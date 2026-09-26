"""Behavior contracts for hermes sessions repair-prompts (#122822)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hermes_cli import sessions_cmd
from hermes_cli.sessions_cmd import _cmd_repair_prompts, _repair_prompts_pin_names
from hermes_state import SessionDB

HEALTHY = (
    "You are Hermes.\n"
    "<available_skills>\n  dogfood: exploratory QA of web apps\n</available_skills>\n"
    "## Skill Safety Rule\nReload [SKILL_PRUNED] placeholders with skill_view.\n"
)
DEGRADED = "You are Hermes.\n(reduced maintenance build without the skills index)\n"


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} test tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _pin(*names: str) -> dict:
    return {"version": "test-sha", "tools": [_tool(name) for name in names]}


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def _args(session_id=None, apply=False, json_out=False):
    return SimpleNamespace(session_id=session_id, apply=apply, json=json_out)


def _accept_confirmations(monkeypatch):
    asked = []
    monkeypatch.setattr(sessions_cmd, "_confirm_prompt", lambda prompt: asked.append(prompt) or True)
    return asked


def test_pin_reader_accepts_current_versioned_and_legacy_shapes():
    current = {"tool_names": json.dumps(_pin("memory", "skills_list"))}
    legacy = {"tool_names": json.dumps(["memory", "skills_list"])}
    malformed = {"tool_names": json.dumps({"version": "x", "tools": [{"bad": True}]})}

    assert _repair_prompts_pin_names(current) == ["memory", "skills_list"]
    assert _repair_prompts_pin_names(legacy) == ["memory", "skills_list"]
    assert _repair_prompts_pin_names(malformed) is None


def test_degraded_row_reported_and_cleared_only_with_apply(db, monkeypatch, capsys):
    degraded = db.create_session("degraded-1", "telegram", system_prompt=DEGRADED)
    db.update_session_tool_names(degraded, _pin("skills_list", "skill_view", "skill_manage"))
    healthy = db.create_session("healthy-1", "telegram", system_prompt=HEALTHY)

    assert _cmd_repair_prompts(db, _args()) == 0
    out = capsys.readouterr().out
    assert "degraded-1" in out
    assert "healthy-1" not in out
    assert db.get_session(degraded)["system_prompt"] == DEGRADED

    asked = _accept_confirmations(monkeypatch)
    assert _cmd_repair_prompts(db, _args(apply=True)) == 0
    assert asked
    assert not (db.get_session(degraded)["system_prompt"] or "")
    assert db.get_session(healthy)["system_prompt"] == HEALTHY


def test_unpinned_row_is_unverifiable_and_apply_json_clears_only_verified(db, monkeypatch, capsys):
    unverifiable = db.create_session("old-unpinned", "telegram", system_prompt=DEGRADED)
    verified = db.create_session("verified", "telegram", system_prompt=DEGRADED)
    db.update_session_tool_names(verified, _pin("skills_list"))

    monkeypatch.setattr(
        sessions_cmd, "_confirm_prompt",
        lambda _prompt: pytest.fail("--apply --json must not prompt"),
    )
    assert _cmd_repair_prompts(db, _args(apply=True, json_out=True)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["apply"] is True
    assert payload["cleared"] == ["verified"]
    assert [row["id"] for row in payload["unverifiable"]] == ["old-unpinned"]
    assert db.get_session(unverifiable)["system_prompt"] == DEGRADED
    assert not (db.get_session(verified)["system_prompt"] or "")


def test_reduced_surface_left_alone_memory_only_pin_cleared(db, monkeypatch, capsys):
    reduced = db.create_session("reduced-1", "telegram", system_prompt=DEGRADED)
    db.update_session_tool_names(reduced, _pin("todo", "web_search"))
    empty = db.create_session("empty-pin", "telegram", system_prompt=DEGRADED)
    db.update_session_tool_names(empty, _pin())
    memory = db.create_session("memory-only-1", "telegram", system_prompt=DEGRADED)
    db.update_session_tool_names(memory, _pin("memory"))

    assert _cmd_repair_prompts(db, _args(json_out=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    flagged = {finding["id"]: finding for finding in payload["findings"]}
    assert set(flagged) == {"memory-only-1"}
    assert flagged["memory-only-1"]["clear_pin"] is True

    _accept_confirmations(monkeypatch)
    assert _cmd_repair_prompts(db, _args(apply=True)) == 0

    assert db.get_session(reduced)["system_prompt"] == DEGRADED
    assert db.get_session(empty)["system_prompt"] == DEGRADED
    memory_row = db.get_session(memory)
    assert not (memory_row["system_prompt"] or "")
    assert not (memory_row["tool_names"] or "")


def test_positional_target_clears_memory_pin_when_prompt_is_already_null(db, monkeypatch):
    target = db.create_session("target-null", "telegram")
    db.update_session_tool_names(target, _pin("memory"))
    assert not (db.get_session(target)["system_prompt"] or "")

    _accept_confirmations(monkeypatch)
    assert _cmd_repair_prompts(db, _args(session_id="target-null", apply=True)) == 0

    row = db.get_session(target)
    assert not (row["system_prompt"] or "")
    assert not (row["tool_names"] or "")


def test_positional_target_clears_exactly_that_row_and_memory_pin(db, monkeypatch, capsys):
    target = db.create_session("target-1", "telegram", system_prompt=HEALTHY)
    db.update_session_tool_names(target, _pin("memory"))
    sibling = db.create_session("sibling-1", "telegram", system_prompt=HEALTHY)
    db.update_session_tool_names(sibling, _pin("skills_list"))

    assert _cmd_repair_prompts(db, _args(session_id="target")) == 0
    assert db.get_session(target)["system_prompt"] == HEALTHY

    _accept_confirmations(monkeypatch)
    assert _cmd_repair_prompts(db, _args(session_id="target", apply=True)) == 0
    target_row = db.get_session(target)
    assert not (target_row["system_prompt"] or "")
    assert not (target_row["tool_names"] or "")
    assert db.get_session(sibling)["system_prompt"] == HEALTHY

    assert _cmd_repair_prompts(db, _args(session_id="no-such-session", apply=True)) == 1
    assert "No session matches" in capsys.readouterr().out
