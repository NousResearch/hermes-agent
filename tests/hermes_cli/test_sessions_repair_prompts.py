"""Behavior contracts for hermes sessions repair-prompts (#122822)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.prompt_builder import SKILLS_GUIDANCE
from hermes_cli import sessions_cmd
from hermes_cli.sessions_cmd import _cmd_repair_prompts
from hermes_state import SessionDB

HEALTHY = (
    "You are Hermes.\n"
    "<available_skills>\n  dogfood: exploratory QA of web apps\n</available_skills>\n"
    f"{SKILLS_GUIDANCE}\n"
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


def _apply_json(db, monkeypatch, capsys):
    monkeypatch.setattr(
        sessions_cmd, "_confirm_prompt",
        lambda _prompt: pytest.fail("--apply --json must not prompt"),
    )
    assert _cmd_repair_prompts(db, _args(apply=True, json_out=True)) == 0
    return json.loads(capsys.readouterr().out)


def test_unpinned_row_is_unverifiable_and_apply_json_clears_only_verified(db, monkeypatch, capsys):
    unverifiable = db.create_session("old-unpinned", "telegram", system_prompt=DEGRADED)
    verified = db.create_session("verified", "telegram", system_prompt=DEGRADED)
    db.update_session_tool_names(verified, _pin("skills_list", "skill_manage"))
    healthy = db.create_session("healthy-1", "telegram", system_prompt=HEALTHY)
    db.update_session_tool_names(healthy, _pin("skills_list", "skill_manage"))

    payload = _apply_json(db, monkeypatch, capsys)

    assert payload["apply"] is True
    assert payload["cleared"] == ["verified"]
    assert [row["id"] for row in payload["unverifiable"]] == ["old-unpinned"]
    assert db.get_session(unverifiable)["system_prompt"] == DEGRADED
    assert not (db.get_session(verified)["system_prompt"] or "")
    assert db.get_session(verified)["tool_names"]  # the skill_manage pin survives the clear
    assert db.get_session(healthy)["system_prompt"] == HEALTHY


def test_scan_leaves_reduced_memory_only_zero_skills_rows_and_session_id_overrides(db, monkeypatch, capsys):
    reduced = db.create_session("reduced-1", "telegram", system_prompt=DEGRADED)
    db.update_session_tool_names(reduced, _pin("todo", "web_search"))
    memory = db.create_session("memory-only-1", "telegram", system_prompt=DEGRADED)
    db.update_session_tool_names(memory, _pin("memory"))
    # Zero skills installed + no skill_manage: the real builder emits neither skills marker.
    zero_skills = "You are Hermes.\nNo skills are installed.\n"
    readonly = db.create_session("zero-skills-readonly", "telegram", system_prompt=zero_skills)
    db.update_session_tool_names(readonly, _pin("terminal", "skills_list", "skill_view"))

    payload = _apply_json(db, monkeypatch, capsys)

    assert payload["cleared"] == []
    assert [row["id"] for row in payload["unverifiable"]] == ["memory-only-1"]
    assert db.get_session(reduced)["system_prompt"] == DEGRADED
    assert db.get_session(memory)["system_prompt"] == DEGRADED
    assert db.get_session(memory)["tool_names"]
    assert db.get_session(readonly)["system_prompt"] == zero_skills

    # An explicit SESSION_ID is the operator override for the unverifiable memory-only row.
    monkeypatch.setattr(sessions_cmd, "_confirm_prompt", lambda _prompt: True)
    assert _cmd_repair_prompts(db, _args(session_id="memory-only", apply=True)) == 0
    assert not (db.get_session(memory)["system_prompt"] or "")

    capsys.readouterr()
    assert _cmd_repair_prompts(db, _args(session_id="memory-only", apply=True)) == 0
    assert "already has no stored prompt" in capsys.readouterr().out
    assert _cmd_repair_prompts(db, _args(session_id="no-such-session", apply=True)) == 1
    assert "No session matches" in capsys.readouterr().out


def test_scan_leaves_a_healthy_prompt_stored_before_the_skill_guidance_rewordings(db, monkeypatch, capsys):
    # Literal skills guidance stored before its opening line was reworded: old opening, then the Skill Safety heading.
    legacy = (
        "You are Hermes.\n"
        "<available_skills>\n  dogfood: exploratory QA of web apps\n</available_skills>\n"
        "After completing a complex task (5+ tool calls), fixing a tricky error, or discovering a non-trivial "
        "workflow, save the approach as a skill with skill_manage so you can reuse it next time.\n\n"
        "## Skill Safety Rule\n"
        "A skill placeholder containing `[SKILL_PRUNED]` lost its content in context compression and is "
        "inaccessible — reload it with skill_view(name='...') before acting on anything that depends on it.\n"
    )
    healthy = db.create_session("legacy-healthy", "telegram", system_prompt=legacy)
    db.update_session_tool_names(healthy, _pin("skills_list", "skill_manage"))

    assert _apply_json(db, monkeypatch, capsys)["cleared"] == []
    assert db.get_session(healthy)["system_prompt"] == legacy
