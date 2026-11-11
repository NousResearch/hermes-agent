"""Behavior contracts for hermes sessions repair-prompts (#122822)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.prompt_builder import SKILLS_GUIDANCE
from hermes_cli import sessions_cmd
from hermes_cli.sessions_cmd import _cmd_repair_prompts
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_session

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
    with open_persistence(profile_scope(home=tmp_path)) as store:
        yield store


def _repair_prompts(db, args):
    with open_persistence(db.scope) as store:
        return _cmd_repair_prompts(args, persistence=store)


def _args(session_id=None, apply=False, json_out=False):
    return SimpleNamespace(session_id=session_id, apply=apply, json=json_out)


def _apply_json(db, monkeypatch, capsys):
    monkeypatch.setattr(
        sessions_cmd, "_confirm_prompt",
        lambda _prompt: pytest.fail("--apply --json must not prompt"),
    )
    assert _repair_prompts(db, _args(apply=True, json_out=True)) == 0
    return json.loads(capsys.readouterr().out)


def test_unpinned_row_is_unverifiable_and_apply_json_clears_only_verified(db, monkeypatch, capsys):
    unverifiable = seed_session(db, "old-unpinned", "telegram", system_prompt=DEGRADED)
    verified = seed_session(db, "verified", "telegram", system_prompt=DEGRADED)
    db.sessions.update_tool_pin(verified, _pin("skills_list", "skill_manage"))
    healthy = seed_session(db, "healthy-1", "telegram", system_prompt=HEALTHY)
    db.sessions.update_tool_pin(healthy, _pin("skills_list", "skill_manage"))

    payload = _apply_json(db, monkeypatch, capsys)

    assert payload["apply"] is True
    assert payload["cleared"] == ["verified"]
    assert [row["id"] for row in payload["unverifiable"]] == ["old-unpinned"]
    assert session_row(db, unverifiable)["system_prompt"] == DEGRADED
    assert not (session_row(db, verified)["system_prompt"] or "")
    assert session_row(db, verified)["tool_names"]  # the skill_manage pin survives the clear
    assert session_row(db, healthy)["system_prompt"] == HEALTHY


def test_scan_leaves_reduced_memory_only_zero_skills_rows_and_session_id_overrides(db, monkeypatch, capsys):
    reduced = seed_session(db, "reduced-1", "telegram", system_prompt=DEGRADED)
    db.sessions.update_tool_pin(reduced, _pin("todo", "web_search"))
    memory = seed_session(db, "memory-only-1", "telegram", system_prompt=DEGRADED)
    db.sessions.update_tool_pin(memory, _pin("memory"))
    # Zero skills installed + no skill_manage: the real builder emits neither skills marker.
    zero_skills = "You are Hermes.\nNo skills are installed.\n"
    readonly = seed_session(db, "zero-skills-readonly", "telegram", system_prompt=zero_skills)
    db.sessions.update_tool_pin(readonly, _pin("terminal", "skills_list", "skill_view"))

    payload = _apply_json(db, monkeypatch, capsys)

    assert payload["cleared"] == []
    assert [row["id"] for row in payload["unverifiable"]] == ["memory-only-1"]
    assert session_row(db, reduced)["system_prompt"] == DEGRADED
    assert session_row(db, memory)["system_prompt"] == DEGRADED
    assert session_row(db, memory)["tool_names"]
    assert session_row(db, readonly)["system_prompt"] == zero_skills

    # An explicit SESSION_ID is the operator override for the unverifiable memory-only row.
    monkeypatch.setattr(sessions_cmd, "_confirm_prompt", lambda _prompt: True)
    assert _repair_prompts(db, _args(session_id="memory-only", apply=True)) == 0
    assert not (session_row(db, memory)["system_prompt"] or "")

    capsys.readouterr()
    assert _repair_prompts(db, _args(session_id="memory-only", apply=True)) == 0
    assert "already has no stored prompt" in capsys.readouterr().out
    assert _repair_prompts(db, _args(session_id="no-such-session", apply=True)) == 1
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
    healthy = seed_session(db, "legacy-healthy", "telegram", system_prompt=legacy)
    db.sessions.update_tool_pin(healthy, _pin("skills_list", "skill_manage"))

    assert _apply_json(db, monkeypatch, capsys)["cleared"] == []
    assert session_row(db, healthy)["system_prompt"] == legacy
