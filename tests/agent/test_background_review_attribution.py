"""Attribution for background-review forks.

``/refine``, a focus-only caller, and the automatic post-turn review all fork with
``write_origin='background_review'``, so the skill ledger records ``actor: curator`` for all three
(and for the scheduled curator walk). Before this, which door a write came through was visible
only in the rotating agent log. These tests pin the two records that now say it: the spawn log
line and the ledger's ``review_trigger`` evidence field (``automatic``, ``focus_only``, or
``refine``, carried through ``/skills approve`` when staged).
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from agent import background_review


class _Agent:
    """Minimal fork parent: only ``session_id`` is read by the spawn log."""

    session_id = "sess-1"


@pytest.mark.parametrize(
    ("explicit", "focus", "entry", "gate", "focused"),
    [
        (False, None, "automatic", "passed", False),
        (False, "legacy caller focus", "focus_only", "bypassed", True),
        (True, None, "explicit", "bypassed", False),
        (True, "cover only the CI paths", "explicit", "bypassed", True),
    ],
)
def test_spawn_names_entry_point_gate_and_focus(
    monkeypatch, caplog, explicit, focus, entry, gate, focused
):
    """Every fork announces which caller asked for it and whether the gate was consulted.

    ``/refine <focus>`` sets BOTH ``explicit=True`` and a focus, which is why ``focused`` is its
    own field: an ``entry`` value alone would lose the steering text's presence.
    """
    worked = []
    monkeypatch.setattr(
        background_review, "_run_review_in_thread", lambda *a, **k: worked.append(True)
    )

    target, _prompt = background_review.spawn_background_review_thread(
        _Agent(), [], review_memory=True, review_skills=True,
        focus=focus, task_cfg={}, explicit=explicit,
    )
    with caplog.at_level(logging.INFO, logger="agent.background_review"):
        target()

    lines = [r.getMessage() for r in caplog.records if "Background review spawn:" in r.getMessage()]
    assert len(lines) == 1, lines
    assert "session=sess-1" in lines[0]
    assert f"entry={entry}" in lines[0]
    assert f"gate={gate}" in lines[0]
    assert f"focused={focused}" in lines[0]
    assert worked, "the log line must not replace the fork worker"


def _capture_ledger(monkeypatch):
    """Patch the ledger writer and the side effects that would touch a real home."""
    import tools.skill_ledger as ledger
    import tools.skill_manager_tool as smt
    from tools import skill_usage

    captured = {}

    def fake_record(action, name, *, before, after_root, evidence):
        captured.update(action=action, name=name, evidence=evidence)

    monkeypatch.setattr(ledger, "record_mutation", fake_record)
    monkeypatch.setattr(smt, "_find_skill", lambda name: {"path": "/nonexistent/SKILL.md"})
    for attr in ("bump_patch", "forget", "record_created"):
        monkeypatch.setattr(skill_usage, attr, lambda *a, **k: None)
    return smt, captured


def _record_patch(smt):
    smt._record_success(
        "patch", "demo", {"success": True}, file_path=None, absorbed_into=None,
        task_id=None, session_id="sess-1", ledger_before=None,
    )


def _run_in_review_fork(tmp_path, monkeypatch, *, explicit: bool, focus: str | None, body) -> None:
    """Run ``body()`` inside a real ``spawn_background_review_thread`` -> ``run_conversation`` turn."""
    home = tmp_path / "home"
    (home / "skills").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [home / "skills"])
    monkeypatch.setattr("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None)

    def _create(**_kw):
        body()
        msg = SimpleNamespace(content="Done.", reasoning_content=None, reasoning=None, tool_calls=None)
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")], model="test-model", usage=usage,
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create)),
        close=lambda: None,
    )
    monkeypatch.setattr("agent.process_bootstrap._OPENAI_CLS_CACHE", lambda **_kw: client)

    from typing import Any
    from run_agent import AIAgent

    parent: Any = AIAgent(
        api_key="test-key", base_url="http://127.0.0.1:9/v1", model="test-model",
        quiet_mode=True, session_id="sess-1", skip_context_files=True, skip_memory=True,
    )
    parent._cached_system_prompt = "stable parent prompt"
    import contextvars
    target, _prompt = background_review.spawn_background_review_thread(
        parent, [], review_memory=True, review_skills=True, focus=focus, task_cfg={}, explicit=explicit,
    )
    contextvars.copy_context().run(target)


@pytest.mark.parametrize(
    ("explicit", "focus", "trigger"),
    [
        (False, None, "automatic"),
        (False, "legacy caller focus", "focus_only"),
        (True, None, "refine"),
        (True, "cover only the CI paths", "refine"),
    ],
)
def test_ledger_records_which_door_the_fork_came_through(
    tmp_path, monkeypatch, explicit, focus, trigger
):
    """The permanent ledger records ``automatic``, ``focus_only``, or ``refine`` from fork entry."""
    smt, captured = _capture_ledger(monkeypatch)
    _run_in_review_fork(
        tmp_path, monkeypatch, explicit=explicit, focus=focus, body=lambda: _record_patch(smt),
    )

    assert captured["evidence"]["review_trigger"] == trigger
    assert captured["evidence"]["session_id"] == "sess-1"


def test_focus_only_fork_stamps_focus_only_without_relaxing_unattended_memory_gate(
    tmp_path, monkeypatch
):
    """A focus-only fork (``explicit=False``, ``focus`` set) stamps ``review_trigger='focus_only'``
    in the ledger while keeping ``is_unattended_review()`` True so the memory-delete gate holds."""
    from typing import Any
    from tools import skill_ledger
    from tools.memory_tool import MemoryStore, memory_tool
    from tools.skill_manager_guards import mark_background_review_skill_read
    from tools.skill_manager_tool import skill_manage

    outcomes: dict[str, Any] = {}

    def _fork_turn():
        store = MemoryStore()
        store.load_from_disk()
        assert store.add("memory", "Old staging host: db-1")["success"] is True
        outcomes["memory_remove"] = json.loads(
            memory_tool(action="remove", target="memory", old_text="Old staging host", store=store)
        )
        outcomes["remaining_memories"] = list(store.memory_entries)

        created = json.loads(skill_manage(
            action="create", name="demo-skill", session_id="sess-1",
            content="---\nname: demo-skill\ndescription: Use when testing.\n---\n\nOriginal step.\n",
        ))
        assert created["success"] is True, created
        mark_background_review_skill_read(tmp_path / "home" / "skills" / "demo-skill" / "SKILL.md")
        outcomes["skill_patch"] = json.loads(skill_manage(
            action="patch", name="demo-skill", session_id="sess-1",
            old_string="Original step.", new_string="Refined step.",
        ))

    _run_in_review_fork(
        tmp_path, monkeypatch, explicit=False, focus="tighten demo-skill", body=_fork_turn,
    )

    # Unattended memory-delete gate still applies: the entry stays and a proposal is staged.
    assert outcomes["memory_remove"].get("proposal_staged") is True, outcomes["memory_remove"]
    assert outcomes["remaining_memories"] == ["Old staging host: db-1"]

    # Skill ledger stamps actor='curator' and review_trigger='focus_only' (not 'automatic').
    assert outcomes["skill_patch"]["success"] is True, outcomes["skill_patch"]
    patch_rows = [r for r in skill_ledger.list_entries(skill="demo-skill") if r["action"] == "patch"]
    assert len(patch_rows) == 1
    assert patch_rows[0]["actor"] == "curator"
    assert patch_rows[0]["evidence"].get("review_trigger") == "focus_only"


def test_ledger_leaves_foreground_writes_unlabelled(monkeypatch):
    """A foreground ``skill_manage`` keeps its evidence shape: no review trigger to report."""
    smt, captured = _capture_ledger(monkeypatch)
    _record_patch(smt)
    assert "review_trigger" not in captured["evidence"]
