"""Attribution for background-review forks.

``/refine`` and the automatic post-turn review both fork with
``write_origin='background_review'``, so the skill ledger records ``actor: curator`` for either
(and for the scheduled curator walk). Before this, which door a write came through was visible
only in the rotating agent log. These tests pin the two records that now say it: the spawn log
line and the ledger's ``review_trigger`` evidence field.
"""
from __future__ import annotations

import logging

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


@pytest.mark.parametrize(("attended", "trigger"), [(False, "automatic"), (True, "refine")])
def test_ledger_records_which_door_the_fork_came_through(monkeypatch, attended, trigger):
    """The permanent ledger says ``automatic`` or ``refine``, not just ``curator``."""
    from tools import skill_provenance as provenance

    smt, captured = _capture_ledger(monkeypatch)
    origin = provenance.set_current_write_origin(provenance.BACKGROUND_REVIEW)
    attended_token = provenance.set_review_attended(attended)
    try:
        _record_patch(smt)
    finally:
        provenance.reset_review_attended(attended_token)
        provenance.reset_current_write_origin(origin)

    assert captured["evidence"]["review_trigger"] == trigger
    assert captured["evidence"]["session_id"] == "sess-1"


def test_ledger_leaves_foreground_writes_unlabelled(monkeypatch):
    """A foreground ``skill_manage`` keeps its evidence shape: no review trigger to report."""
    smt, captured = _capture_ledger(monkeypatch)
    _record_patch(smt)
    assert "review_trigger" not in captured["evidence"]
