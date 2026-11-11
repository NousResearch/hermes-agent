"""Landing guard worker mode : read-only review workers are not told to stop.

Read-heavy workers must checkpoint without treating an incomplete audit as done.
"""

import pytest

from agent import delegation_context
from agent.landing_guard import LandingStreak, LandingStreakConfig
from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController

_STOP_TEXT = "report status to the user and stop"


def _thirty_record_review(write_every=None):
    """Per record: read the memory, then read its governing skill. Optionally checkpoint the report."""
    calls = []
    for i in range(30):
        calls.append(("mcp_openviking_read", {"uri": f"viking://user/memories/pref_{i:03d}"}))
        calls.append(("read_file", {"path": f"/example/hermes/skills/s{i % 9}/SKILL.md", "offset": 1 + 40 * i}))
        if write_every and (i + 1) % write_every == 0:
            calls.append(("write_file", {"path": "/example/hermes/cache/scratch/review.md"}))
    return calls


def _run(streak, calls):
    notices, halted = [], False
    for n, (tool, args) in enumerate(calls, 1):
        obs = streak.observe(tool, args, failed=False)
        if obs.notice:
            notices.append((n, obs.notice))
        halted = halted or obs.halt
    return notices, halted


def test_read_heavy_worker_is_not_told_to_stop_at_20():
    notices, halted = _run(LandingStreak(LandingStreakConfig(), can_halt=True, worker=True), _thirty_record_review())
    streak_notices = [(n, t) for n, t in notices if "tool calls since you last saved" in t]
    assert not halted
    assert all(_STOP_TEXT not in t and "Stop exploring" not in t for _, t in notices)
    assert [n for n, _ in streak_notices] == [40]          # one checkpoint, well past 20
    assert "Do not stop early" in streak_notices[0][1]


def test_worker_that_checkpoints_its_report_never_gets_the_streak_notice():
    notices, halted = _run(LandingStreak(LandingStreakConfig(), can_halt=True, worker=True),
                           _thirty_record_review(write_every=5))
    assert not halted
    assert not [t for _, t in notices if "tool calls" in t and "since" in t]


def test_attended_chat_is_unchanged():
    notices, halted = _run(LandingStreak(LandingStreakConfig(), can_halt=True), _thirty_record_review())
    assert notices[0][0] == 20 and "Stop exploring" in notices[0][1]
    assert halted  # 50-call halt still applies to attended chat


def test_attended_directive_thresholds_unchanged():
    s = LandingStreak(LandingStreakConfig(), can_halt=True)
    s.set_directive("do a PR right now")
    notices, _ = _run(s, _thirty_record_review())
    assert notices[0][0] == 10 and "do a PR right now" in notices[0][1]


def test_worker_mode_can_be_turned_off():
    cfg = LandingStreakConfig.from_mapping({"worker_mode": False})
    notices, _ = _run(LandingStreak(cfg, worker=True), _thirty_record_review())
    assert notices[0][0] == 20 and "Stop exploring" in notices[0][1]


def test_subagent_and_kanban_worker_runs_get_worker_mode(monkeypatch):
    assert ToolCallGuardrailConfig.from_mapping({}, platform="subagent").worker
    assert not ToolCallGuardrailConfig.from_mapping({}, platform="desktop").worker
    monkeypatch.setattr(delegation_context, "owned_kanban_task", lambda: "t_21905319")
    cfg = ToolCallGuardrailConfig.from_mapping({"hard_stop_enabled": True}, platform="kanban")
    assert cfg.worker
    controller = ToolCallGuardrailController(cfg)
    texts = [controller.observe_landing(t, a, failed=False) for t, a in _thirty_record_review()]
    assert controller.halt_decision is None
    assert not any(t and _STOP_TEXT in t for t in texts)


def test_delegated_child_inheriting_the_env_is_judged_by_platform(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    monkeypatch.setattr(delegation_context, "owned_kanban_task", lambda: "")
    assert not ToolCallGuardrailConfig.from_mapping({}, platform="desktop").worker
