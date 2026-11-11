"""No-landing streak guard and act-now directive pin (bounded progress tracking)."""

from types import SimpleNamespace

import pytest

from agent.landing_guard import (
    LandingStreak, LandingStreakConfig, detect_directive, directive_turn_note, is_landing)
from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController


def _explore(i=0):
    return "search_files", {"pattern": f"symbol_{i}"}


def _probe(i=0):
    return "terminal", {"command": f".venv/bin/python -c 'import trimesh; print({i})'"}


@pytest.mark.parametrize("tool,args", [
    ("write_file", {"path": "x"}),
    ("patch", {"path": "x"}),
    ("kanban_comment", {}),
    ("terminal", {"command": "git commit -m x"}),
    ("terminal", {"command": "cd repo && gh pr create --fill"}),
    ("terminal", {"command": "sed -i 's/a/b/' f.yaml"}),
    ("terminal", {"command": "echo hi > out.txt"}),
    ("terminal", {"command": "systemctl --user restart hermes-gateway.service"}),
    ("terminal", {"command": "ssh unraid 'docker restart lidarr'"}),
    ("mcp_homeassistant_ha_config_set_automation", {}),
])
def test_landing_actions(tool, args):
    assert is_landing(tool, args, failed=False)


@pytest.mark.parametrize("tool,args", [
    ("read_file", {"path": "x"}),
    ("search_files", {"pattern": "x"}),
    ("terminal", {"command": "git log -5 --oneline"}),
    ("terminal", {"command": "pytest tests/x.py 2>&1"}),
    ("terminal", {"command": ".venv/bin/python -c 'print(1)'"}),
    ("terminal", {"command": "grep -rn foo . 2>/dev/null"}),
    ("terminal", {"command": "scp host:/data/x.gcode /tmp/hook.gcode"}),
    ("terminal", {"command": "python3 probe.py > /tmp/out.json 2>&1"}),
    ("terminal", {"command": "git checkout -b fix/x"}),
    ("terminal", {"command": ".venv/bin/python -c \"import numpy as np\nprint(np.sum(nz > 0.95))\""}),
    ("terminal", {"command": "python3 - <<'PY'\nimport os\nos.remove('x') if 1 > 0 else None\nPY"}),
    ("mcp_homeassistant_ha_get_state", {}),
])
def test_exploration_is_not_landing(tool, args):
    assert not is_landing(tool, args, failed=False)


def test_failed_write_does_not_land():
    assert not is_landing("patch", {"path": "x"}, failed=True)


def test_notice_final_and_halt_escalate_once_each():
    s = LandingStreak(LandingStreakConfig(), can_halt=True)
    notices = []
    halts = []
    for i in range(1, 51):
        obs = s.observe(*_explore(i), failed=False)
        if obs.notice:
            notices.append((i, obs.notice))
        if obs.halt:
            halts.append(i)
    assert [n for n, _ in notices] == [20, 40]
    assert "Stop exploring" in notices[0][1]
    assert "NEXT action must land" in notices[1][1]
    assert halts == [50]


def test_landing_resets_and_rearms():
    s = LandingStreak(LandingStreakConfig(), can_halt=True)
    for i in range(25):
        s.observe(*_probe(i), failed=False)
    assert s.observe("patch", {"path": "f"}, failed=False).notice is None
    assert s.count == 0
    fired = [i for i in range(1, 21) if s.observe(*_probe(i), failed=False).notice]
    assert fired == [20]


def test_unattended_never_halts():
    s = LandingStreak(LandingStreakConfig(), can_halt=False)
    assert not any(s.observe(*_probe(i), failed=False).halt for i in range(200))


def test_pollers_and_todo_are_neutral():
    s = LandingStreak(LandingStreakConfig(), can_halt=True)
    for i in range(10):
        s.observe(*_probe(i), failed=False)
    for _ in range(30):
        s.observe("process_manage", {"action": "poll"}, failed=False)
        s.observe("todo", {}, failed=False)
    assert s.count == 10


def test_directive_tightens_thresholds_and_names_the_ask():
    s = LandingStreak(LandingStreakConfig(), can_halt=True)
    s.set_directive("Do a PR right now")
    out = [(i, s.observe(*_explore(i), failed=False)) for i in range(1, 36)]
    assert [i for i, o in out if o.notice] == [10, 20]
    assert [i for i, o in out if o.halt] == [35]
    assert "Do a PR right now" in out[9][1].notice


def test_directive_edits_do_not_reset_but_delivery_does():
    s = LandingStreak(LandingStreakConfig(), can_halt=True)
    s.set_directive("Do a PR right now")
    for i in range(9):
        s.observe(*_probe(i), failed=False)
    s.observe("patch", {"path": "f"}, failed=False)
    assert s.count == 10
    s.observe("terminal", {"command": "gh pr create --fill"}, failed=False)
    assert s.count == 0


@pytest.mark.parametrize("text,expected", [
    ("Do a PR right now, Can you also make me a file", "Do a PR right now"),
    ("ok just fix it please", "ok just fix it please"),
    ("stop investigating and ship it", "stop investigating and ship it"),
])
def test_detect_directive(text, expected):
    assert detect_directive(text) == expected


@pytest.mark.parametrize("text", [
    "Please look at the latest discord conversation and figure it out",
    "can you diagnose why hermes fucked up here",
    "[CONTEXT COMPACTION — REFERENCE ONLY] user said do a PR right now",
    "",
    None,
])
def test_no_directive(text):
    assert detect_directive(text) is None


def test_directive_turn_note():
    assert "Do a PR right now" in directive_turn_note("Do a PR right now")
    assert directive_turn_note(None) == ""


def test_config_from_mapping_and_attended_platform():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {"hard_stop_enabled": True, "landing_streak": {"notice_after": 5, "enabled": True}}, platform="desktop")
    assert cfg.landing_streak.notice_after == 5 and cfg.attended
    assert not ToolCallGuardrailConfig.from_mapping({}, platform="cron").attended
    assert LandingStreakConfig.from_mapping({"notice_after": "junk"}).notice_after == 20


def test_controller_records_halt_with_user_facing_message():
    ctl = ToolCallGuardrailController(
        ToolCallGuardrailConfig.from_mapping({"hard_stop_enabled": True}, platform="desktop"))
    ctl.set_turn_directive("Do a PR right now")
    for i in range(35):
        ctl.observe_landing(*_probe(i), failed=False)
    d = ctl.halt_decision
    assert d is not None and d.code == "no_landing_streak_halt"
    assert "didn't change anything" in d.message


def test_controller_reset_clears_directive_and_count():
    ctl = ToolCallGuardrailController(ToolCallGuardrailConfig())
    ctl.set_turn_directive("ship it")
    for i in range(10):
        ctl.observe_landing(*_probe(i), failed=False)
    ctl.reset_for_turn()
    assert ctl._landing.count == 0 and ctl._landing.directive is None


def test_disabled_config_is_silent():
    s = LandingStreak(LandingStreakConfig(enabled=False), can_halt=True)
    assert not any(s.observe(*_probe(i), failed=False).notice for i in range(100))


def _read(path, offset=None, limit=None):
    args = {"path": path}
    if offset is not None:
        args["offset"] = offset
    if limit is not None:
        args["limit"] = limit
    return "read_file", args


def test_overlapping_reread_of_unchanged_file_notices_once():
    s = LandingStreak(LandingStreakConfig(), can_halt=False)
    assert s.observe(*_read("p.py", 3620, 100), failed=False).notice is None
    n = s.observe(*_read("p.py", 3630, 70), failed=False).notice
    assert n and "already read lines 3620-3719 of p.py" in n
    assert s.observe(*_read("p.py", 3640, 60), failed=False).notice is None  # once per path


def test_disjoint_reads_and_continuation_pages_are_fine():
    s = LandingStreak(LandingStreakConfig(), can_halt=False)
    assert s.observe(*_read("p.py"), failed=False).notice is None  # 1-2000
    assert s.observe(*_read("p.py", 2001), failed=False).notice is None  # next page
    assert s.observe(*_read("q.py", 1, 50), failed=False).notice is None


def test_edit_to_the_file_rearms_reads():
    s = LandingStreak(LandingStreakConfig(), can_halt=False)
    s.observe(*_read("p.py", 100, 50), failed=False)
    s.observe("patch", {"path": "p.py"}, failed=False)
    assert s.observe(*_read("p.py", 100, 50), failed=False).notice is None


def test_shell_change_rearms_reads():
    s = LandingStreak(LandingStreakConfig(), can_halt=False)
    s.observe(*_read("p.py", 100, 50), failed=False)
    s.observe("terminal", {"command": "sed -i 's/a/b/' p.py"}, failed=False)
    assert s.observe(*_read("p.py", 100, 50), failed=False).notice is None


def test_probe_notice_once_at_six():
    s = LandingStreak(LandingStreakConfig(), can_halt=False)
    fired = [i for i in range(1, 15) if (o := s.observe(*_probe(i), failed=False)).notice and "inline" in o.notice]
    assert fired == [6]


def test_side_notice_and_streak_notice_combine():
    s = LandingStreak(LandingStreakConfig(notice_after=6), can_halt=False)
    notes = [s.observe(*_probe(i), failed=False).notice for i in range(6)]
    assert "inline interpreter probes" in notes[5] and "Stop exploring" in notes[5]


def test_steer_directive_pins_and_restarts_count():
    s = LandingStreak(LandingStreakConfig(), can_halt=True)
    for i in range(15):
        s.observe(*_probe(i), failed=False)
    assert s.pin_steer("ok stop investigating and do the PR now") is not None
    assert s.count == 0 and s.directive
    fired = [i for i in range(1, 12) if s.observe(*_probe(i), failed=False).notice]
    assert 10 in fired
    assert s.pin_steer("also check the logs") is None and s.directive  # a plain steer changes nothing


def test_controller_steer_passthrough():
    ctl = ToolCallGuardrailController(ToolCallGuardrailConfig())
    assert ctl.pin_steer_directive("just fix it") == "just fix it"


def test_merge_gateway_notes_pins_directive():
    from agent import turn_context as tc
    ctl = ToolCallGuardrailController(ToolCallGuardrailConfig())
    agent = SimpleNamespace(_tool_guardrails=ctl)
    msgs = [{"role": "user", "content": "Do a PR right now, Can you also make me a file"}]
    note = tc._act_now_directive_note(agent, msgs, 0)
    assert "act-now directive" in note and ctl._landing.directive == "Do a PR right now"
    msgs = [{"role": "user", "content": "what's the weather"}]
    assert tc._act_now_directive_note(agent, msgs, 0) == "" and ctl._landing.directive is None
