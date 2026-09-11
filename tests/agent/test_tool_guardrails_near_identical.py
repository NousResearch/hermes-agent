"""Near-identical call streak: a reshuffled command with drifting output is still a loop."""

from agent.tool_guardrails import (
    NEAR_IDENTICAL_ARGS_JACCARD,
    NEAR_IDENTICAL_RESULT_JACCARD,
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
)

_KEY_PRELUDE = (
    "python3 -c \"import json; d=json.load(open('/x/openclaw.json')); "
    "open('/tmp/helper_key.txt','w').write(d.get('env',{}).get('HELPER_API_KEY',''))\""
)
_CHECK = "bash ~/.hermes/skills/gumroad/gumroad-support-auto-triage/scripts/check_changes.sh"

# Three phrasings of one command, cycled — the exact-hash streak never sees two identical neighbours.
_VARIANTS = [
    f"export HOME=/Users/u\n{_KEY_PRELUDE}\nHKEY=$(cat /tmp/helper_key.txt)\nHELPER_API_KEY=\"$HKEY\" {_CHECK}",
    f"{_KEY_PRELUDE} && export HOME=/Users/u && HKEY=$(cat /tmp/helper_key.txt) && HELPER_API_KEY=\"$HKEY\" {_CHECK} > /tmp/o.txt 2>&1; echo EXIT:$?; cat /tmp/o.txt",
    f"{_KEY_PRELUDE}\nexport HOME=/Users/u\nHKEY=$(cat /tmp/helper_key.txt)\nexport HELPER_API_KEY=\"$HKEY\"\n{_CHECK}",
]


def _unattended() -> ToolCallGuardrailController:
    return ToolCallGuardrailController(ToolCallGuardrailConfig.from_mapping({}, platform="webhook"))


def _drive(controller, calls):
    """Feed (tool, args, result) triples; return the first halt decision, or None."""
    for tool, args, result in calls:
        controller.after_call(tool, args, result, failed=False)
        controller.observe_call(tool, args, result, tool_call_id="c")
        if controller.halt_decision is not None:
            return controller.halt_decision
    return None


def test_reshuffled_terminal_command_with_drifting_count_halts_on_unattended_platform():
    controller = _unattended()
    calls = [
        ("terminal", {"command": _VARIANTS[i % 3], "timeout": 120},
         '{"output": "CHANGED:%d/%d", "exit_code": 0, "error": null}' % (44 + i % 5, 44 + i % 5))
        for i in range(40)
    ]
    halt = _drive(controller, calls)

    assert halt is not None
    assert halt.code == "near_identical_call_streak_halt"
    assert halt.count == controller.config.near_identical_block_after
    assert halt.count < 40, "must fire well before the iteration budget"


def test_near_identical_is_warn_only_on_attended_platform():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig.from_mapping({}, platform="cli"))
    calls = [("terminal", {"command": _VARIANTS[i % 3]}, '{"output": "CHANGED:47/47", "exit_code": 0}') for i in range(20)]
    warned = False
    for tool, args, result in calls:
        controller.after_call(tool, args, result, failed=False)
        obs = controller.observe_call(tool, args, result)
        warned = warned or (obs.notice is not None and "near-identical" in obs.notice)
    assert warned
    assert controller.halt_decision is None


def test_edit_then_rerun_cycle_with_changing_output_never_streaks():
    """Same test command re-run after each edit, output genuinely different each time: real work."""
    controller = _unattended()
    calls = []
    for i in range(30):
        calls.append(("patch", {"path": "lib/foo.rb", "old_string": f"a{i}", "new_string": f"b{i}"}, '{"success": true}'))
        outcome = ("FAILED: expected %d got %d\n  spec/foo_spec.rb:%d\n  NoMethodError undefined method_%d" % (i, i + 1, i * 7, i))
        calls.append(("terminal", {"command": "bundle exec rspec spec/foo_spec.rb"}, '{"output": "%s", "exit_code": 1}' % outcome))
    assert _drive(controller, calls) is None


def test_sweep_over_distinct_records_with_similar_args_never_streaks():
    """Same command shape over N different IDs returning N different payloads is iteration, not a loop."""
    controller = _unattended()
    calls = [
        ("terminal", {"command": f"gumroad admin purchases show {i:032x}"},
         '{"output": "id=%032x buyer=user%d@example.com product=%s amount=%d state=%s", "exit_code": 0}'
         % (i, i, ["ebook", "course", "font", "template"][i % 4], 100 + i * 37, ["ok", "refunded", "disputed"][i % 3]))
        for i in range(30)
    ]
    assert _drive(controller, calls) is None


def test_same_command_over_distinct_ids_with_boilerplate_result_never_streaks():
    """`claim --slug <id>` x N returning `CLAIM_OK <id>`: results are ~all boilerplate, only the id moves."""
    controller = _unattended()
    calls = [
        ("terminal", {"command": f"python3 ~/.hermes/scripts/helper_claim.py claim --slug {i:032x} --lane sla"},
         '{"output": "# CLAIM_OK %032x: own reply to send", "exit_code": 0, "error": null}' % i)
        for i in range(40)
    ]
    assert _drive(controller, calls) is None


def test_off_pattern_decoration_does_not_reset_the_loop():
    """A loop that occasionally wraps the same command in `echo EXIT:$?; cat` must still halt."""
    controller = _unattended()
    calls = []
    for i in range(30):
        calls.append(("terminal", {"command": _VARIANTS[i % 3]}, '{"output": "CHANGED:47/47", "exit_code": 0}'))
        if i % 4 == 3:
            calls.append(("terminal", {"command": "wc -l /tmp/triage_changes.txt; cat /tmp/triage_changes.txt"},
                          '{"output": "1 /tmp/triage_changes.txt\\nCHANGED:47/47", "exit_code": 0}'))
    assert _drive(controller, calls) is not None


def test_pollers_exempt_from_near_identical_streak():
    controller = _unattended()
    calls = [("process_manage", {"action": "poll", "session_id": "abc"}, '{"status": "running", "output_preview": "step %d"}' % (i // 5)) for i in range(40)]
    assert _drive(controller, calls) is None


def test_switching_tools_resets_streak():
    controller = _unattended()
    calls = []
    for i in range(20):
        calls.append(("terminal", {"command": _VARIANTS[i % 3]}, '{"output": "CHANGED:47/47", "exit_code": 0}'))
        calls.append(("read_file", {"path": f"/tmp/f{i}.txt"}, f"content {i}"))
    assert _drive(controller, calls) is None


def test_threshold_is_configurable_via_nested_sections():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {"warn_after": {"near_identical_no_progress": 2}, "hard_stop_after": {"near_identical_no_progress": 3}},
        platform="webhook",
    )
    assert cfg.near_identical_warn_after == 2
    assert cfg.near_identical_block_after == 3
    assert 0 < NEAR_IDENTICAL_ARGS_JACCARD < 1 and 0 < NEAR_IDENTICAL_RESULT_JACCARD <= 1
