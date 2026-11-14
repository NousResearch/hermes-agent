"""A committed context boundary must reset the tool-loop guard's window evidence, not only the
repeat-read dedup (#146217): the results those counters describe were removed from the transcript, so
the reload the boundary marker asks for is progress, not a replay."""

import types

from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController


def _controller(platform: str = "telegram") -> ToolCallGuardrailController:
    return ToolCallGuardrailController(
        ToolCallGuardrailConfig.from_mapping({}, platform=platform)
    )


_SKILL_BODY = '{"success": true, "name": "x", "content": "' + "a" * 2000 + '"}'
_OK = '{"exit_code": 0}'


def _reload_then_work(c: ToolCallGuardrailController, window: int):
    """One post-compaction window: the mandatory skill reload, then productive work."""
    decision = c.before_call("skill_view", {"name": "x"})
    if decision.should_halt:
        return decision
    c.after_call("skill_view", {"name": "x"}, _SKILL_BODY, failed=False)
    for i in range(20):
        c.after_call("terminal", {"command": f"step {window}-{i}"}, _OK, failed=False)
    return decision


def test_a_post_compaction_skill_reload_never_hits_the_no_progress_hard_stop():
    c = _controller()
    for window in range(8):
        decision = _reload_then_work(c, window)
        assert not decision.should_halt, (
            f"window {window + 1} halted: {decision.code} count={decision.count}"
        )
        c.reset_after_context_boundary()
    assert c.halt_decision is None


def test_the_same_loop_without_a_boundary_still_hard_stops():
    """The reset is the boundary's doing, not a weakened threshold: six identical reloads with no
    compaction between them still end the turn."""
    c = _controller()
    halted = False
    for window in range(6):
        decision = _reload_then_work(c, window)
        if decision.should_halt:
            halted = True
            assert decision.code == "idempotent_no_progress_block"
            break
    assert halted, (
        "an uninterrupted identical-reload loop must still trip the hard stop"
    )


def test_the_boundary_reset_keeps_call_level_evidence():
    c = _controller()
    c.after_call("terminal", {"command": "false"}, '{"exit_code": 1}', failed=True)
    c._progress_since_failure[next(iter(c._exact_failure_counts))] = True
    c._turn_web_search_count, c._turn_subagent_count = 3, 2
    failures = dict(c._exact_failure_counts)
    progress = dict(c._progress_since_failure)

    c.reset_after_context_boundary()

    assert c._exact_failure_counts == failures
    assert c._progress_since_failure == progress
    assert (c._turn_web_search_count, c._turn_subagent_count) == (3, 2)
    assert c._no_progress == {}
    assert c._identical_streak_count == 0
    assert len(c._call_history) == 0


def test_note_context_boundary_resets_the_agents_live_controller():
    from agent.conversation_compression import note_context_boundary

    c = _controller()
    agent = types.SimpleNamespace(session_id="s-1", _tool_guardrails=c)
    for window in range(5):
        assert not _reload_then_work(c, window).should_halt
    assert c._no_progress, (
        "the reloads must have accumulated repeat-result evidence first"
    )

    note_context_boundary(agent, "task-1")

    assert c._no_progress == {}
    assert c.halt_decision is None
    assert not c.before_call("skill_view", {"name": "x"}).should_halt


def test_note_context_boundary_tolerates_an_agent_without_a_controller():
    from agent.conversation_compression import note_context_boundary

    note_context_boundary(types.SimpleNamespace(session_id="s-2"), "task-2")


# --- the field report's own shape: a cron run, one turn, hard stops on (#146217) ---
# cron is not an attended platform, so the guard's hard stop is on and a scheduled run is a single
# turn: reset_for_turn() never runs inside it, and every compaction is the only chance to forget
# what the boundary removed. The reporter's loop re-reads one small file after each compaction.

_STATUS_JSON = '{"content":"1|{\\"ok\\": true}","total_lines":1}'
_READ_ARGS = {"path": "notes/status.json"}


def _cron_windows(c: ToolCallGuardrailController, boundary) -> list:
    seen = []
    for window in range(1, 7):
        decision = c.before_call("read_file", _READ_ARGS)
        seen.append((window, decision.action, decision.code, decision.count))
        if decision.should_halt:
            break
        c.after_call("read_file", _READ_ARGS, _STATUS_JSON, failed=False)
        for i in range(10):
            c.after_call("terminal", {"command": f"step {window}-{i}"}, _OK, failed=False)
        boundary()
    return seen


def test_a_cron_run_rereading_a_file_after_each_compaction_is_not_a_loop():
    from agent.conversation_compression import note_context_boundary

    c = _controller(platform="cron")
    agent = types.SimpleNamespace(session_id="s-cron", _tool_guardrails=c)

    seen = _cron_windows(c, lambda: note_context_boundary(agent, "task-cron"))

    assert [row[1] for row in seen] == ["allow"] * 6, seen
    assert c.halt_decision is None


def test_the_same_cron_loop_halts_when_only_the_dedup_caches_are_reset():
    """The base behaviour, verbatim from the report: the dedup reset alone leaves the guard's
    window evidence behind, and window 6 is blocked with count=5."""
    from agent.conversation_compression import _reset_read_dedup_caches

    c = _controller(platform="cron")

    seen = _cron_windows(c, lambda: _reset_read_dedup_caches("task-cron", session_id="s-cron"))

    assert seen[-1][0] == 6, seen
    assert seen[-1][2] == "idempotent_no_progress_block"
    assert seen[-1][3] == 5
    assert c.halt_decision is not None
