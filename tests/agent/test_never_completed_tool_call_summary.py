"""Compaction must render never-completed tool calls (timeout / interrupt / lost worker /
orphan recovery) as "not run" — never as completed actions.

Observed live (2026-10-14 audit of PR #138211's follow-on): the executor's non-execution
outcomes were bare prose (``Error executing tool 'x': timed out after 420.0s``,
``[Tool execution cancelled — …]``, ``[Orphan recovery: …]``). The compressor only read
failures from JSON ``error`` / ``success:false`` payloads and refused statuses
(blocked/pending_approval/rejected), so bare prose compressed through the per-tool
summarizers as success: the timeout sample above became ``[terminal] ran `ls` -> exit ?,
1 lines output`` — a never-executed call entering compaction as a completed anchor.

Producers now wrap their wordings via :func:`agent.tool_rejection.outcome_result` into
``{"status": "cancelled"|"timeout"|"unknown", "error": …}`` (single choke points:
``_unfinished_tool_result``, ``_abandoned_sequential_result``, ``_append_skipped_tool_results``,
the dispatch error paths, ``turn_loop_errors``'s unanswered-call fill, and
``replay_cleanup._orphan_recovery``), the compressor reads those statuses structurally,
and the four verb-summarizers (terminal/search_files/execute_code/skill_view) are
failure-first so an error payload can never read as "ran"/"matches"/"lines output".
Bare prose remains readable for transcripts recorded before the statuses existed.
"""

from __future__ import annotations

import json
import os
import sys

# Repo root = three levels up from tests/agent/<file>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent.context_compressor import _summarize_refused_tool_result, _summarize_tool_result


def _outcome(status: str, message: str) -> str:
    from agent.tool_rejection import outcome_result

    return outcome_result(status, message)


def _summarize(name: str, args: dict, content: str) -> str:
    return _summarize_tool_result(name, json.dumps(args), content)


# --- Structural statuses: the live producer payloads --------------------------------------


def test_timeout_outcome_summarizes_as_never_completed():
    # Byte-for-byte the sequential runner's timeout wording (audit's original failure sample).
    content = _outcome("timeout", "Error executing tool 'terminal': timed out after 420.0s")
    summary = _summarize("terminal", {"command": "ls"}, content)
    assert "timed out" in summary
    assert "no result" in summary
    assert "do not report it as done" in summary
    assert "ran" not in summary  # must not render through _sum_terminal as a run


def test_cancelled_outcome_summarizes_as_never_run():
    # Byte-for-byte the interrupt-skip wording _unfinished_tool_result / _append_skipped_tool_results emit.
    content = _outcome("cancelled", "[Tool execution cancelled — terminal was skipped. Interrupted by the user.]")
    summary = _summarize("terminal", {"command": "ls"}, content)
    assert "cancelled" in summary
    assert "not run" in summary
    assert "ran" not in summary


def test_unknown_outcome_summarizes_effect_unknown():
    # Byte-for-byte the lost-worker wording ("thread did not return") and the dangling
    # orphan-recovery notice: the effect may exist, so the summary must demand verification.
    for message in (
        "Error executing tool 'terminal': thread did not return a result",
        "[Orphan recovery: this tool may have executed before Hermes stopped; its effect is UNKNOWN. Inspect current state before retrying.]",
    ):
        summary = _summarize("terminal", {"command": "ls"}, _outcome("unknown", message))
        assert "unknown" in summary
        assert "verify effects" in summary
        assert "ran" not in summary


def test_emit_cancelled_payload_now_summarizes_as_not_run():
    # _ToolCallRef.emit_cancelled has emitted {"error": …, "status": "cancelled"} since
    # before this fix; the compressor simply never read the status. Interrupted mid-tool,
    # the call must not compress as a run.
    content = json.dumps({"error": "Tool execution cancelled. Interrupted by the user", "status": "cancelled"})
    summary = _summarize("terminal", {"command": "ls -la"}, content)
    assert "cancelled" in summary
    assert "not run" in summary
    assert "ran" not in summary


def test_structured_outcomes_survive_reworded_prose():
    # The point of the status payload: the compressor reads the outcome structurally, so a
    # future rewording of the prose can no longer make compaction render the call as
    # completed. Same contract test as the rejected-status one in test_rejected_tool_call_summary.
    for status in ("cancelled", "timeout", "unknown"):
        summary = _summarize("frobnicate_xyz", {}, _outcome(status, "Some future reworded wording."))
        assert "not run" in summary or "no result" in summary
        assert "chars result" not in summary


def test_error_result_payload_marks_failure():
    # The dispatch-exception and outer-loop fills wrap raised errors as {"error": …} via
    # error_result(); the verb-summarizers must read the failure, not the verb.
    content = json.dumps({"error": "Error executing tool 'terminal': ValueError('boom')"})
    summary = _summarize("terminal", {"command": "ls"}, content)
    assert "FAILED" in summary
    assert "ran" not in summary


# --- Legacy transcripts: bare prose, no status field ---------------------------------------


def test_legacy_timeout_prose_summarizes_as_never_completed():
    summary = _summarize("terminal", {"command": "ls"}, "Error executing tool 'terminal': timed out after 420.0s")
    assert "unknown" in summary
    assert "do not report it as done" in summary
    assert "ran" not in summary


def test_legacy_skip_prose_summarizes_as_never_run():
    summary = _summarize("browser", {"url": "x"}, "[Tool execution skipped — browser was skipped. Interrupted by the user.]")
    assert "cancelled" in summary
    assert "not run" in summary


def test_legacy_orphan_prose_summarizes_effect_unknown():
    summary = _summarize(
        "terminal",
        {"command": "ls"},
        "[Orphan recovery: interrupted side-effecting tool may have executed; its effect is UNKNOWN. Inspect state before retrying.]",
    )
    assert "unknown" in summary
    assert "verify effects" in summary
    assert "ran" not in summary


def test_legacy_outer_loop_error_fill_summarizes_as_never_completed():
    # turn_loop_errors appends this for tool calls that never got an answer when the turn died.
    summary = _summarize("browser", {"url": "x"}, "Error executing tool: gateway connection lost")
    assert "unknown" in summary
    assert "do not report it as done" in summary


# --- Verb-summarizer failure-first: an error payload must never read as success -------------


def test_verb_summarizers_report_failure_not_success():
    cases = {
        "terminal": ('{"command": "ls"}', {"error": "spawn failed", "exit_code": 1, "output": ""}, "ran `ls`"),
        "search_files": ('{"pattern": "x"}', {"error": "index missing"}, "matches"),
        "execute_code": ('{"code": "print(1)"}', {"error": "kernel dead"}, "lines output"),
        "skill_view": ('{"name": "s2docs"}', {"error": "skill not found"}, "chars)"),
    }
    for name, (args, payload, success_fragment) in cases.items():
        summary = _summarize_tool_result(name, args, json.dumps(payload))
        assert "FAILED" in summary, name
        assert success_fragment not in summary, name


# --- Negative controls: real successes are unchanged ---------------------------------------


def test_normal_results_are_untouched():
    summary = _summarize("terminal", {"command": "ls -la"}, json.dumps({"output": "file_a file_b", "exit_code": 0}))
    assert "ran `ls -la`" in summary
    search = _summarize("search_files", {"pattern": "x"}, json.dumps({"matches": [], "total_count": 0}))
    assert "0 matches" in search


def test_refused_summarizer_returns_none_for_success():
    # The refused path must not claim ordinary results either.
    assert _summarize_refused_tool_result("terminal", {}, json.dumps({"output": "ok", "exit_code": 0})) is None
