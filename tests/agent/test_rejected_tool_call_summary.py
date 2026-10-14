"""Compaction summaries must render rejected tool calls as "not run" — never completed actions.

Observed live (2026-10-13): a call rejected for an empty tool name ("Tool call rejected: the
tool name was empty") never executed, but the next assistant turn narrated its intended
effect as done with invented evidence (fabricated merge SHAs). When compaction later
summarized that window, the rejection compressed into the generic ``[…] (N chars result)``
line — indistinguishable from a completed action — and the confabulated narration entered
the "completed" anchors as fact.

The refused-result summarizer already had this contract for BLOCKED/approval refusals
(``_sum_write_file``'s comment describes exactly this failure class); this extends it to
dispatch-loop rejections so the post-compaction agent re-issues the call instead of
reporting it done.
"""

from __future__ import annotations

import os
import sys

# Repo root = three levels up from tests/agent/<file>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent.context_compressor import _rejected_tool_call_outcome, _summarize_tool_result


def _rejected_content() -> str:
    # Byte-for-byte the empty-name rejection the dispatch loop emits.
    return (
        "Tool call rejected: the tool name was empty. Nothing was executed — the call did "
        "not run, so do not report its effect as done or describe results you did not "
        "receive; re-issue the tool call with a valid name from your tool list."
    )


def test_empty_name_rejection_summarizes_as_not_run():
    summary = _summarize_tool_result("", "{}", _rejected_content())
    assert "rejected" in summary
    assert "not run" in summary
    assert "do not report it as done" in summary
    # It must NOT read as a completed action.
    assert "chars result" not in summary


def test_wrong_name_rejection_summarizes_as_not_run():
    content = (
        "Tool 'frobnicate_xyz' does not exist. Nothing was executed — the call did not run, "
        "so do not report its effect as done; call one of these instead. Available tools: terminal, read_file"
    )
    summary = _summarize_tool_result("frobnicate_xyz", "{}", content)
    assert "rejected" in summary
    assert "not run" in summary
    assert "chars result" not in summary


def test_legacy_batch_skip_summarizes_as_not_run():
    # Pre-fix mixed batches voided the whole turn with this message; old transcripts still carry it.
    content = "Skipped: another tool call in this turn used an invalid name. Nothing was executed."
    summary = _summarize_tool_result("terminal", '{"command": "ls"}', content)
    assert "skipped" in summary
    assert "not run" in summary
    assert "ran" not in summary  # must not render through _sum_terminal as a run


def test_normal_results_are_untouched():
    # Negative control: ordinary results must not be misread as rejections.
    summary = _summarize_tool_result(
        "terminal", '{"command": "ls -la"}', '{"output": "file_a file_b", "exit_code": 0}'
    )
    assert "ran" in summary
    assert "not run" not in summary

    # A tool result that merely QUOTES rejection-shaped text (e.g. grep output about a
    # rejected call) must not collapse into a rejection line either.
    quoted = '{"total_count": 2, "matches": [{"content": "Tool \'x\' does not exist."}]}'
    summary = _summarize_tool_result("search_files", '{"target": "content"}', quoted)
    assert "rejected" not in summary


def test_blocked_result_contract_still_holds():
    # The pre-existing BLOCKED contract must be unchanged by the new branch.
    summary = _summarize_tool_result("terminal", '{"command": "rm -rf /"}', "BLOCKED: destructive command")
    assert "BLOCKED, not run" in summary


def test_real_loop_rejection_wording_matches_compressor_contract():
    """Contract test: the dispatch loop's ACTUAL rejection wording must be recognized by
    the compressor's rejected-outcome matcher.

    The two sides are coupled only by message text — the compressor prefix-matches
    prose, and every other test in this file feeds canned strings — so rewording the
    rejection in ``agent/conversation_loop.py`` would silently stop the compressor from
    flagging it, with the whole suite green. This feeds the real producer's output
    through the real matcher so a rewording fails here first.
    """
    from agent.conversation_loop import _invalid_tool_name_error_content

    valid_names = {"terminal", "read_file", "search_files"}

    # Empty-name rejection: the live shape must match the matcher…
    empty_msg = _invalid_tool_name_error_content("", valid_names)
    assert _rejected_tool_call_outcome(empty_msg) is not None
    # …and must render as not-run through the summarizer.
    summary = _summarize_tool_result("", "{}", empty_msg)
    assert "rejected" in summary
    assert "not run" in summary
    assert "do not report it as done" in summary

    # Unknown-name rejection: the live shape must match the matcher.
    unknown_msg = _invalid_tool_name_error_content("frobnicate_xyz", valid_names)
    assert _rejected_tool_call_outcome(unknown_msg) is not None

    # Batch-skip: the message ``agent/turn_tool_validation.py`` emits for valid siblings
    # of an invalid call. The literal lives inline in that module's batch-error path,
    # so it is duplicated here; it must keep the prefix the compressor matches on.
    skip_msg = (
        "Skipped: another tool call in this turn used an invalid name. "
        "Please retry this tool call."
    )
    assert _rejected_tool_call_outcome(skip_msg) is not None
