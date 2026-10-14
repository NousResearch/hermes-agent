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
    from agent.conversation_loop import _invalid_tool_name_error_content
    return _invalid_tool_name_error_content("", {"terminal", "read_file"})


def test_empty_name_rejection_summarizes_as_not_run():
    summary = _summarize_tool_result("", "{}", _rejected_content())
    assert "rejected" in summary
    assert "not run" in summary
    assert "do not report it as done" in summary
    # It must NOT read as a completed action.
    assert "chars result" not in summary


def test_wrong_name_rejection_summarizes_as_not_run():
    from agent.conversation_loop import _invalid_tool_name_error_content
    content = _invalid_tool_name_error_content("frobnicate_xyz", {"terminal", "read_file"})
    summary = _summarize_tool_result("frobnicate_xyz", "{}", content)
    assert "rejected" in summary
    assert "not run" in summary
    assert "chars result" not in summary


def test_structured_rejection_survives_reworded_prose():
    # The point of the ``status: rejected`` payload: the compressor reads the outcome
    # structurally, so a future rewording of the prose can no longer make compaction
    # render the call as completed. Unrecognized wording must still compress as not-run.
    from agent.tool_rejection import rejected_result

    content = rejected_result("Some future reworded wording the prose matcher cannot know.")
    summary = _summarize_tool_result("frobnicate_xyz", "{}", content)
    assert "rejected" in summary
    assert "not run" in summary
    assert "do not report it as done" in summary
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
    """Contract test, both directions.

    Structured (live): every rejection the dispatch loop emits now carries
    ``status: rejected`` (agent.tool_rejection.rejected_result), and the
    compressor reads that field — the outcome survives any prose rewording.

    Legacy: transcripts recorded before the structured status carry bare
    rejection prose, matched by the prefix matcher built from the shared
    ``agent.tool_rejection`` constants.

    Historically the two sides were coupled only by message text, so a rewording
    could silently stop the compressor from flagging a rejection, with each
    side's tests green. This test feeds real producer output and representative
    constant-built continuations through the real summarizer so any divergence —
    a dropped ``status`` field or a broken legacy fallback — fails here first.
    """
    import json

    from agent.context_compressor import _summarize_refused_tool_result
    from agent.conversation_loop import _invalid_tool_name_error_content
    from agent.tool_rejection import (
        INVALID_JSON_ARGS_PREFIX,
        LEGACY_SKIPPED_SIBLING_INVALID_NAME_PREFIX,
        REJECTED_STATUS,
        SKIPPED_SIBLING_INVALID_JSON_PREFIX,
        rejected_result,
    )

    valid_names = {"terminal", "read_file", "search_files"}

    def assert_not_run(tool_name: str, content: str) -> str:
        summary = _summarize_tool_result(tool_name, "{}", content)
        assert "not run" in summary, summary
        assert "chars result" not in summary, summary
        return summary

    # --- Structured direction: live producer output. ---
    for content in (
        _invalid_tool_name_error_content("", valid_names),  # empty name
        _invalid_tool_name_error_content("frobnicate_xyz", valid_names),  # unknown name
    ):
        payload = json.loads(content)
        assert payload["status"] == REJECTED_STATUS
        assert isinstance(payload["error"], str) and payload["error"]
        summary = assert_not_run("frobnicate_xyz", content)
        assert "do not report it as done" in summary

    # Invalid-JSON shapes from agent/turn_tool_validation.py's recovery path,
    # wrapped exactly as the producer wraps them: a valid sibling skipped because
    # another call had invalid JSON, and the call whose own arguments were invalid.
    skipped_json = rejected_result(
        SKIPPED_SIBLING_INVALID_JSON_PREFIX + " Nothing was executed for this call — do not report its effect as done; re-issue it."
    )
    summary = assert_not_run("read_file", skipped_json)
    assert "skipped" in summary

    invalid_json = rejected_result(
        INVALID_JSON_ARGS_PREFIX + " Expecting value: line 1 column 1 (char 0). Nothing was executed — the call did not run."
    )
    summary = assert_not_run("read_file", invalid_json)
    assert "invalid JSON" in summary

    # --- Legacy direction: bare prose, no status field (pre-structure transcripts). ---
    for prose in (
        SKIPPED_SIBLING_INVALID_JSON_PREFIX + " Nothing was executed for this call.",
        INVALID_JSON_ARGS_PREFIX + " Expecting value: line 1 column 1 (char 0).",
        LEGACY_SKIPPED_SIBLING_INVALID_NAME_PREFIX + " used an invalid name. Please retry this tool call.",
        "Tool 'frobnicate_xyz' does not exist. Nothing was executed — the call did not run.",
    ):
        assert _rejected_tool_call_outcome(prose) is not None, prose
        assert _summarize_refused_tool_result("read_file", {}, prose) is not None, prose
        summary = _summarize_tool_result("read_file", "{}", prose)
        assert "not run" in summary, summary
