"""Regression tests for gateway handling of compacted context summaries."""

from agent.context_compressor import SUMMARY_PREFIX
from gateway.run import _sanitize_compaction_summary_for_current_turn


def test_stale_compaction_active_task_sections_are_removed_when_latest_turn_unrelated():
    """A compacted history summary must not override an unrelated new message."""
    stale_summary = (
        f"{SUMMARY_PREFIX}\n"
        "## Active Task\n"
        "Deploy the TTM dev server and debug LiteLLM/ngrok routing.\n\n"
        "## Completed Actions\n"
        "- Confirmed prior server status.\n\n"
        "## Pending User Asks\n"
        "- Restart the server and report URL.\n\n"
        "## Remaining Work\n"
        "1. Fix the deployment.\n\n"
        "## Relevant Files\n"
        "- /Users/oc_runtime/Development/ttm\n"
    )

    sanitized = _sanitize_compaction_summary_for_current_turn(
        stale_summary,
        "i've seen codex be able to do it before, so there is a way, you can check it",
    )

    assert "CONTEXT COMPACTION" in sanitized
    assert "## Completed Actions" in sanitized
    assert "## Relevant Files" in sanitized
    assert "## Active Task" not in sanitized
    assert "## Pending User Asks" not in sanitized
    assert "## Remaining Work" not in sanitized
    assert "Deploy the TTM dev server" not in sanitized
    assert "Restart the server" not in sanitized
    assert "Fix the deployment" not in sanitized
    assert "Your current task is identified" not in sanitized


def test_related_compaction_active_task_is_preserved_for_same_task_continuation():
    """Same-topic resumption should keep the compacted active-task handoff."""
    summary = (
        f"{SUMMARY_PREFIX}\n"
        "## Active Task\n"
        "Find root cause for cross-chat context compaction contamination in Hermes.\n\n"
        "## Remaining Work\n"
        "1. Patch Hermes source and run regression tests.\n"
    )

    sanitized = _sanitize_compaction_summary_for_current_turn(
        summary,
        "continue the Hermes context compaction root cause fix and tests",
    )

    assert sanitized == summary


def test_explicit_preserved_active_task_list_keeps_compaction_handoff():
    """The platform's preserved task-list marker is an explicit continuation signal."""
    summary = (
        f"{SUMMARY_PREFIX}\n"
        "## Active Task\n"
        "Find root cause/source code path for cross-chat context-compaction contamination.\n"
    )

    sanitized = _sanitize_compaction_summary_for_current_turn(
        summary,
        "[Your active task list was preserved across context compression]\n- [>] ctx-rootcause",
    )

    assert sanitized == summary
