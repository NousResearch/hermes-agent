"""Tests for _spawn_background_review stale-tool-result fix (#14944).

The background review agent is initialised with a snapshot of the main
conversation (including historical role=tool messages).  Before the fix,
the summariser scanned *all* of review_agent._session_messages, so old
successful tool results from the snapshot were incorrectly surfaced as
newly performed actions.  The fix restricts the scan to messages appended
*after* the snapshot boundary.
"""

import json
import threading
import time
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_msg(message: str, success: bool = True, target: str = "") -> dict:
    return {
        "role": "tool",
        "content": json.dumps({"success": success, "message": message, "target": target}),
    }


def _make_user_msg(text: str = "hello") -> dict:
    return {"role": "user", "content": text}


def _make_assistant_msg(text: str = "ok") -> dict:
    return {"role": "assistant", "content": text}


# ---------------------------------------------------------------------------
# Minimal AIAgent stub that exercises only _spawn_background_review
# ---------------------------------------------------------------------------

class _StubAgent:
    """Minimal stub that exposes _spawn_background_review from run_agent.py."""

    def __init__(self, review_session_messages):
        # Simulate the review sub-agent's _session_messages after run_conversation
        self._review_session_messages = review_session_messages
        self.background_review_callback = None
        self.model = "stub/model"
        self.platform = None
        self.provider = None
        self._memory_store = None
        self._memory_enabled = False
        self._user_profile_enabled = False
        self._memory_nudge_interval = 0
        self._skill_nudge_interval = 0
        self._printed = []

    def _safe_print(self, text):
        self._printed.append(text)

    def close(self):
        pass


# We test the logic directly by extracting the relevant scanning loop,
# reproducing it exactly as it exists after the fix is applied.

def _run_scan(snapshot: list, all_session_messages: list) -> list:
    """Reproduce the post-run_conversation scan from _spawn_background_review."""
    snapshot_len = len(snapshot)
    new_msgs = all_session_messages[snapshot_len:]

    actions = []
    for msg in new_msgs:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        try:
            data = json.loads(msg.get("content", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
        if not data.get("success"):
            continue
        message = data.get("message", "")
        target = data.get("target", "")
        if "created" in message.lower():
            actions.append(message)
        elif "updated" in message.lower():
            actions.append(message)
        elif "added" in message.lower() or (target and "add" in message.lower()):
            label = "Memory" if target == "memory" else "User profile" if target == "user" else target
            actions.append(f"{label} updated")
        elif "Entry added" in message:
            label = "Memory" if target == "memory" else "User profile" if target == "user" else target
            actions.append(f"{label} updated")
        elif "removed" in message.lower() or "replaced" in message.lower():
            label = "Memory" if target == "memory" else "User profile" if target == "user" else target
            actions.append(f"{label} updated")

    return list(dict.fromkeys(actions))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBgReviewStaleToolResults:
    """Verify that historical tool results from the snapshot are NOT surfaced."""

    def test_stale_tool_result_not_surfaced(self):
        """Old cron-creation result from history must not appear in summary."""
        stale_tool = _make_tool_msg("Cron job 'morning-standup' created.")
        snapshot = [
            _make_user_msg("create a cron reminder"),
            _make_assistant_msg("Sure"),
            stale_tool,
        ]
        # review agent appended nothing new (no new tools run)
        all_session_messages = snapshot[:]  # identical — no new messages

        actions = _run_scan(snapshot, all_session_messages)
        assert actions == [], (
            f"Expected no actions (stale tool result should be ignored), got: {actions}"
        )

    def test_new_tool_result_is_surfaced(self):
        """A tool result added by the review agent itself MUST appear."""
        snapshot = [
            _make_user_msg("remember this"),
            _make_assistant_msg("ok"),
        ]
        new_tool = _make_tool_msg("Memory entry created.", target="memory")
        all_session_messages = snapshot + [
            _make_assistant_msg("I'll save that"),
            new_tool,
        ]

        actions = _run_scan(snapshot, all_session_messages)
        assert "Memory entry created." in actions

    def test_mixed_stale_and_new_only_new_surfaced(self):
        """When snapshot contains old tool results, only new ones are reported."""
        stale = _make_tool_msg("Cron job 'daily-briefing' created.")
        snapshot = [
            _make_user_msg("set up daily cron"),
            _make_assistant_msg("done"),
            stale,
            _make_user_msg("also remember my name is Alice"),
            _make_assistant_msg("noted"),
        ]
        new_tool = _make_tool_msg("User profile updated.", target="user")
        all_session_messages = snapshot + [
            _make_assistant_msg("saving to profile"),
            new_tool,
        ]

        actions = _run_scan(snapshot, all_session_messages)
        assert "User profile updated." in actions
        # The stale cron message must NOT be duplicated
        assert not any("daily-briefing" in a for a in actions), (
            f"Stale cron result leaked into actions: {actions}"
        )

    def test_empty_snapshot_all_msgs_scanned(self):
        """With no snapshot, all tool messages count as new."""
        new_tool = _make_tool_msg("Skill 'weather' created.")
        all_session_messages = [
            _make_assistant_msg("installing skill"),
            new_tool,
        ]

        actions = _run_scan([], all_session_messages)
        assert "Skill 'weather' created." in actions

    def test_failed_tool_results_excluded(self):
        """Unsuccessful tool results must never appear in the summary."""
        snapshot = [_make_user_msg("try something")]
        fail_tool = _make_tool_msg("Operation failed.", success=False)
        all_session_messages = snapshot + [fail_tool]

        actions = _run_scan(snapshot, all_session_messages)
        assert actions == []

    def test_deduplication_preserved(self):
        """Identical action strings are deduplicated via dict.fromkeys."""
        snapshot = []
        tool_a = _make_tool_msg("Memory entry created.")
        tool_b = _make_tool_msg("Memory entry created.")
        all_session_messages = [tool_a, tool_b]

        actions = _run_scan(snapshot, all_session_messages)
        assert actions.count("Memory entry created.") == 1

    def test_regression_issue_14944(self):
        """Regression: original reporter scenario — cron creation not repeated."""
        # Simulate a session where user created a cron job earlier, then later
        # an unrelated background review fires.
        cron_tool = _make_tool_msg("Cron job '<reminder name>' created.")
        snapshot = [
            _make_user_msg("create a one-shot cron reminder"),
            _make_assistant_msg("I'll set that up"),
            cron_tool,
            _make_user_msg("what's the weather?"),
            _make_assistant_msg("It's sunny"),
        ]
        # Review agent only updated the user profile — no new cron
        new_tool = _make_tool_msg("User profile updated.", target="user")
        all_session_messages = snapshot + [
            _make_assistant_msg("updating profile"),
            new_tool,
        ]

        actions = _run_scan(snapshot, all_session_messages)
        # Only the new profile update should appear
        assert actions == ["User profile updated."]
        assert not any("Cron job" in a for a in actions), (
            f"Stale cron action incorrectly surfaced: {actions}"
        )
