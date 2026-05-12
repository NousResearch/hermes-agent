"""Tests for gateway progress dedup using pre-truncation keys (issue #24298).

The dedup logic must compare the FULL preview before truncation, not the
truncated display string.  Two commands sharing a long prefix (first ~37
chars) must NOT be collapsed into a single (×N) bubble.
"""

import queue

import pytest

from agent.display import set_tool_preview_max_len


@pytest.fixture(autouse=True)
def _reset_preview_len():
    """Ensure tool preview length is reset between tests."""
    set_tool_preview_max_len(0)  # 0 = use default (40)
    yield
    set_tool_preview_max_len(0)


def _build_progress_callback():
    """Reconstruct the progress_callback closure from gateway/run.py.

    We extract just the dedup-relevant logic to test in isolation without
    spinning up a full gateway + adapter stack.
    """
    progress_queue = queue.Queue()
    last_progress_msg = [None]
    repeat_count = [0]

    def progress_callback(event_type, tool_name=None, preview=None, **kwargs):
        emoji = "⚙"
        if preview:
            from agent.display import get_tool_preview_max_len
            _pl = get_tool_preview_max_len()
            _cap = _pl if _pl > 0 else 40
            # Dedup on the FULL preview BEFORE truncation (issue #24298)
            dedup_key = f'{tool_name}:"{preview}"'
            if len(preview) > _cap:
                preview = preview[:_cap - 3] + "..."
            msg = f'{emoji} {tool_name}: "{preview}"'
        else:
            dedup_key = f"{tool_name}"
            msg = f"{emoji} {tool_name}..."

        if dedup_key == last_progress_msg[0]:
            repeat_count[0] += 1
            progress_queue.put(("__dedup__", msg, repeat_count[0]))
            return
        last_progress_msg[0] = dedup_key
        repeat_count[0] = 0
        progress_queue.put(msg)

    return progress_callback, progress_queue


class TestProgressDedupPreTruncation:
    """Verify dedup uses pre-truncation keys, not truncated messages."""

    def test_identical_previews_are_deduplicated(self):
        """Same command repeated → dedup fires."""
        cb, q = _build_progress_callback()
        cb("tool.started", "terminal", "git rev-parse --abbrev-ref HEAD")
        cb("tool.started", "terminal", "git rev-parse --abbrev-ref HEAD")

        # First message goes through normally
        msg1 = q.get_nowait()
        assert isinstance(msg1, str)
        assert "git rev-parse" in msg1

        # Second is a dedup
        kind, msg, count = q.get_nowait()
        assert kind == "__dedup__"
        assert count == 1

    def test_different_previews_not_deduplicated_despite_shared_prefix(self):
        """Two commands sharing first ~37 chars must NOT be deduped."""
        cb, q = _build_progress_callback()
        # These share a long prefix but are different commands
        preview_a = "cd /home/agent/Coding/my-project && git rev-parse --abbrev-ref HEAD"
        preview_b = "cd /home/agent/Coding/my-project && git log --oneline -5"

        cb("tool.started", "terminal", preview_a)
        cb("tool.started", "terminal", preview_b)

        msg1 = q.get_nowait()
        msg2 = q.get_nowait()

        # Both should be regular messages, not dedup
        assert isinstance(msg1, str)
        assert isinstance(msg2, str)
        # Both previews get truncated to ~37 chars for display,
        # but they must NOT be collapsed into a (×2) dedup entry.
        # Just verify the queue has two separate messages (no dedup).
        assert q.empty()

    def test_truncated_preview_still_shows_correctly(self):
        """Even when previews are truncated for display, the full text
        is used for dedup comparison."""
        cb, q = _build_progress_callback()
        # A long preview that will be truncated
        long_preview = "a" * 100  # 100 chars → truncated to 37...

        cb("tool.started", "terminal", long_preview)
        msg = q.get_nowait()
        assert isinstance(msg, str)
        # Display should be truncated
        assert "..." in msg
        assert "aaa" in msg  # starts with 'a's

    def test_no_preview_no_dedup_for_different_tools(self):
        """Without previews, dedup is by tool_name only."""
        cb, q = _build_progress_callback()
        cb("tool.started", "terminal")
        cb("tool.started", "browser")

        msg1 = q.get_nowait()
        msg2 = q.get_nowait()
        assert "terminal" in msg1
        assert "browser" in msg2
        assert q.empty()

    def test_no_preview_same_tool_is_deduplicated(self):
        """Same tool without preview → dedup fires."""
        cb, q = _build_progress_callback()
        cb("tool.started", "terminal")
        cb("tool.started", "terminal")

        msg1 = q.get_nowait()
        kind, msg, count = q.get_nowait()
        assert kind == "__dedup__"
        assert count == 1

    def test_custom_preview_length_still_dedupes_correctly(self):
        """With a custom (larger) preview length, dedup still works."""
        set_tool_preview_max_len(80)
        cb, q = _build_progress_callback()
        preview_a = "cd /home/agent/Coding/my-project && git rev-parse --abbrev-ref HEAD"
        preview_b = "cd /home/agent/Coding/my-project && git log --oneline -5"

        cb("tool.started", "terminal", preview_a)
        cb("tool.started", "terminal", preview_b)

        msg1 = q.get_nowait()
        msg2 = q.get_nowait()
        assert isinstance(msg1, str)
        assert isinstance(msg2, str)
        assert q.empty()
