"""Tests for gateway progress-message dedup using raw (pre-truncation) key."""


def _simulate_dedup(calls, cap=40):
    """Reproduce the fixed dedup logic and return queued items."""
    last = [None]
    count = [0]
    queued = []
    for tool_name, preview in calls:
        raw = preview or ""
        if preview:
            display = (preview[:cap - 3] + "...") if len(preview) > cap else preview
            msg = f"⚙️ {tool_name}: \"{display}\""
        else:
            msg = f"⚙️ {tool_name}..."
        key = f"{tool_name}\x00{raw}"
        if key == last[0]:
            count[0] += 1
            queued.append(("__dedup__", msg, count[0]))
        else:
            last[0] = key
            count[0] = 0
            queued.append(msg)
    return queued


class TestProgressDedup:
    def test_distinct_long_prefix_not_collapsed(self):
        prefix = "cd /home/agent/coding/my-project && "
        calls = [
            ("terminal", prefix + "git log -1"),
            ("terminal", prefix + "git status -s"),
        ]
        result = _simulate_dedup(calls)
        plain = [q for q in result if isinstance(q, str)]
        dedup = [q for q in result if isinstance(q, tuple) and q[0] == "__dedup__"]
        assert len(plain) == 2
        assert len(dedup) == 0

    def test_identical_calls_collapsed(self):
        calls = [("terminal", "echo hello"), ("terminal", "echo hello")]
        result = _simulate_dedup(calls)
        dedup = [q for q in result if isinstance(q, tuple) and q[0] == "__dedup__"]
        assert len(dedup) == 1
        assert dedup[0][2] == 1

    def test_different_tools_same_preview_not_collapsed(self):
        calls = [("bash", "ls"), ("terminal", "ls")]
        result = _simulate_dedup(calls)
        assert all(isinstance(q, str) for q in result)
        assert len(result) == 2

    def test_empty_preview_dedups_with_itself(self):
        calls = [("tool", ""), ("tool", "")]
        result = _simulate_dedup(calls)
        dedup = [q for q in result if isinstance(q, tuple)]
        assert len(dedup) == 1
