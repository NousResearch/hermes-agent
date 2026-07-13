from gateway.heartbeat_status import format_long_running_heartbeat


def test_rich_heartbeat_includes_iteration_todo_tool_and_latest_activity():
    text = format_long_running_heartbeat(
        29 * 60,
        {
            "api_call_count": 51,
            "max_iterations": 120,
            "current_todo": {
                "content": "Patch heartbeat bubble with richer live details",
                "status": "in_progress",
                "elapsed_seconds": 125.0,
            },
            "current_tool": "terminal",
            "current_tool_preview": "pytest tests/gateway/test_heartbeat_status.py -q",
            "current_tool_elapsed": 90.0,
            "last_completed_tool": {
                "name": "read_file",
                "duration": 0.4,
                "is_error": False,
                "completed_at": 100.0,
            },
            "recent_tool_activity": [
                {"label": "search_files", "duration": 2.0, "state": "done"},
                {"label": "read_file", "duration": 0.4, "state": "done"},
                {"label": "pytest tests/gateway/test_heartbeat_status.py -q", "duration": 90.0, "state": "running"},
            ],
        },
        want_iteration_detail=True,
        now=130.0,
    )

    assert text.splitlines()[0] == "⏳ Working — 29m"
    assert "• iteration: `51/120`" in text
    assert "• todo: `Patch heartbeat bubble with richer live details` · 2m 5s" in text
    assert "• tool: `terminal` · 1m 30s" in text
    assert "• last:" not in text
    assert "• doing: running · `pytest tests/gateway/test_heartbeat_status.py -q` · 1m 30s" in text
    assert "▶" not in text
    assert "search_files" not in text
    assert "read_file" not in text


def test_heartbeat_renders_one_latest_completed_activity_line():
    text = format_long_running_heartbeat(
        123,
        {
            "recent_tool_activity": [
                {"label": "patch", "duration": 1.0, "state": "done"},
                {"label": "terminal", "duration": 0.0, "state": "done"},
                {"label": "patch", "duration": 1.0, "state": "done"},
                "ignored trailing activity",
            ],
        },
    )

    doing_lines = [line for line in text.splitlines() if line.startswith("• doing:")]
    assert doing_lines == ["• doing: done · `patch` · took 1s"]
    assert "▶" not in text


def test_heartbeat_elapsed_seconds_use_human_units():
    text = format_long_running_heartbeat(90, {}, want_iteration_detail=False)

    assert text.splitlines()[0] == "⏳ Working — 1m 30s"


def test_heartbeat_omits_iteration_when_busy_detail_disabled():
    text = format_long_running_heartbeat(
        1,
        {
            "api_call_count": 2,
            "max_iterations": 10,
            "last_activity_desc": "waiting for provider response (streaming)",
        },
        want_iteration_detail=False,
    )

    assert "iteration" not in text
    assert "• status: `waiting for provider response (streaming)`" in text


def test_heartbeat_is_bounded_for_long_previews():
    text = format_long_running_heartbeat(
        3,
        {
            "current_tool": "terminal",
            "current_tool_preview": "x" * 1000,
            "current_todo": {"content": "y" * 1000, "status": "in_progress"},
        },
        want_iteration_detail=True,
    )

    assert len(text) <= 900
    assert len(text.splitlines()) <= 9
