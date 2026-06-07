from gateway.run import (
    _client_friendly_gateway_progress_label,
    _format_gateway_long_running_checkpoint,
    _gateway_progress_subject,
)


def test_gateway_progress_subject_prefers_known_source_and_domain():
    assert _gateway_progress_subject("Search YouTube for demo videos") == "YouTube"
    assert _gateway_progress_subject("https://docs.python.org/3/library/pathlib.html") == "docs.python.org"


def test_gateway_progress_subject_extracts_file_path():
    assert _gateway_progress_subject("Read /tmp/project/gateway/run.py:10") == "gateway/run.py"


def test_client_friendly_tool_labels_hide_internal_tool_names():
    assert _client_friendly_gateway_progress_label("web_search", "latest Hermes docs", active=True) == "Searching latest Hermes docs"
    assert _client_friendly_gateway_progress_label("terminal", "pytest tests/gateway", active=False) == "Ran the tests"
    assert _client_friendly_gateway_progress_label("task_open", "Update task", active=True) == ""


def test_long_running_checkpoint_formats_current_and_completed_work():
    text = _format_gateway_long_running_checkpoint(
        15,
        current=["Searching docs.python.org"],
        completed=[
            "Read gateway/run.py",
            "Ran pytest",
            "Ran pytest",
        ],
    )

    assert text.splitlines()[0] == "⏱ 15-minute check-in"
    assert "Working on\n→ Searching docs.python.org" in text
    assert text.count("✓ Ran pytest") == 1
    assert "Progress so far" in text


def test_long_running_checkpoint_uses_activity_when_current_is_empty():
    text = _format_gateway_long_running_checkpoint(
        5,
        activity={"last_activity_desc": "waiting for stream response"},
    )

    assert "→ Waiting for the answer to start" in text
