from gateway.tool_status_cards import ToolStatusCards


def test_started_tool_renders_running_card():
    cards = ToolStatusCards(app_name="AI Eng Forge")

    cards.started(
        tool_name="skills_list",
        display_name="Run skills_list",
        summary="📚 Reading skill s2-github-pr-review",
    )

    text = cards.render()
    assert "AI Eng Forge" in text
    assert "Run skills_list" in text
    assert "Running" in text
    assert "Reading skill s2-github-pr-review" in text


def test_completed_tool_updates_existing_row_with_duration():
    cards = ToolStatusCards(app_name="AI Eng Forge")

    cards.started(
        call_id="call-1",
        tool_name="skills_list",
        display_name="Run skills_list",
        summary="📚 Reading skill s2-github-pr-review",
    )
    cards.completed(call_id="call-1", tool_name="skills_list", duration=0.34)

    text = cards.render()
    assert "Run skills_list" in text
    assert "Completed in 0.3s" in text
    assert "Running" not in text
    assert "Reading skill s2-github-pr-review" in text


def test_failed_tool_updates_existing_row_with_error_summary():
    cards = ToolStatusCards(app_name="AI Eng Forge")

    cards.started(
        call_id="call-1",
        tool_name="terminal",
        display_name="Run terminal",
        summary="💻 terminal\n```\ncat /secret\n```",
    )
    cards.failed(
        call_id="call-1",
        tool_name="terminal",
        duration=1.25,
        error="Permission denied while opening /secret/path\ntraceback omitted",
    )

    text = cards.render()
    assert "Run terminal" in text
    assert "Failed in 1.2s" in text
    assert "Permission denied while opening /secret/path" in text
    assert "cat /secret" in text
    assert "traceback omitted" not in text


def test_multiple_tools_preserve_start_order_when_completed_out_of_order():
    cards = ToolStatusCards(app_name="AI Eng Forge")

    cards.started(call_id="call-1", tool_name="search_files", display_name="Run search_files")
    cards.started(call_id="call-2", tool_name="read_file", display_name="Run read_file")
    cards.completed(call_id="call-2", tool_name="read_file", duration=0.1)
    cards.completed(call_id="call-1", tool_name="search_files", duration=0.2)

    text = cards.render()
    assert text.index("Run search_files") < text.index("Run read_file")
    assert "Completed in 0.2s" in text
    assert "Completed in 0.1s" in text


def test_completion_without_call_id_matches_oldest_running_tool_name():
    cards = ToolStatusCards(app_name="AI Eng Forge")

    cards.started(call_id="terminal:1", tool_name="terminal", display_name="Run terminal")
    cards.started(call_id="terminal:2", tool_name="terminal", display_name="Run terminal")
    cards.completed(tool_name="terminal", duration=0.5)

    text = cards.render()
    assert text.count("Run terminal") == 2
    assert text.index("Completed in 0.5s") < text.index("Running")


def test_terminal_detail_preserves_fenced_command_preview():
    cards = ToolStatusCards(app_name="AI Eng Forge")

    cards.started(
        call_id="call-1",
        tool_name="terminal",
        display_name="Run terminal",
        summary="💻 terminal\n```\nset -euo pipefail\nnode --version\n```",
    )

    text = cards.render()
    assert "Run terminal" in text
    assert "```" in text
    assert "set -euo pipefail" in text
    assert "node --version" in text
