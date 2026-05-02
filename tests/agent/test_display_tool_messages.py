from agent.display import get_cute_tool_message, set_tool_preview_max_len


def teardown_function():
    set_tool_preview_max_len(0)


def test_terminal_completion_preview_collapses_multiline_commands_when_unlimited():
    set_tool_preview_max_len(0)
    command = "python3 - <<'PY'\nprint('hello')\nPY"

    line = get_cute_tool_message("terminal", {"command": command}, 0.2)

    assert "\n" not in line
    assert "python3 - <<'PY' print('hello') PY" in line


def test_terminal_completion_preview_collapses_multiline_commands_before_truncating():
    set_tool_preview_max_len(1)
    command = "python3 - <<'PY'\n" + "print('hello') " * 5 + "\nPY"

    line = get_cute_tool_message("terminal", {"command": command}, 0.2)

    assert "\n" not in line
    assert "..." in line
    assert "print('hello')" in line


def test_path_completion_preview_stays_single_line_when_unlimited():
    set_tool_preview_max_len(0)

    line = get_cute_tool_message("read_file", {"path": "/tmp/a\nweird/path.txt"}, 0.1)

    assert "\n" not in line
    assert "/tmp/a weird/path.txt" in line
