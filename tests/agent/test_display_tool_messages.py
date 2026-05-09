from agent.display import get_cute_tool_message, set_tool_preview_max_len


def teardown_function():
    set_tool_preview_max_len(0)


def test_terminal_completion_preview_collapses_and_caps_multiline_commands_by_default():
    set_tool_preview_max_len(0)
    command = "python3 - <<'PY'\n" + "print('hello') " * 8 + "\nPY"

    line = get_cute_tool_message("terminal", {"command": command}, 0.2)

    assert "\n" not in line
    assert "..." in line
    assert "python3 - <<'PY'" in line
    preview = line.split("$         ", 1)[1].rsplit("  0.2s", 1)[0]
    assert len(preview) <= 42


def test_terminal_completion_preview_respects_custom_preview_length():
    set_tool_preview_max_len(80)
    command = "python3 - <<'PY'\n" + "print('hello') " * 8 + "\nPY"

    line = get_cute_tool_message("terminal", {"command": command}, 0.2)

    assert "\n" not in line
    assert "..." in line
    preview = line.split("$         ", 1)[1].rsplit("  0.2s", 1)[0]
    assert 42 < len(preview) <= 80


def test_path_completion_preview_stays_single_line_and_capped_by_default():
    set_tool_preview_max_len(0)
    path = "/tmp/" + "very-long-segment/" * 5 + "a\nweird/path.txt"

    line = get_cute_tool_message("read_file", {"path": path}, 0.1)

    assert "\n" not in line
    assert "..." in line
    preview = line.split("read      ", 1)[1].rsplit("  0.1s", 1)[0]
    assert len(preview) <= 35
    assert preview.endswith("weird/path.txt")


def test_path_completion_preview_respects_custom_preview_length():
    set_tool_preview_max_len(60)
    path = "/tmp/" + "very-long-segment/" * 5 + "a\nweird/path.txt"

    line = get_cute_tool_message("read_file", {"path": path}, 0.1)

    assert "\n" not in line
    preview = line.split("read      ", 1)[1].rsplit("  0.1s", 1)[0]
    assert 35 < len(preview) <= 60
    assert preview.endswith("weird/path.txt")
