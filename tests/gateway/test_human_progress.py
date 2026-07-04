from gateway.display_config import resolve_display_setting
from gateway.human_progress import human_tool_progress_message


def test_human_tool_progress_hides_terminal_command_details():
    msg = human_tool_progress_message(
        "terminal",
        {"command": "python -m pytest tests/gateway/test_run.py -q"},
    )

    assert msg == "Запускаю проверку тестами."
    assert "pytest" not in msg
    assert "tests/gateway" not in msg
    assert "terminal" not in msg


def test_human_tool_progress_hides_search_arguments():
    msg = human_tool_progress_message(
        "search_files",
        {"pattern": "SECRET_PATTERN", "path": "/tmp/project", "target": "content"},
    )

    assert msg == "Ищу совпадения внутри файлов."
    assert "SECRET_PATTERN" not in msg
    assert "/tmp/project" not in msg
    assert "search_files" not in msg


def test_human_tool_progress_uses_generic_message_for_unknown_tools():
    msg = human_tool_progress_message(
        "custom_internal_tool",
        {"token": "do-not-leak", "command": "rm -rf /tmp/example"},
    )

    assert msg == "Продолжаю работу над задачей."
    assert "custom_internal_tool" not in msg
    assert "do-not-leak" not in msg
    assert "rm -rf" not in msg


def test_display_config_accepts_human_tool_progress_mode():
    cfg = {"display": {"platforms": {"telegram": {"tool_progress": "human"}}}}

    assert resolve_display_setting(cfg, "telegram", "tool_progress") == "human"
