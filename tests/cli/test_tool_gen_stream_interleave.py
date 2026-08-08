from unittest.mock import patch

from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._stream_box_opened = False
    cli_obj._reasoning_box_opened = False
    cli_obj._stream_buf = ""
    return cli_obj


def test_tool_gen_start_skips_transient_line_while_response_stream_is_open():
    cli_obj = _make_cli()
    cli_obj._stream_box_opened = True

    with patch.object(cli_obj, "_flush_stream") as flush_mock, \
         patch.object(cli_obj, "_close_reasoning_box") as close_reasoning_mock, \
         patch("cli._cprint") as cprint_mock:
        cli_obj._on_tool_gen_start("read_file")

    flush_mock.assert_not_called()
    close_reasoning_mock.assert_not_called()
    cprint_mock.assert_not_called()
    assert cli_obj._stream_box_opened is True


def test_tool_gen_start_prints_status_when_no_stream_ui_is_open():
    cli_obj = _make_cli()

    with patch("agent.display.get_tool_emoji", return_value="📄"), \
         patch("cli._cprint") as cprint_mock:
        cli_obj._on_tool_gen_start("read_file")

    cprint_mock.assert_called_once_with("  ┊ 📄 preparing read_file…")
