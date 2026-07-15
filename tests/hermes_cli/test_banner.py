from unittest.mock import patch

from hermes_cli.banner import cprint


def test_cprint_falls_back_to_plain_print_when_prompt_toolkit_has_no_console(capsys):
    with patch(
        "prompt_toolkit.print_formatted_text",
        side_effect=RuntimeError("no console screen buffer"),
    ):
        cprint("fallback text")

    assert capsys.readouterr().out == "fallback text\n"
