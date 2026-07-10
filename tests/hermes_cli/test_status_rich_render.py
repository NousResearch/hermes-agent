import argparse
from unittest.mock import patch, MagicMock

from pytest import MonkeyPatch
from hermes_cli.status import show_status, check_mark

def test_check_mark_rendering():
    """
    Test that the check mark helper returns the correct rich Text object
    with the proper characters.
    """
    true_mark = check_mark(True)
    assert true_mark.plain == "✓", "Expected a check mark character for True"

    false_mark = check_mark(False)
    assert false_mark.plain == "✗", "Expected an X character for False"


@patch("hermes_cli.status.rich.print")
@patch("hermes_cli.status.load_config")
@patch("hermes_cli.status.get_env_path")
def test_show_status_sections_rendered(mock_get_env_path, mock_load_config, mock_print):
    """
    Test that the base show_status command calls rich.print with the
    correctly formatted section headers and title frames.
    """
    mock_get_env_path.return_value.exists.return_value = False
    mock_load_config.return_value = {}

    args = argparse.Namespace(deep=False)

    show_status(args)

    printed_args = [
        str(call.args[0])
        for call in mock_print.call_args_list
        if call.args
    ]
    joined_output = "\n".join(printed_args)

    assert "⚕ Hermes Agent Status" in joined_output

    assert "◆ Environment" in joined_output
    assert "◆ API Keys" in joined_output
    assert "◆ Auth Providers" in joined_output
    assert "◆ API-Key Providers" in joined_output
    assert "◆ Terminal Backend" in joined_output
    assert "◆ Messaging Platforms" in joined_output
    assert "◆ Gateway Service" in joined_output
    assert "◆ Scheduled Jobs" in joined_output
    assert "◆ Sessions" in joined_output

    assert "◆ Deep Checks" not in joined_output


@patch("hermes_cli.status.rich.print")
@patch("hermes_cli.status.load_config")
@patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-mock-key"})
def test_show_status_deep_checks_rendered(mock_load_config, mock_print):
    """
    Test that the deep checks section is rendered exclusively when the
    deep parameter is passed in the args.
    """
    mock_load_config.return_value = {}
    args = argparse.Namespace(deep=True)

    with patch("httpx.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)

        show_status(args)

    printed_args = [
        str(call.args[0])
        for call in mock_print.call_args_list
        if call.args
    ]
    joined_output = "\n".join(printed_args)

    assert "◆ Deep Checks" in joined_output
    assert "OpenRouter:" in joined_output
    assert "Port 18789:" in joined_output


@patch("hermes_cli.status.rich.print")
def test_show_status_footer_rendered(mock_print):
    """
    Test that the diagnostic footers are correctly appended at the very end.
    """
    args = argparse.Namespace(deep=False)
    show_status(args)

    printed_args = [
        str(call.args[0])
        for call in mock_print.call_args_list
        if call.args
    ]
    joined_output = "\n".join(printed_args)

    assert "Run 'hermes doctor' for detailed diagnostics" in joined_output
    assert "Run 'hermes setup' to configure" in joined_output