"""Tests for /compress using _busy_command — input blocking during compression."""

from unittest.mock import MagicMock, patch

from tests.cli.test_cli_init import _make_cli


def _make_history() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]


def test_compress_sets_command_running(capsys):
    """_compress runs inside _busy_command — _command_running is True during execution."""
    shell = _make_cli()
    history = _make_history()
    shell.conversation_history = history
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = ""
    shell.agent._compress_context.return_value = (list(history), "")

    with patch("agent.model_metadata.estimate_messages_tokens_rough", return_value=100):
        shell._manual_compress("/compress")

    # After _busy_command exits, _command_running should be False
    assert shell._command_running is False
    assert shell._command_status == ""


def test_slow_command_status_compress():
    """_slow_command_status returns a meaningful message for /compress."""
    shell = _make_cli()
    assert shell._slow_command_status("/compress") == "Compressing context..."
    assert shell._slow_command_status("/compress database") == "Compressing context..."


def test_slow_command_status_default_fallback():
    """Unknown commands get the generic fallback."""
    shell = _make_cli()
    assert shell._slow_command_status("/unknown") == "Processing command..."


def test_busy_command_context_manager(capsys):
    """_busy_command sets and clears _command_running, prints status."""
    shell = _make_cli()
    assert shell._command_running is False

    with shell._busy_command("Testing busy"):
        assert shell._command_running is True
        assert shell._command_status == "Testing busy"

    # After exiting context manager, state is cleared
    assert shell._command_running is False
    assert shell._command_status == ""

    # Status was printed
    output = capsys.readouterr().out
    assert "Testing busy" in output


def test_busy_command_clears_on_exception():
    """_busy_command clears state even if the inner code raises."""
    shell = _make_cli()

    try:
        with shell._busy_command("This will fail"):
            assert shell._command_running is True
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass

    # State must be cleaned up
    assert shell._command_running is False
    assert shell._command_status == ""
