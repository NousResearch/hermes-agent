"""Tests for CLI manual compression messaging."""

from unittest.mock import MagicMock, patch

from tests.cli.test_cli_init import _make_cli


def _make_history() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]


def test_manual_compress_reports_noop_without_success_banner(capsys):
    shell = _make_cli()
    history = _make_history()
    shell.conversation_history = history
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = ""
    shell.agent._compress_context.return_value = (list(history), "")

    def _estimate(messages):
        assert messages == history
        return 100

    with patch("agent.model_metadata.estimate_messages_tokens_rough", side_effect=_estimate):
        shell._manual_compress()

    output = capsys.readouterr().out
    assert "No changes from compression" in output
    assert "✅ Compressed" not in output
    assert "Rough transcript estimate: ~100 tokens (unchanged)" in output


def test_manual_compress_explains_when_token_estimate_rises(capsys):
    shell = _make_cli()
    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "Dense summary that still counts as more tokens."},
        history[-1],
    ]
    shell.conversation_history = history
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = ""
    shell.agent._compress_context.return_value = (compressed, "")

    def _estimate(messages):
        if messages == history:
            return 100
        if messages == compressed:
            return 120
        raise AssertionError(f"unexpected transcript: {messages!r}")

    with patch("agent.model_metadata.estimate_messages_tokens_rough", side_effect=_estimate):
        shell._manual_compress()

    output = capsys.readouterr().out
    assert "✅ Compressed: 4 → 3 messages" in output
    assert "Rough transcript estimate: ~100 → ~120 tokens" in output
    assert "denser summaries" in output
