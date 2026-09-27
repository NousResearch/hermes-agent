"""Tests for /compress here [N] — boundary-aware partial compression.

Verifies the CLI handler (_manual_compress) splits the history, compresses
only the head, and re-appends the verbatim tail. Inspired by Claude Code's
Rewind "Summarize up to here" action (v2.1.139, May 2026).
"""

from unittest.mock import MagicMock, patch

from tests.hermes_cli.test_cli_init import _make_cli


def _make_history() -> list[dict[str, str]]:
    # 8 messages = 4 exchanges.
    h: list[dict[str, str]] = []
    for i in range(4):
        h.append({"role": "user", "content": f"u{i}"})
        h.append({"role": "assistant", "content": f"a{i}"})
    return h


def _wire_agent(shell, compressed_head):
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = ""
    shell.agent.session_id = None
    shell.agent.tools = None
    shell.agent._compress_context.return_value = (compressed_head, "")
    shell.agent._compression_skipped_due_to_lock = False


def test_compress_here_reappends_verbatim_tail(capsys):
    """The most recent exchanges are preserved verbatim after the summary."""
    shell = _make_cli()
    history = _make_history()
    shell.conversation_history = history
    # Head compresses to an assistant-role summary so the seam
    # (assistant -> user tail) is already valid — tail rides along whole.
    summary = [{"role": "assistant", "content": "[summary]"}]
    _wire_agent(shell, summary)

    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100):
        shell._manual_compress("/compress here 2")

    # Only the head (everything before the last 2 user-starts) is summarized.
    assert shell.agent._compress_context.call_args.args[0] == history[:4]
    # Result = compressed head + verbatim tail (last 2 exchanges).
    assert shell.conversation_history == summary + history[4:]
    # No consecutive same-role user/assistant messages anywhere.
    roles = [m["role"] for m in shell.conversation_history
             if m["role"] in ("user", "assistant")]
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))
