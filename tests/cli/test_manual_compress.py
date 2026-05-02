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
    shell.agent.tools = None
    shell.agent.session_id = shell.session_id  # no-op compression: no split
    shell.agent._compress_context.return_value = (list(history), "")

    def _estimate(messages, **_kwargs):
        assert messages == history
        return 100

    with patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate):
        shell._manual_compress()

    output = capsys.readouterr().out
    assert "No changes from compression" in output
    assert "✅ Compressed" not in output
    assert "Approx request size: ~100 tokens (unchanged)" in output


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
    shell.agent.tools = None
    shell.agent.session_id = shell.session_id  # no-op: no split
    shell.agent._compress_context.return_value = (compressed, "")

    def _estimate(messages, **_kwargs):
        if messages == history:
            return 100
        if messages == compressed:
            return 120
        raise AssertionError(f"unexpected transcript: {messages!r}")

    with patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate):
        shell._manual_compress()

    output = capsys.readouterr().out
    assert "✅ Compressed: 4 → 3 messages" in output
    assert "Approx request size: ~100 → ~120 tokens" in output
    assert "denser summaries" in output


def test_manual_compress_syncs_session_id_after_split():
    """Regression for cli.session_id desync after /compress.

    _compress_context ends the parent session and creates a new child session,
    mutating agent.session_id. Without syncing, cli.session_id still points
    at the ended parent — causing /status, /resume, exit summary, and the
    next end_session() call (e.g. from /resume <id>) to target the wrong row.
    """
    shell = _make_cli()
    history = _make_history()
    old_id = shell.session_id
    new_child_id = "20260101_000000_child1"

    compressed = [
        {"role": "user", "content": "[summary]"},
        history[-1],
    ]
    shell.conversation_history = history
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = ""
    shell.agent.tools = None
    # Simulate _compress_context mutating agent.session_id as a side effect.
    def _fake_compress(*args, **kwargs):
        shell.agent.session_id = new_child_id
        return (compressed, "")
    shell.agent._compress_context.side_effect = _fake_compress
    shell.agent.session_id = old_id  # starts in sync
    shell._pending_title = "stale title"

    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100):
        shell._manual_compress()

    # CLI session_id must now point at the continuation child, not the parent.
    assert shell.session_id == new_child_id
    assert shell.session_id != old_id
    # Pending title must be cleared — titles belong to the parent lineage and
    # get regenerated for the continuation.
    assert shell._pending_title is None


def test_manual_compress_no_sync_when_session_id_unchanged():
    """If compression is a no-op (agent.session_id didn't change), the CLI
    must NOT clear _pending_title or otherwise disturb session state.
    """
    shell = _make_cli()
    history = _make_history()
    shell.conversation_history = history
    shell.agent = MagicMock()
    shell.agent.compression_enabled = True
    shell.agent._cached_system_prompt = ""
    shell.agent.tools = None
    shell.agent.session_id = shell.session_id
    shell.agent._compress_context.return_value = (list(history), "")
    shell._pending_title = "keep me"

    with patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100):
        shell._manual_compress()

    # No split → pending title untouched.
    assert shell._pending_title == "keep me"
