"""Invariants for the shared manual-/compress core (``agent/conversation_compression_manual``)."""

from __future__ import annotations

import copy
import threading
from unittest.mock import MagicMock

import pytest

from agent.conversation_compression_manual import compress_now, parse_compress_args


def _history():
    return [
        {"role": "user", "content": "one"}, {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"}, {"role": "assistant", "content": "four"},
        {"role": "user", "content": "five"}, {"role": "assistant", "content": "six"},
    ]


def _agent():
    agent = MagicMock()
    agent._cached_system_prompt, agent.tools, agent.context_compressor = "sys", None, None
    agent._compression_skipped_due_to_lock = None
    agent._compress_context.return_value = ([{"role": "assistant", "content": "summary"}], "")
    return agent


@pytest.mark.parametrize("raw", ["--preview", "here 1 --preview", "--dry-run keep the tests", "--aggressive --preview"])
def test_preview_leaves_history_and_agent_byte_identical(raw):
    agent, history = _agent(), _history()
    frozen = copy.deepcopy(history)
    result = compress_now(agent, history, parse_compress_args(raw))
    assert result.status == "preview" and result.lines
    assert history == frozen and result.after_messages == frozen
    agent._compress_context.assert_not_called()


def test_compressed_result_rejoins_verbatim_tail_and_never_mutates_input():
    agent, history = _agent(), _history()
    frozen = copy.deepcopy(history)
    result = compress_now(agent, history, parse_compress_args("here 1"))
    assert result.status == "compressed"
    assert agent._compress_context.call_args.args[0] == frozen[:4]  # head only
    assert result.after_messages[-2:] == frozen[4:]  # last exchange verbatim
    assert history == frozen
    assert agent._compress_context.call_args.kwargs["force"] is True


@pytest.mark.parametrize("surface", ["cli", "gateway", "tui", "acp"])
def test_every_surface_honours_preview_without_compressing(surface, monkeypatch):
    """The prompt-cache-breaking mutation must be gated by the same ``--preview`` on all four surfaces."""
    agent, history = _agent(), _history()
    frozen = copy.deepcopy(history)
    if surface == "cli":
        from hermes_cli.cli_session_mixin import CLISessionMixin
        cli = CLISessionMixin.__new__(CLISessionMixin)
        cli.agent, cli.conversation_history = agent, history
        cli._manual_compress("/compress --preview")
        assert cli.conversation_history == frozen
    elif surface == "gateway":
        import asyncio
        from gateway.run import GatewayRunner
        gw = GatewayRunner.__new__(GatewayRunner)
        gw.session_store = MagicMock()
        entry = MagicMock(session_id="sid")
        gw._async_session_store = MagicMock(
            _store=gw.session_store, get_or_create_session=_coro(entry), load_transcript=_coro(history))
        gw._run_manual_compression = MagicMock(side_effect=AssertionError("must not compress"))
        event = MagicMock(); event.get_command_args.return_value = "--preview"
        reply = asyncio.run(gw._handle_compress_command_inner(event))
        assert "Preview" in reply
    elif surface == "tui":
        from tui_gateway.server import _compress_session_history
        session = {"agent": agent, "history": history, "history_lock": threading.Lock(), "history_version": 3}
        assert _compress_session_history(session, "--preview")[0] == 0
        assert session["history"] == frozen and session["history_version"] == 3
    else:
        from acp_adapter.commands import SlashCommandsMixin
        acp = SlashCommandsMixin.__new__(SlashCommandsMixin)
        acp.session_manager = MagicMock()
        state = MagicMock(history=history, agent=agent, session_id="acp-sid")
        assert "Preview" in acp._cmd_compress("--preview", state)
        assert state.history == frozen
        acp.session_manager.save_session.assert_not_called()
    agent._compress_context.assert_not_called()


def test_windowless_gate_is_gateway_only_so_in_process_surfaces_still_reach_compress_context():
    """``has_content_to_compress`` only knows the local summary window; codex_app_server native compaction
    and the phase-1 tool-result prune inside ``_compress_context`` do useful work without one, so CLI/TUI/
    ACP (the default) must not short-circuit on it. The gateway keeps its historical early answer."""
    agent, history = _agent(), _history()
    agent.context_compressor = MagicMock()
    agent.context_compressor.has_content_to_compress.return_value = False
    agent._compress_context.return_value = (history[:-1], "")  # e.g. a pruned tool result, no summary

    default = compress_now(agent, history, parse_compress_args(""))
    assert default.status == "compressed" and default.removed == 1
    agent._compress_context.assert_called_once()

    agent._compress_context.reset_mock()
    gateway = compress_now(agent, history, parse_compress_args(""), system_message="", skip_without_window=True)
    assert gateway.status == "nothing_to_do" and gateway.after_messages == history
    agent._compress_context.assert_not_called()


def _coro(value):
    async def _inner(*_a, **_k):
        return value
    return _inner
