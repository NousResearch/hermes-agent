"""Gate interactive clarification on resumable-transport capability (#105097).

A webhook route keys every delivery as an independent one-shot session
(``webhook:{route}:{delivery_id}``) — a reply arrives as a brand-new,
unrelated session, never as a message this adapter can route back to a
pending ``clarify_id``. Registering the wait anyway (the old behavior)
can only ever time out: wasted latency for a result nobody could deliver.

``_clarify_callback_sync`` must check the adapter's
``supports_interactive_clarify`` capability BEFORE registering a wait or
sending the prompt, and short-circuit with the same sentinel a real
timeout returns so the agent proceeds exactly as it already does when
nobody answers in time. Interactive adapters (default True) must be
unaffected.
"""

from unittest.mock import MagicMock

import pytest

from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from tools.clarify_tool import TIMEOUT_RESPONSE


def _make_runner(adapter):
    ctx = TurnContext(_status_adapter=adapter, session_key="webhook:orders:abc123")
    return TurnRunner(MagicMock(), ctx)


class _NonResumableAdapter:
    """Stand-in for WebhookAdapter: capability explicitly False."""

    supports_interactive_clarify = False


class _InteractiveAdapter:
    """Stand-in for a normal chat adapter: no ``supports_interactive_clarify``
    override -> the callback's ``getattr(..., True)`` default applies, same
    as every existing interactive platform (Slack, Telegram, Discord, ...)."""

    def __init__(self):
        self.send_clarify = MagicMock()
        self.pause_typing_for_chat = MagicMock()
        self.resume_typing_for_chat = MagicMock()


def test_non_resumable_adapter_skips_wait_and_returns_timeout_sentinel(monkeypatch):
    runner = _make_runner(_NonResumableAdapter())

    register = MagicMock()
    monkeypatch.setattr("tools.clarify_gateway.register", register)
    schedule = MagicMock()
    monkeypatch.setattr(runner, "_schedule", schedule)

    result = runner._clarify_callback_sync("What's the target env?", None)

    assert result == TIMEOUT_RESPONSE
    register.assert_not_called()
    schedule.assert_not_called()


def test_interactive_adapter_still_registers_and_waits(monkeypatch):
    runner = _make_runner(_InteractiveAdapter())

    register = MagicMock()
    monkeypatch.setattr("tools.clarify_gateway.register", register)
    monkeypatch.setattr(runner, "_schedule", MagicMock(return_value="fut"))
    monkeypatch.setattr(runner, "_close_native_stream_boundary", MagicMock())
    monkeypatch.setattr(runner, "_stream_consumer", MagicMock(return_value=None))
    monkeypatch.setattr(
        "gateway.run._clarify_send_then_wait",
        MagicMock(return_value="user picked B"),
    )

    result = runner._clarify_callback_sync("Pick one", ["A", "B"])

    assert result == "user picked B"
    register.assert_called_once()


@pytest.mark.parametrize("adapter_cls", [_NonResumableAdapter, _InteractiveAdapter])
def test_capability_defaults_true_on_base_adapter(adapter_cls):
    """Sanity: only the non-resumable stand-in opts out; the interactive one
    relies on BasePlatformAdapter's True default, proving the gate is additive."""
    from gateway.platforms.base import BasePlatformAdapter

    if adapter_cls is _InteractiveAdapter:
        assert BasePlatformAdapter.supports_interactive_clarify is True


def test_webhook_adapter_opts_out_of_interactive_clarify():
    from gateway.platforms.webhook import WebhookAdapter

    assert WebhookAdapter.supports_interactive_clarify is False
