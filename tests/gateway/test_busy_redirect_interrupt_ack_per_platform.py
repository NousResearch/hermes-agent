"""Regression test: per-platform busy-ack opt-outs for redirect and interrupt.

The gateway busy-input path echoes one of three visible confirmation
bubbles when a user message arrives mid-turn:

  - ``⏩ Steered into current run``  (busy_input_mode=steer)
  - ``↪ Redirected current run``    (busy_input_mode=interrupt + redirect succeeded)
  - ``⚡ Interrupting current task`` (busy_input_mode=interrupt, no redirect)

The steer variant already had a per-platform toggle
(``display.platforms.<platform>.busy_steer_ack_enabled``,
``HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED``). The redirect and interrupt
variants were unconditional, so Matrix group rooms (and any other chat
where the prefix is noise adjacent to the actual reply) had no way to
silence the bubble without disabling the entire busy-ack channel.

This test pins the new per-platform opt-outs:

  - ``busy_redirect_ack_enabled``   (default True, set False to silence)
  - ``busy_interrupt_ack_enabled``  (default True, set False to silence)

The underlying dispatch still happens — only the confirmation echo is
suppressed. A suppressed ack returns ``True`` (handled) so the base
adapter does not fall through to the cold path.
"""

from __future__ import annotations

import sys
import threading
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# Minimal telegram stubs so gateway imports cleanly (mirrors sibling tests).
_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.platforms.base import (  # noqa: E402
    MessageEvent,
    MessageType,
    SessionSource,
    build_session_key,
)
from gateway.run import GatewayRunner  # noqa: E402


def _make_user_event(platform_value: str = "matrix") -> MessageEvent:
    source = SessionSource(
        platform=MagicMock(value=platform_value),
        chat_id="!room:example.org",
        chat_type="group",
        user_id="@user:example.org",
    )
    return MessageEvent(
        text="hold on, do it this way",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg1",
    )


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner.adapters = {}
    runner.config = MagicMock()
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    # No active subagents so interrupt mode is not demoted to queue (#30170).
    runner._agent_has_active_subagents = MagicMock(return_value=False)
    return runner


def _make_adapter(platform_value: str = "matrix") -> MagicMock:
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = MagicMock(value=platform_value)
    return adapter


def _make_running_parent(*, supports_redirect: bool) -> MagicMock:
    parent = MagicMock()
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent.get_activity_summary.return_value = {
        "api_call_count": 2,
        "max_iterations": 60,
        "current_tool": "terminal",
    }
    parent._supports_active_turn_redirect = supports_redirect
    if supports_redirect:
        parent.redirect = MagicMock(return_value=True)
    return parent


@pytest.mark.asyncio
async def test_redirect_ack_suppressed_when_disabled(monkeypatch) -> None:
    """busy_redirect_ack_enabled=False must silence the ↪ bubble.

    The redirect itself still lands in the running turn; only the
    confirmation echo is suppressed, and the function returns True
    so the base adapter treats the event as handled.
    """
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_REDIRECT_ACK_ENABLED", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_INTERRUPT_ACK_ENABLED", raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")

    runner = _make_runner()
    adapter = _make_adapter()
    event = _make_user_event()
    sk = build_session_key(event.source)
    parent = _make_running_parent(supports_redirect=True)
    runner._running_agents[sk] = parent
    runner.adapters[event.source.platform] = adapter

    # Inject gateway config: redirect ack disabled for matrix.
    from gateway import run as run_mod

    monkeypatch.setattr(
        run_mod,
        "_load_gateway_config",
        lambda: {
            "display": {
                "platforms": {
                    "matrix": {
                        "busy_redirect_ack_enabled": False,
                    }
                }
            }
        },
    )

    handled = await runner._handle_active_session_busy_message(event, sk)

    # The redirect was applied; the function still returns True.
    assert handled is True
    parent.redirect.assert_called_once()
    # But the ↪ confirmation bubble was NOT sent.
    adapter._send_with_retry.assert_not_called()
    # And the running turn was not aborted by interrupt() (redirect path
    # skips the interrupt() call entirely).
    parent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_redirect_ack_sent_by_default(monkeypatch) -> None:
    """Default config (busy_redirect_ack_enabled=True) keeps the ↪ bubble.

    Pins back-compat: existing Telegram / Discord / Slack deployments
    that rely on the bubble keep seeing it after the upgrade.
    """
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_REDIRECT_ACK_ENABLED", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_INTERRUPT_ACK_ENABLED", raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")

    runner = _make_runner()
    adapter = _make_adapter()
    event = _make_user_event()
    sk = build_session_key(event.source)
    parent = _make_running_parent(supports_redirect=True)
    runner._running_agents[sk] = parent
    runner.adapters[event.source.platform] = adapter

    from gateway import run as run_mod

    monkeypatch.setattr(run_mod, "_load_gateway_config", lambda: {})

    handled = await runner._handle_active_session_busy_message(event, sk)

    assert handled is True
    assert adapter._send_with_retry.await_count == 1
    sent = adapter._send_with_retry.await_args.kwargs["content"]
    assert "↪ Redirected current run" in sent


@pytest.mark.asyncio
async def test_interrupt_ack_suppressed_when_disabled(monkeypatch) -> None:
    """busy_interrupt_ack_enabled=False must silence the ⚡ bubble.

    Plain interrupt (no redirect) path: the running turn is still
    aborted via running_agent.interrupt(), but the ⚡ confirmation
    bubble is suppressed.
    """
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_REDIRECT_ACK_ENABLED", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_INTERRUPT_ACK_ENABLED", raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")

    runner = _make_runner()
    adapter = _make_adapter()
    event = _make_user_event()
    sk = build_session_key(event.source)
    # Parent WITHOUT redirect support so the path falls into plain interrupt.
    parent = _make_running_parent(supports_redirect=False)
    runner._running_agents[sk] = parent
    runner.adapters[event.source.platform] = adapter

    from gateway import run as run_mod

    monkeypatch.setattr(
        run_mod,
        "_load_gateway_config",
        lambda: {
            "display": {
                "platforms": {
                    "matrix": {
                        "busy_interrupt_ack_enabled": False,
                    }
                }
            }
        },
    )

    handled = await runner._handle_active_session_busy_message(event, sk)

    assert handled is True
    # The running turn WAS interrupted.
    parent.interrupt.assert_called_once()
    # But the ⚡ confirmation bubble was NOT sent.
    adapter._send_with_retry.assert_not_called()


@pytest.mark.asyncio
async def test_interrupt_ack_env_var_overrides_config(monkeypatch) -> None:
    """HERMES_GATEWAY_BUSY_INTERRUPT_ACK_ENABLED=false must silence even if config says True.

    Pin the env-var escape hatch so a misconfigured config.yaml can
    still be overridden at the process level (mirrors how the steer
    toggle already works).
    """
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_INTERRUPT_ACK_ENABLED", "false")

    runner = _make_runner()
    adapter = _make_adapter()
    event = _make_user_event()
    sk = build_session_key(event.source)
    parent = _make_running_parent(supports_redirect=False)
    runner._running_agents[sk] = parent
    runner.adapters[event.source.platform] = adapter

    from gateway import run as run_mod

    # Config would normally allow the bubble, but the env var wins.
    monkeypatch.setattr(
        run_mod,
        "_load_gateway_config",
        lambda: {
            "display": {
                "platforms": {
                    "matrix": {"busy_interrupt_ack_enabled": True}
                }
            }
        },
    )

    handled = await runner._handle_active_session_busy_message(event, sk)

    assert handled is True
    parent.interrupt.assert_called_once()
    adapter._send_with_retry.assert_not_called()
