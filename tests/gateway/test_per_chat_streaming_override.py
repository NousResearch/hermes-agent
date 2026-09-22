"""Per-chat streaming overrides resolve at the turn's streaming gate.

display.platforms.<platform>.chats.<chat_id>.streaming must reach BOTH call sites
that resolve streaming per turn: the main turn runner (run_turn_runner) and the
proxy stream consumer (run_turn._proxy_stream_consumer). The resolver has supported
chat-scoped lookups since the per-chat display overrides feature (#8e05a066d4), but
the streaming call sites passed no ``chat_id`` — a chat-scoped ``streaming: false``
silently resolved to platform/global defaults while every other display key honored
it. A partner group configured for final-answer-first still streamed every mid-turn
text segment as its own bubble.

Contracts:
- resolver-level: a chat-scoped streaming override wins for that chat; other chats
  on the same platform keep the platform-wide value.
- wiring-level: both streaming call sites thread the originating chat_id and honor
  the override — a chat-scoped streaming:false must stop the gate BEFORE adapter
  lookup, while a chat without an override passes through to adapter lookup.
"""

from __future__ import annotations

import types

import pytest

_GROUP = "-1001234567890"
_DM = "1234567890"


def _cfg_with_chat_override():
    return {
        "display": {
            "platforms": {
                "telegram": {
                    "streaming": True,
                    "chats": {_GROUP: {"streaming": False}},
                }
            }
        }
    }


def test_chat_scoped_streaming_override_wins_over_platform():
    from gateway.display_config import resolve_display_setting

    user_config = _cfg_with_chat_override()
    # The overridden chat resolves to its own value.
    assert resolve_display_setting(
        user_config, "telegram", "streaming", chat_id=_GROUP
    ) is False
    # Another chat on the same platform keeps the platform-wide value.
    assert resolve_display_setting(
        user_config, "telegram", "streaming", chat_id=_DM
    ) is True


def _runner_fake(user_config, chat_id):
    """Minimal fake TurnRunner pair (self + ctx) for `_setup_stream_consumer`."""
    from gateway.display_config import resolve_display_setting

    ctx = types.SimpleNamespace(
        # Upstream added a scheduled_heartbeat guard to _setup_stream_consumer (2026-09-14
        # lineage, merged 6005aa1f); the real RunContext carries the field (turn_context.py),
        # so this fixture must too or the guard AttributeErrors. Caught by the 9/16 merge audit.
        scheduled_heartbeat=False,
        # Upstream added a mute_notification_reply guard ahead of the streaming resolution
        # (run_turn_runner.py:908, merged 65b5ac6ef9 2026-09-17); real RunContext carries it.
        # Caught by the 9/17 post-restore test run.
        mute_notification_reply=False,
        streaming_tts_consumer_holder=[None],
        user_config=user_config,
        resolve_display_setting=resolve_display_setting,
        source=types.SimpleNamespace(chat_id=chat_id),
        interim_assistant_messages_enabled=False,
        progress_queue=None,
        _status_thread_metadata=None,
        event_message_id=None,
        _run_still_current=lambda: True,
        stream_consumer_holder=[None],
    )
    consulted = []
    self_obj = types.SimpleNamespace(
        _ctx=ctx,
        _runner=types.SimpleNamespace(
            config=types.SimpleNamespace(streaming=None),
            # Upstream renamed the adapter lookup _adapter_for_source ->
            # _delivery_adapter_for (run_turn_runner.py:267, merged 2026-09-21);
            # the AttributeError was swallowed by the runner's own try/except,
            # so the gate silently never opened for the positive cases. Provide
            # BOTH names so the fixture survives a rename from either side.
            _adapter_for_source=lambda source: consulted.append(1) or None,
            _delivery_adapter_for=lambda source: consulted.append(1) or None,
            _build_stream_consumer_config=None,
        ),
        _track_future_cleanup_id=lambda fut: None,
    )
    return self_obj, consulted


def _run_setup(user_config, chat_id):
    """Run the real `_setup_stream_consumer` over the fakes; return (result, gate_opened)."""
    self_obj, consulted = _runner_fake(user_config, chat_id)
    method = TurnRunner._setup_stream_consumer.__get__(self_obj, type(self_obj))
    result = method("telegram")
    return result, bool(consulted)


def test_setup_stream_consumer_respects_chat_scoped_streaming_off():
    """Chat-scoped streaming:false stops the gate before adapter lookup."""
    result, gate_opened = _run_setup(_cfg_with_chat_override(), _GROUP)
    stream_consumer, delta_cb, interim_cb, want_interim = result
    assert stream_consumer is None and delta_cb is None
    assert gate_opened is False, (
        "chat-scoped streaming:false must resolve BEFORE adapter lookup; the "
        "adapter was consulted, so the override did not reach the gate"
    )


def test_setup_stream_consumer_keeps_streaming_for_other_chats():
    """A chat without an override passes the gate (adapter lookup reached)."""
    result, gate_opened = _run_setup(_cfg_with_chat_override(), _DM)
    stream_consumer, delta_cb, interim_cb, want_interim = result
    assert gate_opened is True, (
        "platform streaming True with no chat override must open the streaming "
        "gate and reach adapter lookup"
    )


def _proxy_fake(user_config, chat_id):
    consulted = []
    self_obj = types.SimpleNamespace(
        config=types.SimpleNamespace(streaming=None),
        # Upstream renamed the adapter lookup _adapter_for_source ->
        # _delivery_adapter_for (run_turn.py:2694, merged 2026-09-21). Provide
        # BOTH names so the fixture survives a rename from either side.
        _adapter_for_source=lambda source: consulted.append(1) or None,
        _delivery_adapter_for=lambda source: consulted.append(1) or None,
    )
    from gateway.config import Platform

    source = types.SimpleNamespace(platform=Platform.TELEGRAM, chat_id=chat_id)
    return self_obj, source, consulted


def test_proxy_stream_consumer_respects_chat_scoped_streaming_off(monkeypatch):
    """Proxy path: chat-scoped streaming:false suppresses the consumer pre-adapter."""
    import gateway.run_turn as run_turn_mod

    monkeypatch.setattr(
        "gateway.run._load_gateway_config", lambda *a, **k: _cfg_with_chat_override()
    )
    self_obj, source, consulted = _proxy_fake(_cfg_with_chat_override(), _GROUP)
    method = run_turn_mod.GatewayTurnMixin._proxy_stream_consumer.__get__(
        self_obj, type(self_obj)
    )
    assert method(source, None, None, lambda: True) is None
    assert not consulted, "chat-scoped streaming:false must stop before adapter lookup"


def test_proxy_stream_consumer_keeps_streaming_for_other_chats(monkeypatch):
    """Proxy path: no chat override -> gate opens, adapter lookup reached."""
    import gateway.run_turn as run_turn_mod

    monkeypatch.setattr(
        "gateway.run._load_gateway_config", lambda *a, **k: _cfg_with_chat_override()
    )
    self_obj, source, consulted = _proxy_fake(_cfg_with_chat_override(), _DM)
    method = run_turn_mod.GatewayTurnMixin._proxy_stream_consumer.__get__(
        self_obj, type(self_obj)
    )
    method(source, None, None, lambda: True)
    assert consulted, (
        "platform streaming True with no chat override must pass the streaming "
        "gate and reach adapter lookup"
    )


from gateway.run_turn_runner import TurnRunner  # noqa: E402  (after fakes defined)