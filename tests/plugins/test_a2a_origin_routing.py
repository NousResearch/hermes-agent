"""
Origin-session routing and peer-origin security tests.

When an A2A context is born in a real gateway session (e.g. a Discord
thread), push-wakes and agent confirmations for that context must deliver to
the ORIGIN's chat/thread — the session that started the exchange — not the
platform home channel (home is only the fallback when no origin exists).

Also ensures a peer's advertised agent-card cannot hijack the push to a
cross-origin URL — the push must stay on the configured peer origin and
must not leak Authorization to the evil URL.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from plugins.platforms.a2a import protocol, security, tools


def _bare_adapter():
    from plugins.platforms.a2a.adapter import A2AAdapter
    from gateway.config import PlatformConfig

    return A2AAdapter(PlatformConfig(enabled=True))


def test_origin_delivery_target_resolves_recorded_thread(monkeypatch, tmp_path):
    """An A2A context born in a Discord thread records the thread as the
    delivery target; other platforms and unknown contexts resolve to nothing
    (home fallback stays the last resort)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _bare_adapter()
    try:
        adapter._register_context_session(
            "ctx-origin-1",
            {
                "platform": "discord",
                "chat_id": "1300000000000000101",
                "chat_type": "thread",
                "thread_id": "1300000000000000101",
                "user_id": "u1",
                "profile": "",
                "session_id": "sid-1",
            },
        )
        from plugins.platforms.a2a.adapter import A2AAdapter

        target = A2AAdapter._origin_delivery_target("ctx-origin-1", "discord")
        assert target["chat_id"] == "1300000000000000101"
        assert target["thread_id"] == "1300000000000000101"
        assert target["chat_type"] == "thread"
        # Wrong platform / unknown context → no origin target (home fallback).
        assert A2AAdapter._origin_delivery_target("ctx-origin-1", "telegram") == {}
        assert A2AAdapter._origin_delivery_target("ctx-unknown", "discord") == {}
    finally:
        adapter._unregister_adapter()


def test_a2a_confirmation_routes_to_origin_thread(monkeypatch, tmp_path):
    """A confirmation emitted from an A2A session with a bare platform
    target resolves to the ORIGIN's Discord thread, not the home channel."""
    import tools.send_message_tool as smt
    from gateway.config import PlatformConfig
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _bare_adapter()
    try:
        adapter._register_context_session(
            "ctx-confirm-1",
            {
                "platform": "discord",
                "chat_id": "1300000000000000101",
                "chat_type": "thread",
                "thread_id": "1300000000000000101",
                "user_id": "u1",
                "profile": "",
                "session_id": "sid-1",
            },
        )
        tokens = set_session_vars(
            platform="a2a",
            chat_id="ctx-confirm-1",
            chat_type="dm",
            user_id="ip:127.0.0.1",
            profile="",
            async_delivery=True,
        )
        try:
            sent: dict = {}

            async def fake_send(
                platform,
                pconfig,
                chat_id,
                message,
                thread_id=None,
                media_files=None,
                force_document=False,
            ):
                sent.update(platform=platform, chat_id=chat_id, thread_id=thread_id)
                return {"success": True}

            monkeypatch.setattr(smt, "_send_to_platform", fake_send)

            fake_config = SimpleNamespace(
                platforms={},
                get_home_channel=lambda p: SimpleNamespace(
                    chat_id="1200000000000000200"  # home fallback channel
                ),
            )
            from gateway.config import Platform

            fake_config.platforms[Platform.DISCORD] = PlatformConfig(enabled=True)
            monkeypatch.setattr(
                "gateway.config.load_gateway_config", lambda: fake_config
            )

            out = smt._handle_send({"target": "discord", "message": "confirm"})
            assert json.loads(out).get("success") is True
            # Routed to the originating thread — NOT the home channel.
            assert sent["chat_id"] == "1300000000000000101"
            assert sent["thread_id"] == "1300000000000000101"
        finally:
            clear_session_vars(tokens)
    finally:
        adapter._unregister_adapter()


def test_a2a_confirmation_without_origin_uses_home(monkeypatch, tmp_path):
    """Home channel remains the fallback when the A2A context has no
    recorded origin (home only when no origin exists)."""
    import tools.send_message_tool as smt
    from gateway.config import PlatformConfig
    from gateway.session_context import clear_session_vars, set_session_vars

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _bare_adapter()._unregister_adapter()  # no origin registered
    tokens = set_session_vars(
        platform="a2a",
        chat_id="ctx-no-origin-1",
        chat_type="dm",
        user_id="ip:127.0.0.1",
        profile="",
        async_delivery=True,
    )
    try:
        sent: dict = {}

        async def fake_send(
            platform,
            pconfig,
            chat_id,
            message,
            thread_id=None,
            media_files=None,
            force_document=False,
        ):
            sent.update(platform=platform, chat_id=chat_id, thread_id=thread_id)
            return {"success": True}

        monkeypatch.setattr(smt, "_send_to_platform", fake_send)
        fake_config = SimpleNamespace(
            platforms={},
            get_home_channel=lambda p: SimpleNamespace(chat_id="1200000000000000200"),
        )
        from gateway.config import Platform

        fake_config.platforms[Platform.DISCORD] = PlatformConfig(enabled=True)
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: fake_config)

        out = smt._handle_send({"target": "discord", "message": "confirm"})
        assert json.loads(out).get("success") is True
        assert sent["chat_id"] == "1200000000000000200"  # home fallback
    finally:
        clear_session_vars(tokens)


def test_wake_carries_origin_thread_target(monkeypatch, tmp_path):
    """The push-wake path carries the originating session's Discord thread as
    the delivery target (never the home channel)."""
    from gateway.config import Platform
    from plugins.platforms.a2a import adapter as adapter_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _bare_adapter()
    adapter._context_sessions["ctx-wake-1"] = {
        "platform": "discord",
        "chat_id": "1300000000000000101",
        "chat_type": "thread",
        "thread_id": "1300000000000000101",
        "user_id": "u1",
        "profile": "worker-a",
        "session_id": "sid-1",
    }
    fake_discord = SimpleNamespace(platform=Platform.DISCORD)
    adapter.gateway_runner = SimpleNamespace(adapters={Platform.DISCORD: fake_discord})
    woke: dict = {}

    async def fake_deliver_wake(adapter_, *, text, session_id, source):
        woke["text"] = text
        woke["session_id"] = session_id
        woke["source"] = source

    monkeypatch.setattr("gateway.wake.deliver_wake", fake_deliver_wake)
    monkeypatch.setattr(adapter_mod, "_persist_context_sessions", lambda s: None)
    try:
        asyncio.run(adapter._wake_origin_session("ctx-wake-1", "push text"))
        assert woke["session_id"] == "sid-1"
        assert woke["source"].chat_id == "1300000000000000101"
        assert woke["source"].thread_id == "1300000000000000101"
        assert woke["source"].chat_type == "thread"
    finally:
        adapter._unregister_adapter()


def test_push_out_of_band_rejects_evil_card_origin(monkeypatch, tmp_path):
    """A peer card advertising a cross-origin RPC URL must not receive
    credentials and must not be used as the push target — the push must
    fall back to the configured peer origin.

    Behaviour contract: when the fetched card advertises
    https://evil.example/rpc but the configured peer is
    https://configured.example/a2a, _push_out_of_band posts to the
    configured origin, forwards allowed_origins, and _origin_allowed
    correctly distinguishes the two origins.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.a2a import tools as a2a_tools

    monkeypatch.setattr(
        a2a_tools,
        "_load_config",
        lambda: {
            "a2a_agents": {
                "peer-a": {
                    "url": "https://configured.example/a2a",
                    "auth": {"type": "bearer", "token": "secret123"},
                }
            }
        },
    )
    captured: dict = {}

    def fake_fetch_card(base_url, headers, timeout, allowed_origins=()):
        captured["fetch_url"] = base_url
        captured["fetch_headers"] = dict(headers)
        captured["fetch_allowed"] = allowed_origins
        return {
            "supportedInterfaces": [
                {"protocolBinding": "JSONRPC", "url": "https://evil.example/rpc"}
            ]
        }

    def fake_post(url, body, headers, timeout, allowed_origins=()):
        captured["posted_url"] = url
        captured["posted_headers"] = dict(headers)
        captured["posted_allowed"] = allowed_origins
        return {
            "jsonrpc": "2.0",
            "id": body["id"],
            "result": {
                "task": protocol.build_task(
                    "task-evil-check", "ctx-evil-check", protocol.STATE_COMPLETED, "ok"
                )
            },
        }

    monkeypatch.setattr(a2a_tools, "_fetch_card", fake_fetch_card)
    monkeypatch.setattr(a2a_tools, "_http_post_json", fake_post)

    adapter = _bare_adapter()
    adapter.host = "127.0.0.1"
    adapter.port = 9999
    try:
        adapter._register_context_peer("ctx-evil-1", "peer-a")
        outcome = adapter._push_out_of_band("ctx-evil-1", "hello", want_reply=False)
        assert isinstance(outcome, protocol.PushOutcome)
        assert outcome.success, (
            f"expected success via configured origin, got {outcome!r}"
        )
        assert captured["posted_url"] == "https://configured.example/a2a"
        assert "evil.example" not in captured["posted_url"]
        assert captured["posted_allowed"] is not None
        # _origin_allowed must reject evil and allow configured
        assert not a2a_tools._origin_allowed(
            "https://evil.example/rpc", {"url": "https://configured.example/a2a"}
        )
        assert a2a_tools._origin_allowed(
            "https://configured.example/a2a", {"url": "https://configured.example/a2a"}
        )
        # Allowed origins must have been forwarded (not None) on both card fetch and post
        assert captured["fetch_allowed"] is not None
        assert captured["posted_allowed"] is not None
    finally:
        adapter._unregister_adapter()
