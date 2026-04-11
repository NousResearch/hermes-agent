"""
Regression test: non-internal messages with user_id=None must not trigger
pairing code generation.

Certain Telegram messages (service messages, channel forwards, anonymous
admin messages) arrive with `from_user = None`. The Telegram adapter
correctly sets `source.user_id = None` for these, but since
`_is_user_authorized()` returns False for None user_id, the unauthorized DM
handler would attempt to generate and send a pairing code — crashing because
there is no valid recipient.

These messages should be silently ignored instead.
"""

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_source(
    platform: Platform = Platform.TELEGRAM,
    user_id: str | None = None,
    chat_id: str = "123",
    chat_type: str = "dm",
) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="tester" if user_id else None,
        chat_type=chat_type,
    )


def _make_event(source: SessionSource, internal: bool = False) -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_id="m1",
        source=source,
        internal=internal,
    )


class TestNoneUserIdIgnoresPairing:
    """Non-internal events with user_id=None must not generate a pairing code."""

    @pytest.mark.asyncio
    async def test_none_user_id_in_dm_is_ignored(self, monkeypatch):
        """A DM event with user_id=None is silently dropped, not rejected as unauthorized."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        import gateway.run as gateway_run
        from gateway.run import GatewayRunner

        monkeypatch.setattr(gateway_run, "_hermes_home", __import__("pathlib").Path("/tmp"))
        (gateway_run._hermes_home / "config.yaml").write_text("", encoding="utf-8")

        runner = GatewayRunner(GatewayConfig())
        adapter = SimpleNamespace(send=AsyncMock())
        runner.adapters[Platform.TELEGRAM] = adapter

        source = _make_source(platform=Platform.TELEGRAM, user_id=None, chat_id="123", chat_type="dm")
        event = _make_event(source, internal=False)

        async def _raise(*_a, **_kw):
            raise RuntimeError("sentinel")
        monkeypatch.setattr(GatewayRunner, "_handle_message_with_agent", _raise)

        try:
            await runner._handle_message(event)
        except RuntimeError:
            pass

        assert adapter.send.await_count == 0, (
            "Adapter.send should NOT be called for user_id=None — "
            "message should be silently ignored"
        )

    @pytest.mark.asyncio
    async def test_none_user_id_in_group_is_ignored(self, monkeypatch):
        """A group event with user_id=None is also silently dropped."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        import gateway.run as gateway_run
        from gateway.run import GatewayRunner

        monkeypatch.setattr(gateway_run, "_hermes_home", __import__("pathlib").Path("/tmp"))
        (gateway_run._hermes_home / "config.yaml").write_text("", encoding="utf-8")

        runner = GatewayRunner(GatewayConfig())
        adapter = SimpleNamespace(send=AsyncMock())
        runner.adapters[Platform.TELEGRAM] = adapter

        source = _make_source(platform=Platform.TELEGRAM, user_id=None, chat_id="456", chat_type="group")
        event = _make_event(source, internal=False)

        async def _raise(*_a, **_kw):
            raise RuntimeError("sentinel")
        monkeypatch.setattr(GatewayRunner, "_handle_message_with_agent", _raise)

        try:
            await runner._handle_message(event)
        except RuntimeError:
            pass

        assert adapter.send.await_count == 0

    @pytest.mark.asyncio
    async def test_none_user_id_does_not_generate_pairing_code(self, monkeypatch):
        """Ensure generate_code is never called when user_id is None."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        import gateway.run as gateway_run
        from gateway.run import GatewayRunner

        monkeypatch.setattr(gateway_run, "_hermes_home", __import__("pathlib").Path("/tmp"))
        (gateway_run._hermes_home / "config.yaml").write_text("", encoding="utf-8")

        runner = GatewayRunner(GatewayConfig())
        adapter = SimpleNamespace(send=AsyncMock())
        runner.adapters[Platform.TELEGRAM] = adapter

        source = _make_source(platform=Platform.TELEGRAM, user_id=None, chat_id="789", chat_type="dm")
        event = _make_event(source, internal=False)

        original_generate = runner.pairing_store.generate_code
        generate_called = False

        def tracking_generate(*args, **kwargs):
            nonlocal generate_called
            generate_called = True
            return original_generate(*args, **kwargs)

        runner.pairing_store.generate_code = tracking_generate

        async def _raise(*_a, **_kw):
            raise RuntimeError("sentinel")
        monkeypatch.setattr(GatewayRunner, "_handle_message_with_agent", _raise)

        try:
            await runner._handle_message(event)
        except RuntimeError:
            pass

        assert not generate_called, (
            "generate_code must NOT be called when user_id is None"
        )

    @pytest.mark.asyncio
    async def test_known_user_without_pairing_still_gets_pairing_code(self, monkeypatch):
        """Sanity check: an unauthorized-but-identified user still gets a pairing code."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        import gateway.run as gateway_run
        from gateway.run import GatewayRunner

        monkeypatch.setattr(gateway_run, "_hermes_home", __import__("pathlib").Path("/tmp"))
        (gateway_run._hermes_home / "config.yaml").write_text("", encoding="utf-8")
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("TELEGRAM_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

        runner = GatewayRunner(GatewayConfig())
        adapter = SimpleNamespace(send=AsyncMock())
        runner.adapters[Platform.TELEGRAM] = adapter

        source = _make_source(
            platform=Platform.TELEGRAM,
            user_id="unknown_user_123",
            chat_id="123",
            chat_type="dm",
        )
        event = _make_event(source, internal=False)

        result = await runner._handle_message(event)

        assert result is None
        assert adapter.send.await_count == 1
        sent_text = adapter.send.await_args.args[1]
        assert "don't recognize" in sent_text
