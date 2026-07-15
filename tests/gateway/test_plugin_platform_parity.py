"""Tests for plugin platform parity: system-prompt hints and standalone sending."""

import importlib

import pytest

from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.config import Platform


# ── Platform hints ───────────────────────────────────────────────────────


class TestPluginPlatformHints:
    """Plugin-registered platform_hint is merged into PLATFORM_HINTS."""

    def test_registered_hint_appears_in_platform_hints(self):
        """A plugin platform's platform_hint is merged into PLATFORM_HINTS at import time."""
        entry = PlatformEntry(
            name="testplatform",
            label="TestPlatform",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            source="plugin",
            platform_hint="You are on TestPlatform. No markdown.",
        )
        platform_registry.register(entry)
        try:
            # Reload prompt_builder so the import-time merge picks up the new entry.
            import agent.prompt_builder as _pb
            importlib.reload(_pb)
            assert "testplatform" in _pb.PLATFORM_HINTS
            assert "No markdown." in _pb.PLATFORM_HINTS["testplatform"]
        finally:
            platform_registry.unregister("testplatform")

    def test_builtin_hint_not_overwritten_by_registry(self):
        """Hardcoded PLATFORM_HINTS win over registry hints for built-in names."""
        entry = PlatformEntry(
            name="telegram",
            label="FakeTelegram",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            source="plugin",
            platform_hint="Fake telegram hint.",
        )
        platform_registry.register(entry)
        try:
            import agent.prompt_builder as _pb
            importlib.reload(_pb)
            assert _pb.PLATFORM_HINTS["telegram"] != "Fake telegram hint."
            assert "Telegram" in _pb.PLATFORM_HINTS["telegram"]
        finally:
            platform_registry.unregister("telegram")


# ── Standalone sender ──────────────────────────────────────────────────────


class TestPluginPlatformStandaloneSend:
    """Plugin-registered standalone_sender_fn is invoked by _send_to_platform."""

    @pytest.mark.asyncio
    async def test_standalone_sender_fn_routed_by_send_to_platform(self):
        """_send_to_platform invokes a plugin's standalone_sender_fn directly."""
        recorded = {}

        async def _stub_sender(
            pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False
        ):
            recorded["args"] = {
                "pconfig": pconfig,
                "chat_id": chat_id,
                "message": message,
                "thread_id": thread_id,
                "media_files": media_files,
                "force_document": force_document,
            }
            return {"success": True, "message_id": "msg-123"}

        entry = PlatformEntry(
            name="testsendplat",
            label="TestSendPlat",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            source="plugin",
            standalone_sender_fn=_stub_sender,
        )
        platform_registry.register(entry)
        try:
            from tools.send_message_tool import _send_to_platform
            from gateway.config import PlatformConfig

            pconfig = PlatformConfig(enabled=True, token="tok")
            plat = Platform("testsendplat")
            result = await _send_to_platform(
                plat,
                pconfig,
                "chat-1",
                "hello world",
                thread_id="th-1",
                media_files=[("/tmp/f.png", False)],
                force_document=True,
            )
            assert isinstance(result, dict)
            assert result.get("success") is True
            assert result.get("message_id") == "msg-123"
            assert recorded["args"]["chat_id"] == "chat-1"
            assert recorded["args"]["message"] == "hello world"
            assert recorded["args"]["thread_id"] == "th-1"
            assert recorded["args"]["media_files"] == [("/tmp/f.png", False)]
            assert recorded["args"]["force_document"] is True
        finally:
            platform_registry.unregister("testsendplat")

    @pytest.mark.asyncio
    async def test_standalone_sender_fn_exception_graceful(self):
        """If standalone_sender_fn raises, the error is caught and returned."""

        async def _bad_sender(
            pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False
        ):
            raise RuntimeError("network down")

        entry = PlatformEntry(
            name="testbadplat",
            label="TestBadPlat",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            source="plugin",
            standalone_sender_fn=_bad_sender,
        )
        platform_registry.register(entry)
        try:
            from tools.send_message_tool import _send_to_platform
            from gateway.config import PlatformConfig

            pconfig = PlatformConfig(enabled=True, token="tok")
            plat = Platform("testbadplat")
            result = await _send_to_platform(
                plat,
                pconfig,
                "chat-1",
                "hello world",
            )
            assert isinstance(result, dict)
            assert result.get("error") == "Plugin standalone send failed: network down"
        finally:
            platform_registry.unregister("testbadplat")
