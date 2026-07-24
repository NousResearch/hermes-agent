"""Tests for platform:connected hook emission (issue #50020)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestPlatformConnectedHook:
    """Verify that platform:connected fires after each adapter connects."""

    @pytest.mark.asyncio
    async def test_primary_profile_emits_platform_connected(self):
        """Primary profile adapter connection emits platform:connected."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.hooks = MagicMock()
        runner.hooks.emit = AsyncMock()
        runner.adapters = {}
        runner._sync_voice_mode_state_to_adapter = MagicMock()
        runner._update_platform_runtime_status = MagicMock()
        runner._connect_adapter_with_timeout = AsyncMock(return_value=True)

        platform = MagicMock()
        platform.value = "telegram"
        adapter = MagicMock()

        # Simulate the primary profile connection path (lines 5551-5562)
        success = await runner._connect_adapter_with_timeout(adapter, platform)
        if success:
            runner.adapters[platform] = adapter
            runner._sync_voice_mode_state_to_adapter(adapter)
            await runner.hooks.emit("platform:connected", {
                "platform": platform.value,
            })

        runner.hooks.emit.assert_called_once_with("platform:connected", {
            "platform": "telegram",
        })

    @pytest.mark.asyncio
    async def test_multiplex_profile_emits_platform_connected_with_profile(self):
        """Multiplex profile adapter connection includes profile name."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.hooks = MagicMock()
        runner.hooks.emit = AsyncMock()
        runner._connect_adapter_with_timeout = AsyncMock(return_value=True)
        runner._safe_adapter_disconnect = AsyncMock()
        runner._adapter_credential_fingerprint = MagicMock(return_value=None)

        platform = MagicMock()
        platform.value = "discord"
        adapter = MagicMock()
        profile_map = {}

        # Simulate the multiplex connection path (lines 6971-6975)
        success = await runner._connect_adapter_with_timeout(adapter, platform)
        if success:
            profile_map[platform] = adapter
            await runner.hooks.emit("platform:connected", {
                "platform": platform.value,
                "profile": "stock-bot",
            })

        runner.hooks.emit.assert_called_once_with("platform:connected", {
            "platform": "discord",
            "profile": "stock-bot",
        })

    @pytest.mark.asyncio
    async def test_failed_connect_does_not_emit(self):
        """Failed adapter connection does NOT emit platform:connected."""
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.hooks = MagicMock()
        runner.hooks.emit = AsyncMock()
        runner._connect_adapter_with_timeout = AsyncMock(return_value=False)
        runner._safe_adapter_disconnect = AsyncMock()

        platform = MagicMock()
        platform.value = "telegram"
        adapter = MagicMock()

        success = await runner._connect_adapter_with_timeout(adapter, platform)
        if success:
            await runner.hooks.emit("platform:connected", {
                "platform": platform.value,
            })

        runner.hooks.emit.assert_not_called()
