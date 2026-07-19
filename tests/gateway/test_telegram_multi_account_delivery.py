"""Per-account outbound routing and lifecycle recovery — #8287.

Outbound must honor the account dimension end-to-end: origin replies leave
through the bot the message arrived on, explicit targets can address a named
bot (``telegram@support:123``), a missing account adapter fails closed
(never the default bot), and a dying account adapter is queued for
reconnection without ever touching the default platform slot.
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.delivery import DeliveryRouter, DeliveryTarget
from gateway.session import SessionSource


@pytest.fixture()
def runner(monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return gateway_run.GatewayRunner(GatewayConfig())


# ── DeliveryTarget parsing ─────────────────────────────────────────────────


def test_parse_account_scoped_chat_target():
    target = DeliveryTarget.parse("telegram@support:123456")
    assert target.platform == Platform.TELEGRAM
    assert target.account == "support"
    assert target.chat_id == "123456"
    assert target.is_explicit


def test_parse_account_scoped_home_target():
    target = DeliveryTarget.parse("telegram@support")
    assert target.platform == Platform.TELEGRAM
    assert target.account == "support"
    assert target.chat_id is None


def test_parse_plain_targets_unchanged():
    assert DeliveryTarget.parse("telegram:123").account is None
    assert DeliveryTarget.parse("telegram").account is None


def test_origin_target_inherits_account():
    origin = SessionSource(
        platform=Platform.TELEGRAM, chat_id="777", chat_type="dm",
        account="support",
    )
    target = DeliveryTarget.parse("origin", origin=origin)
    assert target.is_origin and target.account == "support"
    # Default-account origins stay account-less.
    plain = SessionSource(platform=Platform.TELEGRAM, chat_id="7", chat_type="dm")
    assert DeliveryTarget.parse("origin", origin=plain).account is None


def test_to_string_round_trips_account():
    for raw in ("telegram@support:123", "telegram@support", "telegram:123"):
        assert DeliveryTarget.parse(raw).to_string() == raw


# ── Router resolution ──────────────────────────────────────────────────────


def _router(default_adapter=None, account_adapters=None):
    router = DeliveryRouter(GatewayConfig())
    if default_adapter is not None:
        router.adapters = {Platform.TELEGRAM: default_adapter}
    router.account_adapters = account_adapters or {}
    return router


def test_router_resolves_account_adapter():
    default_adapter, support_adapter = MagicMock(), MagicMock()
    router = _router(default_adapter, {Platform.TELEGRAM: {"support": support_adapter}})
    assert (
        router._adapter_for_target(DeliveryTarget.parse("telegram@support:1"))
        is support_adapter
    )
    assert (
        router._adapter_for_target(DeliveryTarget.parse("telegram:1"))
        is default_adapter
    )


def test_router_fails_closed_for_unknown_account():
    """Account-addressed content must never leave through the default bot."""
    router = _router(MagicMock())
    assert router._adapter_for_target(DeliveryTarget.parse("telegram@support:1")) is None


@pytest.mark.asyncio
async def test_deliver_to_platform_raises_with_account_ref():
    router = _router(MagicMock())
    with pytest.raises(ValueError, match="telegram@support"):
        await router._deliver_to_platform(
            DeliveryTarget.parse("telegram@support:1"), "content", None
        )


# ── Account fatal-error path ───────────────────────────────────────────────


def _fatal_adapter(platform=Platform.TELEGRAM, account="support", retryable=True):
    adapter = MagicMock()
    adapter.platform = platform
    adapter.account_name = account
    adapter.fatal_error_code = "network"
    adapter.fatal_error_message = "boom"
    adapter.fatal_error_retryable = retryable
    adapter.config = PlatformConfig(enabled=True, token="456:support")
    return adapter


@pytest.mark.asyncio
async def test_account_fatal_error_queues_reconnect_not_default_slot(runner):
    default_adapter = MagicMock()
    runner.adapters = {Platform.TELEGRAM: default_adapter}
    adapter = _fatal_adapter()
    runner._account_adapters = {Platform.TELEGRAM: {"support": adapter}}
    runner._safe_adapter_disconnect = AsyncMock()

    await runner._handle_adapter_fatal_error(adapter)

    # Popped from the account registry, queued under (platform, account).
    assert "support" not in (runner._account_adapters.get(Platform.TELEGRAM) or {})
    key = (Platform.TELEGRAM, "support")
    assert key in runner._failed_account_adapters
    assert runner._failed_account_adapters[key]["config"] is adapter.config
    # The default platform slot is untouched — no clobbering, no queueing.
    assert runner.adapters[Platform.TELEGRAM] is default_adapter
    assert Platform.TELEGRAM not in runner._failed_platforms
    runner._safe_adapter_disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_stale_account_fatal_error_is_ignored(runner):
    """A superseded account adapter instance (reconnect already won) must
    not evict the healthy replacement."""
    replacement = MagicMock()
    runner._account_adapters = {Platform.TELEGRAM: {"support": replacement}}
    runner._safe_adapter_disconnect = AsyncMock()

    stale = _fatal_adapter()
    await runner._handle_adapter_fatal_error(stale)

    assert runner._account_adapters[Platform.TELEGRAM]["support"] is replacement
    assert (Platform.TELEGRAM, "support") not in runner._failed_account_adapters
    runner._safe_adapter_disconnect.assert_not_awaited()


@pytest.mark.asyncio
async def test_nonretryable_account_fatal_error_not_queued(runner):
    adapter = _fatal_adapter(retryable=False)
    runner._account_adapters = {Platform.TELEGRAM: {"support": adapter}}
    runner._safe_adapter_disconnect = AsyncMock()

    await runner._handle_adapter_fatal_error(adapter)

    assert runner._failed_account_adapters == {}
    assert "support" not in (runner._account_adapters.get(Platform.TELEGRAM) or {})
