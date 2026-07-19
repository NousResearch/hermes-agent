"""Per-account adapter registry and lifecycle — #8287.

Named bot accounts get their own adapter instances, registered in
``_account_adapters[platform][name]`` (the account-dimension mirror of
``_profile_adapters``), each seeing an ordinary derived ``PlatformConfig``.
Resolution fails closed: a stamped account with no registry entry must never
fall back to the default bot — replying out the wrong bot is worse than not
replying.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.session import SessionSource


@pytest.fixture()
def runner(monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return gateway_run.GatewayRunner(GatewayConfig())


# ── Resolution (authz_mixin) ────────────────────────────────────────────────


def test_default_account_resolves_default_adapter(runner):
    default_adapter = MagicMock()
    runner.adapters = {Platform.TELEGRAM: default_adapter}
    assert runner._authorization_adapter(Platform.TELEGRAM) is default_adapter
    assert (
        runner._authorization_adapter(Platform.TELEGRAM, account="default")
        is default_adapter
    )


def test_named_account_resolves_its_own_adapter(runner):
    default_adapter, support_adapter = MagicMock(), MagicMock()
    runner.adapters = {Platform.TELEGRAM: default_adapter}
    runner._account_adapters = {Platform.TELEGRAM: {"support": support_adapter}}
    assert (
        runner._authorization_adapter(Platform.TELEGRAM, account="support")
        is support_adapter
    )


def test_unknown_account_fails_closed_never_default_bot(runner):
    """The wrong-bot rule: a stamped account whose adapter is missing
    (failed to connect, misconfigured) must NOT fall back to the default
    adapter."""
    runner.adapters = {Platform.TELEGRAM: MagicMock()}
    runner._account_adapters = {}
    assert runner._authorization_adapter(Platform.TELEGRAM, account="support") is None


def test_account_in_secondary_profile_fails_closed(runner):
    runner._profile_adapters = {"coder": {Platform.TELEGRAM: MagicMock()}}
    assert (
        runner._authorization_adapter(
            Platform.TELEGRAM, profile="coder", account="support"
        )
        is None
    )


def test_adapter_for_source_routes_by_account(runner):
    default_adapter, support_adapter = MagicMock(), MagicMock()
    runner.adapters = {Platform.TELEGRAM: default_adapter}
    runner._account_adapters = {Platform.TELEGRAM: {"support": support_adapter}}

    src_default = SessionSource(
        platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"
    )
    src_support = SessionSource(
        platform=Platform.TELEGRAM, chat_id="1", chat_type="dm", account="support"
    )
    assert runner._adapter_for_source(src_default) is default_adapter
    assert runner._adapter_for_source(src_support) is support_adapter


def test_adapter_for_source_tolerates_bare_fixture(runner):
    """SimpleNamespace sources without an ``account`` attr (AGENTS.md
    pitfall #17) must resolve like the default account."""
    default_adapter = MagicMock()
    runner.adapters = {Platform.TELEGRAM: default_adapter}
    bare = SimpleNamespace(platform=Platform.TELEGRAM, profile=None)
    assert runner._adapter_for_source(bare) is default_adapter


# ── Derived per-account config ──────────────────────────────────────────────


def test_account_platform_config_overrides_and_strips_accounts():
    base = PlatformConfig(
        enabled=True,
        token="123:default",
        extra={
            "accounts": {"support": {}},
            "fallback_ips": ["1.2.3.4"],
            "allowed_users": [1],
        },
    )
    derived = gateway_run.GatewayRunner._account_platform_config(
        Platform.TELEGRAM,
        base,
        "support",
        {
            "token": "456:support",
            "allowed_users": [2, 3],
            "home_channel": {"chat_id": "-100999"},
        },
    )
    assert derived.token == "456:support"
    assert isinstance(derived.home_channel, HomeChannel)
    assert derived.home_channel.chat_id == "-100999"
    # Account block overrides platform-level extra; unrelated keys inherit.
    assert derived.extra["allowed_users"] == [2, 3]
    assert derived.extra["fallback_ips"] == ["1.2.3.4"]
    # The accounts map itself never leaks into an account's own config.
    assert "accounts" not in derived.extra
    # The base config is untouched (dataclasses.replace, not mutation).
    assert base.token == "123:default"
    assert base.extra["allowed_users"] == [1]


# ── Lifecycle (_start_account_adapters) ─────────────────────────────────────


def _wire_lifecycle_mocks(runner, connect_results):
    created = []

    def _fake_create(platform, config):
        adapter = MagicMock()
        adapter.platform = platform
        adapter.config = config
        adapter.account_name = None
        created.append(adapter)
        return adapter

    runner._create_adapter = _fake_create
    runner._connect_adapter_with_timeout = AsyncMock(side_effect=connect_results)
    runner._safe_adapter_disconnect = AsyncMock()
    runner._make_adapter_auth_check = MagicMock(return_value=lambda *a, **kw: True)
    runner._sync_voice_mode_state_to_adapter = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _s: None
    runner._handle_message = AsyncMock()
    runner._handle_adapter_fatal_error = AsyncMock()
    runner._handle_active_session_busy_message = AsyncMock()
    runner.session_store = MagicMock()
    runner._busy_text_mode = "full"
    return created


@pytest.mark.asyncio
async def test_start_account_adapters_registers_connected_accounts(runner):
    created = _wire_lifecycle_mocks(runner, [True, True])
    cfg = PlatformConfig(
        enabled=True,
        token="123:default",
        extra={
            "accounts": {
                "support": {"token": "456:support"},
                "sales": {"token": "789:sales"},
            }
        },
    )
    connected = await runner._start_account_adapters(Platform.TELEGRAM, cfg)
    assert connected == 2
    registry = runner._account_adapters[Platform.TELEGRAM]
    assert set(registry) == {"support", "sales"}
    # Stamped before connect, with the derived (account) token.
    assert registry["support"].account_name == "support"
    assert registry["support"].config.token == "456:support"
    # Wired like a default adapter.
    registry["support"].set_message_handler.assert_called_once()
    registry["support"].set_authorization_check.assert_called_once()
    assert len(created) == 2


@pytest.mark.asyncio
async def test_tokenless_account_is_skipped(runner):
    created = _wire_lifecycle_mocks(runner, [True])
    cfg = PlatformConfig(
        enabled=True,
        token="123:default",
        extra={"accounts": {"support": {"display_name": "no token"}}},
    )
    connected = await runner._start_account_adapters(Platform.TELEGRAM, cfg)
    assert connected == 0
    assert runner._account_adapters == {}
    assert created == []


@pytest.mark.asyncio
async def test_failed_account_connect_skips_without_blocking_others(runner):
    _wire_lifecycle_mocks(runner, [False, True])
    cfg = PlatformConfig(
        enabled=True,
        token="123:default",
        extra={
            "accounts": {
                "support": {"token": "456:support"},
                "sales": {"token": "789:sales"},
            }
        },
    )
    connected = await runner._start_account_adapters(Platform.TELEGRAM, cfg)
    assert connected == 1
    registry = runner._account_adapters[Platform.TELEGRAM]
    assert set(registry) == {"sales"}
    runner._safe_adapter_disconnect.assert_awaited()


@pytest.mark.asyncio
async def test_no_accounts_is_a_noop(runner):
    _wire_lifecycle_mocks(runner, [])
    cfg = PlatformConfig(enabled=True, token="123:default")
    assert await runner._start_account_adapters(Platform.TELEGRAM, cfg) == 0
    assert runner._account_adapters == {}


# ── Inbound stamping (real TelegramAdapter) ────────────────────────────────


def test_telegram_adapter_stamps_account_on_inbound_source():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="1:x"))
    adapter.account_name = "support"

    message = MagicMock()
    message.chat.id = 777
    message.chat.type = "private"
    message.chat.title = None
    message.from_user.id = 42
    message.from_user.username = "user"
    message.from_user.full_name = "User"
    message.message_thread_id = None
    message.is_topic_message = False

    source = adapter._source_from_message_for_auth(message)
    assert source.account == "support"
    assert source.platform == Platform.TELEGRAM

    # Default account stays unset — single-bot gateways are unchanged.
    default_adapter = TelegramAdapter(PlatformConfig(enabled=True, token="1:y"))
    assert default_adapter._source_from_message_for_auth(message).account is None
