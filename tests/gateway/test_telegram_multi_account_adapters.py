"""Per-account adapter registry and inbound stamping — #8287.

Named bot accounts get their own adapter instances, registered in
``_account_adapters[platform][name]`` (the account-dimension mirror of
``_profile_adapters``), each seeing an ordinary derived ``PlatformConfig``.

Two contracts are load-bearing here:

* **Resolution fails closed.** A stamped account with no registry entry must
  never fall back to the default bot — replying out the wrong bot is worse
  than not replying.
* **The stamp lands on the NORMAL inbound path.** Every platform's ordinary
  traffic is built by ``BasePlatformAdapter.build_source()``. Stamping only
  the Telegram auth helpers leaves real named-bot messages with
  ``account=None``, which routes them to the default session key and the
  default egress adapter — i.e. the feature silently no-ops.
"""

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


@pytest.fixture()
def runner(monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return gateway_run.GatewayRunner(GatewayConfig())


def _telegram_adapter(token="1:x", account=None):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token=token))
    adapter.account_name = account
    return adapter


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
    """A stamped account whose adapter is missing (failed to connect,
    misconfigured) must NOT fall back to the default adapter."""
    runner.adapters = {Platform.TELEGRAM: MagicMock()}
    runner._account_adapters = {}
    assert runner._authorization_adapter(Platform.TELEGRAM, account="support") is None


def test_account_in_secondary_profile_fails_closed(runner):
    """Named account + secondary profile is not a supported combination yet,
    and is checked before the active-profile fast path."""
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

    src_default = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm")
    src_support = SessionSource(
        platform=Platform.TELEGRAM, chat_id="1", chat_type="dm", account="support"
    )
    assert runner._adapter_for_source(src_default) is default_adapter
    assert runner._adapter_for_source(src_support) is support_adapter


def test_bare_fixture_source_reads_as_default_account(runner):
    """A SimpleNamespace/MagicMock source auto-creates a truthy non-string
    ``account`` (AGENTS.md pitfall #17). That must read as the default
    account, not trip the fail-closed branch and silence every reply."""
    default_adapter = MagicMock()
    runner.adapters = {Platform.TELEGRAM: default_adapter}

    bare = SimpleNamespace(platform=Platform.TELEGRAM, profile=None)
    assert runner._adapter_for_source(bare) is default_adapter

    mock_source = MagicMock()
    mock_source.platform = Platform.TELEGRAM
    mock_source.profile = None
    mock_source._transport_adapter_ref = None
    mock_source.delivered_via_upstream_relay = False
    assert runner._adapter_for_source(mock_source) is default_adapter


def test_mock_account_does_not_corrupt_the_session_key():
    """The same pitfall on the session-key side: a non-string ``account``
    must not be interpolated into the key namespace. Without the guard this
    derives ``agent:main@<MagicMock ...>`` — and a different key every run,
    because the repr embeds the object id."""
    mock_source = MagicMock()
    mock_source.platform = Platform.TELEGRAM
    mock_source.chat_id = "777"
    mock_source.chat_type = "dm"
    mock_source.user_id = "777"
    mock_source.thread_id = None
    key = build_session_key(mock_source)
    assert "MagicMock" not in key
    assert key.split(":")[1] == "main"


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
        {
            "token": "456:support",
            "allowed_users": [2, 3],
            "home_channel": {"chat_id": "-100999"},
        },
    )
    assert derived.token == "456:support"
    assert isinstance(derived.home_channel, HomeChannel)
    assert derived.home_channel.chat_id == "-100999"
    # The platform is implicit inside an account's own home_channel block.
    assert derived.home_channel.platform == Platform.TELEGRAM
    # Account block overrides platform-level extra; unrelated keys inherit.
    assert derived.extra["allowed_users"] == [2, 3]
    assert derived.extra["fallback_ips"] == ["1.2.3.4"]
    # An account must never be able to spawn accounts.
    assert "accounts" not in derived.extra
    # The base config is untouched (dataclasses.replace, not mutation).
    assert base.token == "123:default"
    assert base.extra["allowed_users"] == [1]


# ── Lifecycle (_start_account_adapters) ─────────────────────────────────────


def _wire_prepare_mocks(runner):
    created = []

    def _fake_create(platform, config):
        adapter = MagicMock()
        adapter.platform = platform
        adapter.config = config
        adapter.account_name = None
        created.append(adapter)
        return adapter

    runner._create_adapter = _fake_create
    runner._make_adapter_auth_check = MagicMock(return_value=lambda *a, **kw: True)
    runner._recover_telegram_topic_thread_id = lambda _s: None
    runner._handle_adapter_fatal_error = AsyncMock()
    runner._handle_active_session_busy_message = AsyncMock()
    runner._handle_reaction_event = AsyncMock()
    runner.session_store = MagicMock()
    runner._busy_text_mode = "full"
    return created


def _accounts_config(token="123:default", **accounts):
    return PlatformConfig(
        enabled=True, token=token, extra={"accounts": dict(accounts)}
    )


def test_prepare_account_adapters_wires_and_stamps(runner):
    _wire_prepare_mocks(runner)
    cfg = _accounts_config(
        support={"token": "456:support"}, sales={"token": "789:sales"}
    )
    prepared = runner._prepare_account_adapters(Platform.TELEGRAM, cfg)

    assert [name for name, _, _ in ((a.account_name, c, p) for p, c, a in prepared)] == [
        "support",
        "sales",
    ]
    by_name = {a.account_name: (p, c, a) for p, c, a in prepared}
    _, support_cfg, support = by_name["support"]
    # Its OWN derived token, not the platform default.
    assert support_cfg.token == "456:support"
    assert support.config.token == "456:support"
    # Wired like a default adapter.
    support.set_message_handler.assert_called_once()
    support.set_authorization_check.assert_called_once()
    support.set_platform_event_handler.assert_called_once()
    # Prepare-only: nothing is connected or registered yet — that is what lets
    # accounts join the same concurrent fan-out as every other platform.
    assert runner._account_adapters == {}


def test_tokenless_account_is_skipped(runner):
    created = _wire_prepare_mocks(runner)
    cfg = _accounts_config(support={"display_name": "no token"})
    assert runner._prepare_account_adapters(Platform.TELEGRAM, cfg) == []
    assert created == []


def test_no_accounts_is_a_noop(runner):
    _wire_prepare_mocks(runner)
    cfg = PlatformConfig(enabled=True, token="123:default")
    assert runner._prepare_account_adapters(Platform.TELEGRAM, cfg) == []
    assert runner._account_adapters == {}


# ── Startup fan-out (real GatewayRunner.start) ─────────────────────────────
#
# These drive the real startup path rather than the helper in isolation. That
# matters: the account wiring lives in _connect_platforms' pre-filter and
# aggregation, and the #83791 rewrite made "where it is wired" the part most
# likely to be wrong.


class _ScriptedAdapter(BasePlatformAdapter):
    """Adapter whose connect() records order and returns a scripted result."""

    events: list = []

    def __init__(self, platform, config, label, ok=True, sleep=0.0):
        super().__init__(config, platform)
        self._label = label
        self._ok = ok
        self._sleep = sleep

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        _ScriptedAdapter.events.append((self._label, "start"))
        if self._sleep:
            await asyncio.sleep(self._sleep)
        _ScriptedAdapter.events.append((self._label, "end"))
        if not self._ok:
            self._set_fatal_error("bad_token", "unauthorized", retryable=True)
        return self._ok

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _account_runner(tmp_path, monkeypatch, *, default_token="123:default", **kw):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _ScriptedAdapter.events = []
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token=default_token,
                extra={
                    "accounts": {
                        "support": {"token": "456:support"},
                        "sales": {"token": "789:sales"},
                    }
                },
            )
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    monkeypatch.setattr(runner, "_start_secondary_profile_adapters", lambda: 0)
    return runner


@pytest.mark.asyncio
async def test_named_accounts_connect_concurrently_with_the_default(
    tmp_path, monkeypatch
):
    """Accounts join the same fan-out (#83791), not a serial tail behind it.

    The default is slow; both accounts are instant. Under a serial start the
    default's connect would end before either account's began, so an account
    end can never precede the default's. Only overlap puts them first.
    """
    runner = _account_runner(tmp_path, monkeypatch)

    def _make(platform, cfg):
        if cfg.token == "123:default":
            return _ScriptedAdapter(platform, cfg, "default", sleep=0.3)
        label = "support" if cfg.token == "456:support" else "sales"
        return _ScriptedAdapter(platform, cfg, label)

    monkeypatch.setattr(runner, "_create_adapter", _make)
    await runner.start()

    events = _ScriptedAdapter.events
    order = [f"{label}:{kind}" for label, kind in events]
    assert "support:end" in order and "default:end" in order, order
    assert order.index("support:end") < order.index("default:end"), (
        f"accounts did not overlap the default connect (serial tail?): {order}"
    )
    assert set(runner._account_adapters[Platform.TELEGRAM]) == {"support", "sales"}


@pytest.mark.asyncio
async def test_failing_default_does_not_keep_named_accounts_offline(
    tmp_path, monkeypatch
):
    """The #67455 review finding: a bad or absent default token must not
    block otherwise healthy named bots."""
    runner = _account_runner(tmp_path, monkeypatch)

    def _make(platform, cfg):
        if cfg.token == "123:default":
            return _ScriptedAdapter(platform, cfg, "default", ok=False)
        label = "support" if cfg.token == "456:support" else "sales"
        return _ScriptedAdapter(platform, cfg, label)

    monkeypatch.setattr(runner, "_create_adapter", _make)
    await runner.start()

    # Default failed and is NOT registered...
    assert Platform.TELEGRAM not in runner.adapters
    # ...while both named accounts are live.
    assert set(runner._account_adapters[Platform.TELEGRAM]) == {"support", "sales"}


@pytest.mark.asyncio
async def test_failed_account_never_claims_the_platform_retry_slot(
    tmp_path, monkeypatch
):
    """A named account owns no entry in self.adapters and must not take the
    platform's single _failed_platforms slot — that slot would respawn the
    DEFAULT adapter from the account's config."""
    runner = _account_runner(tmp_path, monkeypatch)

    def _make(platform, cfg):
        if cfg.token == "456:support":
            return _ScriptedAdapter(platform, cfg, "support", ok=False)
        label = {"123:default": "default", "789:sales": "sales"}[cfg.token]
        return _ScriptedAdapter(platform, cfg, label)

    monkeypatch.setattr(runner, "_create_adapter", _make)
    await runner.start()

    # Default is healthy and registered; the failed account did not mark the
    # platform for reconnect, and did not evict the default adapter.
    assert Platform.TELEGRAM in runner.adapters
    assert runner.adapters[Platform.TELEGRAM].config.token == "123:default"
    assert Platform.TELEGRAM not in runner._failed_platforms
    assert set(runner._account_adapters[Platform.TELEGRAM]) == {"sales"}


@pytest.mark.asyncio
async def test_accounts_only_platform_skips_the_tokenless_default_connect(
    tmp_path, monkeypatch
):
    """Named tokens with no default credential: the doomed token-less default
    connect is skipped entirely, but the accounts still come up."""
    runner = _account_runner(tmp_path, monkeypatch, default_token="")

    def _make(platform, cfg):
        label = "support" if cfg.token == "456:support" else "sales"
        return _ScriptedAdapter(platform, cfg, label)

    monkeypatch.setattr(runner, "_create_adapter", _make)
    await runner.start()

    labels = {label for label, _ in _ScriptedAdapter.events}
    assert "default" not in labels, "token-less default should never connect"
    assert set(runner._account_adapters[Platform.TELEGRAM]) == {"support", "sales"}
    assert Platform.TELEGRAM not in runner.adapters


# ── Inbound stamping (real TelegramAdapter) ────────────────────────────────


def test_build_source_stamps_account_on_normal_event_path():
    """The regression guard for the #67455 review's first finding.

    Normal traffic does NOT go through the auth helpers — it goes through
    ``build_source``. If the stamp is missing here, named-bot messages keep
    ``account=None`` and the whole account dimension is inert for real users.
    """
    support = _telegram_adapter(account="support")
    source = support.build_source(chat_id="777", chat_type="dm", user_id="42")
    assert source.account == "support"

    # Default/single-bot adapters stay byte-identical to before.
    default = _telegram_adapter(token="1:y")
    assert default.build_source(chat_id="777", chat_type="dm", user_id="42").account is None


def test_stamped_source_yields_a_per_account_session_key():
    """End-to-end tie-back to the previous slice: the stamp build_source
    applies is what makes the same chat two sessions under two bots."""
    support = _telegram_adapter(account="support")
    default = _telegram_adapter(token="1:y")
    kw = dict(chat_id="777", chat_type="dm", user_id="42")

    support_key = build_session_key(support.build_source(**kw))
    default_key = build_session_key(default.build_source(**kw))

    assert support_key != default_key
    assert support_key.split(":")[1] == "main@support"
    assert default_key == "agent:main:telegram:dm:777"


def test_telegram_auth_helper_stamps_account():
    """The auth-path source must agree with the normal path, or the
    adapter-level guard and the session store derive different keys for the
    same event (the #64934 bug class)."""
    adapter = _telegram_adapter(account="support")

    message = MagicMock()
    message.chat.id = 777
    message.chat.type = "private"
    message.chat.title = None
    message.from_user.id = 42
    message.from_user.username = "user"
    message.from_user.full_name = "User"
    message.message_thread_id = None
    message.is_topic_message = False

    assert adapter._source_from_message_for_auth(message).account == "support"
    assert _telegram_adapter(token="1:y")._source_from_message_for_auth(
        message
    ).account is None
