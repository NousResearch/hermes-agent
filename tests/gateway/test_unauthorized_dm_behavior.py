from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "DISCORD_ALLOWED_USERS",
        "WHATSAPP_ALLOWED_USERS",
        "SLACK_ALLOWED_USERS",
        "SIGNAL_ALLOWED_USERS",
        "EMAIL_ALLOWED_USERS",
        "SMS_ALLOWED_USERS",
        "MATTERMOST_ALLOWED_USERS",
        "MATRIX_ALLOWED_USERS",
        "DINGTALK_ALLOWED_USERS", "FEISHU_ALLOWED_USERS", "WECOM_ALLOWED_USERS",
        "QQ_ALLOWED_USERS", "QQ_GROUP_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "DISCORD_ALLOW_ALL_USERS",
        "WHATSAPP_ALLOW_ALL_USERS",
        "SLACK_ALLOW_ALL_USERS",
        "SIGNAL_ALLOW_ALL_USERS",
        "EMAIL_ALLOW_ALL_USERS",
        "SMS_ALLOW_ALL_USERS",
        "MATTERMOST_ALLOW_ALL_USERS",
        "MATRIX_ALLOW_ALL_USERS",
        "DINGTALK_ALLOW_ALL_USERS", "FEISHU_ALLOW_ALL_USERS", "WECOM_ALLOW_ALL_USERS",
        "QQ_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_event(platform: Platform, user_id: str, chat_id: str) -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_id="m1",
        source=SessionSource(
            platform=platform,
            user_id=user_id,
            chat_id=chat_id,
            user_name="tester",
            chat_type="dm",
        ),
    )


def _make_runner(platform: Platform, config: GatewayConfig):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {platform: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    # Attributes required by _handle_message for the authorized-user path
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._update_prompts = {}
    runner.hooks = SimpleNamespace(dispatch=AsyncMock(return_value=None))
    runner._sessions = {}
    return runner, adapter


def test_whatsapp_lid_user_matches_phone_allowlist_via_session_mapping(monkeypatch, tmp_path):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "15550000001")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    session_dir = tmp_path / "whatsapp" / "session"
    session_dir.mkdir(parents=True)
    (session_dir / "lid-mapping-15550000001.json").write_text('"900000000000001"', encoding="utf-8")
    (session_dir / "lid-mapping-900000000000001_reverse.json").write_text('"15550000001"', encoding="utf-8")

    runner, _adapter = _make_runner(
        Platform.WHATSAPP,
        GatewayConfig(platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)}),
    )

    source = SessionSource(
        platform=Platform.WHATSAPP,
        user_id="900000000000001@lid",
        chat_id="900000000000001@lid",
        user_name="tester",
        chat_type="dm",
    )

    assert runner._is_user_authorized(source) is True


def test_star_wildcard_in_allowlist_authorizes_any_user(monkeypatch):
    """WHATSAPP_ALLOWED_USERS=* should act as allow-all wildcard."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    runner, _adapter = _make_runner(
        Platform.WHATSAPP,
        GatewayConfig(platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)}),
    )

    source = SessionSource(
        platform=Platform.WHATSAPP,
        user_id="99998887776@s.whatsapp.net",
        chat_id="99998887776@s.whatsapp.net",
        user_name="stranger",
        chat_type="dm",
    )
    assert runner._is_user_authorized(source) is True


def test_star_wildcard_works_for_any_platform(monkeypatch):
    """The * wildcard should work generically, not just for WhatsApp."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")

    runner, _adapter = _make_runner(
        Platform.TELEGRAM,
        GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")}),
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="123456789",
        chat_id="123456789",
        user_name="stranger",
        chat_type="dm",
    )
    assert runner._is_user_authorized(source) is True


def test_qq_group_allowlist_authorizes_group_chat_without_user_allowlist(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("QQ_GROUP_ALLOWED_USERS", "group-openid-1")

    runner, _adapter = _make_runner(
        Platform.QQBOT,
        GatewayConfig(platforms={Platform.QQBOT: PlatformConfig(enabled=True)}),
    )

    source = SessionSource(
        platform=Platform.QQBOT,
        user_id="member-openid-999",
        chat_id="group-openid-1",
        user_name="tester",
        chat_type="group",
    )

    assert runner._is_user_authorized(source) is True


def test_qq_group_allowlist_does_not_authorize_other_groups(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("QQ_GROUP_ALLOWED_USERS", "group-openid-1")

    runner, _adapter = _make_runner(
        Platform.QQBOT,
        GatewayConfig(platforms={Platform.QQBOT: PlatformConfig(enabled=True)}),
    )

    source = SessionSource(
        platform=Platform.QQBOT,
        user_id="member-openid-999",
        chat_id="group-openid-2",
        user_name="tester",
        chat_type="group",
    )

    assert runner._is_user_authorized(source) is False


@pytest.mark.asyncio
async def test_unauthorized_dm_pairs_by_default(monkeypatch):
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)},
    )
    runner, adapter = _make_runner(Platform.WHATSAPP, config)
    runner.pairing_store.generate_code.return_value = "ABC12DEF"

    result = await runner._handle_message(
        _make_event(
            Platform.WHATSAPP,
            "15551234567@s.whatsapp.net",
            "15551234567@s.whatsapp.net",
        )
    )

    assert result is None
    runner.pairing_store.generate_code.assert_called_once_with(
        "whatsapp",
        "15551234567@s.whatsapp.net",
        "tester",
    )
    adapter.send.assert_awaited_once()
    assert "ABC12DEF" in adapter.send.await_args.args[1]


@pytest.mark.asyncio
async def test_unauthorized_whatsapp_dm_can_be_ignored(monkeypatch):
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(
                enabled=True,
                extra={"unauthorized_dm_behavior": "ignore"},
            ),
        },
    )
    runner, adapter = _make_runner(Platform.WHATSAPP, config)

    result = await runner._handle_message(
        _make_event(
            Platform.WHATSAPP,
            "15551234567@s.whatsapp.net",
            "15551234567@s.whatsapp.net",
        )
    )

    assert result is None
    runner.pairing_store.generate_code.assert_not_called()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_rate_limited_user_gets_no_response(monkeypatch):
    """When a user is already rate-limited, pairing messages are silently ignored."""
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)},
    )
    runner, adapter = _make_runner(Platform.WHATSAPP, config)
    runner.pairing_store._is_rate_limited.return_value = True

    result = await runner._handle_message(
        _make_event(
            Platform.WHATSAPP,
            "15551234567@s.whatsapp.net",
            "15551234567@s.whatsapp.net",
        )
    )

    assert result is None
    runner.pairing_store.generate_code.assert_not_called()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_rejection_message_records_rate_limit(monkeypatch):
    """After sending a 'too many requests' rejection, rate limit is recorded
    so subsequent messages are silently ignored."""
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)},
    )
    runner, adapter = _make_runner(Platform.WHATSAPP, config)
    runner.pairing_store.generate_code.return_value = None  # triggers rejection

    result = await runner._handle_message(
        _make_event(
            Platform.WHATSAPP,
            "15551234567@s.whatsapp.net",
            "15551234567@s.whatsapp.net",
        )
    )

    assert result is None
    adapter.send.assert_awaited_once()
    assert "Too many" in adapter.send.await_args.args[1]
    runner.pairing_store._record_rate_limit.assert_called_once_with(
        "whatsapp", "15551234567@s.whatsapp.net"
    )


@pytest.mark.asyncio
async def test_global_ignore_suppresses_pairing_reply(monkeypatch):
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        unauthorized_dm_behavior="ignore",
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
    )
    runner, adapter = _make_runner(Platform.TELEGRAM, config)

    result = await runner._handle_message(
        _make_event(
            Platform.TELEGRAM,
            "12345",
            "12345",
        )
    )

    assert result is None
    runner.pairing_store.generate_code.assert_not_called()
    adapter.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# Allowlist-configured platforms default to "ignore" for unauthorized users
# (#9337: Signal gateway sends pairing spam when allowlist is configured)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_signal_with_allowlist_ignores_unauthorized_dm(monkeypatch):
    """When SIGNAL_ALLOWED_USERS is set, unauthorized DMs are silently dropped.

    This is the primary regression test for #9337: before the fix, Signal
    would send pairing codes to ANY sender even when a strict allowlist was
    configured, spamming personal contacts with cryptic bot messages.
    """
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "+15550000001")  # allowlist set

    config = GatewayConfig(
        platforms={Platform.SIGNAL: PlatformConfig(enabled=True)},
    )
    runner, adapter = _make_runner(Platform.SIGNAL, config)

    result = await runner._handle_message(
        _make_event(Platform.SIGNAL, "+15559999999", "+15559999999")  # not in allowlist
    )

    assert result is None
    runner.pairing_store.generate_code.assert_not_called()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_telegram_with_allowlist_ignores_unauthorized_dm(monkeypatch):
    """Same behavior for Telegram: allowlist ⟹ ignore unauthorized DMs."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111111111")

    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)},
    )
    runner, adapter = _make_runner(Platform.TELEGRAM, config)

    result = await runner._handle_message(
        _make_event(Platform.TELEGRAM, "999999999", "999999999")
    )

    assert result is None
    runner.pairing_store.generate_code.assert_not_called()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_global_allowlist_ignores_unauthorized_dm(monkeypatch):
    """GATEWAY_ALLOWED_USERS also triggers the 'ignore' behavior."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "111111111")

    config = GatewayConfig(
        platforms={Platform.SIGNAL: PlatformConfig(enabled=True)},
    )
    runner, adapter = _make_runner(Platform.SIGNAL, config)

    result = await runner._handle_message(
        _make_event(Platform.SIGNAL, "+15559999999", "+15559999999")
    )

    assert result is None
    runner.pairing_store.generate_code.assert_not_called()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_allowlist_still_pairs_by_default(monkeypatch):
    """Without any allowlist, pairing behavior is preserved (open gateway)."""
    _clear_auth_env(monkeypatch)
    # No SIGNAL_ALLOWED_USERS, no GATEWAY_ALLOWED_USERS

    config = GatewayConfig(
        platforms={Platform.SIGNAL: PlatformConfig(enabled=True)},
    )
    runner, adapter = _make_runner(Platform.SIGNAL, config)
    runner.pairing_store.generate_code.return_value = "PAIR1234"

    result = await runner._handle_message(
        _make_event(Platform.SIGNAL, "+15559999999", "+15559999999")
    )

    assert result is None
    runner.pairing_store.generate_code.assert_called_once()
    adapter.send.assert_awaited_once()
    assert "PAIR1234" in adapter.send.await_args.args[1]


def test_explicit_pair_config_overrides_allowlist_default(monkeypatch):
    """Explicit unauthorized_dm_behavior='pair' overrides the allowlist default.

    Operators can opt back in to pairing even with an allowlist by setting
    unauthorized_dm_behavior: pair in their platform config.  We test the
    _get_unauthorized_dm_behavior resolver directly to avoid the full
    _handle_message pipeline which requires extensive runner state.
    """
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "+15550000001")

    config = GatewayConfig(
        platforms={
            Platform.SIGNAL: PlatformConfig(
                enabled=True,
                extra={"unauthorized_dm_behavior": "pair"},  # explicit override
            ),
        },
    )
    runner, _adapter = _make_runner(Platform.SIGNAL, config)

    # The per-platform explicit config should beat the allowlist-derived default
    behavior = runner._get_unauthorized_dm_behavior(Platform.SIGNAL)
    assert behavior == "pair"


def test_allowlist_authorized_user_returns_ignore_for_unauthorized(monkeypatch):
    """_get_unauthorized_dm_behavior returns 'ignore' when allowlist is set.

    We test the resolver directly.  The full _handle_message path for
    authorized users is covered by the integration tests in this module.
    """
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "+15550000001")

    config = GatewayConfig(
        platforms={Platform.SIGNAL: PlatformConfig(enabled=True)},
    )
    runner, _adapter = _make_runner(Platform.SIGNAL, config)

    behavior = runner._get_unauthorized_dm_behavior(Platform.SIGNAL)
    assert behavior == "ignore"


def test_get_unauthorized_dm_behavior_no_allowlist_returns_pair(monkeypatch):
    """Without any allowlist, 'pair' is still the default."""
    _clear_auth_env(monkeypatch)

    config = GatewayConfig(
        platforms={Platform.SIGNAL: PlatformConfig(enabled=True)},
    )
    runner, _adapter = _make_runner(Platform.SIGNAL, config)

    behavior = runner._get_unauthorized_dm_behavior(Platform.SIGNAL)
    assert behavior == "pair"


def test_qqbot_with_allowlist_ignores_unauthorized_dm(monkeypatch):
    """QQBOT is included in the allowlist-aware default (QQ_ALLOWED_USERS).

    Regression guard: the initial #9337 fix omitted QQBOT from the env map
    inside _get_unauthorized_dm_behavior, even though _is_user_authorized
    mapped it to QQ_ALLOWED_USERS.  Without QQBOT here, a QQ operator with a
    strict user allowlist would still get pairing codes sent to strangers.
    """
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("QQ_ALLOWED_USERS", "allowed-openid-1")

    config = GatewayConfig(
        platforms={Platform.QQBOT: PlatformConfig(enabled=True)},
    )
    runner, _adapter = _make_runner(Platform.QQBOT, config)

    behavior = runner._get_unauthorized_dm_behavior(Platform.QQBOT)
    assert behavior == "ignore"
