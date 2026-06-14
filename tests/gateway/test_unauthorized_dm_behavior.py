from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_USERS",
        "DISCORD_ALLOWED_USERS",
        "WHATSAPP_ALLOWED_USERS",
        "SLACK_ALLOWED_USERS",
        "SIGNAL_ALLOWED_USERS",
        "SIGNAL_GROUP_ALLOWED_USERS",
        "TELEGRAM_GROUP_ALLOWED_CHATS",
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
        "LINEAR_ALLOWED_USERS",
        "LINEAR_ALLOW_ALL_USERS",
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
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

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


def test_simplex_allowlist_accepts_display_name(monkeypatch):
    """SIMPLEX_ALLOWED_USERS should match the contact's display name as well
    as the numeric contactId. The SimpleX UI surfaces only display names, so
    operators naturally put those in the env var — and the adapter sets
    user_id=contactId for stability. Both forms must work. (#TBD)"""
    _clear_auth_env(monkeypatch)
    monkeypatch.delenv("SIMPLEX_ALLOWED_USERS", raising=False)
    monkeypatch.setenv("SIMPLEX_ALLOWED_USERS", "hujikuji")

    # Register the simplex plugin so the env-var lookup resolves.
    from gateway.platform_registry import platform_registry, PlatformEntry
    platform_registry.register(PlatformEntry(
        name="simplex",
        label="SimpleX Chat",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        allowed_users_env="SIMPLEX_ALLOWED_USERS",
        allow_all_env="SIMPLEX_ALLOW_ALL_USERS",
    ))

    simplex = Platform("simplex")
    runner, _adapter = _make_runner(
        simplex,
        GatewayConfig(platforms={simplex: PlatformConfig(enabled=True)}),
    )

    # contactId in the allowlist would still work — but the operator chose
    # the display name. Verify the gateway honors it.
    source = SessionSource(
        platform=simplex,
        user_id="4",            # adapter sets this to the numeric contactId
        chat_id="hujikuji",
        user_name="hujikuji",   # adapter sets this to displayName
        chat_type="dm",
    )
    assert runner._is_user_authorized(source) is True


def test_plugin_allow_all_authorizes_linear_agent_session(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("LINEAR_ALLOW_ALL_USERS", "true")

    from gateway.platform_registry import platform_registry, PlatformEntry
    platform_registry.register(PlatformEntry(
        name="linear",
        label="Linear Agent Sessions",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        allowed_users_env="LINEAR_ALLOWED_USERS",
        allow_all_env="LINEAR_ALLOW_ALL_USERS",
    ))

    linear = Platform("linear")
    runner, _adapter = _make_runner(
        linear,
        GatewayConfig(platforms={linear: PlatformConfig(enabled=True)}),
    )

    source = SessionSource(
        platform=linear,
        user_id="linear-agent-session",
        chat_id="agentSession:as_123",
        user_name="Linear Agent Session",
        chat_type="dm",
    )

    assert runner._is_user_authorized(source) is True


def test_simplex_allowlist_accepts_numeric_contact_id(monkeypatch):
    """The numeric contactId form must still work — the new display-name
    matching must not regress existing setups."""
    _clear_auth_env(monkeypatch)
    monkeypatch.delenv("SIMPLEX_ALLOWED_USERS", raising=False)
    monkeypatch.setenv("SIMPLEX_ALLOWED_USERS", "4")

    from gateway.platform_registry import platform_registry, PlatformEntry
    platform_registry.register(PlatformEntry(
        name="simplex",
        label="SimpleX Chat",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        allowed_users_env="SIMPLEX_ALLOWED_USERS",
        allow_all_env="SIMPLEX_ALLOW_ALL_USERS",
    ))

    simplex = Platform("simplex")
    runner, _adapter = _make_runner(
        simplex,
        GatewayConfig(platforms={simplex: PlatformConfig(enabled=True)}),
    )

    source = SessionSource(
        platform=simplex,
        user_id="4",
        chat_id="hujikuji",
        user_name="hujikuji",
        chat_type="dm",
    )
    assert runner._is_user_authorized(source) is True


def test_simplex_allowlist_denies_unlisted(monkeypatch):
    """Sanity check: an unrelated SimpleX user is still rejected."""
    _clear_auth_env(monkeypatch)
    monkeypatch.delenv("SIMPLEX_ALLOWED_USERS", raising=False)
    monkeypatch.setenv("SIMPLEX_ALLOWED_USERS", "hujikuji")

    from gateway.platform_registry import platform_registry, PlatformEntry
    platform_registry.register(PlatformEntry(
        name="simplex",
        label="SimpleX Chat",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        allowed_users_env="SIMPLEX_ALLOWED_USERS",
        allow_all_env="SIMPLEX_ALLOW_ALL_USERS",
    ))

    simplex = Platform("simplex")
    runner, _adapter = _make_runner(
        simplex,
        GatewayConfig(platforms={simplex: PlatformConfig(enabled=True)}),
    )

    source = SessionSource(
        platform=simplex,
        user_id="7",
        chat_id="stranger",
        user_name="stranger",
        chat_type="dm",
    )
    assert runner._is_user_authorized(source) is False


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


def test_telegram_group_user_allowlist_authorizes_forum_sender_without_dm_allowlist(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_USERS", "999")

    runner, _adapter = _make_runner(
        Platform.TELEGRAM,
        GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")}),
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="999",
        chat_id="-1001878443972",
        user_name="tester",
        chat_type="forum",
    )

    assert runner._is_user_authorized(source) is True


def test_telegram_group_user_allowlist_rejects_other_senders(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_USERS", "999")

    runner, _adapter = _make_runner(
        Platform.TELEGRAM,
        GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")}),
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="123",
        chat_id="-1001878443972",
        user_name="tester",
        chat_type="group",
    )

    assert runner._is_user_authorized(source) is False


def test_telegram_group_user_allowlist_wildcard_authorizes_any_sender(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_USERS", "*")

    runner, _adapter = _make_runner(
        Platform.TELEGRAM,
        GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")}),
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="123",
        chat_id="-1001878443972",
        user_name="tester",
        chat_type="group",
    )

    assert runner._is_user_authorized(source) is True


def test_telegram_group_user_allowlist_does_not_authorize_dms(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_USERS", "999")

    runner, _adapter = _make_runner(
        Platform.TELEGRAM,
        GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")}),
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="999",
        chat_id="999",
        user_name="tester",
        chat_type="dm",
    )

    assert runner._is_user_authorized(source) is False


def test_telegram_group_chat_allowlist_authorizes_group_chat_without_user_allowlist(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-1001878443972")

    runner, _adapter = _make_runner(
        Platform.TELEGRAM,
        GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")}),
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="999",
        chat_id="-1001878443972",
        user_name="tester",
        chat_type="forum",
    )

    assert runner._is_user_authorized(source) is True


def test_telegram_group_chat_allowlist_authorizes_anonymous_sender(monkeypatch):
    """TELEGRAM_GROUP_ALLOWED_CHATS must authorize chat traffic with no
    sender user_id (Telegram anonymous-admin posts, sender_chat). The
    docs state the chat allowlist authorizes "every member of that chat,
    regardless of sender" — anonymous senders had been silently dropped
    despite an explicit chat opt-in.
    """
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-1001878443972")

    runner, _adapter = _make_runner(
        Platform.TELEGRAM,
        GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")}),
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id=None,
        chat_id="-1001878443972",
        user_name=None,
        chat_type="group",
    )

    assert runner._is_user_authorized(source) is True


def test_telegram_group_chat_allowlist_rejects_anonymous_sender_in_other_chat(monkeypatch):
    """Anonymous senders in a chat *not* on the allowlist must still be
    rejected — the early no-user-id path must not become an open gate.
    """
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-1001878443972")

    runner, _adapter = _make_runner(
        Platform.TELEGRAM,
        GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")}),
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id=None,
        chat_id="-1009999999999",
        user_name=None,
        chat_type="group",
    )

    assert runner._is_user_authorized(source) is False


@pytest.mark.asyncio
async def test_handle_message_does_not_drop_anonymous_sender_in_allowlisted_chat(monkeypatch):
    """End-to-end: a group message with from_user=None in an allowlisted
    chat must reach the dispatch path — not get silently dropped by the
    no-user-id guard, and not trigger pairing (anonymous senders can't
    be paired anyway).
    """
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-1001878443972")

    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")},
    )
    runner, adapter = _make_runner(Platform.TELEGRAM, config)

    # Force _handle_message to bail with a sentinel right after the
    # auth gate, so a successful "auth passed" call can be distinguished
    # from the buggy "silently dropped" case (which would return None
    # before this hook ever runs).
    reached_dispatch = MagicMock(side_effect=RuntimeError("reached dispatch"))
    runner._session_key_for_source = reached_dispatch

    event = MessageEvent(
        text="hi",
        message_id="m1",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id=None,
            chat_id="-1001878443972",
            user_name=None,
            chat_type="group",
        ),
    )

    with pytest.raises(RuntimeError, match="reached dispatch"):
        await runner._handle_message(event)

    reached_dispatch.assert_called_once()
    runner.pairing_store.generate_code.assert_not_called()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_message_drops_anonymous_sender_outside_allowlist(monkeypatch):
    """Anonymous senders in a chat *not* on the allowlist remain silently
    dropped — the fix must not become a backdoor for unauthorized chats.
    """
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-1001878443972")

    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")},
    )
    runner, adapter = _make_runner(Platform.TELEGRAM, config)

    must_not_run = MagicMock(side_effect=AssertionError("auth gate did not drop"))
    runner._session_key_for_source = must_not_run

    event = MessageEvent(
        text="hi",
        message_id="m1",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id=None,
            chat_id="-1009999999999",
            user_name=None,
            chat_type="group",
        ),
    )

    result = await runner._handle_message(event)

    assert result is None
    must_not_run.assert_not_called()
    runner.pairing_store.generate_code.assert_not_called()
    adapter.send.assert_not_awaited()


def test_telegram_group_users_legacy_chat_ids_still_authorize(monkeypatch):
    """Backward-compat: PR #15027 shipped TELEGRAM_GROUP_ALLOWED_USERS as a
    chat-ID allowlist. PR #17686 renamed it to sender IDs and added
    TELEGRAM_GROUP_ALLOWED_CHATS. Users on the old guidance must keep working:
    chat-ID-shaped values (starting with "-") in the _USERS var are honored as
    chat IDs with a deprecation warning.
    """
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_USERS", "-1001878443972")

    runner, _adapter = _make_runner(
        Platform.TELEGRAM,
        GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")}),
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="999",
        chat_id="-1001878443972",
        user_name="tester",
        chat_type="forum",
    )

    assert runner._is_user_authorized(source) is True


def test_telegram_group_users_legacy_does_not_cross_chats(monkeypatch):
    """Legacy chat-ID value only authorizes the listed chat, not any group."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_USERS", "-1001878443972")

    runner, _adapter = _make_runner(
        Platform.TELEGRAM,
        GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")}),
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="999",
        chat_id="-1009999999999",
        user_name="tester",
        chat_type="group",
    )

    assert runner._is_user_authorized(source) is False


def test_telegram_group_users_mixed_sender_and_legacy_chat(monkeypatch):
    """Mixed values: positive user ID gates senders; negative chat ID gates chat."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_USERS", "999,-1001878443972")

    runner, _adapter = _make_runner(
        Platform.TELEGRAM,
        GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")}),
    )

    # Legacy chat ID path: any sender in the listed chat is authorized
    legacy_chat_source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="123",
        chat_id="-1001878443972",
        user_name="tester",
        chat_type="group",
    )
    assert runner._is_user_authorized(legacy_chat_source) is True

    # Sender path: listed sender user ID authorized in any group
    sender_source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="999",
        chat_id="-1009999999999",
        user_name="tester",
        chat_type="group",
    )
    assert runner._is_user_authorized(sender_source) is True


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
