"""Telegram Business delegated-inbox routing tests."""

from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import _thread_metadata_for_source
from gateway.session import SessionSource, build_session_key
from plugins.platforms.telegram import adapter as telegram


def _adapter(business=None):
    config = PlatformConfig(
        enabled=True,
        token="fake",
        extra={"business": business or {}},
    )
    instance = object.__new__(telegram.TelegramAdapter)
    instance.config = config
    instance._config = config
    instance._platform = Platform.TELEGRAM
    instance.platform = Platform.TELEGRAM
    instance._bot = SimpleNamespace(id=999, username="hermes_bot")
    instance._mention_patterns = []
    instance._dm_topics = {}
    instance._dm_topics_config = []
    instance._reply_to_mode = "first"
    instance._rich_messages_enabled = False
    instance._send_path_degraded = False
    instance._disable_link_previews = False
    instance._telegram_typing_cooldown_until = {}
    instance._telegram_typing_cooldown_seconds = 30.0
    return instance


def _message(
    *,
    text="Sigurd, hello",
    user_id=123,
    chat_id=456,
    business_connection_id="bc-123",
    is_bot=False,
    reply_to_message=None,
):
    return SimpleNamespace(
        text=text,
        caption=None,
        chat=SimpleNamespace(id=chat_id, type="private", title=None, full_name="Customer"),
        from_user=SimpleNamespace(id=user_id, full_name="Customer", is_bot=is_bot),
        message_thread_id=None,
        reply_to_message=reply_to_message,
        message_id=42,
        date=None,
        business_connection_id=business_connection_id,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({}, {"enabled": False, "allowed_chats": [], "trigger_words": []}),
        (
            {"enabled": "YES", "allowed_chats": 42, "trigger_words": "Sigurd"},
            {"enabled": True, "allowed_chats": ["42"], "trigger_words": ["Sigurd"]},
        ),
        (
            {"enabled": "off", "allowed_chats": ["", " 7 ", None], "trigger_words": [" Argus ", ""]},
            {"enabled": False, "allowed_chats": ["7"], "trigger_words": ["Argus"]},
        ),
    ],
)
def test_apply_yaml_config_normalizes_business(raw, expected):
    telegram_cfg = {"business": raw}
    extras = telegram._apply_yaml_config({}, telegram_cfg)
    assert extras["business"] == expected


def test_top_level_telegram_business_config_reaches_platform_extra(tmp_path, monkeypatch):
    import gateway.config as config_mod

    (tmp_path / "config.yaml").write_text(
        """
telegram:
  business:
    enabled: true
    allowed_chats: 456
    trigger_words: Sigurd
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(config_mod, "get_hermes_home", lambda: tmp_path)
    config = config_mod.load_gateway_config()
    assert config.platforms[Platform.TELEGRAM].extra["business"] == {
        "enabled": True,
        "allowed_chats": ["456"],
        "trigger_words": ["Sigurd"],
    }


def test_business_session_keys_are_profile_actor_thread_and_trust_aware():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="456",
        user_id="123",
        thread_id="9",
        profile="coder",
        business_connection_id="bc",
        external_safe_mode=True,
    )
    assert (
        build_session_key(source, profile=source.profile)
        == "agent:coder:telegram:business:bc:456:9:123:external"
    )


def test_business_session_key_uses_unknown_actor_and_trusted_mode():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="456",
        business_connection_id="bc",
    )
    assert build_session_key(source) == "agent:main:telegram:business:bc:456:unknown:trusted"


def test_business_session_key_isolates_business_connections():
    first = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="456",
        user_id="123",
        business_connection_id=" bc-one ",
        external_safe_mode=True,
    )
    second = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="456",
        user_id="123",
        business_connection_id="bc-two",
        external_safe_mode=True,
    )
    assert build_session_key(first) == (
        "agent:main:telegram:business:bc-one:456:123:external"
    )
    assert build_session_key(first) != build_session_key(second)


def test_business_source_persists_only_meaningful_values_and_reads_legacy():
    plain = SessionSource(platform=Platform.TELEGRAM, chat_id="1")
    assert "business_connection_id" not in plain.to_dict()
    assert "external_safe_mode" not in plain.to_dict()

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="1",
        business_connection_id="bc",
        external_safe_mode=True,
    )
    restored = SessionSource.from_dict(source.to_dict())
    assert restored.business_connection_id == "bc"
    assert restored.external_safe_mode is True
    legacy = SessionSource.from_dict(
        {"platform": "telegram", "chat_id": "1", "telegram_business_connection_id": "old"}
    )
    assert legacy.business_connection_id == "old"


def test_business_metadata_is_canonical_and_unthreaded():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="1",
        thread_id="9",
        business_connection_id="bc",
        external_safe_mode=True,
    )
    assert _thread_metadata_for_source(source) == {
        "business_connection_id": "bc",
        "external_safe_mode": True,
        "telegram_business_external_contact": True,
    }


@pytest.mark.parametrize(
    ("text", "matches"),
    [
        ("sigurd hello", True),
        ("SIGURD, hello", True),
        ("sigurd: hello", True),
        ("sigurdian hello", False),
        ("hello sigurd", False),
    ],
)
def test_trigger_words_require_case_insensitive_leading_boundary(text, matches):
    adapter = _adapter({"enabled": True, "trigger_words": ["Sigurd"]})
    assert adapter._business_trigger_text(_message(text=text)) == ("hello" if matches else None)


@pytest.mark.asyncio
async def test_business_guard_order_rejects_disabled_missing_route_empty_bot_and_chat():
    captured = []
    for business, message in [
        ({"enabled": False, "trigger_words": ["Sigurd"]}, _message()),
        ({"enabled": True, "trigger_words": ["Sigurd"]}, _message(business_connection_id=None)),
        ({"enabled": True, "trigger_words": ["Sigurd"]}, _message(text=None)),
        ({"enabled": True, "trigger_words": ["Sigurd"]}, _message(is_bot=True)),
        (
            {"enabled": True, "allowed_chats": ["999"], "trigger_words": ["Sigurd"]},
            _message(),
        ),
    ]:
        adapter = _adapter(business)
        adapter.handle_message = captured.append
        await adapter._handle_business_message(
            SimpleNamespace(update_id=1, business_message=message), SimpleNamespace()
        )
    assert captured == []


@pytest.mark.asyncio
async def test_trigger_reply_and_mention_independently_accept(monkeypatch):
    monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    accepted = []
    for message in [
        _message(),
        _message(text="ordinary", reply_to_message=SimpleNamespace(from_user=SimpleNamespace(id=999))),
        _message(text="@hermes_bot ordinary"),
    ]:
        adapter = _adapter({"enabled": True, "trigger_words": ["Sigurd"]})
        adapter.handle_message = accepted.append
        await adapter._handle_business_message(
            SimpleNamespace(update_id=1, business_message=message), SimpleNamespace()
        )
    assert len(accepted) == 3
    assert all(event.source.external_safe_mode for event in accepted)


@pytest.mark.parametrize("env_name", ["TELEGRAM_ALLOWED_USERS", "GATEWAY_ALLOWED_USERS"])
def test_explicit_operator_ids_are_trusted_but_wildcard_is_ignored(monkeypatch, env_name):
    monkeypatch.setenv(env_name, "123,*")
    adapter = _adapter({"enabled": True})
    assert adapter._is_business_trusted_actor(_message(user_id=123)) is True
    assert adapter._is_business_trusted_actor(_message(user_id=456)) is False


def test_pairing_approval_fallback_and_error_fail_closed(monkeypatch):
    monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    adapter = _adapter({"enabled": True})
    adapter.pairing_store = SimpleNamespace(is_approved=lambda platform, user: user == "123")
    assert adapter._is_business_trusted_actor(_message(user_id=123)) is True
    adapter.pairing_store = SimpleNamespace(
        is_approved=lambda *_: (_ for _ in ()).throw(OSError("broken"))
    )
    assert adapter._is_business_trusted_actor(_message(user_id=123)) is False


@pytest.mark.asyncio
async def test_untrusted_business_event_is_external_safe():
    accepted = []
    adapter = _adapter({"enabled": True, "trigger_words": ["Sigurd"]})
    adapter.handle_message = accepted.append
    await adapter._handle_business_message(
        SimpleNamespace(update_id=1, business_message=_message()), SimpleNamespace()
    )
    event = accepted[0]
    assert event.source.business_connection_id == "bc-123"
    assert event.source.external_safe_mode is True


@pytest.mark.asyncio
async def test_explicit_operator_business_event_is_not_external_safe(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "123")
    accepted = []
    adapter = _adapter({"enabled": True, "trigger_words": ["Sigurd"]})
    adapter.handle_message = accepted.append
    await adapter._handle_business_message(
        SimpleNamespace(update_id=1, business_message=_message()), SimpleNamespace()
    )
    assert accepted[0].source.external_safe_mode is False


@pytest.mark.asyncio
async def test_text_and_typing_propagate_business_connection_id():
    calls = []

    async def send_message(**kwargs):
        calls.append(("text", kwargs))
        return SimpleNamespace(message_id=1)

    async def send_chat_action(**kwargs):
        calls.append(("typing", kwargs))

    adapter = _adapter()
    adapter._bot = SimpleNamespace(
        id=999,
        username="hermes_bot",
        send_message=send_message,
        send_chat_action=send_chat_action,
    )
    result = await adapter.send(
        "456", "hello", metadata={"business_connection_id": "bc-123", "notify": True}
    )
    await adapter.send_typing("456", metadata={"business_connection_id": "bc-123"})
    assert result.success
    assert [kwargs["business_connection_id"] for _, kwargs in calls] == ["bc-123", "bc-123"]


@pytest.mark.asyncio
async def test_ordinary_text_send_omits_business_connection_id():
    calls = []

    async def send_message(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(message_id=1)

    adapter = _adapter()
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot", send_message=send_message)
    assert (await adapter.send("456", "hello", metadata={"notify": True})).success
    assert "business_connection_id" not in calls[0]


@pytest.mark.asyncio
async def test_edit_propagates_business_connection_id():
    calls = []

    async def edit_message_text(**kwargs):
        calls.append(kwargs)

    adapter = _adapter()
    adapter._bot = SimpleNamespace(
        id=999,
        username="hermes_bot",
        edit_message_text=edit_message_text,
    )
    adapter._last_overflow_preview = {}
    result = await adapter.edit_message(
        "456",
        "9",
        "preview",
        metadata={"business_connection_id": "bc-123"},
    )
    assert result.success
    assert calls[0]["business_connection_id"] == "bc-123"


@pytest.mark.asyncio
async def test_draft_propagates_business_connection_id():
    calls = []

    async def send_message_draft(**kwargs):
        calls.append(kwargs)
        return True

    adapter = _adapter()
    adapter._bot = SimpleNamespace(
        id=999,
        username="hermes_bot",
        send_message_draft=send_message_draft,
    )
    adapter._rich_drafts_enabled = False
    result = await adapter.send_draft(
        "456",
        1,
        "preview",
        metadata={"business_connection_id": "bc-123"},
    )
    assert result.success
    assert calls[0]["business_connection_id"] == "bc-123"


def test_all_thread_routed_outbound_paths_receive_business_route():
    adapter = _adapter()
    assert adapter._thread_kwargs_for_send(
        "456", None, {"business_connection_id": "bc-123"}
    ) == {"message_thread_id": None, "business_connection_id": "bc-123"}
    assert adapter._thread_kwargs_for_send("456", None, None) == {
        "message_thread_id": None
    }


@pytest.mark.asyncio
async def test_business_lifecycle_handlers_are_observational(caplog):
    caplog.set_level("INFO")
    adapter = _adapter()
    adapter._business_state = {"sentinel": True}
    before = dict(adapter.__dict__)
    await adapter._handle_business_connection(SimpleNamespace(), SimpleNamespace())
    await adapter._handle_business_messages_deleted(SimpleNamespace(), SimpleNamespace())
    assert adapter.__dict__ == before
    assert "lifecycle update observed" in caplog.text
    assert "deletion lifecycle update observed" in caplog.text


def test_business_metadata_reader_accepts_legacy_key():
    adapter = _adapter()
    assert adapter._business_kwargs({"telegram_business_connection_id": "legacy"}) == {
        "business_connection_id": "legacy"
    }


def test_business_safe_tool_contract_is_exact():
    from gateway.run import (
        _business_contact_web_tool_names,
        _restrict_agent_to_tool_names,
    )

    assert _business_contact_web_tool_names() == {"web_search", "web_extract"}
    agent = SimpleNamespace(
        tools=[
            {"function": {"name": "web_search"}},
            {"function": {"name": "web_extract"}},
            {"function": {"name": "terminal"}},
        ]
    )
    _restrict_agent_to_tool_names(agent, _business_contact_web_tool_names())
    assert [tool["function"]["name"] for tool in agent.tools] == [
        "web_search",
        "web_extract",
    ]
    assert agent.valid_tool_names == {"web_search", "web_extract"}


@pytest.mark.asyncio
async def test_external_business_turn_fails_closed_before_proxy(monkeypatch):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    proxy_calls = []
    runner._get_proxy_url = lambda: "https://unrestricted-agent.example"

    async def proxy(**kwargs):
        proxy_calls.append(kwargs)
        return {"final_response": "unsafe"}

    runner._run_agent_via_proxy = proxy
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="456",
        user_id="123",
        business_connection_id="bc",
        external_safe_mode=True,
    )
    result = await runner._run_agent_inner(
        "read local secrets",
        "",
        [],
        source,
        "session-id",
        session_key=build_session_key(source),
    )
    assert proxy_calls == []
    assert result["final_response"] == (
        "I can’t process external Telegram Business contacts through the "
        "configured remote agent."
    )
    assert result["tools"] == []
