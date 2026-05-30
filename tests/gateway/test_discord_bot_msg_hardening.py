import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionSource
from plugins.platforms.discord.bot_msg_protocol import (
    build_discord_bot_msg_v1,
    is_discord_bot_routing_guard_error,
    is_bot_msg_required_error,
    parse_discord_bot_msg_v1,
)


class _RetryAdapter(BasePlatformAdapter):
    @property
    def name(self):
        return "retry-stub"

    def __init__(self):
        self.calls = []

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content, "reply_to": reply_to, "metadata": metadata})
        return SendResult(
            success=False,
            error="Outbound raw mention of allowed bot 777 requires send_bot_message(...) to create a BOT_MSG v1 envelope",
        )

    async def get_chat_info(self, chat_id):
        return {}

    def _is_terminal_send_error(self, error):
        return is_bot_msg_required_error(error)


class _GuardAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)
        self.calls = []

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content, "reply_to": reply_to, "metadata": metadata})
        return SendResult(success=True, message_id=f"m-{len(self.calls)}")

    async def get_chat_info(self, chat_id):
        return {}


def test_protocol_body_is_opaque_to_header_looking_text():
    body = "line 1\nreply_expected: false\nkind: audit\n---\nstill body"
    envelope = build_discord_bot_msg_v1(
        recipient_bot_id="777",
        body=body,
        reply_expected=True,
        kind="handoff",
        correlation_id="corr-1",
    )

    parsed = parse_discord_bot_msg_v1(envelope, "777")

    assert parsed is not None
    assert parsed["reply_expected"] is True
    assert parsed["kind"] == "handoff"
    assert parsed["body"] == body


@pytest.mark.asyncio
async def test_send_with_retry_treats_bot_msg_required_error_as_terminal():
    adapter = _RetryAdapter()

    result = await adapter._send_with_retry("555", "plain ping <@777>")

    assert result.success is False
    assert is_bot_msg_required_error(result.error)
    assert len(adapter.calls) == 1
    assert not adapter.calls[0]["content"].startswith("(Response formatting failed")

@pytest.mark.asyncio
async def test_send_with_retry_treats_routing_guard_error_as_terminal():
    class GuardErrorAdapter(_RetryAdapter):
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.calls.append({"chat_id": chat_id, "content": content, "reply_to": reply_to, "metadata": metadata})
            return SendResult(success=False, error="BOT_ROUTING_GUARD: blocked ordinary bot-to-bot final response")

        def _is_terminal_send_error(self, error):
            return is_discord_bot_routing_guard_error(error)

    adapter = GuardErrorAdapter()

    result = await adapter._send_with_retry("555", "ACTION_REQUIRED for Galt/default: do this")

    assert result.success is False
    assert is_discord_bot_routing_guard_error(result.error)
    assert len(adapter.calls) == 1
    assert not adapter.calls[0]["content"].startswith("(Response formatting failed")


@pytest.mark.asyncio
async def test_discord_final_response_routing_guard_blocks_operational_bot_to_bot_at_baseline(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "777,888")
    monkeypatch.delenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", raising=False)
    adapter = _GuardAdapter()
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="555",
        chat_name="provide me with the status of the statute pm worker",
        chat_type="thread",
        thread_id="999",
        user_id="888",
        user_name="Statute PM",
        is_bot=True,
    )
    event = MessageEvent(text="trigger", source=source, message_id="111")
    original = "ACTION_REQUIRED for Galt/default: restart the gateway manually"

    result = await adapter._send_text_response_with_routing_guard(
        event=event,
        text_content=original,
        reply_to="111",
        metadata={"thread_id": "999", "notify": True},
    )

    assert result.success is True
    assert len(adapter.calls) == 1
    sent = adapter.calls[0]["content"]
    assert sent != original
    assert sent.startswith("[ROUTING_GUARD]")
    assert (tmp_path / "logs" / "routing_guard").exists()


@pytest.mark.asyncio
async def test_discord_final_response_routing_guard_allows_benign_hard_stop(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _GuardAdapter()
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="555",
        chat_type="thread",
        thread_id="999",
        user_id="human-1",
        user_name="Benjamin",
        is_bot=False,
    )
    event = MessageEvent(text="trigger", source=source, message_id="111")
    original = "We discussed a hard stop at 5pm. No bot action is needed."

    result = await adapter._send_text_response_with_routing_guard(
        event=event,
        text_content=original,
        reply_to="111",
        metadata={"thread_id": "999", "notify": True},
    )

    assert result.success is True
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["content"] == original
    assert not (tmp_path / "logs" / "routing_guard").exists()


def test_send_message_rejects_raw_allowed_discord_bot_mention(monkeypatch):
    import tools.send_message_tool as smt
    from gateway.config import Platform

    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "777")
    monkeypatch.setenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", "1")

    class FakeConfig:
        platforms = {Platform.DISCORD: PlatformConfig(enabled=True, token="***")}

        def get_home_channel(self, platform):
            return None

    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: FakeConfig())

    result = json.loads(smt._handle_send({"target": "discord:555", "message": "bad <@777>"}))

    assert "error" in result
    assert "send_bot_message" in result["error"]


def test_send_bot_approval_decision_builds_non_replying_structured_decision(monkeypatch, tmp_path):
    import tools.send_message_tool as smt
    from gateway.config import Platform

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "777")
    monkeypatch.setenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", "1")
    calls = []

    class FakeConfig:
        platforms = {Platform.DISCORD: PlatformConfig(enabled=True, token="***")}

        def get_home_channel(self, platform):
            return None

    async def fake_send(*args, **kwargs):
        calls.append(kwargs)
        return {"success": True, "message_id": "decision-1"}

    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: FakeConfig())
    monkeypatch.setattr(smt, "_send_bot_message_to_discord", fake_send)

    result = json.loads(
        smt._handle_send_bot_approval_decision(
            {
                "target": "discord:555:999",
                "recipient_bot_id": "777",
                "approval_id": "approval-123",
                "decision": "approve",
                "scope": "once",
                "correlation_id": "approval:approval-123",
            }
        )
    )

    assert result["success"] is True
    assert result["kind"] == "approval_decision"
    assert result["reply_expected"] is False
    assert calls[0]["kind"] == "approval_decision"
    assert calls[0]["reply_expected"] is False
    assert "approval_id: approval-123" in calls[0]["body"]
    assert "decision: approve" in calls[0]["body"]
    assert "scope: once" in calls[0]["body"]


def test_send_bot_message_requires_recipient_allowlist(monkeypatch):
    import tools.send_message_tool as smt
    from gateway.config import Platform

    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "888")
    monkeypatch.setenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", "1")

    class FakeConfig:
        platforms = {Platform.DISCORD: PlatformConfig(enabled=True, token="***")}

        def get_home_channel(self, platform):
            return None

    async def fake_send(*args, **kwargs):  # pragma: no cover - must not be reached
        return {"success": True, "message_id": "sent"}

    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: FakeConfig())
    monkeypatch.setattr(smt, "_send_bot_message_to_discord", fake_send)

    result = json.loads(
        smt._handle_send_bot_message(
            {
                "target": "discord:555",
                "recipient_bot_id": "777",
                "kind": "status",
                "reply_expected": False,
                "body": "body",
                "correlation_id": "corr-1",
            }
        )
    )

    assert "error" in result
    assert "allowlisted" in result["error"]


def test_send_bot_message_idempotency_returns_existing_delivery(monkeypatch, tmp_path):
    import tools.send_message_tool as smt
    from gateway.config import Platform

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "777")
    monkeypatch.setenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", "1")
    calls = []

    class FakeConfig:
        platforms = {Platform.DISCORD: PlatformConfig(enabled=True, token="***")}

        def get_home_channel(self, platform):
            return None

    async def fake_send(*args, **kwargs):
        calls.append(kwargs)
        return {"success": True, "message_id": "m-1"}

    payload = {
        "target": "discord:555",
        "recipient_bot_id": "777",
        "kind": "status",
        "reply_expected": False,
        "body": "body",
        "correlation_id": "corr-1",
    }
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: FakeConfig())
    monkeypatch.setattr(smt, "_send_bot_message_to_discord", fake_send)

    first = json.loads(smt._handle_send_bot_message(dict(payload)))
    second = json.loads(smt._handle_send_bot_message(dict(payload)))

    assert first["success"] is True
    assert first["message_id"] == "m-1"
    assert second["success"] is True
    assert second["skipped"] is True
    assert second["message_id"] == "m-1"
    assert len(calls) == 1


def test_send_bot_message_idempotency_survives_omitted_correlation_id(monkeypatch, tmp_path):
    import tools.send_message_tool as smt
    from gateway.config import Platform

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "777")
    monkeypatch.setenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", "1")
    calls = []

    class FakeConfig:
        platforms = {Platform.DISCORD: PlatformConfig(enabled=True, token="***")}

        def get_home_channel(self, platform):
            return None

    async def fake_send(*args, **kwargs):
        calls.append(kwargs)
        return {"success": True, "message_id": f"m-{len(calls)}"}

    payload = {
        "target": "discord:555",
        "recipient_bot_id": "777",
        "kind": "status",
        "reply_expected": False,
        "body": "body",
    }
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: FakeConfig())
    monkeypatch.setattr(smt, "_send_bot_message_to_discord", fake_send)

    first = json.loads(smt._handle_send_bot_message(dict(payload)))
    second = json.loads(smt._handle_send_bot_message(dict(payload)))

    assert first["success"] is True
    assert first["message_id"] == "m-1"
    assert first["correlation_id"].startswith("botmsg-")
    assert second["success"] is True
    assert second["skipped"] is True
    assert second["message_id"] == "m-1"
    assert second["correlation_id"] == first["correlation_id"]
    assert len(calls) == 1


def test_send_bot_message_rejects_non_numeric_reply_to(monkeypatch, tmp_path):
    import tools.send_message_tool as smt
    from gateway.config import Platform

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "777")
    monkeypatch.setenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", "1")

    class FakeConfig:
        platforms = {Platform.DISCORD: PlatformConfig(enabled=True, token="***")}

        def get_home_channel(self, platform):
            return None

    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: FakeConfig())
    result = json.loads(
        smt._handle_send_bot_message(
            {
                "target": "discord:555",
                "recipient_bot_id": "777",
                "kind": "status",
                "reply_expected": False,
                "body": "body",
                "reply_to": "not-a-snowflake",
            }
        )
    )

    assert "error" in result
    assert "reply_to" in result["error"]


def test_send_bot_message_idempotency_key_includes_protocol_context(monkeypatch, tmp_path):
    import tools.send_message_tool as smt
    from gateway.config import Platform

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "777")
    monkeypatch.setenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", "1")
    calls = []

    class FakeConfig:
        platforms = {Platform.DISCORD: PlatformConfig(enabled=True, token="***")}

        def get_home_channel(self, platform):
            return None

    async def fake_send(*args, **kwargs):
        calls.append(kwargs)
        return {"success": True, "message_id": f"m-{len(calls)}"}

    payload = {
        "target": "discord:555",
        "recipient_bot_id": "777",
        "kind": "status",
        "reply_expected": False,
        "body": "body",
        "correlation_id": "corr-1",
    }
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: FakeConfig())
    monkeypatch.setattr(smt, "_send_bot_message_to_discord", fake_send)

    first = json.loads(smt._handle_send_bot_message(dict(payload)))
    changed_reply = dict(payload, reply_expected=True, correlation_id="corr-2")
    second = json.loads(smt._handle_send_bot_message(changed_reply))
    third = json.loads(smt._handle_send_bot_message(dict(payload, reply_to="123")))

    assert first["message_id"] == "m-1"
    assert second["message_id"] == "m-2"
    assert third["message_id"] == "m-3"
    assert len(calls) == 3


def test_discord_adapter_marks_bot_msg_required_as_terminal_send_error():
    from gateway.platforms.discord import DiscordAdapter

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))

    assert adapter._is_terminal_send_error(
        "Outbound raw mention of allowed bot 777 requires send_bot_message(...) to create a BOT_MSG v1 envelope"
    ) is True
    assert adapter._is_terminal_send_error("ordinary failure") is False


def test_inbound_bot_msg_rejects_body_over_configured_cap(monkeypatch):
    from gateway.platforms.discord import DiscordAdapter

    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "12345")
    monkeypatch.setenv("DISCORD_BOT_MSG_MAX_BODY_CHARS", "4")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=99999))
    bot = SimpleNamespace(bot=True, id=12345)
    content = build_discord_bot_msg_v1(
        recipient_bot_id="99999",
        body="12345",
        reply_expected=True,
        kind="status",
        correlation_id="corr-1",
    )
    msg = SimpleNamespace(author=bot, content=content, channel=SimpleNamespace(id=222), id=1)

    assert adapter._should_accept_bot_message(msg, "mentions") is False
    assert adapter._should_react_malformed_bot_message(msg) is False


@pytest.mark.asyncio
async def test_inbound_approval_decision_resolves_live_request_without_model_turn(monkeypatch):
    from gateway.platforms.discord import DiscordAdapter
    import tools.approval as approval

    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "12345")
    monkeypatch.setenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", "1")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=99999))
    adapter._add_reaction = AsyncMock(return_value=True)
    entry = approval._ApprovalEntry({"approval_id": "approval-live"})
    with approval._lock:
        approval._gateway_queues.clear()
        approval._gateway_queues["session-a"] = [entry]
    try:
        bot = SimpleNamespace(bot=True, id=12345)
        body = "approval_id: approval-live\ndecision: approve\nscope: once"
        bot_msg = {"kind": "approval_decision", "body": body}
        msg = SimpleNamespace(author=bot, content="", channel=SimpleNamespace(id=222), id=1)

        handled = await adapter._handle_bot_approval_decision(msg, bot_msg)

        assert handled is True
        assert entry.result == "once"
        assert entry.event.is_set()
        adapter._add_reaction.assert_awaited_once_with(msg, "✅")
    finally:
        with approval._lock:
            approval._gateway_queues.clear()


@pytest.mark.asyncio
async def test_inbound_approval_decision_rejects_replay(monkeypatch):
    from gateway.platforms.discord import DiscordAdapter

    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "12345")
    monkeypatch.setenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", "1")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=99999))
    adapter._add_reaction = AsyncMock(return_value=True)
    bot = SimpleNamespace(bot=True, id=12345)
    body = "approval_id: not-live\ndecision: approve\nscope: once"
    bot_msg = {"kind": "approval_decision", "body": body}
    msg = SimpleNamespace(author=bot, content="", channel=SimpleNamespace(id=222), id=1)

    handled = await adapter._handle_bot_approval_decision(msg, bot_msg)

    assert handled is True
    adapter._add_reaction.assert_awaited_once_with(msg, "❌")


def test_malformed_bot_msg_reaction_only_for_invalid_envelope(monkeypatch):
    from gateway.platforms.discord import DiscordAdapter

    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "12345")
    monkeypatch.setenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", "1")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=99999))
    bot = SimpleNamespace(bot=True, id=12345)

    malformed = SimpleNamespace(author=bot, content="<@99999> free-form raw mention")
    valid_but_rejected = SimpleNamespace(
        author=bot,
        content=build_discord_bot_msg_v1(
            recipient_bot_id="99999",
            body="too long for a cap, but syntactically valid",
            reply_expected=True,
            kind="status",
            correlation_id="corr-oversize",
        ),
    )

    assert adapter._should_react_malformed_bot_message(malformed) is True
    assert adapter._should_react_malformed_bot_message(valid_but_rejected) is False


def test_protocol_rejects_obsolete_handwritten_envelope_shapes():
    obsolete_with_to_from = "\n".join(
        [
            "<@777>",
            "BOT_MSG v1",
            "to: galt/default",
            "from: nj-statutes-pm",
            "kind: request",
            "reply_expected: true",
            "---",
            "body",
        ]
    )
    missing_correlation = "\n".join(
        [
            "<@777>",
            "BOT_MSG v1",
            "reply_expected: true",
            "kind: action_required",
            "---",
            "body",
        ]
    )
    missing_separator = "\n".join(
        [
            "<@777>",
            "BOT_MSG v1",
            "reply_expected: true",
            "kind: action_required",
            "correlation_id: corr-1",
            "body",
        ]
    )
    invalid_kind = "\n".join(
        [
            "<@777>",
            "BOT_MSG v1",
            "reply_expected: true",
            "kind: request",
            "correlation_id: corr-1",
            "---",
            "body",
        ]
    )

    for content in (obsolete_with_to_from, missing_correlation, missing_separator, invalid_kind):
        assert parse_discord_bot_msg_v1(content, "777") is None


def test_send_bot_message_rejects_invalid_kind_before_send(monkeypatch, tmp_path):
    import tools.send_message_tool as smt
    from gateway.config import Platform

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_USERS", "777")
    monkeypatch.setenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", "1")

    class FakeConfig:
        platforms = {Platform.DISCORD: PlatformConfig(enabled=True, token="***")}

        def get_home_channel(self, platform):
            return None

    async def fake_send(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("invalid BOT_MSG kind reached Discord send path")

    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: FakeConfig())
    monkeypatch.setattr(smt, "_send_bot_message_to_discord", fake_send)
    result = json.loads(
        smt._handle_send_bot_message(
            {
                "target": "discord:555",
                "recipient_bot_id": "777",
                "kind": "request",
                "reply_expected": True,
                "body": "body",
                "correlation_id": "corr-1",
            }
        )
    )

    assert "error" in result
    assert "Invalid BOT_MSG v1 kind" in result["error"]


def test_messaging_toolset_hides_structured_bot_message_tools_at_baseline(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Messaging-platform sessions expose send_message without needing a live
    # gateway.pid under the temporary test HERMES_HOME. Legacy Discord bot
    # control-plane tools remain hidden unless explicitly enabled.
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.delenv("HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT", raising=False)
    import importlib
    import tools.send_message_tool as smt
    import model_tools

    importlib.reload(smt)
    model_tools._clear_tool_defs_cache()
    names = {t["function"]["name"] for t in model_tools.get_tool_definitions(enabled_toolsets=["messaging"], quiet_mode=True)}

    assert "send_message" in names
    assert "send_bot_message" not in names
    assert "send_bot_approval_decision" not in names
