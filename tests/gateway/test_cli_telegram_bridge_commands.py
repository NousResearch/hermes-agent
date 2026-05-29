from datetime import datetime, timezone

from gateway.bridge import BridgeStateStore, BridgeVerdict
from gateway.config import Platform
from gateway.session import SessionSource
from gateway.platforms.base import MessageEvent
from gateway.bridge_commands import (
    bridge_tool_args_hash,
    create_bridge_approval_prompt,
    handle_local_bridge_command,
    handle_gateway_bridge_command,
    maybe_apply_gateway_bridge_binding,
    register_bridge_reply_input_prompt,
)
from hermes_cli.commands import resolve_command


def _now():
    return datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc)


def _telegram_event(text: str, *, chat_id="48264503", user_id="48264503", thread_id=None):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            chat_type="dm",
            user_id=user_id,
            user_name="Boss",
            thread_id=thread_id,
        ),
        message_id="200",
        platform_update_id=1000,
    )


def test_bridge_command_is_registered_for_cli_and_gateway():
    cmd = resolve_command("bridge")
    assert cmd is not None
    assert cmd.name == "bridge"
    assert not cmd.cli_only
    assert not cmd.gateway_only

    approve = resolve_command("bridge_approve")
    assert approve is not None
    assert approve.gateway_only
    assert approve.args_hint == "<nonce>"


def test_local_bridge_bind_mints_single_use_token_limited_to_expected_telegram_identity(tmp_path):
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=_now)

    out = handle_local_bridge_command(
        "/bridge bind telegram --chat 48264503 --user 48264503",
        session_id="cli-session",
        store=store,
    )

    assert "Bridge binding token" in out
    assert "/bridge_bind" in out
    token = out.split("/bridge_bind ", 1)[1].split()[0]

    wrong = store.consume_binding_token(
        token=token,
        telegram_chat_id="48264503",
        telegram_user_id="999",
    )
    assert wrong.verdict is BridgeVerdict.REJECT

    ok = store.consume_binding_token(
        token=token,
        telegram_chat_id="48264503",
        telegram_user_id="48264503",
    )
    assert ok.verdict is BridgeVerdict.ACCEPT
    assert ok.hermes_session_id == "cli-session"


def test_gateway_bridge_bind_consumes_token_and_status_reports_active(tmp_path):
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=_now)
    token = store.create_binding_token(
        bridge_id="bridge-cli-session",
        hermes_session_id="cli-session",
        ttl_seconds=300,
        token="bind-token",
        telegram_chat_id="48264503",
        telegram_user_id="48264503",
    )

    bind_out = handle_gateway_bridge_command(
        _telegram_event(f"/bridge_bind {token.token}"),
        store=store,
    )
    assert "Bridge linked" in bind_out
    assert "cli-session" in bind_out

    status_out = handle_gateway_bridge_command(
        _telegram_event("/bridge_status"),
        store=store,
    )
    assert "active" in status_out
    assert "cli-session" in status_out


def test_gateway_bridge_status_includes_binding_details(tmp_path):
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=_now)
    store.create_binding(
        bridge_id="bridge-cli-session",
        hermes_session_id="cli-session",
        telegram_chat_id="48264503",
        telegram_user_id="48264503",
        telegram_thread_id="42",
    )

    status_out = handle_gateway_bridge_command(
        _telegram_event("/bridge_status", thread_id="42"),
        store=store,
    )

    assert "bridge-cli-session" in status_out
    assert "cli-session" in status_out
    assert "chat=48264503" in status_out
    assert "user=48264503" in status_out
    assert "thread=42" in status_out
    assert "updated=" in status_out


def test_gateway_bridge_disconnect_removes_only_this_telegram_binding(tmp_path):
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=_now)
    store.create_binding(
        bridge_id="bridge-cli-session",
        hermes_session_id="cli-session",
        telegram_chat_id="48264503",
        telegram_user_id="48264503",
    )
    store.create_binding(
        bridge_id="bridge-other-session",
        hermes_session_id="other-session",
        telegram_chat_id="999",
        telegram_user_id="999",
    )

    out = handle_gateway_bridge_command(
        _telegram_event("/bridge_disconnect"),
        store=store,
    )

    assert "disconnected" in out.lower()
    assert "cli-session" in out
    assert store.binding_for_telegram(chat_id="48264503", user_id="48264503") is None
    assert store.binding_for_telegram(chat_id="999", user_id="999") is not None


def test_local_bridge_status_lists_bound_sessions_and_disconnects_by_session(tmp_path):
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=_now)
    store.create_binding(
        bridge_id="bridge-cli-session",
        hermes_session_id="cli-session",
        telegram_chat_id="48264503",
        telegram_user_id="48264503",
    )

    status_out = handle_local_bridge_command("/bridge status", session_id="cli-session", store=store)
    assert "bridge-cli-session" in status_out
    assert "chat=48264503" in status_out
    assert "active" in status_out

    disconnect_out = handle_local_bridge_command("/bridge disconnect", session_id="cli-session", store=store)
    assert "disconnected" in disconnect_out.lower()
    assert store.binding_for_telegram(chat_id="48264503", user_id="48264503") is None


def test_gateway_bridge_off_creates_kill_switch_and_blocks_status(tmp_path):
    kill_switch = tmp_path / "bridge.disabled"
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=_now, kill_switch_path=kill_switch)
    store.create_binding(
        bridge_id="b1",
        hermes_session_id="cli-session",
        telegram_chat_id="48264503",
        telegram_user_id="48264503",
    )

    off_out = handle_gateway_bridge_command(
        _telegram_event("/bridge_off"),
        store=store,
    )
    assert "disabled" in off_out.lower()
    assert kill_switch.exists()

    status_out = handle_gateway_bridge_command(
        _telegram_event("/bridge_status"),
        store=store,
    )
    assert "kill switch" in status_out.lower()


def test_gateway_bridge_commands_are_telegram_dm_only(tmp_path):
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=_now)
    event = MessageEvent(
        text="/bridge_status",
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="discord-chat",
            chat_type="dm",
            user_id="48264503",
        ),
    )

    out = handle_gateway_bridge_command(event, store=store)
    assert "Telegram DM only" in out


class _DummySessionEntry:
    def __init__(self, session_id):
        self.session_id = session_id


class _DummySessionStore:
    def __init__(self):
        self.created_sources = []
        self.switches = []
        self.current_session_id = "telegram-session"

    def get_or_create_session(self, source):
        self.created_sources.append(source)
        return _DummySessionEntry(self.current_session_id)

    def switch_session(self, session_key, target_session_id):
        self.switches.append((session_key, target_session_id))
        self.current_session_id = target_session_id
        return _DummySessionEntry(target_session_id)


def test_bound_telegram_dm_plain_text_switches_to_cli_session(tmp_path):
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=_now)
    store.create_binding(
        bridge_id="bridge-cli-session",
        hermes_session_id="cli-session",
        telegram_chat_id="48264503",
        telegram_user_id="48264503",
    )
    session_store = _DummySessionStore()
    evicted = []

    decision = maybe_apply_gateway_bridge_binding(
        _telegram_event("continue from my phone"),
        session_key="agent:main:telegram:dm:48264503",
        session_store=session_store,
        store=store,
        evict_cached_agent=evicted.append,
    )

    assert decision is not None
    assert decision.verdict is BridgeVerdict.ACCEPT
    assert session_store.switches == [("agent:main:telegram:dm:48264503", "cli-session")]
    assert evicted == ["agent:main:telegram:dm:48264503"]


def test_register_bridge_reply_input_prompt_records_reply_anchor_for_bound_telegram_dm(tmp_path):
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=_now)
    store.create_binding(
        bridge_id="bridge-cli-session",
        hermes_session_id="cli-session",
        telegram_chat_id="48264503",
        telegram_user_id="48264503",
    )

    prompt = register_bridge_reply_input_prompt(
        _telegram_event("phone is waiting"),
        sent_message_id="301",
        prompt_text="Reply to this message with the next instruction.",
        store=store,
        ttl_seconds=600,
    )

    assert prompt.decision.verdict is BridgeVerdict.ACCEPT
    assert prompt.bridge_id == "bridge-cli-session"
    assert prompt.hermes_session_id == "cli-session"
    assert prompt.message_id == "301"
    assert "Reply to this message" in prompt.text

    reply = store.validate_reply_input(
        chat_id="48264503",
        thread_id=None,
        user_id="48264503",
        reply_to_message_id="301",
        inbound_message_id="302",
    )
    assert reply.verdict is BridgeVerdict.ACCEPT
    assert reply.hermes_session_id == "cli-session"


def test_register_bridge_reply_input_prompt_rejects_unbound_or_wrong_platform_events(tmp_path):
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=_now)
    unbound = register_bridge_reply_input_prompt(
        _telegram_event("unbound"),
        sent_message_id="301",
        prompt_text="Reply here",
        store=store,
    )
    assert unbound.decision.verdict is BridgeVerdict.REJECT
    assert "binding" in unbound.decision.reason

    discord_event = MessageEvent(
        text="not telegram",
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="discord-chat",
            chat_type="dm",
            user_id="48264503",
        ),
    )
    wrong_platform = register_bridge_reply_input_prompt(
        discord_event,
        sent_message_id="301",
        prompt_text="Reply here",
        store=store,
    )
    assert wrong_platform.decision.verdict is BridgeVerdict.REJECT
    assert "Telegram DM" in wrong_platform.decision.reason


def test_create_bridge_approval_prompt_uses_single_use_nonce_bound_to_args(tmp_path):
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=_now)
    store.create_binding(
        bridge_id="bridge-cli-session",
        hermes_session_id="cli-session",
        telegram_chat_id="48264503",
        telegram_user_id="48264503",
    )

    prompt = create_bridge_approval_prompt(
        _telegram_event("approval lane"),
        turn_id="turn-1",
        tool_call_id="tool-1",
        tool_name="terminal",
        tool_args={"command": "rm -rf /tmp/demo", "timeout": 30},
        store=store,
        ttl_seconds=300,
    )

    assert prompt.decision.verdict is BridgeVerdict.ACCEPT
    assert prompt.approval is not None
    assert prompt.approval.nonce in prompt.text
    assert "/bridge_approve" in prompt.text
    assert prompt.tool_args_hash == bridge_tool_args_hash(
        "terminal", {"timeout": 30, "command": "rm -rf /tmp/demo"}
    )

    changed_args = store.consume_approval(
        nonce=prompt.approval.nonce,
        chat_id="48264503",
        user_id="48264503",
        hermes_session_id="cli-session",
        tool_args_hash=bridge_tool_args_hash("terminal", {"command": "echo safe", "timeout": 30}),
    )
    assert changed_args.verdict is BridgeVerdict.REJECT
    assert "changed" in changed_args.reason

    ok = store.consume_approval(
        nonce=prompt.approval.nonce,
        chat_id="48264503",
        user_id="48264503",
        hermes_session_id="cli-session",
        tool_args_hash=prompt.tool_args_hash,
    )
    assert ok.verdict is BridgeVerdict.ACCEPT


def test_bridge_does_not_remap_slash_commands_or_paused_bindings(tmp_path):
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=_now)
    store.create_binding(
        bridge_id="bridge-cli-session",
        hermes_session_id="cli-session",
        telegram_chat_id="48264503",
        telegram_user_id="48264503",
    )
    session_store = _DummySessionStore()

    slash_decision = maybe_apply_gateway_bridge_binding(
        _telegram_event("/model"),
        session_key="agent:main:telegram:dm:48264503",
        session_store=session_store,
        store=store,
    )
    assert slash_decision is None
    assert session_store.switches == []

    store.pause_binding("bridge-cli-session", reason="test pause")
    paused_decision = maybe_apply_gateway_bridge_binding(
        _telegram_event("this should not reach cli"),
        session_key="agent:main:telegram:dm:48264503",
        session_store=session_store,
        store=store,
    )
    assert paused_decision is not None
    assert paused_decision.verdict is BridgeVerdict.REJECT
    assert "paused" in paused_decision.reason
    assert session_store.switches == []
