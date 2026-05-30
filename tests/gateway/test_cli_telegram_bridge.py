from datetime import datetime, timedelta, timezone

from gateway.bridge import BridgeStateStore, BridgeVerdict


def _now():
    return datetime(2026, 5, 29, 12, 0, tzinfo=timezone.utc)


def test_update_dedupe_is_durable(tmp_path):
    db = tmp_path / "bridge.sqlite"
    store = BridgeStateStore(db, now_fn=_now)

    assert store.accept_update(platform="telegram", update_id="100") is True
    assert store.accept_update(platform="telegram", update_id="100") is False

    reopened = BridgeStateStore(db, now_fn=_now)
    assert reopened.accept_update(platform="telegram", update_id="100") is False
    assert reopened.accept_update(platform="telegram", update_id="101") is True


def test_reply_input_requires_live_registered_outbound_same_user_and_chat(tmp_path):
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=_now)
    store.create_binding(
        bridge_id="b1",
        hermes_session_id="cli-session",
        telegram_chat_id="48264503",
        telegram_user_id="48264503",
        telegram_thread_id=None,
    )
    store.record_outbound_message(
        bridge_id="b1",
        chat_id="48264503",
        thread_id=None,
        message_id="174",
        purpose="input_prompt",
        input_expected=True,
        ttl_seconds=600,
    )

    wrong_user = store.validate_reply_input(
        chat_id="48264503",
        thread_id=None,
        user_id="999",
        reply_to_message_id="174",
        inbound_message_id="176",
    )
    assert wrong_user.verdict is BridgeVerdict.REJECT
    assert "user" in wrong_user.reason

    ok = store.validate_reply_input(
        chat_id="48264503",
        thread_id=None,
        user_id="48264503",
        reply_to_message_id="174",
        inbound_message_id="175",
    )
    assert ok.verdict is BridgeVerdict.ACCEPT
    assert ok.bridge_id == "b1"
    assert ok.hermes_session_id == "cli-session"

    missing = store.validate_reply_input(
        chat_id="48264503",
        thread_id=None,
        user_id="48264503",
        reply_to_message_id="does-not-exist",
        inbound_message_id="177",
    )
    assert missing.verdict is BridgeVerdict.REJECT


def test_reply_input_rejects_stale_or_non_input_messages(tmp_path):
    now = _now()
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=lambda: now)
    store.create_binding(
        bridge_id="b1",
        hermes_session_id="cli-session",
        telegram_chat_id="48264503",
        telegram_user_id="48264503",
    )
    store.record_outbound_message(
        bridge_id="b1",
        chat_id="48264503",
        thread_id=None,
        message_id="normal-output",
        purpose="mirror_output",
        input_expected=False,
        ttl_seconds=600,
    )
    store.record_outbound_message(
        bridge_id="b1",
        chat_id="48264503",
        thread_id=None,
        message_id="expired",
        purpose="input_prompt",
        input_expected=True,
        ttl_seconds=1,
    )

    assert store.validate_reply_input(
        chat_id="48264503",
        thread_id=None,
        user_id="48264503",
        reply_to_message_id="normal-output",
        inbound_message_id="200",
    ).verdict is BridgeVerdict.REJECT

    later = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=lambda: now + timedelta(seconds=2))
    assert later.validate_reply_input(
        chat_id="48264503",
        thread_id=None,
        user_id="48264503",
        reply_to_message_id="expired",
        inbound_message_id="201",
    ).verdict is BridgeVerdict.REJECT


def test_approval_nonce_is_single_use_and_bound_to_session_user_chat_and_args(tmp_path):
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=_now)
    store.create_binding(
        bridge_id="b1",
        hermes_session_id="cli-session",
        telegram_chat_id="48264503",
        telegram_user_id="48264503",
    )
    approval = store.create_approval(
        bridge_id="b1",
        turn_id="turn-1",
        tool_call_id="tool-1",
        tool_name="terminal",
        tool_args_hash="sha256:abc",
        ttl_seconds=300,
        nonce="nonce-123",
    )

    wrong = store.consume_approval(
        nonce=approval.nonce,
        chat_id="48264503",
        user_id="48264503",
        hermes_session_id="other-session",
        tool_args_hash="sha256:abc",
    )
    assert wrong.verdict is BridgeVerdict.REJECT

    wrong_turn = store.consume_approval(
        nonce=approval.nonce,
        chat_id="48264503",
        user_id="48264503",
        hermes_session_id="cli-session",
        turn_id="other-turn",
        tool_call_id="tool-1",
        tool_args_hash="sha256:abc",
    )
    assert wrong_turn.verdict is BridgeVerdict.REJECT
    assert "turn" in wrong_turn.reason

    wrong_tool_call = store.consume_approval(
        nonce=approval.nonce,
        chat_id="48264503",
        user_id="48264503",
        hermes_session_id="cli-session",
        turn_id="turn-1",
        tool_call_id="other-tool",
        tool_args_hash="sha256:abc",
    )
    assert wrong_tool_call.verdict is BridgeVerdict.REJECT
    assert "tool call" in wrong_tool_call.reason

    missing_exact_binding = store.consume_approval(
        nonce=approval.nonce,
        chat_id="48264503",
        user_id="48264503",
        hermes_session_id="cli-session",
        tool_args_hash="sha256:abc",
        require_user_approval=True,
    )
    assert missing_exact_binding.verdict is BridgeVerdict.REJECT
    assert "turn" in missing_exact_binding.reason

    ok = store.consume_approval(
        nonce=approval.nonce,
        chat_id="48264503",
        user_id="48264503",
        hermes_session_id="cli-session",
        turn_id="turn-1",
        tool_call_id="tool-1",
        tool_args_hash="sha256:abc",
    )
    assert ok.verdict is BridgeVerdict.ACCEPT

    reused = store.consume_approval(
        nonce=approval.nonce,
        chat_id="48264503",
        user_id="48264503",
        hermes_session_id="cli-session",
        turn_id="turn-1",
        tool_call_id="tool-1",
        tool_args_hash="sha256:abc",
    )
    assert reused.verdict is BridgeVerdict.REJECT
    assert "consumed" in reused.reason


def test_local_opt_in_binding_token_is_single_use(tmp_path):
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=_now)
    token = store.create_binding_token(
        bridge_id="b1",
        hermes_session_id="cli-session",
        ttl_seconds=300,
        token="bind-token",
    )

    first = store.consume_binding_token(
        token=token.token,
        telegram_chat_id="48264503",
        telegram_user_id="48264503",
        telegram_thread_id=None,
    )
    assert first.verdict is BridgeVerdict.ACCEPT
    assert first.bridge_id == "b1"
    assert first.hermes_session_id == "cli-session"

    second = store.consume_binding_token(
        token=token.token,
        telegram_chat_id="48264503",
        telegram_user_id="48264503",
        telegram_thread_id=None,
    )
    assert second.verdict is BridgeVerdict.REJECT
    assert "consumed" in second.reason

    # The token created an active binding that can now validate a reply anchor.
    store.record_outbound_message(
        bridge_id="b1",
        chat_id="48264503",
        thread_id=None,
        message_id="174",
        purpose="input_prompt",
        input_expected=True,
        ttl_seconds=600,
    )
    reply = store.validate_reply_input(
        chat_id="48264503",
        thread_id=None,
        user_id="48264503",
        reply_to_message_id="174",
        inbound_message_id="175",
    )
    assert reply.verdict is BridgeVerdict.ACCEPT


def test_local_opt_in_binding_token_can_be_limited_to_expected_telegram_user(tmp_path):
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=_now)
    token = store.create_binding_token(
        bridge_id="b1",
        hermes_session_id="cli-session",
        ttl_seconds=300,
        token="bind-token",
        telegram_user_id="48264503",
        telegram_chat_id="48264503",
    )

    wrong_user = store.consume_binding_token(
        token=token.token,
        telegram_chat_id="48264503",
        telegram_user_id="999",
        telegram_thread_id=None,
    )
    assert wrong_user.verdict is BridgeVerdict.REJECT
    assert "user" in wrong_user.reason

    wrong_chat = store.consume_binding_token(
        token=token.token,
        telegram_chat_id="999",
        telegram_user_id="48264503",
        telegram_thread_id=None,
    )
    assert wrong_chat.verdict is BridgeVerdict.REJECT
    assert "chat" in wrong_chat.reason

    ok = store.consume_binding_token(
        token=token.token,
        telegram_chat_id="48264503",
        telegram_user_id="48264503",
        telegram_thread_id=None,
    )
    assert ok.verdict is BridgeVerdict.ACCEPT


def test_expired_local_opt_in_binding_token_is_rejected(tmp_path):
    now = _now()
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=lambda: now)
    token = store.create_binding_token(
        bridge_id="b1",
        hermes_session_id="cli-session",
        ttl_seconds=1,
        token="bind-token",
    )

    later = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=lambda: now + timedelta(seconds=2))
    expired = later.consume_binding_token(
        token=token.token,
        telegram_chat_id="48264503",
        telegram_user_id="48264503",
        telegram_thread_id=None,
    )
    assert expired.verdict is BridgeVerdict.REJECT
    assert "expired" in expired.reason


def test_paused_binding_and_kill_switch_fail_closed(tmp_path):
    disabled = tmp_path / "bridge.disabled"
    store = BridgeStateStore(tmp_path / "bridge.sqlite", now_fn=_now, kill_switch_path=disabled)
    store.create_binding(
        bridge_id="b1",
        hermes_session_id="cli-session",
        telegram_chat_id="48264503",
        telegram_user_id="48264503",
    )
    store.record_outbound_message(
        bridge_id="b1",
        chat_id="48264503",
        thread_id=None,
        message_id="174",
        purpose="input_prompt",
        input_expected=True,
        ttl_seconds=600,
    )

    store.pause_binding("b1", reason="operator pause")
    paused = store.validate_reply_input(
        chat_id="48264503",
        thread_id=None,
        user_id="48264503",
        reply_to_message_id="174",
        inbound_message_id="175",
    )
    assert paused.verdict is BridgeVerdict.REJECT
    assert "paused" in paused.reason

    store.resume_binding("b1")
    disabled.write_text("disabled\n")
    killed = store.validate_reply_input(
        chat_id="48264503",
        thread_id=None,
        user_id="48264503",
        reply_to_message_id="174",
        inbound_message_id="176",
    )
    assert killed.verdict is BridgeVerdict.REJECT
    assert "kill switch" in killed.reason
