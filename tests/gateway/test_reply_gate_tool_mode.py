"""Reply-gate tool-mode delivery inversion (Phase 1).

Covers the tool-gated delivery mechanism, the fallback ratchet, the
ReplyPolicy resolver, target="current" addressing, and the group-mode
interruption note. The feature is config-flagged OFF by default
(reply_gate_mode="prompt"); these tests assert that opting in flips delivery
only for free-response GROUP turns and leaves DMs / mention-gated groups /
prompt mode byte-identical.
"""

import json
import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin
from gateway.reply_policy import ReplyPolicy, resolve_reply_policy
from gateway.run import _build_interruption_system_note
from gateway.session import SessionEntry, SessionSource
from tools import approval
from tools.send_message_tool import _handle_send


# --------------------------------------------------------------------------- #
# ReplyPolicy resolver (T-POL)
# --------------------------------------------------------------------------- #


class _FakeWhatsApp(WhatsAppBehaviorMixin):
    """Minimal mixin host driven by a real PlatformConfig (config-propagation)."""

    def __init__(self, extra):
        self.config = PlatformConfig(enabled=True, extra=extra)


def _wa_reply_policy(extra, data):
    return _FakeWhatsApp(extra)._resolve_whatsapp_reply_policy(data)


def test_resolver_pure_matrix():
    assert resolve_reply_policy(
        is_group=False, in_free_response_set=False, require_mention=True
    ) == ReplyPolicy.DM
    assert resolve_reply_policy(
        is_group=True, in_free_response_set=True, require_mention=True
    ) == ReplyPolicy.FREE_RESPONSE
    assert resolve_reply_policy(
        is_group=True, in_free_response_set=False, require_mention=False
    ) == ReplyPolicy.FREE_RESPONSE
    assert resolve_reply_policy(
        is_group=True, in_free_response_set=False, require_mention=True
    ) == ReplyPolicy.MENTION_GATED


def test_whatsapp_matrix_via_real_config():
    # T-POL-1: free_response_chats member (even with require_mention on) →
    # FREE_RESPONSE. Driven through the real mixin config parsers, not a mock.
    assert _wa_reply_policy(
        {"require_mention": True, "free_response_chats": "123@g.us,999@g.us"},
        {"isGroup": True, "chatId": "123@g.us"},
    ) == ReplyPolicy.FREE_RESPONSE
    # require_mention false → FREE_RESPONSE
    assert _wa_reply_policy(
        {"require_mention": False},
        {"isGroup": True, "chatId": "555@g.us"},
    ) == ReplyPolicy.FREE_RESPONSE
    # require_mention true, not in the set → MENTION_GATED
    assert _wa_reply_policy(
        {"require_mention": True, "free_response_chats": ["123@g.us"]},
        {"isGroup": True, "chatId": "555@g.us"},
    ) == ReplyPolicy.MENTION_GATED
    # DM → DM
    assert _wa_reply_policy(
        {"require_mention": True},
        {"isGroup": False, "chatId": "111@s.whatsapp.net"},
    ) == ReplyPolicy.DM


def test_unstamped_event_defaults_to_dm():
    # T-POL-2: a freshly-built source (platform not yet wired) defaults DM, so
    # the inversion never applies — partial platform rollout is safe.
    src = SessionSource(platform=Platform.TELEGRAM, chat_id="-1", chat_type="group")
    assert src.reply_policy == ReplyPolicy.DM


# --------------------------------------------------------------------------- #
# Config flag (T-GATE-5 groundwork)
# --------------------------------------------------------------------------- #


def test_config_defaults_and_coercion():
    c = GatewayConfig()
    assert c.reply_gate_mode == "prompt"
    assert c.reply_gate_tool_fallback is True
    # garbage → prompt (never raise on bad config)
    assert GatewayConfig.from_dict({"reply_gate_mode": "bogus"}).reply_gate_mode == "prompt"
    assert GatewayConfig.from_dict({"reply_gate_mode": "TOOL"}).reply_gate_mode == "tool"
    rt = GatewayConfig.from_dict(
        {"reply_gate_mode": "tool", "reply_gate_tool_fallback": False}
    )
    assert (rt.reply_gate_mode, rt.reply_gate_tool_fallback) == ("tool", False)
    # to_dict round-trip preserves values
    assert GatewayConfig.from_dict(rt.to_dict()).reply_gate_mode == "tool"


# --------------------------------------------------------------------------- #
# target="current" resolution (T-GATE-6)
# --------------------------------------------------------------------------- #


def test_target_current_errors_outside_gateway_turn():
    for args in ({"target": "current", "message": "hi"}, {"message": "hi"}):
        r = json.loads(_handle_send(args))
        assert "error" in r
        assert "live gateway turn" in r["error"]


def test_target_current_resolves_registered_turn(monkeypatch):
    src = SessionSource(platform=Platform.WHATSAPP, chat_id="123@g.us", chat_type="group")
    captured = {}

    async def _fake_send_to_platform(platform, pconfig, chat_id, message, **kw):
        captured["platform"] = platform
        captured["chat_id"] = chat_id
        captured["message"] = message
        return {"success": True, "message_id": "m1"}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", _fake_send_to_platform)
    monkeypatch.setattr("model_tools._run_async", lambda coro: __import__("asyncio").get_event_loop().run_until_complete(coro))
    # A configured, enabled WhatsApp platform so _handle_send passes the pconfig gate.
    cfg = GatewayConfig()
    cfg.platforms[Platform.WHATSAPP] = PlatformConfig(enabled=True, token="x")
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: cfg)

    tok = approval.set_current_message_source(src)
    try:
        r = json.loads(_handle_send({"target": "current", "message": "hello there"}))
    finally:
        approval.reset_current_message_source(tok)

    assert r.get("success") is True
    assert captured["chat_id"] == "123@g.us"
    assert captured["message"] == "hello there"
    # Marker recorded so the post-turn block can dedup.
    assert src.reply_gate_tool_sends == 1


# --------------------------------------------------------------------------- #
# Interruption note group-mode variant (T-COLL-3 companion)
# --------------------------------------------------------------------------- #


def test_interruption_note_group_vs_dm():
    dm = _build_interruption_system_note("restart_timeout", has_new_message=False)
    grp = _build_interruption_system_note(
        "restart_timeout", has_new_message=False, group_mode=True
    )
    assert "Report to the user that the session was restored" in dm
    assert "do NOT narrate this restart/interruption into the group" in grp
    assert "Report to the user that the session was restored" not in grp
    # DM default output unchanged (byte-identical to pre-variant behavior).
    assert _build_interruption_system_note("restart_timeout", has_new_message=False) == dm


# --------------------------------------------------------------------------- #
# Post-turn delivery inversion (T-GATE / T-FALL) — E2E through the runner
# --------------------------------------------------------------------------- #

_SESSION_KEY = "agent:main:whatsapp:group:123@g.us:u1"


def _source(reply_policy=ReplyPolicy.FREE_RESPONSE, chat_type="group", tool_sends=0):
    src = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="123@g.us",
        chat_type=chat_type,
        user_id="u1",
    )
    src.reply_policy = reply_policy
    src.reply_gate_tool_sends = tool_sends
    return src


def _event(src):
    return MessageEvent(text="side chatter", source=src, message_id="msg-1")


def _runner(monkeypatch, tmp_path, config):
    runner = gateway_run.GatewayRunner(config)
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _s: True
    runner._set_session_env = lambda _c: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _s: None
    runner._cache_session_source = lambda _k, _s: None
    runner._is_session_run_current = lambda _k, _g: True
    runner._reply_anchor_for_event = lambda _e: None
    runner._get_guild_id = lambda _e: None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=_SESSION_KEY,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.WHATSAPP,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_a, **_kw: 100_000,
    )
    return runner


def _agent_result(text):
    return {
        "final_response": text,
        "messages": [
            {"role": "user", "content": "side chatter"},
            {"role": "assistant", "content": text},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
    }


async def _run(runner, src, text):
    runner._run_agent = AsyncMock(return_value=_agent_result(text))
    return await runner._handle_message_with_agent(_event(src), src, _SESSION_KEY, 1)


@pytest.mark.asyncio
async def test_tool_gated_silence_is_non_event(monkeypatch, tmp_path, caplog):
    # T-GATE-1: tool mode, free-response group, chatty tail, zero tool sends,
    # ratchet OFF → nothing delivered; transcript still persisted; telemetry.
    runner = _runner(
        monkeypatch, tmp_path,
        GatewayConfig(reply_gate_mode="tool", reply_gate_tool_fallback=False),
    )
    src = _source()
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        response = await _run(runner, src, "just some banter, nothing to do here")
    assert response == ""
    appended = [c.args[1] for c in runner.session_store.append_to_transcript.call_args_list]
    assert any(m.get("role") == "assistant" for m in appended)
    assert any("reply_gate: tool-mode" in r.message and "tool_sends=0" in r.message
               for r in caplog.records)


@pytest.mark.asyncio
async def test_tool_send_dedup_suppresses_tail(monkeypatch, tmp_path):
    # T-GATE-2 / T-FALL-3: a current-target tool send happened this turn → the
    # free-text tail is NOT also delivered (dedup), even with fallback ON.
    runner = _runner(
        monkeypatch, tmp_path,
        GatewayConfig(reply_gate_mode="tool", reply_gate_tool_fallback=True),
    )
    src = _source(tool_sends=1)
    response = await _run(runner, src, "and here is a chatty tail too")
    assert response == ""


@pytest.mark.asyncio
async def test_dm_byte_identical_regardless_of_mode(monkeypatch, tmp_path):
    # T-GATE-3: DM policy bypasses the inversion in both modes.
    text = "here is your answer"
    for mode in ("prompt", "tool"):
        runner = _runner(monkeypatch, tmp_path, GatewayConfig(reply_gate_mode=mode))
        src = _source(reply_policy=ReplyPolicy.DM, chat_type="dm")
        assert await _run(runner, src, text) == text


@pytest.mark.asyncio
async def test_mention_gated_byte_identical(monkeypatch, tmp_path):
    # T-GATE-4: mention-gated groups bypass the inversion in both modes.
    text = "here is your answer"
    for mode in ("prompt", "tool"):
        runner = _runner(monkeypatch, tmp_path, GatewayConfig(reply_gate_mode=mode))
        src = _source(reply_policy=ReplyPolicy.MENTION_GATED)
        assert await _run(runner, src, text) == text


@pytest.mark.asyncio
async def test_prompt_mode_branch_inert(monkeypatch, tmp_path):
    # T-GATE-5: default prompt mode → free-response group tail delivers as today.
    runner = _runner(monkeypatch, tmp_path, GatewayConfig())  # reply_gate_mode="prompt"
    src = _source()
    text = "a normal free-response reply"
    assert await _run(runner, src, text) == text


@pytest.mark.asyncio
async def test_fallback_ratchet_delivers_substantive_tail(monkeypatch, tmp_path, caplog):
    # T-FALL-1: ratchet ON, zero tool sends, substantive non-marker tail →
    # delivered (old behavior), telemetry marks fallback=1.
    runner = _runner(
        monkeypatch, tmp_path,
        GatewayConfig(reply_gate_mode="tool", reply_gate_tool_fallback=True),
    )
    src = _source()
    text = "the real answer the model forgot to send via tool"
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        response = await _run(runner, src, text)
    assert response == text
    assert any("reply_gate: tool-mode" in r.message and "fallback=1" in r.message
               for r in caplog.records)


@pytest.mark.asyncio
async def test_fallback_ratchet_still_suppresses_marker(monkeypatch, tmp_path):
    # T-FALL-2: ratchet ON, zero tool sends, tail IS a silence marker →
    # suppressed (the ratchet only rescues substantive answers).
    runner = _runner(
        monkeypatch, tmp_path,
        GatewayConfig(reply_gate_mode="tool", reply_gate_tool_fallback=True),
    )
    src = _source()
    assert await _run(runner, src, "NO_REPLY") == ""


@pytest.mark.asyncio
async def test_fallback_off_suppresses_substantive_tail(monkeypatch, tmp_path):
    # T-FALL-4: ratchet OFF → the same substantive tail delivers nothing.
    runner = _runner(
        monkeypatch, tmp_path,
        GatewayConfig(reply_gate_mode="tool", reply_gate_tool_fallback=False),
    )
    src = _source()
    assert await _run(runner, src, "the real answer") == ""


@pytest.mark.asyncio
async def test_tool_gated_empty_tail_not_normalized(monkeypatch, tmp_path):
    # T-GATE-7 (partial): a tool-gated turn with an empty tail is NOT turned
    # into a "no response" placeholder by the empty-response normalizer.
    runner = _runner(
        monkeypatch, tmp_path,
        GatewayConfig(reply_gate_mode="tool", reply_gate_tool_fallback=True),
    )
    src = _source()
    response = await _run(runner, src, "")
    assert response == ""
