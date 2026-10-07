"""guardrail_rejection_reset: a router guardrail-rejection reply auto-rotates the session so the
user's next message starts fresh without typing /new. Off by default; opt-in via
config.yaml `guardrail_rejection_reset: {enabled, pattern}`."""

import asyncio
from types import SimpleNamespace

import gateway.run as gr
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource

REJECTION_REPLY = (
    "This channel is scoped to EAT Cafe operations and F&B topics. Please ask EAT analytics, "
    "preorder, menu, inventory, or F&B questions. (policy compliance probability 0.02 "
    "(threshold 0.5))\n\n[smart-router/v2.26.2]"
)
NORMAL_REPLY = "Last week's Cafe sales were 12,450 tokens ($124.50).\n\n[smart-router/L1|v2.26.2]"
PATTERN = r"(?s)This channel is scoped to EAT Cafe.*\[smart-router/v"

# _hmwa_guardrail_rejection_reset lives on GatewayRunner's turn mixin and only touches a handful
# of self attributes; async_session_store is a read-only property on the real class, so the method
# is invoked unbound against a plain stub.
_METHOD = GatewayRunner._hmwa_guardrail_rejection_reset


def _run(response, cfg, new_sid="fresh", reset_returns_entry=True):
    calls = {}

    async def reset_session(key):
        calls["reset_session"] = key
        return SimpleNamespace(session_id=new_sid) if reset_returns_entry else None

    def sync_binding(source, entry, reason):
        calls["sync_binding"] = reason

    stub = SimpleNamespace(
        async_session_store=SimpleNamespace(reset_session=reset_session),
        _evict_cached_agent=lambda key: calls.setdefault("evict", key),
        _clear_conversation_scope=lambda key, reason: calls.setdefault("clear_scope", (key, reason)),
        _sync_telegram_topic_binding=sync_binding,
    )
    saved = gr._load_gateway_config
    gr._load_gateway_config = lambda: {"guardrail_rejection_reset": cfg}
    try:
        entry = SimpleNamespace(session_id="old")
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="c", user_id="u")
        out = asyncio.run(_METHOD(stub, response, entry, "key", source))
        return out, calls
    finally:
        gr._load_gateway_config = saved


def test_rejection_reply_rotates_session():
    out, calls = _run(REJECTION_REPLY, {"enabled": True, "pattern": PATTERN})
    assert calls.get("reset_session") == "key"
    assert calls.get("evict") == "key"
    assert calls["clear_scope"][1] == "guardrail_rejection_reset"
    assert calls.get("sync_binding") == "guardrail-rejection-reset"
    assert out.session_id == "fresh"


def test_normal_reply_no_reset():
    out, calls = _run(NORMAL_REPLY, {"enabled": True, "pattern": PATTERN})
    assert "reset_session" not in calls
    assert out.session_id == "old"


def test_disabled_no_reset():
    out, calls = _run(REJECTION_REPLY, {"enabled": False, "pattern": PATTERN})
    assert "reset_session" not in calls
    assert out.session_id == "old"


def test_missing_config_no_reset():
    out, calls = _run(REJECTION_REPLY, {})
    assert "reset_session" not in calls
    assert out.session_id == "old"


def test_reset_session_returning_none_keeps_old_entry():
    out, calls = _run(REJECTION_REPLY, {"enabled": True, "pattern": PATTERN}, reset_returns_entry=False)
    assert calls.get("reset_session") == "key"
    assert out.session_id == "old"
