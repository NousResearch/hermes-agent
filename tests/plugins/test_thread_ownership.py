"""Tests for the thread_ownership plugin (local last-mention ownership).

We exercise the `pre_gateway_dispatch` hook directly. The plugin reads fields
off a MessageEvent-shaped object and returns:
  - None / {"action": "allow"}            → gateway dispatches normally
  - {"action": "skip", "reason": "..."}   → drop the message (no turn, no output)

Ownership is computed locally from each message's <@id> mention list, so these
tests need no network — only the bot's own user_id (seeded in the fixture).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from plugins import thread_ownership

# Realistic Slack user ids: uppercase alphanumeric, no underscore (matches the
# plugin's <@id> regex, same shape as real ids like U0AQ1AA1HDF).
MY_ID = "U0GARRY1"
OTHER_ID = "U0ZEPHYR1"


def _slack_event(
    *,
    text: str,
    channel_id: str = "Cabc",
    thread_ts: str | None = "T1",
    reply_to_message_id: str | None = None,
    raw_text: str | None = None,
) -> SimpleNamespace:
    """Build a duck-typed MessageEvent matching hermes's shape.

    Real Slack passes the ORIGINAL message text in ``raw_message["text"]``, while
    ``event.text`` may have this bot's own mention stripped. ``raw_text`` lets a
    test set the original separately; it defaults to ``text`` (un-stripped).
    """
    return SimpleNamespace(
        text=text,
        raw_message={"text": raw_text if raw_text is not None else text},
        reply_to_message_id=reply_to_message_id,
        source=SimpleNamespace(platform="slack", chat_id=channel_id, thread_id=thread_ts),
    )


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Reset per-thread ownership + identity caches between tests."""
    thread_ownership._owns.clear()
    thread_ownership._bot_user_id_cache = None
    for key in ("RAILWAY_SERVICE_NAME", "HOSTNAME", "SLACK_BOT_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RAILWAY_SERVICE_NAME", "garry")
    thread_ownership._bot_user_id_cache = MY_ID  # seed to skip auth.test
    yield
    thread_ownership._owns.clear()
    thread_ownership._bot_user_id_cache = None


# ── identity resolution ──────────────────────────────────────────────────────


def test_agent_id_comes_from_railway_service_name(monkeypatch):
    monkeypatch.setenv("RAILWAY_SERVICE_NAME", "alice")
    assert thread_ownership._resolve_agent_id() == "alice"


def test_agent_id_falls_back_to_hostname(monkeypatch):
    monkeypatch.delenv("RAILWAY_SERVICE_NAME", raising=False)
    monkeypatch.setenv("HOSTNAME", "container-abc")
    assert thread_ownership._resolve_agent_id() == "container-abc"


def test_bot_user_id_fetched_from_slack_auth_test(monkeypatch):
    thread_ownership._bot_user_id_cache = None
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"ok": True, "user_id": "U_FROM_API"}
    with patch.object(thread_ownership.httpx, "post", return_value=fake_resp) as post:
        assert thread_ownership._resolve_bot_user_id() == "U_FROM_API"
        assert thread_ownership._resolve_bot_user_id() == "U_FROM_API"  # cached
    post.assert_called_once()


def test_bot_user_id_failure_is_cached(monkeypatch):
    thread_ownership._bot_user_id_cache = None
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"ok": False, "error": "invalid_auth"}
    with patch.object(thread_ownership.httpx, "post", return_value=fake_resp) as post:
        assert thread_ownership._resolve_bot_user_id() is None
        assert thread_ownership._resolve_bot_user_id() is None
    post.assert_called_once()


# ── passthrough / safe no-op cases ───────────────────────────────────────────


def test_non_slack_event_passthrough():
    e = SimpleNamespace(
        text="hi", reply_to_message_id=None,
        source=SimpleNamespace(platform="telegram", chat_id="123", thread_id="T1"),
    )
    assert thread_ownership._on_pre_gateway_dispatch(event=e) is None


def test_top_level_message_passthrough():
    # No thread_ts → root @-gating handles it; we don't intervene even if it
    # @-mentions another bot.
    e = _slack_event(text=f"<@{OTHER_ID}> hi", thread_ts=None)
    assert thread_ownership._on_pre_gateway_dispatch(event=e) is None


def test_dm_always_allows():
    # DM channel (id starts "D") is 1:1 — never gate, even a mention of another id.
    e = _slack_event(text=f"<@{OTHER_ID}> hi", channel_id="D999", thread_ts="T1")
    assert thread_ownership._on_pre_gateway_dispatch(event=e) is None


def test_unresolved_bot_id_fails_open():
    thread_ownership._bot_user_id_cache = thread_ownership._BOT_USER_ID_SENTINEL
    e = _slack_event(text=f"<@{OTHER_ID}> hi")  # would otherwise skip
    assert thread_ownership._on_pre_gateway_dispatch(event=e) is None


def test_unconfigured_agent_id_is_noop(monkeypatch):
    monkeypatch.delenv("RAILWAY_SERVICE_NAME", raising=False)
    monkeypatch.delenv("HOSTNAME", raising=False)
    e = _slack_event(text=f"<@{OTHER_ID}> hi")
    assert thread_ownership._on_pre_gateway_dispatch(event=e) is None


# ── ownership logic ──────────────────────────────────────────────────────────


def test_mention_me_allows_and_takes_ownership():
    e = _slack_event(text=f"hey <@{MY_ID}> what's up")
    assert thread_ownership._on_pre_gateway_dispatch(event=e) is None
    assert thread_ownership._get_owns("Cabc:T1") is True


def test_stripped_self_mention_still_detected_via_raw_message():
    """Regression: hermes strips this bot's own <@id> from event.text on a direct
    @-mention, so we must read the original from raw_message['text']. Without this
    the bot ignores a direct @-mention (the prod bug we hit)."""
    e = _slack_event(text="在吗", raw_text=f"<@{MY_ID}> 在吗")  # event.text already stripped
    assert thread_ownership._on_pre_gateway_dispatch(event=e) is None
    assert thread_ownership._get_owns("Cabc:T1") is True


def test_mention_other_bot_skips_and_yields_ownership():
    thread_ownership._set_owns("Cabc:T1", True)  # I used to own it
    e = _slack_event(text=f"<@{OTHER_ID}> hi")
    out = thread_ownership._on_pre_gateway_dispatch(event=e)
    assert isinstance(out, dict) and out["action"] == "skip"
    assert thread_ownership._get_owns("Cabc:T1") is False  # ownership moved away


def test_non_mention_owner_allows():
    thread_ownership._set_owns("Cabc:T1", True)
    assert thread_ownership._on_pre_gateway_dispatch(event=_slack_event(text="yes please")) is None


def test_non_mention_non_owner_skips():
    thread_ownership._set_owns("Cabc:T1", False)
    out = thread_ownership._on_pre_gateway_dispatch(event=_slack_event(text="yes please"))
    assert isinstance(out, dict) and out["action"] == "skip"


def test_non_mention_unknown_thread_skips():
    # No prior ownership recorded → not the owner → stay silent.
    out = thread_ownership._on_pre_gateway_dispatch(event=_slack_event(text="lol"))
    assert isinstance(out, dict) and out["action"] == "skip"


def test_last_mention_wins_when_multiple_bots_mentioned():
    # @OTHER then @ME → I'm last → I own AND reply.
    e = _slack_event(text=f"<@{OTHER_ID}> <@{MY_ID}> thoughts?")
    assert thread_ownership._on_pre_gateway_dispatch(event=e) is None
    assert thread_ownership._get_owns("Cabc:T1") is True


def test_mentioned_but_not_last_replies_but_does_not_own():
    # @ME then @OTHER → I'm mentioned (reply to THIS msg) but OTHER is last (owner).
    e = _slack_event(text=f"<@{MY_ID}> <@{OTHER_ID}> over to you")
    assert thread_ownership._on_pre_gateway_dispatch(event=e) is None  # I'm mentioned → reply
    assert thread_ownership._get_owns("Cabc:T1") is False  # but I don't own follow-ups
    # subsequent plain follow-up → I stay silent
    out = thread_ownership._on_pre_gateway_dispatch(event=_slack_event(text="cool"))
    assert isinstance(out, dict) and out["action"] == "skip"


def test_full_transfer_then_followup_is_silent():
    # I own → someone @s another bot → I yield → plain follow-up → I stay silent.
    thread_ownership._set_owns("Cabc:T1", True)
    thread_ownership._on_pre_gateway_dispatch(event=_slack_event(text=f"<@{OTHER_ID}> take this"))
    out = thread_ownership._on_pre_gateway_dispatch(event=_slack_event(text="thanks"))
    assert isinstance(out, dict) and out["action"] == "skip"


def test_ownership_is_per_thread():
    thread_ownership._set_owns("Cabc:T1", True)
    # A different thread I don't own → plain follow-up → skip.
    out = thread_ownership._on_pre_gateway_dispatch(event=_slack_event(text="hi", thread_ts="T2"))
    assert isinstance(out, dict) and out["action"] == "skip"


def test_human_only_mention_is_treated_as_addressed_elsewhere():
    # A mention of a human (not me) → I yield, same as any non-self mention.
    thread_ownership._set_owns("Cabc:T1", True)
    out = thread_ownership._on_pre_gateway_dispatch(event=_slack_event(text="<@U0HUMAN1> thanks"))
    assert isinstance(out, dict) and out["action"] == "skip"


def test_register_wires_hook_callback():
    ctx = MagicMock()
    thread_ownership.register(ctx)
    ctx.register_hook.assert_called_once()
    args, _ = ctx.register_hook.call_args
    assert args[0] == "pre_gateway_dispatch"
    assert callable(args[1])
