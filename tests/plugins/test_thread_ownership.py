"""Tests for the thread_ownership plugin (local last-mention ownership).

We exercise the `pre_gateway_dispatch` hook directly. The plugin reads fields
off a MessageEvent-shaped object and returns:
  - None / {"action": "allow"}            → gateway dispatches normally
  - {"action": "skip", "reason": "..."}   → drop the message (no turn, no output)

Ownership is computed locally from each message's <@id> mention list. Identity
(this bot's Slack user_id) is read per-workspace from the live Slack adapter's
``_team_bot_user_ids`` map, reached via the ``gateway`` kwarg the hook receives —
so these tests seed a fake gateway/adapter instead of hitting the network.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins import thread_ownership

# Realistic Slack user ids: uppercase alphanumeric, no underscore (matches the
# plugin's <@id> regex, same shape as real ids like U0AQ1AA1HDF).
MY_ID = "U0GARRY1"
OTHER_ID = "U0ZEPHYR1"
TEAM_ID = "T0TEAM1"


def _gateway(*, team_bot_user_ids=None, channel_team=None) -> SimpleNamespace:
    """Fake GatewayRunner exposing a Slack adapter with a per-workspace id map.

    Mirrors the real shape the hook sees: ``gateway.adapters[Platform.SLACK]``
    carrying ``_team_bot_user_ids`` (team_id → bot_user_id) and ``_channel_team``
    (channel_id → team_id). Keyed by "slack" — the plugin matches the platform
    key case-insensitively, and ``str(Platform.SLACK)`` contains "slack".
    """
    adapter = SimpleNamespace(
        platform="slack",
        _team_bot_user_ids={TEAM_ID: MY_ID} if team_bot_user_ids is None else team_bot_user_ids,
        _channel_team=channel_team or {},
    )
    return SimpleNamespace(adapters={"slack": adapter})


def _slack_event(
    *,
    text: str,
    channel_id: str = "Cabc",
    thread_ts: str | None = "T1",
    reply_to_message_id: str | None = None,
    raw_text: str | None = None,
    team_id: str = TEAM_ID,
) -> SimpleNamespace:
    """Build a duck-typed MessageEvent matching hermes's shape.

    Real Slack passes the ORIGINAL message text in ``raw_message["text"]`` and
    the workspace in ``raw_message["team"]``, while ``event.text`` may have this
    bot's own mention stripped. ``raw_text`` lets a test set the original
    separately; it defaults to ``text`` (un-stripped).
    """
    return SimpleNamespace(
        text=text,
        raw_message={"text": raw_text if raw_text is not None else text, "team": team_id},
        reply_to_message_id=reply_to_message_id,
        source=SimpleNamespace(platform="slack", chat_id=channel_id, thread_id=thread_ts),
    )


def _dispatch(event, gateway=None):
    """Invoke the hook with a default single-workspace gateway unless overridden."""
    return thread_ownership._on_pre_gateway_dispatch(
        event=event, gateway=_gateway() if gateway is None else gateway
    )


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset per-thread ownership between tests."""
    thread_ownership._owns.clear()
    yield
    thread_ownership._owns.clear()


# ── identity resolution (per-workspace, from the live adapter) ───────────────


def test_identity_resolved_per_workspace_from_adapter():
    gw = _gateway(team_bot_user_ids={TEAM_ID: MY_ID})
    assert thread_ownership._resolve_my_id(gw, TEAM_ID, "Cabc") == MY_ID


def test_identity_multi_workspace_picks_event_workspace():
    # Two workspaces; the bot has a distinct user id in each. The event's own
    # workspace must select its id — the bug the review flagged.
    other_team = "T0OTHER1"
    gw = _gateway(team_bot_user_ids={TEAM_ID: MY_ID, other_team: "U0ELSEWHERE"})
    assert thread_ownership._resolve_my_id(gw, TEAM_ID, "Cabc") == MY_ID
    assert thread_ownership._resolve_my_id(gw, other_team, "Cabc") == "U0ELSEWHERE"


def test_identity_multi_workspace_unknown_team_fails_open():
    # Unknown workspace must NOT fall back to another workspace's id.
    gw = _gateway(team_bot_user_ids={TEAM_ID: MY_ID, "T0OTHER1": "U0ELSEWHERE"})
    assert thread_ownership._resolve_my_id(gw, "T0UNKNOWN", "Cabc") is None


def test_identity_single_workspace_tolerates_team_key_mismatch():
    # One workspace → the sole id is unambiguous even if the event's team key
    # differs (e.g. enterprise-grid payload shapes).
    gw = _gateway(team_bot_user_ids={TEAM_ID: MY_ID})
    assert thread_ownership._resolve_my_id(gw, "T0DIFFERENT", "Cabc") == MY_ID


def test_identity_falls_back_to_channel_team_when_event_omits_team():
    gw = _gateway(
        team_bot_user_ids={TEAM_ID: MY_ID, "T0OTHER1": "U0X"},
        channel_team={"Cabc": TEAM_ID},
    )
    assert thread_ownership._resolve_my_id(gw, "", "Cabc") == MY_ID


def test_identity_no_gateway_or_adapter_returns_none():
    assert thread_ownership._resolve_my_id(None, TEAM_ID, "Cabc") is None
    assert thread_ownership._resolve_my_id(SimpleNamespace(adapters={}), TEAM_ID, "Cabc") is None


# ── passthrough / safe no-op cases ───────────────────────────────────────────


def test_non_slack_event_passthrough():
    e = SimpleNamespace(
        text="hi", reply_to_message_id=None,
        source=SimpleNamespace(platform="telegram", chat_id="123", thread_id="T1"),
    )
    assert _dispatch(e) is None


def test_top_level_message_passthrough():
    # No thread_ts → root @-gating handles it; we don't intervene even if it
    # @-mentions another bot.
    e = _slack_event(text=f"<@{OTHER_ID}> hi", thread_ts=None)
    assert _dispatch(e) is None


def test_dm_always_allows():
    # DM channel (id starts "D") is 1:1 — never gate, even a mention of another id.
    e = _slack_event(text=f"<@{OTHER_ID}> hi", channel_id="D999", thread_ts="T1")
    assert _dispatch(e) is None


def test_unresolved_bot_id_fails_open():
    # Adapter can't resolve my id for this event's workspace → fail open.
    gw = _gateway(team_bot_user_ids={"T0OTHER1": "U0X", "T0OTHER2": "U0Y"})
    e = _slack_event(text=f"<@{OTHER_ID}> hi", team_id="T0UNKNOWN")  # would otherwise skip
    assert _dispatch(e, gateway=gw) is None


def test_missing_slack_adapter_fails_open():
    # No Slack adapter reachable (e.g. torn down) → never silence.
    e = _slack_event(text=f"<@{OTHER_ID}> hi")  # would otherwise skip
    assert _dispatch(e, gateway=SimpleNamespace(adapters={})) is None


# ── ownership logic ──────────────────────────────────────────────────────────


def test_mention_me_allows_and_takes_ownership():
    e = _slack_event(text=f"hey <@{MY_ID}> what's up")
    assert _dispatch(e) is None
    assert thread_ownership._get_owns("Cabc:T1") is True


def test_stripped_self_mention_still_detected_via_raw_message():
    """Regression: hermes strips this bot's own <@id> from event.text on a direct
    @-mention, so we must read the original from raw_message['text']. Without this
    the bot ignores a direct @-mention (the prod bug we hit)."""
    e = _slack_event(text="在吗", raw_text=f"<@{MY_ID}> 在吗")  # event.text already stripped
    assert _dispatch(e) is None
    assert thread_ownership._get_owns("Cabc:T1") is True


def test_mention_other_bot_skips_and_yields_ownership():
    thread_ownership._set_owns("Cabc:T1", True)  # I used to own it
    e = _slack_event(text=f"<@{OTHER_ID}> hi")
    out = _dispatch(e)
    assert isinstance(out, dict) and out["action"] == "skip"
    assert thread_ownership._get_owns("Cabc:T1") is False  # ownership moved away


def test_non_mention_owner_allows():
    thread_ownership._set_owns("Cabc:T1", True)
    assert _dispatch(_slack_event(text="yes please")) is None


def test_non_mention_non_owner_skips():
    thread_ownership._set_owns("Cabc:T1", False)
    out = _dispatch(_slack_event(text="yes please"))
    assert isinstance(out, dict) and out["action"] == "skip"


def test_non_mention_unknown_thread_skips():
    # No prior ownership recorded → not the owner → stay silent.
    out = _dispatch(_slack_event(text="lol"))
    assert isinstance(out, dict) and out["action"] == "skip"


def test_last_mention_wins_when_multiple_bots_mentioned():
    # @OTHER then @ME → I'm last → I own AND reply.
    e = _slack_event(text=f"<@{OTHER_ID}> <@{MY_ID}> thoughts?")
    assert _dispatch(e) is None
    assert thread_ownership._get_owns("Cabc:T1") is True


def test_mentioned_but_not_last_replies_but_does_not_own():
    # @ME then @OTHER → I'm mentioned (reply to THIS msg) but OTHER is last (owner).
    e = _slack_event(text=f"<@{MY_ID}> <@{OTHER_ID}> over to you")
    assert _dispatch(e) is None  # I'm mentioned → reply
    assert thread_ownership._get_owns("Cabc:T1") is False  # but I don't own follow-ups
    # subsequent plain follow-up → I stay silent
    out = _dispatch(_slack_event(text="cool"))
    assert isinstance(out, dict) and out["action"] == "skip"


def test_full_transfer_then_followup_is_silent():
    # I own → someone @s another bot → I yield → plain follow-up → I stay silent.
    thread_ownership._set_owns("Cabc:T1", True)
    _dispatch(_slack_event(text=f"<@{OTHER_ID}> take this"))
    out = _dispatch(_slack_event(text="thanks"))
    assert isinstance(out, dict) and out["action"] == "skip"


def test_ownership_is_per_thread():
    thread_ownership._set_owns("Cabc:T1", True)
    # A different thread I don't own → plain follow-up → skip.
    out = _dispatch(_slack_event(text="hi", thread_ts="T2"))
    assert isinstance(out, dict) and out["action"] == "skip"


def test_human_only_mention_is_treated_as_addressed_elsewhere():
    # A mention of a human (not me) → I yield, same as any non-self mention.
    thread_ownership._set_owns("Cabc:T1", True)
    out = _dispatch(_slack_event(text="<@U0HUMAN1> thanks"))
    assert isinstance(out, dict) and out["action"] == "skip"


def test_register_wires_hook_callback():
    ctx = MagicMock()
    thread_ownership.register(ctx)
    ctx.register_hook.assert_called_once()
    args, _ = ctx.register_hook.call_args
    assert args[0] == "pre_gateway_dispatch"
    assert callable(args[1])
