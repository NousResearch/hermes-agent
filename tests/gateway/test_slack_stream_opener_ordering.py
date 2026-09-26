"""Ordering race: the body-less stream opener claims the message ts, then eats the completion.

Slack emits a peer agent's streamed post in two envelopes, on the SAME message ts:

  1. the creation event, carrying no body at all — ``event.get("text", "")`` is ``""`` and there
     are no ``blocks`` to fold in, so the adapter dispatches an empty turn (gateway/run_turn.py
     logs it as ``msg=''``);
  2. ``subtype=message_changed``, carrying the finalised text, whose inner ``message.ts`` is that
     same message ts.

The handler claims the ts *before* the slow enrichment awaits
(``_handle_slack_message_impl`` → ``_remember_processed_message_ts(_claim_ts)``), and
``_normalize_changed_message`` drops any ``message_changed`` whose inner ``message.ts`` is already
claimed. When the opener lands first, it therefore claims the ts and then discards envelope 2 —
the only copy of the body in the whole exchange.

The same race has a PREFIX shape: a draft/preview creation event that carries a strict PREFIX of the
final text (production: a 1646-character post reached the gateway as ``Sh``), later completed by the
same-ts ``message_changed``. A later revision of one logical message must REPLACE the body already
delivered for that message — the session must end up with the most complete revision, exactly once:
never the stale prefix, and never the prefix plus the completion as two messages. A genuinely empty
message with no completion is still delivered as-is (no blanket "drop empty" rule).

This is ONE payload and ONE variable: arrival order. The same two envelopes produce a delivered
body in both orders or the test is red, and the favourable order (completion first, no prior
claim) is pinned alongside the racing order so a fix cannot buy one order by breaking the other.

Production evidence (Slack's own record; /home/hermesuser/.hermes/logs/gateway.log, inbound
entries with ``msg=''``): the same payload shape arrived INTACT for ts 1790423744.042979 (1205
chars) and ts 1790423794.532019 (1696 chars), and arrived BLANK at the gateway for ts
1790423483.110059 (681), 1790423490.924079 (773), 1790423667.553539 (1745) and
1790423951.653739 (1593) — a difference in arrival order, not in payload.
"""

import asyncio
import importlib
import sys
from importlib.machinery import PathFinder
from types import ModuleType
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig


def _load_installed_package(name):
    if PathFinder.find_spec(name) is None:
        return None
    prefix = f"{name}."
    displaced = {
        m: sys.modules.pop(m)
        for m in tuple(sys.modules)
        if (m == name or m.startswith(prefix)) and not isinstance(sys.modules[m], ModuleType)
    }
    try:
        return importlib.import_module(name)
    except ImportError:
        sys.modules.update(displaced)
        return None


_load_installed_package("slack_bolt")
_load_installed_package("slack_sdk")

_slack_mod = importlib.import_module("plugins.platforms.slack.adapter")
SlackAdapter = _slack_mod.SlackAdapter

CHANNEL = "C0BF1EYUA9H"
TEAM = "T025KND0E"
# The blank one from the production record: its four siblings in the same log arrived intact.
MESSAGE_TS = "1790423483.110059"
PEER_BOT_ID = "B0PEERAGENT"
PEER_BOT_USER = "U0PEERAGENT"
STREAMED_BODY = "Openclaw's streamed answer, body only on the changed event"
# The draft/preview shape: the transient revision carries a strict prefix of the final text.
# "Sh" is literally what the gateway received for a 1646-character streamed post.
PREFIX = STREAMED_BODY[:2]


def _make_adapter(delivered):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    adapter._bot_user_id = "U0BCLP7DB7B"
    adapter.config.extra["allow_bots"] = "all"
    adapter.config.extra["free_response_channels"] = CHANNEL
    adapter._resolve_user_name = AsyncMock(return_value="Openclaw")

    async def _capture(event):
        delivered.append(event)

    adapter.handle_message = _capture
    return adapter


def _stream_opener_event(text=None):
    """Slack's creation envelope for a streamed peer post, on the message's own ts.

    Deliberately no ``text`` key (not even ``""``) and no ``blocks`` — the unescaped shape that
    falls through ``original_text = event.get("text", "")`` with nothing to fold in. ``text``
    set instead is the draft/preview shape: the creation event carries a strict prefix and the
    same-ts completion carries the rest.
    """
    event = {
        "type": "message",
        "bot_id": PEER_BOT_ID,
        "user": PEER_BOT_USER,
        "channel": CHANNEL,
        "channel_type": "channel",
        "team": TEAM,
        "ts": MESSAGE_TS,
        "event_ts": MESSAGE_TS,
    }
    if text is not None:
        event["text"] = text
    return event


def _finalised_body_event(text=STREAMED_BODY):
    """The completion envelope: same message ts, body in the nested ``message``."""
    return {
        "type": "message",
        "subtype": "message_changed",
        "channel": CHANNEL,
        "channel_type": "channel",
        "team": TEAM,
        "ts": MESSAGE_TS,
        "event_ts": MESSAGE_TS,
        "message": {
            "type": "message",
            "bot_id": PEER_BOT_ID,
            "user": PEER_BOT_USER,
            "text": text,
            "ts": MESSAGE_TS,
        },
    }


def _body():
    return {"team_id": TEAM, "event_id": "Ev0ORDERINGTEST"}


def _run_events(*factories):
    """Deliver each envelope in order; return the bodies the agent was handed, in delivery order."""
    delivered = []
    adapter = _make_adapter(delivered)

    async def scenario():
        for factory in factories:
            await adapter._handle_slack_message(factory(), _body())

    asyncio.run(scenario())
    return [event.text for event in delivered]


def _run(order, envelopes=None):
    """Deliver both envelopes of the one payload in ``order``; return what the agent received."""
    envelopes = envelopes or {
        "completion": _finalised_body_event, "opener": _stream_opener_event,
    }
    return _run_events(*[envelopes[which] for which in order])


ORDERINGS = [
    pytest.param(["completion", "opener"], id="completion_first_favourable"),
    pytest.param(["opener", "completion"], id="opener_first_race"),
]


@pytest.mark.parametrize("order", ORDERINGS)
def test_streamed_body_reaches_the_agent_in_both_orderings(order):
    """Ordering is the only variable: both orders must hand the agent the finalised body once."""
    delivered_texts = _run(order)
    bodies = [text for text in delivered_texts if text.strip()]

    assert bodies == [STREAMED_BODY], (
        f"order={'->'.join(order)}: agent received bodies={bodies!r} "
        f"(all deliveries={delivered_texts!r}). "
        "The body-less opener claimed ts="
        f"{MESSAGE_TS}, so the message_changed envelope carrying the only copy of the body was "
        "dropped as an already-routed duplicate."
    )


@pytest.mark.parametrize("order", ORDERINGS)
def test_strict_prefix_revision_replaces_the_delivered_body(order):
    """A transient revision carrying a strict prefix must not stand: the completion replaces it.

    Asserted on the delivered body itself, by equality, after BOTH envelopes: the session ends up
    with the full text, exactly once. A build that keeps the prefix delivers ``['Sh']``; a build
    that delivers the prefix and then the completion as a second message delivers
    ``['Sh', <full>]`` — both fail this equality (wrong content / wrong count).
    """
    envelopes = {
        "opener": lambda: _stream_opener_event(text=PREFIX),
        "completion": _finalised_body_event,
    }
    delivered = _run(order, envelopes)

    assert delivered == [STREAMED_BODY], (
        f"order={'->'.join(order)}: agent received bodies={delivered!r} "
        f"(expected the most complete revision, once: {STREAMED_BODY!r}). "
        f"transient revision={PREFIX!r}, completion={STREAMED_BODY!r} on ts={MESSAGE_TS}: "
        "the prefix must be replaced by the completion, not stand and not be delivered twice."
    )


@pytest.mark.parametrize("order", ORDERINGS)
def test_empty_opener_is_replaced_by_the_same_ts_completion(order):
    """A body-less opener followed by the same-ts completion ends at the full body, once."""
    delivered = _run(order)

    assert delivered == [STREAMED_BODY], (
        f"order={'->'.join(order)}: agent received bodies={delivered!r} "
        f"(expected {[STREAMED_BODY]!r}). "
        "The empty opener claimed ts="
        f"{MESSAGE_TS}; the completion on that ts carries the only copy of the body."
    )


def test_repeated_completion_of_the_same_text_is_not_delivered_twice():
    """One logical message, one body: a repeat of the same text must add nothing.

    The opener delivers nothing, the first completion supplies the body, and the second carries
    the same text again. Equality on the delivered list catches both a build that never replaces
    the opener (``['']``) and one that re-delivers every completion (``[<full>, <full>]``).
    """
    delivered = _run_events(
        _stream_opener_event,
        _finalised_body_event,
        _finalised_body_event,
    )

    assert delivered == [STREAMED_BODY], (
        f"agent received bodies={delivered!r} (expected exactly one {STREAMED_BODY!r}). "
        "The second completion repeats text already delivered for ts="
        f"{MESSAGE_TS} and must not produce a second delivery."
    )


def test_genuinely_empty_message_without_a_completion_is_still_delivered():
    """No later revision: a genuinely empty body reaches the session as-is, never swallowed."""
    delivered = _run_events(_stream_opener_event)

    assert delivered == [""], (
        f"agent received bodies={delivered!r} (expected ['' ]). "
        "A genuinely empty message with no completion must still be delivered — the fix must not "
        "introduce a blanket 'drop empty message' rule."
    )
