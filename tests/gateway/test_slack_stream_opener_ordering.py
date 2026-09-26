"""A stream/preview sender's transient envelope must never be dispatched as the turn.

Slack emits a peer agent's streamed post as TWO envelopes per logical message, on the SAME
message ts:

  1. the creation/opener envelope — no body at all, or a strict PREFIX of the final text
     (wire shape for a native streamer: ``streaming_state`` "in_progress"); and
  2. a same-ts ``message_changed`` completion carrying the full body.

The adapter cannot tell at envelope (1) whether more is coming, so it must NOT dispatch it: it
holds it and dispatches exactly ONE turn, from the finalised body. The invariant this file pins:

    the agent is never handed an empty or truncated body for a logical message that has a fuller
    revision, and never receives two turns for one message; a genuinely empty message with no
    revision is still delivered; nothing is held indefinitely.

Production evidence (Slack's own record, ``/home/hermesuser/.hermes/logs/gateway.log``): the same
payload shape arrived INTACT for ts 1790423744.042979 (1205 chars) and ts 1790423794.532019 (1696
chars), and arrived BLANK at the gateway for ts 1790423483.110059 (681), 1790423490.924079 (773),
1790423667.553539 (1745) and 1790423951.653739 (1593) — a difference in arrival order, not in
payload. A 1646-character streamed post reached the gateway as ``Sh``.

HARNESS CONTRACT — why every assertion here is a body-level equality on dispatch
--------------------------------------------------------------------------------
``_capture`` records ``event.text`` INSIDE the consumer callback, at call time: exactly the body
``handle_message`` was handed. Reading ``event.text`` off the captured MessageEvent after the run
(the previous shape of this file) cannot see the defect at all — an in-place rewrite of the
already-dispatched object makes such an assertion read the post-mutation record instead of the
delivered body. Assertions compare the FULL delivered list by equality and never filter with
``if text.strip()``, so an extra empty turn fails the comparison instead of being hidden.

Case labels: DETECTOR = fails on the defect build (the point of the case); PIN = passes before and
after, pinning behaviour the fix must not regress.
"""

import asyncio
import importlib
import sys
import time
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
# Slack's own event ts for the completion (a real edit carries a distinct event_ts, and usually an
# ``edited`` marker too): the old shape handed the completion the message ts, which let the
# deduplicator hide the spurious opener turn this file now catches.
EDIT_EVENT_TS = "1790423483.110999"
EDIT_MARKER_TS = "1790423483.111000"
PEER_BOT_ID = "B0PEERAGENT"
PEER_BOT_USER = "U0PEERAGENT"
# The gateway's own bot user id in this harness (a summons target for the mention case).
MENTIONED_BOT_USER = "U0BCLP7DB7B"
FULL_BODY = "Openclaw's streamed answer, body only on the changed event"
# The draft/preview shape: the transient revision carries a strict prefix of the final text.
# "Sh" is literally what the gateway received for a 1646-character streamed post.
PREFIX = FULL_BODY[:2]
# Rich text carried ONLY in blocks (flat ``text`` empty) — the same content loss on another carrier.
BLOCKS_BODY = "deploy #42 succeeded"
HUMAN_USER = "U0374GH838U"
HUMAN_TS = "1790424000.000100"
HUMAN_BODY = "please summarise the deploy log"
HUMAN_EDIT_EVENT_TS = "1790424001.000200"
HUMAN_REWRITE_EVENT_TS = "1790424003.000400"
DELETE_EVENT_TS = "1790424004.000500"
# Both bounded fallback windows, short so a case that leans on the fallback does not stall CI.
# Env-driven (the adapter's documented knob); harmless to code that predates it.
HOLD_ENV = {
    "SLACK_TRANSIENT_HOLD_SECONDS": "0.3",
    "SLACK_STREAM_HOLD_MAX_SECONDS": "0.3",
}


@pytest.fixture(autouse=True)
def _short_hold_windows(monkeypatch):
    for key, value in HOLD_ENV.items():
        monkeypatch.setenv(key, value)


def _make_adapter(delivered, handed=None):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    adapter._bot_user_id = "U0BCLP7DB7B"
    adapter.config.extra["allow_bots"] = "all"
    adapter.config.extra["free_response_channels"] = CHANNEL
    adapter._resolve_user_name = AsyncMock(return_value="Openclaw")

    async def _capture(event):
        # DISPATCH-TIME SNAPSHOT — what the agent is handed, taken before any later envelope can
        # rewrite the object this event points at.
        delivered.append(event.text)
        if handed is not None:
            handed.append(event)

    adapter.handle_message = _capture
    return adapter


def _stream_opener_event(text=None, *, streaming_state=None, edited=None):
    """Slack's creation envelope for a streamed peer post, on the message's own ts.

    No ``text`` key at all (not even ``""``) is the unescaped body-less shape. ``text`` set is the
    draft/preview shape: the creation event carries a strict prefix and the same-ts completion
    carries the rest. ``streaming_state`` is Slack's own marker on a native streamer's envelopes
    (BotMessageEvent: "in_progress" → "completed"/"errored").
    """
    event: dict = {
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
    if streaming_state is not None:
        event["streaming_state"] = streaming_state
    if edited is not None:
        event["edited"] = edited
    return event


def _finalised_body_event(
    text=FULL_BODY, *, blocks=None, event_ts=MESSAGE_TS, edited_ts=None, streaming_state=None
):
    """The completion envelope: same message ts, body in the nested ``message``."""
    message = {
        "type": "message",
        "bot_id": PEER_BOT_ID,
        "user": PEER_BOT_USER,
        "text": text,
        "ts": MESSAGE_TS,
    }
    if blocks is not None:
        message["blocks"] = blocks
    if streaming_state is not None:
        message["streaming_state"] = streaming_state
    if edited_ts is not None:
        message["edited"] = {"user": PEER_BOT_USER, "ts": edited_ts}
    return {
        "type": "message",
        "subtype": "message_changed",
        "channel": CHANNEL,
        "channel_type": "channel",
        "team": TEAM,
        "ts": MESSAGE_TS,
        "event_ts": event_ts,
        "message": message,
    }


def _blocks_only_completion_event():
    """A completion whose whole body lives in ``blocks``; flat ``text`` is empty."""
    return _finalised_body_event(
        text="",
        blocks=[
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [{"type": "text", "text": BLOCKS_BODY}],
                    }
                ],
            }
        ],
    )


def _human_event(text=HUMAN_BODY):
    return {
        "type": "message",
        "user": HUMAN_USER,
        "text": text,
        "channel": CHANNEL,
        "channel_type": "channel",
        "team": TEAM,
        "ts": HUMAN_TS,
        "client_msg_id": "cmid-human-1",
    }


def _human_edit_event(text, *, event_ts):
    """A human's own edit of the message they posted: ``edited`` + a human author."""
    return {
        "type": "message",
        "subtype": "message_changed",
        "channel": CHANNEL,
        "channel_type": "channel",
        "team": TEAM,
        "ts": event_ts,
        "event_ts": event_ts,
        "message": {
            "type": "message",
            "user": HUMAN_USER,
            "text": text,
            "ts": HUMAN_TS,
            "client_msg_id": "cmid-human-1",
            "edited": {"user": HUMAN_USER, "ts": event_ts},
        },
    }


def _human_delete_event():
    return {
        "type": "message",
        "subtype": "message_deleted",
        "channel": CHANNEL,
        "channel_type": "channel",
        "team": TEAM,
        "ts": DELETE_EVENT_TS,
        "deleted_ts": HUMAN_TS,
        "previous_message": _human_event(),
    }


def _body():
    return {"team_id": TEAM, "event_id": "Ev0ORDERINGTEST"}


async def _settle(adapter, delivered, timeout=3.0):
    """Let bounded holds mature (or be cancelled) and their releases finish dispatching.

    A hold that is still pending after this point means the fix swallowed the message; the
    assertion on the delivered list then reports it instead of the test hanging on nothing.
    """
    deadline = time.monotonic() + timeout
    stable, previous = 0, -1
    while time.monotonic() < deadline:
        await asyncio.sleep(0.02)
        held = getattr(adapter, "_held_transient_events", None) or {}
        if not held and len(delivered) == previous:
            stable += 1
            if stable >= 2:
                return
        else:
            stable = 0
        previous = len(delivered)


def _deliver(*factories, handed=None):
    """Deliver each envelope in order; return the bodies the consumer was handed, in order."""
    delivered = []
    adapter = _make_adapter(delivered, handed=handed)

    async def scenario():
        for factory in factories:
            await adapter._handle_slack_message(factory(), _body())
        await _settle(adapter, delivered)

    asyncio.run(scenario())
    return delivered


ORDERINGS = [
    pytest.param([_stream_opener_event, _finalised_body_event], id="opener_first_race"),
    pytest.param([_finalised_body_event, _stream_opener_event], id="completion_first"),
]


@pytest.mark.parametrize("envelopes", ORDERINGS)
def test_body_less_opener_and_completion_hand_over_the_full_body_once(envelopes):
    """The body-less opener (+ its same-ts completion) must land as ONE full-body turn.

    SENSITIVITY: DETECTOR for the racing order — a build that dispatches the opener hands the agent
    ``['']`` where ``[FULL_BODY]`` is required (the completion is then thrown away). PIN for the
    completion-first order (already suppressed by dedup before the fix).
    """
    delivered = _deliver(*envelopes)

    assert delivered == [FULL_BODY], (
        f"agent was handed {delivered!r} (expected exactly one {FULL_BODY!r}). "
        f"The body-less opener claimed ts={MESSAGE_TS}, so the message_changed envelope carrying "
        "the only copy of the body was either dispatched as an empty turn or dropped as a duplicate."
    )


@pytest.mark.parametrize(
    "envelopes",
    [
        pytest.param(
            [lambda: _stream_opener_event(PREFIX), _finalised_body_event], id="opener_first_race"
        ),
        pytest.param(
            [_finalised_body_event, lambda: _stream_opener_event(PREFIX)], id="completion_first"
        ),
    ],
)
def test_prefix_opener_and_completion_hand_over_the_full_body_once(envelopes):
    """A strict-prefix preview (+ its same-ts completion) must land as ONE full-body turn.

    SENSITIVITY: DETECTOR for the racing order — a still-delivers-the-prefix build hands the agent
    ``['Sh']``; a build that delivers both hands ``['Sh', <full>]`` (wrong content, wrong count).
    PIN for the completion-first order.
    """
    delivered = _deliver(*envelopes)

    assert delivered == [FULL_BODY], (
        f"agent was handed {delivered!r} (expected the most complete revision, once: "
        f"{FULL_BODY!r}). transient revision={PREFIX!r}, completion={FULL_BODY!r} on "
        f"ts={MESSAGE_TS}: the transient envelope must never be dispatched."
    )


def test_completion_then_late_opener_with_realistic_edit_event_ts_is_one_delivery():
    """Completion first, its transient opener LATE — and the completion carries Slack's own edit ts.

    Real completions arrive with their own ``event_ts`` (and an ``edited`` marker on the nested
    message), so the deduplicator does not suppress a later envelope carrying the message ts.
    Exactly one delivery, and it is the full body: no spurious empty second turn.

    SENSITIVITY: DETECTOR — a build without a delivered-identity guard hands the agent
    ``[FULL_BODY, '']`` (the late opener becomes a second, empty turn).
    """
    completion = _finalised_body_event(
        event_ts=EDIT_EVENT_TS, edited_ts=EDIT_MARKER_TS, streaming_state="completed"
    )
    delivered = _deliver(
        lambda: completion, _stream_opener_event, lambda: _stream_opener_event(PREFIX)
    )

    assert delivered == [FULL_BODY], (
        f"agent was handed {delivered!r} (expected exactly one {FULL_BODY!r}). The completion "
        f"carried its own event_ts={EDIT_EVENT_TS}, so the late opener on ts={MESSAGE_TS} was not "
        "covered by dedup and must be recognised as an already-delivered logical message."
    )


def test_blocks_only_completion_is_not_lost():
    """A completion whose body is ONLY in ``blocks`` must still reach the agent.

    SENSITIVITY: DETECTOR — comparing (or deriving) the revision body from flat ``text`` alone hands
    the agent ``['']`` and the block content is lost entirely, the same content loss as the
    truncation on a different carrier. Asserts the CONSUMER's body, not that a fold-in helper ran.
    """
    delivered = _deliver(_stream_opener_event, _blocks_only_completion_event)

    assert delivered == [BLOCKS_BODY], (
        f"agent was handed {delivered!r} (expected [{BLOCKS_BODY!r}]). A blocks-only completion has "
        "an empty flat text: folding the blocks in is what keeps its content from being dropped."
    )


def test_native_streaming_state_envelopes_hand_over_the_final_body_once():
    """Slack's own streaming marker: in_progress opener/append, then the completed final body.

    Uses the wire shape a native streamer (``chat.startStream``/``appendStream``/``stopStream``)
    produces — the production sender's documented behaviour.

    SENSITIVITY: DETECTOR — an in_progress envelope is a partial body by definition; a build that
    dispatches it hands the agent the prefix (``['Sh']``) or the empty opener (``['']``).
    """
    in_progress_prefix = _stream_opener_event(PREFIX, streaming_state="in_progress")
    in_progress_append = _finalised_body_event(
        text="Openclaw's streamed ans", event_ts=EDIT_EVENT_TS, streaming_state="in_progress"
    )
    final = _finalised_body_event(
        event_ts="1790423483.112000", edited_ts=EDIT_MARKER_TS, streaming_state="completed"
    )

    delivered = _deliver(
        lambda: in_progress_prefix,
        lambda: in_progress_append,
        lambda: final,
    )

    assert delivered == [FULL_BODY], (
        f"agent was handed {delivered!r} (expected exactly one {FULL_BODY!r}). While "
        "streaming_state is in_progress the body is still growing: only the final envelope may "
        "start the turn."
    )


def test_mentioned_peer_bot_envelope_is_held_and_delivers_the_full_body():
    """A bot envelope that @mentions us is held too: a prefix is a prefix, mentioned or not.

    A summons must not be an exemption — a streaming peer's own answer may mention us mid-stream, so
    exempting mentions would re-open the truncation for the messages that matter most. The turn
    still happens, exactly once, with our mention stripped from the final body.

    SENSITIVITY: DETECTOR — a build that dispatches the creation envelope hands the agent the
    prefix (``['Op']``); a build that exempts summonses from the hold does exactly the same.
    """
    mention = f"<@{MENTIONED_BOT_USER}> "
    delivered = _deliver(
        lambda: _stream_opener_event(mention + PREFIX),
        lambda: _finalised_body_event(mention + FULL_BODY),
    )

    assert delivered == [FULL_BODY], (
        f"agent was handed {delivered!r} (expected exactly one {FULL_BODY!r} with our mention "
        "stripped): a bot envelope that summons us is still a transient envelope."
    )


def test_repeated_identical_completion_is_one_delivery():
    """One logical message, one body: a repeat of the same completion adds nothing.

    SENSITIVITY: DETECTOR — a build that dispatched the opener hands ``['']``; one that
    re-dispatches every completion hands the body twice.
    """
    delivered = _deliver(_stream_opener_event, _finalised_body_event, _finalised_body_event)

    assert delivered == [FULL_BODY], (
        f"agent was handed {delivered!r} (expected exactly one {FULL_BODY!r}). The second "
        f"completion repeats text already delivered for ts={MESSAGE_TS} and must not add a turn."
    )


def test_genuinely_empty_message_without_a_completion_is_still_delivered():
    """No later revision ever arrives: the bounded fallback delivers the message as-is.

    SENSITIVITY: PIN — a genuinely empty message is a real turn and must not be swallowed by a
    blanket "drop empty" rule. (The fallback window is driven from the environment by the autouse
    fixture, so this asserts the timeout path, not a manual release.)
    """
    delivered = _deliver(_stream_opener_event)

    assert delivered == [""], (
        f"agent was handed {delivered!r} (expected ['']). Nothing finalised this message, so the "
        "bounded hold must expire and deliver it — never hold it indefinitely."
    )


def test_human_append_edit_is_not_treated_as_a_stream_revision():
    """A human's own append-edit of an answered message keeps the previous semantics.

    The delivered turn is NOT rewritten and no second turn appears: the ``edited`` marker plus a
    human author means "a person edited their own message", not "a stream finalised".

    SENSITIVITY: PIN on the dispatch-time snapshot (both builds hand the original body). The second
    assertion is a DETECTOR against the rejected in-place-mutation build, which rewrote the already
    dispatched MessageEvent's ``text``: that is a body the agent never saw.
    """
    handed = []
    delivered = _deliver(
        _human_event,
        lambda: _human_edit_event(HUMAN_BODY + " — and the error lines", event_ts=HUMAN_EDIT_EVENT_TS),
        handed=handed,
    )

    assert delivered == [HUMAN_BODY], (
        f"agent was handed {delivered!r} (expected exactly one {HUMAN_BODY!r}). An append-edit of an "
        "already-answered message must not re-deliver anything."
    )
    assert handed and handed[0].text == HUMAN_BODY, (
        f"the delivered MessageEvent was rewritten to {handed[0].text!r}: a later envelope must "
        "never mutate the record of a turn the agent was already handed."
    )


def test_human_rewrite_edit_is_still_dropped():
    """A human REWRITE (not an extension) of an answered message starts no turn.

    SENSITIVITY: PIN — pre-existing semantics (an edit of an already-routed message never
    re-triggers), asserted at the consumer so a fix cannot silently reintroduce it.
    """
    delivered = _deliver(
        _human_event,
        lambda: _human_edit_event(
            "completely different question", event_ts=HUMAN_REWRITE_EVENT_TS
        ),
    )

    assert delivered == [HUMAN_BODY], (
        f"agent was handed {delivered!r} (expected exactly one {HUMAN_BODY!r}). A rewrite of an "
        "already-answered message must not re-trigger a turn."
    )


def test_human_delete_is_still_dropped():
    """A ``message_deleted`` subtype is not a person speaking: no turn, ever.

    SENSITIVITY: PIN — pre-existing housekeeping-subtype semantics.
    """
    delivered = _deliver(_human_event, _human_delete_event)

    assert delivered == [HUMAN_BODY], (
        f"agent was handed {delivered!r} (expected exactly one {HUMAN_BODY!r}). A deletion must not "
        "start a turn."
    )
