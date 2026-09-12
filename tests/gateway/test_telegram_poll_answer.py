"""Telegram poll-answer harvesting — unit tests for the adapter path.

Covers the gateway adapter's ``poll_answer`` handling (Mark-directed: make
poll votes harvestable by agents):

* AC-1: known poll_id -> normalized ``poll_vote`` event carries option_ids,
  option_texts, user, and the origin chat/thread from the send-time cache.
* AC-2: unknown poll_id -> WARNING logged, no event, nothing routed.
* AC-4: LRU eviction at POLL_ORIGIN_CACHE_MAX; a vote on the evicted poll is
  WARNING-dropped, no crash.
* sendPoll origin capture from the raw Bot API envelope (the do_request
  observer on the general request path).
* The synthetic vote event's contract (user-role text, internal=True,
  allow_gateway_control=False, no message_id so votes aren't transcript-
  deduped against each other).

Pure stdlib + pytest + unittest.mock, no live network (AGENTS.md).
"""

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter():
    """TelegramAdapter via object.__new__ (the repo's bare-adapter pattern).

    No __init__ runs: the poll path must self-heal its lazily created
    attributes exactly like every other getattr()-guarded adapter path.
    """
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    # build_source() resolves profile routes via gateway_runner; bare test
    # adapter has none (class default None) -> profile stays None, same as a
    # single-profile gateway.
    return adapter


def _prime_poll(
    adapter,
    poll_id="Q" * 32,
    *,
    chat_id="-100123",
    chat_type="group",
    message_id="501",
    thread_id="77",
    question="Ship the fix now?",
    options=("yes", "no", "wait for tests"),
):
    adapter._remember_sent_poll(
        poll_id,
        chat_id=chat_id,
        chat_type=chat_type,
        message_id=message_id,
        thread_id=thread_id,
        question=question,
        options=list(options),
    )
    return poll_id


def _poll_answer_update(poll_id, option_ids=(0,), user_id=42, update_id=999,
                        username="mark"):
    user = SimpleNamespace(
        id=user_id, username=username, full_name="Mark Norris", first_name="Mark",
    )
    pa = SimpleNamespace(poll_id=poll_id, option_ids=list(option_ids), user=user)
    return SimpleNamespace(update_id=update_id, poll_answer=pa)


# ── AC-1: known poll_id normalizes to a full poll_vote event ──────────


def test_normalize_poll_answer_known_poll_carries_decision_fields():
    adapter = _make_adapter()
    poll_id = _prime_poll(adapter)

    event = adapter._normalize_poll_answer_event(
        _poll_answer_update(poll_id, option_ids=[2])
    )

    assert event is not None
    assert event["platform"] == "telegram"
    assert event["event_type"] == "poll_vote"
    payload = event["payload"]
    # The decision itself: which options, resolved to human-readable text.
    assert payload["option_ids"] == [2]
    assert payload["option_texts"] == ["wait for tests"]
    assert payload["question"] == "Ship the fix now?"
    # Voter identity.
    assert payload["user"]["id"] == "42"
    assert payload["user"]["name"] == "mark"
    # Origin routing context from the send-time cache.
    assert payload["chat_id"] == "-100123"
    assert payload["thread_id"] == "77"
    assert payload["message_id"] == "501"
    assert payload["poll_id"] == poll_id
    assert payload["update_id"] == 999


def test_normalize_poll_answer_multi_and_retraction():
    adapter = _make_adapter()
    poll_id = _prime_poll(adapter)

    multi = adapter._normalize_poll_answer_event(
        _poll_answer_update(poll_id, option_ids=[0, 1])
    )
    assert multi["payload"]["option_texts"] == ["yes", "no"]

    # A retraction carries empty option_ids — still a real event, still
    # attributed, with no fabricated choice.
    retracted = adapter._normalize_poll_answer_event(
        _poll_answer_update(poll_id, option_ids=[])
    )
    assert retracted is not None
    assert retracted["payload"]["option_ids"] == []
    assert retracted["payload"]["option_texts"] == []


def test_normalize_poll_answer_out_of_range_index_is_tolerated():
    """option_ids beyond the cached options list degrade, never crash."""
    adapter = _make_adapter()
    poll_id = _prime_poll(adapter)

    event = adapter._normalize_poll_answer_event(
        _poll_answer_update(poll_id, option_ids=[9])
    )
    assert event["payload"]["option_ids"] == [9]
    assert event["payload"]["option_texts"] == []


def test_normalize_poll_answer_dm_without_thread():
    adapter = _make_adapter()
    poll_id = _prime_poll(adapter, thread_id=None, chat_type="dm")

    event = adapter._normalize_poll_answer_event(
        _poll_answer_update(poll_id, option_ids=[0])
    )
    assert event["payload"]["thread_id"] is None
    assert event["payload"]["chat_id"] == "-100123"


def test_platform_event_dispatch_routes_poll_answer():
    """The catch-all normalizer must wire poll_answer to its contract."""
    adapter = _make_adapter()
    poll_id = _prime_poll(adapter)

    event = adapter._normalize_platform_event(
        _poll_answer_update(poll_id, option_ids=[0])
    )
    assert event is not None and event["event_type"] == "poll_vote"

    # Unknown polls and unrelated update types keep returning None.
    assert adapter._normalize_platform_event(
        _poll_answer_update("never-seen")
    ) is None
    assert adapter._normalize_platform_event(
        SimpleNamespace(poll_answer=None, message_reaction=None, edited_message=None)
    ) is None


def test_source_from_poll_answer_for_auth():
    adapter = _make_adapter()
    poll_id = _prime_poll(adapter)

    source = adapter._source_from_poll_answer_for_auth(
        _poll_answer_update(poll_id, option_ids=[0])
    )
    assert source.platform == Platform.TELEGRAM
    assert source.chat_id == "-100123"
    assert source.user_id == "42"
    assert source.thread_id == "77"

    # Unknown poll or missing voter -> ValueError so the post-auth boundary
    # fails closed (same contract as reaction/edit sources).
    with pytest.raises(ValueError):
        adapter._source_from_poll_answer_for_auth(
            _poll_answer_update("no-such-poll")
        )
    anonymous = _poll_answer_update(poll_id)
    anonymous.poll_answer.user = None
    with pytest.raises(ValueError):
        adapter._source_from_poll_answer_for_auth(anonymous)


# ── AC-2: unknown poll_id -> WARNING, no event, nothing routed ────────


def test_unknown_poll_id_logs_warning_and_no_event(caplog):
    adapter = _make_adapter()
    with caplog.at_level(logging.WARNING):
        asyncio.run(adapter._handle_poll_answer(
            _poll_answer_update("Q" * 32, option_ids=[0]), None
        ))

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "unknown poll_id" in msg
    assert "Q" * 32 in msg  # the warning names the poll it dropped


# ── AC-4: LRU bound, oldest evicted; vote on evicted poll is safe ─────


def test_poll_origin_cache_evicts_oldest_and_survives_late_vote(caplog):
    adapter = _make_adapter()
    cap = adapter._poll_origin_cache()

    for i in range(adapter.POLL_ORIGIN_CACHE_MAX):
        adapter._remember_sent_poll(
            f"p{i}", chat_id="-1000", chat_type="dm", message_id=str(1000 + i),
            thread_id=None, question=f"q{i}", options=["a", "b"],
        )
    assert len(cap) == adapter.POLL_ORIGIN_CACHE_MAX

    # Touch the oldest poll (a lookup refreshes its recency), THEN push one
    # more entry: LRU order must evict p1 (the oldest untouched), not p0.
    assert adapter._lookup_sent_poll("p0") is not None
    adapter._remember_sent_poll(
        "p512", chat_id="-1000", chat_type="dm", message_id="1512",
        thread_id=None, question="q512", options=["a", "b"],
    )
    assert len(cap) == adapter.POLL_ORIGIN_CACHE_MAX
    assert "p0" in cap and "p512" in cap
    assert "p1" not in cap

    # A vote on the evicted poll degrades to WARNING + drop, no crash.
    with caplog.at_level(logging.WARNING):
        asyncio.run(adapter._handle_poll_answer(
            _poll_answer_update("p1", option_ids=[0]), None
        ))
    assert any("unknown poll_id" in r.getMessage() for r in caplog.records)


def test_remember_sent_poll_respects_bounds():
    adapter = _make_adapter()
    adapter._remember_sent_poll(
        "pX", chat_id="-1", chat_type="private", message_id="1",
        thread_id=None, question="q" * 500, options=["o" * 600] * 40,
    )
    origin = adapter._lookup_sent_poll("pX")
    assert len(origin["question"]) <= 200
    assert len(origin["options"]) <= 10
    assert all(len(o) <= 512 for o in origin["options"])


# ── sendPoll origin capture (the sendPoll-time memory) ────────────────


def test_observe_send_poll_result_from_wire_envelope():
    adapter = _make_adapter()
    envelope = json.dumps({
        "ok": True,
        "result": {
            "message_id": 55,
            "chat": {"id": -10099, "type": "supergroup"},
            "date": 1788000000,
            "poll": {
                "id": "ABC123",
                "question": "Pick one",
                "options": [{"text": "left", "voter_count": 0},
                            {"text": "right", "voter_count": 0}],
                "total_voter_count": 0,
                "is_anonymous": False,
                "type": "regular",
                "allows_multiple_answers": False,
            },
            "message_thread_id": 88,
        },
    }).encode()
    request_data = SimpleNamespace(parameters={
        "question": "Pick one",
        "options": json.dumps([{"text": "left"}, {"text": "right"}]),
    })

    origin = adapter._observe_send_poll_result(request_data, (200, envelope))
    assert origin is not None
    assert origin["poll_id"] == "ABC123"
    assert origin["chat_id"] == "-10099"
    assert origin["message_id"] == "55"
    assert origin["thread_id"] == "88"
    # supergroup + thread -> forum (same normalization _build_message_event uses)
    assert origin["chat_type"] == "forum"
    assert origin["options"] == ["left", "right"]

    # The captured origin is what makes the later vote harvestable.
    event = adapter._normalize_poll_answer_event(
        _poll_answer_update("ABC123", option_ids=[1])
    )
    assert event["payload"]["option_texts"] == ["right"]


@pytest.mark.parametrize("result", [
    (500, b'{"ok": false}'),
    (200, b"not json"),
    (200, b'{"ok": true, "result": {"message_id": 1}}'),
    (200, b'{"ok": false, "description": "Bad Request"}'),
])
def test_observe_send_poll_result_rejects_unusable_envelopes(result):
    adapter = _make_adapter()
    assert adapter._observe_send_poll_result(None, result) is None
    assert len(adapter._poll_origin_cache()) == 0


def test_instrumented_request_captures_send_poll_on_do_request():
    """The real capture path: a do_request round-trip records the origin."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = _make_adapter()

    class _BaseRequest:
        __slots__ = ()

        async def do_request(self, url, method, request_data=None, **kwargs):
            return 200, json.dumps({
                "ok": True,
                "result": {
                    "message_id": 7,
                    "chat": {"id": 111, "type": "private"},
                    "poll": {"id": "Z9", "question": "Go?",
                             "options": [{"text": "go", "voter_count": 0},
                                         {"text": "stop", "voter_count": 0}]},
                },
            }).encode()

    request = _BaseRequest()
    # PTB request objects are slotted; the __class__ swap must stay legal.
    instrumented = TelegramAdapter._instrument_send_poll_capture(adapter, request)
    assert type(instrumented) is not _BaseRequest

    async def _run():
        return await instrumented.do_request(
            "https://api.telegram.org/botTOK/sendPoll", "POST",
            request_data=SimpleNamespace(parameters={
                "question": "Go?",
                "options": json.dumps([{"text": "go"}, {"text": "stop"}]),
            }),
        )

    status, _payload = asyncio.run(_run())
    assert status == 200
    origin = adapter._lookup_sent_poll("Z9")
    assert origin["chat_id"] == "111"
    assert origin["chat_type"] == "dm"

    # Non-sendPoll requests are untouched — no origin recorded.
    async def _run_other():
        return await instrumented.do_request(
            "https://api.telegram.org/botTOK/sendMessage", "POST",
            request_data=SimpleNamespace(parameters={"text": "hi"}),
        )

    asyncio.run(_run_other())
    assert len(adapter._poll_origin_cache()) == 1

    # Re-instrumenting the same class is a no-op guard (idempotent __class__ swap).
    again = TelegramAdapter._instrument_send_poll_capture(adapter, instrumented)
    assert type(again) is type(instrumented)


# ── Vote event contract (what reaches the session path) ───────────────


def test_poll_vote_builds_internal_user_event_into_owning_session():
    adapter = _make_adapter()
    poll_id = _prime_poll(adapter)
    captured = {}

    async def _capture(event):
        captured["event"] = event

    adapter.handle_message = _capture
    adapter._is_callback_user_authorized = lambda *a, **k: True

    update = _poll_answer_update(poll_id, option_ids=[0], user_id=42)
    asyncio.run(adapter._handle_poll_answer(update, None))

    event = captured["event"]
    assert event.text == "[Poll vote] Ship the fix now?: yes"
    assert event.internal is True              # never pairs, never interrupts
    assert event.allow_gateway_control is False  # option labels are not commands
    assert event.message_id is None            # votes must not transcript-dedupe
    assert event.platform_update_id == 999
    assert event.source.chat_id == "-100123"   # the poll's chat, not the voter's
    assert event.source.thread_id == "77"
    assert event.source.user_id == "42"        # Mark's identity, so auth and
    # per-user session keys bind the vote to the same session that sent the poll
    assert event.metadata["poll_vote"]["option_ids"] == [0]
    assert event.metadata["poll_vote"]["option_texts"] == ["yes"]
    assert event.metadata["poll_vote"]["poll_id"] == poll_id


def test_poll_vote_retraction_text_is_explicit():
    adapter = _make_adapter()
    poll_id = _prime_poll(adapter)
    captured = {}

    async def _capture(event):
        captured["event"] = event

    adapter.handle_message = _capture
    adapter._is_callback_user_authorized = lambda *a, **k: True
    asyncio.run(adapter._handle_poll_answer(
        _poll_answer_update(poll_id, option_ids=[]), None
    ))
    assert captured["event"].text == "[Poll vote] Ship the fix now?: (vote retracted)"


def test_poll_vote_unauthorized_voter_is_dropped(caplog):
    adapter = _make_adapter()
    poll_id = _prime_poll(adapter)
    dispatched = []

    async def _capture(event):
        dispatched.append(event)

    adapter.handle_message = _capture
    adapter._is_callback_user_authorized = lambda *a, **k: False

    with caplog.at_level(logging.WARNING):
        asyncio.run(adapter._handle_poll_answer(
            _poll_answer_update(poll_id, option_ids=[0]), None
        ))
    assert dispatched == []
    assert any("unauthorized" in r.getMessage() for r in caplog.records)


def test_handler_registered_on_app():
    from plugins.platforms.telegram.adapter import TelegramAdapter
    import plugins.platforms.telegram.adapter as tg

    adapter = _make_adapter()
    registered = []

    class _App:
        def add_handler(self, handler, group=0):
            registered.append((type(handler).__name__, group))

    # PollAnswerHandler is a real class whenever PTB is importable (it is in
    # the project venv); assert on the registration site's behavior instead of
    # a class-identity snapshot.
    original = tg.PollAnswerHandler
    class _FakePollAnswerHandler:
        def __init__(self, cb):
            self.cb = cb
    tg.PollAnswerHandler = _FakePollAnswerHandler
    try:
        TelegramAdapter._register_handlers(adapter, _App())
    finally:
        tg.PollAnswerHandler = original

    names = [n for n, _g in registered]
    assert "_FakePollAnswerHandler" in names
    # The vote handler must land in the default group (0): PTB dispatches the
    # first match PER GROUP, so group 99 (the platform-event observer) still
    # fires for poll_answer — both lanes are intended.
    groups = {n: g for n, g in registered}
    assert groups["_FakePollAnswerHandler"] == 0
