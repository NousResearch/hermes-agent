"""A direct-call ingress's review ownership follows the turn, not the object it was dispatched with.

``gateway.run_turn_review.begin_gateway_review_admission`` parks a direct-call ingress's completion
on the event it is handed when that event carries ``_gateway_review_completes_on_event`` (the
authority drain's executed row, both ``/retry`` re-sends, the CLI->gateway handoff), and the owner
that completes it reads the object IT dispatched (``hand_over_review_completion``,
``rehome_review_delivery_completion``, the handoff's ``finally``). ``_handle_message`` can run the
turn on a DIFFERENT object: a ``pre_gateway_dispatch`` ``rewrite`` is a ``dataclasses.replace``
copy, and the FIFO orphan rescue (#99882) runs the oldest orphan instead of the incoming event. A
dynamic attribute survives neither, so the completion was parked on the chat's live adapter guard
(or on the copy), where no owner of the dispatched object reads it: the turn's token stayed live
for the process lifetime (every later automatic review on the session refused ``live_turn_active``)
and its candidate was neither spawned nor dropped, without a log line. The marker now rides the
turn, and the completion is re-homed onto the dispatched object when the handler returns.
"""

from __future__ import annotations

import asyncio
import threading
import types
from unittest.mock import AsyncMock

import pytest

from agent import review_admission
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_turn_review import (
    _GatewayReviewAdmission,
    begin_gateway_review_admission,
)
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source

_FLAG = "_gateway_review_completes_on_event"
_COMPLETION = "_gateway_review_delivery_complete"


def _event(text: str, source) -> MessageEvent:
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


def _direct_call(text: str, source) -> MessageEvent:
    event = _event(text, source)
    setattr(event, _FLAG, True)
    return event


async def _rewrite(name, **_kwargs):
    """A plugin that rewrites every inbound text (``{"action": "rewrite"}``)."""
    if name != "pre_gateway_dispatch":
        return []
    return [{"action": "rewrite", "text": _kwargs["event"].text + " (rewritten)"}]


async def _allow(_name, **_kwargs):
    return []


def _profile_key(runner, source) -> str:
    """How ``begin_gateway_review_admission`` keys the turn's owner."""
    from hermes_constants import hermes_home_key

    try:
        return hermes_home_key(runner._resolve_profile_home_for_source(source))
    except Exception:
        return review_admission.current_profile_key()


def _runner(monkeypatch, hook):
    """The real ``_handle_message`` over a bare runner and a real base adapter; the plugin hook
    is ``hook``; authorization and the post-turn hooks are stand-ins."""
    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", hook)
    runner, adapter = make_restart_runner()
    adapter.gateway_runner = runner
    monkeypatch.setattr(adapter.config, "typing_indicator", False, raising=False)
    runner._delivery_adapter_for = runner._intake_adapter_for = lambda _source: adapter
    runner._is_user_authorized_for_source = lambda _source, **_kwargs: True
    runner._admit_bot_message_for_source = lambda _source: True
    runner._run_post_turn_hooks = AsyncMock()
    source = make_restart_source()
    quick_key = runner._session_key_for_source(source)
    assert quick_key == adapter._event_session_key(_event("", source))
    return runner, adapter, source, quick_key


def _turn_that_only_registers(runner, session_id: str) -> list[MessageEvent]:
    """``_handle_message_with_agent`` stand-in: the real registration and carrier parking,
    then an answer; records the event the turn ran on."""
    ran: list[MessageEvent] = []

    async def turn(event, source, quick_key, _run_generation):
        ran.append(event)
        await begin_gateway_review_admission(
            runner,
            event,
            source,
            types.SimpleNamespace(session_id=session_id),
            quick_key,
        )
        return "reply"

    runner._handle_message_with_agent = turn
    return ran


def _assert_completes_from(dispatched: MessageEvent, session_id: str, *others) -> None:
    """The completion is on ``dispatched`` and nowhere else; completing it releases the token."""
    completion = getattr(dispatched, _COMPLETION, None)
    assert callable(completion), "the direct-call owner's object carries no completion"
    for other in others:
        assert not hasattr(other, _COMPLETION), f"completion parked on {other!r}"
    admission = completion.__self__
    assert (
        review_admission.live_turn_block_reason(session_id, None, admission.profile_key)
        == review_admission.REASON_LIVE_TURN
    )
    completion(delivery_succeeded=True)
    assert (
        review_admission.live_turn_block_reason(session_id, None, admission.profile_key)
        is None
    )


@pytest.mark.asyncio
async def test_rewrite_copy_completes_from_the_dispatched_event(monkeypatch):
    """A ``pre_gateway_dispatch`` rewrite runs the turn on a ``dataclasses.replace`` copy. The
    completion must end up on the object the direct-call owner dispatched, never on the chat's
    live adapter guard (that task completes its own turn) and never on the copy (nobody reads it)."""
    session_id = "swapped-event-rewrite"
    runner, adapter, source, quick_key = _runner(monkeypatch, _rewrite)
    ran = _turn_that_only_registers(runner, session_id)
    live_guard = (
        asyncio.Event()
    )  # the chat's outgoing adapter task, its reply on the wire
    adapter._active_sessions[quick_key] = live_guard
    dispatched = _direct_call("retry me", source)

    assert await runner._handle_message(dispatched) == "reply"

    (turn,) = ran
    assert turn is not dispatched and turn.text == "retry me (rewritten)"
    _assert_completes_from(dispatched, session_id, live_guard, turn)


@pytest.mark.asyncio
async def test_rescued_orphan_turn_completes_from_the_dispatched_event(monkeypatch):
    """The FIFO orphan rescue runs the oldest orphan as this turn and parks the incoming event
    behind it. The orphan's turn is still the direct-call owner's turn: its completion ends up
    on the dispatched object, not on the orphan."""
    session_id = "swapped-event-orphan"
    runner, adapter, source, quick_key = _runner(monkeypatch, _allow)
    ran = _turn_that_only_registers(runner, session_id)
    orphan = _event("orphaned follow-up", source)
    runner._session_state(quick_key).conversation.queued_events.append(orphan)
    dispatched = _direct_call("admitted row", source)

    assert await runner._handle_message(dispatched) == "reply"

    assert ran == [orphan]
    assert adapter._pending_messages[quick_key] is dispatched  # parked behind the chain
    _assert_completes_from(dispatched, session_id, orphan)


class _InlineThread:
    """``finish``'s spawn hop, run inline so the spawn is observed synchronously."""

    def __init__(self, *, target, args=(), kwargs=None, daemon=None, name=None):
        self._run = lambda: target(*args, **(kwargs or {}))

    def start(self):
        self._run()


@pytest.mark.asyncio
async def test_nested_retry_rewritten_by_a_plugin_completes_with_the_command_send(
    monkeypatch,
):
    """``/retry`` typed while the outgoing reply is on the wire nests the retried turn through the
    real ``_handle_message``; a rewriting plugin swaps its event for a copy. The retried turn's
    token is released by the command's inline send and its candidate goes to the outgoing turn,
    which spawns it once its own token is released (``test_retry_nested_review_ownership``
    pins the same contract without the rewrite)."""
    import gateway.run_turn_review as run_turn_review_module

    session_id = "swapped-event-nested-retry"
    runner, adapter, source, quick_key = _runner(monkeypatch, _rewrite)
    monkeypatch.setattr(
        run_turn_review_module,
        "threading",
        types.SimpleNamespace(
            Thread=_InlineThread, Lock=threading.Lock, Event=threading.Event
        ),
    )
    profile_key = _profile_key(runner, source)
    # The outgoing turn: registered, its reply on the wire, its completion already taken by its
    # task and published on the guard for a nested turn's candidate.
    outgoing = await _GatewayReviewAdmission.begin(runner, session_id, profile_key)
    outgoing_guard = asyncio.Event()
    outgoing_guard._gateway_review_delivery_outgoing = outgoing.finish
    adapter._active_sessions[quick_key] = outgoing_guard
    spawns = []

    def _spawn(**kwargs):
        spawns.append(
            (
                kwargs["messages_snapshot"][0]["content"],
                review_admission.other_live_turn(
                    session_id, kwargs["_spawning_turn_token"], profile_key
                ),
            )
        )

    agent = types.SimpleNamespace(
        session_id=session_id, _spawn_background_review=_spawn
    )

    async def _resolve(event, _source):
        return event.source, types.SimpleNamespace(session_id=session_id), quick_key

    async def _prepare(event, *_args):
        completion = getattr(event, _COMPLETION, None) or getattr(
            adapter._active_sessions[quick_key], _COMPLETION, None
        )
        admission = completion.__self__
        admission.bind_agent(agent)
        admission.capture_candidate(
            agent,
            [{"role": "user", "content": event.text}],
            review_memory=True,
            review_skills=False,
        )
        return "reply", []

    async def _load_transcript(_session_id):
        return [
            {"role": "user", "content": "retried prompt"},
            {"role": "assistant", "content": "retracted reply"},
        ]

    async def _rewrite_transcript(_session_id, _history, **_kwargs):
        return True

    async def _get_or_create_session(_source):
        return types.SimpleNamespace(session_id=session_id, last_prompt_tokens=0)

    runner._hmwa_resolve_session = _resolve
    runner._hmwa_prepare_turn = _prepare
    runner._record_model_friction = lambda *_args, **_kwargs: None
    runner.session_store = store = types.SimpleNamespace()
    runner._async_session_store = types.SimpleNamespace(
        _store=store,
        get_or_create_session=_get_or_create_session,
        load_transcript=_load_transcript,
        rewrite_transcript=_rewrite_transcript,
    )

    async def _handler(event):
        assert event.text == "/retry"
        return await runner._handle_retry_command(event)

    adapter.set_message_handler(_handler)

    await adapter._handle_message_while_active(_event("/retry", source), quick_key)

    assert adapter.sent == ["reply"]
    assert not hasattr(outgoing_guard, _COMPLETION)
    # Only the outgoing turn's token is left: the retried turn's was released with the send.
    assert (
        review_admission.other_live_turn(session_id, outgoing.token, profile_key)
        is False
    )
    assert spawns == []  # adopted by the outgoing turn, not spawned beside it
    outgoing.finish(delivery_succeeded=True)
    assert spawns == [("retried prompt (rewritten)", False)]
    assert review_admission.other_live_turn(session_id, None, profile_key) is False


@pytest.mark.asyncio
async def test_drained_row_that_runs_a_rescued_orphan_completes_with_the_drain_send(
    tmp_path, monkeypatch
):
    """The authority drain executes a row with no live waiter (recovered after a restart) while
    the adapter's overflow holds an orphan: the rescue runs the orphan as the row's turn. The
    completion must reach the drain's handover through the row's event, so the drain completes it
    with its own send (``deliver_settled``) and the candidate spawns once, after that send."""
    from tests.gateway.test_background_review_authority_ownership import (
        _TimelineAdapter,
        _install_turn,
        _owner,
    )
    from tests.gateway.test_shared_busy_receive_loop import _statuses
    from tests.waiting import async_wait_until

    timeline, turns = [], {}
    adapter = _TimelineAdapter(timeline)
    runner, authority = await _owner(tmp_path, monkeypatch, adapter)
    source = adapter.build_source(
        chat_id="busy-chat", chat_type="dm", user_id="fixture-user"
    )
    _install_turn(runner, monkeypatch, timeline, turns)
    quick_key = runner._session_key_for_source(source)
    orphan = MessageEvent(text="orphan", source=source, message_id="m0")
    runner._session_state(quick_key).conversation.queued_events.append(orphan)

    # Committed and scheduled without ``admit_message``'s delivery waiter: the drain owes the reply.
    await authority.admit_native(
        MessageEvent(text="row", source=source, message_id="m1")
    )
    await async_wait_until(
        lambda: (
            _statuses(authority) == {"row": "terminal"}
            and not authority.pending_deliveries
        ),
        timeout=20,
    )
    sid, profile_key = turns["orphan"]
    await async_wait_until(
        lambda: review_admission.live_turn_block_reason(sid, None, profile_key) is None,
        timeout=10,
    )
    await async_wait_until(lambda: ("spawn", sid) in timeline, timeout=10)

    assert timeline.count(("sent", "reply to orphan")) == 1
    assert timeline.count(("spawn", sid)) == 1
    assert timeline.index(("sent", "reply to orphan")) < timeline.index(("spawn", sid))
    assert getattr(authority, "review_completions", {}) == {}
