"""A nested ``/retry`` turn's review ownership completes with the send that delivers it.

``/retry`` typed while the outgoing reply is on the wire is dispatched inline
(``BasePlatformAdapter._dispatch_inline_reply``), and the runner's handler nests the retried
turn through a synthetic event it builds itself. The inline dispatch read the session guard
before calling the handler; the nested turn reaches the runner's carrier lookup only after
several awaits. In that window the outgoing task's send can complete with a follow-up queued:
its unwind releases the guard and hands the session to a drain task, which installs a fresh
guard and dispatches the follow-up -- straight back into the busy path, because the nested
turn holds the runner's sentinel. A nested completion parked on that guard is what the drain
task takes when its handler returns, and it completes it unconfirmed
(``pending_followup_handoff``) while the nested turn is still executing: the retried turn's
candidate, captured later, is dropped silently, and the inline send that delivers its reply
completes nothing. So the nested turn is a direct-call ingress of the ``/retry`` handler: its
completion rides its own event, re-homed onto the command event the inline dispatch completes
with its send.
"""

from __future__ import annotations

import asyncio
import threading
import types

import pytest

from agent import review_admission
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource

SESSION_ID = "drain-swapped-guard-retry-session"
SESSION_KEY = "drain-swapped-guard-retry-key"


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user"
        ),
    )


class _InlineThread:
    """``finish``'s spawn hop, run inline so the spawn is observed synchronously."""

    def __init__(self, *, target, args=(), kwargs=None, daemon=None, name=None):
        self._run = lambda: target(*args, **(kwargs or {}))

    def start(self):
        self._run()


@pytest.mark.asyncio
async def test_inline_retry_completion_survives_a_drain_task_replacing_the_guard(
    monkeypatch,
):
    import gateway.run_turn_review as run_turn_review_module

    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    monkeypatch.setattr(adapter.config, "typing_indicator", False, raising=False)
    monkeypatch.setattr(
        run_turn_review_module,
        "threading",
        types.SimpleNamespace(
            Thread=_InlineThread, Lock=threading.Lock, Event=threading.Event
        ),
    )
    runner = object.__new__(GatewayRunner)
    profile_key = review_admission.current_profile_key()
    outgoing_guard = asyncio.Event()
    # The outgoing turn: its reply is on the wire, a follow-up is queued behind it.
    adapter._active_sessions[SESSION_KEY] = outgoing_guard
    queued = _event("queued follow-up")
    drain_started, nested_registered = asyncio.Event(), asyncio.Event()
    drain: list[asyncio.Task] = []
    sends, spawns, drained = [], [], []

    def _spawn(**kwargs):
        spawns.append(
            (
                kwargs["messages_snapshot"][0]["content"],
                review_admission.other_live_turn(
                    SESSION_ID, kwargs["_spawning_turn_token"], profile_key
                ),
            )
        )

    agent = types.SimpleNamespace(
        session_id=SESSION_ID, _spawn_background_review=_spawn
    )

    async def _resolve(event, _source):
        # The outgoing task's send completed while /retry rewound the transcript: its unwind
        # released the guard it owned and handed the session to a drain task, which installs
        # a fresh guard and dispatches the queued follow-up -- into the busy path, before this
        # turn reaches its carrier lookup.
        adapter._cleanup_finished_session_task(SESSION_KEY, outgoing_guard)
        task = asyncio.create_task(
            adapter._process_message_background(queued, SESSION_KEY)
        )
        adapter._track_session_task(SESSION_KEY, task)
        drain.append(task)
        await drain_started.wait()
        return event.source, types.SimpleNamespace(session_id=SESSION_ID), SESSION_KEY

    async def _prepare(event, *_args):
        # Registered, completion parked. The follow-up's dispatch now returns (busy: this turn
        # holds the runner's sentinel) and the drain task unwinds before this turn's agent runs.
        completion = getattr(
            event, "_gateway_review_delivery_complete", None
        ) or getattr(
            adapter._active_sessions[SESSION_KEY],
            "_gateway_review_delivery_complete",
            None,
        )
        admission = completion.__self__
        nested_registered.set()
        await drain[0]
        admission.bind_agent(agent)
        admission.capture_candidate(
            agent,
            [{"role": "user", "content": event.text}],
            review_memory=True,
            review_skills=False,
        )
        return "reply", []

    async def _get_or_create_session(_source):
        return types.SimpleNamespace(session_id=SESSION_ID, last_prompt_tokens=0)

    async def _load_transcript(_session_id):
        return [
            {"role": "user", "content": "retried prompt"},
            {"role": "assistant", "content": "retracted reply"},
        ]

    async def _rewrite_transcript(_session_id, _history, **_kwargs):
        return True

    async def _nested(event):
        return await runner._handle_message_with_agent(
            event, event.source, SESSION_KEY, 2
        )

    runner._hmwa_resolve_session = _resolve
    runner._hmwa_prepare_turn = _prepare
    runner._delivery_adapter_for = lambda _source: adapter
    runner._record_model_friction = lambda *_args, **_kwargs: None
    runner.session_store = store = types.SimpleNamespace()
    runner._async_session_store = types.SimpleNamespace(
        _store=store,
        get_or_create_session=_get_or_create_session,
        load_transcript=_load_transcript,
        rewrite_transcript=_rewrite_transcript,
    )
    runner._handle_message = _nested

    async def _handler(event):
        if event.text == "/retry":
            return await runner._handle_retry_command(event)
        # The runner's busy path: the nested turn holds the sentinel, so the follow-up goes
        # back to the slot for the next drain (returning once the nested turn is registered,
        # so the drain task's own unwind is what reads the guard it installed).
        adapter._pending_messages[SESSION_KEY] = event
        drain_started.set()
        await nested_registered.wait()
        return None

    async def _send(chat_id, content, reply_to=None, metadata=None):
        sends.append(content)
        return SendResult(success=True, message_id=f"sent-{len(sends)}")

    adapter.set_message_handler(_handler)
    adapter.send = _send
    adapter._spawn_drain_task = lambda pending, _key, **_kwargs: drained.append(pending)

    await adapter._handle_message_while_active(_event("/retry"), SESSION_KEY)

    assert sends == ["reply"]
    assert drained == [queued]
    assert adapter._active_sessions == {}
    assert review_admission.other_live_turn(SESSION_ID, None, profile_key) is False
    assert spawns == [("retried prompt", False)], (
        "the retried turn's candidate did not spawn once after its own confirmed delivery"
    )
