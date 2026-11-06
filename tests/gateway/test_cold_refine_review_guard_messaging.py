"""A cold messaging ``/refine`` runs under the session's review registry slot.

``/refine`` for a messaging session whose cached agent is gone
(``gateway/session_cold_review.start_cold_refine``) builds a fresh parent on the stored
session id and makes the resident path's own ``_spawn_background_review(explicit=True)``
call from it. The run it starts is therefore the registry's answer for
``(profile, session_id)`` -- never an id-less ``anon:`` guard -- so a live turn's gateway
admission on that session cancels it and waits for its exit, two sessions of one profile
run their cold reviews side by side, and a second ``/refine`` on a session whose review is
still running is refused as ``start_failed`` instead of being reported as started. Stubs as
in ``test_local_refine_review_guard_sessions``.
"""

import contextlib
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import review_admission
from tests.waiting import wait_until


def _runner(sessions):
    """A ``GatewayRunner`` whose agent cache is empty (every session is cold) and whose
    session store answers ``sessions`` (``session_key -> session_id``) with a one-turn
    transcript."""
    from gateway.run import GatewayRunner

    async def peek_session_id(session_key):
        return sessions[session_key]

    async def load_transcript(session_id):
        return [
            {"role": "user", "content": "h"},
            {"role": "assistant", "content": "a"},
        ]

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_key_for_source = lambda source: source.chat_id
    runner._delivery_adapter_for = lambda source: None
    runner._thread_metadata_for_source = lambda source: None
    # ``async_session_store`` is a property that re-validates its facade against ``session_store``.
    runner.session_store = store = object()
    runner._async_session_store = SimpleNamespace(
        _store=store, peek_session_id=peek_session_id, load_transcript=load_transcript
    )
    return runner


def _event(chat_id):
    event = MagicMock()
    event.source = SimpleNamespace(platform="telegram", chat_id=chat_id)
    event.get_command_args.return_value = ""
    return event


def _stub_cold_refine(monkeypatch, review):
    """Stub the cold parent, the fork builder and the fence's interrupt; return the parents
    built and the cold-refine threads started."""
    from agent import background_review, memory_provider
    from gateway import session_cold_review
    from run_agent import AIAgent

    parents, threads = [], []

    @contextlib.contextmanager
    def cold_review_parent(runner, source, session_key, session_id):
        parent = object.__new__(AIAgent)
        parent.session_id = session_id
        parent.valid_tool_names = set()
        parents.append(parent)
        yield parent

    monkeypatch.setattr(session_cold_review, "cold_review_parent", cold_review_parent)
    monkeypatch.setattr(
        background_review,
        "spawn_background_review_thread",
        lambda agent, *a, review_run=None, **k: (lambda: review(agent, review_run), ""),
    )
    monkeypatch.setattr(
        background_review, "_interrupt_background_review", lambda *a, **k: None
    )
    real_spawn = memory_provider.spawn_context_thread

    def spawn_context_thread(target, **kwargs):
        thread = real_spawn(target, **kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(memory_provider, "spawn_context_thread", spawn_context_thread)
    return parents, threads


def _review(started, release):
    """A review body that admits a fork (so a cancel reaches a running request) and holds
    its slot until released or fenced."""
    from agent.background_review import finish_background_review_run

    def review(agent, run):
        if run.begin_request(SimpleNamespace(session_id=agent.session_id)):
            started.append(agent.session_id)
            wait_until(
                lambda: release.is_set() or run.cancel_requested.is_set(),
                what="the cold review's release or its cancel",
            )
        finish_background_review_run(agent, run)

    return review


@pytest.mark.asyncio
async def test_two_sessions_cold_refines_both_run_and_a_running_session_refuses_another(
    monkeypatch,
):
    sessions = {"chat-a": "cold-refine-session-a", "chat-b": "cold-refine-session-b"}
    runner = _runner(sessions)
    started, release = [], threading.Event()
    parents, threads = _stub_cold_refine(monkeypatch, _review(started, release))
    try:
        first = await runner._handle_refine_command(_event("chat-a"))
        wait_until(lambda: started, what="the first cold review started")
        second = await runner._handle_refine_command(_event("chat-b"))
        wait_until(
            lambda: len(started) == 2, what="the second session's cold review started"
        )
        duplicate = await runner._handle_refine_command(_event("chat-a"))
        assert sorted(started) == sorted(sessions.values()), started
        # Each run is the registry's answer for ITS session, keyed by the stored session id:
        # an automatic review there is excluded in-process, and no id-less bucket exists.
        for parent in parents[:2]:
            assert (
                review_admission.current_review_run(parent.session_id)
                is parent._background_review_run
            )
        assert review_admission.current_review_run("") is None
    finally:
        release.set()
        for thread in threads:
            thread.join(10)
    for reply in (first, second):
        assert "Reviewing this conversation" in reply, reply
    assert "failed to start" in duplicate and "already running" in duplicate, duplicate
    assert len(started) == 2, (
        "a /refine on a running session started another",
        started,
    )
    for session_id in sessions.values():
        assert review_admission.current_review_run(session_id) is None


@pytest.mark.asyncio
async def test_a_live_turns_gateway_admission_cancels_the_cold_refine(monkeypatch):
    from gateway.run_turn_review import _GatewayReviewAdmission

    session_id = "cold-refine-session-live"
    runner = _runner({"chat-live": session_id})
    started, release = [], threading.Event()
    parents, threads = _stub_cold_refine(monkeypatch, _review(started, release))
    profile_key = review_admission.current_profile_key()
    try:
        reply = await runner._handle_refine_command(_event("chat-live"))
        wait_until(lambda: started, what="the cold review started")
        run = parents[0]._background_review_run
        assert review_admission.current_review_run(session_id, profile_key) is run
        # The live turn's admission is keyed exactly as the cold run was published: it fences
        # the run, interrupts the admitted fork and returns only once the request has exited.
        admission = await _GatewayReviewAdmission.begin(runner, session_id, profile_key)
        try:
            assert run.cancel_requested.is_set()
            assert run.request_done.is_set()
            assert review_admission.current_review_run(session_id, profile_key) is None
        finally:
            review_admission.note_turn_finished(
                session_id, admission.token, profile_key
            )
    finally:
        release.set()
        for thread in threads:
            thread.join(10)
    assert "Reviewing this conversation" in reply, reply
    assert started == [session_id], started
