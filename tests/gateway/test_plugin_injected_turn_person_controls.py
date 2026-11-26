"""A person's Desktop/TUI steer or redirect, and a Stop on any surface, end a plugin-injected turn's wait
for the session's background work at once, as a queued message does: none of them is queued, so the
arrival signal that ends the wait for queued input never reports them."""
from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace

import pytest

from tests.gateway.test_plugin_injected_turn import PROMPT, WAITS, _gateway_session, _start_background_work
from tests.waiting import async_wait_until

pytest_plugins = ("tests.gateway.test_plugin_injected_turn",)  # the ``probe`` fixture

FAST = 5.0


def _ref_and_actor(runner, probe):
    from gateway.session_contract import SessionActor, SessionRef
    authority = runner.session_authority
    sid = authority.logical_owner(runner.session_store.lookup_by_session_key(probe.session_key).session_id)
    actor = SessionActor("human", authority.profile_id, frozenset({"session:submit", "session:control"}), "desktop")
    return authority, SessionRef(authority.profile_id, sid), actor


@pytest.mark.asyncio
@pytest.mark.parametrize("waits_in", ["resident", "refine"])
async def test_desktop_steer_ends_the_wait(monkeypatch, probe, caplog, waits_in):
    from gateway.session_busy_controls import correct

    caplog.set_level(logging.INFO, logger="agent.background_review")
    async with _gateway_session(monkeypatch, probe) as runner:
        bg = await _start_background_work(monkeypatch, probe)
        if waits_in == "resident":
            bg.refine.review.stop.set()
            bg.refine.thread.join(10)
        assert probe.ctx.inject_message(PROMPT, session_key=probe.session_key) is True
        await async_wait_until(lambda: WAITS in caplog.text, timeout=10, what="the wait")
        authority, ref, actor = _ref_and_actor(runner, probe)
        generation = authority._handle(ref).execution_generation
        steered = []
        bg.parent.steer = lambda text, **kw: steered.append(text) or True
        if waits_in == "resident":
            authority.adopted[ref.session_id] = (generation, bg.parent)  # the turn adopted it
        reply = await correct(SimpleNamespace(authority=authority, actor=actor), ref,
                              {"session_id": ref.session_id, "text": "B", "execution_generation": generation},
                              verb="steer")
        t0 = time.monotonic()
        try:
            await async_wait_until(lambda: bg.fork.interrupted, timeout=FAST, what="fork superseded")
            ok = True
        except Exception:
            ok = False
        elapsed = time.monotonic() - t0
        bg.fork.stop.set(); bg.refine.review.stop.set()
        await asyncio.to_thread(bg.handle.wait, 10)
        bg.refine.thread.join(10)
    assert reply["status"] == "queued"
    assert ok, f"person's Desktop steer did not end the wait within {FAST}s ({elapsed:.1f}s)"


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["authority.interrupt", "messaging /stop", "messaging /new"])
async def test_a_stop_ends_the_refine_path_wait(monkeypatch, probe, caplog, how):
    from gateway.platforms.event import MessageEvent

    caplog.set_level(logging.INFO, logger="agent.background_review")
    async with _gateway_session(monkeypatch, probe) as runner:
        bg = await _start_background_work(monkeypatch, probe)
        assert probe.ctx.inject_message(PROMPT, session_key=probe.session_key) is True
        await async_wait_until(lambda: WAITS in caplog.text, timeout=10, what="the wait")
        authority, ref, actor = _ref_and_actor(runner, probe)
        if how == "authority.interrupt":
            await authority.interrupt(actor, ref, authority._handle(ref).execution_generation)
        else:
            await probe.adapter.handle_message(MessageEvent(text=how.split()[1], source=probe.source, message_id="S"))
        try:
            await async_wait_until(lambda: bg.refine.review.interrupted, timeout=FAST, what="review superseded")
            ok = True
        except Exception:
            ok = False
        bg.fork.stop.set(); bg.refine.review.stop.set()
        await asyncio.to_thread(bg.handle.wait, 10)
        bg.refine.thread.join(10)
    assert ok, f"person's Stop ({how}) did not end the /refine-path wait within {FAST}s"
