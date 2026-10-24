"""The compression-exhaustion auto-reset drops the exhausted turn's review candidate.

A turn whose pre-API compression times out answers with the timeout notice (a non-empty
``final_response``, so ``finalize_turn`` still hands a review candidate to the gateway's
admission) and is tagged ``compression_exhausted``: the gateway rotates the chat to a fresh,
parentless session and re-homes the follow-ups queued behind the turn onto it
(``reset_and_rehome_messages``), whose FIFO may claim its head before the exhausted turn's reply
goes out. Review admission keys the fresh session apart from the exhausted one (its registry by
session id, its durable lease by lineage root), and the route probe reads only ``queued`` rows,
so the candidate spawned by the exhausted turn's confirmed delivery would replay the abandoned
transcript beside the follow-up's turn: the cost overlap the follow-up fence exists to prevent,
for a transcript with nothing left to learn. The reset drops the candidate instead, with a
body-free slug, and the turn's ownership is still released with its delivery.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest
import pytest_asyncio

from agent import review_admission
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.run_turn_review import (
    _GatewayReviewAdmission,
    begin_gateway_review_admission,
)
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_contract import SessionRef
from gateway.session_ingress_context import native_callback
from hermes_constants import get_hermes_home
from hermes_state_runtime import list_session_admissions
from plugins.platforms.discord.adapter import DiscordAdapter

_SNAPSHOT = [
    {"role": "user", "content": "the oversized turn"},
    {"role": "assistant", "content": "the timeout notice"},
]


def _stub_agent(session_id: str, spawned: list) -> SimpleNamespace:
    return SimpleNamespace(
        session_id=session_id,
        _spawn_background_review=lambda **kwargs: spawned.append(kwargs),
    )


def test_a_dropped_candidate_is_not_spawned_by_the_turns_confirmed_delivery(caplog):
    profile_key = review_admission.current_profile_key()
    spawned: list = []
    kept = _GatewayReviewAdmission(
        "kept-session",
        profile_key,
        review_admission.note_turn_started("kept-session", profile_key),
    )
    dropped = _GatewayReviewAdmission(
        "reset-session",
        profile_key,
        review_admission.note_turn_started("reset-session", profile_key),
    )
    for admission in (kept, dropped):
        agent = _stub_agent(admission.session_id, spawned)
        admission.bind_agent(agent)
        admission.capture_candidate(
            agent, _SNAPSHOT, review_memory=True, review_skills=False
        )
    try:
        with caplog.at_level(logging.INFO, logger="gateway.run"):
            assert (
                dropped.drop_candidate(review_admission.REASON_EXHAUSTION_RESET) is True
            )
        # Ownership is not released by the drop: the turn's reply is still on its way out.
        assert (
            review_admission.other_live_turn("reset-session", None, profile_key) is True
        )
        assert dropped.finish(delivery_succeeded=True) is None
        assert (
            review_admission.other_live_turn("reset-session", None, profile_key)
            is False
        )
        assert dropped.drop_candidate(review_admission.REASON_EXHAUSTION_RESET) is False
        # The same confirmed delivery spawns a candidate nobody dropped.
        spawner = kept.finish(delivery_succeeded=True)
        assert spawner is not None
        spawner.join(5)
    finally:
        for admission in (kept, dropped):
            review_admission.note_turn_finished(
                admission.session_id, admission.token, profile_key
            )
    assert [kwargs["_review_session_id"] for kwargs in spawned] == ["kept-session"]
    assert (
        f"Background review skipped (owner={review_admission.owner_tag(profile_key, 'reset-session')}): "
        f"{review_admission.REASON_EXHAUSTION_RESET}"
    ) in caplog.text
    assert "reset-session" not in caplog.text


@pytest_asyncio.fixture
async def native(monkeypatch):
    """Real GatewayRunner, DiscordAdapter, SessionAuthority and drain over SQLite; scheduling is
    parked so the test starts the fresh session's FIFO itself (the way the reset's
    ``release()`` would) at the moment that matters."""
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(
        PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False)
    )

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="fixture-reply")

    monkeypatch.setattr(adapter, "send", send)
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(
        runner, profile_id="default", instance_id="fixture"
    )
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    try:
        yield SimpleNamespace(
            runner=runner, adapter=adapter, authority=authority, source=source
        )
    finally:
        from gateway.session_runtime_workers import mutation_tasks

        await asyncio.gather(*mutation_tasks(authority))
        authority.db.close()


async def _admit(native, text):
    event = MessageEvent(text=text, source=native.source, message_id=text)
    with native_callback(native.runner, event, get_hermes_home()):
        return await native.authority.admit_native(event)


@pytest.mark.asyncio
async def test_exhausted_turn_review_is_dropped_while_its_rehomed_followup_runs(
    native, monkeypatch, caplog
):
    from gateway.session_results import execution_result

    runner, authority, db = native.runner, native.authority, native.authority.db
    old = (await _admit(native, "A")).ref
    await _admit(native, "B")  # queued behind A, re-homed by A's reset
    spawned: list = []
    completions: list = []
    fresh: dict = {}
    b_started, b_gate = asyncio.Event(), asyncio.Event()

    async def turn(event, source, session_key, run_generation):
        entry = runner.session_store.lookup_by_session_key(session_key)
        captured = execution_result.get()
        if event.text != "A":
            # Parked only on the fresh session: a follow-up the reset left behind on the old
            # one must run through, so a regression fails the assertions instead of hanging.
            if entry.session_id == fresh.get("sid"):
                b_started.set()
                await b_gate.wait()
            captured["result"] = {"final_response": "reply", "messages": []}
            return "reply"
        # The real ownership of this turn, bound and holding the candidate the finalizer froze.
        admission = await begin_gateway_review_admission(
            runner, event, source, entry, session_key
        )
        fresh["admission"] = admission
        agent = _stub_agent(entry.session_id, spawned)
        admission.bind_agent(agent)
        admission.capture_candidate(
            agent, _SNAPSHOT, review_memory=True, review_skills=False
        )
        finish = event._gateway_review_delivery_complete

        def completing(
            **kwargs,
        ):  # the drain's completion, its spawn hop observed in-line
            spawner = finish(**kwargs)
            completions.append((kwargs, spawner))
            if spawner is not None:
                spawner.join(5)
            return spawner

        event._gateway_review_delivery_complete = completing
        result = {
            "failed": True,
            "compression_exhausted": True,
            "final_response": "the timeout notice",
            "messages": [],
        }
        captured["result"] = result
        # Production's turn cleanup releases the running slot before the reset runs.
        runner._release_running_agent_state(session_key, run_generation=run_generation)
        _, new_entry = await runner._hmwa_compression_exhaustion_reset(
            result,
            "context exhausted",
            entry,
            session_key,
            source,
            gateway_review_admission=admission,
        )
        fresh["sid"] = new_entry.session_id
        # The fresh FIFO claims the re-homed B before this turn's reply is delivered.
        fresh["drain"] = asyncio.create_task(
            authority._drain(SessionRef(authority.profile_id, new_entry.session_id))
        )
        await asyncio.wait_for(b_started.wait(), 5)
        return "the timeout notice"

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        await authority._drain(old)
    try:
        new = fresh["sid"]
        assert new != old.session_id
        rows = list_session_admissions(db, session_id=new)
        assert [(row["request_id"], row["status"]) for row in rows] == [
            ("B", "started")
        ]
        # A's reply was confirmed delivered by the drain, which released its ownership ...
        assert [kwargs["delivery_succeeded"] for kwargs, _ in completions] == [True]
        profile_key = fresh["admission"].profile_key
        assert (
            review_admission.other_live_turn(old.session_id, None, profile_key) is False
        )
        # ... and spawned nothing: the exhausted transcript's review was dropped with the reset.
        assert [spawner for _, spawner in completions] == [None]
        assert spawned == []
        assert (
            f"Background review skipped (owner={review_admission.owner_tag(profile_key, old.session_id)}): "
            f"{review_admission.REASON_EXHAUSTION_RESET}"
        ) in caplog.text
    finally:
        b_gate.set()
        if "drain" in fresh:
            await asyncio.wait_for(fresh["drain"], 10)
