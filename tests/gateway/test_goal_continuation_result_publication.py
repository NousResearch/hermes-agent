"""RED-first crash boundaries for claimed-result publication ownership.

These tests exercise the production runner/ledger seams with an isolated
HERMES_HOME.  A claimed continuation result must acquire durable delivery
ownership before its claim can be acknowledged or its turn can be cleared.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "isolated-hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


def _source(*, profile=None, chat_id="publication-chat"):
    from gateway.config import Platform
    from gateway.session import SessionSource

    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=chat_id,
        chat_type="channel",
        user_id="publication-user",
        thread_id="publication-thread",
        profile=profile,
    )


def _source_json(source):
    payload = source.to_dict()
    payload["is_bot"] = bool(source.is_bot)
    payload["role_authorized"] = False
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _continuation(source=None, *, message_id="publication-head"):
    from gateway.platforms.event import MessageEvent, MessageType
    from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE

    return MessageEvent(
        text=CONTINUATION_PROMPT_TEMPLATE.format(goal="publish the completed result"),
        message_type=MessageType.TEXT,
        source=source or _source(),
        message_id=message_id,
        metadata={"identity": message_id},
        internal=False,
        allow_gateway_control=False,
        goal_continuation=True,
    )


def _partial_runner(source, session_key, session_id):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(
        platform=source.platform,
        _pending_messages={},
        extract_media=lambda text: ([], text),
        extract_images=lambda text: ([], text),
        extract_local_files=lambda text: ([], text),
    )
    runner.adapters = {source.platform: adapter}
    runner._profile_adapters = {}
    runner._sessions = {}
    runner._goal_continuation_retries = {}
    runner._background_tasks = set()
    runner._startup_restore_in_progress = True
    runner._startup_restore_tasks = []
    runner._startup_restore_queue = []
    runner._draining = False
    runner._external_drain_active = False
    runner._pending_approvals = {}
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner.session_store = SimpleNamespace(
        _entries={
            session_key: SimpleNamespace(
                session_key=session_key,
                session_id=session_id,
                origin=source,
            )
        }
    )
    runner._session_key_for_source = lambda candidate: (
        session_key if candidate.chat_id == source.chat_id else "wrong-route"
    )
    runner._adapter_for_source = lambda candidate: (
        adapter if candidate is not None and candidate.platform == source.platform else None
    )
    runner._persist_active_agents = lambda: None
    runner._is_user_authorized = lambda _candidate: True
    return runner, adapter


def _claim(runner, adapter, event, *, session_id):
    retry = runner._claim_goal_continuation_retry(
        runner._session_key_for_source(event.source),
        adapter,
        event,
        session_id=session_id,
    )
    assert retry is not None
    return retry


def _rows():
    from gateway import delivery_ledger as dl

    with dl._connect() as conn:
        return conn.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, state, attempts, claim_id, claim_event_id
               FROM delivery_obligations ORDER BY created_at, obligation_id"""
        ).fetchall()


def _row_for_event(event):
    from gateway import delivery_ledger as dl
    from gateway.goal_continuation_claims import event_claim_identity

    identity = event_claim_identity(event)
    assert identity is not None
    with dl._connect() as conn:
        return conn.execute(
            """SELECT obligation_id, content, state, attempts, claim_id,
                      claim_event_id
               FROM delivery_obligations
               WHERE claim_id=? AND claim_event_id=?""",
            identity,
        ).fetchone()


@pytest.mark.asyncio
async def test_local_result_is_staged_before_claim_ack(isolated_home, monkeypatch):
    """Sequence A/B: durable output ownership must precede claim retirement."""
    from gateway.goal_continuation_claims import load_claims

    source = _source()
    key = "agent:main:discord:channel:publication-local"
    session_id = "sid-publication-local"
    runner, adapter = _partial_runner(source, key, session_id)
    runner.session_store.get_active_turn_token = lambda _key: "turn-local-owned"
    adapter.extract_media = lambda text: (
        [("C:/private/result.png", False)],
        text.split("\nMEDIA:", 1)[0],
    )
    adapter.extract_images = lambda text: ([], text)
    adapter.extract_local_files = lambda text: ([], text)
    event = _continuation(source)
    _claim(runner, adapter, event, session_id=session_id)

    original_complete = runner._complete_goal_continuation_claim_event
    observed = []

    def assert_staged_before_ack(session_key, candidate_adapter, candidate_event):
        observed.append(_row_for_event(candidate_event))
        return original_complete(session_key, candidate_adapter, candidate_event)

    monkeypatch.setattr(
        runner,
        "_complete_goal_continuation_claim_event",
        assert_staged_before_ack,
    )
    runner._run_agent_inner = AsyncMock(
        return_value={
            "final_response": "durable local result\nMEDIA:C:/private/result.png",
            "messages": [],
        }
    )

    result = await runner._run_agent(
        event.text,
        "",
        [],
        source,
        session_id,
        session_key=key,
        claimed_event=event,
    )

    assert observed and observed[0][1:3] == ("durable local result", "pending")
    assert result["_delivery_obligation_id"] == observed[0][0]
    assert load_claims(home=isolated_home) == []
    assert adapter._pending_messages == {}
    from gateway import delivery_ledger as dl

    assert dl.completed_active_turn_tokens() == {key: {"turn-local-owned"}}
    with dl._connect() as conn:
        raw_content = conn.execute(
            "SELECT raw_content FROM delivery_obligations WHERE obligation_id=?",
            (result["_delivery_obligation_id"],),
        ).fetchone()[0]
    assert raw_content == "durable local result\nMEDIA:C:/private/result.png"


@pytest.mark.asyncio
async def test_unclaimed_message_keeps_ordinary_execution_and_streaming_path(
    isolated_home,
):
    from gateway.platforms.event import MessageEvent

    source = _source(chat_id="ordinary")
    key = "agent:main:discord:channel:ordinary"
    runner, _adapter = _partial_runner(source, key, "sid-ordinary")
    ordinary_event = MessageEvent(
        text="ordinary message",
        source=source,
        message_id="ordinary-message",
    )
    runner._run_agent_inner = AsyncMock(
        return_value={"final_response": "ordinary result", "messages": []}
    )

    result = await runner._run_agent(
        ordinary_event.text,
        "",
        [],
        source,
        "sid-ordinary",
        session_key=key,
        claimed_event=ordinary_event,
    )

    assert result["final_response"] == "ordinary result"
    assert "_delivery_obligation_id" not in result
    assert runner._run_agent_inner.await_args.kwargs[
        "durable_claimed_event"
    ] is False
    assert _rows() == []


@pytest.mark.asyncio
async def test_proxy_result_uses_same_stage_before_ack_boundary(isolated_home, monkeypatch):
    """Proxy execution must not bypass the claimed-result commit boundary."""
    from gateway.goal_continuation_claims import load_claims

    source = _source(chat_id="publication-proxy")
    key = "agent:main:discord:channel:publication-proxy"
    session_id = "sid-publication-proxy"
    runner, adapter = _partial_runner(source, key, session_id)
    runner.config.multiplex_profiles = False
    runner.session_store.get_active_turn_token = lambda _key: "turn-proxy-owned"
    event = _continuation(source, message_id="proxy-head")
    _claim(runner, adapter, event, session_id=session_id)

    runner._get_proxy_url = lambda: "http://isolated-proxy.invalid"
    runner._run_agent_via_proxy = AsyncMock(
        return_value={"final_response": "durable proxy result", "messages": []}
    )
    result = await runner._run_agent(
        event.text,
        "",
        [],
        source,
        session_id,
        session_key=key,
        claimed_event=event,
    )

    rows = _rows()
    assert len(rows) == 1
    row = rows[0]
    assert row[5:7] == ("durable proxy result", "pending")
    assert result["_delivery_obligation_id"] == row[0]
    assert load_claims(home=isolated_home) == []
    assert runner._run_agent_via_proxy.await_args.kwargs[
        "defer_result_publication"
    ] is True


@pytest.mark.asyncio
async def test_queued_delivery_marks_the_pre_staged_result_delivered(isolated_home):
    from gateway import delivery_ledger as dl

    source = _source()
    key = "agent:main:discord:channel:publication-chat"
    runner, _ = _partial_runner(source, key, "sid-direct")
    obligation_id = dl.record_claimed_result(
        session_key=key,
        claim_id="claim-direct",
        claim_event_id="event-direct",
        platform="discord",
        chat_id=source.chat_id,
        thread_id=source.thread_id,
        content="queued completed result",
        adapter_profile=None,
    )
    adapter = SimpleNamespace(
        send=AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="sent-1")
        ),
        edit_message=AsyncMock(),
        extract_media=lambda text: ([], text),
    )

    await runner._deliver_queued_first_response(
        "queued completed result",
        source=source,
        adapter=adapter,
        deliver_media=False,
        delivery_obligation_id=obligation_id,
    )

    adapter.send.assert_awaited_once()
    row = _rows()[0]
    assert row[0] == obligation_id
    assert row[6] == "delivered"


@pytest.mark.asyncio
async def test_claim_ack_failure_keeps_staged_result_without_rearming_execution(
    isolated_home, monkeypatch
):
    """A failed ack may retry delivery/ack, but it must not rerun the agent."""
    from gateway.goal_continuation_claims import load_claims
    from gateway.run import GoalContinuationPublicationError

    source = _source(chat_id="ack-failure")
    key = "agent:main:discord:channel:ack-failure"
    session_id = "sid-ack-failure"
    runner, adapter = _partial_runner(source, key, session_id)
    event = _continuation(source, message_id="ack-failure-head")
    _claim(runner, adapter, event, session_id=session_id)
    runner._run_agent_inner = AsyncMock(
        return_value={"final_response": "result survives ack failure", "messages": []}
    )
    monkeypatch.setattr(
        runner, "_complete_goal_continuation_claim_event", lambda *_args: False
    )

    with pytest.raises(GoalContinuationPublicationError):
        await runner._run_agent(
            event.text,
            "",
            [],
            source,
            session_id,
            session_key=key,
            claimed_event=event,
        )

    assert _row_for_event(event)[1:3] == (
        "result survives ack failure",
        "pending",
    )
    assert len(load_claims(home=isolated_home)) == 1
    assert adapter._pending_messages == {}
    runner._run_agent_inner.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_obligation_failure_never_acknowledges_or_publishes(
    isolated_home, monkeypatch
):
    """Fail closed if durable completed-result staging cannot be established."""
    from gateway import delivery_ledger as dl
    from gateway.goal_continuation_claims import load_claims
    from gateway.run import GoalContinuationPublicationError

    source = _source(chat_id="record-failure")
    key = "agent:main:discord:channel:record-failure"
    session_id = "sid-record-failure"
    runner, adapter = _partial_runner(source, key, session_id)
    event = _continuation(source, message_id="record-failure-head")
    _claim(runner, adapter, event, session_id=session_id)
    runner._run_agent_inner = AsyncMock(
        return_value={"final_response": "must not become volatile only", "messages": []}
    )
    acked = []
    monkeypatch.setattr(
        runner,
        "_complete_goal_continuation_claim_event",
        lambda *_args: acked.append(True) or True,
    )
    original_record_claimed_result = dl.record_claimed_result
    monkeypatch.setattr(
        dl,
        "record_claimed_result",
        lambda **_kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("private db failure")),
    )

    with pytest.raises(GoalContinuationPublicationError, match="durable result staging failed"):
        await runner._run_agent(
            event.text,
            "",
            [],
            source,
            session_id,
            session_key=key,
            claimed_event=event,
        )

    assert acked == []
    assert len(load_claims(home=isolated_home)) == 1
    assert adapter._pending_messages == {}
    monkeypatch.setattr(
        dl, "record_claimed_result", original_record_claimed_result
    )

    restarted, restarted_adapter = _partial_runner(source, key, session_id)
    assert restarted._reconcile_completed_goal_continuation_claims() == 1
    assert restarted._recover_goal_continuation_claims(schedule=False) == 0
    assert restarted_adapter._pending_messages == {}
    rows = _rows()
    assert len(rows) == 1
    assert rows[0][5:7] == ("must not become volatile only", "pending")


@pytest.mark.asyncio
async def test_staged_result_recovery_retires_claim_without_rerunning_agent(
    isolated_home,
):
    """Restart reconciliation must consume completed heads, never execute them again."""
    from gateway import delivery_ledger as dl
    from gateway.goal_continuation_claims import (
        event_claim_identity,
        load_claims,
        stage_completed_result,
    )

    source = _source(chat_id="restart")
    key = "agent:main:discord:channel:restart"
    session_id = "sid-restart"
    runner, adapter = _partial_runner(source, key, session_id)
    event = _continuation(source, message_id="restart-head")
    _claim(runner, adapter, event, session_id=session_id)
    claim_id, event_id = event_claim_identity(event)
    stage_completed_result(
        key,
        claim_id,
        event_id,
        "completed before restart",
        active_turn_token="turn-cleared-before-delivery",
        home=isolated_home,
    )
    oid = dl.record_claimed_result(
        session_key=key,
        claim_id=claim_id,
        claim_event_id=event_id,
        platform=source.platform.value,
        chat_id=source.chat_id,
        thread_id=source.thread_id,
        content="completed before restart",
        adapter_profile=source.profile,
        active_turn_token="turn-cleared-before-delivery",
        raw_content="completed before restart",
        source_json=_source_json(source),
        message_ref=event.message_id,
    )
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1 WHERE obligation_id=?",
            (oid,),
        )

    restarted, restarted_adapter = _partial_runner(source, key, session_id)
    assert restarted._reconcile_completed_goal_continuation_claims() == 1
    assert load_claims(home=isolated_home) == []
    assert restarted._recover_goal_continuation_claims(schedule=False) == 0
    assert restarted_adapter._pending_messages == {}
    claimed = dl.sweep_recoverable()
    assert [row["content"] for row in claimed] == ["completed before restart"]


@pytest.mark.asyncio
async def test_delivered_result_recovery_retires_claim_without_duplicate_publication(
    isolated_home,
):
    """Crash after send/ledger ACK but before claim ACK must not resend or rerun."""
    from gateway import delivery_ledger as dl
    from gateway.goal_continuation_claims import (
        event_claim_identity,
        load_claims,
        stage_completed_result,
    )

    source = _source(chat_id="delivered-before-claim-ack")
    key = "agent:main:discord:channel:delivered-before-claim-ack"
    session_id = "sid-delivered-before-claim-ack"
    runner, adapter = _partial_runner(source, key, session_id)
    event = _continuation(source, message_id="delivered-before-claim-ack-head")
    _claim(runner, adapter, event, session_id=session_id)
    claim_id, event_id = event_claim_identity(event)
    stage_completed_result(
        key,
        claim_id,
        event_id,
        "already delivered",
        home=isolated_home,
    )
    oid = dl.record_claimed_result(
        session_key=key,
        claim_id=claim_id,
        claim_event_id=event_id,
        platform=source.platform.value,
        chat_id=source.chat_id,
        thread_id=source.thread_id,
        content="already delivered",
        adapter_profile=source.profile,
        raw_content="already delivered",
        source_json=_source_json(source),
        message_ref=event.message_id,
    )
    dl.mark_delivered(oid)

    restarted, restarted_adapter = _partial_runner(source, key, session_id)
    assert restarted._reconcile_completed_goal_continuation_claims() == 1
    assert load_claims(home=isolated_home) == []
    assert restarted._recover_goal_continuation_claims(schedule=False) == 0
    assert restarted_adapter._pending_messages == {}
    assert dl.sweep_recoverable() == []


def test_record_claimed_result_is_exactly_idempotent_and_conflicts_fail_closed(
    isolated_home,
):
    """Stable output identity cannot reset delivery state or accept changed bytes."""
    from gateway import delivery_ledger as dl
    from gateway.delivery_ledger import DeliveryObligationConflict

    kwargs = dict(
        session_key="agent:main:discord:channel:idempotent",
        claim_id="claim-idempotent",
        claim_event_id="event-idempotent",
        platform="discord",
        chat_id="idempotent-chat",
        thread_id="idempotent-thread",
        content="stable result bytes",
        adapter_profile="default",
    )
    first = dl.record_claimed_result(**kwargs)
    dl.mark_delivered(first)
    second = dl.record_claimed_result(**kwargs)

    assert first == second
    assert _rows()[0][6] == "delivered"
    assert len(_rows()) == 1
    with pytest.raises(DeliveryObligationConflict):
        dl.record_claimed_result(**{**kwargs, "content": "different result bytes"})
    assert len(_rows()) == 1


def test_claimed_result_identity_excludes_owner_pid_and_contains_no_payload(
    isolated_home, monkeypatch
):
    """A process restart keeps one privacy-preserving output ID."""
    from gateway import delivery_ledger as dl

    kwargs = dict(
        session_key="agent:main:discord:channel:stable-id",
        claim_id="claim-stable-id",
        claim_event_id="event-stable-id",
        platform="discord",
        chat_id="stable-id-chat",
        thread_id=None,
        content="PRIVATE_RESULT_SENTINEL",
        adapter_profile=None,
    )
    monkeypatch.setattr(dl, "_owner_stamp", lambda: (111, 1))
    first = dl.record_claimed_result(**kwargs)
    monkeypatch.setattr(dl, "_owner_stamp", lambda: (222, 2))
    second = dl.record_claimed_result(**kwargs)

    assert first == second
    assert "PRIVATE_RESULT_SENTINEL" not in first
    assert "claim-stable-id" not in first
    assert len(_rows()) == 1


def test_second_crash_during_replay_is_bounded_and_visibly_ambiguous(isolated_home):
    """A replay crash remains one row with a bounded attempt counter and marker."""
    from gateway import delivery_ledger as dl

    oid = dl.record_claimed_result(
        session_key="agent:main:discord:channel:second-crash",
        claim_id="claim-second-crash",
        claim_event_id="event-second-crash",
        platform="discord",
        chat_id="second-crash-chat",
        thread_id=None,
        content="recover me once per ownership claim",
        adapter_profile=None,
    )
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1 WHERE obligation_id=?",
            (oid,),
        )
    first = dl.sweep_recoverable()
    assert len(first) == 1 and first[0]["needs_marker"] is False
    dl.mark_attempting(oid)
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1 WHERE obligation_id=?",
            (oid,),
        )
    second = dl.sweep_recoverable()
    assert len(second) == 1
    assert second[0]["needs_marker"] is True
    assert second[0]["attempts"] == 2
    assert len(_rows()) == 1


def _race_sweep(home: str, gate, output):
    os.environ["HERMES_HOME"] = home
    from hermes_constants import set_hermes_home_override

    set_hermes_home_override(home)
    from gateway import delivery_ledger as dl

    gate.wait(10)
    output.put([row["obligation_id"] for row in dl.sweep_recoverable()])


def test_overlapping_recovery_writers_claim_one_publication(isolated_home):
    """Two restart writers cannot both own and publish the same completed result."""
    from gateway import delivery_ledger as dl

    oid = dl.record_claimed_result(
        session_key="agent:main:discord:channel:recovery-race",
        claim_id="claim-recovery-race",
        claim_event_id="event-recovery-race",
        platform="discord",
        chat_id="recovery-race-chat",
        thread_id=None,
        content="one recovery owner",
        adapter_profile=None,
    )
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1 WHERE obligation_id=?",
            (oid,),
        )

    ctx = multiprocessing.get_context("spawn")
    gate = ctx.Event()
    output = ctx.Queue()
    processes = [
        ctx.Process(target=_race_sweep, args=(str(isolated_home), gate, output))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    gate.set()
    results = [output.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(20)
        assert process.exitcode == 0

    assert sorted(map(len, results)) == [0, 1]
    assert sorted(row for result in results for row in result) == [oid]


def test_claimed_result_rows_remain_bounded(isolated_home, monkeypatch):
    """Completed-result ownership uses the ledger's hard row bound."""
    from gateway import delivery_ledger as dl

    monkeypatch.setattr(dl, "_MAX_ROWS", 3)
    for index in range(3):
        oid = dl.record_claimed_result(
            session_key=f"agent:main:discord:channel:bounded-{index}",
            claim_id=f"claim-bounded-{index}",
            claim_event_id=f"event-bounded-{index}",
            platform="discord",
            chat_id=f"bounded-{index}",
            thread_id=None,
            content=f"bounded result {index}",
            adapter_profile=None,
        )
        dl.mark_delivered(oid)
    newest_id = dl.record_claimed_result(
        session_key="agent:main:discord:channel:bounded-newest",
        claim_id="claim-bounded-newest",
        claim_event_id="event-bounded-newest",
        platform="discord",
        chat_id="bounded-newest",
        thread_id=None,
        content="new result after terminal pruning",
        adapter_profile=None,
    )
    rows = _rows()
    assert len(rows) == 3
    assert newest_id in {row[0] for row in rows}


def test_ordinary_record_cannot_prune_pending_claimed_results(
    isolated_home, monkeypatch
):
    from gateway import delivery_ledger as dl

    monkeypatch.setattr(dl, "_MAX_ROWS", 3)
    for index in range(3):
        dl.record_claimed_result(
            session_key=f"session-protected-{index}",
            claim_id=f"claim-protected-{index}",
            claim_event_id=f"event-protected-{index}",
            platform="discord",
            chat_id="chat-protected",
            thread_id=None,
            content=f"protected-{index}",
        )

    with pytest.raises(dl.DeliveryObligationCapacityError):
        dl.record_obligation(
            obligation_id="ordinary-overflow",
            session_key="ordinary-session",
            platform="discord",
            chat_id="ordinary-chat",
            thread_id=None,
            content="ordinary output",
        )

    rows = _rows()
    assert len(rows) == 3
    assert {row[8] for row in rows} == {
        "claim-protected-0",
        "claim-protected-1",
        "claim-protected-2",
    }


def test_prepare_claimed_result_atomically_enters_attempting(isolated_home):
    from gateway import delivery_ledger as dl

    obligation_id = dl.record_claimed_result(
        session_key="session-attempting",
        claim_id="claim-attempting",
        claim_event_id="event-attempting",
        platform="discord",
        chat_id="attempting-chat",
        thread_id=None,
        content="raw output",
    )

    assert dl.prepare_claimed_result_delivery(
        obligation_id,
        session_key="session-attempting",
        platform="discord",
        chat_id="attempting-chat",
        thread_id=None,
        content="visible output",
        adapter_profile=None,
    ) is True
    row = _rows()[0]
    assert row[5:7] == ("visible output", "attempting")
    assert dl.mark_claimed_result_delivered(obligation_id) is True


def test_stale_writer_cannot_prepare_or_finish_claimed_delivery(
    isolated_home, monkeypatch
):
    from gateway import delivery_ledger as dl

    obligation_id = dl.record_claimed_result(
        session_key="session-owner",
        claim_id="claim-owner",
        claim_event_id="event-owner",
        platform="discord",
        chat_id="owner-chat",
        thread_id=None,
        content="owner output",
    )
    monkeypatch.setattr(dl, "_owner_stamp", lambda: (999999999, 1.0))

    with pytest.raises(dl.DeliveryObligationConflict):
        dl.prepare_claimed_result_delivery(
            obligation_id,
            session_key="session-owner",
            platform="discord",
            chat_id="owner-chat",
            thread_id=None,
            content="stale rewrite",
            adapter_profile=None,
        )
    assert dl.mark_claimed_result_delivered(obligation_id) is False

    row = _rows()[0]
    assert row[5:7] == ("owner output", "pending")


def test_claimed_result_binds_exact_active_turn_token(isolated_home):
    from gateway import delivery_ledger as dl

    obligation_id = dl.record_claimed_result(
        session_key="session-active-owned",
        claim_id="claim-active-owned",
        claim_event_id="event-active-owned",
        platform="discord",
        chat_id="active-owned",
        thread_id=None,
        content="completed before active clear",
        active_turn_token="turn-token-exact",
    )

    assert dl.completed_active_turn_tokens() == {
        "session-active-owned": {"turn-token-exact"}
    }
    assert dl.get_claimed_result(
        "claim-active-owned", "event-active-owned"
    )["obligation_id"] == obligation_id
    with pytest.raises(dl.DeliveryObligationConflict):
        dl.record_claimed_result(
            session_key="session-active-owned",
            claim_id="claim-active-owned",
            claim_event_id="event-active-owned",
            platform="discord",
            chat_id="active-owned",
            thread_id=None,
            content="completed before active clear",
            active_turn_token="turn-token-other",
        )


def test_hard_exit_after_result_staging_before_claim_ack_recovers_without_execution(
    isolated_home,
):
    """The exact reviewer crash seam leaves both durable identities reconcilable."""
    repo = Path(__file__).resolve().parents[2]
    script = r'''
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

from hermes_constants import set_hermes_home_override
set_hermes_home_override(os.environ["HERMES_HOME"])
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE

source = SessionSource(platform=Platform.DISCORD, chat_id="hard-exit-chat", chat_type="channel", thread_id="hard-exit-thread")
event = MessageEvent(text=CONTINUATION_PROMPT_TEMPLATE.format(goal="hard exit result"), message_type=MessageType.TEXT, source=source, message_id="hard-exit-head", internal=False, allow_gateway_control=False, goal_continuation=True)
key = "agent:main:discord:channel:hard-exit"
runner = object.__new__(GatewayRunner)
adapter = SimpleNamespace(
    platform=source.platform,
    _pending_messages={},
    extract_media=lambda text: ([], text),
    extract_images=lambda text: ([], text),
    extract_local_files=lambda text: ([], text),
)
runner.adapters = {source.platform: adapter}
runner._profile_adapters = {}
runner._sessions = {}
runner._goal_continuation_retries = {}
runner._background_tasks = set()
runner._running_agents = {}
runner._session_run_generation = {}
runner.config = SimpleNamespace(multiplex_profiles=False)
runner.session_store = SimpleNamespace(_entries={key: SimpleNamespace(session_key=key, session_id="sid-hard-exit", origin=source)})
runner._session_key_for_source = lambda _source: key
runner._adapter_for_source = lambda _source: adapter
runner._persist_active_agents = lambda: None
runner._is_user_authorized = lambda _source: True
async def completed(*_args, **_kwargs):
    return {"final_response": "hard-exit completed result", "messages": []}
runner._run_agent_inner = completed
runner._complete_goal_continuation_claim_event = lambda *_args: os._exit(23)
async def main():
    assert runner._claim_goal_continuation_retry(
        key, adapter, event, session_id="sid-hard-exit"
    ) is not None
    await runner._run_agent(
        event.text,
        "",
        [],
        source,
        "sid-hard-exit",
        session_key=key,
        claimed_event=event,
    )
asyncio.run(main())
'''
    env = dict(os.environ)
    env.update(HERMES_HOME=str(isolated_home), PYTHONPATH=str(repo))
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 23, proc.stderr

    from gateway.goal_continuation_claims import load_claims

    claims = load_claims(home=isolated_home)
    assert len(claims) == 1
    event = claims[0].events[0]
    assert _row_for_event(event)[1:3] == ("hard-exit completed result", "pending")
    restarted, adapter = _partial_runner(event.source, claims[0].session_key, claims[0].session_id)
    assert restarted._reconcile_completed_goal_continuation_claims() == 1
    assert restarted._recover_goal_continuation_claims(schedule=False) == 0
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_pause_clear_shutdown_cannot_delete_completed_delivery_ownership(
    isolated_home,
):
    """Conversation lifecycle may retire the input claim, never the staged output row."""
    from gateway import delivery_ledger as dl
    from gateway.goal_continuation_claims import (
        event_claim_identity,
        load_claims,
        stage_completed_result,
    )

    source = _source(chat_id="lifecycle")
    key = "agent:main:discord:channel:lifecycle"
    session_id = "sid-lifecycle"
    runner, adapter = _partial_runner(source, key, session_id)
    event = _continuation(source, message_id="lifecycle-head")
    _claim(runner, adapter, event, session_id=session_id)
    claim_id, event_id = event_claim_identity(event)
    stage_completed_result(
        key,
        claim_id,
        event_id,
        "lifecycle-owned result",
        home=isolated_home,
    )
    oid = dl.record_claimed_result(
        session_key=key,
        claim_id=claim_id,
        claim_event_id=event_id,
        platform=source.platform.value,
        chat_id=source.chat_id,
        thread_id=source.thread_id,
        content="lifecycle-owned result",
        adapter_profile=None,
    )
    setattr(event, "_hermes_execution_completed", True)
    setattr(event, "_hermes_delivery_obligation_id", oid)

    assert runner._drop_goal_continuation_retry(key) == 1
    assert load_claims(home=isolated_home) == []
    rows = _rows()
    assert len(rows) == 1
    assert rows[0][0] == oid and rows[0][5:7] == (
        "lifecycle-owned result",
        "pending",
    )
