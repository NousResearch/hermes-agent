"""Crash-safe durability for claimed gateway goal continuations."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Install an isolated profile before any runner or store is constructed."""
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


def _source(*, profile=None, chat_id="crash-safe-chat"):
    from gateway.config import Platform
    from gateway.session import SessionSource

    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=chat_id,
        chat_type="channel",
        user_id="crash-safe-user",
        thread_id="crash-safe-thread",
        profile=profile,
    )


def _event(text, *, source=None, message_id=None, continuation=False, media=False):
    from gateway.platforms.base import MessageEvent, MessageType

    return MessageEvent(
        text=text,
        message_type=MessageType.PHOTO if media else MessageType.TEXT,
        source=source or _source(),
        message_id=message_id or text,
        media_urls=["C:/isolated/media.png"] if media else [],
        media_types=["image/png"] if media else [],
        metadata={"identity": message_id or text},
        internal=False,
        allow_gateway_control=not continuation,
        goal_continuation=continuation,
    )


def _continuation(source=None, *, message_id="continuation-head"):
    from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE

    return _event(
        CONTINUATION_PROMPT_TEMPLATE.format(goal="finish crash-safe recovery"),
        source=source,
        message_id=message_id,
        continuation=True,
    )


def _run_production_drain_failure(isolated_home, mode):
    """Run the real queued-turn drain and fail at its first post-dequeue await."""
    repo = Path(__file__).resolve().parents[2]
    script = r'''
import asyncio
import os
from pathlib import Path
import sys

repo = Path(os.environ["REPO_ROOT"])
sys.path.insert(0, str(repo / "tests" / "gateway"))

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE
from test_run_progress_topics import QueuedGoalDispatchAgent, _run_with_agent

async def main():
    from pytest import MonkeyPatch

    monkeypatch = MonkeyPatch()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )
    continuation = MessageEvent(
        text=CONTINUATION_PROMPT_TEMPLATE.format(goal="finish production crash recovery"),
        source=source,
        message_id="production-continuation",
        internal=False,
        allow_gateway_control=False,
        goal_continuation=True,
    )
    successors = [
        MessageEvent(text="production successor 1", source=source, message_id="production-successor-1"),
        MessageEvent(text="production successor 2", source=source, message_id="production-successor-2"),
    ]
    holder = {}

    def before_run(runner, adapter, session_key, _source):
        holder.update(runner=runner, adapter=adapter, session_key=session_key)
        if os.environ["FAILURE_MODE"] == "hard-exit":
            async def hard_exit_after_production_dequeue(*_args, **_kwargs):
                os._exit(23)

            runner._deliver_queued_first_response = hard_exit_after_production_dequeue
        else:
            class FailOnPostDequeueClear:
                def is_set(self):
                    return False

                def clear(self):
                    raise RuntimeError("injected production drain failure")

            adapter._active_sessions[session_key] = FailOnPostDequeueClear()

    QueuedGoalDispatchAgent.calls = 0
    QueuedGoalDispatchAgent.messages = []
    run_root = Path(os.environ["HERMES_HOME"]) / "run-root"
    run_root.mkdir()
    try:
        await _run_with_agent(
            monkeypatch,
            run_root,
            QueuedGoalDispatchAgent,
            session_id="sid-production-drain",
            pending_event=continuation,
            overflow_events=successors,
            before_run=before_run,
        )
    except RuntimeError as exc:
        assert str(exc) == "injected production drain failure"
        assert holder["adapter"]._pending_messages[holder["session_key"]] is continuation
        from gateway.goal_continuation_claims import load_claims
        claim = load_claims()[0]
        assert [event.message_id for event in claim.events] == [
            "production-continuation",
            "production-successor-1",
            "production-successor-2",
        ]
    else:
        raise AssertionError("production drain failure did not fire")
    finally:
        monkeypatch.undo()

asyncio.run(main())
'''
    env = dict(os.environ)
    env.update(
        HERMES_HOME=str(isolated_home),
        PYTHONPATH=str(repo),
        REPO_ROOT=str(repo),
        FAILURE_MODE=mode,
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_production_drain_hard_exit_recovers_typed_head_and_fifo(isolated_home):
    """The real drain must publish head and successors before its first await."""
    proc = _run_production_drain_failure(isolated_home, "hard-exit")
    assert proc.returncode == 23, proc.stderr

    from gateway.goal_continuation_claims import load_claims

    claim = load_claims(home=isolated_home)[0]
    assert [event.message_id for event in claim.events] == [
        "production-continuation",
        "production-successor-1",
        "production-successor-2",
    ]
    assert [event.goal_continuation for event in claim.events] == [True, False, False]

    source = claim.events[0].source
    runner, adapter = _partial_runner(source, claim.session_key, claim.session_id)
    assert runner._recover_goal_continuation_claims(schedule=False) == 1
    assert runner._goal_continuation_retries[claim.session_key].event.message_id == (
        "production-continuation"
    )
    assert adapter._pending_messages[claim.session_key].message_id == (
        "production-successor-1"
    )
    assert [event.message_id for event in runner._session_state(
        claim.session_key
    ).conversation.queued_events] == ["production-successor-2"]


def test_production_drain_exception_restores_durable_head(isolated_home):
    """A propagating ordinary exception must restore the claimed head in memory."""
    proc = _run_production_drain_failure(isolated_home, "exception")
    assert proc.returncode == 0, proc.stderr


def test_claim_is_durable_before_exclusive_removal_after_hard_exit(isolated_home):
    """The existing claim seam must publish recovery state before ownership moves."""
    repo = Path(__file__).resolve().parents[2]
    script = r'''
import asyncio
import os
import threading
from types import SimpleNamespace

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE

async def main():
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="crash-safe-chat",
        chat_type="channel",
        user_id="crash-safe-user",
        thread_id="crash-safe-thread",
    )
    event = MessageEvent(
        text=CONTINUATION_PROMPT_TEMPLATE.format(goal="finish crash-safe recovery"),
        message_type=MessageType.TEXT,
        source=source,
        message_id="continuation-head",
        internal=False,
        allow_gateway_control=False,
        goal_continuation=True,
    )
    key = "agent:main:discord:channel:crash-safe"
    runner = object.__new__(GatewayRunner)
    runner._sessions = {}
    runner._goal_continuation_retries = {}
    runner.session_store = SimpleNamespace(
        _entries={key: SimpleNamespace(session_id="sid-crash-safe", origin=source)},
        _lock=threading.RLock(),
        _ensure_loaded_locked=lambda: None,
    )
    adapter = SimpleNamespace(_pending_messages={})
    retry = runner._claim_goal_continuation_retry(key, adapter, event)
    assert retry is not None
    os._exit(23)

asyncio.run(main())
'''
    env = dict(os.environ)
    env["HERMES_HOME"] = str(isolated_home)
    env["PYTHONPATH"] = str(repo)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 23, proc.stderr

    records = list(isolated_home.rglob("*.json"))
    assert len(records) == 1, "claimed continuation vanished on hard exit"
    payload = json.loads(records[0].read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["session_key"] == "agent:main:discord:channel:crash-safe"
    assert payload["session_id"] == "sid-crash-safe"
    assert len(payload["events"]) == 1
    assert payload["events"][0]["text"].startswith(
        "[Continuing toward your standing goal]\nGoal: finish crash-safe recovery"
    )
    assert payload["events"][0]["goal_continuation"] is True


def _partial_runner(source, session_key, session_id):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(platform=source.platform, _pending_messages={})
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
    runner._delivery_adapter_for = lambda candidate: (
        adapter if candidate is not None and candidate.platform == source.platform else None
    )
    runner._persist_active_agents = lambda: None
    runner._is_user_authorized = lambda _candidate: True
    return runner, adapter


def test_subprocess_crash_recovers_typed_head_and_distinct_fifo(isolated_home):
    """A hard process exit leaves the claimed head and later arrivals recoverable."""
    repo = Path(__file__).resolve().parents[2]
    script = r'''
import os
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.goal_continuation_claims import publish_claim, append_claim_event
from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE

source = SessionSource(
    platform=Platform.DISCORD,
    chat_id="crash-safe-chat",
    chat_type="channel",
    user_id="crash-safe-user",
    thread_id="crash-safe-thread",
)
head = MessageEvent(
    text=CONTINUATION_PROMPT_TEMPLATE.format(goal="finish crash-safe recovery"),
    message_type=MessageType.TEXT,
    source=source,
    message_id="continuation-head",
    internal=False,
    allow_gateway_control=False,
    goal_continuation=True,
)
claim = publish_claim("agent:main:discord:channel:crash-safe", "sid-crash-safe", [head])
for index in (1, 2):
    append_claim_event(
        claim.session_key,
        claim.claim_id,
        MessageEvent(
            text=f"successor-{index}",
            message_type=MessageType.TEXT,
            source=source,
            message_id=f"successor-{index}",
        ),
    )
os._exit(23)
'''
    env = dict(os.environ)
    env["HERMES_HOME"] = str(isolated_home)
    env["PYTHONPATH"] = str(repo)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 23, proc.stderr

    from gateway.goal_continuation_claims import load_claims

    claims = load_claims(home=isolated_home)
    assert len(claims) == 1
    claim = claims[0]
    assert [event.text for event in claim.events] == [
        _continuation().text,
        "successor-1",
        "successor-2",
    ]
    assert [event.goal_continuation for event in claim.events] == [True, False, False]
    assert [event.message_id for event in claim.events] == [
        "continuation-head",
        "successor-1",
        "successor-2",
    ]
    assert all(event.internal is False for event in claim.events)


@pytest.mark.asyncio
async def test_claim_write_failure_restores_exact_head_without_owner(
    isolated_home, monkeypatch
):
    """Exclusive in-memory ownership is forbidden until the claim is durable."""
    from gateway import goal_continuation_claims as claims

    source = _source()
    key = "agent:main:discord:channel:write-failure"
    runner, adapter = _partial_runner(source, key, "sid-write-failure")
    head = _continuation(source)
    first = _event("first successor", source=source)
    second = _event("second successor", source=source)
    adapter._pending_messages[key] = first
    runner._session_state(key).conversation.queued_events.append(second)

    def fail_publish(*_args, **_kwargs):
        raise claims.GoalContinuationClaimError("private payload must not escape")

    monkeypatch.setattr(claims, "publish_claim", fail_publish)
    claimed = runner._claim_goal_continuation_retry(
        key, adapter, head, session_id="sid-write-failure"
    )

    assert claimed is None
    assert adapter._pending_messages[key] is head
    assert runner._session_state(key).conversation.queued_events == [first, second]
    assert key not in runner._goal_continuation_retries
    assert not list(isolated_home.rglob("*.json"))


@pytest.mark.parametrize("kind", ["corrupt", "oversized"])
def test_corrupt_or_oversized_claim_fails_closed_and_is_preserved(
    isolated_home, kind
):
    from gateway.goal_continuation_claims import (
        MAX_CLAIM_BYTES,
        GoalContinuationClaimError,
        claim_directory,
        load_claims,
    )

    directory = claim_directory(home=isolated_home)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "claim-invalid.json"
    if kind == "corrupt":
        path.write_text("{not valid json", encoding="utf-8")
    else:
        path.write_bytes(b"x" * (MAX_CLAIM_BYTES + 1))

    with pytest.raises(GoalContinuationClaimError):
        load_claims(home=isolated_home)
    assert path.exists()


@pytest.mark.parametrize("mismatch", ["profile", "session"])
def test_runner_rejects_misbound_claim_without_staging_typed_event(
    isolated_home, mismatch
):
    from gateway.goal_continuation_claims import (
        GoalContinuationClaimError,
        load_claims,
        publish_claim,
    )

    stored_source = _source(profile="other" if mismatch == "profile" else None)
    key = "agent:main:discord:channel:misbound"
    session_id = "sid-stored"
    publish_claim(key, session_id, [_continuation(stored_source)], home=isolated_home)

    runner_source = _source()
    runner, adapter = _partial_runner(
        runner_source,
        key,
        "sid-current" if mismatch == "session" else session_id,
    )
    with pytest.raises(GoalContinuationClaimError):
        runner._recover_goal_continuation_claims(schedule=False)

    assert adapter._pending_messages == {}
    assert runner._session_state(key).conversation.queued_events == []
    assert runner._goal_continuation_retries == {}
    assert len(load_claims(home=isolated_home)) == 1


@pytest.mark.asyncio
async def test_retry_arrivals_are_durable_before_fifo_admission(isolated_home):
    """Both text and media successors are mirrored without merge or overtaking."""
    from gateway.goal_continuation_claims import load_claims

    source = _source()
    key = "agent:main:discord:channel:arrivals"
    runner, adapter = _partial_runner(source, key, "sid-arrivals")
    head = _continuation(source)
    retry = runner._claim_goal_continuation_retry(
        key, adapter, head, session_id="sid-arrivals"
    )
    assert retry is not None

    first = _event("first", source=source, message_id="first")
    second = _event("second", source=source, message_id="second", media=True)
    assert await runner._handle_active_session_busy_message(first, key) is True
    assert await runner._handle_active_session_busy_message(second, key) is True

    claim = load_claims(home=isolated_home)[0]
    assert [event.message_id for event in claim.events] == [
        "continuation-head",
        "first",
        "second",
    ]
    assert adapter._pending_messages[key] is first
    assert runner._session_state(key).conversation.queued_events == [second]
    assert first.media_urls == []
    assert second.media_urls == ["C:/isolated/media.png"]


@pytest.mark.asyncio
async def test_successor_append_target_failure_still_recovers_admitted_fifo(
    isolated_home, monkeypatch
):
    """A busy canonical claim target cannot make an admitted successor volatile."""
    from gateway import goal_continuation_claims as claims

    source = _source()
    key = "agent:main:discord:channel:append-fallback"
    session_id = "sid-append-fallback"
    runner, adapter = _partial_runner(source, key, session_id)
    head = _continuation(source, message_id="append-fallback-head")
    retry = runner._claim_goal_continuation_retry(
        key, adapter, head, session_id=session_id
    )
    assert retry is not None
    canonical_path = claims.claim_path(key, home=isolated_home)
    original_write = claims._write_payload

    def fail_canonical_replacement(path, payload, *, must_not_exist=False):
        if Path(path) == canonical_path and not must_not_exist:
            raise claims.GoalContinuationClaimError("simulated busy claim target")
        return original_write(path, payload, must_not_exist=must_not_exist)

    monkeypatch.setattr(claims, "_write_payload", fail_canonical_replacement)
    successor = _event(
        "admitted after busy target",
        source=source,
        message_id="append-fallback-successor",
    )
    assert await runner._handle_active_session_busy_message(successor, key)

    restarted, restarted_adapter = _partial_runner(source, key, session_id)
    assert restarted._recover_goal_continuation_claims(schedule=False) == 1
    assert restarted._goal_continuation_retries[key].event.message_id == (
        "append-fallback-head"
    )
    assert restarted_adapter._pending_messages[key].message_id == (
        "append-fallback-successor"
    )


@pytest.mark.asyncio
async def test_total_successor_publication_failure_does_not_admit_volatile_event(
    isolated_home, monkeypatch
):
    """A successor is not accepted when no durable event record can be created."""
    from gateway import goal_continuation_claims as claims

    source = _source()
    key = "agent:main:discord:channel:append-total-failure"
    runner, adapter = _partial_runner(source, key, "sid-append-total-failure")
    head = _continuation(source, message_id="append-total-failure-head")
    retry = runner._claim_goal_continuation_retry(
        key, adapter, head, session_id="sid-append-total-failure"
    )
    assert retry is not None
    successor = _event(
        "must not become volatile",
        source=source,
        message_id="append-total-failure-successor",
    )

    def fail_append(*_args, **_kwargs):
        raise claims.GoalContinuationClaimError("private sidecar write failure")

    monkeypatch.setattr(claims, "append_claim_event", fail_append)
    with pytest.raises(
        RuntimeError, match="durable continuation successor publication failed"
    ) as raised:
        await runner._handle_active_session_busy_message(successor, key)

    assert "private sidecar write failure" not in str(raised.value)
    assert adapter._pending_messages[key] is head
    assert successor not in runner._session_state(key).conversation.queued_events


@pytest.mark.asyncio
async def test_pause_retirement_preserves_genuine_prefix_collision_successors(
    isolated_home,
):
    from gateway.goal_continuation_claims import load_claims

    source = _source()
    key = "agent:main:discord:channel:pause"
    runner, adapter = _partial_runner(source, key, "sid-pause")
    head = _continuation(source)
    collision = _event(head.text, source=source, message_id="real-user-collision")
    adapter._pending_messages[key] = collision
    retry = runner._claim_goal_continuation_retry(
        key, adapter, head, session_id="sid-pause"
    )
    assert retry is not None

    removed = runner._clear_goal_pending_continuations(key, adapter)

    assert removed == 1
    assert retry.dropped is True
    assert adapter._pending_messages[key] is collision
    assert collision.goal_continuation is False
    durable_successors = load_claims(home=isolated_home)
    assert [event.message_id for event in durable_successors[0].events] == [
        "real-user-collision"
    ]

    restarted, restarted_adapter = _partial_runner(source, key, "sid-pause")
    assert restarted._recover_goal_continuation_claims(schedule=False) == 1
    assert restarted._goal_continuation_retries[key].event.message_id == (
        "real-user-collision"
    )
    assert restarted_adapter._pending_messages == {}


def test_claim_completion_is_head_ordered_and_exactly_once(isolated_home):
    from gateway.goal_continuation_claims import (
        complete_claim_event,
        event_claim_identity,
        load_claims,
        publish_claim,
    )

    source = _source()
    head = _continuation(source)
    successor = _event("successor", source=source, message_id="successor")
    claim = publish_claim(
        "agent:main:discord:channel:complete",
        "sid-complete",
        [head, successor],
        home=isolated_home,
    )
    head_identity = event_claim_identity(head)
    successor_identity = event_claim_identity(successor)
    assert head_identity is not None
    assert successor_identity is not None
    head_id = head_identity[1]
    successor_id = successor_identity[1]

    assert complete_claim_event(
        claim.session_key, claim.claim_id, head_id, home=isolated_home
    ) is True
    remaining = load_claims(home=isolated_home)
    assert [event.text for event in remaining[0].events] == ["successor"]
    assert complete_claim_event(
        claim.session_key, claim.claim_id, successor_id, home=isolated_home
    ) is True
    assert load_claims(home=isolated_home) == []
    assert complete_claim_event(
        claim.session_key, claim.claim_id, successor_id, home=isolated_home
    ) is False


def test_immutable_successor_sidecar_promotes_and_completes_exactly_once(
    isolated_home,
):
    from gateway.goal_continuation_claims import (
        append_claim_event,
        complete_claim_event,
        event_claim_identity,
        load_claims,
        publish_claim,
    )

    source = _source()
    key = "agent:main:discord:channel:sidecar-complete"
    head = _continuation(source, message_id="sidecar-complete-head")
    successor = _event(
        "sidecar successor", source=source, message_id="sidecar-complete-successor"
    )
    claim = publish_claim(key, "sid-sidecar-complete", [head], home=isolated_home)
    append_claim_event(key, claim.claim_id, successor, home=isolated_home)
    head_identity = event_claim_identity(head)
    successor_identity = event_claim_identity(successor)
    assert head_identity is not None and successor_identity is not None

    assert complete_claim_event(
        key, claim.claim_id, head_identity[1], home=isolated_home
    )
    remaining = load_claims(home=isolated_home)
    assert [event.message_id for event in remaining[0].events] == [
        "sidecar-complete-successor"
    ]
    assert remaining[0].synthetic_head_pending is False
    assert complete_claim_event(
        key, claim.claim_id, successor_identity[1], home=isolated_home
    )
    assert load_claims(home=isolated_home) == []
    assert not list(isolated_home.rglob("successor-*.json"))


def test_stale_promoted_sidecar_cannot_resurrect_completed_successor(
    isolated_home, monkeypatch
):
    """Completed sidecar identities remain retired if unlink cleanup fails."""
    from gateway import goal_continuation_claims as claims

    source = _source()
    key = "agent:main:discord:channel:stale-sidecar"
    head = _continuation(source, message_id="stale-sidecar-head")
    first = _event("first sidecar", source=source, message_id="stale-sidecar-first")
    second = _event("second sidecar", source=source, message_id="stale-sidecar-second")
    claim = claims.publish_claim(
        key, "sid-stale-sidecar", [head], home=isolated_home
    )
    claims.append_claim_event(key, claim.claim_id, first, home=isolated_home)
    claims.append_claim_event(key, claim.claim_id, second, home=isolated_home)
    head_identity = claims.event_claim_identity(head)
    first_identity = claims.event_claim_identity(first)
    assert head_identity is not None and first_identity is not None

    monkeypatch.setattr(claims, "_discard_successor_records", lambda _records: None)
    assert claims.complete_claim_event(
        key, claim.claim_id, head_identity[1], home=isolated_home
    )
    assert claims.complete_claim_event(
        key, claim.claim_id, first_identity[1], home=isolated_home
    )

    recovered = claims.load_claims(home=isolated_home)
    assert [event.message_id for event in recovered[0].events] == [
        "stale-sidecar-second"
    ]


@pytest.mark.asyncio
async def test_queue_cap_counts_claimed_durable_head(isolated_home):
    source = _source()
    key = "agent:main:discord:channel:cap"
    runner, adapter = _partial_runner(source, key, "sid-cap")
    successor = _event("successor", source=source)
    adapter._pending_messages[key] = successor
    retry = runner._claim_goal_continuation_retry(
        key, adapter, _continuation(source), session_id="sid-cap"
    )

    assert retry is not None
    assert runner._queue_depth(key, adapter=adapter) == 2


def test_claim_publication_does_not_swallow_baseexception(isolated_home, monkeypatch):
    """Process-level interruption remains authoritative at the fsync boundary."""
    from gateway import goal_continuation_claims as claims

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt("synthetic hard interruption")

    monkeypatch.setattr(claims.os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt, match="synthetic hard interruption"):
        claims.publish_claim(
            "agent:main:discord:channel:baseexception",
            "sid-baseexception",
            [_continuation(_source())],
            home=isolated_home,
        )


def test_loader_rejects_wrong_event_field_types_without_consuming_record(isolated_home):
    from gateway.goal_continuation_claims import (
        GoalContinuationClaimError,
        load_claims,
        publish_claim,
    )

    source = _source()
    claim = publish_claim(
        "agent:main:discord:channel:wrong-field-type",
        "sid-wrong-field-type",
        [_continuation(source)],
        home=isolated_home,
    )
    payload = json.loads(claim.path.read_text(encoding="utf-8"))
    payload["events"][0]["user_id"] = {"not": "a user id"}
    claim.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GoalContinuationClaimError):
        load_claims(home=isolated_home)
    assert claim.path.exists()


def test_loader_rejects_duplicate_event_ids_without_consuming_record(isolated_home):
    from gateway.goal_continuation_claims import (
        GoalContinuationClaimError,
        load_claims,
        publish_claim,
    )

    source = _source()
    claim = publish_claim(
        "agent:main:discord:channel:duplicate-event-id",
        "sid-duplicate-event-id",
        [_continuation(source), _event("successor", source=source)],
        home=isolated_home,
    )
    payload = json.loads(claim.path.read_text(encoding="utf-8"))
    payload["events"][1]["event_id"] = payload["events"][0]["event_id"]
    claim.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GoalContinuationClaimError):
        load_claims(home=isolated_home)
    assert claim.path.exists()


def test_recovery_preflights_all_bindings_before_staging_any_claim(isolated_home):
    """One invalid binding must not partially release another durable FIFO."""
    from gateway.goal_continuation_claims import (
        GoalContinuationClaimError,
        load_claims,
        publish_claim,
    )

    first_source = _source(chat_id="preflight-one")
    second_source = _source(chat_id="preflight-two")
    first_key = "agent:main:discord:channel:preflight-one"
    second_key = "agent:main:discord:channel:preflight-two"
    publish_claim(
        first_key,
        "sid-preflight-one",
        [_continuation(first_source, message_id="preflight-head-one")],
        home=isolated_home,
    )
    publish_claim(
        second_key,
        "sid-preflight-two",
        [_continuation(second_source, message_id="preflight-head-two")],
        home=isolated_home,
    )

    runner, adapter = _partial_runner(first_source, first_key, "sid-preflight-one")
    runner.session_store._entries[second_key] = SimpleNamespace(
        session_id="sid-preflight-two",
        origin=second_source,
    )
    ordered_claims = load_claims(home=isolated_home)
    invalid_key = ordered_claims[1].session_key
    runner.session_store._entries[invalid_key].session_id = "sid-preflight-current"
    keys = {
        first_source.chat_id: first_key,
        second_source.chat_id: second_key,
    }
    runner._session_key_for_source = lambda source: keys[source.chat_id]

    with pytest.raises(GoalContinuationClaimError):
        runner._recover_goal_continuation_claims(schedule=False)

    assert runner._goal_continuation_retries == {}
    assert adapter._pending_messages == {}
    assert runner._peek_session_state(first_key) is None
    assert runner._peek_session_state(second_key) is None


def test_recovery_does_not_persist_stale_role_authorization(isolated_home):
    """A recovered event must pass current authorization instead of cached role trust."""
    from gateway.goal_continuation_claims import load_claims, publish_claim

    source = _source()
    source.role_authorized = True
    publish_claim(
        "agent:main:discord:channel:stale-role",
        "sid-stale-role",
        [_continuation(source)],
        home=isolated_home,
    )

    recovered_source = load_claims(home=isolated_home)[0].events[0].source
    assert recovered_source is not None
    assert recovered_source.role_authorized is False


def test_concurrent_sessions_recover_independently(isolated_home):
    from gateway.goal_continuation_claims import load_claims, publish_claim

    first_source = _source(chat_id="chat-one")
    second_source = _source(chat_id="chat-two")
    publish_claim(
        "agent:main:discord:channel:one",
        "sid-one",
        [_continuation(first_source, message_id="head-one")],
        home=isolated_home,
    )
    publish_claim(
        "agent:main:discord:channel:two",
        "sid-two",
        [_continuation(second_source, message_id="head-two")],
        home=isolated_home,
    )

    claims = load_claims(home=isolated_home)
    assert [(claim.session_id, claim.events[0].message_id) for claim in claims] == [
        ("sid-one", "head-one"),
        ("sid-two", "head-two"),
    ]


def test_recovery_stages_head_then_media_successors_without_deleting_claim(
    isolated_home,
):
    from gateway.goal_continuation_claims import load_claims, publish_claim

    source = _source()
    key = "agent:main:discord:channel:recover"
    session_id = "sid-recover"
    head = _continuation(source)
    first = _event("first", source=source, message_id="first", media=True)
    second = _event("second", source=source, message_id="second")
    publish_claim(key, session_id, [head, first, second], home=isolated_home)

    runner, adapter = _partial_runner(source, key, session_id)
    assert runner._recover_goal_continuation_claims(schedule=False) == 1

    retry = runner._goal_continuation_retries[key]
    assert retry.event.text == head.text
    assert retry.event.goal_continuation is True
    assert adapter._pending_messages[key].message_id == "first"
    assert runner._session_state(key).conversation.queued_events[0].message_id == "second"
    assert len(load_claims(home=isolated_home)) == 1


@pytest.mark.asyncio
async def test_same_object_aliases_are_persisted_once(isolated_home):
    from gateway.goal_continuation_claims import load_claims

    source = _source()
    key = "agent:main:discord:channel:alias"
    runner, adapter = _partial_runner(source, key, "sid-alias")
    head = _continuation(source)
    successor = _event("successor", source=source)
    adapter._pending_messages[key] = successor
    runner._session_state(key).conversation.queued_events[:] = [head, successor]

    retry = runner._claim_goal_continuation_retry(
        key, adapter, head, session_id="sid-alias"
    )

    assert retry is not None
    claim = load_claims(home=isolated_home)[0]
    assert [event.message_id for event in claim.events] == [
        "continuation-head",
        "successor",
    ]


def test_shutdown_spool_does_not_downgrade_claim_owned_event(isolated_home):
    from gateway.goal_continuation_claims import load_claims, publish_claim
    from gateway.shutdown_flush import flush_overflow_to_file, flush_pending_to_file

    source = _source()
    key = "agent:main:discord:channel:shutdown"
    head = _continuation(source)
    successor = _event("successor", source=source)
    publish_claim(key, "sid-shutdown", [head, successor], home=isolated_home)

    assert flush_pending_to_file({key: successor}, reason="adapter_shutdown") == 0
    assert flush_overflow_to_file({key: [successor]}, reason="shutdown") == 0
    assert len(load_claims(home=isolated_home)) == 1
    pending_dir = isolated_home / "pending_messages"
    assert not pending_dir.exists() or list(pending_dir.glob("pending-*.json")) == []


@pytest.mark.asyncio
async def test_queue_cap_rejection_never_enters_durable_fifo(isolated_home):
    """A later event rejected by the live queue cap must not reappear on restart."""
    from gateway.goal_continuation_claims import load_claims

    source = _source()
    key = "agent:main:discord:channel:durable-cap"
    runner, adapter = _partial_runner(source, key, "sid-durable-cap")
    runner._draining = False
    runner._BUSY_QUEUE_MAX_PENDING = 1
    runner._is_user_authorized = lambda _source: True
    head = _continuation(source)
    retry = runner._claim_goal_continuation_retry(
        key, adapter, head, session_id="sid-durable-cap"
    )
    assert retry is not None

    rejected = _event("rejected at cap", source=source)
    assert await runner._handle_active_session_busy_message(rejected, key) is True

    claim = load_claims(home=isolated_home)[0]
    assert [event.text for event in claim.events] == [head.text]
    assert adapter._pending_messages == {}
    assert runner._session_state(key).conversation.queued_events == []


@pytest.mark.asyncio
async def test_arrival_during_cancelled_owner_gap_extends_durable_fifo(isolated_home):
    """A restored tagged head protects arrivals before its replacement owner starts."""
    from gateway.goal_continuation_claims import load_claims

    source = _source()
    key = "agent:main:discord:channel:owner-gap"
    runner, adapter = _partial_runner(source, key, "sid-owner-gap")
    runner._draining = False
    runner._busy_text_mode = "queue"
    runner._busy_input_mode = "queue"
    runner._is_user_authorized = lambda _source: True
    head = _continuation(source)
    retry = runner._claim_goal_continuation_retry(
        key, adapter, head, session_id="sid-owner-gap"
    )
    assert retry is not None
    runner._restore_dequeued_event_front(key, adapter, head)
    runner._finish_goal_continuation_retry(key, retry)

    successor = _event("successor in cancellation gap", source=source)
    assert await runner._handle_active_session_busy_message(successor, key) is True

    claim = load_claims(home=isolated_home)[0]
    assert [event.text for event in claim.events] == [head.text, successor.text]
    assert adapter._pending_messages[key] is head
    assert runner._session_state(key).conversation.queued_events == [successor]


@pytest.mark.asyncio
async def test_runner_completion_ack_advances_then_retires_durable_fifo(isolated_home):
    """A completed event is removed once and its successor remains recoverable."""
    from gateway.goal_continuation_claims import load_claims

    source = _source()
    key = "agent:main:discord:channel:ack"
    runner, adapter = _partial_runner(source, key, "sid-ack")
    head = _continuation(source)
    successor = _event("successor", source=source, message_id="successor")
    adapter._pending_messages[key] = successor
    retry = runner._claim_goal_continuation_retry(
        key, adapter, head, session_id="sid-ack"
    )
    assert retry is not None

    assert runner._complete_goal_continuation_claim_event(key, adapter, head) is True
    remaining = load_claims(home=isolated_home)
    assert len(remaining) == 1
    assert [event.message_id for event in remaining[0].events] == ["successor"]
    assert runner._goal_continuation_retries[key].event is successor

    assert (
        runner._complete_goal_continuation_claim_event(key, adapter, successor)
        is True
    )
    assert load_claims(home=isolated_home) == []
    assert key not in runner._goal_continuation_retries


def test_isolated_claim_paths_never_touch_live_profile(isolated_home):
    from gateway.goal_continuation_claims import claim_path, publish_claim

    key = "agent:main:discord:channel:isolation"
    publish_claim(key, "sid-isolation", [_continuation()], home=isolated_home)
    path = claim_path(key, home=isolated_home).resolve()
    assert path.is_relative_to(isolated_home.resolve())
    assert "AppData/Local/hermes" not in path.as_posix()


def test_busy_target_replacement_failure_preserves_previous_claim_bytes(
    isolated_home, monkeypatch
):
    """A blocked Windows rename cannot fall back to rewriting the live record."""
    from gateway import goal_continuation_claims as claims

    source = _source()
    key = "agent:main:discord:channel:blocked-replace"
    head = _continuation(source)
    claim = claims.publish_claim(
        key, "sid-blocked-replace", [head], home=isolated_home
    )
    path = claims.claim_path(key, home=isolated_home)
    before = path.read_bytes()

    import errno

    def reject_replace(*_args, **_kwargs):
        raise OSError(errno.EBUSY, "simulated busy target")

    monkeypatch.setattr(claims.os, "replace", reject_replace)
    with pytest.raises(claims.GoalContinuationClaimError):
        claims.append_claim_event(
            key,
            claim.claim_id,
            _event("must remain pending", source=source),
            home=isolated_home,
        )

    assert path.read_bytes() == before
    assert [event.message_id for event in claims.load_claims(home=isolated_home)[0].events] == [
        "continuation-head"
    ]


@pytest.mark.asyncio
async def test_initial_claim_post_replace_fsync_failure_reconciles_owner_and_restart(
    isolated_home, monkeypatch
):
    """A visible initial record remains one successful caller-owned publication."""
    from gateway import goal_continuation_claims as claims

    source = _source()
    key = "agent:main:discord:channel:initial-fsync-ambiguity"
    session_id = "sid-initial-fsync-ambiguity"
    runner, adapter = _partial_runner(source, key, session_id)
    head = _continuation(source, message_id="initial-fsync-head")

    def fail_directory_fsync(_path):
        raise OSError("injected post-replace directory fsync failure")

    monkeypatch.setattr(claims, "_fsync_directory", fail_directory_fsync)
    retry = runner._claim_goal_continuation_retry(
        key, adapter, head, session_id=session_id
    )

    assert retry is not None
    identity = claims.event_claim_identity(head)
    assert identity is not None
    assert retry.claim_id == identity[0]
    assert runner._claim_goal_continuation_retry(
        key, adapter, head, session_id=session_id
    ) is retry
    durable = claims.load_claims(home=isolated_home)
    assert len(durable) == 1
    assert [event.message_id for event in durable[0].events] == ["initial-fsync-head"]

    restarted, _restarted_adapter = _partial_runner(source, key, session_id)
    assert restarted._recover_goal_continuation_claims(schedule=False) == 1
    assert restarted._goal_continuation_retries[key].event.message_id == (
        "initial-fsync-head"
    )


def test_successor_post_replace_fsync_failure_reconciles_retry_and_restart(
    isolated_home, monkeypatch
):
    """A visible sidecar is admitted once and keeps its event identity on retry."""
    from gateway import goal_continuation_claims as claims

    source = _source()
    key = "agent:main:discord:channel:successor-fsync-ambiguity"
    session_id = "sid-successor-fsync-ambiguity"
    head = _continuation(source, message_id="successor-fsync-head")
    claim = claims.publish_claim(key, session_id, [head], home=isolated_home)
    successor = _event(
        "successor after fsync ambiguity",
        source=source,
        message_id="successor-fsync-event",
    )

    def fail_directory_fsync(_path):
        raise OSError("injected successor directory fsync failure")

    monkeypatch.setattr(claims, "_fsync_directory", fail_directory_fsync)
    event_id = claims.append_claim_event(
        key, claim.claim_id, successor, home=isolated_home
    )
    assert claims.append_claim_event(
        key, claim.claim_id, successor, home=isolated_home
    ) == event_id
    assert claims.event_claim_identity(successor) == (claim.claim_id, event_id)
    assert len(list(claim.path.parent.glob("successor-*.json"))) == 1
    assert [event.message_id for event in claims.load_claims(home=isolated_home)[0].events] == [
        "successor-fsync-head",
        "successor-fsync-event",
    ]

    restarted, restarted_adapter = _partial_runner(source, key, session_id)
    assert restarted._recover_goal_continuation_claims(schedule=False) == 1
    assert restarted._goal_continuation_retries[key].event.message_id == (
        "successor-fsync-head"
    )
    assert restarted_adapter._pending_messages[key].message_id == (
        "successor-fsync-event"
    )


def test_claim_count_cap_is_enforced_before_publication(isolated_home, monkeypatch):
    """The versioned spool cannot grow beyond its declared record cap."""
    from gateway import goal_continuation_claims as claims

    monkeypatch.setattr(claims, "MAX_CLAIMS", 1)
    source = _source()
    first_key = "agent:main:discord:channel:claim-cap-one"
    second_key = "agent:main:discord:channel:claim-cap-two"
    claims.publish_claim(
        first_key,
        "sid-claim-cap-one",
        [_continuation(source, message_id="claim-cap-one")],
        home=isolated_home,
    )

    with pytest.raises(claims.GoalContinuationClaimError):
        claims.publish_claim(
            second_key,
            "sid-claim-cap-two",
            [_continuation(source, message_id="claim-cap-two")],
            home=isolated_home,
        )

    remaining = claims.load_claims(home=isolated_home)
    assert [(claim.session_key, claim.events[0].message_id) for claim in remaining] == [
        (first_key, "claim-cap-one")
    ]


@pytest.mark.asyncio
async def test_startup_restore_gate_queues_internal_successor_until_recovery(
    isolated_home,
):
    """Plugin/internal arrivals cannot overtake an un-recovered durable head."""
    source = _source()
    key = "agent:main:discord:channel:startup-internal"
    runner, _adapter = _partial_runner(source, key, "sid-startup-internal")
    runner._startup_restore_in_progress = True
    event = _event("startup internal successor", source=source)
    event.internal = True

    async def execute_if_gate_is_bypassed(*_args, **_kwargs):
        return "overtook durable recovery"

    runner._handle_message_with_agent = execute_if_gate_is_bypassed
    assert await runner._handle_message(event) is None
    assert runner._startup_restore_queue == [event]


@pytest.mark.asyncio
async def test_startup_gate_second_crash_recovers_exact_successor_fifo(isolated_home):
    """Arrivals behind an unrecovered head survive another startup crash."""
    from gateway.goal_continuation_claims import publish_claim

    source = _source()
    key = "agent:main:discord:channel:startup-second-crash"
    session_id = "sid-startup-second-crash"
    head = _continuation(source, message_id="startup-second-crash-head")
    publish_claim(key, session_id, [head], home=isolated_home)
    first_runner, _first_adapter = _partial_runner(source, key, session_id)

    successors = [
        _event("startup user successor", source=source, message_id="startup-user"),
        _event("startup internal successor", source=source, message_id="startup-internal"),
        _event("startup plugin successor", source=source, message_id="startup-plugin"),
    ]
    successors[1].internal = True
    successors[2].internal = True
    successors[2].metadata["hermes_plugin_id"] = "isolated-plugin"
    for event in successors:
        assert await first_runner._handle_message(event) is None

    # Discard the first runner without draining: this is the second startup crash.
    restarted, restarted_adapter = _partial_runner(source, key, session_id)
    assert restarted._recover_goal_continuation_claims(schedule=False) == 1
    recovered = restarted._goal_continuation_retries[key].event
    assert recovered.message_id == "startup-second-crash-head"
    assert recovered.goal_continuation is True
    assert restarted_adapter._pending_messages[key].message_id == "startup-user"
    assert [
        event.message_id
        for event in restarted._session_state(key).conversation.queued_events
    ] == ["startup-internal", "startup-plugin"]


def test_startup_gate_after_recovery_preserves_slot_and_overflow_fifo(isolated_home):
    """A late gated arrival joins both durable and already-staged FIFO state."""
    from gateway.goal_continuation_claims import load_claims, publish_claim

    source = _source()
    key = "agent:main:discord:channel:startup-staged"
    session_id = "sid-startup-staged"
    head = _continuation(source, message_id="startup-staged-head")
    slot = _event("startup staged slot", source=source, message_id="startup-staged-slot")
    overflow = _event(
        "startup staged overflow", source=source, message_id="startup-staged-overflow"
    )
    publish_claim(key, session_id, [head, slot, overflow], home=isolated_home)
    runner, adapter = _partial_runner(source, key, session_id)
    assert runner._recover_goal_continuation_claims(schedule=False) == 1

    late = _event("startup late arrival", source=source, message_id="startup-late")
    assert runner._queue_startup_restore_event(late) is True

    assert adapter._pending_messages[key].message_id == "startup-staged-slot"
    assert [
        event.message_id for event in runner._session_state(key).conversation.queued_events
    ] == ["startup-staged-overflow", "startup-late"]
    assert runner._startup_restore_queue == []
    assert [event.message_id for event in load_claims(home=isolated_home)[0].events] == [
        "startup-staged-head",
        "startup-staged-slot",
        "startup-staged-overflow",
        "startup-late",
    ]


def test_startup_gate_claim_cap_rejects_without_volatile_admission(
    isolated_home, monkeypatch
):
    """The normal per-session cap includes an unrecovered durable head."""
    from gateway.goal_continuation_claims import load_claims, publish_claim

    source = _source()
    key = "agent:main:discord:channel:startup-claim-cap"
    session_id = "sid-startup-claim-cap"
    existing = [
        _continuation(source, message_id="startup-cap-head"),
        _event("startup cap one", source=source, message_id="startup-cap-one"),
        _event("startup cap two", source=source, message_id="startup-cap-two"),
    ]
    publish_claim(key, session_id, existing, home=isolated_home)
    runner, _adapter = _partial_runner(source, key, session_id)
    monkeypatch.setattr(runner, "_BUSY_QUEUE_MAX_PENDING", len(existing))

    rejected = _event(
        "startup cap rejected", source=source, message_id="startup-cap-rejected"
    )
    assert runner._queue_startup_restore_event(rejected) is False
    assert runner._startup_restore_queue == []
    assert [event.message_id for event in load_claims(home=isolated_home)[0].events] == [
        "startup-cap-head",
        "startup-cap-one",
        "startup-cap-two",
    ]


def test_startup_gate_without_claim_uses_per_session_cap(isolated_home, monkeypatch):
    """Ordinary gated arrivals cannot grow the process-local queue without bound."""
    source = _source(chat_id="startup-no-claim")
    key = "agent:main:discord:channel:startup-no-claim"
    runner, _adapter = _partial_runner(source, key, "sid-startup-no-claim")
    monkeypatch.setattr(runner, "_BUSY_QUEUE_MAX_PENDING", 2)
    events = [
        _event(f"startup ordinary {index}", source=source, message_id=f"startup-{index}")
        for index in range(3)
    ]

    assert runner._queue_startup_restore_event(events[0]) is True
    assert runner._queue_startup_restore_event(events[1]) is True
    assert runner._queue_startup_restore_event(events[2]) is False
    assert runner._startup_restore_queue == events[:2]


def test_startup_gate_durable_failure_rejects_before_volatile_admission(
    isolated_home, monkeypatch
):
    """A successor is not accepted when its claim sidecar cannot publish."""
    from gateway import goal_continuation_claims as claims

    source = _source()
    key = "agent:main:discord:channel:startup-publish-failure"
    session_id = "sid-startup-publish-failure"
    claims.publish_claim(
        key,
        session_id,
        [_continuation(source, message_id="startup-publish-head")],
        home=isolated_home,
    )
    runner, _adapter = _partial_runner(source, key, session_id)

    def fail_append(*_args, **_kwargs):
        raise claims.GoalContinuationClaimError("private startup publication detail")

    monkeypatch.setattr(claims, "append_claim_event", fail_append)
    with pytest.raises(RuntimeError, match="durable startup successor publication failed"):
        runner._queue_startup_restore_event(
            _event(
                "startup rejected after failure",
                source=source,
                message_id="startup-rejected-after-failure",
            )
        )

    assert runner._startup_restore_queue == []


@pytest.mark.asyncio
async def test_recovered_inactive_claim_retires_before_releasing_successor(
    isolated_home,
):
    """An inactive startup head cannot leave a durable ghost ahead of its successor."""
    from gateway.goal_continuation_claims import load_claims, publish_claim

    source = _source()
    key = "agent:main:discord:channel:recovered-inactive"
    session_id = "sid-recovered-inactive"
    head = _continuation(source, message_id="recovered-inactive-head")
    successor = _event(
        "Automated continuation: user-authored collision",
        source=source,
        message_id="recovered-inactive-successor",
    )
    publish_claim(key, session_id, [head, successor], home=isolated_home)
    runner, adapter = _partial_runner(source, key, session_id)
    assert runner._recover_goal_continuation_claims(schedule=False) == 1
    recovered_head = runner._goal_continuation_retries[key].event
    runner._goal_still_active_for_session = lambda _session_id: False

    assert not await runner._wait_for_goal_continuation_admission(
        event=recovered_head,
        source=source,
        session_id=session_id,
        session_key=key,
        adapter=adapter,
        retain_on_admit=True,
    )

    durable_successors = load_claims(home=isolated_home)
    assert [event.message_id for event in durable_successors[0].events] == [
        "recovered-inactive-successor"
    ]
    assert key not in runner._goal_continuation_retries
    assert adapter._pending_messages[key].message_id == "recovered-inactive-successor"

    restarted, restarted_adapter = _partial_runner(source, key, session_id)
    assert restarted._recover_goal_continuation_claims(schedule=False) == 1
    assert restarted._goal_continuation_retries[key].event.message_id == (
        "recovered-inactive-successor"
    )
    assert restarted_adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_cancelled_claimed_execution_restores_head_for_rearmed_owner(isolated_home):
    """Cancellation before durable completion cannot strand the claimed head."""
    source = _source()
    key = "agent:main:discord:channel:cancelled-execution"
    runner, adapter = _partial_runner(source, key, "sid-cancelled-execution")
    runner.config.multiplex_profiles = False
    head = _continuation(source)
    retry = runner._claim_goal_continuation_retry(
        key, adapter, head, session_id="sid-cancelled-execution"
    )
    assert retry is not None

    async def cancel_before_completion(*_args, **_kwargs):
        raise asyncio.CancelledError

    runner._run_agent_inner = cancel_before_completion
    with pytest.raises(asyncio.CancelledError):
        await runner._run_agent(
            head.text,
            "",
            [],
            source,
            "sid-cancelled-execution",
            session_key=key,
            claimed_event=head,
        )

    assert adapter._pending_messages[key] is head
    assert key not in runner._goal_continuation_retries


def test_recovery_rejects_same_profile_successor_bound_to_another_route(
    isolated_home,
):
    """Every recovered event must bind to the claim's exact session route."""
    from gateway.goal_continuation_claims import (
        GoalContinuationClaimError,
        publish_claim,
    )

    head_source = _source(chat_id="route-a")
    other_source = _source(chat_id="route-b")
    key = "agent:main:discord:channel:route-a"
    publish_claim(
        key,
        "sid-route-a",
        [
            _continuation(head_source, message_id="route-head"),
            _event("other route", source=other_source, message_id="route-successor"),
        ],
        home=isolated_home,
    )
    runner, adapter = _partial_runner(head_source, key, "sid-route-a")
    runner._session_key_for_source = lambda source: (
        key
        if source.chat_id == "route-a"
        else "agent:main:discord:channel:route-b"
    )

    with pytest.raises(GoalContinuationClaimError):
        runner._recover_goal_continuation_claims(schedule=False)

    assert runner._goal_continuation_retries == {}
    assert adapter._pending_messages == {}


def test_active_claim_rejects_an_untyped_semantic_head(isolated_home):
    """An active synthetic claim cannot be recovered with an ordinary head."""
    from gateway.goal_continuation_claims import (
        GoalContinuationClaimError,
        load_claims,
        publish_claim,
    )

    claim = publish_claim(
        "agent:main:discord:channel:semantic-head",
        "sid-semantic-head",
        [_continuation(_source(), message_id="semantic-head")],
        home=isolated_home,
    )
    payload = json.loads(claim.path.read_text(encoding="utf-8"))
    payload["events"][0]["goal_continuation"] = False
    claim.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GoalContinuationClaimError):
        load_claims(home=isolated_home)


def test_loader_rejects_canonical_event_already_marked_consumed(isolated_home):
    """A corrupt consumed ledger cannot authorize replay of a retired head."""
    from gateway.goal_continuation_claims import (
        GoalContinuationClaimError,
        load_claims,
        publish_claim,
    )

    claim = publish_claim(
        "agent:main:discord:channel:consumed-overlap",
        "sid-consumed-overlap",
        [_continuation(_source(), message_id="consumed-overlap-head")],
        home=isolated_home,
    )
    payload = json.loads(claim.path.read_text(encoding="utf-8"))
    payload["consumed_event_ids"] = [payload["events"][0]["event_id"]]
    claim.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GoalContinuationClaimError):
        load_claims(home=isolated_home)


@pytest.mark.asyncio
async def test_conversation_boundary_stops_when_durable_retirement_fails(
    isolated_home, monkeypatch
):
    """Reset/clear cannot advance after a synthetic claim retirement failure."""
    from gateway import goal_continuation_claims as claims

    source = _source()
    key = "agent:main:discord:channel:retirement-failure"
    runner, adapter = _partial_runner(source, key, "sid-retirement-failure")
    head = _continuation(source)
    successor = _event("must remain queued", source=source)
    runner._session_state(key).conversation.queued_events.append(successor)
    retry = runner._claim_goal_continuation_retry(
        key, adapter, head, session_id="sid-retirement-failure"
    )
    assert retry is not None

    def fail_retirement(*_args, **_kwargs):
        raise claims.GoalContinuationClaimError("private raw storage failure")

    monkeypatch.setattr(claims, "retire_claim", fail_retirement)
    with pytest.raises(
        RuntimeError, match="durable goal continuation retirement is unavailable"
    ) as raised:
        runner._clear_conversation_scope(key, reason="test_reset")

    assert "private raw storage failure" not in str(raised.value)
    assert retry.dropped is False
    assert runner._session_state(key).conversation.queued_events == [successor]


@pytest.mark.asyncio
async def test_reset_preserves_durable_successors_across_new_session_binding(
    isolated_home,
):
    """A session reset drops synthetic authority without losing queued users."""
    source = _source()
    key = "agent:main:discord:channel:reset-successors"
    old_session_id = "sid-reset-old"
    runner, adapter = _partial_runner(source, key, old_session_id)
    head = _continuation(source, message_id="reset-head")
    first = _event("first after reset", source=source, message_id="reset-first")
    second = _event("second after reset", source=source, message_id="reset-second")
    adapter._pending_messages[key] = first
    runner._session_state(key).conversation.queued_events.append(second)
    retry = runner._claim_goal_continuation_retry(
        key, adapter, head, session_id=old_session_id
    )
    assert retry is not None

    runner._clear_conversation_scope(key, reason="session_reset")

    assert adapter._pending_messages[key] is first
    assert runner._session_state(key).conversation.queued_events == [second]
    restarted, restarted_adapter = _partial_runner(source, key, "sid-reset-new")
    assert restarted._recover_goal_continuation_claims(schedule=False) == 1
    assert restarted._goal_continuation_retries[key].event.message_id == "reset-first"
    assert restarted_adapter._pending_messages[key].message_id == "reset-second"


def test_completed_queue_cannot_cross_session_without_lifecycle_rebind(
    isolated_home,
):
    """Ordinary durable tails need an explicit lifecycle grant before rebinding."""
    from gateway.goal_continuation_claims import (
        GoalContinuationClaimError,
        complete_claim_event,
        event_claim_identity,
        publish_claim,
    )

    source = _source()
    key = "agent:main:discord:channel:no-implicit-rebind"
    head = _continuation(source, message_id="no-implicit-rebind-head")
    successor = _event(
        "ordinary tail", source=source, message_id="no-implicit-rebind-tail"
    )
    claim = publish_claim(
        key, "sid-no-implicit-rebind-old", [head, successor], home=isolated_home
    )
    head_identity = event_claim_identity(head)
    assert head_identity is not None
    assert complete_claim_event(
        key, claim.claim_id, head_identity[1], home=isolated_home
    )

    restarted, adapter = _partial_runner(
        source, key, "sid-no-implicit-rebind-new"
    )
    with pytest.raises(GoalContinuationClaimError):
        restarted._recover_goal_continuation_claims(schedule=False)
    assert restarted._goal_continuation_retries == {}
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_authorization_rejection_retires_recovered_head_before_successor(
    isolated_home,
):
    """An unauthorized recovered head is terminal before its tail can drain."""
    from gateway.goal_continuation_claims import load_claims, publish_claim

    source = _source()
    key = "agent:main:discord:channel:recovered-unauthorized"
    session_id = "sid-recovered-unauthorized"
    publish_claim(
        key,
        session_id,
        [
            _continuation(source, message_id="unauthorized-head"),
            _event("authorized tail", source=source, message_id="unauthorized-tail"),
        ],
        home=isolated_home,
    )
    runner, adapter = _partial_runner(source, key, session_id)
    assert runner._recover_goal_continuation_claims(schedule=False) == 1
    head = runner._goal_continuation_retries[key].event
    setattr(head, "_hermes_startup_restore_replay", True)
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._is_user_authorized_for_source = lambda _source: False

    assert await runner._handle_message(head) is None

    durable = load_claims(home=isolated_home)
    assert [event.message_id for event in durable[0].events] == [
        "unauthorized-tail"
    ]
    assert key not in runner._goal_continuation_retries
    assert adapter._pending_messages[key].message_id == "unauthorized-tail"


@pytest.mark.asyncio
async def test_global_pause_retires_recovered_head_before_successor(
    isolated_home, monkeypatch
):
    """Emergency pause rejects the head terminally before exposing its tail."""
    from agent import estop
    from gateway.goal_continuation_claims import load_claims, publish_claim

    source = _source()
    key = "agent:main:discord:channel:recovered-global-pause"
    session_id = "sid-recovered-global-pause"
    publish_claim(
        key,
        session_id,
        [
            _continuation(source, message_id="global-pause-head"),
            _event("tail after pause", source=source, message_id="global-pause-tail"),
        ],
        home=isolated_home,
    )
    runner, adapter = _partial_runner(source, key, session_id)
    assert runner._recover_goal_continuation_claims(schedule=False) == 1
    head = runner._goal_continuation_retries[key].event
    setattr(head, "_hermes_startup_restore_replay", True)
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._is_user_authorized_for_source = lambda _source: True
    runner._is_session_running = lambda _session_key: True
    monkeypatch.setattr(estop, "paused_reply", lambda: "gateway paused")

    assert await runner._handle_message(head) == "gateway paused"

    durable = load_claims(home=isolated_home)
    assert [event.message_id for event in durable[0].events] == [
        "global-pause-tail"
    ]
    assert key not in runner._goal_continuation_retries
    assert adapter._pending_messages[key].message_id == "global-pause-tail"


@pytest.mark.asyncio
async def test_pause_retires_durable_head_during_rearmed_owner_gap(isolated_home):
    """Pause cannot remove a restored head while leaving its durable claim live."""
    from gateway.goal_continuation_claims import load_claims

    source = _source()
    key = "agent:main:discord:channel:owner-gap-pause"
    runner, adapter = _partial_runner(source, key, "sid-owner-gap-pause")
    head = _continuation(source, message_id="owner-gap-pause-head")
    successor = _event(
        "owner gap successor", source=source, message_id="owner-gap-pause-tail"
    )
    runner._session_state(key).conversation.queued_events.append(successor)
    retry = runner._claim_goal_continuation_retry(
        key, adapter, head, session_id="sid-owner-gap-pause"
    )
    assert retry is not None
    runner._restore_dequeued_event_front(key, adapter, head)
    runner._finish_goal_continuation_retry(key, retry)
    assert key not in runner._goal_continuation_retries
    assert adapter._pending_messages[key] is head

    assert runner._clear_goal_pending_continuations(key, adapter) == 1

    durable = load_claims(home=isolated_home)
    assert [event.message_id for event in durable[0].events] == [
        "owner-gap-pause-tail"
    ]
    assert adapter._pending_messages[key] is successor


@pytest.mark.asyncio
async def test_plugin_rewrite_cannot_strip_recovered_head_ownership(
    isolated_home, monkeypatch
):
    """A pre-dispatch rewrite is terminal rather than orphaning durable identity."""
    from hermes_cli import lifecycle
    from gateway.goal_continuation_claims import load_claims, publish_claim

    source = _source()
    key = "agent:main:discord:channel:recovered-plugin-rewrite"
    session_id = "sid-recovered-plugin-rewrite"
    publish_claim(
        key,
        session_id,
        [
            _continuation(source, message_id="plugin-rewrite-head"),
            _event("tail after plugin", source=source, message_id="plugin-rewrite-tail"),
        ],
        home=isolated_home,
    )
    runner, adapter = _partial_runner(source, key, session_id)
    assert runner._recover_goal_continuation_claims(schedule=False) == 1
    head = runner._goal_continuation_retries[key].event
    setattr(head, "_hermes_startup_restore_replay", True)
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._is_user_authorized_for_source = lambda _source: False
    monkeypatch.setattr(
        lifecycle,
        "invoke_hook",
        lambda *_args, **_kwargs: [{"action": "rewrite", "text": "rewritten"}],
    )

    assert await runner._handle_message(head) is None

    durable = load_claims(home=isolated_home)
    assert [event.message_id for event in durable[0].events] == [
        "plugin-rewrite-tail"
    ]
    assert key not in runner._goal_continuation_retries
    assert adapter._pending_messages[key].message_id == "plugin-rewrite-tail"
