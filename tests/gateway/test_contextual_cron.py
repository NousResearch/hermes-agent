from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
import inspect
import logging
import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gateway.contextual_cron import (
    ContextualCronGateway,
    ContextualCronOutcome,
)
from gateway.session import Platform, SessionEntry, SessionSource, SessionStore
from gateway.run import GatewayRunner, TurnRunner


def test_delivery_authorizer_rejects_a_deleted_sealed_route():
    runner = object.__new__(GatewayRunner)
    cast(Any, runner).session_store = SimpleNamespace(
        peek_session_entry=lambda _key: None
    )
    cast(Any, runner)._is_user_authorized = lambda source: True

    assert runner._authorize_contextual_delivery_from_scheduler(
        {
            "origin": {
                "platform": "telegram",
                "chat_type": "dm",
                "chat_id": "42",
                "user_id": "42",
            },
            "_contextual_authority": {
                "execution_id": "execution",
                "session_key": "telegram:dm:42",
                "binding_version": 2,
                "route_instance_id": "route-instance-a",
                "session_id": "session-a",
                "routing_revision": 7,
            },
        }
    ) is False


def test_delivery_authorizer_marks_a_lost_locked_claim_as_attempted():
    runner = object.__new__(GatewayRunner)
    cast(Any, runner).session_store = SimpleNamespace(
        claim_contextual_delivery_authority=lambda *_a, **_k: (True, False)
    )
    cast(Any, runner)._is_user_authorized = lambda source: True
    target = {
        "origin": {
            "platform": "telegram",
            "chat_type": "dm",
            "chat_id": "42",
            "user_id": "42",
        },
        "_contextual_authority": {
            "execution_id": "execution",
            "session_key": "telegram:dm:42",
            "binding_version": 2,
            "route_instance_id": "route-instance-a",
            "session_id": "session-a",
            "routing_revision": 7,
        },
    }

    assert runner._authorize_contextual_delivery_from_scheduler(target) is True
    assert target["_contextual_delivery_claim_attempted"] is True
    assert "_contextual_delivery_claimed" not in target


def test_delivery_authority_claim_is_linearized_with_the_route():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_type="dm",
        chat_id="42",
        user_id="42",
    )
    now = datetime.now(timezone.utc)
    entry = SessionEntry(
        session_key="telegram:dm:42",
        session_id="session-a",
        route_instance_id="route-instance-a",
        routing_revision=7,
        created_at=now,
        updated_at=now,
        origin=source,
    )
    store = object.__new__(SessionStore)
    cast(Any, store)._lock = threading.RLock()
    cast(Any, store)._entries = {entry.session_key: entry}
    cast(Any, store)._ensure_loaded_locked = lambda: None
    events = []

    assert store.claim_contextual_delivery_authority(
        entry.session_key,
        source=source,
        binding_version=2,
        expected_route_instance_id=entry.route_instance_id,
        expected_session_id=entry.session_id,
        expected_routing_revision=entry.routing_revision,
        authorize=lambda: events.append("authorized") or True,
        claim=lambda: events.append("claimed") or True,
    ) == (True, True)
    assert events == ["authorized", "claimed"]

    events.clear()
    assert store.claim_contextual_delivery_authority(
        entry.session_key,
        source=source,
        binding_version=2,
        expected_route_instance_id="recreated-route",
        expected_session_id=entry.session_id,
        expected_routing_revision=entry.routing_revision,
        authorize=lambda: events.append("authorized") or True,
        claim=lambda: events.append("claimed") or True,
    ) == (False, False)
    assert events == []


class _LiveStore:
    def __init__(self, entry: SessionEntry, transcript: list[dict]):
        self.entry = entry
        self.transcript = transcript

    def peek_session_entry(self, session_key: str):
        if self.entry is None:
            return None
        return self.entry if session_key == self.entry.session_key else None

    async def load_transcript(self, session_id: str):
        assert session_id == self.entry.session_id
        return list(self.transcript)

    async def load_transcript_with_fence_strict(self, session_id: str):
        return await self.load_transcript(session_id), len(self.transcript)


async def _hold_contextual_turn_lease(runner, item):
    from gateway.turn_lease import SessionTurnLeaseRegistry

    registry = SessionTurnLeaseRegistry()
    token = await registry.acquire(
        item.admitted_session_id,
        owner_key=f"contextual-cron:{item.execution_id}",
        generation=0,
        timeout=1,
    )
    assert token is not None
    cast(Any, runner)._turn_leases = registry
    item.turn_lease_token = token
    return registry, token


class _TracerRunner:
    def __init__(self, store: _LiveStore):
        self.session_store = store
        self.async_session_store = store
        self.seen_histories: list[list[dict]] = []

    def _is_user_authorized(self, source: SessionSource) -> bool:
        return True

    def _contextual_cron_session_busy(self, session_key: str) -> bool:
        return False

    async def _run_contextual_cron_turn(self, item, entry, history):
        self.seen_histories.append(list(history))
        chosen = next(
            (
                message["content"].split("chosen word is ", 1)[1].rstrip(".")
                for message in history
                if message.get("role") == "user"
                and "chosen word is " in str(message.get("content"))
            ),
            "unknown",
        )
        return ContextualCronOutcome.notify(f"Recovered chosen word: {chosen}")


def test_contextual_error_and_already_sent_paths_are_transport_fail_closed():
    source = inspect.getsource(GatewayRunner._handle_message_with_agent)
    assert (
        "if not _is_contextual_cron:\n            self._bind_adapter_run_generation("
        in source
    )
    assert (
        'not _is_contextual_cron\n                and agent_result.get("already_sent")'
        in source
    )
    assert "_err_adapter = (\n                    None if _is_contextual_cron" in source
    assert "_stts_adapter = (\n                None if _is_contextual_cron" in source
    assert "_apply_contextual_transcript_visibility(" in source
    assert "_get_contextual_turn_authority" in source
    preprocessing = source.split("# V1 contextual prompts", maxsplit=1)[1].split(
        "if message_text is None", maxsplit=1
    )[0]
    assert 'message_text = event.text or ""' in preprocessing
    assert "else:" in preprocessing
    assert "self._prepare_profile_scoped_inbound_message_text(" in preprocessing
    assert "_get_contextual_turn_authority" in source
    assert (
        "not _is_contextual_cron\n            and not history\n"
        "            and not await self.async_session_store.has_any_sessions()"
        in source
    )
    home_notice_guard = source.split(
        "# One-time prompt if no home channel is set", maxsplit=1
    )[1].split("platform_name =", maxsplit=1)[0]
    assert "not _is_contextual_cron" in home_notice_guard
    assert "await self._deliver_platform_notice" in source
    assert "_contextual_authority.admitted_routing_revision" in source
    assert "session_entry.routing_revision != _admitted_routing_revision" in source
    assert (
        "if (\n                _is_contextual_cron\n"
        "                and agent_result.get(\"session_id\")\n"
        "                and agent_result[\"session_id\"] != session_entry.session_id"
        in source
    )

    run_source = inspect.getsource(TurnRunner.run_sync)
    for callback in (
        "tool_progress_callback",
        "tool_start_callback",
        "stream_delta_callback",
        "interim_assistant_callback",
        "status_callback",
    ):
        assert f"agent.{callback} = (None if ctx.suppress_output else" in run_source
    assert (
        "agent._gateway_turn_context_notes = (\n"
        "            \"\"\n"
        "            if ctx.suppress_output"
        in run_source
    )
    assert (
        "_native_imgs = (\n"
        "                []\n"
        "                if ctx.suppress_output"
        in run_source
    )
    assert "if not ctx.suppress_output:\n                unregister_gateway_notify" in run_source
    assert "if not ctx.suppress_output:\n                try:\n                    from tools.clarify_gateway" in run_source
    assert "if not ctx.suppress_output and ctx._status_adapter and ctx.session_key:" in run_source
    assert "agent.thinking_progress = False if ctx.suppress_output else ctx._thinking_enabled" in run_source
    assert "contextual_execution=bool(ctx.suppress_output)" in run_source
    assert "api_mode == \"codex_app_server\"" in run_source
    assert "fallback_model=self._runner._refresh_fallback_model()" in run_source
    from agent import agent_init

    init_source = inspect.getsource(agent_init.init_agent)
    assert "if agent._contextual_execution:" in init_source
    assert "fallback_model = None" in init_source
    assert "Contextual execution attempted to rotate its admitted session" in run_source


def test_contextual_model_boundary_bypasses_extension_planes():
    from agent import conversation_loop

    class _Engine:
        def __init__(self):
            self.called = False
            self.context_length = 100

        def select_context(self, *args, **kwargs):
            self.called = True
            return []

        def on_turn_complete(self, *args, **kwargs):
            self.called = True

    engine = _Engine()
    agent = SimpleNamespace(
        _contextual_execution=True,
        context_compressor=engine,
        session_id="session-1",
    )
    messages = [{"role": "user", "content": "hello"}]
    selected = conversation_loop._apply_context_engine_selection(
        agent,
        messages,
        messages,
        messages[-1],
        logger=logging.getLogger(__name__),
    )
    conversation_loop._notify_context_engine_turn_complete(
        agent,
        messages,
        logger=logging.getLogger(__name__),
    )

    assert selected is messages
    assert engine.called is False
    source = inspect.getsource(conversation_loop.run_conversation)
    assert "_use_streaming = not _contextual_execution" in source
    assert "if _contextual_execution:\n                        return agent._interruptible_api_call" in source
    assert "if _contextual_execution:\n                        response = _perform_api_call" in source


def test_zero_routing_revision_preserves_legacy_session_record_shape():
    now = datetime.now(timezone.utc)
    entry = SessionEntry(
        session_key="telegram:u:c",
        session_id="session-1",
        created_at=now,
        updated_at=now,
    )

    assert "routing_revision" not in entry.to_dict()
    assert SessionEntry.from_dict(entry.to_dict()).routing_revision == 0

    entry.routing_revision = 1
    assert entry.to_dict()["routing_revision"] == 1


def test_route_instance_identity_round_trips():
    now = datetime.now(timezone.utc)
    entry = SessionEntry(
        session_key="telegram:u:c",
        session_id="session-1",
        created_at=now,
        updated_at=now,
        route_instance_id="route-instance-a",
    )

    stored = entry.to_dict()
    assert stored["route_instance_id"] == "route-instance-a"
    assert SessionEntry.from_dict(stored).route_instance_id == "route-instance-a"

    legacy = dict(stored)
    legacy.pop("route_instance_id")
    first = SessionEntry.from_dict(legacy).route_instance_id
    second = SessionEntry.from_dict(legacy).route_instance_id
    assert first
    assert first == second


@pytest.mark.asyncio
async def test_contextual_cron_tracer_recovers_only_the_stable_session_context():
    """The opt-in lane sees live history; the legacy isolated control does not."""

    session_key = "telegram:dm:42:42"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="42",
        chat_type="dm",
        user_id="42",
    )
    now = datetime.now(timezone.utc)
    entry = SessionEntry(
        session_id="session-marzipan",
        session_key=session_key,
        created_at=now,
        updated_at=now,
        origin=source,
    )
    store = _LiveStore(
        entry,
        [
            {"role": "user", "content": "The chosen word is marzipan."},
            {"role": "assistant", "content": "I will remember it."},
        ],
    )
    runner = _TracerRunner(store)
    sealed: list[tuple[str, str, str]] = []
    gateway = ContextualCronGateway(
        runner,
        seal_admission=lambda execution_id, key, session_id: sealed.append(
            (execution_id, key, session_id)
        )
        or True,
    )

    contextual = await gateway.dispatch(
        {
            "id": "contextual",
            "prompt": "What was the chosen word?",
            "session_target": "current",
            "session_key": session_key,
        },
        execution_id="execution-1",
    )

    # Deterministic isolated control: the existing isolated lane starts with an
    # empty history, so the same tracer cannot recover the seeded value.
    isolated = await runner._run_contextual_cron_turn(
        SimpleNamespace(prompt="What was the chosen word?"),
        entry,
        [],
    )

    assert contextual.kind == "notify"
    assert "marzipan" in contextual.final_response
    assert "marzipan" not in isolated.final_response
    assert sealed == [("execution-1", session_key, "session-marzipan")]
    assert runner.seen_histories[0] == store.transcript


@pytest.mark.asyncio
async def test_contextual_turn_fails_closed_when_fenced_transcript_load_fails():
    class BrokenStore(_LiveStore):
        async def load_transcript_with_fence_strict(self, session_id: str):
            del session_id
            raise RuntimeError("session db unavailable")

    store = BrokenStore(_entry(), [{"role": "user", "content": "secret"}])
    runner = _QueueRunner(store)
    finished = []
    gateway = _queue_gateway(runner, [], finished)

    outcome = await gateway.dispatch(
        {
            "id": "strict-load",
            "prompt": "must not run",
            "session_target": "current",
            "session_key": store.entry.session_key,
        },
        execution_id="strict-load",
    )

    assert outcome.kind == "failure"
    assert "session db unavailable" in str(outcome.error)
    assert runner.started == []
    assert finished == [("strict-load", "failure")]


@pytest.mark.asyncio
async def test_contextual_turn_rejects_proxy_before_any_execution(monkeypatch):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    monkeypatch.setattr(runner, "_get_proxy_url", lambda: "http://proxy.invalid")
    item = SimpleNamespace(
        execution_id="proxy-exec",
        admitted_session_id="session-1",
        session_key="telegram:dm:42:42",
        prompt="do not send",
    )

    outcome = await runner._run_contextual_cron_turn(item, _entry(), [])

    assert outcome.kind == "rejected"
    assert "proxy" in str(outcome.error).lower()


class _QueueRunner(_TracerRunner):
    def __init__(self, store):
        super().__init__(store)
        self.busy = False
        self.authorized = True
        self.started: list[str] = []
        self.release_first = __import__("asyncio").Event()

    def _is_user_authorized(self, source):
        return self.authorized

    def _contextual_cron_session_busy(self, session_key):
        return self.busy

    async def _run_contextual_cron_turn(self, item, entry, history):
        self.started.append(item.execution_id)
        if item.execution_id == "first":
            await self.release_first.wait()
        return ContextualCronOutcome.notify(item.execution_id)


def _entry(session_id="session-1"):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="42",
        chat_type="dm",
        user_id="42",
    )
    now = datetime.now(timezone.utc)
    return SessionEntry(
        session_id=session_id,
        session_key="telegram:dm:42:42",
        created_at=now,
        updated_at=now,
        origin=source,
    )


def _queue_gateway(runner, sealed, finished=None):
    return ContextualCronGateway(
        runner,
        busy_poll_seconds=0.001,
        seal_admission=lambda execution_id, key, sid: sealed.append(
            (execution_id, key, sid)
        )
        or True,
        finish_admission=(
            (lambda execution_id, outcome: finished.append((execution_id, outcome.kind)))
            if finished is not None
            else None
        ),
    )


class _OutboxRunner(_QueueRunner):
    def __init__(self, store):
        super().__init__(store)
        self.events: list[str] = []
        self.fail_apply = False

    async def _run_contextual_cron_turn(self, item, entry, history):
        self.events.append("run")
        item.transcript_session_id = entry.session_id
        item.transcript_entries = [
            {
                "role": "user",
                "content": "hidden trigger",
                "display_kind": "hidden",
                "message_id": f"contextual-cron:{item.execution_id}:0",
            }
        ]
        return ContextualCronOutcome.no_action()

    async def _apply_contextual_cron_transcript(self, item):
        self.events.append("apply")
        if self.fail_apply:
            self.fail_apply = False
            raise RuntimeError("session store unavailable")


@pytest.mark.asyncio
async def test_contextual_queue_reauthorizes_after_model_before_commit():
    import asyncio

    runner = _QueueRunner(_LiveStore(_entry(), []))
    finished = []
    gateway = _queue_gateway(runner, [], finished)
    pending = asyncio.create_task(
        gateway.dispatch(
            {
                "id": "job",
                "prompt": "check",
                "session_target": "current",
                "session_key": "telegram:dm:42:42",
            },
            execution_id="first",
        )
    )
    while runner.started != ["first"]:
        await asyncio.sleep(0)
    runner.authorized = False
    runner.release_first.set()

    outcome = await pending

    assert outcome.kind == "rejected"
    assert "authorization" in str(outcome.error).lower()
    assert finished == [("first", "rejected")]


@pytest.mark.asyncio
async def test_post_model_authorization_error_resolves_item_and_lane_continues():
    import asyncio

    runner = _QueueRunner(_LiveStore(_entry(), []))
    finished = []
    raised = False

    def authorize(source):
        del source
        nonlocal raised
        if runner.release_first.is_set() and runner.started == ["first"] and not raised:
            raised = True
            raise RuntimeError("auth backend unavailable")
        return True

    runner._is_user_authorized = authorize
    gateway = _queue_gateway(runner, [], finished)
    first_task = asyncio.create_task(
        gateway.dispatch(
            {
                "id": "job",
                "prompt": "continue",
                "session_target": "current",
                "session_key": "telegram:dm:42:42",
            },
            execution_id="first",
        )
    )
    while runner.started != ["first"]:
        await asyncio.sleep(0)
    runner.release_first.set()

    first = await asyncio.wait_for(first_task, timeout=1)
    second = await asyncio.wait_for(
        gateway.dispatch(
            {
                "id": "job",
                "prompt": "continue again",
                "session_target": "current",
                "session_key": "telegram:dm:42:42",
            },
            execution_id="second",
        ),
        timeout=1,
    )

    assert first.kind == "retryable"
    assert "auth backend unavailable" in str(first.error)
    assert second.kind == "notify"
    assert runner.started == ["first", "second"]
    assert finished == [("first", "retryable"), ("second", "notify")]


@pytest.mark.asyncio
async def test_late_authorization_error_defers_outbox_and_releases_lane():
    import asyncio

    runner = _OutboxRunner(_LiveStore(_entry(), []))
    late_failure = True
    auth_calls = 0

    def authorize(source):
        del source
        nonlocal auth_calls
        auth_calls += 1
        if late_failure and auth_calls >= 4:
            raise RuntimeError("late auth backend unavailable")
        return True

    runner._is_user_authorized = authorize
    gateway = ContextualCronGateway(
        runner,
        busy_poll_seconds=0.001,
        transcript_retry_seconds=0.001,
        seal_admission=lambda *_: True,
        finish_admission=lambda *_: {"transcript_state": "pending"},
    )

    first_task = asyncio.create_task(
        gateway.dispatch(
            {
                "id": "job",
                "prompt": "first",
                "session_target": "current",
                "session_key": "telegram:dm:42:42",
            },
            execution_id="late-auth-first",
        )
    )
    first = await asyncio.wait_for(first_task, timeout=0.5)
    late_failure = False
    second = await asyncio.wait_for(
        gateway.dispatch(
            {
                "id": "job",
                "prompt": "second",
                "session_target": "current",
                "session_key": "telegram:dm:42:42",
            },
            execution_id="late-auth-second",
        ),
        timeout=0.5,
    )

    assert first.kind == "unknown"
    assert "authorization" in str(first.error).lower()
    assert second.kind == "no_action"
    assert runner.events.count("run") == 2


@pytest.mark.asyncio
async def test_contextual_queue_stages_outbox_before_applying_transcript():
    runner = _OutboxRunner(_LiveStore(_entry(), []))

    def finish(execution_id, outcome, item):
        assert execution_id == "outbox-order"
        assert outcome.kind == "no_action"
        assert item.transcript_session_id == "session-1"
        assert item.transcript_entries[0]["display_kind"] == "hidden"
        runner.events.append("persist")
        return {"transcript_state": "pending"}

    gateway = ContextualCronGateway(
        runner,
        seal_admission=lambda *_args: True,
        finish_admission=finish,
    )
    outcome = await gateway.dispatch(
        {
            "id": "job",
            "prompt": "check",
            "session_target": "current",
            "session_key": "telegram:dm:42:42",
        },
        execution_id="outbox-order",
    )

    assert outcome.kind == "no_action"
    assert runner.events == ["run", "persist", "apply"]


@pytest.mark.asyncio
async def test_contextual_queue_retries_durable_outbox_before_resolving():
    runner = _OutboxRunner(_LiveStore(_entry(), []))
    runner.fail_apply = True

    def finish(_execution_id, _outcome, _item):
        runner.events.append("persist")
        return {"transcript_state": "pending"}

    gateway = ContextualCronGateway(
        runner,
        seal_admission=lambda *_args: True,
        finish_admission=finish,
        transcript_retry_seconds=0.001,
    )
    outcome = await gateway.dispatch(
        {
            "id": "job",
            "prompt": "check",
            "session_target": "current",
            "session_key": "telegram:dm:42:42",
        },
        execution_id="outbox-recovery",
    )

    assert outcome.kind == "no_action"
    assert runner.events == ["run", "persist", "apply", "apply"]


@pytest.mark.asyncio
async def test_contextual_queue_terminalizes_causal_conflict_without_retrying(
    monkeypatch,
):
    import asyncio
    import threading

    from gateway.contextual_cron import ContextualCronTranscriptConflict

    class ConflictRunner(_OutboxRunner):
        async def _apply_contextual_cron_transcript(self, item):
            del item
            self.events.append("apply")
            raise ContextualCronTranscriptConflict(
                "Contextual transcript advanced before outbox application."
            )

    runner = ConflictRunner(_LiveStore(_entry(), []))
    marked = []
    marked_signal = threading.Event()

    def mark_conflict(execution_id, *, error):
        marked.append((execution_id, error))
        marked_signal.set()
        return True

    monkeypatch.setattr(
        "cron.executions.mark_contextual_transcript_conflict",
        mark_conflict,
    )

    def finish(_execution_id, _outcome, _item):
        runner.events.append("persist")
        return {"transcript_state": "pending"}

    gateway = ContextualCronGateway(
        runner,
        seal_admission=lambda *_args: True,
        finish_admission=finish,
        transcript_retry_seconds=0.001,
    )
    dispatch = asyncio.create_task(
        gateway.dispatch(
            {
                "id": "job",
                "prompt": "check",
                "session_target": "current",
                "session_key": "telegram:dm:42:42",
            },
            execution_id="outbox-conflict",
        )
    )
    await asyncio.wait_for(asyncio.to_thread(marked_signal.wait, 1), timeout=1.1)
    try:
        outcome = await asyncio.wait_for(dispatch, timeout=1)
        assert outcome.kind == "unknown"
        assert marked == [
            (
                "outbox-conflict",
                "Contextual transcript advanced before outbox application.",
            )
        ]
        assert runner.events == ["run", "persist", "apply"]
    finally:
        drainer = gateway._drainers.get("telegram:dm:42:42")
        if drainer is not None and not drainer.done():
            drainer.cancel()
            await asyncio.gather(drainer, return_exceptions=True)
        if not dispatch.done():
            dispatch.cancel()
            await asyncio.gather(dispatch, return_exceptions=True)


@pytest.mark.asyncio
async def test_contextual_transcript_entries_stage_without_touching_session_store():
    class Store:
        def __init__(self):
            self.appended = []

        async def append_to_transcript(self, *args, **kwargs):
            self.appended.append((args, kwargs))

    runner = object.__new__(GatewayRunner)
    runner._async_session_store = Store()
    event = SimpleNamespace(metadata={})

    await GatewayRunner._write_or_stage_transcript_entry(
        runner,
        event,
        "session-1",
        {"role": "user", "content": "hidden trigger", "display_kind": "hidden"},
        skip_db=False,
        contextual_execution_id="stage-only",
    )
    await GatewayRunner._write_or_stage_transcript_entry(
        runner,
        event,
        "session-1",
        {"role": "assistant", "content": "answer"},
        skip_db=False,
        contextual_execution_id="stage-only",
    )

    assert runner._async_session_store.appended == []
    assert event.metadata["contextual_cron_transcript_session_id"] == "session-1"
    assert [
        entry["message_id"]
        for entry in event.metadata["contextual_cron_transcript_entries"]
    ] == ["contextual-cron:stage-only:0", "contextual-cron:stage-only:1"]


@pytest.mark.asyncio
async def test_ordinary_transcript_entry_keeps_direct_persistence_path():
    class Store:
        def __init__(self):
            self.appended = []

        async def append_to_transcript(self, *args, **kwargs):
            self.appended.append((args, kwargs))

    runner = object.__new__(GatewayRunner)
    runner._async_session_store = Store()
    event = SimpleNamespace(metadata={})
    entry = {"role": "user", "content": "hello", "message_id": "platform-1"}

    await GatewayRunner._write_or_stage_transcript_entry(
        runner,
        event,
        "session-1",
        entry,
        skip_db=True,
        contextual_execution_id=None,
    )

    assert runner._async_session_store.appended == [
        (("session-1", entry), {"skip_db": True})
    ]
    assert event.metadata == {}


@pytest.mark.asyncio
async def test_contextual_transcript_outbox_applies_idempotently_before_ack(
    monkeypatch,
):
    from gateway.contextual_cron import ContextualCronQueueItem

    class Store:
        def __init__(self):
            self.persisted = {"contextual-cron:outbox-apply:0"}
            self.appended = []
            self.history = [
                {
                    "role": "user",
                    "content": "hidden trigger",
                    "display_kind": "hidden",
                    "message_id": "contextual-cron:outbox-apply:0",
                }
            ]

        async def load_transcript_strict(self, _session_id):
            return [dict(entry) for entry in self.history]

        async def has_platform_message_id(self, _session_id, message_id):
            return message_id in self.persisted

        async def append_to_transcript(self, session_id, entry, *, skip_db=False):
            assert session_id == "session-1"
            assert skip_db is False
            self.appended.append(dict(entry))
            self.persisted.add(entry["message_id"])
            self.history.append(dict(entry))

        async def update_session(self, session_key, **updates):
            assert session_key == "telegram:dm:42:42"
            assert updates == {"last_prompt_tokens": 77, "touch_activity": False}

    store = Store()
    runner = object.__new__(GatewayRunner)
    runner._async_session_store = store
    applied = []
    monkeypatch.setattr(
        "cron.executions.mark_contextual_transcript_applied",
        lambda execution_id: applied.append(execution_id) or True,
    )
    loop = __import__("asyncio").get_running_loop()
    item = ContextualCronQueueItem(
        job_id="job",
        execution_id="outbox-apply",
        prompt="check",
        session_key="telegram:dm:42:42",
        admitted_session_id="session-1",
        source=_entry().origin,
        future=loop.create_future(),
        transcript_session_id="session-1",
        transcript_entries=[
            {
                "role": "user",
                "content": "hidden trigger",
                "display_kind": "hidden",
                "message_id": "contextual-cron:outbox-apply:0",
            },
            {
                "role": "assistant",
                "content": "answer",
                "message_id": "contextual-cron:outbox-apply:1",
            },
        ],
        transcript_base_message_count=0,
        transcript_base_revision=0,
        last_prompt_tokens=77,
    )

    await GatewayRunner._apply_contextual_cron_transcript(runner, item)

    assert [entry["message_id"] for entry in store.appended] == [
        "contextual-cron:outbox-apply:1"
    ]
    assert applied == ["outbox-apply"]


@pytest.mark.asyncio
async def test_contextual_transcript_outbox_stays_pending_after_partial_apply(
    monkeypatch,
):
    from gateway.contextual_cron import ContextualCronQueueItem

    class FailingStore:
        async def load_transcript_strict(self, _session_id):
            return []

        async def has_platform_message_id(self, _session_id, _message_id):
            return False

        async def append_to_transcript(self, *_args, **_kwargs):
            raise RuntimeError("session db unavailable")

    runner = object.__new__(GatewayRunner)
    runner._async_session_store = FailingStore()
    applied = []
    monkeypatch.setattr(
        "cron.executions.mark_contextual_transcript_applied",
        lambda execution_id: applied.append(execution_id) or True,
    )
    loop = __import__("asyncio").get_running_loop()
    item = ContextualCronQueueItem(
        job_id="job",
        execution_id="outbox-pending",
        prompt="check",
        session_key="telegram:dm:42:42",
        admitted_session_id="session-1",
        source=_entry().origin,
        future=loop.create_future(),
        transcript_session_id="session-1",
        transcript_entries=[
            {
                "role": "user",
                "content": "hidden trigger",
                "display_kind": "hidden",
                "message_id": "contextual-cron:outbox-pending:0",
            }
        ],
        transcript_base_message_count=0,
        transcript_base_revision=0,
    )

    with pytest.raises(RuntimeError, match="session db unavailable"):
        await GatewayRunner._apply_contextual_cron_transcript(runner, item)

    assert applied == []


def test_contextual_transcript_persistence_preserves_hidden_display_metadata():
    from gateway.session import SessionStore

    captured = []

    class DB:
        def append_message(self, **kwargs):
            captured.append(kwargs)
            return 1

    store = object.__new__(SessionStore)
    cast(Any, store)._db = DB()
    store._append_transcript_message(
        "session-1",
        {
            "role": "user",
            "content": "hidden prompt",
            "message_id": "contextual-cron:hidden:0",
            "display_kind": "hidden",
            "display_metadata": {"producer": "contextual_cron"},
        },
    )

    assert captured[0]["display_kind"] == "hidden"
    assert captured[0]["display_metadata"] == {"producer": "contextual_cron"}


@pytest.mark.asyncio
async def test_contextual_transcript_apply_rejects_a_later_human_turn(
    monkeypatch,
):
    from gateway.contextual_cron import (
        ContextualCronQueueItem,
        ContextualCronTranscriptConflict,
    )

    class AdvancedStore:
        def __init__(self):
            self.messages = [
                {"role": "user", "content": "later human turn", "message_id": "human:1"}
            ]
            self.appended = []

        async def load_transcript_strict(self, _session_id):
            return list(self.messages)

        async def has_platform_message_id_strict(self, _session_id, candidate):
            return any(row.get("message_id") == candidate for row in self.messages)

        async def append_to_transcript(self, _session_id, entry, *, skip_db=False):
            self.appended.append(dict(entry))
            self.messages.append(dict(entry))

    store = AdvancedStore()
    runner = object.__new__(GatewayRunner)
    runner._async_session_store = store
    monkeypatch.setattr(
        "cron.executions.mark_contextual_transcript_applied", lambda _execution_id: True
    )
    loop = __import__("asyncio").get_running_loop()
    item = ContextualCronQueueItem(
        job_id="job",
        execution_id="causal-conflict",
        prompt="check",
        session_key="telegram:dm:42:42",
        admitted_session_id="session-1",
        source=_entry().origin,
        future=loop.create_future(),
        transcript_session_id="session-1",
        transcript_entries=[
            {
                "role": "assistant",
                "content": "scheduled answer",
                "message_id": "contextual-cron:causal-conflict:0",
            }
        ],
        transcript_base_message_count=0,
        transcript_base_revision=0,
    )

    with pytest.raises(ContextualCronTranscriptConflict, match="advanced"):
        await GatewayRunner._apply_contextual_cron_transcript(runner, item)

    assert store.appended == []
    assert store.messages == [
        {"role": "user", "content": "later human turn", "message_id": "human:1"}
    ]


@pytest.mark.asyncio
async def test_concurrent_transcript_recovery_has_one_application_winner(
    monkeypatch, tmp_path
):
    import asyncio
    import cron.executions as executions
    from gateway.contextual_cron import ContextualCronQueueItem

    monkeypatch.setattr(
        executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db"
    )
    record = executions.create_execution("job", source="builtin")
    executions.mark_execution_running(record["id"])
    message_id = f"contextual-cron:{record['id']}:0"
    entries = [
        {
            "role": "user",
            "content": "hidden",
            "display_kind": "hidden",
            "message_id": message_id,
        }
    ]
    executions.persist_contextual_agent_result(
        record["id"],
        outcome="no_action",
        transcript_session_id="session-1",
        transcript_entries=entries,
        transcript_base_message_count=0,
        transcript_base_revision=0,
    )

    class RacingStore:
        def __init__(self):
            self.persisted = set()
            self.history = []
            self.append_calls = 0
            self.first_append_entered = asyncio.Event()
            self.release_first_append = asyncio.Event()

        async def load_transcript_strict(self, _session_id):
            return [dict(entry) for entry in self.history]

        async def has_platform_message_id_strict(self, _session_id, candidate):
            return candidate in self.persisted

        async def append_to_transcript(self, _session_id, entry, *, skip_db=False):
            assert skip_db is False
            self.append_calls += 1
            if self.append_calls == 1:
                self.first_append_entered.set()
                await self.release_first_append.wait()
            self.persisted.add(entry["message_id"])
            self.history.append(dict(entry))

    store = RacingStore()
    runner = object.__new__(GatewayRunner)
    runner._async_session_store = store
    loop = asyncio.get_running_loop()
    item = ContextualCronQueueItem(
        job_id="job",
        execution_id=record["id"],
        prompt="check",
        session_key="telegram:dm:42:42",
        admitted_session_id="session-1",
        source=_entry().origin,
        future=loop.create_future(),
        transcript_session_id="session-1",
        transcript_entries=entries,
        transcript_base_message_count=0,
        transcript_base_revision=0,
    )

    first = asyncio.create_task(
        GatewayRunner._apply_contextual_cron_transcript(runner, item)
    )
    await asyncio.wait_for(store.first_append_entered.wait(), timeout=1)
    second = asyncio.create_task(
        GatewayRunner._apply_contextual_cron_transcript(runner, item)
    )
    await asyncio.sleep(0.05)
    assert store.append_calls == 1
    store.release_first_append.set()
    await asyncio.gather(first, second)

    assert store.append_calls == 1
    applied_record = executions.get_execution(record["id"])
    assert applied_record is not None
    assert applied_record["transcript_state"] == "applied"


def test_cross_process_transcript_recovery_has_one_append_winner(
    monkeypatch, tmp_path
):
    import multiprocessing
    import os
    import sqlite3
    import time
    from pathlib import Path
    from types import SimpleNamespace

    import cron.executions as executions

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    executions.EXECUTIONS_FILE = tmp_path / "executions.db"
    record = executions.create_execution(
        "job", source="builtin", requires_job_accounting=True
    )
    assert executions.seal_contextual_admission(
        record["id"],
        session_key="telegram:1",
        admitted_session_id="session-1",
    ) is True
    executions.persist_contextual_agent_result(
        record["id"],
        outcome="notify",
        final_response="done",
        transcript_session_id="session-1",
        transcript_entries=[
            {
                "role": "assistant",
                "content": "done",
                "message_id": f"contextual-cron:{record['id']}:0",
                "display_kind": "normal",
            }
        ],
        transcript_base_message_count=0,
        transcript_base_revision=0,
    )

    transcript_db = tmp_path / "transcript.db"
    with sqlite3.connect(transcript_db) as conn:
        conn.execute("CREATE TABLE messages (message_id TEXT NOT NULL)")

    gate = multiprocessing.get_context("fork").Event()
    ready = multiprocessing.get_context("fork").Queue()

    def apply_in_process() -> None:
        os.environ["HERMES_HOME"] = str(tmp_path / "home")
        executions.EXECUTIONS_FILE = Path(tmp_path / "executions.db")

        class Store:
            def load_transcript_strict(self, _session_id):
                with sqlite3.connect(transcript_db) as conn:
                    return [
                        {"message_id": row[0]}
                        for row in conn.execute(
                            "SELECT message_id FROM messages ORDER BY rowid"
                        )
                    ]

            def has_platform_message_id_strict(self, _session_id, message_id):
                with sqlite3.connect(transcript_db) as conn:
                    return conn.execute(
                        "SELECT 1 FROM messages WHERE message_id=? LIMIT 1",
                        (message_id,),
                    ).fetchone() is not None

            def append_contextual_transcript_message_once(self, _session_id, entry):
                # Widen the unprotected race window: without the process/file
                # lock both workers observe absence and both append.
                time.sleep(0.1)
                with sqlite3.connect(transcript_db) as conn:
                    conn.execute(
                        "INSERT INTO messages(message_id) VALUES (?)",
                        (entry["message_id"],),
                    )

            def update_session(self, *_args, **_kwargs):
                return None

        item = SimpleNamespace(
            execution_id=record["id"],
            admitted_session_id="session-1",
            transcript_session_id="session-1",
            transcript_entries=[
                {
                    "role": "assistant",
                    "content": "done",
                    "message_id": f"contextual-cron:{record['id']}:0",
                    "display_kind": "normal",
                }
            ],
            transcript_base_message_count=0,
            transcript_base_revision=0,
            last_prompt_tokens=None,
            session_key="telegram:1",
        )
        ready.put(True)
        gate.wait(timeout=5)
        runner = object.__new__(GatewayRunner)
        GatewayRunner._apply_contextual_cron_transcript_sync(runner, Store(), item)

    ctx = multiprocessing.get_context("fork")
    workers = [ctx.Process(target=apply_in_process) for _ in range(2)]
    for worker in workers:
        worker.start()
    assert ready.get(timeout=5) is True
    assert ready.get(timeout=5) is True
    gate.set()
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0

    with sqlite3.connect(transcript_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    applied_record = executions.get_execution(record["id"])
    assert applied_record is not None
    assert applied_record["transcript_state"] == "applied"


@pytest.mark.asyncio
async def test_gateway_startup_replays_every_valid_pending_contextual_transcript(
    monkeypatch,
):
    records = [
        {
            "id": "malformed",
            "job_id": "job-0",
            "session_key": "telegram:dm:42:42",
            "admitted_session_id": "session-1",
            "transcript_session_id": "session-1",
            "transcript_json": "not-json",
            "transcript_base_message_count": None,
            "transcript_base_revision": None,
            "transcript_last_prompt_tokens": 1,
        },
        {
            "id": "recover-1",
            "job_id": "job-1",
            "session_key": "telegram:dm:42:42",
            "admitted_session_id": "session-1",
            "transcript_session_id": "session-1",
            "transcript_json": '[{"role":"user","content":"hidden",'
            '"display_kind":"hidden",'
            '"message_id":"contextual-cron:recover-1:0"}]',
            "transcript_base_message_count": 0,
            "transcript_base_revision": 0,
            "transcript_last_prompt_tokens": 42,
        },
    ]
    monkeypatch.setattr(
        "cron.executions.list_pending_contextual_transcripts", lambda: records
    )
    runner = object.__new__(GatewayRunner)
    cast(Any, runner).session_store = _LiveStore(_entry(), [])
    cast(Any, runner)._is_user_authorized = lambda _source: True
    seen = []
    conflicts = []
    monkeypatch.setattr(
        "cron.executions.mark_contextual_transcript_conflict",
        lambda execution_id, *, error: conflicts.append((execution_id, error)) or True,
    )

    async def apply(item):
        seen.append(item)

    runner._apply_contextual_cron_transcript = apply

    recovered = await GatewayRunner._recover_contextual_cron_transcripts(runner)

    assert recovered == 1
    assert [item.execution_id for item in seen] == ["recover-1"]
    assert seen[0].transcript_session_id == "session-1"
    assert seen[0].transcript_base_message_count == 0
    assert seen[0].last_prompt_tokens == 42
    assert conflicts == [
        ("malformed", "Contextual transcript outbox payload is invalid.")
    ]


@pytest.mark.asyncio
async def test_recovery_route_revision_change_terminalizes_unapplied_outbox(monkeypatch):
    record = {
        "id": "recover-stale-route",
        "job_id": "job-1",
        "session_key": "telegram:dm:42:42",
        "admitted_session_id": "session-1",
        "admitted_routing_revision": 1,
        "transcript_session_id": "session-1",
        "transcript_json": '[{"role":"assistant","content":"scheduled",'
        '"message_id":"contextual-cron:recover-stale-route:0"}]',
        "transcript_base_message_count": 0,
        "transcript_base_revision": 0,
        "transcript_last_prompt_tokens": None,
    }
    monkeypatch.setattr(
        "cron.executions.list_pending_contextual_transcripts", lambda: [record]
    )
    conflicts = []
    monkeypatch.setattr(
        "cron.executions.mark_contextual_transcript_conflict",
        lambda execution_id, *, error: conflicts.append((execution_id, error)) or True,
    )
    entry = replace(_entry(), routing_revision=2)
    runner = object.__new__(GatewayRunner)
    cast(Any, runner).session_store = _LiveStore(entry, [])
    applied = []

    async def apply(item):
        applied.append(item)

    cast(Any, runner)._apply_contextual_cron_transcript = apply

    recovered = await GatewayRunner._recover_contextual_cron_transcripts(runner)

    assert recovered == 0
    assert applied == []
    assert conflicts and conflicts[0][0] == "recover-stale-route"
    assert "route changed" in conflicts[0][1]


@pytest.mark.asyncio
async def test_recovery_route_instance_change_terminalizes_unapplied_outbox(
    monkeypatch,
):
    record = {
        "id": "recover-route-aba",
        "job_id": "job-1",
        "session_key": "telegram:dm:42:42",
        "admitted_route_instance_id": "route-instance-a",
        "admitted_binding_version": 2,
        "delivery_target_json": (
            '{"origin":{"platform":"telegram","chat_type":"dm",'
            '"chat_id":"42","user_id":"42","profile":""}}'
        ),
        "admitted_session_id": "session-1",
        "admitted_routing_revision": 0,
        "transcript_session_id": "session-1",
        "transcript_json": '[{"role":"assistant","content":"scheduled",'
        '"message_id":"contextual-cron:recover-route-aba:0"}]',
        "transcript_base_message_count": 0,
        "transcript_base_revision": 0,
        "transcript_last_prompt_tokens": None,
    }
    monkeypatch.setattr(
        "cron.executions.list_pending_contextual_transcripts", lambda: [record]
    )
    conflicts = []
    monkeypatch.setattr(
        "cron.executions.mark_contextual_transcript_conflict",
        lambda execution_id, *, error: conflicts.append((execution_id, error)) or True,
    )
    entry = _entry()
    entry.route_instance_id = "route-instance-b"
    runner = object.__new__(GatewayRunner)
    cast(Any, runner).session_store = _LiveStore(entry, [])
    applied = []
    cast(Any, runner)._apply_contextual_cron_transcript = applied.append

    recovered = await GatewayRunner._recover_contextual_cron_transcripts(runner)

    assert recovered == 0
    assert applied == []
    assert conflicts and conflicts[0][0] == "recover-route-aba"
    assert "route changed" in conflicts[0][1]


@pytest.mark.asyncio
async def test_recovery_preserves_nonzero_admitted_route_revision_for_current_route(
    monkeypatch,
):
    record = {
        "id": "recover-current-nonzero-route",
        "job_id": "job-1",
        "session_key": "telegram:dm:42:42",
        "admitted_session_id": "session-1",
        "admitted_routing_revision": 7,
        "transcript_session_id": "session-1",
        "transcript_json": '[{"role":"assistant","content":"scheduled",'
        '"message_id":"contextual-cron:recover-current-nonzero-route:0"}]',
        "transcript_base_message_count": 0,
        "transcript_base_revision": 0,
        "transcript_last_prompt_tokens": None,
    }
    monkeypatch.setattr(
        "cron.executions.list_pending_contextual_transcripts", lambda: [record]
    )
    conflicts = []
    monkeypatch.setattr(
        "cron.executions.mark_contextual_transcript_conflict",
        lambda execution_id, *, error: conflicts.append((execution_id, error)) or True,
    )
    runner = object.__new__(GatewayRunner)
    cast(Any, runner).session_store = _LiveStore(
        replace(_entry(), routing_revision=7), []
    )
    cast(Any, runner)._is_user_authorized = lambda _source: True
    applied = []

    async def apply(item):
        applied.append(item)

    cast(Any, runner)._apply_contextual_cron_transcript = apply

    recovered = await GatewayRunner._recover_contextual_cron_transcripts(runner)

    assert recovered == 1
    assert conflicts == []
    assert len(applied) == 1
    assert applied[0].admitted_routing_revision == 7
    assert applied[0].contextual_route_current is True


@pytest.mark.asyncio
async def test_v2_recovery_reauthorizes_sealed_creator_not_route_origin(monkeypatch):
    import json

    record = {
        "id": "recover-shared-creator",
        "job_id": "job-1",
        "session_key": "telegram:dm:42:42",
        "admitted_session_id": "session-1",
        "admitted_routing_revision": 0,
        "admitted_route_instance_id": "route-instance-a",
        "admitted_binding_version": 2,
        "delivery_target_json": json.dumps(
            {
                "id": "job-1",
                "deliver": "local",
                "origin": {
                    "platform": "telegram",
                    "chat_type": "dm",
                    "chat_id": "42",
                    "user_id": "creator-b",
                    "profile": "",
                },
            }
        ),
        "transcript_session_id": "session-1",
        "transcript_json": '[{"role":"assistant","content":"scheduled",'
        '"message_id":"contextual-cron:recover-shared-creator:0"}]',
        "transcript_base_message_count": 0,
        "transcript_base_revision": 0,
        "transcript_last_prompt_tokens": None,
    }
    monkeypatch.setattr(
        "cron.executions.list_pending_contextual_transcripts", lambda: [record]
    )
    conflicts = []
    monkeypatch.setattr(
        "cron.executions.mark_contextual_transcript_conflict",
        lambda execution_id, *, error: conflicts.append((execution_id, error)) or True,
    )
    route_entry = _entry()
    route_entry.route_instance_id = "route-instance-a"
    assert route_entry.origin is not None
    route_entry.origin.user_id = "creator-a"
    runner = object.__new__(GatewayRunner)
    cast(Any, runner).session_store = _LiveStore(route_entry, [])
    authorized_users = []
    cast(Any, runner)._is_user_authorized = lambda source: (
        authorized_users.append(source.user_id) or source.user_id == "creator-b"
    )
    applied = []

    async def apply(item):
        applied.append(item)

    cast(Any, runner)._apply_contextual_cron_transcript = apply

    recovered = await GatewayRunner._recover_contextual_cron_transcripts(runner)

    assert recovered == 1
    assert conflicts == []
    assert authorized_users == ["creator-b"]
    assert applied[0].source.user_id == "creator-b"


@pytest.mark.asyncio
async def test_recovery_receipt_can_be_acknowledged_after_route_revision_change(
    monkeypatch,
):
    record = {
        "id": "recover-receipted",
        "job_id": "job-1",
        "session_key": "telegram:dm:42:42",
        "admitted_session_id": "session-1",
        "admitted_routing_revision": 1,
        "transcript_session_id": "session-1",
        "transcript_json": '[{"role":"assistant","content":"scheduled",'
        '"message_id":"contextual-cron:recover-receipted:0"}]',
        "transcript_base_message_count": 0,
        "transcript_base_revision": 0,
        "transcript_last_prompt_tokens": None,
    }
    monkeypatch.setattr(
        "cron.executions.list_pending_contextual_transcripts", lambda: [record]
    )

    class ReceiptedStore(_LiveStore):
        def get_contextual_transcript_application(self, execution_id):
            assert execution_id == "recover-receipted"
            return {"execution_id": execution_id}

    runner = object.__new__(GatewayRunner)
    cast(Any, runner).session_store = ReceiptedStore(
        replace(_entry(), routing_revision=2), []
    )
    applied = []

    async def apply(item):
        applied.append(item)

    cast(Any, runner)._apply_contextual_cron_transcript = apply

    recovered = await GatewayRunner._recover_contextual_cron_transcripts(runner)

    assert recovered == 1
    assert len(applied) == 1
    assert applied[0].contextual_route_current is False


@pytest.mark.asyncio
async def test_contextual_transcript_recovery_propagates_deferred_record(
    monkeypatch,
):
    record = {
        "id": "recover-deferred",
        "job_id": "job-1",
        "session_key": "telegram:dm:42:42",
        "admitted_session_id": "session-1",
        "admitted_routing_revision": 0,
        "transcript_session_id": "session-1",
        "transcript_json": '[{"role":"assistant","content":"scheduled",'
        '"message_id":"contextual-cron:recover-deferred:0"}]',
        "transcript_base_message_count": 0,
        "transcript_base_revision": 0,
        "transcript_last_prompt_tokens": 42,
    }
    monkeypatch.setattr(
        "cron.executions.list_pending_contextual_transcripts", lambda: [record]
    )
    runner = object.__new__(GatewayRunner)
    cast(Any, runner).session_store = _LiveStore(_entry(), [])

    async def apply(_item):
        raise RuntimeError("state db unavailable")

    cast(Any, runner)._apply_contextual_cron_transcript_with_recovery_fence = apply

    with pytest.raises(RuntimeError, match="remain deferred"):
        await GatewayRunner._recover_contextual_cron_transcripts(runner)


@pytest.mark.asyncio
async def test_contextual_transcript_recovery_reauthorizes_before_apply():
    from gateway.contextual_cron import ContextualCronTranscriptConflict

    runner = object.__new__(GatewayRunner)
    cast(Any, runner).session_store = _LiveStore(_entry(), [])
    cast(Any, runner)._is_user_authorized = lambda _source: False
    applied = []

    async def apply(item):
        applied.append(item)

    cast(Any, runner)._apply_contextual_cron_transcript = apply
    item = SimpleNamespace(
        execution_id="revoked",
        session_key="telegram:dm:42:42",
        admitted_session_id="session-1",
        admitted_routing_revision=0,
    )

    with pytest.raises(ContextualCronTranscriptConflict, match="authorization"):
        await GatewayRunner._apply_contextual_cron_transcript_with_recovery_fence(
            runner, item
        )
    assert applied == []


@pytest.mark.asyncio
async def test_contextual_transcript_recovery_waits_for_the_session_turn_lease(
    monkeypatch,
):
    import asyncio

    from gateway.turn_lease import SessionTurnLeaseRegistry

    record = {
        "id": "recover-after-human",
        "job_id": "job-1",
        "session_key": "telegram:dm:42:42",
        "admitted_session_id": "session-1",
        "admitted_routing_revision": 0,
        "transcript_session_id": "session-1",
        "transcript_json": '[{"role":"assistant","content":"scheduled",'
        '"message_id":"contextual-cron:recover-after-human:0"}]',
        "transcript_base_message_count": 0,
        "transcript_base_revision": 0,
        "transcript_last_prompt_tokens": 42,
    }
    monkeypatch.setattr(
        "cron.executions.list_pending_contextual_transcripts", lambda: [record]
    )
    runner = object.__new__(GatewayRunner)
    cast(Any, runner).session_store = _LiveStore(_entry(), [])
    cast(Any, runner)._is_user_authorized = lambda _source: True
    runner._turn_leases = SessionTurnLeaseRegistry()
    apply_started = asyncio.Event()

    async def apply(item):
        del item
        apply_started.set()

    runner._apply_contextual_cron_transcript = apply
    held = await runner._turn_leases.acquire(
        "session-1", owner_key="human-turn", generation=1
    )
    recovery = asyncio.create_task(
        GatewayRunner._recover_contextual_cron_transcripts(runner)
    )
    await asyncio.sleep(0.03)
    assert apply_started.is_set() is False
    assert recovery.done() is False

    runner._turn_leases.release(held)
    assert await asyncio.wait_for(recovery, timeout=1) == 1
    assert apply_started.is_set() is True


@pytest.mark.asyncio
async def test_contextual_transcript_recovery_waits_for_the_adapter_guard():
    import asyncio

    class Adapter:
        def __init__(self):
            self.lock = asyncio.Lock()

        async def acquire_contextual_cron_guard(self, _session_key):
            await self.lock.acquire()
            return object()

        async def release_contextual_cron_guard(self, _session_key, _guard):
            self.lock.release()

    adapter = Adapter()
    human_guard = await adapter.acquire_contextual_cron_guard("telegram:dm:42:42")
    runner = object.__new__(GatewayRunner)
    cast(Any, runner).session_store = _LiveStore(_entry(), [])
    cast(Any, runner)._is_user_authorized = lambda _source: True
    runner._adapter_for_source = lambda source: adapter
    apply_started = asyncio.Event()

    async def apply(item):
        del item
        apply_started.set()

    runner._apply_contextual_cron_transcript = apply
    item = SimpleNamespace(
        execution_id="recover-after-adapter",
        session_key="telegram:dm:42:42",
        admitted_session_id="session-1",
    )
    recovery = asyncio.create_task(
        GatewayRunner._apply_contextual_cron_transcript_with_recovery_fence(
            runner, item
        )
    )
    await asyncio.sleep(0.03)
    assert apply_started.is_set() is False

    await adapter.release_contextual_cron_guard(
        "telegram:dm:42:42", human_guard
    )
    await asyncio.wait_for(recovery, timeout=1)
    assert apply_started.is_set() is True
    assert adapter.lock.locked() is False


@pytest.mark.asyncio
async def test_contextual_transcript_recovery_revalidates_route_inside_both_fences():
    from gateway.contextual_cron import ContextualCronTranscriptConflict
    from gateway.turn_lease import SessionTurnLeaseRegistry

    store = _LiveStore(_entry(), [])

    class RouteChangingAdapter:
        async def acquire_contextual_cron_guard(self, _session_key):
            store.entry = replace(
                store.entry,
                session_id="replacement-session",
                routing_revision=1,
            )
            return object()

        async def release_contextual_cron_guard(self, _session_key, _guard):
            return None

    runner = cast(Any, object.__new__(GatewayRunner))
    runner.session_store = store
    runner._turn_leases = SessionTurnLeaseRegistry()
    runner._adapter_for_source = lambda _source: RouteChangingAdapter()
    applied = []

    async def apply(item):
        applied.append(item)

    runner._apply_contextual_cron_transcript = apply
    item = SimpleNamespace(
        execution_id="recover-route-race",
        session_key="telegram:dm:42:42",
        admitted_session_id="session-1",
        admitted_routing_revision=0,
    )

    with pytest.raises(ContextualCronTranscriptConflict, match="route changed"):
        await GatewayRunner._apply_contextual_cron_transcript_with_recovery_fence(
            runner, item
        )

    assert applied == []


@pytest.mark.asyncio
async def test_contextual_transcript_recovery_terminalizes_a_causal_conflict(
    monkeypatch,
):
    from gateway.contextual_cron import ContextualCronTranscriptConflict

    record = {
        "id": "recover-conflict",
        "job_id": "job-1",
        "session_key": "telegram:dm:42:42",
        "admitted_session_id": "session-1",
        "admitted_routing_revision": 0,
        "transcript_session_id": "session-1",
        "transcript_json": '[{"role":"assistant","content":"scheduled",'
        '"message_id":"contextual-cron:recover-conflict:0"}]',
        "transcript_base_message_count": 0,
        "transcript_base_revision": 0,
    }
    monkeypatch.setattr(
        "cron.executions.list_pending_contextual_transcripts", lambda: [record]
    )
    marked = []
    monkeypatch.setattr(
        "cron.executions.mark_contextual_transcript_conflict",
        lambda execution_id, *, error: marked.append((execution_id, error)) or True,
    )
    runner = object.__new__(GatewayRunner)

    async def apply_with_fence(item):
        del item
        raise ContextualCronTranscriptConflict(
            "Contextual transcript advanced before recovery."
        )

    runner._apply_contextual_cron_transcript_with_recovery_fence = apply_with_fence

    recovered = await GatewayRunner._recover_contextual_cron_transcripts(runner)

    assert recovered == 0
    assert marked == [
        ("recover-conflict", "Contextual transcript advanced before recovery.")
    ]


@pytest.mark.asyncio
async def test_contextual_queue_is_fifo_and_waits_behind_active_user_turn():
    import asyncio

    runner = _QueueRunner(_LiveStore(_entry(), []))
    runner.busy = True
    gateway = _queue_gateway(runner, [])
    job = {
        "id": "job",
        "prompt": "continue",
        "session_target": "current",
        "session_key": "telegram:dm:42:42",
    }
    first = asyncio.create_task(gateway.dispatch(job, execution_id="first"))
    second = asyncio.create_task(gateway.dispatch(job, execution_id="second"))
    await asyncio.sleep(0.01)
    assert runner.started == []

    runner.busy = False
    await asyncio.sleep(0.01)
    assert runner.started == ["first"]
    runner.release_first.set()

    assert (await first).final_response == "first"
    assert (await second).final_response == "second"
    assert runner.started == ["first", "second"]


@pytest.mark.asyncio
async def test_contextual_lane_waits_on_resolved_session_turn_lease():
    import asyncio

    from gateway.turn_lease import SessionTurnLeaseRegistry

    runner = _QueueRunner(_LiveStore(_entry(), []))
    turn_leases = SessionTurnLeaseRegistry()
    cast(Any, runner)._turn_leases = turn_leases
    held = await turn_leases.acquire(
        "session-1", owner_key="human-turn", generation=1
    )
    gateway = ContextualCronGateway(
        runner,
        seal_admission=lambda *_: True,
        finish_admission=lambda *_: None,
    )
    job = {
        "id": "job",
        "prompt": "continue",
        "session_target": "current",
        "session_key": "telegram:dm:42:42",
    }

    task = asyncio.create_task(gateway.dispatch(job, execution_id="exec-lease"))
    await asyncio.sleep(0.03)
    assert runner.started == []

    turn_leases.release(held)
    result = await asyncio.wait_for(task, timeout=1)
    assert result.kind == "notify"
    assert runner.started == ["exec-lease"]


@pytest.mark.asyncio
async def test_contextual_lane_holds_turn_lease_until_outbox_is_applied():
    import asyncio

    from gateway.turn_lease import SessionTurnLeaseRegistry

    class Runner(_QueueRunner):
        def __init__(self, store):
            super().__init__(store)
            self._turn_leases = SessionTurnLeaseRegistry()
            self.apply_started = asyncio.Event()
            self.allow_apply = asyncio.Event()

        async def _run_contextual_cron_turn(self, item, entry, history):
            item.transcript_session_id = entry.session_id
            item.transcript_entries = [
                {
                    "role": "assistant",
                    "content": "done",
                    "display_kind": "normal",
                    "message_id": f"contextual-cron:{item.execution_id}:0",
                }
            ]
            return ContextualCronOutcome.notify("done")

        async def _apply_contextual_cron_transcript(self, _item):
            self.apply_started.set()
            await self.allow_apply.wait()

    runner = Runner(_LiveStore(_entry(), []))
    gateway = ContextualCronGateway(
        runner,
        seal_admission=lambda *_: True,
        finish_admission=lambda *_: {"transcript_state": "pending"},
    )
    dispatch = asyncio.create_task(
        gateway.dispatch(
            {
                "id": "job",
                "prompt": "continue",
                "session_target": "current",
                "session_key": "telegram:dm:42:42",
            },
            execution_id="lease-through-outbox",
        )
    )
    await asyncio.wait_for(runner.apply_started.wait(), timeout=1)

    human = asyncio.create_task(
        runner._turn_leases.acquire(
            "session-1", owner_key="human-after-cron", generation=1
        )
    )
    await asyncio.sleep(0.02)
    assert human.done() is False
    assert dispatch.done() is False

    runner.allow_apply.set()
    assert (await asyncio.wait_for(dispatch, timeout=1)).kind == "notify"
    human_token = await asyncio.wait_for(human, timeout=1)
    runner._turn_leases.release(human_token)


@pytest.mark.asyncio
async def test_contextual_drainer_cancellation_resolves_all_pending_unknown():
    import asyncio

    runner = _QueueRunner(_LiveStore(_entry(), []))
    gateway = _queue_gateway(runner, [])
    job = {
        "id": "job",
        "prompt": "continue",
        "session_target": "current",
        "session_key": "telegram:dm:42:42",
    }
    first = asyncio.create_task(gateway.dispatch(job, execution_id="first"))
    second = asyncio.create_task(gateway.dispatch(job, execution_id="second"))
    for _ in range(20):
        if runner.started:
            break
        await asyncio.sleep(0.005)
    assert runner.started == ["first"]

    gateway._drainers["telegram:dm:42:42"].cancel()
    first_outcome, second_outcome = await asyncio.wait_for(
        asyncio.gather(first, second), timeout=1
    )
    assert first_outcome.kind == "unknown"
    assert second_outcome.kind == "unknown"
    assert runner.started == ["first"]


@pytest.mark.asyncio
async def test_v1_reset_before_admission_remains_strictly_stale():
    replacement = _entry("replacement-session")
    store = _LiveStore(replacement, [])
    runner = _QueueRunner(store)
    sealed = []
    gateway = ContextualCronGateway(
        runner,
        seal_admission=lambda *args: sealed.append(args) or True,
        finish_admission=lambda *_args: None,
        busy_poll_seconds=0.001,
    )

    outcome = await gateway.dispatch(
        {
            "id": "legacy-reset-before-admission",
            "session_target": "current",
            "session_key": "telegram:dm:42:42",
            "context_binding": {
                "profile": "",
                "session_key": "telegram:dm:42:42",
                "session_id": "original-session",
                "routing_revision": 0,
                "platform": "telegram",
                "chat_type": "dm",
                "chat_id": "42",
                "thread_id": "",
                "user_id": "42",
            },
            "_contextual_binding_version": 1,
            "prompt": "legacy strict binding",
        },
        execution_id="legacy-reset-before-admission",
    )

    assert outcome.kind == "stale"
    assert sealed == []
    assert runner.started == []


@pytest.mark.asyncio
async def test_reset_before_admission_follows_same_logical_route():
    replacement = _entry("replacement-session")
    replacement.route_instance_id = "route-instance-a"
    store = _LiveStore(replacement, [])
    runner = _QueueRunner(store)
    sealed = []
    gateway = ContextualCronGateway(
        runner,
        seal_admission=lambda *args: sealed.append(args) or True,
        finish_admission=lambda *_: None,
    )

    outcome = await gateway.dispatch(
        {
            "id": "job",
            "prompt": "continue",
            "session_target": "current",
            "session_key": "telegram:dm:42:42",
            "_contextual_binding_version": 2,
            "context_binding": {
                "session_key": "telegram:dm:42:42",
                "route_instance_id": "route-instance-a",
                "profile": "",
                "platform": "telegram",
                "chat_type": "dm",
                "chat_id": "42",
                "thread_id": "",
                "user_id": "42",
            },
        },
        execution_id="reset-before-admission",
    )

    assert outcome.kind == "notify"
    assert sealed == [
        (
            "reset-before-admission",
            "telegram:dm:42:42",
            "replacement-session",
            0,
            "route-instance-a",
            2,
        )
    ]
    assert runner.started == ["reset-before-admission"]


@pytest.mark.asyncio
async def test_v2_route_delete_recreate_aba_is_stale_before_seal():
    recreated = _entry("replacement-session")
    recreated.route_instance_id = "route-instance-b"
    store = _LiveStore(recreated, [])
    runner = _QueueRunner(store)
    sealed = []
    gateway = ContextualCronGateway(
        runner,
        seal_admission=lambda *args: sealed.append(args) or True,
        finish_admission=lambda *_: None,
    )

    outcome = await gateway.dispatch(
        {
            "id": "job",
            "prompt": "continue",
            "session_target": "current",
            "session_key": recreated.session_key,
            "_contextual_binding_version": 2,
            "context_binding": {
                "session_key": recreated.session_key,
                "route_instance_id": "route-instance-a",
                "profile": "",
                "platform": "telegram",
                "chat_type": "dm",
                "chat_id": "42",
                "thread_id": "",
                "user_id": "42",
            },
        },
        execution_id="route-aba",
    )

    assert outcome.kind == "stale"
    assert sealed == []
    assert runner.started == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scope_id", "workspace-b"),
        ("parent_chat_id", "parent-b"),
        ("chat_id_alt", "chat-alt-b"),
    ],
)
@pytest.mark.asyncio
async def test_v2_same_route_instance_with_principal_drift_is_stale(field, value):
    changed = _entry("same-session")
    changed.route_instance_id = "route-instance-a"
    assert changed.origin is not None
    setattr(changed.origin, field, value)
    store = _LiveStore(changed, [])
    runner = _QueueRunner(store)
    sealed = []
    gateway = ContextualCronGateway(
        runner,
        seal_admission=lambda *args: sealed.append(args) or True,
        finish_admission=lambda *_: None,
    )

    outcome = await gateway.dispatch(
        {
            "id": "job",
            "prompt": "continue",
            "session_target": "current",
            "session_key": changed.session_key,
            "_contextual_binding_version": 2,
            "context_binding": {
                "session_key": changed.session_key,
                "route_instance_id": "route-instance-a",
                "profile": "",
                "platform": "telegram",
                "chat_type": "dm",
                "chat_id": "42",
                "thread_id": "",
                "user_id": "42",
            },
        },
        execution_id="principal-drift",
    )

    assert outcome.kind == "stale"
    assert sealed == []
    assert runner.started == []


def test_v2_prospective_thread_and_real_thread_are_one_logical_route():
    from dataclasses import replace

    from gateway.session import _same_route_identity, build_session_key

    entry = _entry("thread-session")
    assert entry.origin is not None
    prospective = replace(
        entry.origin,
        platform=Platform.DISCORD,
        chat_type="channel",
        chat_id="channel-1",
        parent_chat_id=None,
        thread_id=None,
        prospective_thread_id="thread-42",
    )
    realized = replace(
        entry.origin,
        platform=Platform.DISCORD,
        chat_type="thread",
        chat_id="thread-42",
        parent_chat_id="channel-1",
        thread_id="thread-42",
        prospective_thread_id=None,
    )
    entry.origin = realized
    entry.route_instance_id = "route-instance-a"

    assert build_session_key(prospective) == build_session_key(realized)
    assert _same_route_identity(prospective, realized)
    assert (
        ContextualCronGateway._logical_binding_rejection(
            entry,
            {
                "route_instance_id": "route-instance-a",
                "profile": "",
                "platform": "discord",
                "chat_type": "thread",
                "chat_id": "thread-42",
                "thread_id": "thread-42",
                "user_id": "42",
                "scope_id": "",
                "parent_chat_id": "channel-1",
                "user_id_alt": "",
                "chat_id_alt": "",
            },
        )
        is None
    )


@pytest.mark.asyncio
async def test_v2_shared_thread_uses_captured_creator_authority():
    from gateway.session import build_session_key

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_type="thread",
        chat_id="thread-42",
        parent_chat_id="channel-1",
        thread_id="thread-42",
        user_id="creator-a",
    )
    now = datetime.now(timezone.utc)
    entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="thread-session",
        created_at=now,
        updated_at=now,
        origin=source,
        route_instance_id="route-instance-a",
    )
    runner = _QueueRunner(_LiveStore(entry, []))
    gateway = ContextualCronGateway(
        runner,
        seal_admission=lambda *_: True,
        finish_admission=lambda *_: None,
    )

    outcome = await gateway.dispatch(
        {
            "id": "job-b",
            "prompt": "continue",
            "session_target": "current",
            "session_key": entry.session_key,
            "_contextual_binding_version": 2,
            "context_binding": {
                "session_key": entry.session_key,
                "route_instance_id": "route-instance-a",
                "profile": "",
                "platform": "discord",
                "chat_type": "thread",
                "chat_id": "thread-42",
                "thread_id": "thread-42",
                "user_id": "creator-b",
                "scope_id": "",
                "parent_chat_id": "channel-1",
                "user_id_alt": "",
                "chat_id_alt": "",
            },
        },
        execution_id="shared-creator-b",
    )

    assert outcome.kind == "notify"
    assert runner.started == ["shared-creator-b"]


@pytest.mark.asyncio
async def test_v2_route_instance_swap_during_model_is_stale_before_commit():
    entry = _entry("session-1")
    entry.route_instance_id = "route-instance-a"
    store = _LiveStore(entry, [])
    runner = _QueueRunner(store)

    async def run_turn(item, _entry_value, _history):
        runner.started.append(item.execution_id)
        store.entry.route_instance_id = "route-instance-b"
        return ContextualCronOutcome.notify("must-not-commit")

    setattr(cast(Any, runner), "_run_contextual_cron_turn", run_turn)
    gateway = ContextualCronGateway(
        runner,
        seal_admission=lambda *_: True,
        finish_admission=lambda *_: None,
    )

    outcome = await gateway.dispatch(
        {
            "id": "job",
            "prompt": "continue",
            "session_target": "current",
            "session_key": entry.session_key,
            "_contextual_binding_version": 2,
            "context_binding": {
                "session_key": entry.session_key,
                "route_instance_id": "route-instance-a",
                "profile": "",
                "platform": "telegram",
                "chat_type": "dm",
                "chat_id": "42",
                "thread_id": "",
                "user_id": "42",
            },
        },
        execution_id="route-swap-during-model",
    )

    assert runner.started == ["route-swap-during-model"]
    assert outcome.kind == "stale"
    assert outcome.final_response == ""


@pytest.mark.asyncio
async def test_reset_after_admission_is_stale():
    store = _LiveStore(_entry("new-before-admission"), [])
    runner = _QueueRunner(store)
    sealed = []

    def seal(execution_id, key, session_id):
        sealed.append((execution_id, key, session_id))
        store.entry = _entry("reset-after-admission")
        return True

    gateway = ContextualCronGateway(
        runner,
        seal_admission=seal,
        finish_admission=lambda *_: None,
        busy_poll_seconds=0.001,
    )
    outcome = await gateway.dispatch(
        {
            "id": "job",
            "prompt": "continue",
            "session_target": "current",
            "session_key": "telegram:dm:42:42",
            "context_binding": {
                "session_key": "telegram:dm:42:42",
                "session_id": "new-before-admission",
                "routing_revision": 0,
            },
        },
        execution_id="reset-boundary",
    )

    assert sealed[0][2] == "new-before-admission"
    assert outcome.kind == "stale"
    assert runner.started == []


@pytest.mark.asyncio
async def test_missing_or_revoked_session_is_rejected_without_fallback():
    missing_store = _LiveStore(_entry(), [])
    missing_store.entry = None
    missing_runner = _QueueRunner(missing_store)
    missing = await _queue_gateway(missing_runner, []).dispatch(
        {
            "id": "job",
            "prompt": "continue",
            "session_target": "current",
            "session_key": "telegram:dm:42:42",
        },
        execution_id="missing",
    )
    assert missing.kind == "rejected"
    assert missing_runner.started == []

    store = _LiveStore(_entry(), [])
    revoked_runner = _QueueRunner(store)
    revoked_runner.busy = True
    gateway = _queue_gateway(revoked_runner, [])
    task = __import__("asyncio").create_task(
        gateway.dispatch(
            {
                "id": "job",
                "prompt": "continue",
                "session_target": "current",
                "session_key": "telegram:dm:42:42",
            },
            execution_id="revoked",
        )
    )
    await __import__("asyncio").sleep(0.01)
    revoked_runner.authorized = False
    revoked_runner.busy = False
    revoked = await task
    assert revoked.kind == "rejected"
    assert revoked_runner.started == []


def test_contextual_no_action_hides_every_generated_tool_row(tmp_path):
    from gateway.run import _apply_contextual_transcript_visibility
    from hermes_state import SessionDB

    canary = "CONTEXTUAL-NO-ACTION-TOOL-CANARY"
    generated = [
        {"role": "user", "content": "internal scheduled prompt"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-secret",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"' + canary + '"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-secret",
            "content": canary,
        },
        {"role": "assistant", "content": "[SILENT]"},
    ]
    staged = [
        _apply_contextual_transcript_visibility(
            dict(message),
            contextual=True,
            intentional_silence=True,
        )
        for message in generated
    ]
    assert all(message.get("display_kind") == "hidden" for message in staged)

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("contextual-no-action", source="test", model="test/model")
    db.append_messages_batch("contextual-no-action", staged)
    try:
        assert db.get_messages("contextual-no-action") == []
        privileged = db.get_messages(
            "contextual-no-action",
            include_hidden=True,
        )
    finally:
        db.close()
    assert canary in repr(privileged)

    notifying_tool = _apply_contextual_transcript_visibility(
        {"role": "tool", "content": "public tool result"},
        contextual=True,
        intentional_silence=False,
    )
    assert "display_kind" not in notifying_tool


@pytest.mark.asyncio
async def test_runner_internal_turn_is_hidden_and_intentional_silence_is_no_action(
    monkeypatch,
):
    from unittest.mock import AsyncMock

    from gateway.run import GatewayRunner
    from gateway.contextual_cron import ContextualCronQueueItem

    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = lambda _source: None
    hook_calls = []
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *args, **kwargs: hook_calls.append((args, kwargs)),
    )

    async def handle(event, source, session_key, generation):
        from gateway.session_context import _get_contextual_turn_authority
        from hermes_cli.lifecycle import has_hook, invoke_hook

        assert event.internal is True
        assert event.metadata["contextual_cron"] is True
        assert event.metadata["contextual_cron_admitted_session_id"] == "session-1"
        assert event.metadata["contextual_cron_admitted_routing_revision"] == 3
        authority = _get_contextual_turn_authority()
        assert authority.execution_id == "execution"
        assert authority.admitted_session_id == "session-1"
        assert authority.creator_source is source
        assert source.user_id == "creator-b"
        assert has_hook("agent:start") is False
        assert invoke_hook("agent:start") == []
        event.metadata["contextual_cron_result"] = {
            "completed": True,
            "intentional_silence": True,
            "error": None,
        }
        event.metadata["contextual_cron_transcript_session_id"] = "session-1"
        event.metadata["contextual_cron_transcript_entries"] = [
            {
                "role": "user",
                "content": "hidden",
                "display_kind": "hidden",
                "message_id": "contextual-cron:execution:0",
            }
        ]
        event.metadata["contextual_cron_last_prompt_tokens"] = 17
        return ""

    state = SimpleNamespace(turn=SimpleNamespace(lease=None, agent=None, started_ts=0))
    runner._handle_message_with_agent = handle
    runner._claim_active_session_slot = lambda _key, _source: (None, None)
    runner._session_state = lambda _key: state
    runner._persist_active_agents = lambda: None
    runner._begin_session_run_generation = lambda _key: 1
    runner._restore_moa_one_shot = lambda *_args: None
    restored_one_turn = []
    runner._restore_pending_one_turn_model_override = (
        lambda *args: restored_one_turn.append(args)
    )
    runner._release_running_agent_state = lambda *_args, **_kwargs: True
    runner._release_turn_lease = lambda *_args, **_kwargs: True
    loop = __import__("asyncio").get_running_loop()
    creator = _entry().origin
    assert creator is not None
    creator.user_id = "creator-b"
    route_entry = _entry()
    item = ContextualCronQueueItem(
        job_id="job",
        execution_id="execution",
        prompt="check quietly",
        session_key="telegram:dm:42:42",
        admitted_session_id="session-1",
        admitted_routing_revision=3,
        source=creator,
        future=loop.create_future(),
    )
    cast(Any, runner)._detach_preheld_turn_lease = lambda *_args: True
    registry, token = await _hold_contextual_turn_lease(runner, item)
    try:
        outcome = await GatewayRunner._run_contextual_cron_turn(
            runner,
            item,
            route_entry,
            [],
        )
    finally:
        assert registry.release(token) is True
    assert outcome.kind == "no_action"
    assert item.transcript_session_id == "session-1"
    assert item.transcript_entries is not None
    assert item.transcript_entries[0]["message_id"] == "contextual-cron:execution:0"
    assert item.last_prompt_tokens == 17
    assert restored_one_turn == []
    assert hook_calls == []


@pytest.mark.asyncio
async def test_contextual_turn_fails_closed_without_explicit_completion_metadata():
    from gateway.contextual_cron import ContextualCronQueueItem

    runner = object.__new__(GatewayRunner)
    runner._adapter_for_source = lambda _source: None

    async def incomplete_handler(*_args):
        return "provider error rendered as text"

    state = SimpleNamespace(turn=SimpleNamespace(lease=None, agent=None, started_ts=0))
    runner._handle_message_with_agent = incomplete_handler
    runner._claim_active_session_slot = lambda _key, _source: (None, None)
    runner._session_state = lambda _key: state
    runner._persist_active_agents = lambda: None
    runner._begin_session_run_generation = lambda _key: 1
    runner._restore_moa_one_shot = lambda *_args: None
    runner._release_running_agent_state = lambda *_args, **_kwargs: True
    runner._release_turn_lease = lambda *_args, **_kwargs: True
    runner._detach_preheld_turn_lease = lambda *_args, **_kwargs: True
    runner_any = cast(Any, runner)
    runner_any._turn_leases = SimpleNamespace(
        is_current_holder=lambda *_args, **_kwargs: True
    )
    turn_lease_token = object()
    loop = __import__("asyncio").get_running_loop()

    outcome = await GatewayRunner._run_contextual_cron_turn(
        runner,
        ContextualCronQueueItem(
            job_id="job",
            execution_id="incomplete",
            prompt="check",
            session_key="telegram:dm:42:42",
            admitted_session_id="session-1",
            admitted_routing_revision=0,
            source=_entry().origin,
            future=loop.create_future(),
            turn_lease_token=turn_lease_token,
        ),
        _entry(),
        [],
    )

    assert outcome.kind == "failure"
    assert "failed closed" in str(outcome.error)


def test_internal_session_metadata_update_does_not_touch_human_activity_clock():
    import threading

    from gateway.session import SessionStore

    entry = _entry()
    original = entry.updated_at
    store = object.__new__(SessionStore)
    store._lock = threading.Lock()
    store._loaded = True
    store._entries = {entry.session_key: entry}
    store._save_entry = lambda session_key: None
    store._record_gateway_session_peer = lambda *_args, **_kwargs: None

    store.update_session(
        entry.session_key,
        last_prompt_tokens=123,
        touch_activity=False,
    )

    assert entry.updated_at == original
    assert entry.last_prompt_tokens == 123


@pytest.mark.asyncio
async def test_duplicate_after_future_eviction_replays_scheduler_terminal_result(
    monkeypatch, tmp_path
):
    """The gateway admits/runs; the scheduler alone persists and can replay."""
    import cron.executions as executions

    monkeypatch.setattr(
        executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db"
    )
    record = executions.create_execution("job", source="builtin")
    executions.mark_execution_running(record["id"])

    runner = _QueueRunner(_LiveStore(_entry(), []))
    gateway = ContextualCronGateway(runner)
    job = {
        "id": "job",
        "prompt": "continue",
        "session_target": "current",
        "session_key": "telegram:dm:42:42",
    }

    first = await gateway.dispatch(job, execution_id=record["id"])
    assert first == ContextualCronOutcome.notify(record["id"])
    # Gateway queue/future eviction does not race the scheduler's delivery-aware
    # terminal write with a second, less-informed terminal update.
    persisted = executions.get_execution(record["id"])
    assert persisted is not None
    assert persisted["status"] == "running"

    finished = executions.finish_contextual_execution(
        record["id"],
        outcome=first.kind,
        final_response=first.final_response,
    )
    assert finished is not None

    duplicate = await gateway.dispatch(job, execution_id=record["id"])
    assert duplicate == first
    assert runner.started == [record["id"]]


@pytest.mark.asyncio
async def test_concurrent_duplicate_dispatch_has_one_admission_and_one_execution():
    import asyncio
    import threading

    store = _LiveStore(_entry(), [])
    seal_started = threading.Event()
    allow_seal = threading.Event()
    seals = []

    def seal_contextual_admission(session_key, execution_id, seal):
        seals.append(execution_id)
        seal_started.set()
        assert allow_seal.wait(1)
        assert seal(execution_id, session_key, store.entry.session_id, 0)
        return store.entry

    cast(Any, store).seal_contextual_admission = seal_contextual_admission
    runner = _QueueRunner(store)
    gateway = ContextualCronGateway(
        runner,
        seal_admission=lambda *_args: True,
    )
    job = {
        "id": "job",
        "prompt": "continue",
        "session_target": "current",
        "session_key": store.entry.session_key,
    }

    first = asyncio.create_task(gateway.dispatch(job, execution_id="duplicate"))
    assert await asyncio.to_thread(seal_started.wait, 1)
    second = asyncio.create_task(gateway.dispatch(job, execution_id="duplicate"))
    await asyncio.sleep(0)
    allow_seal.set()
    outcomes = await asyncio.gather(first, second)

    assert outcomes == [
        ContextualCronOutcome.notify("duplicate"),
        ContextualCronOutcome.notify("duplicate"),
    ]
    assert seals == ["duplicate"]
    assert runner.started == ["duplicate"]


@pytest.mark.asyncio
async def test_live_contextual_turn_bypasses_normal_incoming_handler():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    calls = []

    async def forbidden_incoming_handler(_event):
        raise AssertionError("contextual cron entered the normal incoming handler")

    async def live_agent_handler(event, source, session_key, generation):
        calls.append((event, source, session_key, generation))
        event.metadata["contextual_cron_result"] = {"completed": True}
        return "scheduled result"

    runner._handle_message = forbidden_incoming_handler
    runner._handle_message_with_agent = live_agent_handler
    runner._adapter_for_source = lambda _source: None
    state = SimpleNamespace(turn=SimpleNamespace(lease=None, agent=None, started_ts=0))
    runner._claim_active_session_slot = lambda _key, _source: (None, None)
    runner._session_state = lambda _key: state
    runner._persist_active_agents = lambda: None
    runner._begin_session_run_generation = lambda _key: 9
    runner._restore_moa_one_shot = lambda *_args: None
    runner._restore_pending_one_turn_model_override = lambda *_args: None
    runner._release_running_agent_state = lambda *_args, **_kwargs: True
    runner._release_turn_lease = lambda *_args, **_kwargs: True

    item = SimpleNamespace(
        prompt="continue",
        execution_id="exec-live",
        session_key="telegram:dm:42:42",
        admitted_session_id="session-1",
        admitted_routing_revision=7,
    )
    entry = _entry()
    cast(Any, runner)._detach_preheld_turn_lease = lambda *_args: True
    registry, token = await _hold_contextual_turn_lease(runner, item)

    try:
        outcome = await GatewayRunner._run_contextual_cron_turn(
            runner, item, entry, [{"role": "user", "content": "prior"}]
        )
    finally:
        assert registry.release(token) is True

    assert outcome == ContextualCronOutcome.notify("scheduled result")
    assert len(calls) == 1
    event, source, session_key, generation = calls[0]
    assert source is entry.origin
    assert session_key == entry.session_key
    assert generation == 9
    assert event.internal is True
    assert event.metadata["contextual_cron_lease_preheld"] is True
    assert event.metadata["contextual_cron_admitted_routing_revision"] == 7


@pytest.mark.asyncio
async def test_live_contextual_turn_rejects_forged_lease_before_state_publication():
    from gateway.turn_lease import SessionTurnLeaseRegistry

    runner = GatewayRunner.__new__(GatewayRunner)
    runner_any = cast(Any, runner)
    registry = SessionTurnLeaseRegistry()
    genuine = await registry.acquire(
        "session-1",
        owner_key="contextual-cron:exec-forged",
        generation=0,
        timeout=1,
    )
    assert genuine is not None
    runner_any._get_proxy_url = lambda: None
    runner_any._turn_leases = registry
    runner_any._claim_active_session_slot = pytest.fail
    runner_any._handle_message_with_agent = pytest.fail
    item = SimpleNamespace(
        prompt="continue",
        execution_id="exec-forged",
        session_key="telegram:dm:42:42",
        admitted_session_id="session-1",
        admitted_routing_revision=7,
        turn_lease_token=object(),
    )
    try:
        outcome = await GatewayRunner._run_contextual_cron_turn(
            runner,
            item,
            _entry(),
            [{"role": "user", "content": "prior"}],
        )
    finally:
        assert registry.release(genuine) is True

    assert outcome.kind == "rejected"


@pytest.mark.asyncio
async def test_live_contextual_turn_does_not_release_lane_owned_turn_lease():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner_any = cast(Any, runner)

    async def live_agent_handler(event, _source, _session_key, _generation):
        event.metadata["contextual_cron_result"] = {"completed": True}
        return "scheduled result"

    state = SimpleNamespace(
        turn=SimpleNamespace(
            lease=None,
            agent=None,
            started_ts=0,
            lease_token=None,
            lease_generation=None,
        )
    )
    runner_any._get_proxy_url = lambda: None
    runner_any._handle_message_with_agent = live_agent_handler
    runner_any._adapter_for_source = lambda _source: None
    runner_any._claim_active_session_slot = lambda _key, _source: (None, None)
    runner_any._session_state = lambda _key: state
    runner_any._peek_session_state = lambda _key: state
    runner_any._persist_active_agents = lambda: None
    runner_any._begin_session_run_generation = lambda _key: 9
    runner_any._restore_moa_one_shot = lambda *_args: None
    runner_any._release_running_agent_state = lambda *_args, **_kwargs: True

    item = SimpleNamespace(
        prompt="continue",
        execution_id="exec-live-lease",
        session_key="telegram:dm:42:42",
        admitted_session_id="session-1",
        admitted_routing_revision=7,
    )
    registry, token = await _hold_contextual_turn_lease(runner, item)
    try:
        outcome = await GatewayRunner._run_contextual_cron_turn(
            runner, item, _entry(), [{"role": "user", "content": "prior"}]
        )
        assert registry.is_current_holder(
            token,
            session_id="session-1",
            owner_key="contextual-cron:exec-live-lease",
            generation=0,
        )
    finally:
        assert registry.release(token) is True

    assert outcome == ContextualCronOutcome.notify("scheduled result")
    assert state.turn.lease_token is None
    assert state.turn.lease_generation is None


def test_contextual_user_delta_prepends_missing_prompt_source():
    from gateway.run import _ensure_contextual_user_delta

    assistant_only = [{"role": "assistant", "content": "done"}]
    result = _ensure_contextual_user_delta(
        assistant_only, content="scheduled prompt", timestamp="then"
    )

    assert result == [
        {"role": "user", "content": "scheduled prompt", "timestamp": "then"},
        {"role": "assistant", "content": "done"},
    ]
    assert _ensure_contextual_user_delta(
        result, content="duplicate", timestamp="later"
    ) is result


@pytest.mark.asyncio
async def test_contextual_turn_logs_redact_hidden_prompt_and_route_identity(caplog):
    from gateway.platforms.base import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_context import _bind_contextual_turn_authority

    runner = GatewayRunner.__new__(GatewayRunner)
    runner_any = cast(Any, runner)
    entry = _entry("session-private-route")
    assert entry.origin is not None

    async def missing_entry(_session_key):
        return None

    store = SimpleNamespace()
    runner_any.session_store = store
    runner_any._async_session_store = SimpleNamespace(
        _store=store,
        peek_session_entry=missing_entry,
    )
    event = MessageEvent(
        text="do-not-log-contextual-secret",
        source=entry.origin,
        internal=True,
    )
    caplog.set_level(logging.INFO, logger="gateway.run")

    with _bind_contextual_turn_authority(
        execution_id="exec-private-log",
        session_key=entry.session_key,
        admitted_session_id=entry.session_id,
        admitted_routing_revision=0,
    ):
        result = await GatewayRunner._handle_message_with_agent(
            runner, event, entry.origin, entry.session_key, 1
        )

    assert result is None
    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert "do-not-log-contextual-secret" not in rendered
    assert entry.session_key not in rendered
    assert entry.session_id not in rendered
    assert "contextual cron" in rendered.lower()


def test_agent_exception_path_uses_stage_aware_transcript_writer():
    from gateway.run import GatewayRunner

    source = inspect.getsource(GatewayRunner._handle_message_with_agent)
    exception_path = source.split("except Exception as e:", maxsplit=1)[1]

    assert "await self._write_or_stage_transcript_entry(" in exception_path
    assert "await self.async_session_store.append_to_transcript(" not in exception_path
    assert (
        'if _is_contextual_cron:\n                logger.error("Contextual cron agent execution failed.")'
        in exception_path
    )


@pytest.mark.asyncio
async def test_adapter_guard_race_retries_without_terminally_dropping_occurrence():
    from gateway.contextual_cron import ContextualCronGuardBusy

    runner = _QueueRunner(_LiveStore(_entry(), []))
    calls = 0

    async def race_once(item, entry, history):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ContextualCronGuardBusy()
        return ContextualCronOutcome.notify("after-user-turn")

    cast(Any, runner)._run_contextual_cron_turn = race_once
    gateway = _queue_gateway(runner, [])

    outcome = await gateway.dispatch(
        {
            "id": "job",
            "prompt": "continue",
            "session_target": "current",
            "session_key": "telegram:dm:42:42",
        },
        execution_id="guard-race",
    )

    assert outcome == ContextualCronOutcome.notify("after-user-turn")
    assert calls == 2


@pytest.mark.asyncio
async def test_route_revision_detects_reset_resume_aba_after_admission():
    import asyncio
    from dataclasses import replace

    store = _LiveStore(_entry(), [])
    runner = _QueueRunner(store)
    runner.busy = True
    gateway = _queue_gateway(runner, [])
    job = {
        "id": "job",
        "prompt": "continue",
        "session_target": "current",
        "session_key": "telegram:dm:42:42",
    }

    task = asyncio.create_task(gateway.dispatch(job, execution_id="aba"))
    await asyncio.sleep(0.01)
    # Simulate old→new→old. Concrete id alone matches again, but the route
    # revision proves the admitted mapping changed in between.
    store.entry = replace(store.entry, routing_revision=2)
    runner.busy = False

    outcome = await asyncio.wait_for(task, timeout=1)
    assert outcome.kind == "stale"
    assert runner.started == []


def test_auto_reset_then_resume_keeps_route_revision_monotonic():
    import threading

    from gateway.session import SessionStore

    source = _entry().origin
    key = _entry().session_key
    old = _entry()
    old.routing_revision = 1
    old.route_instance_id = "route-instance-a"
    store = object.__new__(SessionStore)
    store._lock = threading.Lock()
    store._loaded = True
    store._entries = {key: old}
    store._db = None
    store._save_entries = lambda: None
    store._save = lambda: None
    store._generate_session_key = lambda _source: key
    store._compression_tip_for_session_id = lambda session_id: session_id
    store._is_session_ended_in_db = lambda _session_id: False
    store._should_reset = lambda _entry, _source: "idle"

    reset = store._get_or_create_session_impl(source)
    assert reset.session_id != old.session_id
    assert reset.routing_revision == 2
    assert reset.route_instance_id == "route-instance-a"

    resumed = store.switch_session(key, old.session_id)
    assert resumed is not None
    assert resumed.session_id == old.session_id
    assert resumed.routing_revision == 3
    assert resumed.route_instance_id == "route-instance-a"


@pytest.mark.parametrize(
    ("field", "value", "preserves_route"),
    [
        ("user_id", "99", True),
        ("user_id_alt", "user-alt-b", True),
        ("scope_id", "workspace-b", False),
    ],
)
def test_auto_reset_separates_conversation_identity_from_participant_authority(
    field, value, preserves_route
):
    import threading
    from dataclasses import replace

    from gateway.session import SessionStore

    old = _entry()
    assert old.origin is not None
    old.route_instance_id = "route-instance-a"
    incoming = replace(old.origin, **{field: value})
    store = object.__new__(SessionStore)
    store_any = cast(Any, store)
    store_any._lock = threading.Lock()
    store_any._loaded = True
    store_any._entries = {old.session_key: old}
    store_any._db = None
    store_any._save_entries = lambda: None
    store_any._save = lambda: None
    store_any._generate_session_key = lambda _source: old.session_key
    store_any._compression_tip_for_session_id = lambda session_id: session_id
    store_any._is_session_ended_in_db = lambda _session_id: False
    store_any._should_reset = lambda _entry, _source: "idle"

    reset = store._get_or_create_session_impl(incoming)

    assert reset.routing_revision == 1
    assert (reset.route_instance_id == "route-instance-a") is preserves_route
    assert reset.origin is incoming


def test_admission_and_reset_share_one_routing_linearization_lock():
    import threading
    import time

    from gateway.session import SessionStore

    entry = _entry()
    store = object.__new__(SessionStore)
    store._lock = threading.Lock()
    store._loaded = True
    store._entries = {entry.session_key: entry}
    store._save = lambda: None
    store._db = None

    seal_started = threading.Event()
    allow_seal = threading.Event()
    admitted = []
    reset_result = []

    def seal(execution_id, key, session_id, revision):
        seal_started.set()
        assert allow_seal.wait(1)
        admitted.append((execution_id, key, session_id, revision))
        return True

    admit_thread = threading.Thread(
        target=lambda: store.seal_contextual_admission(
            entry.session_key, "exec", seal
        )
    )
    admit_thread.start()
    assert seal_started.wait(1)

    reset_thread = threading.Thread(
        target=lambda: reset_result.append(store.reset_session(entry.session_key))
    )
    reset_thread.start()
    time.sleep(0.02)
    assert reset_thread.is_alive(), "reset crossed the admission seal boundary"

    allow_seal.set()
    admit_thread.join(1)
    reset_thread.join(1)

    assert admitted == [("exec", entry.session_key, "session-1", 0)]
    assert reset_result[0].session_id != "session-1"
    assert reset_result[0].routing_revision == 1


def test_admission_seal_rejects_route_instance_swap_inside_routing_lock():
    import threading

    from gateway.session import ContextualRouteInstanceMismatch, SessionStore

    store = object.__new__(SessionStore)
    store._lock = threading.Lock()
    store._loaded = True
    entry = _entry("replacement-session")
    entry.route_instance_id = "route-instance-b"
    store._entries = {entry.session_key: entry}
    sealed = []

    with pytest.raises(ContextualRouteInstanceMismatch):
        store.seal_contextual_admission(
            entry.session_key,
            "route-race",
            lambda *args: sealed.append(args) or True,
            expected_route_instance_id="route-instance-a",
        )

    assert sealed == []


@pytest.mark.asyncio
async def test_compression_route_refuses_to_publish_when_lease_rebind_is_blocked():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    token = object()
    state = SimpleNamespace(
        turn=SimpleNamespace(lease_token=token, lease_generation=7)
    )
    calls = []
    setattr(runner, "_peek_session_state", lambda session_key: state)
    setattr(runner, "_rebind_turn_lease", lambda *args: False)

    async def route(*_args):
        calls.append("route")
        return object()

    fake_store = object()
    setattr(runner, "session_store", fake_store)
    setattr(runner, "_async_session_store", SimpleNamespace(
        _store=fake_store,
        advance_compression_session=route,
    ))

    advanced = await runner._publish_gateway_compression_route(
        "telegram:u:c",
        "parent",
        "child",
        7,
    )

    assert advanced is None
    assert calls == []


@pytest.mark.asyncio
async def test_compression_route_rolls_lease_back_when_backing_cas_loses():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    state = SimpleNamespace(
        turn=SimpleNamespace(lease_token=object(), lease_generation=8)
    )
    rebinds = []
    setattr(runner, "_peek_session_state", lambda session_key: state)
    setattr(
        runner,
        "_rebind_turn_lease",
        lambda session_key, run_generation, new_session_id: (
            rebinds.append(new_session_id) or True
        ),
    )

    async def lose_cas(*_args):
        return None

    fake_store = object()
    setattr(runner, "session_store", fake_store)
    setattr(runner, "_async_session_store", SimpleNamespace(
        _store=fake_store,
        advance_compression_session=lose_cas,
    ))

    advanced = await runner._publish_gateway_compression_route(
        "telegram:u:c",
        "parent",
        "child",
        8,
    )

    assert advanced is None
    assert rebinds == ["child", "parent"]


@pytest.mark.asyncio
async def test_compression_route_rejects_success_after_turn_owner_is_invalidated():
    from gateway.run import GatewayRunner
    from gateway.turn_lease import SessionTurnLeaseRegistry

    runner = GatewayRunner.__new__(GatewayRunner)
    route = _entry()
    state = SimpleNamespace(
        turn=SimpleNamespace(lease_token=None, lease_generation=9)
    )
    registry = SessionTurnLeaseRegistry()
    token = await registry.acquire(
        route.session_id,
        owner_key=route.session_key,
        generation=9,
    )
    state.turn.lease_token = token
    runner_any = cast(Any, runner)
    runner_any._turn_leases = registry
    runner_any._peek_session_state = lambda _key: state

    class Store:
        def __init__(self):
            self.entry = route

        def peek_session_entry(self, _key):
            return self.entry

    store = Store()

    async def advance(*_args):
        advanced = replace(route, session_id="child", routing_revision=1)
        store.entry = advanced
        state.turn = SimpleNamespace(lease_token=None, lease_generation=10)
        registry.release(token)
        return advanced

    runner_any.session_store = store
    runner_any._async_session_store = SimpleNamespace(
        _store=store,
        advance_compression_session=advance,
    )

    published = await runner._publish_gateway_compression_route(
        route.session_key,
        route.session_id,
        "child",
        9,
    )

    assert published is None


def test_compression_route_cas_mutates_backing_entry_and_rolls_back_save_failure():
    import threading
    from datetime import datetime, timezone

    from gateway.session import SessionEntry, SessionStore

    store = SessionStore.__new__(SessionStore)
    store._lock = threading.Lock()
    store._loaded = True
    store._db = None
    store._routing_generation = 0
    store._entries = {
        "telegram:u:c": SessionEntry(
            session_key="telegram:u:c",
            session_id="parent",
            created_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
            updated_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
        )
    }
    store._save = lambda: None

    advanced = store.advance_compression_session(
        "telegram:u:c",
        "parent",
        "child",
    )
    assert advanced is not None
    assert advanced.session_id == "child"
    assert store._entries["telegram:u:c"].session_id == "child"
    assert store._entries["telegram:u:c"].routing_revision == 1

    def fail_save():
        raise OSError("disk full")

    store._save = fail_save
    with pytest.raises(OSError, match="disk full"):
        store.advance_compression_session(
            "telegram:u:c",
            "child",
            "grandchild",
        )
    assert store._entries["telegram:u:c"].session_id == "child"
    assert store._entries["telegram:u:c"].routing_revision == 1
def test_served_contextual_cron_profile_homes_are_deduplicated(
    monkeypatch, tmp_path
):
    import hermes_constants
    import hermes_cli.profiles as profiles

    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    primary.mkdir()
    secondary.mkdir()
    runner = cast(Any, object.__new__(GatewayRunner))
    runner.config = SimpleNamespace(multiplex_profiles=True)
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: primary)
    monkeypatch.setattr(
        profiles,
        "profiles_to_serve",
        lambda **_kwargs: [
            ("primary", primary),
            ("secondary", secondary),
            ("secondary-copy", secondary),
        ],
    )

    assert runner._served_contextual_cron_profile_homes() == [
        str(primary.resolve()),
        str(secondary.resolve()),
    ]


@pytest.mark.asyncio
async def test_contextual_startup_recovery_fences_resume_until_all_profiles_succeed(
    tmp_path,
):
    primary = str((tmp_path / "primary").resolve())
    secondary = str((tmp_path / "secondary").resolve())
    events: list[str] = []
    failing = {secondary}
    runner = cast(Any, object.__new__(GatewayRunner))
    runner._served_contextual_cron_profile_homes = lambda: [primary, secondary]

    async def recover(*, cron_home):
        events.append(f"recover:{cron_home}")
        if cron_home in failing:
            raise RuntimeError("secondary unavailable")
        return 1

    async def redeliver():
        events.append("redeliver")

    def resume():
        events.append("resume")

    async def finish_restore():
        events.append("release")

    runner._recover_contextual_cron_transcripts = recover
    runner._redeliver_pending_obligations = redeliver
    runner._schedule_resume_pending_sessions = resume
    runner._finish_startup_restore = finish_restore

    assert not await runner._release_startup_restore_after_contextual_recovery()
    assert events == [f"recover:{primary}", f"recover:{secondary}"]

    failing.clear()
    assert await runner._release_startup_restore_after_contextual_recovery()
    assert events == [
        f"recover:{primary}",
        f"recover:{secondary}",
        f"recover:{primary}",
        f"recover:{secondary}",
        "redeliver",
        "resume",
        "release",
    ]
