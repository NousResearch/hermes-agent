"""Rollover admission across real store, lane lease and native FIFO boundaries."""
from __future__ import annotations

import asyncio
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.rollover_packet import RolloverBlocked, RolloverPolicy, validate_packet
from gateway.run_rollover import LaneRollover, queue_during_rollover
from gateway.session import SessionSource, SessionStore
from gateway.turn_lease import SessionTurnLeaseRegistry
from hermes_state import SessionDB


def _packet(path, authority, mode="continue-authorized"):
    path.write_text(
        "# Handoff: Synthetic work\n\n## Continuation Directive\n"
        f"- **Standing mandate:** [Authority]({authority})\n"
        f"- **Current objective:** Continue only the task in [Authority]({authority})\n"
        f"- **Current mode:** `{mode}`\n"
        "- **Next action and authority:** Read the approved synthetic task; no external actions.\n"
        "- **Pending obligations:** none\n"
        "- **Completed work:** preparation only\n"
        "- **Superseded/stopped/cancelled instructions:** none\n"
        "- **Do not repeat until reconciled:** none\n",
        encoding="utf-8",
    )
    return f"# Handoff: Synthetic work\nLast Finished Session: test\nFirst read this: {path}\n\n- Continue only the approved task.  \n"


@pytest.fixture
def lane(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    import gateway.run_rollover as rollover
    import gateway.run as gateway_run
    from hermes_state import AsyncSessionDB

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    if store._db is not None:
        store._db.close()
    db = SessionDB(db_path=tmp_path / "state.db")
    store._db = db
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="owner")
    entry = store.get_or_create_session(source)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = store
    runner._session_db = AsyncSessionDB(db)
    runner._turn_leases = SessionTurnLeaseRegistry()
    runner._voice_mode = {}
    runner._agent_cache_lock = threading.RLock()
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._mark_durable_active_turn = AsyncMock()
    runner._clear_durable_active_turn = AsyncMock()
    adapter = SimpleNamespace(_pending_messages={}, _text_debounce_store=dict)
    adapter.get_pending_message = lambda key: adapter._pending_messages.pop(key, None)
    runner._delivery_adapter_for = lambda _: adapter
    key = entry.session_key
    generation = runner._begin_session_run_generation(key)
    policy = RolloverPolicy("handoff", tmp_path / "packets", "First read {packet}; no new authority.", 1000, 200)
    from ruamel.yaml import YAML
    with (tmp_path / "config.yaml").open("w") as config_file:
        YAML().dump({"session_rollover": {
            "enabled": True, "session_keys": [key], "handoff_skill": policy.handoff_skill,
            "packet_dir": str(policy.packet_dir), "bootstrap_prompt": policy.bootstrap_prompt,
            "runway_tokens": policy.runway_tokens, "safety_margin_tokens": policy.safety_margin_tokens,
        }}, config_file)
    guard = SimpleNamespace(phase="requested")
    state = runner._session_state(key)
    state.persistent.rollover = LaneRollover(policy, entry.session_id, guard)
    agent = SimpleNamespace(
        session_id=entry.session_id, _active_children_lock=threading.RLock(), _active_children=set(),
        _process_owner_task_ids=set(), _memory_manager=None, _session_messages=[],
        context_compressor=SimpleNamespace(on_session_end=lambda *_: None), release_clients=lambda: None,
    )
    state.turn.agent = agent
    runner._agent_cache = {key: (agent, "signature", 0)}
    authority = tmp_path / "authority.txt"
    authority.write_text("Only synthetic local work is authorized.", encoding="utf-8")
    calls = []
    authored = []

    async def model(prompt, source, current, key, generation, *, history):
        receipt = store.get_rollover(key)
        calls.append((receipt["phase"], current.session_id, list(history)))
        if receipt["phase"] == "authoring":
            message = _packet(Path(receipt["packet_path"]), authority)
            authored.append(message)
            return {
                "completed": True, "rollover_restart_message": message, "final_response": message,
                "rollover_transcript_watermark": db.get_active_message_watermark(current.session_id),
            }
        return {"completed": True, "final_response": "Recentered: continue-authorized; only the synthetic next action."}

    runner._rollover_internal_turn = model
    monkeypatch.setattr(rollover, "handoff_instruction", lambda *_: "Internal handoff authoring")
    data = SimpleNamespace(runner=runner, store=store, db=db, source=source, entry=entry,
                           key=key, generation=generation, adapter=adapter, calls=calls,
                           authored=authored, authority=authority, agent=agent)
    yield data
    executor = getattr(runner, "_housekeeping_executor", None)
    if executor is not None:
        executor.shutdown(wait=True)
    db.close()


async def _claim(data):
    token = await data.runner._turn_leases.acquire(
        data.entry.session_id, owner_key=data.key, generation=data.generation,
    )
    data.runner._session_state(data.key).turn.lease_tokens[data.generation] = token


async def _rotate(data):
    handled, response = await data.runner._rollover_after_turn(
        MessageEvent(text="accepted work", source=data.source), data.source, data.entry,
        data.key, data.generation, {"rollover_requested": True},
    )
    assert handled
    return response


@pytest.mark.asyncio
async def test_arrivals_during_offloop_commit_survive_and_wait_for_bootstrap(lane, monkeypatch):
    await _claim(lane)
    entered, release = threading.Event(), threading.Event()
    commit = lane.store.commit_rollover

    def held_commit(*args, **kwargs):
        entered.set()
        if not release.wait(10):
            raise RuntimeError("test did not release commit")
        return commit(*args, **kwargs)

    monkeypatch.setattr(lane.store, "commit_rollover", held_commit)
    task = asyncio.create_task(_rotate(lane))
    try:
        assert await asyncio.to_thread(entered.wait, 10), "cutover never reached the native store"
        # This runs while the DB worker is deliberately held: no event-loop blocking.
        one = MessageEvent(text="/unrecognized ordinary input", source=lane.source)
        two = MessageEvent(text="keep monitoring only", source=lane.source)
        assert queue_during_rollover(lane.runner, one, lane.key)
        assert queue_during_rollover(lane.runner, two, lane.key)
        assert not queue_during_rollover(
            lane.runner, MessageEvent(text="/stop", source=lane.source), lane.key,
        )
        release.set()
        reply = await task
        receipt = lane.store.get_rollover(lane.key)
        successor = lane.store.lookup_by_session_key(lane.key)
        assert receipt["phase"] == "admitting", reply
        assert receipt["restart_message"] == lane.authored[0]
        assert receipt["bootstrap_prompt"].endswith(lane.authored[0])
        assert successor.session_id != lane.entry.session_id
        assert lane.db.get_session(lane.entry.session_id)["end_reason"] == "session_reset"
        assert lane.db.get_session(successor.session_id)["parent_session_id"] == lane.entry.session_id
        assert lane.calls == [("authoring", lane.entry.session_id, []), ("bootstrapping", successor.session_id, [])]
        assert lane.adapter._pending_messages[lane.key] is one
        assert lane.runner._session_state(lane.key).conversation.queued_events == [two]
        assert lane.runner._held_turn_lease(lane.key, lane.generation) is not None
        # Native adapter drains in order. The first retained message cannot release
        # the directive while a newer input remains; the final one admits it once.
        from gateway.run import _dequeue_pending_event
        for arrival in (one, two):
            pending = _dequeue_pending_event(lane.adapter, lane.key)
            assert pending is arrival
            admitted, _, _ = lane.runner._hm_rescue_orphaned_fifo(
                pending, lane.source, False, lane.key,
            )
            assert admitted is arrival
            assert await lane.runner._run_agent_drain_pending(
                {"completed": True}, lane.adapter, lane.source, lane.key,
            ) == (None, None)
            recovered, _ = await lane.runner._rollover_recover_before_turn(
                arrival, lane.source, successor, lane.key, lane.generation,
            )
            assert recovered is False
            handled, _ = await lane.runner._rollover_after_turn(
                arrival, lane.source, successor, lane.key, lane.generation, {"completed": True},
            )
            assert handled is False  # the native caller still delivers each retained reply
        admission = _dequeue_pending_event(lane.adapter, lane.key)
        assert admission.internal and not admission.allow_gateway_control
        lane.runner._release_turn_lease(lane.key, lane.generation)
        lane.generation = lane.runner._begin_session_run_generation(lane.key)
        handled, _ = await lane.runner._rollover_recover_before_turn(
            admission, lane.source, successor, lane.key, lane.generation,
        )
        assert handled is True
        assert lane.store.get_rollover(lane.key)["phase"] == "complete"
        assert lane.calls[-1][0:2] == ("admitting", successor.session_id)
    finally:
        release.set()
        await task
        lane.runner._release_turn_lease(lane.key, lane.generation)


@pytest.mark.asyncio
async def test_input_during_packet_authoring_defers_without_losing_old_owner(lane):
    await _claim(lane)
    lane.store.set_session_yolo(lane.key, True)
    model = lane.runner._rollover_internal_turn
    arrival = MessageEvent(text="Stop; await my answer instead", source=lane.source)

    async def author_then_receive(*args, **kwargs):
        result = await model(*args, **kwargs)
        queue_during_rollover(lane.runner, arrival, lane.key)
        return result

    lane.runner._rollover_internal_turn = author_then_receive
    try:
        reply = await _rotate(lane)
        receipt = lane.store.get_rollover(lane.key)
        assert "deferred" in reply
        assert receipt["phase"] == "invalidated"
        assert lane.store.peek_session_id(lane.key) == lane.entry.session_id
        assert lane.store.lookup_by_session_key(lane.key).yolo is True
        assert lane.db.get_session(receipt["new_session_id"]) is None
        assert lane.adapter._pending_messages[lane.key] is arrival
    finally:
        lane.runner._release_turn_lease(lane.key, lane.generation)


@pytest.mark.asyncio
async def test_cleanup_failure_blocks_before_identity_commit(lane, monkeypatch):
    from tools import terminal_tool

    class BrokenEnvironment:
        def cleanup(self):
            raise OSError("sandbox cleanup failed")

    owner = terminal_tool._resolve_container_task_id(lane.entry.session_id)
    monkeypatch.setitem(terminal_tool._active_environments, owner, BrokenEnvironment())
    await _claim(lane)
    try:
        reply = await _rotate(lane)
        receipt = lane.store.get_rollover(lane.key)
        assert "sandbox cleanup failed" in reply
        assert receipt["phase"] == "blocked"
        assert receipt["owner_committed"] is False
        assert lane.store.peek_session_id(lane.key) == lane.entry.session_id
        assert lane.db.get_session(lane.entry.session_id)["end_reason"] is None
        assert lane.db.get_session(receipt["new_session_id"]) is None
    finally:
        lane.runner._release_turn_lease(lane.key, lane.generation)


@pytest.mark.asyncio
async def test_pending_worker_completion_prevents_cutover(lane):
    lane.db._conn.execute(
        "INSERT INTO async_delegations(delegation_id, origin_session, parent_session_id, state, "
        "dispatched_at, updated_at, delivery_state) VALUES (?, ?, ?, 'completed', 1, 1, 'pending')",
        ("work-unit", lane.key, lane.entry.session_id),
    )
    await _claim(lane)
    try:
        reply = await _rotate(lane)
        assert "completion has not been reconciled" in reply
        assert lane.store.get_rollover(lane.key) is None
        assert lane.calls == []
        assert lane.store.peek_session_id(lane.key) == lane.entry.session_id
    finally:
        lane.runner._release_turn_lease(lane.key, lane.generation)


def test_packet_admission_preserves_message_and_rejects_unresolved_authority(tmp_path):
    authority = tmp_path / "authority.txt"
    authority.write_text("Await the user's answer.", encoding="utf-8")
    packet = tmp_path / "handoff.md"
    message = _packet(packet, authority, mode="await-user")
    binding = validate_packet(packet, message)
    assert binding["mode"] == "await-user"
    assert binding["restart_message"].endswith("task.  \n")
    authority.unlink()
    with pytest.raises(RolloverBlocked, match="Unresolved handoff reference"):
        validate_packet(packet, message)


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_independent_infrastructure_remains_running_after_rollover(lane):
    import shlex
    import sys
    from tools.process_registry import process_registry

    command = f"{shlex.quote(sys.executable)} -c 'import threading; threading.Event().wait(30)'"
    process = process_registry.spawn_local(
        command, cwd=str(lane.authority.parent), task_id=lane.entry.session_id,
        owner_task_id=lane.entry.session_id, session_key=lane.key,
    )
    await _claim(lane)
    try:
        assert process.process.poll() is None
        reply = await _rotate(lane)
        assert lane.store.get_rollover(lane.key)["phase"] == "complete", reply
        assert process.process.poll() is None
        assert lane.store.peek_session_id(lane.key) != lane.entry.session_id
    finally:
        process_registry.kill_process(process.id, consume_output=True)
        lane.runner._release_turn_lease(lane.key, lane.generation)


@pytest.mark.asyncio
async def test_internal_handoff_keeps_redacted_prefix_and_streamed_final_is_not_resent(lane):
    from dataclasses import replace
    from ruamel.yaml import YAML
    from gateway.run_rollover import GatewayRolloverMixin
    from gateway.session import build_session_context

    config_path = lane.authority.parent / "config.yaml"
    yaml = YAML()
    config = yaml.load(config_path)
    config["privacy"] = {"redact_pii": True}
    with config_path.open("w") as handle:
        yaml.dump(config, handle)
    source = replace(lane.source, user_id="438901267", chat_id="894215679")
    context = build_session_context(source, lane.runner.config, lane.entry)
    pinned = lane.runner._pinned_session_context_prompt(context, True, lane.key)
    assert source.user_id not in pinned and source.chat_id not in pinned
    captured = []

    async def provider_boundary(**kwargs):
        captured.append(kwargs["context_prompt"])
        return {"completed": True, "already_sent": True, "final_response": "Recentered.",
                "messages": [], "media_already_delivered": True}

    lane.runner._run_agent = provider_boundary
    result = await GatewayRolloverMixin._rollover_internal_turn(
        lane.runner, "Read the bound handoff.", source, lane.entry,
        lane.key, lane.generation, history=[],
    )
    assert captured == [pinned]
    event = MessageEvent(text="accepted input", source=source)
    response = await lane.runner._rollover_deliver_result(
        result, event, source, lane.entry, lane.key, lane.generation,
    )
    assert response is None
    assert event._streamed_final_response == "Recentered."


@pytest.mark.asyncio
async def test_pending_approval_prevents_cutover_without_answering_it(lane, monkeypatch):
    from tools.approval import _gateway_queues
    from tools.approval_gateway_wait import _ApprovalEntry

    approval = _ApprovalEntry({"command": "synthetic consequential operation"})
    monkeypatch.setitem(_gateway_queues, lane.key, [approval])
    await _claim(lane)
    try:
        reply = await _rotate(lane)
        assert "approval or unanswered question" in reply
        assert not approval.event.is_set()
        assert lane.store.get_rollover(lane.key) is None
        assert lane.store.peek_session_id(lane.key) == lane.entry.session_id
    finally:
        lane.runner._release_turn_lease(lane.key, lane.generation)


@pytest.mark.asyncio
async def test_startup_recovers_healthy_successor_despite_other_lane_failure(lane, monkeypatch):
    import hermes_constants
    from gateway.config import PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from gateway.run import GatewayRunner
    from hermes_state import AsyncSessionDB

    class Transport(BasePlatformAdapter):
        async def connect(self):
            raise AssertionError("No real transport is allowed")

        async def disconnect(self):
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            return SendResult(success=True, message_id="recovery-reply")

        async def send_typing(self, chat_id, metadata=None):
            return None

        async def get_chat_info(self, chat_id):
            return {"id": chat_id}

    home = lane.authority.parent
    broken = lane.store.get_or_create_session(
        SessionSource(platform=Platform.TELEGRAM, chat_id="broken-route", user_id=lane.source.user_id),
    )
    successors = {}
    for entry in (broken, lane.entry):
        packet = home / f"recovery-{entry.session_id}.md"
        message = _packet(packet, lane.authority, mode="await-user")
        receipt = lane.store.begin_rollover(
            entry.session_key, entry.session_id, lane.generation, profile_home=str(home), packet_path=str(packet),
        )
        binding = validate_packet(packet, message)
        binding.pop("mode")
        binding.update(
            profile_home=str(home), bootstrap_prompt="Read the bound packet.\n\n" + message,
            transcript_watermark=lane.db.get_active_message_watermark(entry.session_id),
        )
        lane.store.prepare_rollover(
            entry.session_key, entry.session_id, lane.generation, binding,
            expected_new_session_id=receipt["new_session_id"],
        )
        successors[entry.session_key] = lane.store.commit_rollover(
            entry.session_key, entry.session_id, lane.generation,
            expected_new_session_id=receipt["new_session_id"],
        )
    successor = successors[lane.key]
    config = GatewayConfig(sessions_dir=home / "sessions")
    runner = GatewayRunner(config)
    runner.session_store = lane.store
    runner._session_db = AsyncSessionDB(lane.db)
    adapter = Transport(PlatformConfig(enabled=True), Platform.TELEGRAM)
    adapter.gateway_runner = runner
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._is_user_authorized_for_source = lambda source: source.user_id == lane.source.user_id
    monkeypatch.setattr(hermes_constants, "get_routing_process_hermes_home", lambda: home)
    list_pending = lane.store.list_pending_rollovers
    restore_source = runner._restored_source

    def ordered_pending(**kwargs):
        return sorted(list_pending(**kwargs), key=lambda row: row["session_key"] != broken.session_key)

    def restore_available_source(entry):
        if entry.session_key == broken.session_key:
            raise RuntimeError("The first lane's source is unavailable")
        return restore_source(entry)

    monkeypatch.setattr(lane.store, "list_pending_rollovers", ordered_pending)
    monkeypatch.setattr(runner, "_restored_source", restore_available_source)
    calls = []

    async def model_boundary(**kwargs):
        calls.append((kwargs["session_id"], kwargs["history"]))
        return {"completed": True, "final_response": "Awaiting the existing question.", "messages": []}

    runner._run_agent = model_boundary
    adapter.set_message_handler(runner._handle_message)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    try:
        await runner._restore_rollover_sessions(recover_interrupted=True)
        assert lane.store.get_rollover(lane.key)["phase"] == "complete"
        assert lane.store.peek_session_id(lane.key) == successor.session_id
        assert calls == [(successor.session_id, [])]
        assert lane.store.get_rollover(broken.session_key)["phase"] == "committed"
        await runner._restore_rollover_sessions(recover_interrupted=False)
        assert calls == [(successor.session_id, [])]
    finally:
        runner.close_all_session_db_handles()
        for name in ("_agent_executor", "_housekeeping_executor"):
            executor = getattr(runner, name, None)
            if executor is not None:
                executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_replaced_admitting_lane_does_not_suppress_native_hygiene(lane):
    from ruamel.yaml import YAML
    from gateway.run_rollover import lane_rollover

    lane_rollover(lane.runner, lane.key).phase = "admitting"
    replacement = lane.store.reset_session(lane.key)
    with (lane.authority.parent / "config.yaml").open("w") as handle:
        YAML().dump({"session_rollover": {"enabled": False}}, handle)
    event = MessageEvent(text="new conversation", source=lane.source)
    handled, _ = await lane.runner._rollover_recover_before_turn(
        event, lane.source, replacement, lane.key, lane.generation,
    )
    assert handled is False
    assert await lane.runner._rollover_before_hygiene(
        event, lane.source, replacement, lane.key, [],
    ) is False
    assert lane_rollover(lane.runner, lane.key) is None


@pytest.mark.asyncio
async def test_bind_refusal_does_not_record_an_executed_or_partial_turn(lane):
    from gateway.rollover_packet import RolloverPreflightBlocked
    from gateway.run_rollover import bind_rollover_guard

    context = SimpleNamespace(session_key=lane.key, moa_config=None)
    agent = SimpleNamespace(session_id=lane.entry.session_id, api_mode="codex_app_server")
    with pytest.raises(RolloverPreflightBlocked) as refusal:
        bind_rollover_guard(lane.runner, context, agent)
    prepared = SimpleNamespace(
        history=[], message_text="accepted input", persist_user_message=None,
        persist_user_timestamp=None, persist_user_display_kind=None,
        persistence_owner="unstarted-turn", persistence_session_id=lane.entry.session_id,
    )
    await lane.runner._hmwa_agent_error_reply(
        refusal.value, MessageEvent(text="accepted input", source=lane.source),
        lane.source, lane.entry, lane.key, prepared,
    )
    assert lane.store.load_transcript(lane.entry.session_id) == []
    assert lane.db.get_session(lane.entry.session_id)["message_count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["authoring", "bootstrapping", "admitting"])
async def test_cancelled_transition_blocks_its_owner_without_replay(lane, phase):
    entered = asyncio.Event()
    unsettled = asyncio.Event()
    original = lane.runner._rollover_internal_turn

    async def cancellable_model(*args, **kwargs):
        if lane.store.get_rollover(lane.key)["phase"] == phase:
            entered.set()
            await unsettled.wait()
        return await original(*args, **kwargs)

    lane.runner._rollover_internal_turn = cancellable_model
    await _claim(lane)
    task = asyncio.create_task(_rotate(lane))
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        receipt = lane.store.get_rollover(lane.key)
        assert receipt["phase"] == "blocked"
        assert receipt["owner_committed"] is (phase != "authoring")
        owner = receipt["old_session_id"] if phase == "authoring" else receipt["new_session_id"]
        entry = lane.store.lookup_by_session_key(lane.key)
        assert entry.session_id == owner
        calls = list(lane.calls)
        handled, _ = await lane.runner._rollover_recover_before_turn(
            MessageEvent(text="do not replay", source=lane.source), lane.source,
            entry, lane.key, lane.generation,
        )
        assert handled is True and lane.calls == calls
    finally:
        unsettled.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        lane.runner._release_turn_lease(lane.key, lane.generation)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_during", ["bootstrap", "before_admission"])
async def test_changed_packet_blocks_same_successor_before_continuation(lane, changed_during):
    def change_packet():
        receipt = lane.store.get_rollover(lane.key)
        packet = Path(receipt["packet_path"])
        packet.write_text(packet.read_text().replace("`continue-authorized`", "`idle`"))

    original_model = lane.runner._rollover_internal_turn
    original_admit = lane.runner._rollover_admit

    async def model(*args, **kwargs):
        result = await original_model(*args, **kwargs)
        if changed_during == "bootstrap" and lane.store.get_rollover(lane.key)["phase"] == "bootstrapping":
            change_packet()
        return result

    async def admit(*args, **kwargs):
        if changed_during == "before_admission":
            change_packet()
        return await original_admit(*args, **kwargs)

    lane.runner._rollover_internal_turn = model
    lane.runner._rollover_admit = admit
    await _claim(lane)
    try:
        await _rotate(lane)
        receipt = lane.store.get_rollover(lane.key)
        assert receipt["phase"] == "blocked"
        assert receipt["owner_committed"] is True
        assert lane.store.peek_session_id(lane.key) == receipt["new_session_id"]
        assert not any(phase == "admitting" for phase, *_ in lane.calls)
    finally:
        lane.runner._release_turn_lease(lane.key, lane.generation)


@pytest.mark.asyncio
async def test_recovery_with_changed_packet_blocks_without_requeue_loop(lane):
    class CrashBeforeBootstrap(BaseException):
        pass

    bootstrap = lane.runner._rollover_bootstrap

    async def crash_before_bootstrap(*_args):
        raise CrashBeforeBootstrap

    lane.runner._rollover_bootstrap = crash_before_bootstrap
    await _claim(lane)
    try:
        with pytest.raises(CrashBeforeBootstrap):
            await _rotate(lane)
        receipt = lane.store.get_rollover(lane.key)
        assert receipt["phase"] == "committed"
        lane.runner._release_turn_lease(lane.key, lane.generation)
        lane.generation = lane.runner._begin_session_run_generation(lane.key)
        lane.runner._session_state(lane.key).persistent.rollover = None
        lane.runner._rollover_bootstrap = bootstrap
        Path(receipt["packet_path"]).write_text("The bound packet was replaced.")
        entry = lane.store.lookup_by_session_key(lane.key)
        arrival = MessageEvent(text="accepted after restart", source=lane.source)
        handled, _ = await lane.runner._rollover_recover_before_turn(
            arrival, lane.source, entry, lane.key, lane.generation,
        )
        assert handled is True
        assert lane.store.get_rollover(lane.key)["phase"] == "blocked"
        assert lane.store.peek_session_id(lane.key) == receipt["new_session_id"]
        pending = lane.adapter.get_pending_message(lane.key)
        assert pending is arrival
        calls = list(lane.calls)
        handled, _ = await lane.runner._rollover_recover_before_turn(
            pending, lane.source, entry, lane.key, lane.generation,
        )
        assert handled is True and lane.calls == calls
        assert lane.key not in lane.adapter._pending_messages
    finally:
        lane.runner._release_turn_lease(lane.key, lane.generation)
