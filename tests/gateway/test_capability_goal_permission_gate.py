from __future__ import annotations

import asyncio
import dataclasses
import json
import time
import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.discord_connector_protocol import (
    DiscordConnectorEvent,
    DiscordConnectorKind,
    DiscordConnectorTarget,
    DiscordConnectorTargetType,
)
from gateway.platforms.base import MessageEvent, MessageType
from gateway.relay.discord_connector_transport import (
    DiscordConnectorRelayTransport,
    _event_to_gateway,
    authenticated_discord_connector_ingress,
)
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CapabilityDescriptor
from gateway.session import (
    SessionEntry,
    SessionSource,
    SessionStore,
    build_session_key,
)


GUILD_ID = "1282725267068157972"
ROOT_CHANNEL_ID = "1526858760100909066"
OWNER_ID = "1279454038731264061"
THREAD_ID = "1527000000000000001"
RECEIPT_SHA256 = "a" * 64


def _runtime_binding() -> dict[str, object]:
    now = int(time.time() * 1_000)
    return {
        "run_id": "capability-goal-test-run",
        "fixture_sha256": "1" * 64,
        "release_sha": "2" * 40,
        "capability_plan_sha256": "3" * 64,
        "full_canary_plan_sha256": "4" * 64,
        "fixture_publication_receipt_sha256": "5" * 64,
        "valid_from_unix_ms": now - 60_000,
        "valid_until_unix_ms": now + 60_000,
        "guild_id": GUILD_ID,
        "root_channel_id": ROOT_CHANNEL_ID,
        "owner_user_id": OWNER_ID,
    }


@pytest.fixture(autouse=True)
def _isolated_goal_home(tmp_path, monkeypatch):
    """Keep every synthetic GoalManager row outside the operator's state DB."""

    from hermes_cli import goals

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("INVOCATION_ID", "d" * 32)
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


class _Adapter:
    def __init__(self) -> None:
        self._pending_messages: dict[str, MessageEvent] = {}
        self.sends: list[str] = []

    async def send(self, _chat_id, content, **_kwargs):
        self.sends.append(content)
        return SimpleNamespace(success=True, message_id="1527000000000000999")


def _connector_event(
    text: str = "/goal finish the exact canary acceptance",
    *,
    event_id: str = "1527000000000000100",
    guild_id: str = GUILD_ID,
    channel_id: str = ROOT_CHANNEL_ID,
    owner_id: str = OWNER_ID,
    target_type: DiscordConnectorTargetType = (
        DiscordConnectorTargetType.PUBLIC_GUILD_CHANNEL
    ),
    parent_channel_id: str | None = None,
) -> MessageEvent:
    target = DiscordConnectorTarget(
        target_type=target_type,
        guild_id=guild_id,
        channel_id=channel_id,
        parent_channel_id=parent_channel_id,
    )
    event = DiscordConnectorEvent.from_mapping(
        {
            "event_id": event_id,
            "target": target.to_mapping(),
            "author_id": owner_id,
            "author_name": "Emo",
            "author_is_bot": False,
            "content": text,
            "created_at_unix_ms": int(time.time() * 1_000),
            "reply_to_message_id": None,
        }
    )
    return _event_to_gateway(
        event,
        delivery_id=str(uuid.uuid4()),
        delivery_receipt_sha256=RECEIPT_SHA256,
    )


def _runner(session_id: str) -> tuple[object, _Adapter]:
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._require_capability_canary = True
    runner._require_production_model_sovereignty = False
    runner._startup_restore_in_progress = False
    runner._draining = False
    runner.config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, token="redacted")
        }
    )
    adapter = _Adapter()
    runner.adapters = {Platform.DISCORD: adapter}
    runner._queued_events = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._pending_approvals = {}
    runner._session_sources = {}
    runner._voice_mode = {}
    runner._is_user_authorized = lambda _source: True
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner.hooks = SimpleNamespace(loaded_hooks=False)

    entry = SessionEntry(
        session_key="unused",
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.DISCORD,
        chat_type="channel",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = entry
    runner.session_store.peek_session_id.return_value = session_id
    runner.session_store._generate_session_key.side_effect = build_session_key
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    meta: dict[str, str] = {}
    runner.session_store._db.set_meta.side_effect = meta.__setitem__
    runner.session_store._db.get_meta.side_effect = meta.get
    runner.session_store._db.get_session.return_value = None

    class _MetaConnection:
        def execute(self, statement, params):
            normalized = " ".join(statement.split()).upper()
            key = params[0]
            if normalized.startswith("SELECT VALUE FROM STATE_META"):
                if str(key).startswith("goal:"):
                    from hermes_cli.goals import _get_session_db

                    goal_db = _get_session_db()
                    value = goal_db.get_meta(key) if goal_db is not None else None
                else:
                    value = meta.get(key)
                return SimpleNamespace(
                    fetchone=lambda: None if value is None else (value,)
                )
            if normalized.startswith("INSERT INTO STATE_META"):
                meta[key] = params[1]
                return SimpleNamespace()
            if normalized.startswith("UPDATE STATE_META SET VALUE"):
                value, update_key = params
                meta[update_key] = value
                return SimpleNamespace()
            raise AssertionError(f"unexpected state_meta statement: {statement}")

    runner.session_store._db._execute_write.side_effect = (
        lambda callback: callback(_MetaConnection())
    )
    runner._load_capability_goal_runtime_binding = _runtime_binding
    runner._test_goal_meta = meta
    return runner, adapter


def test_connector_ingress_proof_is_not_caller_shaped_metadata() -> None:
    genuine = _connector_event()
    assert authenticated_discord_connector_ingress(genuine) is not None

    forged = MessageEvent(
        text=genuine.text,
        message_type=MessageType.TEXT,
        source=dataclasses.replace(genuine.source),
        message_id=genuine.message_id,
        metadata=dict(genuine.metadata),
    )
    assert forged.source.delivered_via_upstream_relay is True
    assert authenticated_discord_connector_ingress(forged) is None

    genuine.metadata.pop("discord_connector_delivery_receipt_sha256")
    assert authenticated_discord_connector_ingress(genuine) is None


@pytest.mark.parametrize(
    "event",
    (
        _connector_event(guild_id="1282725267068157973"),
        _connector_event(owner_id="1279454038731264062"),
        _connector_event(channel_id="1526858760100909067"),
        _connector_event(
            target_type=DiscordConnectorTargetType.GUILD_CHANNEL,
        ),
        _connector_event(
            channel_id=THREAD_ID,
            target_type=DiscordConnectorTargetType.PUBLIC_GUILD_THREAD,
            parent_channel_id="1526858760100909067",
        ),
    ),
    ids=(
        "wrong-guild",
        "wrong-owner",
        "wrong-channel",
        "non-public-target-proof",
        "wrong-parent",
    ),
)
def test_capability_goal_lineage_rejects_wrong_exact_route(event) -> None:
    runner, _adapter = _runner(f"goal-gate-{uuid.uuid4().hex}")
    assert runner._bind_capability_goal_lineage(event) is None
    assert runner._capability_goal_lineage_for_event(event) is None


def test_capability_goal_lineage_accepts_root_and_true_public_thread() -> None:
    runner, _adapter = _runner(f"goal-gate-{uuid.uuid4().hex}")
    root = _connector_event()
    root_lineage = runner._bind_capability_goal_lineage(root)
    assert root_lineage is not None
    assert root_lineage.root_channel_id == ROOT_CHANNEL_ID

    thread = _connector_event(
        channel_id=THREAD_ID,
        target_type=DiscordConnectorTargetType.PUBLIC_GUILD_THREAD,
        parent_channel_id=ROOT_CHANNEL_ID,
    )
    thread_lineage = runner._bind_capability_goal_lineage(thread)
    assert thread_lineage is not None
    assert thread_lineage.channel_id == THREAD_ID
    assert thread_lineage.root_channel_id == ROOT_CHANNEL_ID


def test_capability_goal_lineage_rejects_missing_receipt_digest() -> None:
    runner, _adapter = _runner(f"goal-gate-{uuid.uuid4().hex}")
    event = _connector_event()
    event.metadata.pop("discord_connector_delivery_receipt_sha256")

    assert authenticated_discord_connector_ingress(event) is None
    assert runner._bind_capability_goal_lineage(event) is None


@pytest.mark.asyncio
async def test_general_capability_canary_goal_remains_disabled() -> None:
    runner, _adapter = _runner(f"goal-gate-{uuid.uuid4().hex}")
    forged = MessageEvent(
        text="/goal caller-shaped attempt",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id=ROOT_CHANNEL_ID,
            chat_type="channel",
            user_id=OWNER_ID,
            user_name="Emo",
            scope_id=GUILD_ID,
            is_bot=False,
            delivered_via_upstream_relay=True,
        ),
        message_id="1527000000000000101",
        metadata={
            "discord_connector_delivery_id": str(uuid.uuid4()),
            "discord_connector_delivery_receipt_sha256": RECEIPT_SHA256,
            "discord_connector_event_sha256": "b" * 64,
            "discord_connector_event_created_at_unix_ms": 1,
            "discord_connector_target_type": "guild_channel",
        },
    )

    result = await runner._handle_message(forged)

    assert result == (
        "Standing-goal commands are disabled in this bounded capability canary."
    )


@pytest.mark.asyncio
async def test_direct_goal_control_cannot_mutate_without_sealed_lineage() -> None:
    from hermes_cli.goals import GoalManager

    session_id = f"goal-gate-{uuid.uuid4().hex}"
    runner, _adapter = _runner(session_id)
    GoalManager(session_id).set("must remain active")
    event = MessageEvent(
        text="/goal clear",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id=ROOT_CHANNEL_ID,
            chat_type="channel",
            user_id=OWNER_ID,
            user_name="Emo",
            scope_id=GUILD_ID,
            delivered_via_upstream_relay=True,
        ),
    )

    assert await runner._handle_goal_command(event) == (
        "Standing-goal commands are disabled in this bounded capability canary."
    )
    assert GoalManager(session_id).is_active()


@pytest.mark.asyncio
async def test_generic_internal_event_cannot_inherit_source_seal_or_clear_goal() -> None:
    from hermes_cli.goals import GoalManager

    session_id = f"goal-gate-{uuid.uuid4().hex}"
    runner, _adapter = _runner(session_id)
    ingress = _connector_event(text="authenticated owner ingress")
    lineage = runner._bind_capability_goal_lineage(ingress)
    assert lineage is not None
    GoalManager(session_id).set("must survive generic internal input")

    synthetic = MessageEvent(
        text="/goal clear",
        message_type=MessageType.TEXT,
        source=ingress.source,
        internal=True,
    )
    assert authenticated_discord_connector_ingress(synthetic) is None
    assert runner._bind_capability_goal_lineage(synthetic) is None
    assert runner._capability_goal_lineage_for_event(synthetic) is None
    assert (
        await runner._handle_active_session_busy_message(
            synthetic,
            build_session_key(synthetic.source),
        )
        is True
    )
    assert await runner._handle_goal_command(synthetic) == (
        "Standing-goal commands are disabled in this bounded capability canary."
    )
    assert await runner._handle_message(synthetic) is None
    assert GoalManager(session_id).is_active()


@pytest.mark.asyncio
async def test_busy_exact_connector_event_is_sealed_before_fifo_queue() -> None:
    session_id = f"goal-gate-{uuid.uuid4().hex}"
    runner, _adapter = _runner(session_id)
    runner._busy_text_mode = "queue"
    runner._busy_input_mode = "interrupt"
    event = _connector_event(text="real owner follow-up")

    assert runner._capability_goal_lineage_for_event(event) is None
    assert (
        await runner._handle_active_session_busy_message(
            event,
            build_session_key(event.source),
        )
        is False
    )
    assert runner._capability_goal_lineage_for_event(event) is not None


@pytest.mark.asyncio
async def test_exact_connector_goal_creation_preserves_zero_turn_cap(
    monkeypatch,
) -> None:
    from hermes_cli import goals

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"goals": {"max_turns": 0}},
    )
    session_id = f"goal-gate-{uuid.uuid4().hex}"
    runner, adapter = _runner(session_id)
    event = _connector_event()

    response = await runner._handle_message(event)

    state = goals.GoalManager(session_id).state
    assert state is not None
    assert state.max_turns == 0
    assert state.status == "active"
    assert "no automatic turn cap" in response
    kickoff = next(iter(adapter._pending_messages.values()))
    # The canary wrapper identifies and seals the exact newly queued kickoff
    # before returning from the authenticated /goal command. Generic events
    # that merely share this source never receive this event-level seal.
    assert runner._capability_goal_lineage_for_event(kickoff) is not None
    assert runner._bind_capability_goal_lineage(kickoff) is not None


@pytest.mark.asyncio
async def test_goal_start_fails_closed_and_removes_kickoff_when_record_write_fails(
) -> None:
    from hermes_cli.goals import GoalManager

    session_id = f"goal-gate-{uuid.uuid4().hex}"
    runner, adapter = _runner(session_id)
    runner._load_capability_goal_runtime_binding = lambda: (_ for _ in ()).throw(
        RuntimeError("reviewed runtime unavailable")
    )

    response = await runner._handle_message(_connector_event())

    state = GoalManager(session_id).state
    assert state is not None and state.status == "cleared"
    assert "was not started" in response
    assert adapter._pending_messages == {}
    assert runner._queued_events == {}


@pytest.mark.asyncio
async def test_goal_status_does_not_replace_first_authenticated_lineage() -> None:
    from hermes_cli.goals import GoalManager

    session_id = f"goal-gate-{uuid.uuid4().hex}"
    runner, _adapter = _runner(session_id)
    first = _connector_event(text="/goal preserve the first lineage")
    first_lineage = runner._bind_capability_goal_lineage(first)
    assert first_lineage is not None
    GoalManager(session_id).set("preserve the first lineage")
    entry = runner.session_store.get_or_create_session(first.source)
    entry.session_key = build_session_key(first.source)
    entry.origin = first.source
    original = await runner._persist_capability_goal_lineage_record(
        session_entry=entry,
        lineage=first_lineage,
    )

    status = _connector_event(
        text="/goal status",
        event_id="1527000000000000103",
    )
    assert runner._bind_capability_goal_lineage(status) is not None
    assert "Goal" in await runner._handle_goal_command(status)

    preserved = json.loads(
        runner._test_goal_meta[f"capability_goal_lineage:{session_id}"]
    )
    assert preserved == original
    assert preserved["first_event_id"] == first.message_id


def _begin_outcome(session_id: str, outcome: str) -> tuple[str, str]:
    from hermes_cli.goals import GoalManager

    manager = GoalManager(session_id, default_max_turns=0)
    manager.set("complete the canary lifecycle", max_turns=0)
    turn_id = f"turn-{uuid.uuid4().hex}"
    generation_id = manager.begin_model_turn(turn_id)
    assert manager.record_model_outcome(
        outcome,
        f"model authored {outcome}",
        originating_turn_id=turn_id,
        goal_generation_id=generation_id,
    )
    return turn_id, generation_id


@pytest.mark.asyncio
async def test_sealed_goal_outcome_continue_then_complete() -> None:
    from gateway.run import _capability_goal_finalization_meta_key
    from hermes_cli.goals import GoalManager

    continue_session = f"goal-gate-{uuid.uuid4().hex}"
    runner, adapter = _runner(continue_session)
    ingress = _connector_event(text="work on the goal")
    lineage = runner._bind_capability_goal_lineage(ingress)
    assert lineage is not None
    turn_id, generation_id = _begin_outcome(continue_session, "continue")
    entry = runner.session_store.get_or_create_session(ingress.source)
    entry.session_key = build_session_key(ingress.source)
    entry.origin = ingress.source
    await runner._persist_capability_goal_lineage_record(
        session_entry=entry,
        lineage=lineage,
    )

    await runner._finalize_gateway_goal_turn(
        source=ingress.source,
        response={
            "final_response": "first concrete step complete",
            "session_id": continue_session,
            "turn_id": turn_id,
            "goal_generation_id": generation_id,
        },
    )

    continuation = next(iter(adapter._pending_messages.values()))
    assert runner._is_sealed_capability_goal_continuation(continuation)
    state = GoalManager(continue_session).state
    assert state is not None and state.status == "active"
    assert state.turns_used == 1
    first_before_raw = runner._test_goal_meta[
        _capability_goal_finalization_meta_key(
            continue_session,
            turn_id,
            "before",
        )
    ]
    first_after_raw = runner._test_goal_meta[
        _capability_goal_finalization_meta_key(
            continue_session,
            turn_id,
            "after",
        )
    ]
    first_before = json.loads(first_before_raw)
    first_after = json.loads(first_after_raw)
    assert (
        first_before["schema"]
        == "muncho-capability-goal-finalization-intent.v1"
    )
    assert first_after["schema"] == "muncho-capability-goal-finalization.v1"
    assert first_after["intent_sha256"] == first_before["intent_sha256"]
    assert "complete the canary lifecycle" not in first_before_raw
    assert "model authored continue" not in first_before_raw
    assert "complete the canary lifecycle" not in first_after_raw
    assert "model authored continue" not in first_after_raw

    # Simulate the FIFO drain consuming the sealed automatic event, then let
    # the same standing goal's next model turn author its exact completion.
    adapter._pending_messages.clear()
    manager = GoalManager(continue_session, default_max_turns=0)
    complete_turn = f"turn-{uuid.uuid4().hex}"
    complete_generation = manager.begin_model_turn(complete_turn)
    assert manager.record_model_outcome(
        "complete",
        "model-authored verification is complete",
        originating_turn_id=complete_turn,
        goal_generation_id=complete_generation,
    )
    await runner._finalize_gateway_goal_turn(
        source=continuation.source,
        response={
            "final_response": "verified complete",
            "session_id": continue_session,
            "turn_id": complete_turn,
            "goal_generation_id": complete_generation,
        },
    )
    complete_state = GoalManager(continue_session).state
    assert complete_state is not None and complete_state.status == "done"
    assert complete_state.turns_used == 2
    assert (
        runner._test_goal_meta[
            f"capability_goal_lineage:{continue_session}"
        ]
        == ""
    )
    assert adapter._pending_messages == {}
    second_before_raw = runner._test_goal_meta[
        _capability_goal_finalization_meta_key(
            continue_session,
            complete_turn,
            "before",
        )
    ]
    second_after_raw = runner._test_goal_meta[
        _capability_goal_finalization_meta_key(
            continue_session,
            complete_turn,
            "after",
        )
    ]
    second_before = json.loads(second_before_raw)
    second_after = json.loads(second_after_raw)
    assert first_after["state_after"] == second_before["state_before"]
    assert (
        first_after["state_after_sha256"]
        == second_before["state_before_sha256"]
    )
    assert not any(
        "pending" in key or "active_model_turn" in key
        for key in first_after["state_after"]
    )
    assert second_after["decision_verdict"] == "done"
    assert "complete the canary lifecycle" not in second_before_raw
    assert "model-authored verification is complete" not in second_before_raw
    assert "complete the canary lifecycle" not in second_after_raw
    assert "model-authored verification is complete" not in second_after_raw


@pytest.mark.asyncio
async def test_terminal_clear_preserves_concurrently_resumed_generation() -> None:
    from hermes_cli.goals import GoalManager

    session_id = f"goal-gate-{uuid.uuid4().hex}"
    runner, _adapter = _runner(session_id)
    ingress = _connector_event(text="finish the current generation")
    lineage = runner._bind_capability_goal_lineage(ingress)
    assert lineage is not None
    manager = GoalManager(session_id, default_max_turns=0)
    manager.set("finish without erasing a newer resume", max_turns=0)
    turn_id = f"turn-{uuid.uuid4().hex}"
    generation_id = manager.begin_model_turn(turn_id)
    assert manager.record_model_outcome(
        "complete",
        "current generation is complete",
        originating_turn_id=turn_id,
        goal_generation_id=generation_id,
    )
    entry = runner.session_store.get_or_create_session(ingress.source)
    entry.session_key = build_session_key(ingress.source)
    entry.origin = ingress.source
    await runner._persist_capability_goal_lineage_record(
        session_entry=entry,
        lineage=lineage,
    )

    original_complete = runner._complete_capability_goal_finalization_receipt
    resumed_generation: dict[str, str] = {}

    async def _resume_during_receipt_write(**kwargs):
        receipt = await original_complete(**kwargs)
        resumed = GoalManager(session_id, default_max_turns=0).resume()
        assert resumed is not None and resumed.status == "active"
        resumed_generation["id"] = resumed.generation_id
        await runner._persist_capability_goal_lineage_record(
            session_entry=entry,
            lineage=lineage,
        )
        return receipt

    runner._complete_capability_goal_finalization_receipt = (
        _resume_during_receipt_write
    )
    await runner._finalize_gateway_goal_turn(
        source=ingress.source,
        response={
            "final_response": "verified terminal result",
            "session_id": session_id,
            "turn_id": turn_id,
            "goal_generation_id": generation_id,
        },
    )

    current = GoalManager(session_id).state
    assert current is not None and current.status == "active"
    assert current.generation_id == resumed_generation["id"]
    lineage_raw = runner._test_goal_meta[
        f"capability_goal_lineage:{session_id}"
    ]
    assert lineage_raw
    assert json.loads(lineage_raw)["goal_generation_id"] == resumed_generation["id"]


@pytest.mark.asyncio
async def test_current_terminal_clear_preserves_generation_resumed_after_snapshot(
) -> None:
    from hermes_cli.goals import GoalManager

    session_id = f"goal-gate-{uuid.uuid4().hex}"
    runner, _adapter = _runner(session_id)
    ingress = _connector_event(text="clear the old generation")
    lineage = runner._bind_capability_goal_lineage(ingress)
    assert lineage is not None
    manager = GoalManager(session_id, default_max_turns=0)
    original_state = manager.set("preserve a resume racing with clear", max_turns=0)
    entry = runner.session_store.get_or_create_session(ingress.source)
    entry.session_key = build_session_key(ingress.source)
    entry.origin = ingress.source
    original_record = await runner._persist_capability_goal_lineage_record(
        session_entry=entry,
        lineage=lineage,
    )
    manager.clear()

    original_conditional_clear = (
        runner._clear_capability_goal_lineage_record_if_terminal
    )
    resumed_generation: dict[str, str] = {}

    async def _resume_after_lineage_snapshot(**expected):
        assert expected["goal_generation_id"] == original_state.generation_id
        assert (
            expected["lineage_record_sha256"]
            == original_record["record_sha256"]
        )
        resumed = GoalManager(session_id, default_max_turns=0).resume()
        assert resumed is not None and resumed.status == "active"
        resumed_generation["id"] = resumed.generation_id
        await runner._persist_capability_goal_lineage_record(
            session_entry=entry,
            lineage=lineage,
        )
        return await original_conditional_clear(**expected)

    runner._clear_capability_goal_lineage_record_if_terminal = (
        _resume_after_lineage_snapshot
    )
    assert not await runner._clear_capability_goal_current_terminal_lineage(
        session_id
    )

    current = GoalManager(session_id).state
    assert current is not None and current.status == "active"
    assert current.generation_id == resumed_generation["id"]
    lineage_raw = runner._test_goal_meta[
        f"capability_goal_lineage:{session_id}"
    ]
    assert lineage_raw
    assert json.loads(lineage_raw)["goal_generation_id"] == resumed_generation["id"]


@pytest.mark.asyncio
async def test_real_queued_user_event_preempts_automatic_continuation() -> None:
    session_id = f"goal-gate-{uuid.uuid4().hex}"
    runner, adapter = _runner(session_id)
    ingress = _connector_event(text="first authenticated turn")
    queued_user = _connector_event(
        text="new owner direction",
        event_id="1527000000000000101",
    )
    goal_lineage = runner._bind_capability_goal_lineage(ingress)
    assert goal_lineage
    assert runner._bind_capability_goal_lineage(queued_user)
    session_key = build_session_key(ingress.source)
    adapter._pending_messages[session_key] = queued_user
    turn_id, generation_id = _begin_outcome(session_id, "continue")
    entry = runner.session_store.get_or_create_session(ingress.source)
    entry.session_key = session_key
    entry.origin = ingress.source
    await runner._persist_capability_goal_lineage_record(
        session_entry=entry,
        lineage=goal_lineage,
    )

    await runner._finalize_gateway_goal_turn(
        source=ingress.source,
        response={
            "final_response": "partial result",
            "session_id": session_id,
            "turn_id": turn_id,
            "goal_generation_id": generation_id,
        },
    )

    assert adapter._pending_messages[session_key] is queued_user
    assert not runner._is_sealed_capability_goal_continuation(queued_user)
    receipt = json.loads(
        runner._test_goal_meta[f"capability_goal_preemption:{session_id}"]
    )
    queued_proof = authenticated_discord_connector_ingress(queued_user)
    assert queued_proof is not None
    assert receipt["goal_generation_id"] == generation_id
    assert receipt["originating_turn_id"] == turn_id
    assert receipt["queued_event_id"] == queued_proof.event_id
    assert receipt["queued_event_sha256"] == queued_proof.event_sha256
    assert receipt["queued_delivery_id"] == queued_proof.delivery_id
    assert receipt["automatic_continuation_was_pending"] is True
    assert receipt["automatic_continuation_duplicate_count"] == 0
    assert receipt["queue_path"] == "adapter.pending"
    assert len(receipt["preemption_sha256"]) == 64
    preemption_key = f"capability_goal_preemption:{session_id}"
    original_raw = runner._test_goal_meta[preemption_key]
    assert (
        await runner._write_capability_goal_append_only_meta(
            preemption_key,
            original_raw,
        )
        is True
    )

    later = _connector_event(
        text="different queued owner direction",
        event_id="1527000000000000104",
    )
    assert runner._bind_capability_goal_lineage(later) is not None
    with pytest.raises(RuntimeError, match="append-only receipt conflicts"):
        await runner._persist_capability_goal_preemption_receipt(
            source=ingress.source,
            capability_goal_lineage=goal_lineage,
            queued_event=later,
            queue_path="adapter.pending",
            session_id=session_id,
            session_key=session_key,
            goal_generation_id=generation_id,
            originating_turn_id=turn_id,
            automatic_duplicate_count=0,
        )
    assert runner._test_goal_meta[preemption_key] == original_raw


@pytest.mark.parametrize(
    "case",
    ("forged-metadata", "synthetic-event", "cross-session", "cross-generation"),
)
@pytest.mark.asyncio
async def test_preemption_receipt_rejects_non_connector_or_cross_bound_event(
    case,
) -> None:
    from hermes_cli.goals import GoalManager

    session_id = f"goal-gate-{uuid.uuid4().hex}"
    runner, _adapter = _runner(session_id)
    ingress = _connector_event(text="active goal turn")
    goal_lineage = runner._bind_capability_goal_lineage(ingress)
    assert goal_lineage is not None
    manager = GoalManager(session_id, default_max_turns=0)
    state = manager.set("preserve exact preemption authority", max_turns=0)
    session_key = build_session_key(ingress.source)
    entry = runner.session_store.get_or_create_session(ingress.source)
    entry.session_key = session_key
    entry.origin = ingress.source
    await runner._persist_capability_goal_lineage_record(
        session_entry=entry,
        lineage=goal_lineage,
    )

    genuine = _connector_event(
        text="real queued owner direction",
        event_id="1527000000000000102",
    )
    assert runner._bind_capability_goal_lineage(genuine) is not None
    queued_event = genuine
    receipt_session_id = session_id
    receipt_generation = state.generation_id
    if case == "forged-metadata":
        queued_event = MessageEvent(
            text=genuine.text,
            message_type=MessageType.TEXT,
            source=dataclasses.replace(genuine.source),
            message_id=genuine.message_id,
            metadata=dict(genuine.metadata),
        )
        assert runner._bind_capability_goal_lineage(queued_event) is None
    elif case == "synthetic-event":
        queued_event = MessageEvent(
            text="synthetic queue item",
            message_type=MessageType.TEXT,
            source=ingress.source,
            message_id=ingress.message_id,
        )
        assert runner._bind_capability_goal_lineage(queued_event) is None
    elif case == "cross-session":
        receipt_session_id = f"goal-gate-{uuid.uuid4().hex}"
    elif case == "cross-generation":
        receipt_generation = "e" * 32

    with pytest.raises(RuntimeError):
        await runner._persist_capability_goal_preemption_receipt(
            source=ingress.source,
            capability_goal_lineage=goal_lineage,
            queued_event=queued_event,
            queue_path="adapter.pending",
            session_id=receipt_session_id,
            session_key=session_key,
            goal_generation_id=receipt_generation,
            originating_turn_id=f"turn-{uuid.uuid4().hex}",
            automatic_duplicate_count=0,
        )
    assert f"capability_goal_preemption:{session_id}" not in runner._test_goal_meta


@pytest.mark.asyncio
async def test_fresh_gateway_runner_recovers_durable_goal_and_finalizes(
    _isolated_goal_home,
    monkeypatch,
) -> None:
    """A serialized source loses relay trust; journal readback re-seals it."""

    from hermes_cli.goals import GoalManager

    home = _isolated_goal_home
    config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, token="redacted"),
            Platform.RELAY: PlatformConfig(enabled=True, token="redacted"),
        }
    )
    sessions_dir = home / "gateway-sessions"

    first_store = SessionStore(sessions_dir, config)
    first_runner, _first_adapter = _runner("unused-first-runner")
    first_runner.config = config
    first_runner.session_store = first_store
    first_runner._async_session_store = None
    first_runner._load_capability_goal_runtime_binding = _runtime_binding

    ingress = _connector_event()
    result = await first_runner._handle_message(ingress)
    first_entry = first_store.get_or_create_session(ingress.source)
    assert "Goal set" in result
    assert GoalManager(first_entry.session_id).is_active()

    db = first_store._db
    assert db is not None
    record_key = f"capability_goal_lineage:{first_entry.session_id}"
    record_raw = db.get_meta(record_key)
    assert record_raw
    record = json.loads(record_raw)
    assert record["session_id"] == first_entry.session_id
    pre_restart_record_sha256 = record["record_sha256"]
    assert first_store.mark_resume_pending(first_entry.session_key)

    # A genuinely fresh SessionStore reload reconstructs SessionSource from
    # durable JSON. The process-local relay trust bit and lineage seal are both
    # deliberately absent after that boundary.
    second_store = SessionStore(sessions_dir, config)
    second_entry = second_store.lookup_by_session_id(first_entry.session_id)
    assert second_entry is not None and second_entry.origin is not None
    assert second_entry.origin.delivered_via_upstream_relay is False
    assert not hasattr(second_entry.origin, "_hermes_capability_goal_lineage")

    transport = object.__new__(DiscordConnectorRelayTransport)

    async def _read_ack(kind, payload):
        assert kind is DiscordConnectorKind.EVENT_ACK_READBACK
        assert payload == {
            "delivery_id": record["first_delivery_id"],
            "event_id": record["first_event_id"],
            "event_sha256": record["first_event_sha256"],
        }
        return {
            "status": "ok",
            "replayed": False,
            "result": {
                "delivery_id": payload["delivery_id"],
                "event_id": payload["event_id"],
                "event_sha256": payload["event_sha256"],
                "state": "acked",
                "acked": True,
            },
            "receipt_sha256": "6" * 64,
        }

    transport._request = _read_ack
    descriptor = CapabilityDescriptor(
        contract_version=1,
        platform="discord",
        label="Discord",
        max_message_length=2_000,
        supports_draft_streaming=False,
        supports_edit=False,
        supports_threads=True,
        markdown_dialect="discord",
        len_unit="chars",
    )
    relay = RelayAdapter(PlatformConfig(), descriptor, transport=transport)

    second_runner, _unused_adapter = _runner("unused-second-runner")
    second_runner.config = config
    second_runner.session_store = second_store
    second_runner._async_session_store = None
    second_runner.adapters = {Platform.RELAY: relay}
    second_runner._load_capability_goal_runtime_binding = _runtime_binding
    second_runner._isolated_runtime = False
    second_runner._background_tasks = set()
    second_runner._startup_restore_tasks = []
    second_runner._persist_active_agents = lambda: None
    second_runner._restart_loop_guard_config = lambda: (100, 60)
    second_runner._is_user_authorized = (
        lambda source: source.delivered_via_upstream_relay is True
        and source.user_id == OWNER_ID
    )
    monkeypatch.setenv("INVOCATION_ID", "7" * 32)
    monkeypatch.setattr(
        "gateway.restart_loop_guard.check_and_record",
        lambda *_args, **_kwargs: False,
    )

    finalized = []

    async def _defer_notice(_source, _message):
        return None

    second_runner._defer_goal_status_notice_after_delivery = _defer_notice

    async def _handle_restored(event):
        assert event.source is second_entry.origin
        assert event.source.delivered_via_upstream_relay is True
        assert second_runner._capability_goal_lineage_for_event(event) is not None
        manager = GoalManager(first_entry.session_id, default_max_turns=0)
        turn_id = f"restart-turn-{uuid.uuid4().hex}"
        generation_id = manager.begin_model_turn(turn_id)
        assert manager.record_model_outcome(
            "continue",
            "restart continuation remains active",
            originating_turn_id=turn_id,
            goal_generation_id=generation_id,
        )
        await second_runner._finalize_gateway_goal_turn(
            source=event.source,
            response={
                "final_response": "first restart step verified",
                "session_id": first_entry.session_id,
                "turn_id": turn_id,
                "goal_generation_id": generation_id,
            },
        )
        finalized.append(event)

    relay.handle_message = _handle_restored

    assert second_runner._schedule_resume_pending_sessions() == 1
    await asyncio.gather(*list(second_runner._background_tasks))

    assert len(finalized) == 1
    state = GoalManager(first_entry.session_id).state
    assert state is not None and state.status == "active"
    recovery_key = f"capability_goal_lineage_recovery:{first_entry.session_id}"
    recovery_raw = db.get_meta(recovery_key)
    assert recovery_raw
    recovery = json.loads(recovery_raw)
    assert recovery["lineage_record_sha256"] == pre_restart_record_sha256
    assert recovery["connector_ack_readback_receipt_sha256"] == "6" * 64
    assert recovery["connector_journal_state"] == "acked"
    assert recovery["gateway_invocation_id"] == "7" * 32
    assert recovery["gateway_main_pid"] > 1
    assert len(recovery["gateway_process_identity_sha256"]) == 64
    assert (
        await second_runner._write_capability_goal_append_only_meta(
            recovery_key,
            recovery_raw,
        )
        is True
    )

    # The canary contract permits exactly one controlled process recovery.
    # A second fresh process can independently re-prove the connector ACK,
    # but it cannot replace the first-wins process-bound recovery receipt.
    third_store = SessionStore(sessions_dir, config)
    third_entry = third_store.lookup_by_session_id(first_entry.session_id)
    assert third_entry is not None and third_entry.origin is not None
    assert third_entry.origin.delivered_via_upstream_relay is False
    third_runner, _third_adapter = _runner("unused-third-runner")
    third_runner.config = config
    third_runner.session_store = third_store
    third_runner._async_session_store = None
    third_runner._load_capability_goal_runtime_binding = _runtime_binding
    third_event = MessageEvent(
        text="",
        message_type=MessageType.TEXT,
        source=third_entry.origin,
        internal=True,
    )
    monkeypatch.setenv("INVOCATION_ID", "8" * 32)
    assert (
        await third_runner._recover_capability_goal_lineage_for_startup(
            adapter=relay,
            event=third_event,
            session_key=third_entry.session_key,
        )
        is None
    )
    assert third_entry.origin.delivered_via_upstream_relay is False
    assert not hasattr(
        third_entry.origin,
        "_hermes_capability_goal_lineage",
    )
    assert not hasattr(third_event, "_hermes_capability_goal_lineage")
    third_db = third_store._db
    assert third_db is not None
    assert (
        third_db.get_meta(recovery_key) == recovery_raw
    )

    # The one accepted restart may continue through its exact sealed source
    # and reach native terminal finalization; the rejected process has no
    # authority to do so.
    restored_event = finalized[0]
    assert (
        second_runner._capability_goal_lineage_for_event(restored_event)
        is not None
    )
    relay._pending_messages.clear()
    manager = GoalManager(first_entry.session_id, default_max_turns=0)
    complete_turn_id = f"restart-complete-{uuid.uuid4().hex}"
    complete_generation_id = manager.begin_model_turn(complete_turn_id)
    assert manager.record_model_outcome(
        "complete",
        "accepted restart finalization verified",
        originating_turn_id=complete_turn_id,
        goal_generation_id=complete_generation_id,
    )
    monkeypatch.setenv("INVOCATION_ID", "7" * 32)
    await second_runner._finalize_gateway_goal_turn(
        source=restored_event.source,
        response={
            "final_response": "verified after the accepted restart",
            "session_id": first_entry.session_id,
            "turn_id": complete_turn_id,
            "goal_generation_id": complete_generation_id,
        },
    )
    terminal_state = GoalManager(first_entry.session_id).state
    assert terminal_state is not None and terminal_state.status == "done"
    # Terminal completion retires only the active lineage hint. The immutable
    # recovery receipt remains available to the canary evidence collector.
    assert db.get_meta(record_key) == ""
    assert (
        db.get_meta(recovery_key) == recovery_raw
    )


@pytest.mark.parametrize(
    "case",
    (
        "cross-session-key",
        "cross-goal-generation",
        "cross-run",
        "cross-fixture",
        "cross-release",
        "stale-window",
        "wrong-owner",
        "wrong-channel",
        "dm-source",
    ),
)
@pytest.mark.asyncio
async def test_restart_recovery_rejects_stale_or_cross_bound_authority(
    case,
    monkeypatch,
) -> None:
    from gateway.run import _capability_goal_sha256
    from hermes_cli.goals import GoalManager

    session_id = f"goal-gate-{uuid.uuid4().hex}"
    first_runner, _adapter = _runner(session_id)
    ingress = _connector_event(text="establish durable authority")
    lineage = first_runner._bind_capability_goal_lineage(ingress)
    assert lineage is not None
    manager = GoalManager(session_id, default_max_turns=0)
    manager.set("finish after a verified restart", max_turns=0)
    session_key = build_session_key(ingress.source)
    entry = first_runner.session_store.get_or_create_session(ingress.source)
    entry.session_key = session_key
    entry.origin = ingress.source
    await first_runner._persist_capability_goal_lineage_record(
        session_entry=entry,
        lineage=lineage,
    )
    key = f"capability_goal_lineage:{session_id}"
    record = json.loads(first_runner._test_goal_meta[key])
    runtime = _runtime_binding()

    if case == "cross-session-key":
        record["session_key_sha256"] = "8" * 64
    elif case == "cross-goal-generation":
        record["goal_generation_id"] = "9" * 32
    elif case == "cross-run":
        runtime["run_id"] = "different-reviewed-run"
    elif case == "cross-fixture":
        runtime["fixture_sha256"] = "a" * 64
    elif case == "cross-release":
        runtime["release_sha"] = "b" * 40
    elif case == "stale-window":
        now = int(time.time() * 1_000)
        runtime["valid_from_unix_ms"] = now - 60_000
        runtime["valid_until_unix_ms"] = now - 1
    elif case == "wrong-owner":
        record["owner_user_id"] = "1279454038731264062"
    elif case == "wrong-channel":
        record["channel_id"] = "1526858760100909067"

    unsigned = {
        name: value for name, value in record.items() if name != "record_sha256"
    }
    record["record_sha256"] = _capability_goal_sha256(unsigned)
    first_runner._test_goal_meta[key] = json.dumps(
        record,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    restored_source = dataclasses.replace(
        ingress.source,
        delivered_via_upstream_relay=False,
        chat_type="dm" if case == "dm-source" else ingress.source.chat_type,
    )
    restored_entry = SessionEntry(
        session_key=session_key,
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=restored_source,
        platform=Platform.DISCORD,
        chat_type=restored_source.chat_type,
        resume_pending=True,
        resume_reason="restart_timeout",
    )
    second_runner, _second_adapter = _runner(session_id)
    second_runner._load_capability_goal_runtime_binding = lambda: runtime
    second_runner.session_store.peek_session_id.return_value = session_id
    second_runner.session_store.lookup_by_session_id.return_value = restored_entry
    second_runner.session_store._generate_session_key.return_value = session_key
    second_runner.session_store._db.get_meta.side_effect = (
        first_runner._test_goal_meta.get
    )
    second_runner.session_store._db.get_session.return_value = {
        "id": session_id,
        "source": Platform.DISCORD.value,
        "user_id": OWNER_ID,
        "session_key": session_key,
        "chat_id": ROOT_CHANNEL_ID,
        "chat_type": "channel",
        "thread_id": None,
        "ended_at": None,
    }

    transport = object.__new__(DiscordConnectorRelayTransport)
    connector_calls = []

    async def _unexpected_request(kind, payload):
        connector_calls.append((kind, payload))
        raise AssertionError("invalid durable binding reached connector authority")

    transport._request = _unexpected_request
    relay = RelayAdapter(
        PlatformConfig(),
        CapabilityDescriptor(
            contract_version=1,
            platform="discord",
            label="Discord",
            max_message_length=2_000,
            supports_draft_streaming=False,
            supports_edit=False,
            supports_threads=True,
            markdown_dialect="discord",
            len_unit="chars",
        ),
        transport=transport,
    )
    event = MessageEvent(
        text="",
        message_type=MessageType.TEXT,
        source=restored_source,
        internal=True,
        metadata={
            # Caller-shaped metadata cannot repair any failed durable binding.
            "discord_connector_event_sha256": record["first_event_sha256"],
        },
    )
    monkeypatch.setenv("INVOCATION_ID", "c" * 32)

    assert (
        await second_runner._recover_capability_goal_lineage_for_startup(
            adapter=relay,
            event=event,
            session_key=session_key,
        )
        is None
    )
    assert restored_source.delivered_via_upstream_relay is False
    assert second_runner._capability_goal_lineage_for_event(event) is None
    assert connector_calls == []
