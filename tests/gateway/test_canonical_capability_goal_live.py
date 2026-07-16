from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest

from gateway import canonical_capability_goal_live as live
from gateway import canonical_capability_canary_e2e as evidence_contract
from gateway.canonical_projection_export import PROJECTION_EXPORT_SCHEMA
from gateway.discord_connector_protocol import (
    DiscordConnectorEvent,
    DiscordConnectorTarget,
)
from plugins import muncho_canary_evidence as plugin


NOW = 1_800_000_000_000
RUN_ID = "11111111-1111-4111-8111-111111111111"
REVISION = "a" * 40
RELEASE_SHA256 = hashlib.sha256(b"release").hexdigest()
FIXTURE_SHA256 = hashlib.sha256(b"fixture").hexdigest()
TARGET = {
    "target_type": "public_guild_channel",
    "guild_id": "123456789012345678",
    "channel_id": "223456789012345678",
}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _process_identity() -> dict[str, object]:
    return {
        "collector_pid": os.getpid(),
        "collector_uid": os.getuid(),
        "collector_gid": os.getgid(),
        "module_origin_sha256": _digest("collector-origin"),
        "module_sha256": _digest("collector-module"),
        "boot_id_sha256": _digest("boot"),
        "process_start_time_ticks": 42,
    }


def _service(pid: int = 4242, invocation: str = "1" * 32):
    unsigned = {
        "service_unit": live.GATEWAY_UNIT_NAME,
        "main_pid": pid,
        "invocation_id": invocation,
        "active_enter_timestamp_monotonic": 100,
        "n_restarts": 0,
    }
    return live.GatewayServiceIdentity(
        **unsigned,
        identity_sha256=live._sha256_json(unsigned),
    )


def _collector(**overrides):
    values = {
        "revision": REVISION,
        "release_sha256": RELEASE_SHA256,
        "run_id": RUN_ID,
        "fixture_sha256": FIXTURE_SHA256,
        "valid_from_unix_ms": NOW - 1_000,
        "valid_until_unix_ms": NOW + 60_000,
        "public_target": TARGET,
        "owner_user_id": "1279454038731264061",
        "api_observer_config_sha256": _digest("api-observer-config"),
        "gateway_uid": 1001,
        "gateway_gid": 1002,
        "collector_process_identity_reader": _process_identity,
        "plugin_module_validator": lambda *_args: None,
        "now_ms": lambda: NOW,
    }
    values.update(overrides)
    return live.SegmentedGoalEvidenceCollector(**values)


def _projection_binding_fixture():
    event_id = "31111111-1111-4111-8111-111111111111"
    case_id = "case:goal-live-projection"
    runtime = {
        "request_id": "goal-live-projection",
        "platform": "discord",
        "session_key_sha256": _digest("session-key"),
    }
    idempotency_key = "goal-live:plan:active"
    content_sha256 = _digest("canonical-content")
    occurred_at = "2027-01-15T08:00:00+00:00"
    event = {
        "event_id": event_id,
        "schema_version": "canonical_event.v1",
        "event_type": "task.plan.updated",
        "occurred_at": occurred_at,
        "case_id": case_id,
        "source": {"observed_session": runtime},
        "actor": {"type": "model", "id": "gpt-5.6-sol"},
        "subject": {"type": "case", "id": case_id},
        "evidence": [],
        "decision": {"decided_by": "model:gpt-5.6-sol"},
        "status": {"state": "active"},
        "next_action": {},
        "safety": {"secret_value_recorded": False},
        "payload": {
            "idempotency_key": idempotency_key,
            "canonical_content_sha256": content_sha256,
            "plan": {
                "plan_id": "plan:goal-live-projection",
                "revision": 2,
                "state": "active",
                "resume_cursor": {"next_step_id": "step:verify"},
            },
        },
    }
    proof = {
        "event_id": event_id,
        "canonical_content_sha256": content_sha256,
        "origin": "model:gpt-5.6-sol",
        "trusted_runtime": runtime,
        "appended_at": occurred_at,
    }
    service = _service()
    peer = live.GoalPeer(service.main_pid, 1001, 1002)

    def collected(event_name, payload, ordinal):
        return live.GoalCollectedFrame(
            value={"event": event_name, "payload": payload},
            frame_sha256=_digest(f"projection-frame-{ordinal}"),
            segment_chain_head_sha256=_digest(f"projection-chain-{ordinal}"),
            previous_segment_terminal_sha256="0" * 64,
            peer=peer,
            service_identity=service,
        )

    frames = [
        collected(
            "goal_canonical_event",
            {
                "event_id": event_id,
                "case_id": case_id,
                "event_type": "task.plan.updated",
                "canonical_content_sha256": content_sha256,
                "idempotency_key_sha256": hashlib.sha256(
                    idempotency_key.encode()
                ).hexdigest(),
            },
            1,
        ),
        collected(
            "goal_canonical_readback",
            {
                "case_id": case_id,
                "plan_identities": [
                    {
                        "event_id": event_id,
                        "plan_id": "plan:goal-live-projection",
                        "revision": 2,
                        "state": "active",
                        "next_step_id": "step:verify",
                    }
                ],
            },
            2,
        ),
    ]
    export = {
        "schema": PROJECTION_EXPORT_SCHEMA,
        "events": [event],
        "provenance": [proof],
    }
    return frames, export


def _goal_assembler_fixture():
    semantic = evidence_contract.build_semantic_config_contract()
    fixture_target = {
        "target_type": "public_channel",
        "guild_id": TARGET["guild_id"],
        "channel_id": TARGET["channel_id"],
    }
    bot_identities = {
        "production_bot_user_id": "323456789012345678",
        "connector_bot_user_id": "423456789012345678",
        "routeback_bot_user_id": "523456789012345678",
    }
    topology = evidence_contract.build_capability_role_topology_contract(
        public_discord_target=fixture_target,
        discord_bot_identities=bot_identities,
    )
    fixture = {
        "run_id": RUN_ID,
        "release_sha": REVISION,
        "capability_plan_sha256": _digest("capability-plan"),
        "full_canary_plan_sha256": _digest("full-plan"),
        "owner_id": "1279454038731264061",
        "valid_from_unix_ms": NOW - 1_000,
        "valid_until_unix_ms": NOW + 1_000,
        "model_route": {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "model": "gpt-5.6-sol",
        },
        "semantic_config_contract": semantic,
        "semantic_config_contract_sha256": live._sha256_json(semantic),
        "required_toolsets": list(evidence_contract.REQUIRED_TOOLSETS),
        "required_toolsets_sha256": live._sha256_json(
            {"toolsets": list(evidence_contract.REQUIRED_TOOLSETS)}
        ),
        "public_discord_target": fixture_target,
        "discord_bot_identities": bot_identities,
        "capability_role_topology_contract": topology,
        "capability_role_topology_contract_sha256": live._sha256_json(
            topology
        ),
    }
    fixture_sha256 = _digest("assembler-fixture")
    approval_sha256 = _digest("owner-approval")
    kickoff = (
        "Complete the approved durable Canonical Task Workspace canary and "
        "resume the exact plan after the controlled restart."
    )
    goal_command = f"/goal {kickoff}"
    preemption_message = (
        "Owner direction: preserve the same plan and verify the next safe step."
    )
    challenge_unsigned = {
        "schema": live.GOAL_CHALLENGE_SCHEMA,
        "run_id": RUN_ID,
        "fixture_sha256": fixture_sha256,
        "challenge_id": "41111111-1111-4111-8111-111111111111",
        "public_target": copy.deepcopy(TARGET),
        "owner_user_id": fixture["owner_id"],
        "goal_command_sha256": hashlib.sha256(goal_command.encode()).hexdigest(),
        "kickoff_message_sha256": hashlib.sha256(kickoff.encode()).hexdigest(),
        "preemption_message_sha256": hashlib.sha256(
            preemption_message.encode()
        ).hexdigest(),
        "message_content_recorded": False,
        "published_at_unix_ms": NOW,
    }
    challenge = live.GoalOwnerChallenge(
        transient_input={
            "goal_command": goal_command,
            "kickoff_message": kickoff,
            "preemption_message": preemption_message,
        },
        receipt={
            **challenge_unsigned,
            "challenge_sha256": live._sha256_json(challenge_unsigned),
        },
        path=Path("/run/goal-owner-challenge.json"),
    )

    target = DiscordConnectorTarget.from_mapping(TARGET)

    def connector_row(
        *, event_id: str, content: str, delivery: str, offered: int, acked: int
    ):
        event = DiscordConnectorEvent(
            event_id=event_id,
            target=target,
            author_id=fixture["owner_id"],
            author_name="Emil",
            author_is_bot=False,
            content=content,
            created_at_unix_ms=offered - 1,
        )
        return live.ConnectorJournalRow(
            event=event,
            event_sha256=event.sha256,
            state="acked",
            delivery_id=delivery,
            lease_until_unix_ms=None,
            offered_at_unix_ms=offered,
            acked_at_unix_ms=acked,
            row_sha256=_digest(f"connector-row:{event_id}"),
        )

    goal_row = connector_row(
        event_id="1527000000000000001",
        content=goal_command,
        delivery="delivery-goal",
        offered=NOW + 1,
        acked=NOW + 2,
    )
    preemption_row = connector_row(
        event_id="1527000000000000002",
        content=preemption_message,
        delivery="delivery-preemption",
        offered=NOW + 22,
        acked=NOW + 23,
    )

    pre_service = _service(pid=21001, invocation="1" * 32)
    post_unsigned = {
        "service_unit": live.GATEWAY_UNIT_NAME,
        "main_pid": 21002,
        "invocation_id": "2" * 32,
        "active_enter_timestamp_monotonic": 200,
        "n_restarts": 1,
    }
    post_service = live.GatewayServiceIdentity(
        **post_unsigned,
        identity_sha256=live._sha256_json(post_unsigned),
    )
    restart = live.GatewayRestartObservation(
        pre_service_identity=pre_service,
        post_service_identity=post_service,
        restart_requested_at_unix_ms=NOW + 50,
        restart_completed_at_unix_ms=NOW + 60,
    )
    prompt_sha256 = _digest("stable-prompt")
    tools_sha256 = _digest("stable-tools")
    model_sha256 = hashlib.sha256(b"gpt-5.6-sol").hexdigest()
    turns = (
        ("goal-turn-1", "continue", pre_service, "a" * 32, 10, 20, 21),
        ("goal-turn-2", "continue", pre_service, "a" * 32, 30, 40, 41),
        ("goal-turn-3", "complete", post_service, "b" * 32, 70, 80, 81),
    )
    frames = []

    def collected(
        *,
        event: str,
        payload: dict,
        service,
        segment: str,
        turn_id: str | None,
        observed: int,
        label: str,
    ):
        return live.GoalCollectedFrame(
            value={
                "event": event,
                "segment_id": segment,
                "turn_id": turn_id,
                "session_id": f"capability_{RUN_ID}",
                "observed_at_unix_ms": observed,
                "payload": payload,
            },
            frame_sha256=_digest(f"frame:{label}"),
            segment_chain_head_sha256=_digest(f"chain:{label}"),
            previous_segment_terminal_sha256="0" * 64,
            peer=live.GoalPeer(service.main_pid, 1001, 1002),
            service_identity=service,
        )

    for ordinal, (
        turn_id,
        outcome,
        service,
        segment,
        started,
        completed,
        outcome_observed,
    ) in enumerate(turns, start=1):
        api_sha256 = _digest(f"api:{ordinal}")
        tool_sha256 = _digest(f"todo:{ordinal}")
        frames.extend(
            (
                collected(
                    event="goal_pre_api_request",
                    payload={
                        "api_request_id_sha256": api_sha256,
                        "system_prompt_sha256": prompt_sha256,
                        "tool_schema_sha256": tools_sha256,
                        "started_at_unix_ms": NOW + started,
                    },
                    service=service,
                    segment=segment,
                    turn_id=turn_id,
                    observed=NOW + started,
                    label=f"pre:{ordinal}",
                ),
                collected(
                    event="goal_post_api_request",
                    payload={
                        "api_request_id_sha256": api_sha256,
                        "response_model_sha256": model_sha256,
                        "response_observed_at_unix_ms": NOW + completed,
                    },
                    service=service,
                    segment=segment,
                    turn_id=turn_id,
                    observed=NOW + completed,
                    label=f"post:{ordinal}",
                ),
                collected(
                    event="goal_model_outcome",
                    payload={
                        "api_request_id_sha256": api_sha256,
                        "tool_call_id_sha256": tool_sha256,
                        "outcome": outcome,
                        "reason_sha256": _digest(f"reason:{ordinal}"),
                    },
                    service=service,
                    segment=segment,
                    turn_id=turn_id,
                    observed=NOW + outcome_observed,
                    label=f"outcome:{ordinal}",
                ),
                collected(
                    event="goal_turn_end",
                    payload={"completed": True, "interrupted": False},
                    service=service,
                    segment=segment,
                    turn_id=turn_id,
                    observed=NOW + outcome_observed,
                    label=f"end:{ordinal}",
                ),
            )
        )

    case_id = "case:goal-assembler"
    plan_id = "plan:goal-assembler"

    def plan_event(
        *, event_id: str, revision: int, state: str, next_step: str | None
    ):
        idempotency = f"goal-assembler:plan:{revision}"
        content_sha256 = _digest(f"plan-content:{revision}")
        occurred_at = f"2027-01-15T08:00:0{revision}+00:00"
        runtime = {
            "request_id": f"goal-plan-{revision}",
            "platform": "discord",
        }
        event = {
            "event_id": event_id,
            "schema_version": "canonical_event.v1",
            "event_type": "task.plan.updated",
            "occurred_at": occurred_at,
            "case_id": case_id,
            "source": {"observed_session": runtime},
            "actor": {"type": "model", "id": "gpt-5.6-sol"},
            "subject": {"type": "case", "id": case_id},
            "evidence": [],
            "decision": {"decided_by": "model:gpt-5.6-sol"},
            "status": {"state": state},
            "next_action": {},
            "safety": {"secret_value_recorded": False},
            "payload": {
                "idempotency_key": idempotency,
                "canonical_content_sha256": content_sha256,
                "plan": {
                    "plan_id": plan_id,
                    "revision": revision,
                    "state": state,
                    "resume_cursor": {"next_step_id": next_step},
                },
            },
        }
        proof = {
            "event_id": event_id,
            "canonical_content_sha256": content_sha256,
            "origin": "model:gpt-5.6-sol",
            "trusted_runtime": runtime,
            "appended_at": occurred_at,
        }
        return event, proof, idempotency

    active_event, active_proof, active_idempotency = plan_event(
        event_id="61111111-1111-4111-8111-111111111111",
        revision=1,
        state="active",
        next_step="step:after-restart",
    )
    terminal_event, terminal_proof, terminal_idempotency = plan_event(
        event_id="71111111-1111-4111-8111-111111111111",
        revision=2,
        state="completed",
        next_step=None,
    )
    route_receipt = {
        "platform": "discord",
        "adapter_receipt": True,
        "receipt_readback_verified": True,
        "message_id": "1527000000000000099",
        "channel_id": TARGET["channel_id"],
        "content_sha256": _digest("terminal-route-content"),
        "public_receipt_sha256": _digest("terminal-route-public-receipt"),
    }
    route_idempotency = "goal-assembler:routeback:terminal"
    route_content_sha256 = _digest("terminal-route-canonical-content")
    route_runtime = {"request_id": "goal-routeback", "platform": "discord"}
    route_occurred_at = "2027-01-15T08:00:04+00:00"
    route_event = {
        "event_id": "81111111-1111-4111-8111-111111111111",
        "schema_version": "canonical_event.v1",
        "event_type": "route_back.sent",
        "occurred_at": route_occurred_at,
        "case_id": case_id,
        "source": {"observed_session": route_runtime},
        "actor": {"type": "service", "id": "canonical_writer"},
        "subject": {"type": "route_back", "id": TARGET["channel_id"]},
        "evidence": [],
        "decision": {"decided_by": "routeback_finalize_sent"},
        "status": {"state": "route_back.sent"},
        "next_action": {},
        "safety": {"secret_value_recorded": False, "outbound": True},
        "payload": {
            "idempotency_key": route_idempotency,
            "canonical_content_sha256": route_content_sha256,
            "receipt": route_receipt,
            "route_back": {
                "target_ref": {
                    "target_type": "public_guild_channel",
                    "guild_id": TARGET["guild_id"],
                    "channel_id": TARGET["channel_id"],
                },
                "receipt": route_receipt,
                "execution_binding": {
                    "target_channel_id": TARGET["channel_id"],
                    "content_sha256": route_receipt["content_sha256"],
                },
            },
        },
    }
    route_proof = {
        "event_id": route_event["event_id"],
        "canonical_content_sha256": route_content_sha256,
        "origin": "routeback_finalize_sent",
        "trusted_runtime": route_runtime,
        "appended_at": route_occurred_at,
    }

    def append_plan_frames(event, idempotency, service, segment, observed, label):
        plan = event["payload"]["plan"]
        frames.extend(
            (
                collected(
                    event="goal_canonical_event",
                    payload={
                        "event_id": event["event_id"],
                        "case_id": case_id,
                        "event_type": "task.plan.updated",
                        "canonical_content_sha256": event["payload"][
                            "canonical_content_sha256"
                        ],
                        "idempotency_key_sha256": hashlib.sha256(
                            idempotency.encode()
                        ).hexdigest(),
                    },
                    service=service,
                    segment=segment,
                    turn_id=(
                        "goal-turn-2" if service == pre_service else "goal-turn-3"
                    ),
                    observed=NOW + observed,
                    label=f"canonical:{label}",
                ),
                collected(
                    event="goal_canonical_readback",
                    payload={
                        "case_id": case_id,
                        "readback_sha256": _digest(f"readback:{label}"),
                        "plan_identities": [
                            {
                                "event_id": event["event_id"],
                                "plan_id": plan_id,
                                "revision": plan["revision"],
                                "state": plan["state"],
                                "next_step_id": plan["resume_cursor"][
                                    "next_step_id"
                                ],
                            }
                        ],
                    },
                    service=service,
                    segment=segment,
                    turn_id=(
                        "goal-turn-2" if service == pre_service else "goal-turn-3"
                    ),
                    observed=NOW + observed + 1,
                    label=f"readback:{label}",
                ),
            )
        )

    append_plan_frames(
        active_event,
        active_idempotency,
        pre_service,
        "a" * 32,
        45,
        "active",
    )
    append_plan_frames(
        terminal_event,
        terminal_idempotency,
        post_service,
        "b" * 32,
        82,
        "terminal",
    )
    frames.append(
        collected(
            event="goal_canonical_event",
            payload={
                "event_id": route_event["event_id"],
                "case_id": case_id,
                "event_type": "route_back.sent",
                "canonical_content_sha256": route_content_sha256,
                "idempotency_key_sha256": hashlib.sha256(
                    route_idempotency.encode()
                ).hexdigest(),
            },
            service=post_service,
            segment="b" * 32,
            turn_id="goal-turn-3",
            observed=NOW + 84,
            label="canonical:routeback",
        )
    )
    projection = {
        "schema": PROJECTION_EXPORT_SCHEMA,
        "events": [active_event, terminal_event, route_event],
        "provenance": [active_proof, terminal_proof, route_proof],
    }
    projection_binding = live.validate_goal_canonical_projection_binding(
        frames,
        projection,
    )

    generation_id = "9" * 32
    session_id = f"capability_{RUN_ID}"
    binding = {
        "run_id": RUN_ID,
        "fixture_sha256": fixture_sha256,
        "release_sha": REVISION,
        "capability_plan_sha256": fixture["capability_plan_sha256"],
        "full_canary_plan_sha256": fixture["full_canary_plan_sha256"],
    }

    def native(schema: str, digest_field: str, **fields):
        unsigned = {"schema": schema, **binding, **fields}
        return {**unsigned, digest_field: live._sha256_json(unsigned)}

    states = [
        {
            "session_id": session_id,
            "generation_id": generation_id,
            "max_turns": 0,
            "status": status,
            "turns_used": ordinal,
        }
        for ordinal, status in enumerate(("active", "active", "active", "done"))
    ]
    finalizations = {}
    for ordinal, (turn_id, outcome, service, _segment, *_times) in enumerate(
        turns, start=1
    ):
        process_sha256 = live._service_process_sha256(service)
        intent = native(
            live.GOAL_FINALIZATION_INTENT_SCHEMA,
            "intent_sha256",
            session_id=session_id,
            goal_generation_id=generation_id,
            originating_turn_id=turn_id,
            pending_outcome_exact=True,
            model_outcome=outcome,
            model_reason_sha256=_digest(f"reason:{ordinal}"),
            state_before=states[ordinal - 1],
            state_before_sha256=live._sha256_json(states[ordinal - 1]),
            gateway_process_identity_sha256=process_sha256,
        )
        finalization = native(
            live.GOAL_FINALIZATION_SCHEMA,
            "finalization_sha256",
            session_id=session_id,
            goal_generation_id=generation_id,
            originating_turn_id=turn_id,
            intent_sha256=intent["intent_sha256"],
            model_outcome=outcome,
            model_reason_sha256=_digest(f"reason:{ordinal}"),
            decision_verdict="continue" if outcome == "continue" else "done",
            should_continue=outcome == "continue",
            state_before=states[ordinal - 1],
            state_before_sha256=live._sha256_json(states[ordinal - 1]),
            state_after=states[ordinal],
            state_after_sha256=live._sha256_json(states[ordinal]),
            gateway_service_unit=service.service_unit,
            gateway_invocation_id=service.invocation_id,
            gateway_main_pid=service.main_pid,
            gateway_process_identity_sha256=process_sha256,
            observed_after_unix_ms=NOW + turns[ordinal - 1][-1],
        )
        finalizations[turn_id] = (intent, finalization)
    recovery = native(
        live.GOAL_RECOVERY_SCHEMA,
        "recovery_sha256",
        session_id=session_id,
        goal_generation_id=generation_id,
        connector_event_id=goal_row.event.event_id,
        connector_event_sha256=goal_row.event.sha256,
        connector_delivery_id=goal_row.delivery_id,
        connector_journal_state="acked",
        gateway_service_unit=post_service.service_unit,
        gateway_invocation_id=post_service.invocation_id,
        gateway_main_pid=post_service.main_pid,
        restored_at_unix_ms=NOW + 65,
    )
    preemption = native(
        live.GOAL_PREEMPTION_SCHEMA,
        "preemption_sha256",
        session_id=session_id,
        goal_generation_id=generation_id,
        originating_turn_id="goal-turn-1",
        queued_event_id=preemption_row.event.event_id,
        queued_event_sha256=preemption_row.event.sha256,
        queued_delivery_id=preemption_row.delivery_id,
        queued_owner_user_id=fixture["owner_id"],
        queued_guild_id=TARGET["guild_id"],
        queued_channel_id=TARGET["channel_id"],
        automatic_continuation_was_pending=True,
        automatic_continuation_duplicate_count=0,
        queue_path="adapter.pending",
        preempted_at_unix_ms=NOW + 24,
    )
    native_receipts = live.GoalNativeReceiptBundle(
        recovery=recovery,
        preemption=preemption,
        finalizations=finalizations,
    )
    production_diff = {"diff_sha256": _digest("production-diff")}
    return {
        "fixture": fixture,
        "fixture_sha256": fixture_sha256,
        "approval_sha256": approval_sha256,
        "challenge": challenge,
        "connector_rows": (goal_row, preemption_row),
        "frames": tuple(frames),
        "native_receipts": native_receipts,
        "restart": restart,
        "projection_binding": projection_binding,
        "production_diff": production_diff,
    }


def _frame(collector, *, event, sequence, payload, session=None, turn=None, segment="2" * 32):
    return {
        "schema": plugin.GOAL_FRAME_SCHEMA,
        "segment_id": segment,
        "sequence": sequence,
        "event": event,
        "release_sha": REVISION,
        "release_sha256": RELEASE_SHA256,
        "run_id": RUN_ID,
        "fixture_sha256": FIXTURE_SHA256,
        "collector_service_identity_sha256": (
            collector._collector_service_identity_sha256
        ),
        "session_id": session,
        "turn_id": turn,
        "observed_at_unix_ms": NOW,
        "payload": payload,
    }


def _ready_payload(collector, pid=4242):
    return {
        "plugin_name": "muncho_canary_evidence.goal_continuation",
        "gateway_pid": pid,
        "config_sha256": _digest("config"),
        "fixture_sha256": FIXTURE_SHA256,
        "collector_service_identity_sha256": (
            collector._collector_service_identity_sha256
        ),
        "collector_socket_identity_sha256": _digest("socket"),
        "module_origin": f"/opt/muncho-canary-releases/{REVISION}/plugins/muncho_canary_evidence/__init__.py",
        "module_sha256": _digest("plugin"),
    }


def _admit(collector, frame, service):
    peer = live.GoalPeer(service.main_pid, collector.gateway_uid, collector.gateway_gid)
    collector._validate_frame(frame, peer, service)
    segment = frame["segment_id"]
    if frame["sequence"] == 1:
        collector._segment_order.append(segment)
        collector._segment_peers[segment] = peer
        collector._segment_services[segment] = service
        collector._segment_heads[segment] = "0" * 64
        collector._segment_requests[segment] = {}
        collector._segment_tool_ids[segment] = set()
        collector._segment_pre_count[segment] = 0
    collector._frames.append(
        live.GoalCollectedFrame(
            value=copy.deepcopy(frame),
            frame_sha256=_digest(f"frame-{len(collector._frames)}"),
            segment_chain_head_sha256=_digest(f"chain-{len(collector._frames)}"),
            previous_segment_terminal_sha256="0" * 64,
            peer=peer,
            service_identity=service,
        )
    )


def test_observer_config_roundtrips_real_fixture_normalization(tmp_path: Path):
    fixture_target = {
        "target_type": "public_channel",
        "guild_id": TARGET["guild_id"],
        "channel_id": TARGET["channel_id"],
    }
    observer_target = live.normalize_goal_observer_public_target(fixture_target)
    collector = _collector(public_target=observer_target)
    config_path = tmp_path / "goal-observer.json"
    config_path.write_bytes(live._canonical_bytes(collector.observer_config()))
    config_path.chmod(0o440)

    observed = plugin.load_goal_config(
        config_path,
        expected_owner_uid=os.getuid(),
        expected_owner_gid=config_path.lstat().st_gid,
    )

    assert observed.run_id == RUN_ID
    assert observed.public_target == observer_target
    assert observed.public_target["guild_id"] == fixture_target["guild_id"]
    assert observed.public_target["channel_id"] == fixture_target["channel_id"]
    assert observed.collector.socket_path == live.DEFAULT_GOAL_COLLECTOR_SOCKET


def test_collector_rejects_non_uuid_run_id_early():
    with pytest.raises(live.GoalLiveEvidenceError) as failure:
        _collector(run_id="capability-run-1")
    assert failure.value.code == "goal_collector_configuration_invalid"


def test_writer_projection_binding_joins_frames_provenance_and_plan_readback():
    frames, export = _projection_binding_fixture()

    observed = live.validate_goal_canonical_projection_binding(frames, export)

    assert observed.case_id == "case:goal-live-projection"
    assert tuple(observed.canonical_event_pairs) == (
        "31111111-1111-4111-8111-111111111111",
    )
    assert tuple(observed.readback_plan_pairs) == (
        "31111111-1111-4111-8111-111111111111",
    )


def test_goal_evidence_assembler_joins_all_native_authorities():
    source = _goal_assembler_fixture()

    observed = live.build_goal_continuation_evidence(
        fixture=source["fixture"],
        fixture_sha256=source["fixture_sha256"],
        owner_approval_receipt_sha256=source["approval_sha256"],
        challenge=source["challenge"],
        connector_rows=source["connector_rows"],
        frames=source["frames"],
        native_receipts=source["native_receipts"],
        restart=source["restart"],
        projection_binding=source["projection_binding"],
        production_diff=source["production_diff"],
    )

    assert [item["outcome"] for item in observed["model_outcomes"]] == [
        "continue",
        "continue",
        "complete",
    ]
    assert observed["gateway_restart"]["continuation_turn_id"] == "goal-turn-2"
    assert observed["ctw_recovery"]["terminal_state"] == "completed"
    assert observed["terminal_routeback"]["message_id"] == (
        "1527000000000000099"
    )
    assert observed["user_preemption_queue_e2e"][
        "transport_redelivery_required"
    ] is False
    assert observed["terminal"]["production_diff_sha256"] == source[
        "production_diff"
    ]["diff_sha256"]


def test_goal_evidence_assembler_rejects_substituted_native_receipt():
    source = _goal_assembler_fixture()
    receipts = source["native_receipts"]
    finalizations = copy.deepcopy(dict(receipts.finalizations))
    intent, finalization = finalizations["goal-turn-2"]
    finalization["gateway_main_pid"] += 1
    source["native_receipts"] = live.GoalNativeReceiptBundle(
        recovery=receipts.recovery,
        preemption=receipts.preemption,
        finalizations=finalizations,
    )

    with pytest.raises(live.GoalLiveEvidenceError) as failure:
        live.build_goal_continuation_evidence(
            fixture=source["fixture"],
            fixture_sha256=source["fixture_sha256"],
            owner_approval_receipt_sha256=source["approval_sha256"],
            challenge=source["challenge"],
            connector_rows=source["connector_rows"],
            frames=source["frames"],
            native_receipts=source["native_receipts"],
            restart=source["restart"],
            projection_binding=source["projection_binding"],
            production_diff=source["production_diff"],
        )
    assert failure.value.code == "goal_native_receipt_invalid"


def test_writer_projection_export_reader_binds_exact_file_and_receipt(tmp_path: Path):
    _frames, export = _projection_binding_fixture()
    path = tmp_path / "canonical-events.json"
    raw = live._canonical_bytes(export) + b"\n"
    path.write_bytes(raw)
    os.chown(path, os.getuid(), os.getgid())
    path.chmod(0o640)
    provenance = export["provenance"]
    receipt = {
        "event_count": 1,
        "provenance_count": 1,
        "provenance_sha256": live._sha256_json(provenance),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
        "owner_uid": os.getuid(),
        "group_gid": os.getgid(),
        "mode": "0640",
        "stdout_receipt": {"event_count": 1, "success": True},
    }

    observed = live.read_writer_projection_export(
        path,
        expected_writer_uid=os.getuid(),
        expected_projector_gid=os.getgid(),
        export_receipt=receipt,
    )

    assert observed == export
    receipt["provenance_sha256"] = "f" * 64
    with pytest.raises(live.GoalLiveEvidenceError) as failure:
        live.read_writer_projection_export(
            path,
            expected_writer_uid=os.getuid(),
            expected_projector_gid=os.getgid(),
            export_receipt=receipt,
        )
    assert failure.value.code == "goal_projection_export_invalid"


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("frame_idempotency", "f" * 64),
        ("plan_next_step", "step:substituted"),
        ("provenance_origin", "model:substituted"),
    ),
)
def test_writer_projection_binding_fails_closed_on_substitution(mutation, value):
    frames, export = _projection_binding_fixture()
    if mutation == "frame_idempotency":
        frames[0].value["payload"]["idempotency_key_sha256"] = value
    elif mutation == "plan_next_step":
        frames[1].value["payload"]["plan_identities"][0]["next_step_id"] = value
    else:
        export["provenance"][0]["origin"] = value

    with pytest.raises(live.GoalLiveEvidenceError) as failure:
        live.validate_goal_canonical_projection_binding(frames, export)
    assert failure.value.code == "goal_projection_binding_invalid"


def test_collector_rejects_unknown_event_and_post_without_pre():
    collector = _collector()
    service = _service()
    ready = _frame(
        collector,
        event="goal_plugin_ready",
        sequence=1,
        payload=_ready_payload(collector),
    )
    _admit(collector, ready, service)
    unknown = _frame(
        collector,
        event="goal_unknown",
        sequence=2,
        session="session-1",
        turn="turn-1",
        payload={},
    )
    with pytest.raises(live.GoalLiveEvidenceError):
        collector._validate_frame(
            unknown,
            live.GoalPeer(4242, 1001, 1002),
            service,
        )

    post = _frame(
        collector,
        event="goal_post_api_request",
        sequence=2,
        session="session-1",
        turn="turn-1",
        payload={
            "request_ordinal": 1,
            "api_request_id_sha256": _digest("api"),
            "finish_reason_sha256": _digest("finish"),
            "response_model_sha256": _digest("model"),
            "response_payload_sha256": _digest("response"),
            "assistant_tool_call_id_sha256s": [],
            "response_observed_at_unix_ms": NOW,
        },
    )
    with pytest.raises(live.GoalLiveEvidenceError) as failure:
        collector._validate_frame(
            post,
            live.GoalPeer(4242, 1001, 1002),
            service,
        )
    assert failure.value.code == "goal_collector_event_payload_invalid"


def test_collector_accepts_exact_pre_post_outcome_turn_order():
    collector = _collector()
    service = _service()
    session = "session-1"
    turn = "turn-1"
    api = _digest("api")
    tool = _digest("todo-tool")
    _admit(
        collector,
        _frame(
            collector,
            event="goal_plugin_ready",
            sequence=1,
            payload=_ready_payload(collector),
        ),
        service,
    )
    _admit(
        collector,
        _frame(
            collector,
            event="goal_pre_api_request",
            sequence=2,
            session=session,
            turn=turn,
            payload={
                "request_ordinal": 1,
                "task_id_sha256": _digest("task"),
                "api_request_id_sha256": api,
                "runtime_api_call_count": 0,
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "model": "gpt-5.6-sol",
                "base_url_sha256": hashlib.sha256(
                    b"https://chatgpt.com/backend-api/codex"
                ).hexdigest(),
                "system_prompt_sha256": _digest("prompt"),
                "tool_schema_sha256": _digest("tools"),
                "reasoning_effort": "high",
                "started_at_unix_ms": NOW,
            },
        ),
        service,
    )
    _admit(
        collector,
        _frame(
            collector,
            event="goal_post_api_request",
            sequence=3,
            session=session,
            turn=turn,
            payload={
                "request_ordinal": 1,
                "api_request_id_sha256": api,
                "finish_reason_sha256": _digest("finish"),
                "response_model_sha256": _digest("model"),
                "response_payload_sha256": _digest("response"),
                "assistant_tool_call_id_sha256s": [tool],
                "response_observed_at_unix_ms": NOW,
            },
        ),
        service,
    )
    _admit(
        collector,
        _frame(
            collector,
            event="goal_model_outcome",
            sequence=4,
            session=session,
            turn=turn,
            payload={
                "api_request_id_sha256": api,
                "tool_call_id_sha256": tool,
                "outcome": "continue",
                "reason_sha256": _digest("reason"),
                "recorded": True,
                "result_sha256": _digest("result"),
            },
        ),
        service,
    )
    _admit(
        collector,
        _frame(
            collector,
            event="goal_turn_end",
            sequence=5,
            session=session,
            turn=turn,
            payload={
                "completed": True,
                "interrupted": False,
                "model_sha256": _digest("gpt-5.6-sol"),
            },
        ),
        service,
    )

    assert len(collector.frames) == 5
    assert turn in collector._ended_turn_ids


def test_collector_independently_rejects_reported_module_mismatch():
    collector = _collector(
        plugin_module_validator=lambda *_args: live._fail(
            "goal_collector_plugin_module_invalid"
        )
    )
    ready = _frame(
        collector,
        event="goal_plugin_ready",
        sequence=1,
        payload=_ready_payload(collector),
    )
    with pytest.raises(live.GoalLiveEvidenceError) as failure:
        collector._validate_frame(
            ready,
            live.GoalPeer(4242, 1001, 1002),
            _service(),
        )
    assert failure.value.code == "goal_collector_plugin_module_invalid"


def test_packaged_plugin_attestation_reads_exact_file_and_rejects_substitution(
    tmp_path: Path,
):
    module = (
        tmp_path
        / REVISION
        / "venv/lib/python/site-packages/plugins/muncho_canary_evidence/__init__.py"
    )
    module.parent.mkdir(parents=True)
    module.write_bytes(b"trusted packaged plugin")
    module.chmod(0o444)
    digest = hashlib.sha256(module.read_bytes()).hexdigest()

    live.validate_packaged_goal_plugin_module(
        str(module),
        digest,
        REVISION,
        release_base=tmp_path,
        expected_uid=os.getuid(),
    )
    with pytest.raises(live.GoalLiveEvidenceError):
        live.validate_packaged_goal_plugin_module(
            str(module),
            _digest("substituted"),
            REVISION,
            release_base=tmp_path,
            expected_uid=os.getuid(),
        )
    replacement = module.with_name("replacement.py")
    replacement.write_bytes(b"trusted packaged plugin")
    replacement.chmod(0o444)
    module.unlink()
    module.symlink_to(replacement)
    with pytest.raises(live.GoalLiveEvidenceError):
        live.validate_packaged_goal_plugin_module(
            str(module),
            digest,
            REVISION,
            release_base=tmp_path,
            expected_uid=os.getuid(),
        )


def test_challenge_durable_receipt_contains_no_plaintext(tmp_path: Path, monkeypatch):
    collector = _collector()
    run_root = tmp_path / RUN_ID
    run_root.mkdir()
    captured = {}

    def publish(path, payload, **_kwargs):
        captured["path"] = path
        captured["value"] = json.loads(payload)
        return (1, 2, 0, 0, 0o400, len(payload))

    monkeypatch.setattr(live, "_publish_exclusive", publish)
    challenge = collector.publish_challenge(
        receipt_root=tmp_path,
        goal_command="/goal secret owner task",
        kickoff_message="begin secret owner task",
        preemption_message="urgent secret owner input",
    )

    raw = live._canonical_bytes(captured["value"])
    assert b"secret owner" not in raw
    assert captured["value"]["message_content_recorded"] is False
    assert challenge.transient_input["goal_command"] == "/goal secret owner task"
    assert "goal_command" not in challenge.receipt


def test_api_observer_retirement_marker_is_config_pinned_and_one_shot(monkeypatch):
    collector = _collector()
    config = collector.observer_config()
    expected_sha256 = config["api_observer_retirement"]["marker_sha256"]
    published = False
    captured = {}

    monkeypatch.setattr(
        live.os.path,
        "lexists",
        lambda path: published
        if Path(path) == live.API_OBSERVER_RETIREMENT_PATH
        else False,
    )

    def publish(path, payload, **_kwargs):
        nonlocal published
        published = True
        captured["path"] = path
        captured["value"] = json.loads(payload)
        return (1, 2, 0, 0, 0o440, len(payload))

    monkeypatch.setattr(live, "_publish_exclusive", publish)
    marker = collector.publish_api_observer_retirement()

    assert marker == captured["value"]
    assert marker["marker_sha256"] == expected_sha256
    assert marker["historical_api_observer_terminal"] is True
    assert captured["path"] == live.API_OBSERVER_RETIREMENT_PATH
    with pytest.raises(live.GoalLiveEvidenceError) as failure:
        collector.publish_api_observer_retirement()
    assert failure.value.code == "goal_api_observer_retirement_replayed"


def test_controlled_restart_requires_marker_two_continues_and_rotates_identity(
    monkeypatch,
    tmp_path: Path,
):
    pre = _service()
    post = _service(pid=5252, invocation="3" * 32)
    post = live.GatewayServiceIdentity(
        **{
            **post.to_mapping(),
            "active_enter_timestamp_monotonic": 200,
            "identity_sha256": live._sha256_json(
                {
                    "service_unit": live.GATEWAY_UNIT_NAME,
                    "main_pid": 5252,
                    "invocation_id": "3" * 32,
                    "active_enter_timestamp_monotonic": 200,
                    "n_restarts": 0,
                }
            ),
        }
    )
    current = [pre]
    collector = _collector(service_identity_reader=lambda: current[0])
    segment = "2" * 32
    collector._segment_order.append(segment)
    collector._segment_services[segment] = pre
    peer = live.GoalPeer(pre.main_pid, 1001, 1002)
    for ordinal in (1, 2):
        collector._frames.append(
            live.GoalCollectedFrame(
                value={
                    "event": "goal_model_outcome",
                    "payload": {"outcome": "continue"},
                    "observed_at_unix_ms": NOW,
                },
                frame_sha256=_digest(f"outcome-{ordinal}"),
                segment_chain_head_sha256=_digest(f"chain-{ordinal}"),
                previous_segment_terminal_sha256="0" * 64,
                peer=peer,
                service_identity=pre,
            )
        )
    marker_path = tmp_path / "api-observer-retired.json"
    marker_path.write_bytes(b"sealed retirement marker")
    marker_path.chmod(0o440)
    monkeypatch.setattr(live, "API_OBSERVER_RETIREMENT_PATH", marker_path)
    collector._published[marker_path] = live._identity(marker_path)
    commands = []
    validated_frames = []

    def restart(command, **_kwargs):
        commands.append(command)
        current[0] = post
        second = "4" * 32
        collector._segment_order.append(second)
        collector._segment_services[second] = post
        return object()

    observation = collector.controlled_gateway_restart(
        deadline=live.time.monotonic() + 5,
        pre_restart_validator=lambda frames: (
            validated_frames.append(tuple(frames)) or True
        ),
        runner=restart,
    )

    assert commands == [
        ["/usr/bin/systemctl", "restart", live.GATEWAY_UNIT_NAME]
    ]
    assert observation.pre_service_identity == pre
    assert observation.post_service_identity == post
    assert len(validated_frames) == 1
    assert len(validated_frames[0]) == 2


@pytest.mark.parametrize(
    ("validator", "expected_code"),
    (
        (lambda _frames: False, "goal_gateway_restart_native_receipt_timeout"),
        (lambda _frames: None, "goal_gateway_restart_validator_invalid"),
    ),
)
def test_controlled_restart_fails_before_systemctl_without_durable_seals(
    validator,
    expected_code,
):
    pre = _service()
    collector = _collector(service_identity_reader=lambda: pre)
    segment = "2" * 32
    collector._segment_order.append(segment)
    collector._segment_services[segment] = pre
    peer = live.GoalPeer(pre.main_pid, 1001, 1002)
    for ordinal in (1, 2):
        collector._frames.append(
            live.GoalCollectedFrame(
                value={
                    "event": "goal_model_outcome",
                    "payload": {"outcome": "continue"},
                    "observed_at_unix_ms": NOW,
                },
                frame_sha256=_digest(f"unsealed-outcome-{ordinal}"),
                segment_chain_head_sha256=_digest(
                    f"unsealed-chain-{ordinal}"
                ),
                previous_segment_terminal_sha256="0" * 64,
                peer=peer,
                service_identity=pre,
            )
        )
    commands = []

    with pytest.raises(live.GoalLiveEvidenceError) as failure:
        collector.controlled_gateway_restart(
            deadline=live.time.monotonic() - 1,
            pre_restart_validator=validator,
            runner=lambda command, **_kwargs: commands.append(command),
        )

    assert failure.value.code == expected_code
    assert commands == []


def _sqlite_db(tmp_path: Path, name: str = "state.db") -> Path:
    tmp_path.chmod(0o700)
    path = tmp_path / name
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE state_meta(key TEXT PRIMARY KEY, value TEXT)")
    connection.execute("INSERT INTO state_meta VALUES('key','value')")
    connection.commit()
    connection.close()
    path.chmod(0o600)
    return path


def test_sqlite_reader_accepts_exact_wal_and_rejects_unexpected_sidecar(tmp_path: Path):
    path = _sqlite_db(tmp_path)
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("INSERT OR REPLACE INTO state_meta VALUES('wal','present')")
    writer.commit()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists():
            sidecar.chmod(0o600)
    rows = live._sqlite_read_rows(
        path,
        query="SELECT value FROM state_meta WHERE key='wal'",
        expected_uid=os.getuid(),
        expected_gid=path.lstat().st_gid,
        expected_parent_uid=os.getuid(),
        expected_parent_gid=path.parent.lstat().st_gid,
        code="test_sqlite_invalid",
    )
    assert rows == [("present",)]

    unexpected = Path(str(path) + "-journal")
    unexpected.write_bytes(b"unexpected")
    unexpected.chmod(0o600)
    with pytest.raises(live.GoalLiveEvidenceError) as failure:
        live._sqlite_read_rows(
            path,
            query="SELECT 1",
            expected_uid=os.getuid(),
            expected_gid=path.lstat().st_gid,
            expected_parent_uid=os.getuid(),
            expected_parent_gid=path.parent.lstat().st_gid,
            code="test_sqlite_invalid",
        )
    assert failure.value.code == "test_sqlite_invalid"
    writer.close()


def test_native_goal_receipts_are_one_exact_stable_snapshot(tmp_path: Path):
    path = _sqlite_db(tmp_path)
    session_id = "capability_goal_native_snapshot"
    turn_id = "goal-turn-1"
    turn_sha256 = hashlib.sha256(turn_id.encode()).hexdigest()

    def receipt(schema, digest_field, label):
        unsigned = {"schema": schema, "label": label}
        return live._canonical_bytes(
            {**unsigned, digest_field: live._sha256_json(unsigned)}
        ).decode()

    values = {
        f"capability_goal_lineage_recovery:{session_id}": receipt(
            live.GOAL_RECOVERY_SCHEMA,
            "recovery_sha256",
            "recovery",
        ),
        f"capability_goal_preemption:{session_id}": receipt(
            live.GOAL_PREEMPTION_SCHEMA,
            "preemption_sha256",
            "preemption",
        ),
        (
            f"capability_goal_finalization:{session_id}:{turn_sha256}:before"
        ): receipt(
            live.GOAL_FINALIZATION_INTENT_SCHEMA,
            "intent_sha256",
            "before",
        ),
        (
            f"capability_goal_finalization:{session_id}:{turn_sha256}:after"
        ): receipt(
            live.GOAL_FINALIZATION_SCHEMA,
            "finalization_sha256",
            "after",
        ),
    }
    connection = sqlite3.connect(path)
    connection.executemany(
        "INSERT INTO state_meta(key,value) VALUES(?,?)",
        tuple(values.items()),
    )
    connection.commit()
    connection.close()
    path.chmod(0o600)

    observed = live.read_goal_native_receipts(
        session_id,
        [turn_id],
        path=path,
        expected_uid=os.getuid(),
        expected_gid=path.lstat().st_gid,
        expected_parent_uid=os.getuid(),
        expected_parent_gid=path.parent.lstat().st_gid,
    )

    assert observed.recovery["label"] == "recovery"
    assert observed.preemption["label"] == "preemption"
    assert observed.finalizations[turn_id][0]["label"] == "before"
    assert observed.finalizations[turn_id][1]["label"] == "after"

    seals = live.read_goal_native_finalizations(
        session_id,
        [turn_id],
        path=path,
        expected_uid=os.getuid(),
        expected_gid=path.lstat().st_gid,
        expected_parent_uid=os.getuid(),
        expected_parent_gid=path.parent.lstat().st_gid,
    )
    assert seals[turn_id][0]["label"] == "before"
    assert seals[turn_id][1]["label"] == "after"


def test_native_goal_finalizations_distinguish_not_yet_sealed(tmp_path: Path):
    path = _sqlite_db(tmp_path)
    with pytest.raises(live.GoalLiveEvidenceError) as failure:
        live.read_goal_native_finalizations(
            "capability_goal_waiting",
            ["goal-turn-1"],
            path=path,
            expected_uid=os.getuid(),
            expected_gid=path.lstat().st_gid,
            expected_parent_uid=os.getuid(),
            expected_parent_gid=path.parent.lstat().st_gid,
        )
    assert failure.value.code == "goal_native_receipt_unavailable"


def test_sqlite_reader_rejects_symlink_and_identity_substitution(tmp_path: Path, monkeypatch):
    path = _sqlite_db(tmp_path)
    database_gid = path.lstat().st_gid
    parent_gid = path.parent.lstat().st_gid
    link = tmp_path / "linked.db"
    link.symlink_to(path)
    with pytest.raises(live.GoalLiveEvidenceError):
        live._sqlite_read_rows(
            link,
            query="SELECT 1",
            expected_uid=os.getuid(),
            expected_gid=database_gid,
            expected_parent_uid=os.getuid(),
            expected_parent_gid=parent_gid,
            code="test_sqlite_invalid",
        )

    original = live._sqlite_identity_snapshot
    calls = 0

    def substituted(*args, **kwargs):
        nonlocal calls
        calls += 1
        value = original(*args, **kwargs)
        if calls == 2:
            parent, entries = value
            changed = list(entries)
            name, identity = changed[0]
            changed[0] = (name, (identity[0], identity[1] + 1, *identity[2:]))
            return parent, tuple(changed)
        return value

    monkeypatch.setattr(live, "_sqlite_identity_snapshot", substituted)
    with pytest.raises(live.GoalLiveEvidenceError) as failure:
        live._sqlite_read_rows(
            path,
            query="SELECT 1",
            expected_uid=os.getuid(),
            expected_gid=database_gid,
            expected_parent_uid=os.getuid(),
            expected_parent_gid=parent_gid,
            code="test_sqlite_invalid",
        )
    assert failure.value.code == "test_sqlite_invalid"


def test_connector_reader_uses_exact_wal_snapshot(tmp_path: Path):
    tmp_path.chmod(0o700)
    path = tmp_path / "connector.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE connector_events_v1(
        event_id TEXT PRIMARY KEY,event_sha256 TEXT,event_json TEXT,state TEXT,
        delivery_id TEXT,lease_until_unix_ms INTEGER,offered_at_unix_ms INTEGER,
        acked_at_unix_ms INTEGER)"""
    )
    event = DiscordConnectorEvent(
        event_id="323456789012345678",
        target=DiscordConnectorTarget.from_mapping(TARGET),
        author_id="1279454038731264061",
        author_name="Emo",
        author_is_bot=False,
        content="transient owner input",
        created_at_unix_ms=NOW,
        reply_to_message_id=None,
    )
    event_json = live._canonical_bytes(event.to_mapping()).decode()
    connection.execute(
        "INSERT INTO connector_events_v1 VALUES(?,?,?,?,?,?,?,?)",
        (
            event.event_id,
            event.sha256,
            event_json,
            "acked",
            "delivery-1",
            None,
            NOW,
            NOW + 1,
        ),
    )
    connection.commit()
    connection.close()
    path.chmod(0o600)

    rows = live.read_connector_rows(
        path,
        expected_uid=os.getuid(),
        expected_gid=path.lstat().st_gid,
        expected_parent_uid=os.getuid(),
        expected_parent_gid=path.parent.lstat().st_gid,
    )
    assert len(rows) == 1
    assert rows[0].event.sha256 == event.sha256
    assert rows[0].state == "acked"
