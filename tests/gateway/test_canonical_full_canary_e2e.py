from __future__ import annotations

import hashlib
import json
import uuid
from copy import deepcopy
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import gateway.canonical_full_canary_e2e as e2e_module
from gateway.canonical_full_canary_e2e import (
    CANONICAL_TRUTH_RECEIPT_SCHEMA,
    EVIDENCE_SCHEMA,
    FIXTURE_SCHEMA,
    INVARIANT_RECEIPT_SCHEMA,
    MODEL_CALL_RECEIPT_SCHEMA,
    PRIVATE_DENIAL_RECEIPT_SCHEMA,
    REASONING_RECEIPT_SCHEMA,
    SOURCE_RECEIPT_SCHEMA,
    TASK_OUTCOME_RECEIPT_SCHEMA,
    CanaryEvidenceError,
    main,
    verify_evidence,
)
from gateway.discord_edge_protocol import (
    DiscordEdgeAuthorityKind,
    DiscordEdgeIntent,
    DiscordEdgeOperation,
    DiscordEdgeReceiptOutcome,
    DiscordPublicTarget,
    make_request,
    sign_capability,
    sign_receipt,
    verify_request_capability_for_reconciliation,
)
from gateway.discord_edge_writer_authority import (
    derive_routeback_edge_idempotency_key,
)


RELEASE_SHA = "a" * 40
RELEASE_ARTIFACT_SHA256 = "f" * 64
RUN_ID = "11111111-1111-4111-8111-111111111111"
SESSION_ID = "api_server:canary-session"
TURN_ID = "turn:full-canary:1"
CASE_ID = "case:full-canary:1"
PLAN_ID = "plan:full-canary:1"
GUILD_ID = "100000000000000001"
CHANNEL_ID = "100000000000000002"
MESSAGE_ID = "100000000000000003"
BOT_ID = "100000000000000004"
OWNER_ID = "100000000000000005"
NOW_MS = 2_000_000_000_000
SESSION_KEY_SHA256 = "d" * 64
CAPABILITY_EPOCH_SHA256 = "e" * 64
APPROVAL_SOURCE_SHA256 = "8" * 64
SCOPE_GRANT_ID = "canary-grant:full-canary:1"


def _canonical_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _public_hex(key):
    return (
        key
        .public_key()
        .public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        .hex()
    )


def _readiness():
    return {
        "version": "canonical-writer-readiness-v1",
        "observed_at_unix": NOW_MS // 1000,
        "observed_at_boottime_ns": 999,
        "boot_id_sha256": "b" * 64,
        "gateway_pid": 4242,
        "gateway_start_time_ticks": 31337,
        "writer_request_id": "22222222-2222-4222-8222-222222222222",
        "writer_service": "canonical_writer",
        "writer_protocol": "v1",
        "database_identity": "canonical_brain_migration_owner",
        "gateway_module_origin": "/opt/muncho/releases/a/gateway/readiness.py",
        "gateway_module_sha256": "c" * 64,
        "gateway_dumpable": False,
        "gateway_core_soft_limit": 0,
        "gateway_core_hard_limit": 0,
        "effective_import_paths": ["/opt/muncho/releases/a/site-packages"],
        "unexpected_import_paths": [],
        "loaded_module_origins": ["/opt/muncho/releases/a/gateway/readiness.py"],
        "unexpected_import_origins": [],
        "loaded_module_origins_complete": True,
        "effective_environment_variable_names": ["NOTIFY_SOCKET"],
        "effective_environment_variable_value_sha256": {
            "NOTIFY_SOCKET": "d" * 64,
        },
    }


def _model_call(ordinal, effort, tool_ids):
    return {
        "schema": MODEL_CALL_RECEIPT_SCHEMA,
        "provenance": "live_gateway_model_adapter",
        "release_sha": RELEASE_SHA,
        "canary_run_id": RUN_ID,
        "session_id": SESSION_ID,
        "turn_id": TURN_ID,
        "request_ordinal": ordinal,
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "model": "gpt-5.6-sol",
        "reasoning_effort": effort,
        "api_request_sha256": f"{ordinal}" * 64,
        "response_payload_sha256": f"{ordinal + 3}" * 64,
        "response_model": "gpt-5.6-sol",
        "response_observed_at_unix_ms": NOW_MS + ordinal,
        "assistant_tool_call_ids": tool_ids,
    }


def _plan_event(revision, step_ids, criterion_ids, verification_ids):
    completed = revision - 1
    final = completed == len(step_ids)
    steps = []
    for index, step_id in enumerate(step_ids):
        status = (
            "completed"
            if index < completed
            else ("in_progress" if index == completed and not final else "pending")
        )
        steps.append({
            "id": step_id,
            "status": status,
            "depends_on": [] if index == 0 else [step_ids[index - 1]],
        })
    current = None if final else step_ids[completed]
    return {
        "event_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"plan:{revision}")),
        "event_type": "task.plan.updated",
        "case_id": CASE_ID,
        "readback_verified": True,
        "canonical_content_sha256": hashlib.sha256(
            f"plan:{revision}".encode()
        ).hexdigest(),
        "plan": {
            "plan_id": PLAN_ID,
            "revision": revision,
            "state": "completed" if final else "active",
            "current_step_id": current,
            "resume_cursor": {"next_step_id": current},
            "steps": steps,
            "criterion_ids": criterion_ids,
            "verification_event_ids": verification_ids if final else [],
        },
    }


def _bundle():
    writer_key = Ed25519PrivateKey.generate()
    edge_key = Ed25519PrivateKey.generate()
    step_ids = ["collect", "analyze", "verify"]
    criterion_ids = ["criterion:collect", "criterion:analyze", "criterion:verify"]
    verification_event_id = "33333333-3333-4333-8333-333333333333"
    canonical_idempotency_key = "routeback:full-canary:1"
    content = "Full canary completed with exact verified receipts."
    content_sha256 = hashlib.sha256(content.encode()).hexdigest()
    prompt = "Owner-approved live canary prompt"
    prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
    target = DiscordPublicTarget.from_mapping({
        "target_type": "public_guild_channel",
        "guild_id": GUILD_ID,
        "channel_id": CHANNEL_ID,
    })
    authorization_id = "routeauth:full-canary:1"
    intent = DiscordEdgeIntent(
        operation=DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
        target=target,
        payload={"content": content},
        idempotency_key=derive_routeback_edge_idempotency_key(
            case_id=CASE_ID,
            canonical_idempotency_key=canonical_idempotency_key,
        ),
    )
    capability_envelope = sign_capability(
        writer_key,
        intent,
        authority_kind=DiscordEdgeAuthorityKind.CANONICAL_ROUTEBACK,
        authority_ref=authorization_id,
        issued_at_unix_ms=NOW_MS,
        expires_at_unix_ms=NOW_MS + 60_000,
    )
    request = make_request(
        intent,
        capability_envelope,
        request_id="44444444-4444-4444-8444-444444444444",
        now_unix_ms=NOW_MS,
    )
    capability = verify_request_capability_for_reconciliation(
        request, writer_key.public_key()
    )
    edge_receipt = sign_receipt(
        edge_key,
        request,
        capability,
        outcome=DiscordEdgeReceiptOutcome.VERIFIED,
        discord_object_id=MESSAGE_ID,
        bot_user_id=BOT_ID,
        adapter_accepted=True,
        readback_verified=True,
        readback_content_sha256=content_sha256,
        occurred_at_unix_ms=NOW_MS + 2,
        receipt_id="55555555-5555-4555-8555-555555555555",
    )
    fixture = {
        "schema": FIXTURE_SCHEMA,
        "canary_run_id": RUN_ID,
        "release_sha": RELEASE_SHA,
        "release_artifact_sha256": RELEASE_ARTIFACT_SHA256,
        "valid_from_unix_ms": NOW_MS - 1_000,
        "valid_until_unix_ms": NOW_MS + 60_000,
        "case_id": CASE_ID,
        "owner_discord_user_id": OWNER_ID,
        "source": {
            "platform": "api_server",
            "control_protocol": "authenticated_loopback_api_server.v1",
            "host": "127.0.0.1",
            "port": 8642,
            "session_create_endpoint": "/api/sessions",
            "chat_stream_endpoint_template": ("/api/sessions/{session_id}/chat/stream"),
        },
        "model_route": {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "model": "gpt-5.6-sol",
            "initial_effort": "high",
            "elevated_effort": "xhigh",
        },
        "task_policy": {
            "minimum_completed_steps": 3,
            "prompt": prompt,
            "prompt_sha256": prompt_sha256,
        },
        "public_routeback": {
            "target": target.to_dict(),
            "canonical_idempotency_key": canonical_idempotency_key,
        },
        "discord_public_keys": {
            "writer_capability_ed25519_hex": _public_hex(writer_key),
            "edge_receipt_ed25519_hex": _public_hex(edge_key),
        },
    }
    fixture_sha = _digest(fixture)
    routeback_event = {
        "event_id": "66666666-6666-4666-8666-666666666666",
        "event_type": "route_back.sent",
        "case_id": CASE_ID,
        "readback_verified": True,
        "canonical_content_sha256": "e" * 64,
        "canonical_idempotency_key": canonical_idempotency_key,
        "authorization_id": authorization_id,
        "target_ref": {"channel_id": CHANNEL_ID},
        "receipt": {
            "platform": "discord",
            "adapter_receipt": True,
            "receipt_readback_verified": True,
            "message_id": MESSAGE_ID,
            "channel_id": CHANNEL_ID,
            "content_sha256": content_sha256,
        },
    }
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "fixture_sha256": fixture_sha,
        "collected_at_unix_ms": NOW_MS + 13,
        "runtime_provenance": {
            "execution_mode": "live_isolated_canary",
            "synthetic": False,
            "release_sha": RELEASE_SHA,
            "canary_run_id": RUN_ID,
            "owner_discord_user_id": OWNER_ID,
            "full_canary_start_receipt_sha256": "0" * 64,
            "gateway_service_identity_sha256": "1" * 64,
            "canonical_writer_service_identity_sha256": "2" * 64,
            "discord_edge_service_identity_sha256": "3" * 64,
            "collector_receipt_sha256": "4" * 64,
        },
        "writer_readiness": _readiness(),
        "source_receipt": {
            "schema": SOURCE_RECEIPT_SCHEMA,
            "provenance": "live_gateway_authenticated_loopback_api",
            "release_sha": RELEASE_SHA,
            "canary_run_id": RUN_ID,
            "session_id": SESSION_ID,
            "turn_id": TURN_ID,
            "platform": "api_server",
            "control_protocol": "authenticated_loopback_api_server.v1",
            "host": "127.0.0.1",
            "port": 8642,
            "session_create_endpoint": "/api/sessions",
            "chat_stream_endpoint": f"/api/sessions/{SESSION_ID}/chat/stream",
            "session_create_request_id": "88888888-8888-4888-8888-888888888888",
            "chat_stream_request_id": "99999999-9999-4999-8999-999999999999",
            "api_run_id": "run_11111111111111111111111111111111",
            "api_message_id": "msg_11111111111111111111111111111111",
            "loopback_peer_verified": True,
            "credential_provenance_receipt_sha256": "c" * 64,
            "session_key_sha256": SESSION_KEY_SHA256,
            "capability_epoch_sha256": CAPABILITY_EPOCH_SHA256,
            "message_content_sha256": prompt_sha256,
            "observed_at_unix_ms": NOW_MS,
        },
        "model_calls": [
            _model_call(1, "high", ["call:reasoning", "call:step:collect"]),
            _model_call(2, "xhigh", ["call:step:analyze"]),
            _model_call(3, "xhigh", ["call:step:verify"]),
            _model_call(4, "xhigh", []),
        ],
        "reasoning_directive": {
            "schema": REASONING_RECEIPT_SCHEMA,
            "provenance": "live_gateway_assistant_tool_call",
            "release_sha": RELEASE_SHA,
            "canary_run_id": RUN_ID,
            "session_id": SESSION_ID,
            "turn_id": TURN_ID,
            "tool_name": "todo",
            "tool_call_id": "call:reasoning",
            "model_authored": True,
            "directive": {"effort": "xhigh"},
            "reasoning_control": {
                "status": "applied",
                "scope": "current_turn",
                "expires": "end_of_current_turn",
                "effective": {"effort": "xhigh"},
                "change_count": 1,
            },
            "produced_by_model_call_ordinal": 1,
            "applied_before_model_call_ordinal": 2,
            "todo_result_sha256": "5" * 64,
        },
        "canonical_truth": {
            "schema": CANONICAL_TRUTH_RECEIPT_SCHEMA,
            "provenance": "canonical_writer_live_readback",
            "release_sha": RELEASE_SHA,
            "canary_run_id": RUN_ID,
            "writer_query_request_id": "77777777-7777-4777-8777-777777777777",
            "observed_at_unix_ms": NOW_MS + 12,
            "query_status": "CANONICAL_BRAIN_QUERY_PASS",
            "query_view": "resume_bundle",
            "case_id": CASE_ID,
            "support_incomplete": False,
            "plan_projection_complete": True,
            "completion_receipts_satisfied": True,
            "missing_verification_event_ids": [],
            "scope_events": [
                {
                    "event_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                    "event_type": "canary.scope.preapproved",
                    "case_id": CASE_ID,
                    "occurred_at_unix_ms": NOW_MS - 500,
                    "readback_verified": True,
                    "canonical_content_sha256": "1" * 64,
                    "scope": {
                        "grant_id": SCOPE_GRANT_ID,
                        "case_id": CASE_ID,
                        "release_sha256": RELEASE_ARTIFACT_SHA256,
                        "fixture_sha256": fixture_sha,
                        "run_id": RUN_ID,
                        "session_key_sha256": SESSION_KEY_SHA256,
                        "expires_at_unix_ms": NOW_MS + 60_000,
                        "approved_by": OWNER_ID,
                        "approval_source_sha256": APPROVAL_SOURCE_SHA256,
                        "state": "preapproved",
                    },
                },
                {
                    "event_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                    "event_type": "canary.scope.claimed",
                    "case_id": CASE_ID,
                    "occurred_at_unix_ms": NOW_MS + 1,
                    "readback_verified": True,
                    "canonical_content_sha256": "2" * 64,
                    "scope": {
                        "grant_id": SCOPE_GRANT_ID,
                        "case_id": CASE_ID,
                        "release_sha256": RELEASE_ARTIFACT_SHA256,
                        "fixture_sha256": fixture_sha,
                        "run_id": RUN_ID,
                        "approval_source_sha256": APPROVAL_SOURCE_SHA256,
                        "session_key_sha256": SESSION_KEY_SHA256,
                        "capability_epoch_sha256": CAPABILITY_EPOCH_SHA256,
                        "expires_at_unix_ms": NOW_MS + 60_000,
                        "state": "claimed",
                    },
                },
                {
                    "event_id": "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
                    "event_type": "canary.scope.revoked",
                    "case_id": CASE_ID,
                    "occurred_at_unix_ms": NOW_MS + 10,
                    "readback_verified": True,
                    "canonical_content_sha256": "3" * 64,
                    "scope": {
                        "grant_id": SCOPE_GRANT_ID,
                        "session_key_sha256": SESSION_KEY_SHA256,
                        "capability_epoch_sha256": CAPABILITY_EPOCH_SHA256,
                        "reason": "api_server_run_finished",
                        "session_tombstone_recorded": True,
                        "state": "revoked",
                    },
                },
            ],
            "scope_retirement": {
                "grant_id": SCOPE_GRANT_ID,
                "session_key_sha256": SESSION_KEY_SHA256,
                "capability_epoch_sha256": CAPABILITY_EPOCH_SHA256,
                "authority_active": False,
                "revocation_event_id": "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
                "session_tombstone_commit_receipt_verified": True,
                "observed_at_unix_ms": NOW_MS + 11,
            },
            "plan_events": [
                _plan_event(
                    revision,
                    step_ids,
                    criterion_ids,
                    [verification_event_id],
                )
                for revision in range(1, len(step_ids) + 2)
            ],
            "verification_events": [
                {
                    "event_id": verification_event_id,
                    "event_type": "task.verification.recorded",
                    "case_id": CASE_ID,
                    "readback_verified": True,
                    "canonical_content_sha256": "6" * 64,
                    "verification": {
                        "verification_id": "verification:full-canary:1",
                        "plan_id": PLAN_ID,
                        "plan_revision": len(step_ids),
                        "criterion_ids": criterion_ids,
                        "outcome": "passed",
                        "receipt": {"kind": "live_canary", "sha256": "7" * 64},
                    },
                }
            ],
            "routeback_event": routeback_event,
        },
        "public_routeback": {
            "provenance": "live_discord_edge_signed_receipt",
            "discord_edge_request": request.to_message(),
            "discord_edge_receipt": edge_receipt.to_message(),
        },
        "private_denial": {
            "schema": PRIVATE_DENIAL_RECEIPT_SCHEMA,
            "provenance": "live_gateway_to_discord_edge_probe",
            "release_sha": RELEASE_SHA,
            "canary_run_id": RUN_ID,
            "session_id": SESSION_ID,
            "turn_id": TURN_ID,
            "observed_at_unix_ms": NOW_MS + 6,
            "discord_edge_service_identity_sha256": "3" * 64,
            "socket_identity_sha256": "8" * 64,
            "attempt_frame_sha256": "9" * 64,
            "attempted_operation": "public.message.send",
            "attempted_target_type": "dm",
            "connection_closed_without_response": True,
            "signed_receipt_observed": False,
            "journal_snapshot_before": {"record_count": 1, "logical_sha256": "a" * 64},
            "journal_snapshot_after": {"record_count": 1, "logical_sha256": "a" * 64},
        },
        "task_outcome": {
            "schema": TASK_OUTCOME_RECEIPT_SCHEMA,
            "provenance": "live_gateway_turn_completion",
            "release_sha": RELEASE_SHA,
            "canary_run_id": RUN_ID,
            "session_id": SESSION_ID,
            "turn_id": TURN_ID,
            "api_run_id": "run_11111111111111111111111111111111",
            "api_message_id": "msg_11111111111111111111111111111111",
            "case_id": CASE_ID,
            "plan_id": PLAN_ID,
            "stream_terminal_event": "run.completed",
            "completed": True,
            "partial": False,
            "interrupted": False,
            "failed": False,
            "turn_exit_reason": "text_response(finish_reason=stop)",
            "completed_at_unix_ms": NOW_MS + 9,
            "completed_steps": [
                {
                    "step_id": step_id,
                    "completion_ordinal": ordinal,
                    "tool_receipt_sha256": f"{ordinal + 6}" * 64,
                }
                for ordinal, step_id in enumerate(step_ids, start=1)
            ],
            "model_call_count": 4,
            "tool_call_count": 4,
            "final_response_sha256": "b" * 64,
        },
    }
    return fixture, evidence


def _verify(fixture, evidence):
    return verify_evidence(
        fixture,
        evidence,
        start_receipt_sha256=evidence["runtime_provenance"][
            "full_canary_start_receipt_sha256"
        ],
        fixture_sha256=_digest(fixture),
        evidence_sha256=_digest(evidence),
    )


def test_full_live_evidence_contract_passes_all_invariants():
    fixture, evidence = _bundle()

    receipt = _verify(fixture, evidence)

    assert receipt["schema"] == INVARIANT_RECEIPT_SCHEMA
    assert receipt["ok"] is True
    assert receipt["release_sha"] == RELEASE_SHA
    assert receipt["canary_run_id"] == RUN_ID
    assert len(receipt["invariants"]) == 8
    digest_payload = dict(receipt)
    digest = digest_payload.pop("invariant_receipt_sha256")
    assert digest == _digest(digest_payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda truth: truth["scope_events"][0].update(
            event_type="canary.scope.claimed"
        ),
        lambda truth: truth["scope_events"][1]["scope"].update(
            grant_id="canary-grant:another"
        ),
        lambda truth: truth["scope_events"][1]["scope"].update(
            capability_epoch_sha256="7" * 64
        ),
        lambda truth: truth["scope_events"][2]["scope"].update(
            reason="gateway_session_boundary"
        ),
        lambda truth: truth["scope_events"][2]["scope"].update(
            session_tombstone_recorded=False
        ),
        lambda truth: truth["scope_retirement"].update(authority_active=True),
        lambda truth: truth["scope_retirement"].update(
            session_tombstone_commit_receipt_verified=False
        ),
    ],
)
def test_one_shot_scope_must_be_exactly_claimed_and_durably_retired(mutate):
    fixture, evidence = _bundle()
    mutate(evidence["canonical_truth"])

    with pytest.raises(CanaryEvidenceError) as exc:
        _verify(fixture, evidence)

    assert exc.value.code == "canonical_scope_truth_invalid"


def test_scope_is_bound_to_the_sealed_release_artifact_digest():
    fixture, evidence = _bundle()
    fixture["release_artifact_sha256"] = "0" * 64
    evidence["fixture_sha256"] = _digest(fixture)

    with pytest.raises(CanaryEvidenceError) as exc:
        _verify(fixture, evidence)

    assert exc.value.code == "canonical_scope_truth_invalid"


def test_library_api_recomputes_fixture_and_evidence_digests():
    fixture, evidence = _bundle()
    fixture_digest = _digest(fixture)
    evidence_digest = _digest(evidence)
    fixture["task_policy"]["minimum_completed_steps"] = 4

    with pytest.raises(CanaryEvidenceError) as exc:
        verify_evidence(
            fixture,
            evidence,
            start_receipt_sha256=evidence["runtime_provenance"][
                "full_canary_start_receipt_sha256"
            ],
            fixture_sha256=fixture_digest,
            evidence_sha256=evidence_digest,
        )

    assert exc.value.code == "fixture_digest_invalid"


def test_fixture_preserves_model_authorship_of_runtime_identity_plan_and_content():
    fixture, evidence = _bundle()

    assert "session_id" not in fixture
    assert "turn_id" not in fixture
    assert "plan_id" not in fixture
    assert set(fixture["task_policy"]) == {
        "minimum_completed_steps",
        "prompt",
        "prompt_sha256",
    }
    assert "content_sha256" not in fixture["public_routeback"]
    assert evidence["canonical_truth"]["plan_events"][0]["plan"]["plan_id"] == PLAN_ID
    assert (
        evidence["public_routeback"]["discord_edge_receipt"]["payload"][
            "content_sha256"
        ]
        == evidence["canonical_truth"]["routeback_event"]["receipt"]["content_sha256"]
    )


def test_fixture_cannot_smuggle_external_plan_decomposition():
    fixture, evidence = _bundle()
    fixture["plan_id"] = PLAN_ID

    with pytest.raises(CanaryEvidenceError) as exc:
        _verify(fixture, evidence)

    assert exc.value.code == "fixture_shape_invalid"


def test_plan_progression_rejects_two_steps_completed_in_one_revision():
    fixture, evidence = _bundle()
    plan = evidence["canonical_truth"]["plan_events"][1]["plan"]
    plan["steps"][1]["status"] = "completed"
    plan["current_step_id"] = "verify"
    plan["resume_cursor"]["next_step_id"] = "verify"

    with pytest.raises(CanaryEvidenceError) as exc:
        _verify(fixture, evidence)

    assert exc.value.code == "canonical_plan_truth_invalid"


def test_plan_progression_completes_the_previous_canonical_cursor():
    fixture, evidence = _bundle()
    first_plan = evidence["canonical_truth"]["plan_events"][0]["plan"]
    first_plan["current_step_id"] = "analyze"
    first_plan["resume_cursor"]["next_step_id"] = "analyze"

    with pytest.raises(CanaryEvidenceError) as exc:
        _verify(fixture, evidence)

    assert exc.value.code == "canonical_plan_truth_invalid"


def test_prompt_bytes_are_bound_without_interpreting_them():
    fixture, evidence = _bundle()
    fixture["task_policy"]["prompt"] += " changed"

    with pytest.raises(CanaryEvidenceError) as exc:
        _verify(fixture, evidence)

    assert exc.value.code == "fixture_task_invalid"


def test_live_source_must_be_authenticated_loopback_control():
    fixture, evidence = _bundle()
    evidence["source_receipt"]["host"] = "0.0.0.0"

    with pytest.raises(CanaryEvidenceError) as exc:
        _verify(fixture, evidence)

    assert exc.value.code == "source_receipt_invalid"


def test_live_source_receipt_outside_owner_window_cannot_be_replayed():
    fixture, evidence = _bundle()
    evidence["source_receipt"]["observed_at_unix_ms"] = (
        fixture["valid_until_unix_ms"] + 1
    )

    with pytest.raises(CanaryEvidenceError) as exc:
        _verify(fixture, evidence)

    assert exc.value.code == "source_receipt_invalid"


def test_model_receipt_binds_sanitized_post_api_payload_not_unavailable_provider_id():
    fixture, evidence = _bundle()
    receipt = evidence["model_calls"][0]

    assert "response_payload_sha256" in receipt
    assert "response_model" in receipt
    assert "provider_response_id_sha256" not in receipt

    receipt["provider_response_id_sha256"] = "f" * 64
    with pytest.raises(CanaryEvidenceError) as exc:
        _verify(fixture, evidence)

    assert exc.value.code == "model_call_receipt_invalid"


def test_model_receipts_require_globally_unique_tool_call_ids():
    fixture, evidence = _bundle()
    evidence["model_calls"][1]["assistant_tool_call_ids"] = ["call:step:collect"]

    with pytest.raises(CanaryEvidenceError) as exc:
        _verify(fixture, evidence)

    assert exc.value.code == "model_call_receipt_invalid"


def test_final_model_receipt_must_be_the_terminal_text_response_call():
    fixture, evidence = _bundle()
    evidence["model_calls"][-1]["assistant_tool_call_ids"] = ["call:late"]
    evidence["task_outcome"]["tool_call_count"] += 1

    with pytest.raises(CanaryEvidenceError) as exc:
        _verify(fixture, evidence)

    assert exc.value.code == "model_call_receipt_invalid"


def test_task_outcome_counts_exact_observed_model_and_tool_calls():
    fixture, evidence = _bundle()
    evidence["task_outcome"]["tool_call_count"] += 1

    with pytest.raises(CanaryEvidenceError) as exc:
        _verify(fixture, evidence)

    assert exc.value.code == "task_outcome_invalid"


@pytest.mark.parametrize(
    "field,value",
    [
        ("stream_terminal_event", "run.partial"),
        ("completed", False),
        ("failed", True),
        ("interrupted", True),
        ("turn_exit_reason", "model_completed"),
        ("turn_exit_reason", "text_response(finish_reason=length)"),
    ],
)
def test_task_outcome_requires_exact_healthy_api_terminal_facts(field, value):
    fixture, evidence = _bundle()
    evidence["task_outcome"][field] = value

    with pytest.raises(CanaryEvidenceError) as exc:
        _verify(fixture, evidence)

    assert exc.value.code == "task_outcome_invalid"


def test_verifier_cross_binds_exact_lifecycle_start_receipt_digest():
    fixture, evidence = _bundle()

    with pytest.raises(CanaryEvidenceError) as exc:
        verify_evidence(
            fixture,
            evidence,
            start_receipt_sha256="f" * 64,
            fixture_sha256=_digest(fixture),
            evidence_sha256=_digest(evidence),
        )

    assert exc.value.code == "live_provenance_invalid"


@pytest.mark.parametrize(
    "mutate,code",
    [
        (
            lambda _fixture, evidence: evidence["runtime_provenance"].update(
                synthetic=True
            ),
            "live_provenance_invalid",
        ),
        (
            lambda _fixture, evidence: evidence["model_calls"][1].update(
                reasoning_effort="high"
            ),
            "model_call_receipt_invalid",
        ),
        (
            lambda _fixture, evidence: evidence["reasoning_directive"].update(
                model_authored=False
            ),
            "adaptive_reasoning_evidence_invalid",
        ),
        (
            lambda _fixture, evidence: evidence["canonical_truth"]["plan_events"][2][
                "plan"
            ]["steps"][0].update(status="blocked"),
            "canonical_plan_truth_invalid",
        ),
        (
            lambda _fixture, evidence: evidence["canonical_truth"][
                "verification_events"
            ][0]["verification"].update(outcome="inconclusive"),
            "canonical_verification_truth_invalid",
        ),
        (
            lambda _fixture, evidence: evidence["private_denial"][
                "journal_snapshot_after"
            ].update(record_count=2),
            "private_denial_invalid",
        ),
        (
            lambda _fixture, evidence: evidence["task_outcome"].update(partial=True),
            "task_outcome_invalid",
        ),
    ],
)
def test_every_invariant_fails_closed(mutate, code):
    fixture, evidence = _bundle()
    mutate(fixture, evidence)

    with pytest.raises(CanaryEvidenceError) as exc:
        _verify(fixture, evidence)

    assert exc.value.code == code


def test_tampered_discord_signature_cannot_create_sent_truth():
    fixture, evidence = _bundle()
    signature = evidence["public_routeback"]["discord_edge_receipt"]["signature"]
    evidence["public_routeback"]["discord_edge_receipt"]["signature"] = (
        "A" if signature[0] != "A" else "B"
    ) + signature[1:]

    with pytest.raises(CanaryEvidenceError) as exc:
        _verify(fixture, evidence)

    assert exc.value.code == "public_routeback_invalid"


def test_signed_receipt_must_also_exist_as_canonical_routeback_sent():
    fixture, evidence = _bundle()
    evidence["canonical_truth"]["routeback_event"]["event_type"] = "route_back.blocked"

    with pytest.raises(CanaryEvidenceError) as exc:
        _verify(fixture, evidence)

    assert exc.value.code == "canonical_routeback_truth_invalid"


def test_cli_is_digest_bound_and_prints_only_canonical_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    fixture, evidence = _bundle()
    fixture_body = _canonical_bytes(fixture)
    evidence_body = _canonical_bytes(evidence)
    fixture_path = tmp_path / "fixture.json"
    evidence_path = tmp_path / "evidence.json"
    fixture_path.write_bytes(fixture_body)
    evidence_path.write_bytes(evidence_body)

    result = main([
        "verify",
        "--start-receipt-sha256",
        evidence["runtime_provenance"]["full_canary_start_receipt_sha256"],
        "--fixture",
        str(fixture_path),
        "--fixture-sha256",
        hashlib.sha256(fixture_body).hexdigest(),
        "--evidence",
        str(evidence_path),
        "--evidence-sha256",
        hashlib.sha256(evidence_body).hexdigest(),
    ])

    output = capsys.readouterr().out
    receipt = json.loads(output)
    assert result == 0
    assert receipt["ok"] is True
    assert output == _canonical_bytes(receipt).decode() + "\n"
    assert content_not_exposed(output)


def content_not_exposed(output):
    return (
        "Full canary completed" not in output
        and "Owner-approved live canary prompt" not in output
        and "NOTIFY_SOCKET" not in output
    )


def test_cli_rejects_digest_mismatch_without_echoing_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    fixture, evidence = _bundle()
    fixture_path = tmp_path / "fixture.json"
    evidence_path = tmp_path / "evidence.json"
    fixture_path.write_bytes(_canonical_bytes(fixture))
    evidence_path.write_bytes(_canonical_bytes(evidence))

    result = main([
        "verify",
        "--start-receipt-sha256",
        evidence["runtime_provenance"]["full_canary_start_receipt_sha256"],
        "--fixture",
        str(fixture_path),
        "--fixture-sha256",
        "f" * 64,
        "--evidence",
        str(evidence_path),
        "--evidence-sha256",
        _digest(evidence),
    ])

    receipt = json.loads(capsys.readouterr().out)
    assert result == 2
    assert receipt["ok"] is False
    assert receipt["failure_code"] == "fixture_artifact_invalid"
    assert set(receipt) == {
        "schema",
        "ok",
        "fixture_sha256",
        "evidence_sha256",
        "failure_code",
        "invariant_receipt_sha256",
    }


def test_cli_rejects_symlink_artifact(tmp_path: Path, capsys):
    fixture, evidence = _bundle()
    real_fixture = tmp_path / "fixture-real.json"
    fixture_link = tmp_path / "fixture.json"
    evidence_path = tmp_path / "evidence.json"
    real_fixture.write_bytes(_canonical_bytes(fixture))
    fixture_link.symlink_to(real_fixture)
    evidence_path.write_bytes(_canonical_bytes(evidence))

    result = main([
        "verify",
        "--start-receipt-sha256",
        evidence["runtime_provenance"]["full_canary_start_receipt_sha256"],
        "--fixture",
        str(fixture_link),
        "--fixture-sha256",
        _digest(fixture),
        "--evidence",
        str(evidence_path),
        "--evidence-sha256",
        _digest(evidence),
    ])

    receipt = json.loads(capsys.readouterr().out)
    assert result == 2
    assert receipt["failure_code"] == "fixture_artifact_invalid"


def test_bound_artifact_replacement_is_rejected_before_any_replacement_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture, _evidence = _bundle()
    path = tmp_path / "fixture.json"
    replacement = tmp_path / "replacement.json"
    body = _canonical_bytes(fixture)
    path.write_bytes(body)
    replacement.write_bytes(b'{"secret":"must-not-be-read"}')
    original_open = e2e_module.os.open
    original_read = e2e_module.os.read
    replacement_read = False

    def replace_then_open(target, flags, *args):
        if Path(target) == path:
            replacement.replace(path)
        return original_open(target, flags, *args)

    def guarded_read(descriptor, size):
        nonlocal replacement_read
        replacement_read = True
        return original_read(descriptor, size)

    monkeypatch.setattr(e2e_module.os, "open", replace_then_open)
    monkeypatch.setattr(e2e_module.os, "read", guarded_read)

    with pytest.raises(CanaryEvidenceError) as exc:
        e2e_module._read_bound_artifact(  # noqa: SLF001
            path,
            hashlib.sha256(body).hexdigest(),
            "fixture_artifact_invalid",
        )

    assert exc.value.code == "fixture_artifact_invalid"
    assert replacement_read is False


def test_unit_fixture_cannot_be_relabelled_synthetic():
    fixture, evidence = _bundle()
    synthetic = deepcopy(evidence)
    synthetic["runtime_provenance"]["execution_mode"] = "unit_fixture"
    synthetic["runtime_provenance"]["synthetic"] = True

    with pytest.raises(CanaryEvidenceError) as exc:
        _verify(fixture, synthetic)

    assert exc.value.code == "live_provenance_invalid"
