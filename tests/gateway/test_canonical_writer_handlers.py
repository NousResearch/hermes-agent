import datetime as dt
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.canonical_writer_handlers import (
    MODEL_FORBIDDEN_EVENT_TYPES,
    REQUEST_SCHEMAS,
    SUPPORTED_OPERATIONS,
    CanonicalWriterHandlers,
    CanonicalWriterTypedDispatcher,
    InMemoryCanonicalWriterBackend,
    InMemoryCanonicalWriterStore,
    RuntimeContext,
)
from gateway.canonical_writer_protocol import CanonicalWriterOperation, make_request
from gateway.discord_edge_protocol import (
    DiscordEdgeReceiptOutcome,
    parse_request_for_reconciliation,
    sign_receipt,
    verify_request_capability_for_reconciliation,
)
from gateway.discord_edge_writer_authority import (
    CanonicalWriterDiscordAuthority,
    derive_routeback_edge_idempotency_key,
)
from gateway.canonical_writer_service import DispatchContext, PeerCredentials


NOW = dt.datetime(2026, 7, 12, 8, 0, tzinfo=dt.timezone.utc)
SESSION_HASH = "a" * 64
COMMAND_HASH = "b" * 64
SOURCE_HASH = "c" * 64
ROUTEBACK_CONTENT = "Exact handler route-back content"
CONTENT_HASH = hashlib.sha256(ROUTEBACK_CONTENT.encode("utf-8")).hexdigest()
CAPABILITY_EPOCH_HASH = "e" * 64
DISCORD_GUILD_ID = "100000000000000001"
DISCORD_CHANNEL_ID = "100000000000000002"
DISCORD_MESSAGE_ID = "100000000000000003"
DISCORD_BOT_ID = "100000000000000004"
WRITER_PRIVATE_KEY = Ed25519PrivateKey.generate()
EDGE_PRIVATE_KEY = Ed25519PrivateKey.generate()


def _runtime(
    *,
    session_hash=SESSION_HASH,
    epoch_hash=CAPABILITY_EPOCH_HASH,
    thread_id="requester-thread",
    platform="discord",
):
    return RuntimeContext(
        request_id="request-1",
        platform=platform,
        session_key_sha256=session_hash,
        capability_epoch_sha256=epoch_hash,
        user_id="owner-1",
        chat_id=thread_id,
        thread_id=thread_id,
        message_id="message-1",
        owner_authenticated=True,
    )


def _handlers(store=None):
    backend = InMemoryCanonicalWriterBackend(store, clock=lambda: NOW)
    authority = CanonicalWriterDiscordAuthority(
        capability_private_key=WRITER_PRIVATE_KEY,
        edge_receipt_public_key=EDGE_PRIVATE_KEY.public_key(),
        clock_unix_ms=lambda: int(NOW.timestamp() * 1000),
    )
    return CanonicalWriterHandlers(
        backend,
        discord_edge_authority=authority,
    ), backend


def _routeback_claim_payload(*, case_id="case:1", key="routeback:1"):
    edge_key = derive_routeback_edge_idempotency_key(
        case_id=case_id,
        canonical_idempotency_key=key,
    )
    return {
        "case_id": case_id,
        "target_ref": {
            "target_type": "public_guild_channel",
            "guild_id": DISCORD_GUILD_ID,
            "channel_id": DISCORD_CHANNEL_ID,
        },
        "message_summary": "Send exact result",
        "source_refs": {"thread_id": "requester-thread"},
        "idempotency_key": key,
        "execution_binding": {
            "target_channel_id": DISCORD_CHANNEL_ID,
            "content_sha256": CONTENT_HASH,
        },
        "discord_edge_intent": {
            "operation": "public.message.send",
            "target": {
                "target_type": "public_guild_channel",
                "guild_id": DISCORD_GUILD_ID,
                "channel_id": DISCORD_CHANNEL_ID,
            },
            "payload": {"content": ROUTEBACK_CONTENT},
            "idempotency_key": edge_key,
        },
    }


def _signed_routeback_receipt(edge_request, outcome):
    request = parse_request_for_reconciliation(edge_request)
    capability = verify_request_capability_for_reconciliation(
        request,
        WRITER_PRIVATE_KEY.public_key(),
    )
    if outcome is DiscordEdgeReceiptOutcome.VERIFIED:
        values = {
            "discord_object_id": DISCORD_MESSAGE_ID,
            "bot_user_id": DISCORD_BOT_ID,
            "adapter_accepted": True,
            "readback_verified": True,
            "readback_content_sha256": CONTENT_HASH,
            "blocker_code": None,
        }
    elif outcome is DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED:
        values = {
            "discord_object_id": DISCORD_MESSAGE_ID,
            "bot_user_id": DISCORD_BOT_ID,
            "adapter_accepted": True,
            "readback_verified": False,
            "readback_content_sha256": None,
            "blocker_code": "readback_failed",
        }
    elif outcome is DiscordEdgeReceiptOutcome.DISPATCH_UNCERTAIN:
        values = {
            "discord_object_id": None,
            "bot_user_id": None,
            "adapter_accepted": None,
            "readback_verified": False,
            "readback_content_sha256": None,
            "blocker_code": "dispatch_outcome_uncertain",
        }
    else:
        values = {
            "discord_object_id": None,
            "bot_user_id": None,
            "adapter_accepted": False,
            "readback_verified": False,
            "readback_content_sha256": None,
            "blocker_code": "blocked_before_dispatch",
        }
    return sign_receipt(
        EDGE_PRIVATE_KEY,
        request,
        capability,
        outcome=outcome,
        occurred_at_unix_ms=int(NOW.timestamp() * 1000),
        **values,
    ).to_message()


def _routeback_terminal_payload(claim, edge_request, edge_receipt):
    return {
        key: claim[key]
        for key in (
            "case_id",
            "target_ref",
            "message_summary",
            "source_refs",
            "idempotency_key",
            "execution_binding",
        )
    } | {
        "discord_edge_request": edge_request,
        "discord_edge_receipt": edge_receipt,
    }


def _append(handlers, runtime, *, event_type="case.note", case_id="case:1", body=None, key=None):
    payload = {
        "event_type": event_type,
        "case_id": case_id,
        "summary": "mechanical test event",
        "source_refs": {"thread_id": runtime.thread_id or "thread-1"},
        "payload": body or {},
    }
    if key is not None:
        payload["idempotency_key"] = key
    return handlers.dispatch(
        CanonicalWriterOperation.EVENT_APPEND_MODEL.value,
        payload,
        runtime=runtime,
    )


def _seed_active_plan(handlers, runtime, *, case_id="case:1", plan_id="plan:1", key="plan:1:r1"):
    result = _transition_plan(
        handlers,
        runtime,
        case_id=case_id,
        plan=_valid_plan(plan_id=plan_id),
        key=key,
    )
    assert result["ok"] is True


def _transition_plan(handlers, runtime, *, case_id="case:1", plan, key):
    return handlers.dispatch(
        CanonicalWriterOperation.PLAN_TRANSITION.value,
        {
            "case_id": case_id,
            "summary": "typed plan transition",
            "source_refs": {"thread_id": runtime.thread_id or "thread-1"},
            "payload": {"plan": plan},
            "idempotency_key": key,
        },
        runtime=runtime,
    )


def _valid_plan(*, plan_id="plan:1", supersedes_plan_id=""):
    plan = {
        "plan_id": plan_id,
        "revision": 1,
        "objective": "Exercise the typed writer boundary",
        "state": "active",
        "success_criteria": [{"id": "verified", "content": "Tests pass"}],
        "steps": [{
            "id": "execute",
            "content": "Run the bounded test",
            "status": "in_progress",
            "depends_on": [],
        }],
        "current_step_id": "execute",
        "resume_cursor": {"next_step_id": "execute", "summary": "Continue test"},
    }
    if supersedes_plan_id:
        plan["supersedes_plan_id"] = supersedes_plan_id
        plan["supersedes_plan_revision"] = 1
    return plan


def _grant_payload(*, approval_id="approval:1", source_hash=SOURCE_HASH, max_uses=1):
    return {
        "approval_id": approval_id,
        "case_id": "case:1",
        "plan_id": "plan:1",
        "plan_revision": 1,
        "approval_source_sha256": source_hash,
        "command_hashes": [COMMAND_HASH],
        "expires_at": (NOW + dt.timedelta(hours=1)).isoformat(),
        "max_uses": max_uses,
    }


def _consume_payload(*, key="consume:1"):
    return {"command_sha256": COMMAND_HASH, "idempotency_key": key}


def _initiating_mutation_cases():
    return (
        (
            CanonicalWriterOperation.EVENT_APPEND_MODEL,
            {
                "event_type": "case.note",
                "case_id": "case:retired",
                "summary": "stale model event",
                "source_refs": {"thread_id": "requester-thread"},
                "payload": {},
                "idempotency_key": "retired:event",
            },
        ),
        (
            CanonicalWriterOperation.PLAN_TRANSITION,
            {
                "case_id": "case:retired",
                "summary": "stale plan transition",
                "source_refs": {"thread_id": "requester-thread"},
                "payload": {"plan": _valid_plan(plan_id="plan:retired")},
                "idempotency_key": "retired:plan",
            },
        ),
        (
            CanonicalWriterOperation.VERIFICATION_APPEND,
            {
                "case_id": "case:retired",
                "summary": "stale verification",
                "source_refs": {"thread_id": "requester-thread"},
                "payload": {
                    "verification": {
                        "verification_id": "verification:retired",
                        "plan_id": "plan:retired",
                        "plan_revision": 1,
                        "summary": "bounded check passed",
                        "outcome": "passed",
                        "criterion_ids": ["verified"],
                        "receipt": {"kind": "test", "ref": "pytest:retired"},
                    }
                },
                "idempotency_key": "retired:verification",
            },
        ),
        (
            CanonicalWriterOperation.ROUTEBACK_CLAIM,
            _routeback_claim_payload(
                case_id="case:retired",
                key="retired:claim",
            ),
        ),
        (
            CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED,
            {
                "preclaim": True,
                "case_id": "case:retired",
                "target_ref": {
                    "id": "blocked-target:retired",
                    "target_kind": "unresolved_public_target",
                },
                "message_summary": "stale preclaim block",
                "source_refs": {"thread_id": "requester-thread"},
                "blocker_reason": "public target remained unresolved",
                "idempotency_key": "retired:preclaim",
            },
        ),
    )


def test_operations_and_schemas_exactly_follow_central_protocol_without_sql():
    protocol_ops = {item.value for item in CanonicalWriterOperation}

    assert SUPPORTED_OPERATIONS == protocol_ops
    assert set(REQUEST_SCHEMAS) == protocol_ops
    assert all("sql" not in operation for operation in SUPPORTED_OPERATIONS)


def test_ping_and_read_only_do_not_require_platform_or_session_binding():
    handlers, _ = _handlers()
    runtime = RuntimeContext(request_id="read-only-request")

    response = handlers.dispatch(CanonicalWriterOperation.PING.value, {}, runtime=runtime)

    assert response["ok"] is True
    assert response["result"]["status"] == "ok"


@pytest.mark.parametrize("operation,payload", _initiating_mutation_cases())
@pytest.mark.parametrize("missing_binding", ["session", "epoch"])
def test_every_initiating_mutation_requires_exact_session_epoch_binding(
    operation,
    payload,
    missing_binding,
):
    handlers, backend = _handlers()
    runtime = RuntimeContext(
        request_id=f"missing-{missing_binding}",
        platform="discord",
        session_key_sha256="" if missing_binding == "session" else SESSION_HASH,
        capability_epoch_sha256=(
            "" if missing_binding == "epoch" else CAPABILITY_EPOCH_HASH
        ),
        user_id="owner-1",
        chat_id="requester-thread",
        thread_id="requester-thread",
        owner_authenticated=True,
    )

    response = handlers.dispatch(operation.value, payload, runtime=runtime)

    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_runtime"
    assert backend.store.events == []


@pytest.mark.parametrize("operation,payload", _initiating_mutation_cases())
def test_retired_session_epoch_blocks_every_initiating_mutation(
    operation,
    payload,
):
    handlers, backend = _handlers()
    runtime = _runtime()
    revoked = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_REVOKE_SESSION.value,
        {"reason": "explicit session rotation"},
        runtime=runtime,
    )
    before = backend.store.snapshot()

    response = handlers.dispatch(operation.value, payload, runtime=runtime)

    assert revoked["ok"] is True
    assert response["ok"] is False
    assert response["error"]["code"] == "session_epoch_retired"
    assert backend.store.snapshot() == before


def test_session_retirement_wins_shared_lock_before_waiting_stale_mutation():
    handlers, backend = _handlers()
    runtime = _runtime()
    reached_backend = threading.Event()
    original_append = backend.event_append

    def waiting_append(request, observed_runtime):
        reached_backend.set()
        return original_append(request, observed_runtime)

    backend.event_append = waiting_append
    with ThreadPoolExecutor(max_workers=1) as pool:
        with backend.store.lock:
            stale = pool.submit(
                _append,
                handlers,
                runtime,
                case_id="case:retirement-race",
                key="retirement-race:event",
            )
            assert reached_backend.wait(timeout=2)
            revoked = handlers.dispatch(
                CanonicalWriterOperation.CAPABILITY_REVOKE_SESSION.value,
                {"reason": "rotation won serialization"},
                runtime=runtime,
            )
        stale_result = stale.result(timeout=2)

    assert revoked["ok"] is True
    assert stale_result["ok"] is False
    assert stale_result["error"]["code"] == "session_epoch_retired"
    assert len(backend.store.events) == 1
    assert backend.store.events[0]["event_type"] == (
        "approval.capability.session_revoked"
    )
    assert revoked["result"]["authority_active"] is False
    assert revoked["result"]["inserted"] is True
    assert revoked["result"]["deduped"] is False


@pytest.mark.parametrize(
    "finalize_operation,edge_outcome,event_type",
    [
        (
            CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT,
            DiscordEdgeReceiptOutcome.VERIFIED,
            "route_back.sent",
        ),
        (
            CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED,
            DiscordEdgeReceiptOutcome.DISPATCH_UNCERTAIN,
            "route_back.blocked",
        ),
    ],
)
def test_exact_original_claimant_can_record_terminal_truth_after_epoch_retirement(
    finalize_operation,
    edge_outcome,
    event_type,
):
    handlers, backend = _handlers()
    runtime = _runtime()
    claim = _routeback_claim_payload(
        case_id="case:terminal-truth",
        key=f"terminal-truth:{event_type}",
    )
    claimed = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        claim,
        runtime=runtime,
    )
    edge_request = claimed["result"]["discord_edge_request"]
    edge_receipt = _signed_routeback_receipt(edge_request, edge_outcome)
    revoked = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_REVOKE_SESSION.value,
        {"reason": "rotate after external send was claimed"},
        runtime=runtime,
    )

    finalized = handlers.dispatch(
        finalize_operation.value,
        _routeback_terminal_payload(claim, edge_request, edge_receipt),
        runtime=runtime,
    )

    assert claimed["ok"] is True
    assert revoked["ok"] is True
    assert finalized["ok"] is True
    assert finalized["result"]["outcome"] == event_type.rsplit(".", 1)[-1]
    assert backend.store.events[-1]["event_type"] == event_type


def test_runtime_is_separate_and_caller_payload_cannot_override_it():
    handlers, backend = _handlers()
    runtime = _runtime()
    rejected = handlers.dispatch(
        CanonicalWriterOperation.EVENT_APPEND_MODEL.value,
        {
            "event_type": "case.note",
            "case_id": "case:1",
            "summary": "bad override",
            "source_refs": {},
            "runtime_context": {"session_key_sha256": "f" * 64},
        },
        runtime=runtime,
    )
    accepted = _append(
        handlers,
        runtime,
        body={"observed_session": {"session_key_sha256": "f" * 64}},
    )

    assert rejected["error"]["code"] == "runtime_override_forbidden"
    assert accepted["ok"] is True
    assert backend.store.events[-1]["runtime"]["session_key_sha256"] == SESSION_HASH
    assert backend.store.events[-1]["body"]["payload"]["observed_session"][
        "session_key_sha256"
    ] == "f" * 64


@pytest.mark.parametrize("event_type", sorted(MODEL_FORBIDDEN_EVENT_TYPES))
def test_model_append_cannot_mint_privileged_process_receipts(event_type):
    handlers, backend = _handlers()

    response = _append(handlers, _runtime(), event_type=event_type)

    assert response["ok"] is False
    assert response["error"]["code"] == "privileged_event_forbidden"
    assert backend.store.events == []


def test_typed_preclaim_routeback_blocked_path_never_forges_authorization():
    handlers, backend = _handlers()

    response = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED.value,
        {
            "preclaim": True,
            "case_id": "case:1",
            "target_ref": {
                "id": "blocked-target:abc",
                "target_kind": "forbidden_or_unresolved_target",
            },
            "message_summary": "could not resolve public target",
            "source_refs": {"thread_id": "requester-thread"},
            "blocker_reason": "target unresolved before claim",
            "idempotency_key": "preclaim:blocked:1",
        },
        runtime=_runtime(),
    )

    assert "route_back.blocked" in MODEL_FORBIDDEN_EVENT_TYPES
    assert response["ok"] is True
    assert response["result"]["preclaim"] is True
    assert response["result"]["preclaim_block_id"].startswith("routeblock:")
    assert "authorization_id" not in response["result"]
    assert backend.store.routeback_authorizations == {}
    assert len(backend.store.routeback_lifecycle_terminals) == 1
    assert backend.store.events[-1]["event_type"] == "route_back.blocked"


def test_preclaim_terminal_blocks_later_claim_across_session_rotation():
    handlers, backend = _handlers()
    claim = _routeback_claim_payload(key="preclaim:global-lifecycle:1")
    payload = {
        "preclaim": True,
        "case_id": claim["case_id"],
        "target_ref": claim["target_ref"],
        "message_summary": "blocked before claim",
        "source_refs": claim["source_refs"],
        "blocker_reason": "public target unavailable",
        "idempotency_key": claim["idempotency_key"],
    }
    assert handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED.value,
        payload,
        runtime=_runtime(),
    )["ok"]

    retry = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        {**claim, "message_summary": payload["message_summary"]},
        runtime=_runtime(session_hash="f" * 64),
    )

    assert retry["result"]["terminal_event_type"] == "route_back.blocked"
    assert "authorization_id" not in retry["result"]
    assert backend.store.routeback_authorizations == {}
    assert [event["event_type"] for event in backend.store.events] == [
        "route_back.blocked"
    ]


@pytest.mark.parametrize(
    "runtime_kwargs",
    [
        {"session_hash": "f" * 64},
        {"epoch_hash": "d" * 64},
        {"thread_id": "another-requester-thread"},
        {"platform": "telegram"},
    ],
)
def test_nonterminal_claim_dedupe_requires_exact_runtime_scope(runtime_kwargs):
    handlers, backend = _handlers()
    claim = _routeback_claim_payload(key="routeback:cross-session:1")
    first = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        claim,
        runtime=_runtime(),
    )
    retry = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        claim,
        runtime=_runtime(**runtime_kwargs),
    )

    assert first["ok"] is True
    assert retry["error"]["code"] == "scope_mismatch"
    assert len(backend.store.routeback_authorizations) == 1
    assert [event["event_type"] for event in backend.store.events] == [
        "route_back.intent.created"
    ]


def test_nonterminal_deduped_claim_mints_fresh_short_lived_edge_request():
    handlers, backend = _handlers()
    runtime = _runtime()
    claim = _routeback_claim_payload(key="routeback:fresh-edge-recovery:1")

    first = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        claim,
        runtime=runtime,
    )["result"]
    retry = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        claim,
        runtime=runtime,
    )["result"]

    first_request = parse_request_for_reconciliation(first["discord_edge_request"])
    retry_request = parse_request_for_reconciliation(retry["discord_edge_request"])
    assert first["inserted"] is True
    assert retry["inserted"] is False
    assert retry["deduped"] is True
    assert first_request.request_id != retry_request.request_id
    assert first_request.intent == retry_request.intent
    assert first_request.intent.idempotency_key == derive_routeback_edge_idempotency_key(
        case_id=claim["case_id"],
        canonical_idempotency_key=claim["idempotency_key"],
    )
    assert len(backend.store.routeback_authorizations) == 1


def test_edge_idempotency_is_scoped_by_case_not_raw_caller_key():
    handlers, _backend = _handlers()
    canonical_key = "routeback:case-scoped-edge:1"
    first_claim = _routeback_claim_payload(
        case_id="case:edge-scope-a",
        key=canonical_key,
    )
    second_claim = _routeback_claim_payload(
        case_id="case:edge-scope-b",
        key=canonical_key,
    )

    first = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        first_claim,
        runtime=_runtime(),
    )["result"]
    second = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        second_claim,
        runtime=_runtime(),
    )["result"]

    first_request = parse_request_for_reconciliation(first["discord_edge_request"])
    second_request = parse_request_for_reconciliation(second["discord_edge_request"])
    assert first_request.intent.idempotency_key != second_request.intent.idempotency_key
    assert first_request.intent.idempotency_key.startswith("canonical-routeback:")
    assert second_request.intent.idempotency_key.startswith("canonical-routeback:")


def test_model_append_optional_idempotency_is_deterministic():
    handlers, backend = _handlers()
    runtime = _runtime()

    first = _append(handlers, runtime)
    second = _append(handlers, runtime)

    assert first["result"]["inserted"] is True
    assert second["result"]["deduped"] is True
    assert len(backend.store.events) == 1


def test_service_side_append_reuses_authoritative_mechanical_validators():
    handlers, backend = _handlers()

    response = _append(handlers, _runtime(), event_type="not.allowed", body={})

    assert response["error"]["code"] == "invalid_event"
    assert "event_type_not_allowed" in response["error"]["message"]
    assert backend.store.events == []


def test_service_side_append_blocks_nested_secret_before_backend():
    handlers, backend = _handlers()

    response = _append(
        handlers,
        _runtime(),
        body={"nested": {"credentials": {"token": "secret-value"}}},
    )

    assert response["error"]["code"] == "invalid_event"
    assert "secret_like_content_blocked" in response["error"]["message"]
    assert backend.store.events == []


@pytest.mark.parametrize(
    "operation",
    [
        CanonicalWriterOperation.PLAN_TRANSITION,
        CanonicalWriterOperation.VERIFICATION_APPEND,
    ],
)
def test_typed_plan_and_verification_ops_validate_inside_service(operation):
    handlers, backend = _handlers()
    response = handlers.dispatch(
        operation.value,
        {
            "case_id": "case:1",
            "summary": "malformed typed event",
            "source_refs": {},
            "payload": {},
        },
        runtime=_runtime(),
    )

    assert response["error"]["code"] == "invalid_event"
    assert backend.store.events == []


def test_routeback_claim_and_typed_finalize_are_atomic_and_replay_safe():
    handlers, backend = _handlers()
    runtime = _runtime()
    claim = _routeback_claim_payload()
    claimed = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        claim,
        runtime=runtime,
    )
    edge_request = claimed["result"]["discord_edge_request"]
    edge_receipt = _signed_routeback_receipt(
        edge_request,
        DiscordEdgeReceiptOutcome.VERIFIED,
    )
    finalized = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT.value,
        _routeback_terminal_payload(claim, edge_request, edge_receipt),
        runtime=runtime,
    )
    replay = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        claim,
        runtime=runtime,
    )
    context = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CONTEXT.value,
        {"thread_id": DISCORD_CHANNEL_ID},
        runtime=RuntimeContext(
            request_id="read-routeback",
            platform="discord",
            thread_id=DISCORD_CHANNEL_ID,
        ),
    )

    assert claimed["result"]["inserted"] is True
    assert claimed["result"]["claimed_at"]
    assert finalized["ok"] is True
    assert replay["result"]["terminal_event_type"] == "route_back.sent"
    assert replay["result"]["claimed_at"] == claimed["result"]["claimed_at"]
    assert backend.store.events[-1]["event_type"] == "route_back.sent"
    assert context["result"]["cases"] == [{
        "case_id": "case:1",
        "source_thread_id": "requester-thread",
    }]


def test_routeback_sent_rejects_unverified_process_receipt():
    handlers, _backend = _handlers()
    runtime = _runtime()
    claim = _routeback_claim_payload(key="routeback:unverified")
    claimed = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        claim,
        runtime=runtime,
    )

    response = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT.value,
        {
            **_routeback_terminal_payload(
                claim,
                claimed["result"]["discord_edge_request"],
                _signed_routeback_receipt(
                    claimed["result"]["discord_edge_request"],
                    DiscordEdgeReceiptOutcome.VERIFIED,
                ),
            ),
            "receipt": {
                "platform": "discord",
                "adapter_receipt": True,
                "receipt_readback_verified": False,
            },
        },
        runtime=runtime,
    )

    assert response["error"]["code"] == "invalid_request"


def test_routeback_accepted_unverified_remains_nonterminal():
    handlers, backend = _handlers()
    runtime = _runtime()
    claim = _routeback_claim_payload(key="routeback:partial-receipt")
    claimed = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        claim,
        runtime=runtime,
    )["result"]
    edge_request = claimed["discord_edge_request"]
    edge_receipt = _signed_routeback_receipt(
        edge_request,
        DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED,
    )
    terminal = _routeback_terminal_payload(claim, edge_request, edge_receipt)

    first = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED.value,
        terminal,
        runtime=runtime,
    )
    replay = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED.value,
        terminal,
        runtime=runtime,
    )

    assert first["error"]["code"] == "discord_edge_outcome_pending"
    assert replay["error"]["code"] == "discord_edge_outcome_pending"
    authorization = backend.store.routeback_authorizations[
        claimed["authorization_id"]
    ]
    assert authorization["terminal"] is None
    assert all(event["event_type"] != "route_back.blocked" for event in backend.store.events)


def test_routeback_blocked_rejects_verified_edge_truth():
    handlers, _backend = _handlers()
    runtime = _runtime()
    claim = _routeback_claim_payload(key="routeback:verified-blocked")
    claimed = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        claim,
        runtime=runtime,
    )["result"]
    edge_request = claimed["discord_edge_request"]
    response = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED.value,
        _routeback_terminal_payload(
            claim,
            edge_request,
            _signed_routeback_receipt(
                edge_request,
                DiscordEdgeReceiptOutcome.VERIFIED,
            ),
        ),
        runtime=runtime,
    )

    assert response["error"]["code"] == "invalid_discord_edge_outcome"


@pytest.mark.parametrize(
    "change",
    [
        {"field": "channel_id", "value": "100000000000000099"},
        {"field": "content_sha256", "value": "f" * 64},
        {"field": "idempotency_key", "value": "routeback:other"},
    ],
)
def test_routeback_blocked_rejects_unbound_signed_receipt(change):
    handlers, _backend = _handlers()
    runtime = _runtime()
    claim = _routeback_claim_payload(key="routeback:bad-signed-receipt")
    claimed = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        claim,
        runtime=runtime,
    )["result"]
    edge_request = claimed["discord_edge_request"]
    edge_receipt = _signed_routeback_receipt(
        edge_request,
        DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED,
    )
    tampered_payload = dict(edge_receipt["payload"])
    tampered_payload[change["field"]] = change["value"]
    edge_receipt = {**edge_receipt, "payload": tampered_payload}

    response = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED.value,
        _routeback_terminal_payload(claim, edge_request, edge_receipt),
        runtime=runtime,
    )

    assert response["error"]["code"] == "invalid_discord_edge_evidence"


@pytest.mark.parametrize(
    "forbidden_target",
    [
        {"channel_id": "public", "recipient_id": "user-1"},
        {"channel_id": "public", "nested": {"dm_channel_id": "dm-1"}},
        {"channel_id": "public", "target_type": "private_dm"},
        {"channel_id": "public", "lane": "owner_dm"},
        {"channel_id": "public", "channel_type": "group_dm"},
        {"channel_id": "public", "channel_type": "group"},
        {"channel_id": "public", "channel_type": "private"},
        {"channel_id": "public", "channel_type": "private_channel"},
        {"channel_id": "public", "channel_type": "private_thread"},
        {"channel_id": "public", "target_kind": "private"},
        {"channel_id": "public", "target_kind": "private_thread"},
        {"channel_id": "public", "target_kind": "group"},
    ],
)
def test_routeback_claim_recursively_blocks_all_dm_reference_shapes(forbidden_target):
    handlers, _ = _handlers()
    claim = _routeback_claim_payload(key="routeback:dm")
    claim["target_ref"] = {
        **forbidden_target,
        "channel_id": DISCORD_CHANNEL_ID,
    }
    response = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        claim,
        runtime=_runtime(),
    )

    assert response["error"]["code"] == "dm_forbidden"


def test_idempotency_key_uses_the_same_utf8_byte_bound_as_transport():
    handlers, _ = _handlers()
    payload = _routeback_claim_payload()
    payload["idempotency_key"] = "é" * 129
    payload["discord_edge_intent"]["idempotency_key"] = "é" * 129
    response = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        payload,
        runtime=_runtime(),
    )

    assert response["error"]["code"] == "invalid_request"
    assert "protocol byte bound" in response["error"]["message"]


def test_capability_grant_consume_is_exact_and_process_receipts_are_typed():
    handlers, backend = _handlers()
    runtime = _runtime()
    _seed_active_plan(handlers, runtime)

    granted = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(max_uses=2),
        runtime=runtime,
    )
    consumed = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME.value,
        _consume_payload(),
        runtime=runtime,
    )

    assert granted["ok"] is True
    assert granted["result"]["authority_active"] is True
    assert consumed["result"]["authorized"] is True
    assert consumed["result"]["remaining_uses"] == 1
    assert consumed["result"]["command_sha256"] == COMMAND_HASH
    assert [event["event_type"] for event in backend.store.events[-2:]] == [
        "approval.capability.recorded",
        "capability.check.recorded",
    ]


def test_capability_consume_is_atomic_under_concurrency():
    handlers, _ = _handlers()
    runtime = _runtime()
    _seed_active_plan(handlers, runtime)
    assert handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(max_uses=1),
        runtime=runtime,
    )["ok"]

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(
            lambda index: handlers.dispatch(
                CanonicalWriterOperation.CAPABILITY_CONSUME.value,
                _consume_payload(key=f"concurrent:{index}"),
                runtime=runtime,
            ),
            range(24),
        ))

    assert sum(result["ok"] is True for result in results) == 1
    assert all(
        result["ok"] or result["error"]["code"] == "capability_exhausted"
        for result in results
    )


def test_capability_consume_idempotency_retry_never_double_decrements():
    handlers, backend = _handlers()
    runtime = _runtime()
    _seed_active_plan(handlers, runtime)
    handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(max_uses=2),
        runtime=runtime,
    )
    payload = _consume_payload(key="stable-consume-attempt")

    first = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME.value,
        payload,
        runtime=runtime,
    )
    retry = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME.value,
        payload,
        runtime=runtime,
    )

    assert first["result"]["remaining_uses"] == 1
    assert retry["result"]["remaining_uses"] == 1
    assert retry["result"]["deduped"] is True
    assert backend.store.capabilities["approval:1"]["remaining_uses"][COMMAND_HASH] == 1


def test_command_only_consume_fails_closed_when_durable_lookup_is_ambiguous():
    handlers, _ = _handlers()
    runtime = _runtime()
    _seed_active_plan(handlers, runtime)
    handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(approval_id="approval:1", source_hash="1" * 64),
        runtime=runtime,
    )
    handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(approval_id="approval:2", source_hash="2" * 64),
        runtime=runtime,
    )

    response = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME.value,
        {"command_sha256": COMMAND_HASH, "idempotency_key": "ambiguous:1"},
        runtime=runtime,
    )

    assert response["error"]["code"] == "capability_ambiguous"


def test_approval_source_replay_cannot_refresh_or_mint_second_capability():
    handlers, backend = _handlers()
    runtime = _runtime()
    _seed_active_plan(handlers, runtime)
    payload = _grant_payload(max_uses=2)
    first = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        payload,
        runtime=runtime,
    )
    deduped = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        payload,
        runtime=runtime,
    )
    changed_expiry = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        {
            **payload,
            "expires_at": (NOW + dt.timedelta(hours=2)).isoformat(),
        },
        runtime=runtime,
    )
    replay = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        {**payload, "approval_id": "approval:2", "max_uses": 99},
        runtime=runtime,
    )

    assert first["ok"] is True
    assert deduped["result"]["deduped"] is True
    assert changed_expiry["error"]["code"] == "approval_source_replay"
    assert replay["error"]["code"] == "approval_source_replay"
    assert len(backend.store.capabilities) == 1
    assert backend.store.capabilities["approval:1"]["remaining_uses"][COMMAND_HASH] == 2


def test_capability_ledger_survives_restart_snapshot_without_counter_reset():
    handlers, backend = _handlers()
    runtime = _runtime()
    _seed_active_plan(handlers, runtime)
    handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(max_uses=2),
        runtime=runtime,
    )
    handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME.value,
        _consume_payload(key="consume:after-restart"),
        runtime=runtime,
    )
    restarted_store = InMemoryCanonicalWriterStore.from_snapshot(backend.store.snapshot())
    restarted, _ = _handlers(restarted_store)

    second = restarted.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME.value,
        _consume_payload(key="consume:exhausted"),
        runtime=runtime,
    )
    exhausted = restarted.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME.value,
        _consume_payload(),
        runtime=runtime,
    )

    assert second["result"]["remaining_uses"] == 0
    assert exhausted["error"]["code"] == "capability_exhausted"


def test_capability_expiry_is_enforced_by_atomic_durable_lookup():
    current = [NOW]
    backend = InMemoryCanonicalWriterBackend(clock=lambda: current[0])
    handlers = CanonicalWriterHandlers(backend)
    runtime = _runtime()
    _seed_active_plan(handlers, runtime)
    handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(),
        runtime=runtime,
    )
    current[0] = NOW + dt.timedelta(hours=2)

    response = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME.value,
        _consume_payload(),
        runtime=runtime,
    )

    assert response["error"]["code"] == "capability_expired"
    assert backend.store.capabilities["approval:1"]["state"] == "expired"


def test_terminal_plan_event_revokes_durable_capability():
    handlers, backend = _handlers()
    runtime = _runtime()
    _seed_active_plan(handlers, runtime)
    handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(),
        runtime=runtime,
    )
    completed = _valid_plan()
    completed.update({
        "revision": 2,
        "state": "completed",
        "steps": [{
            "id": "execute",
            "content": "Run the bounded test",
            "status": "completed",
            "depends_on": [],
        }],
        "current_step_id": None,
        "resume_cursor": {"next_step_id": None, "summary": "Complete"},
        "verification_event_ids": ["11111111-1111-4111-8111-111111111111"],
    })

    terminal = _transition_plan(
        handlers,
        runtime,
        plan=completed,
        key="plan:1:r2:complete",
    )

    assert terminal["ok"] is True
    assert backend.store.capabilities["approval:1"]["state"] == "revoked"


def test_active_plan_recheck_and_supersession_revoke_old_capability():
    handlers, backend = _handlers()
    runtime = _runtime()
    _seed_active_plan(handlers, runtime)
    handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(),
        runtime=runtime,
    )
    _transition_plan(
        handlers,
        runtime,
        plan=_valid_plan(
            plan_id="plan:2",
            supersedes_plan_id="plan:1",
        ),
        key="plan:2:r1",
    )

    consumed = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME.value,
        _consume_payload(),
        runtime=runtime,
    )

    assert consumed["error"]["code"] == "capability_missing"
    assert backend.store.capabilities["approval:1"]["state"] == "revoked"


def test_same_plan_later_revision_invalidates_exact_revision_capability():
    handlers, backend = _handlers()
    runtime = _runtime()
    _seed_active_plan(handlers, runtime)
    assert handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(),
        runtime=runtime,
    )["ok"]
    revision_two = _valid_plan()
    revision_two["revision"] = 2
    assert _transition_plan(
        handlers,
        runtime,
        plan=revision_two,
        key="plan:1:r2:active",
    )["ok"]

    consumed = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME.value,
        _consume_payload(key="stale-revision"),
        runtime=runtime,
    )

    assert consumed["error"]["code"] == "capability_missing"
    assert backend.store.capabilities["approval:1"]["state"] == "revoked"


def test_revoke_tombstone_blocks_later_grants_in_same_routing_epoch():
    handlers, backend = _handlers()
    runtime = _runtime()
    _seed_active_plan(handlers, runtime)
    assert handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(approval_id="approval:1", source_hash="1" * 64),
        runtime=runtime,
    )["ok"]
    first = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_REVOKE.value,
        {"plan_id": "plan:1", "reason": "owner stopped plan"},
        runtime=runtime,
    )
    later_grant = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(approval_id="approval:2", source_hash="2" * 64),
        runtime=runtime,
    )
    second = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_REVOKE.value,
        {"plan_id": "plan:1", "reason": "owner stopped plan"},
        runtime=runtime,
    )

    receipts = [
        event for event in backend.store.events
        if event["event_type"] == "approval.capability.revoked"
    ]
    assert first["result"]["revoked"] == 1
    assert later_grant["error"]["code"] == "capability_scope_revoked"
    assert second["result"]["revoked"] == 0
    assert second["result"]["scope_revoked"] is True
    assert len(receipts) == 1


def test_session_revoke_and_projection_read_are_bounded_typed_ops():
    handlers, _ = _handlers()
    runtime = _runtime()
    _seed_active_plan(handlers, runtime)
    handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(),
        runtime=runtime,
    )
    revoked = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_REVOKE_SESSION.value,
        {"reason": "gateway restart"},
        runtime=runtime,
    )
    projected = handlers.dispatch(
        CanonicalWriterOperation.PROJECTION_READ_EVENTS.value,
        {"case_id": "case:1", "limit": 10},
        runtime=RuntimeContext(request_id="projector-read"),
    )

    assert revoked["result"]["revoked"] == 1
    assert revoked["result"] == {
        "success": True,
        "session_key_sha256": SESSION_HASH,
        "capability_epoch_sha256": CAPABILITY_EPOCH_HASH,
        "scope_type": "session",
        "scope_revoked": True,
        "authority_active": False,
        "revocation_event_id": revoked["result"]["revocation_event_id"],
        "inserted": True,
        "deduped": False,
        "revoked": 1,
        "canary_scopes_revoked": 0,
    }
    assert projected["ok"] is True
    assert 1 <= len(projected["result"]["events"]) <= 10


def test_plan_revoke_uses_only_plan_id_and_runtime_session_binding():
    handlers, backend = _handlers()
    runtime = _runtime()
    _seed_active_plan(handlers, runtime)
    handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_GRANT.value,
        _grant_payload(),
        runtime=runtime,
    )

    revoked = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_REVOKE.value,
        {"plan_id": "plan:1", "reason": "plan stopped"},
        runtime=runtime,
    )

    assert revoked["result"] == {
        "success": True,
        "capability_epoch_sha256": CAPABILITY_EPOCH_HASH,
        "scope_type": "plan",
        "scope_revoked": True,
        "revoked": 1,
    }
    assert backend.store.capabilities["approval:1"]["state"] == "revoked"


def test_lease_shadow_schema_matches_gateway_audit_proxy_exactly():
    handlers, backend = _handlers()
    payload = {
        "intent_event_id": "11111111-1111-4111-8111-111111111111",
        "intent_kind": "discord_send",
        "case": {"case_id": "case:gateway-audit:abc", "title": "audit"},
        "runtime_lease_enforcement": {
            "config_path": "canonical_brain.runtime_lease",
            "blocking_effective": True,
        },
        "enforcement_enabled": True,
        "send_path_blocking_enabled": True,
        "audit_runtime_id": "cloud-muncho",
        "source_platform": "discord",
        "session_key_ref": "urn:hermes:session:abc",
    }

    response = handlers.dispatch(
        CanonicalWriterOperation.LEASE_SHADOW_RECORD.value,
        payload,
        runtime=_runtime(),
    )

    assert response["ok"] is True
    assert backend.store.events[-1]["event_type"] == "lease.shadow.recorded"
    assert backend.store.events[-1]["body"]["lease_shadow"]["intent_event_id"] == payload[
        "intent_event_id"
    ]


def test_protocol_to_service_adapter_uses_runtime_without_peer_identity():
    handlers, backend = _handlers()
    adapter = CanonicalWriterTypedDispatcher(handlers)
    request = make_request(
        CanonicalWriterOperation.EVENT_APPEND_MODEL,
        {
            "event_type": "case.note",
            "case_id": "case:wire",
            "summary": "wire e2e",
            "source_refs": {"thread_id": "thread-wire"},
            "payload": {},
        },
        runtime={
            "platform": "discord",
            "session_key_sha256": SESSION_HASH,
            "capability_epoch_sha256": CAPABILITY_EPOCH_HASH,
            "thread_id": "thread-wire",
        },
        sequence=1,
        timeout_seconds=5,
        idempotency_key="wire:e2e",
    )
    context = DispatchContext(
        request_id=request.request_id,
        sequence=request.sequence,
        deadline_unix_ms=request.deadline_unix_ms,
        idempotency_key=request.idempotency_key,
        peer=PeerCredentials(pid=999, uid=1000, gid=1000),
        runtime={**request.runtime, "peer": {"pid": 999, "uid": 1000, "gid": 1000}},
    )

    result = adapter.dispatch(request.operation, request.payload, context)

    assert result.status == "inserted"
    assert backend.store.events[-1]["runtime"] == {
        "platform": "discord",
        "session_key_sha256": SESSION_HASH,
        "capability_epoch_sha256": CAPABILITY_EPOCH_HASH,
        "thread_id": "thread-wire",
    }
    assert "peer" not in backend.store.events[-1]["runtime"]


def test_service_adapter_blocks_handler_errors_without_exception_details():
    handlers, _ = _handlers()
    adapter = CanonicalWriterTypedDispatcher(handlers)
    request = make_request(
        CanonicalWriterOperation.EVENT_APPEND_MODEL,
        {
            "event_type": "route_back.sent",
            "case_id": "case:wire",
            "summary": "forbidden",
            "source_refs": {},
        },
        runtime={
            "session_key_sha256": SESSION_HASH,
            "capability_epoch_sha256": CAPABILITY_EPOCH_HASH,
        },
        sequence=1,
        timeout_seconds=5,
    )
    context = DispatchContext(
        request_id=request.request_id,
        sequence=request.sequence,
        deadline_unix_ms=request.deadline_unix_ms,
        idempotency_key=None,
        peer=PeerCredentials(pid=999, uid=1000, gid=1000),
        runtime={
            "session_key_sha256": SESSION_HASH,
            "capability_epoch_sha256": CAPABILITY_EPOCH_HASH,
        },
    )

    result = adapter.dispatch(request.operation, request.payload, context)

    assert result.status == "blocked"
    assert result.result["success"] is False
    assert result.result["error_code"] == "privileged_event_forbidden"


def test_service_adapter_ignores_wire_claimed_internal_and_owner_authority():
    observed = []

    class _Backend(InMemoryCanonicalWriterBackend):
        def ping(self, runtime):
            observed.append(runtime)
            return {"status": "ok"}

    handlers = CanonicalWriterHandlers(_Backend(clock=lambda: NOW))
    adapter = CanonicalWriterTypedDispatcher(
        handlers,
        owner_user_ids=frozenset({"real-owner"}),
    )
    context = DispatchContext(
        request_id="wire-authority-forgery",
        sequence=1,
        deadline_unix_ms=9999999999999,
        idempotency_key=None,
        peer=PeerCredentials(pid=999, uid=1000, gid=1000),
        runtime={
            "platform": "discord",
            "user_id": "attacker",
            "owner_authenticated": True,
            "service_internal": True,
        },
    )

    result = adapter.dispatch(CanonicalWriterOperation.PING, {}, context)

    assert result.status == "ok"
    assert len(observed) == 1
    assert observed[0].user_id == "attacker"
    assert observed[0].owner_authenticated is False
    assert observed[0].service_internal is False
