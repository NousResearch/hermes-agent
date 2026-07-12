import datetime as dt
import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.canonical_writer_handlers import (
    CanonicalWriterHandlers,
    InMemoryCanonicalWriterBackend,
    RuntimeContext,
)
from gateway.canonical_writer_protocol import CanonicalWriterOperation
from gateway.discord_edge_protocol import (
    DiscordEdgeAuthorityKind,
    DiscordEdgeErrorCode,
    DiscordEdgeReceiptOutcome,
    SignedDiscordEdgeEnvelope,
    parse_request_for_reconciliation,
    sign_receipt,
    verify_request_capability_for_reconciliation,
)
from gateway.discord_edge_writer_authority import (
    CanonicalWriterDiscordAuthority,
    derive_routeback_edge_idempotency_key,
)


NOW_MS = 1_800_000_000_000
NOW = dt.datetime.fromtimestamp(NOW_MS / 1_000, tz=dt.timezone.utc)
SESSION_SHA256 = "a" * 64
EPOCH_SHA256 = "b" * 64
GUILD_ID = "100000000000000001"
CHANNEL_ID = "100000000000000002"
MESSAGE_ID = "100000000000000003"
BOT_ID = "100000000000000004"

WRITER_PRIVATE_KEY = Ed25519PrivateKey.generate()
EDGE_PRIVATE_KEY = Ed25519PrivateKey.generate()


def _runtime(
    *,
    session_sha256: str = SESSION_SHA256,
    epoch_sha256: str = EPOCH_SHA256,
    platform: str = "discord",
    thread_id: str = CHANNEL_ID,
) -> RuntimeContext:
    return RuntimeContext(
        request_id="writer-request-1",
        platform=platform,
        session_key_sha256=session_sha256,
        capability_epoch_sha256=epoch_sha256,
        user_id="owner-1",
        chat_id=thread_id,
        thread_id=thread_id,
        message_id="100000000000000005",
        owner_authenticated=True,
    )


def _authority() -> CanonicalWriterDiscordAuthority:
    return CanonicalWriterDiscordAuthority(
        capability_private_key=WRITER_PRIVATE_KEY,
        edge_receipt_public_key=EDGE_PRIVATE_KEY.public_key(),
        clock_unix_ms=lambda: NOW_MS,
    )


def _handlers(*, authority=True, public_routeback_targets=None):
    backend = InMemoryCanonicalWriterBackend(
        clock=lambda: NOW,
        public_routeback_targets=public_routeback_targets,
    )
    handlers = CanonicalWriterHandlers(
        backend,
        discord_edge_authority=_authority() if authority else None,
    )
    return handlers, backend


def _claim_payload(
    *,
    key="routeback:signed:1",
    content="Exact public route-back content",
):
    content_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    edge_key = derive_routeback_edge_idempotency_key(
        case_id="case:signed-routeback",
        canonical_idempotency_key=key,
    )
    return {
        "case_id": "case:signed-routeback",
        "target_ref": {
            "target_type": "public_guild_channel",
            "guild_id": GUILD_ID,
            "channel_id": CHANNEL_ID,
        },
        "message_summary": "Send the exact authorized result",
        "source_refs": {"thread_id": "100000000000000006"},
        "idempotency_key": key,
        "execution_binding": {
            "target_channel_id": CHANNEL_ID,
            "content_sha256": content_sha256,
        },
        "discord_edge_intent": {
            "operation": "public.message.send",
            "target": {
                "target_type": "public_guild_channel",
                "guild_id": GUILD_ID,
                "channel_id": CHANNEL_ID,
            },
            "payload": {"content": content},
            "idempotency_key": edge_key,
        },
    }


def _claim(handlers, payload=None):
    claim_payload = payload or _claim_payload()
    response = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        claim_payload,
        runtime=_runtime(),
    )
    assert response["ok"] is True
    return claim_payload, response["result"]


def _signed_receipt(
    edge_request,
    *,
    outcome=DiscordEdgeReceiptOutcome.VERIFIED,
    private_key=EDGE_PRIVATE_KEY,
):
    request = parse_request_for_reconciliation(edge_request)
    capability = verify_request_capability_for_reconciliation(
        request,
        WRITER_PRIVATE_KEY.public_key(),
    )
    if outcome is DiscordEdgeReceiptOutcome.VERIFIED:
        kwargs = {
            "discord_object_id": MESSAGE_ID,
            "bot_user_id": BOT_ID,
            "adapter_accepted": True,
            "readback_verified": True,
            "readback_content_sha256": request.intent.content_sha256,
            "blocker_code": None,
        }
    elif outcome is DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED:
        kwargs = {
            "discord_object_id": MESSAGE_ID,
            "bot_user_id": BOT_ID,
            "adapter_accepted": True,
            "readback_verified": False,
            "readback_content_sha256": None,
            "blocker_code": "readback_timeout",
        }
    elif outcome is DiscordEdgeReceiptOutcome.DISPATCH_UNCERTAIN:
        kwargs = {
            "discord_object_id": None,
            "bot_user_id": None,
            "adapter_accepted": None,
            "readback_verified": False,
            "readback_content_sha256": None,
            "blocker_code": "transport_closed",
        }
    else:
        kwargs = {
            "discord_object_id": None,
            "bot_user_id": None,
            "adapter_accepted": False,
            "readback_verified": False,
            "readback_content_sha256": None,
            "blocker_code": "target_not_public",
        }
    return sign_receipt(
        private_key,
        request,
        capability,
        outcome=outcome,
        occurred_at_unix_ms=NOW_MS,
        **kwargs,
    ).to_message()


def _terminal_payload(claim_payload, edge_request, edge_receipt):
    return {
        key: claim_payload[key]
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


def _recovery_payload(
    claim_payload,
    *,
    recovery_kind,
    edge_request=None,
    edge_receipt=None,
):
    payload = {
        key: claim_payload[key]
        for key in (
            "case_id",
            "target_ref",
            "message_summary",
            "source_refs",
            "idempotency_key",
            "execution_binding",
            "discord_edge_intent",
        )
    } | {"recovery_kind": recovery_kind}
    if recovery_kind == "edge_evidence":
        payload["discord_edge_request"] = edge_request
        payload["discord_edge_receipt"] = edge_receipt
    return payload


def _tamper_signature(envelope):
    changed = dict(envelope)
    signature = str(changed["signature"])
    changed["signature"] = ("A" if signature[0] != "A" else "B") + signature[1:]
    return changed


def test_nonterminal_deduped_claim_returns_fresh_writer_signed_request():
    handlers, _backend = _handlers()
    payload, first = _claim(handlers)
    replay = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        payload,
        runtime=_runtime(),
    )["result"]

    request = parse_request_for_reconciliation(first["discord_edge_request"])
    capability = verify_request_capability_for_reconciliation(
        request,
        WRITER_PRIVATE_KEY.public_key(),
    )

    assert first["inserted"] is True
    assert capability.authority_kind is DiscordEdgeAuthorityKind.CANONICAL_ROUTEBACK
    assert capability.authority_ref == first["authorization_id"]
    replay_request = parse_request_for_reconciliation(
        replay["discord_edge_request"]
    )
    replay_capability = verify_request_capability_for_reconciliation(
        replay_request,
        WRITER_PRIVATE_KEY.public_key(),
    )
    assert request.intent.idempotency_key == derive_routeback_edge_idempotency_key(
        case_id=payload["case_id"],
        canonical_idempotency_key=payload["idempotency_key"],
    )
    assert replay["inserted"] is False
    assert replay_request.intent == request.intent
    assert replay_request.request_id != request.request_id
    assert replay_capability.issued_at_unix_ms > capability.issued_at_unix_ms


def test_epoch_restart_recovers_verified_edge_truth_before_terminal_finalize():
    handlers, backend = _handlers(public_routeback_targets=frozenset())
    payload, claim = _claim(
        handlers,
        _claim_payload(key="routeback:restart:verified-before-terminal"),
    )
    edge_request = claim["discord_edge_request"]
    edge_receipt = _signed_receipt(edge_request)
    restarted = _runtime(epoch_sha256="c" * 64)

    recovered = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_RECOVER.value,
        _recovery_payload(
            payload,
            recovery_kind="edge_evidence",
            edge_request=edge_request,
            edge_receipt=edge_receipt,
        ),
        runtime=restarted,
    )
    finalized = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT.value,
        _terminal_payload(payload, edge_request, edge_receipt),
        runtime=restarted,
    )

    assert recovered["ok"] is True
    assert recovered["result"]["recovered"] is True
    assert recovered["result"]["recovery_kind"] == "edge_evidence"
    assert "discord_edge_request" not in recovered["result"]
    assert recovered["result"]["discord_edge_evidence"]["outcome"] == "verified"
    authorization = backend.store.routeback_authorizations[
        claim["authorization_id"]
    ]
    assert authorization["capability_epoch_sha256"] == EPOCH_SHA256
    assert recovered["result"]["recovered_epoch_sha256"] == "c" * 64
    assert finalized["ok"] is True
    assert [event["event_type"] for event in backend.store.events] == [
        "route_back.intent.created",
        "route_back.sent",
    ]


def test_epoch_restart_no_record_rebinds_only_exact_lane_and_live_acl():
    handlers, backend = _handlers(
        public_routeback_targets=frozenset({CHANNEL_ID})
    )
    payload, claim = _claim(
        handlers,
        _claim_payload(key="routeback:restart:claim-before-edge"),
    )
    restarted = _runtime(epoch_sha256="c" * 64)

    recovered = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_RECOVER.value,
        _recovery_payload(payload, recovery_kind="edge_no_record"),
        runtime=restarted,
    )

    assert recovered["ok"] is True
    assert recovered["result"]["recovered"] is True
    fresh_request = parse_request_for_reconciliation(
        recovered["result"]["discord_edge_request"]
    )
    original_request = parse_request_for_reconciliation(
        claim["discord_edge_request"]
    )
    assert fresh_request.request_id != original_request.request_id
    assert fresh_request.intent == original_request.intent
    assert len(backend.store.routeback_authorizations) == 1
    assert [event["event_type"] for event in backend.store.events] == [
        "route_back.intent.created"
    ]


@pytest.mark.parametrize(
    "runtime",
    [
        _runtime(session_sha256="f" * 64, epoch_sha256="c" * 64),
        _runtime(epoch_sha256="c" * 64, thread_id="100000000000000099"),
        _runtime(epoch_sha256="c" * 64, platform="telegram"),
    ],
)
def test_epoch_restart_recovery_cannot_cross_session_or_source_lane(runtime):
    handlers, backend = _handlers(
        public_routeback_targets=frozenset({CHANNEL_ID})
    )
    payload, claim = _claim(
        handlers,
        _claim_payload(key="routeback:restart:wrong-lane"),
    )

    rejected = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_RECOVER.value,
        _recovery_payload(payload, recovery_kind="edge_no_record"),
        runtime=runtime,
    )

    assert rejected["error"]["code"] == "scope_mismatch"
    authorization = backend.store.routeback_authorizations[
        claim["authorization_id"]
    ]
    assert authorization["capability_epoch_sha256"] == EPOCH_SHA256
    assert len(backend.store.routeback_authorizations) == 1


def test_recovery_acl_change_blocks_fresh_send_but_not_signed_truth():
    handlers, backend = _handlers(
        public_routeback_targets=frozenset({CHANNEL_ID})
    )
    payload, claim = _claim(
        handlers,
        _claim_payload(key="routeback:restart:acl-rotated"),
    )
    edge_request = claim["discord_edge_request"]
    edge_receipt = _signed_receipt(edge_request)
    backend._public_routeback_targets = frozenset()
    restarted = _runtime(epoch_sha256="c" * 64)

    fresh_send = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_RECOVER.value,
        _recovery_payload(payload, recovery_kind="edge_no_record"),
        runtime=restarted,
    )
    signed_truth = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_RECOVER.value,
        _recovery_payload(
            payload,
            recovery_kind="edge_evidence",
            edge_request=edge_request,
            edge_receipt=edge_receipt,
        ),
        runtime=restarted,
    )

    assert fresh_send["error"]["code"] == "target_not_approved"
    assert signed_truth["ok"] is True
    assert signed_truth["result"]["recovered"] is True
    assert len(backend.store.routeback_authorizations) == 1


def test_claim_fails_closed_without_writer_owned_authority():
    handlers, backend = _handlers(authority=False)

    response = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        _claim_payload(),
        runtime=_runtime(),
    )

    assert response["error"]["code"] == "discord_edge_authority_unavailable"
    assert backend.store.routeback_authorizations == {}
    assert backend.store.events == []


def test_claimed_finalization_fails_closed_when_authority_becomes_unavailable():
    handlers, backend = _handlers()
    payload, claim = _claim(handlers)
    edge_request = claim["discord_edge_request"]
    edge_receipt = _signed_receipt(edge_request)
    unavailable = CanonicalWriterHandlers(backend)

    response = unavailable.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT.value,
        _terminal_payload(payload, edge_request, edge_receipt),
        runtime=_runtime(),
    )

    assert response["error"]["code"] == "discord_edge_authority_unavailable"
    assert backend.store.routeback_authorizations[claim["authorization_id"]][
        "terminal"
    ] is None


@pytest.mark.parametrize("mismatch", ["target", "content", "idempotency"])
def test_claim_rejects_edge_intent_binding_mismatch(mismatch):
    handlers, backend = _handlers()
    payload = _claim_payload()
    if mismatch == "target":
        payload["discord_edge_intent"]["target"]["channel_id"] = (
            "100000000000000099"
        )
    elif mismatch == "content":
        payload["discord_edge_intent"]["payload"]["content"] = "Other content"
    else:
        payload["discord_edge_intent"]["idempotency_key"] = "routeback:other"

    response = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        payload,
        runtime=_runtime(),
    )

    assert response["error"]["code"] == "scope_mismatch"
    assert backend.store.routeback_authorizations == {}


def test_verified_signed_receipt_is_the_only_source_of_sent_truth():
    handlers, backend = _handlers()
    payload, claim = _claim(handlers)
    edge_request = claim["discord_edge_request"]
    edge_receipt = _signed_receipt(edge_request)

    result = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT.value,
        _terminal_payload(payload, edge_request, edge_receipt),
        runtime=_runtime(),
    )

    assert result["ok"] is True
    assert result["result"]["receipt"] == {
        "platform": "discord",
        "adapter_receipt": True,
        "receipt_readback_verified": True,
        "message_id": MESSAGE_ID,
        "channel_id": CHANNEL_ID,
        "content_sha256": payload["execution_binding"]["content_sha256"],
    }
    assert result["result"]["discord_edge_evidence"]["outcome"] == "verified"
    assert backend.store.events[-1]["event_type"] == "route_back.sent"


@pytest.mark.parametrize(
    "outcome,expected_blocker,has_partial_receipt",
    [
        (
            DiscordEdgeReceiptOutcome.DISPATCH_UNCERTAIN,
            "discord_edge:dispatch_uncertain:transport_closed",
            False,
        ),
        (
            DiscordEdgeReceiptOutcome.BLOCKED_BEFORE_DISPATCH,
            "discord_edge:blocked_before_dispatch:target_not_public",
            False,
        ),
    ],
)
def test_nonverified_signed_outcome_derives_blocked_truth(
    outcome,
    expected_blocker,
    has_partial_receipt,
):
    handlers, backend = _handlers()
    payload, claim = _claim(handlers, _claim_payload(key=f"routeback:{outcome.value}"))
    edge_request = claim["discord_edge_request"]
    edge_receipt = _signed_receipt(edge_request, outcome=outcome)

    result = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED.value,
        _terminal_payload(payload, edge_request, edge_receipt),
        runtime=_runtime(),
    )

    assert result["ok"] is True
    assert result["result"]["blocker_reason"] == expected_blocker
    assert bool(result["result"]["partial_receipt"]) is has_partial_receipt
    assert backend.store.events[-1]["event_type"] == "route_back.blocked"


def test_accepted_unverified_cannot_finalize_canonical_blocked_truth():
    handlers, backend = _handlers()
    payload, claim = _claim(
        handlers,
        _claim_payload(key="routeback:accepted-unverified-pending"),
    )
    edge_request = claim["discord_edge_request"]
    edge_receipt = _signed_receipt(
        edge_request,
        outcome=DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED,
    )

    result = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED.value,
        _terminal_payload(payload, edge_request, edge_receipt),
        runtime=_runtime(),
    )

    assert result["error"]["code"] == "discord_edge_outcome_pending"
    authorization = backend.store.routeback_authorizations[
        claim["authorization_id"]
    ]
    assert authorization["terminal"] is None
    assert all(event["event_type"] != "route_back.blocked" for event in backend.store.events)


def test_claimed_blocker_cannot_be_supplied_by_gateway():
    handlers, _backend = _handlers()
    payload, claim = _claim(handlers)
    edge_request = claim["discord_edge_request"]
    edge_receipt = _signed_receipt(
        edge_request,
        outcome=DiscordEdgeReceiptOutcome.DISPATCH_UNCERTAIN,
    )
    terminal = _terminal_payload(payload, edge_request, edge_receipt)
    terminal["blocker_reason"] = "gateway_claimed_blocker"

    result = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED.value,
        terminal,
        runtime=_runtime(),
    )

    assert result["error"]["code"] == "invalid_request"


def test_forged_writer_capability_and_edge_receipt_signatures_are_rejected():
    handlers, _backend = _handlers()
    payload, claim = _claim(handlers)
    edge_request = claim["discord_edge_request"]
    edge_receipt = _signed_receipt(edge_request)

    forged_request = dict(edge_request)
    forged_request["capability"] = _tamper_signature(edge_request["capability"])
    forged_capability = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT.value,
        _terminal_payload(payload, forged_request, edge_receipt),
        runtime=_runtime(),
    )
    forged_receipt = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT.value,
        _terminal_payload(
            payload,
            edge_request,
            _tamper_signature(edge_receipt),
        ),
        runtime=_runtime(),
    )

    assert forged_capability["error"]["code"] == "invalid_discord_edge_evidence"
    assert forged_receipt["error"]["code"] == "invalid_discord_edge_evidence"


def test_signed_outcomes_cannot_cross_sent_and_blocked_finalizers():
    handlers, _backend = _handlers()
    payload, claim = _claim(handlers)
    edge_request = claim["discord_edge_request"]
    unverified = _signed_receipt(
        edge_request,
        outcome=DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED,
    )
    verified = _signed_receipt(edge_request)

    sent = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT.value,
        _terminal_payload(payload, edge_request, unverified),
        runtime=_runtime(),
    )
    blocked = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED.value,
        _terminal_payload(payload, edge_request, verified),
        runtime=_runtime(),
    )

    assert sent["error"]["code"] == "invalid_discord_edge_outcome"
    assert blocked["error"]["code"] == "invalid_discord_edge_outcome"


def test_finalizer_rejects_target_content_and_idempotency_rebinding():
    handlers, _backend = _handlers()
    payload, claim = _claim(handlers)
    edge_request = claim["discord_edge_request"]
    edge_receipt = _signed_receipt(edge_request)

    for field in ("target", "content", "idempotency"):
        terminal = _terminal_payload(payload, edge_request, edge_receipt)
        if field == "target":
            terminal["target_ref"] = {
                **terminal["target_ref"],
                "channel_id": "100000000000000099",
            }
            terminal["execution_binding"] = {
                **terminal["execution_binding"],
                "target_channel_id": "100000000000000099",
            }
        elif field == "content":
            terminal["execution_binding"] = {
                **terminal["execution_binding"],
                "content_sha256": "f" * 64,
            }
        else:
            terminal["idempotency_key"] = "routeback:rebound"
        response = handlers.dispatch(
            CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT.value,
            terminal,
            runtime=_runtime(),
        )
        assert response["error"]["code"] in {
            "invalid_discord_edge_evidence",
            "scope_mismatch",
        }


def test_preclaim_blocked_remains_unsigned_and_never_mints_dispatch_authority():
    handlers, backend = _handlers(authority=False)

    response = handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED.value,
        {
            "preclaim": True,
            "case_id": "case:preclaim",
            "target_ref": {"id": "unresolved-public-target"},
            "message_summary": "public target could not be resolved",
            "source_refs": {"thread_id": CHANNEL_ID},
            "blocker_reason": "target unresolved before claim",
            "idempotency_key": "routeback:preclaim",
        },
        runtime=_runtime(),
    )

    assert response["ok"] is True
    assert response["result"]["preclaim"] is True
    assert "discord_edge_request" not in response["result"]
    assert backend.store.routeback_authorizations == {}


def test_authority_rejects_non_send_and_private_target_intents():
    authority = _authority()
    non_send = _claim_payload()["discord_edge_intent"]
    non_send["operation"] = "public.message.edit"
    non_send["payload"] = {"message_id": MESSAGE_ID, "content": "edit"}
    private_target = _claim_payload()["discord_edge_intent"]
    private_target["target"]["target_type"] = "dm"

    with pytest.raises(ValueError, match="public.message.send"):
        authority.parse_public_send_intent(non_send)
    with pytest.raises(ValueError, match="public Discord protocol boundary"):
        authority.parse_public_send_intent(private_target)


def test_edge_receipt_envelope_is_an_exact_signed_shape():
    handlers, _backend = _handlers()
    payload, claim = _claim(handlers)
    edge_request = claim["discord_edge_request"]
    edge_receipt = _signed_receipt(edge_request)
    envelope = SignedDiscordEdgeEnvelope.from_mapping(
        edge_receipt,
        code=DiscordEdgeErrorCode.INVALID_RECEIPT,
        label="test receipt",
    )

    assert envelope.to_message() == edge_receipt
    assert payload["execution_binding"]["content_sha256"] == (
        parse_request_for_reconciliation(edge_request).intent.content_sha256
    )
