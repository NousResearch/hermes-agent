from __future__ import annotations

import datetime as dt
import pytest

from gateway.canonical_canary_bootstrap import (
    CanaryScopeBootstrapRequest,
    CanaryScopePreclaimRetirementRequest,
)
from gateway.canonical_writer_handlers import (
    CanonicalWriterHandlers,
    InMemoryCanonicalWriterBackend,
    RuntimeContext,
)
from gateway.canonical_writer_protocol import (
    PROTOCOL_VERSION,
    CanonicalWriterOperation,
    ErrorCode,
    ProtocolError,
    parse_request,
)


SESSION = "a" * 64
EPOCH = "b" * 64
FRESH_EPOCH = "c" * 64
RELEASE = "d" * 64
FIXTURE = "e" * 64
APPROVAL = "f" * 64
NOW = dt.datetime(2026, 7, 13, 8, 0, tzinfo=dt.timezone.utc)


def _preapproval(*, expires_at: dt.datetime | None = None) -> dict[str, object]:
    return {
        "grant_id": "canary-grant-1",
        "case_id": "case:canary-1",
        "release_sha256": RELEASE,
        "fixture_sha256": FIXTURE,
        "run_id": "canary-run-1",
        "session_key_sha256": SESSION,
        "expires_at": (expires_at or NOW + dt.timedelta(minutes=30)).isoformat(),
        "approved_by": "1279454038731264061",
        "approval_source_sha256": APPROVAL,
    }


def _claim() -> dict[str, object]:
    payload = _preapproval()
    return {
        key: payload[key]
        for key in (
            "grant_id",
            "case_id",
            "release_sha256",
            "fixture_sha256",
            "run_id",
            "approval_source_sha256",
        )
    }


def _api_runtime(epoch: str = EPOCH, *, request_id: str = "api-run") -> RuntimeContext:
    return RuntimeContext(
        request_id=request_id,
        platform="api_server",
        session_key_sha256=SESSION,
        capability_epoch_sha256=epoch,
        chat_id="api-thread-1",
        thread_id="api-thread-1",
    )


def _append_payload(case_id: str, key: str) -> dict[str, object]:
    return {
        "event_type": "case.note",
        "case_id": case_id,
        "summary": "Model-authored progress receipt",
        "source_refs": {},
        "payload": {"progress": {"state": "observed"}},
        "idempotency_key": key,
    }


def _handlers(clock):
    backend = InMemoryCanonicalWriterBackend(clock=clock)
    return CanonicalWriterHandlers(backend), backend


def _bootstrap_preapprove(
    backend: InMemoryCanonicalWriterBackend,
    *,
    expires_at: dt.datetime | None = None,
    request_id: str = "bootstrap-preapproval",
):
    payload = _preapproval(expires_at=expires_at)
    return backend.bootstrap_canary_scope_preapprove(
        CanaryScopeBootstrapRequest(
            grant_id=str(payload["grant_id"]),
            case_id=str(payload["case_id"]),
            release_sha256=str(payload["release_sha256"]),
            fixture_sha256=str(payload["fixture_sha256"]),
            run_id=str(payload["run_id"]),
            session_key_sha256=str(payload["session_key_sha256"]),
            expires_at=dt.datetime.fromisoformat(str(payload["expires_at"])),
            approved_by=str(payload["approved_by"]),
            approval_source_sha256=str(payload["approval_source_sha256"]),
            provisioning_receipt_sha256="9" * 64,
        ),
        RuntimeContext(
            request_id=request_id,
            platform="writer_service",
            session_key_sha256=SESSION,
            service_internal=True,
        ),
    )


def _preclaim_retirement_request(
    *,
    expires_at: dt.datetime | None = None,
) -> CanaryScopePreclaimRetirementRequest:
    payload = _preapproval(expires_at=expires_at)
    return CanaryScopePreclaimRetirementRequest(
        grant_id=str(payload["grant_id"]),
        case_id=str(payload["case_id"]),
        release_sha256=str(payload["release_sha256"]),
        fixture_sha256=str(payload["fixture_sha256"]),
        run_id=str(payload["run_id"]),
        session_key_sha256=str(payload["session_key_sha256"]),
        expires_at=dt.datetime.fromisoformat(str(payload["expires_at"])),
        approved_by=str(payload["approved_by"]),
        approval_source_sha256=str(payload["approval_source_sha256"]),
        provisioning_receipt_sha256="9" * 64,
    )


def _retirement_runtime(request_id: str) -> RuntimeContext:
    return RuntimeContext(
        request_id=request_id,
        platform="writer_service",
        session_key_sha256=SESSION,
        service_internal=True,
    )


def test_preclaim_retirement_before_claim_is_one_append_only_tombstone():
    handlers, backend = _handlers(lambda: NOW)
    preapproved = _bootstrap_preapprove(backend)
    request = _preclaim_retirement_request()

    first = backend.retire_canary_scope_preapproval(
        request,
        _retirement_runtime("retire-first"),
    )
    replay = backend.retire_canary_scope_preapproval(
        request,
        _retirement_runtime("retire-replay"),
    )
    rejected_claim = handlers.dispatch(
        CanonicalWriterOperation.CANARY_SCOPE_CLAIM.value,
        _claim(),
        runtime=_api_runtime(),
    )

    assert first["outcome"] == "retired"
    assert first["authority_active"] is False
    assert first["inserted"] is True
    assert first["deduped"] is False
    assert first["preapproval_event_id"] == preapproved["receipt_event_id"]
    assert first["bootstrap_consumption_event_id"] == preapproved[
        "bootstrap_consumption_event_id"
    ]
    assert replay["retirement_event_id"] == first["retirement_event_id"]
    assert replay["retired_at"] == first["retired_at"]
    assert replay["inserted"] is False
    assert replay["deduped"] is True
    assert rejected_claim["error"]["code"] == (
        "canary_scope_preapproval_retired"
    )
    retired_events = [
        event
        for event in backend.store.events
        if event["event_type"] == "canary.scope.preapproval_retired"
    ]
    assert len(retired_events) == 1
    assert retired_events[0]["event_id"] == first["retirement_event_id"]
    assert retired_events[0]["origin"] == "canary_scope_preapproval_retire"


def test_preclaim_retirement_after_claim_reuses_exact_session_tombstone():
    handlers, backend = _handlers(lambda: NOW)
    _bootstrap_preapprove(backend)
    claimed = handlers.dispatch(
        CanonicalWriterOperation.CANARY_SCOPE_CLAIM.value,
        _claim(),
        runtime=_api_runtime(),
    )
    request = _preclaim_retirement_request()

    first = backend.retire_canary_scope_preapproval(
        request,
        _retirement_runtime("claimed-retire-first"),
    )
    replay = backend.retire_canary_scope_preapproval(
        request,
        _retirement_runtime("claimed-retire-replay"),
    )

    assert first["outcome"] == "claimed"
    assert first["claim_event_id"] == claimed["result"]["claim_event_id"]
    assert first["scope_retired"] is False
    assert first["authority_active"] is False
    assert first["retirement_event_id"] is None
    assert first["inserted"] is True
    assert first["deduped"] is False
    assert replay["revocation_event_id"] == first["revocation_event_id"]
    assert replay["inserted"] is False
    assert replay["deduped"] is True
    assert not any(
        event["event_type"] == "canary.scope.preapproval_retired"
        for event in backend.store.events
    )
    session_tombstones = [
        event
        for event in backend.store.events
        if event["event_type"] == "approval.capability.session_revoked"
        and "session_scope_revocation" in event["body"]
    ]
    assert len(session_tombstones) == 1
    assert session_tombstones[0]["event_id"] == first["revocation_event_id"]
    assert session_tombstones[0]["origin"] == "capability_revoke_session"


def test_preclaim_retirement_not_preapproved_is_terminal_safe_noop():
    _, backend = _handlers(lambda: NOW)

    result = backend.retire_canary_scope_preapproval(
        _preclaim_retirement_request(),
        _retirement_runtime("not-preapproved"),
    )

    assert result["outcome"] == "not_preapproved"
    assert result["reason"] == "preapproval_not_committed"
    assert result["authority_active"] is False
    assert result["inserted"] is False
    assert result["deduped"] is False
    assert backend.store.events == []


def test_preapproval_claim_mutation_and_retirement_are_exact_and_append_only():
    current = [NOW]
    handlers, backend = _handlers(lambda: current[0])

    preapproved = _bootstrap_preapprove(backend)
    assert preapproved["inserted"] is True
    stored_preapproval = backend.store.canary_scope_preapprovals["canary-grant-1"]
    assert preapproved["preapproved_at"] == stored_preapproval["preapproved_at"]
    preapproved_retry = _bootstrap_preapprove(
        backend,
        request_id="bootstrap-preapproval-retry",
    )
    assert preapproved_retry["deduped"] is True
    assert preapproved_retry["preapproved_at"] == preapproved["preapproved_at"]
    assert preapproved_retry["receipt_event_id"] == preapproved["receipt_event_id"]

    before_claim = handlers.dispatch(
        CanonicalWriterOperation.EVENT_APPEND_MODEL.value,
        _append_payload("case:canary-1", "before-claim"),
        runtime=_api_runtime(),
    )
    assert before_claim["error"]["code"] == "scope_mismatch"

    claimed = handlers.dispatch(
        CanonicalWriterOperation.CANARY_SCOPE_CLAIM.value,
        _claim(),
        runtime=_api_runtime(),
    )
    assert claimed["ok"] is True
    assert claimed["result"]["authority_active"] is True
    assert claimed["result"]["expires_at"] == _preapproval()["expires_at"]
    stored_claim = backend.store.canary_scope_claims["canary-grant-1"]
    assert claimed["result"]["claimed_at"] == stored_claim["claimed_at"]

    retry = handlers.dispatch(
        CanonicalWriterOperation.CANARY_SCOPE_CLAIM.value,
        _claim(),
        runtime=_api_runtime(request_id="claim-retry"),
    )
    assert retry["ok"] is True
    assert retry["result"]["deduped"] is True
    assert retry["result"]["claim_event_id"] == claimed["result"]["claim_event_id"]
    assert retry["result"]["claimed_at"] == claimed["result"]["claimed_at"]
    assert retry["result"]["expires_at"] == claimed["result"]["expires_at"]

    appended = handlers.dispatch(
        CanonicalWriterOperation.EVENT_APPEND_MODEL.value,
        _append_payload("case:canary-1", "after-claim"),
        runtime=_api_runtime(),
    )
    assert appended["ok"] is True

    fresh_epoch = handlers.dispatch(
        CanonicalWriterOperation.EVENT_APPEND_MODEL.value,
        _append_payload("case:canary-1", "fresh-epoch"),
        runtime=_api_runtime(FRESH_EPOCH, request_id="fresh-epoch"),
    )
    assert fresh_epoch["error"]["code"] == "scope_mismatch"

    revoked = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_REVOKE_SESSION.value,
        {"reason": "canary run completed"},
        runtime=_api_runtime(),
    )
    assert revoked["ok"] is True
    assert revoked["result"]["revoked"] == 0
    assert revoked["result"]["canary_scopes_revoked"] == 1
    assert [
        event["event_type"] for event in backend.store.events
    ].count("canary.scope.revoked") == 1

    revoked_retry = handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_REVOKE_SESSION.value,
        {"reason": "different retry text cannot rewrite tombstone"},
        runtime=_api_runtime(request_id="revoke-retry"),
    )
    assert revoked_retry["result"]["canary_scopes_revoked"] == 1
    assert [
        event["event_type"] for event in backend.store.events
    ].count("canary.scope.revoked") == 1

    after_revoke = handlers.dispatch(
        CanonicalWriterOperation.CASE_QUERY.value,
        {"case_id": "case:canary-1", "limit": 20},
        runtime=_api_runtime(request_id="after-revoke"),
    )
    assert after_revoke["error"]["code"] == "scope_mismatch"


def test_public_protocol_has_no_canary_preapproval_operation():
    with pytest.raises(ValueError):
        CanonicalWriterOperation("canary.scope_preapprove")
    with pytest.raises(ProtocolError) as failure:
        parse_request(
            {
                "protocol": PROTOCOL_VERSION,
                "request_id": "1b334bb8-29fd-4f50-85d2-69071d5024b3",
                "sequence": 1,
                "operation": "canary.scope_preapprove",
                "deadline_unix_ms": 1_010_000,
                "runtime": {},
                "payload": _preapproval(),
            },
            now=1_000,
        )
    assert failure.value.code is ErrorCode.UNKNOWN_OPERATION


def test_wrong_expired_and_replayed_claims_are_blocked():
    current = [NOW]
    handlers, backend = _handlers(lambda: current[0])
    _bootstrap_preapprove(backend)

    wrong = _claim()
    wrong["fixture_sha256"] = "0" * 64
    wrong_result = handlers.dispatch(
        CanonicalWriterOperation.CANARY_SCOPE_CLAIM.value,
        wrong,
        runtime=_api_runtime(),
    )
    assert wrong_result["error"]["code"] == "scope_mismatch"

    current[0] = NOW + dt.timedelta(minutes=31)
    expired = handlers.dispatch(
        CanonicalWriterOperation.CANARY_SCOPE_CLAIM.value,
        _claim(),
        runtime=_api_runtime(),
    )
    assert expired["error"]["code"] == "canary_scope_expired"

    current[0] = NOW
    claimed = handlers.dispatch(
        CanonicalWriterOperation.CANARY_SCOPE_CLAIM.value,
        _claim(),
        runtime=_api_runtime(),
    )
    assert claimed["ok"] is True
    replayed = handlers.dispatch(
        CanonicalWriterOperation.CANARY_SCOPE_CLAIM.value,
        _claim(),
        runtime=_api_runtime(FRESH_EPOCH, request_id="replayed-claim"),
    )
    assert replayed["error"]["code"] == "canary_scope_replayed"


def test_thread_reads_recheck_each_canary_case_but_ordinary_discord_is_unchanged():
    handlers, backend = _handlers(lambda: NOW)
    _bootstrap_preapprove(backend)
    handlers.dispatch(
        CanonicalWriterOperation.CANARY_SCOPE_CLAIM.value,
        _claim(),
        runtime=_api_runtime(),
    )
    active = handlers.dispatch(
        CanonicalWriterOperation.CASE_QUERY.value,
        {"thread_id": "api-thread-1", "limit": 20},
        runtime=_api_runtime(),
    )
    assert any(
        event["case_id"] == "case:canary-1" for event in active["result"]["events"]
    )

    fresh = handlers.dispatch(
        CanonicalWriterOperation.CASE_QUERY.value,
        {"thread_id": "api-thread-1", "limit": 20},
        runtime=_api_runtime(FRESH_EPOCH, request_id="fresh-thread-query"),
    )
    assert fresh["result"]["events"] == []

    discord_runtime = RuntimeContext(
        request_id="discord-append",
        platform="discord",
        session_key_sha256="1" * 64,
        capability_epoch_sha256="2" * 64,
        chat_id="public-thread-1",
        thread_id="public-thread-1",
    )
    ordinary = handlers.dispatch(
        CanonicalWriterOperation.EVENT_APPEND_MODEL.value,
        _append_payload("case:ordinary-discord", "ordinary"),
        runtime=discord_runtime,
    )
    assert ordinary["ok"] is True
    ordinary_read = handlers.dispatch(
        CanonicalWriterOperation.CASE_QUERY.value,
        {"thread_id": "public-thread-1", "limit": 20},
        runtime=RuntimeContext(
            request_id="discord-read",
            platform="discord",
            session_key_sha256="1" * 64,
            capability_epoch_sha256="3" * 64,
            chat_id="public-thread-1",
            thread_id="public-thread-1",
        ),
    )
    assert [event["case_id"] for event in ordinary_read["result"]["events"]] == [
        "case:ordinary-discord"
    ]
