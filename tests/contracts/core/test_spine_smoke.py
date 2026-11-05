"""Smoke suite for the shared persistence types and backend fixture.

Checks that shared symbols import from ``persistence``, identifiers pass through payloads, pages and structured errors without conversion, timestamps round-trip through ISO strings, scope and paging primitives enforce the tested invariants, and errors carry their structured fields. The shared fixture assembles a real backend and checks isolation between two scopes routed through the same provider instance.

Cross-capability sharing is exercised with the core types rather than capability records. The two-scope test uses the reference backend's scoped-storage surface directly. Isolation checks through public capability protocols belong to their capability suites; nothing here counts as that coverage.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone, UTC

import pytest

from persistence import (
    MUTATION_GUARD_REFUSALS,
    PERSISTENCE_ERROR_CAUSES,
    STORE_FAILURES,
    AuthorityRefusal,
    CapabilityUnsupported,
    ClaimToken,
    CompressionBusy,
    CompressionClosed,
    CompressionInProgress,
    Conflict,
    ContinuationToken,
    CorruptStoreError,
    DeclarativeDrift,
    DelegationId,
    DeletedWalGeneration,
    DiskFull,
    ExportTooLarge,
    FenceStale,
    HeldRowRewritten,
    JsonValue,
    MalformedSchema,
    MessageId,
    NotFound,
    Page,
    PageRequest,
    PermanentStoreError,
    PersistenceCapabilities,
    PersistenceError,
    ProfileScope,
    ProofRejected,
    PruneRowStale,
    PruneRowUnresolved,
    RetiredSessionError,
    RowVisibility,
    SearchIndexCorrupt,
    SessionId,
    SessionRowMissing,
    SessionWriteGuarded,
    StaleFenceKind,
    StoreContention,
    StoreQuarantined,
    StoreReplaced,
    StoreUnavailable,
    TransientStoreError,
    TurnLeaseLost,
    UnknownExecution,
    busy_refusal,
    ensure_utc,
)
from tests.contracts._scopes import profile_scope

# ── Core surface: identity, records, conventions ────────────────────────────


def test_cross_capability_records_share_ids_and_timestamps_without_glue():
    # The same identifiers pass through JsonValue payloads, pages and structured errors without conversion. The timestamp is serialized as an ISO string. MessageId is an int; the other identifiers are strings.
    sid = SessionId("sess-1")
    mid = MessageId(41)
    did = DelegationId("dlg-1")
    token = ClaimToken("claim-1")
    stamp = ensure_utc(datetime(2026, 9, 11, 14, 30, 0, tzinfo=UTC))

    record: dict[str, JsonValue] = {
        "session": sid,
        "message": mid,
        "delegation": did,
        "claim": token,
        "created_at": stamp.isoformat(),
        "nested": [1, 2.5, None, True, {"deep": "value"}],
    }
    payload: JsonValue = record
    page: Page[JsonValue] = Page(items=(payload,), continuation=None)
    assert record["session"] == sid
    assert record["message"] == mid
    assert page.items[0] is record

    # The error carries the same session id as the page's record, and the timestamp string parses back to the original instant.
    err = ExportTooLarge(sid, 25, 10)
    assert err.session_id == sid
    assert ensure_utc(stamp) == stamp
    assert datetime.fromisoformat(str(record["created_at"])) == stamp


def test_timestamp_convention_rejects_naive_and_normalizes_to_utc():
    with pytest.raises(ValueError):
        ensure_utc(datetime(2026, 9, 11, 12, 0, 0))
    offset = timezone(timedelta(hours=2))
    local = datetime(2026, 9, 11, 12, 0, 0, tzinfo=offset)
    normalized = ensure_utc(local)
    assert normalized.tzinfo == UTC
    assert normalized == local


def test_visibility_vocabulary_is_the_three_observed_states():
    # A transcript row is in exactly one of these three states; the enum is the
    # shared vocabulary capability modules use for include_inactive semantics.
    assert {v.value for v in RowVisibility} == {
        "active",
        "rewound",
        "compaction_archived",
    }


def test_page_request_rejects_nonpositive_limits():
    with pytest.raises(ValueError):
        PageRequest(limit=0)
    with pytest.raises(ValueError):
        PageRequest(limit=-3)
    request = PageRequest(limit=1)
    assert request.cursor is None


def test_paging_records_are_frozen_and_carry_opaque_continuation():
    token = ContinuationToken("cursor-opaque")
    page: Page[str] = Page(items=("a", "b"), continuation=token)
    with pytest.raises(dataclasses.FrozenInstanceError):
        page.items = ()  # type: ignore[misc]
    resumed = PageRequest(limit=5, cursor=page.continuation)
    assert resumed.cursor == token


# ── Scope and identity records ───────────────────────────────────────────────


def test_profile_scope_requires_a_profile_and_is_frozen():
    with pytest.raises(TypeError):
        ProfileScope()  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        profile_scope("")
    scope = profile_scope("p")
    with pytest.raises(dataclasses.FrozenInstanceError):
        scope.name = "other"  # type: ignore[misc]
    assert scope == profile_scope("p")


# ── Error taxonomy ───────────────────────────────────────────────────────────


def test_error_taxonomy_is_rooted_and_carries_structured_fields():
    for exc_type in (
        NotFound,
        Conflict,
        TransientStoreError,
        PermanentStoreError,
        StoreUnavailable,
        StoreReplaced,
        ExportTooLarge,
        CapabilityUnsupported,
        AuthorityRefusal,
    ):
        assert issubclass(exc_type, PersistenceError)

    not_found = NotFound("session", "sess-9")
    assert (not_found.entity, not_found.entity_id) == ("session", "sess-9")

    conflict = Conflict("session", "sess-7")
    assert (conflict.entity, conflict.entity_id) == ("session", "sess-7")

    too_large = ExportTooLarge(SessionId("sess-x"), 250, 100)
    assert (too_large.session_id, too_large.message_count, too_large.limit) == ("sess-x", 250, 100)
    assert isinstance(too_large, ValueError)

    unsupported = CapabilityUnsupported("full_text_search")
    assert unsupported.capability == "full_text_search"


@pytest.mark.parametrize("error, cause, builtin", [
    (SessionRowMissing(SessionId("s")), "session_row_missing", None),
    (CompressionClosed(SessionId("s"), SessionId("t")), "compression_closed", RuntimeError),
    (CompressionBusy(SessionId("s")), "compression", RuntimeError),
    (CompressionInProgress(SessionId("s")), "compression", RuntimeError),
    (TurnLeaseLost(SessionId("s")), "turn_lease", RuntimeError),
    (SessionWriteGuarded(SessionId("s")), "unknown", RuntimeError),
    (HeldRowRewritten(SessionId("s")), "unknown", RuntimeError),
    (PruneRowStale(SessionId("s")), "unknown", RuntimeError),
    (TransientStoreError(), "disk", None),
    (StoreContention(), "locked", None),
    (DiskFull(), "disk", None),
    (StoreQuarantined(), "corrupt", None),
    (MalformedSchema(), "corrupt", None),
    (SearchIndexCorrupt(), "fts_index", None),
    (DeclarativeDrift("missing column"), "unknown", RuntimeError),
    (StoreReplaced(), "replaced", RuntimeError),
    (DeletedWalGeneration(), "deleted_wal", RuntimeError),
    (AuthorityRefusal("queue_full"), "unknown", ValueError),
])
def test_every_error_names_its_cause_and_keeps_its_builtin_base(error, cause, builtin):
    assert error.cause == cause and cause in PERSISTENCE_ERROR_CAUSES
    if builtin is not None:
        assert isinstance(error, builtin)


def test_lease_and_guard_refusals_keep_the_store_texts_their_classification_reads():
    assert "turn lease" in str(TurnLeaseLost(SessionId("s"))).lower()
    assert "being compressed" in str(CompressionInProgress(SessionId("s")))
    assert "compression lease" in str(CompressionBusy(SessionId("s"))).lower()
    assert "closed by compression" in str(CompressionClosed(SessionId("s"), SessionId("t")))
    assert "turn lease" in str(SessionWriteGuarded(SessionId("s")))
    assert str(TurnLeaseLost(SessionId("s"), "custom text")) == "custom text"
    assert issubclass(CompressionInProgress, CompressionBusy) and issubclass(PruneRowStale, PruneRowUnresolved)
    assert issubclass(StoreQuarantined, CorruptStoreError) and issubclass(DeletedWalGeneration, StoreReplaced)
    assert issubclass(StoreContention, TransientStoreError) and issubclass(DiskFull, TransientStoreError)


def test_authority_refusal_is_the_one_refusal_name_with_bounded_one_line_detail():
    import persistence
    from persistence.core import errors
    assert not hasattr(persistence, "RuntimeStoreError") and not hasattr(errors, "RuntimeStoreError")
    refusal = AuthorityRefusal("invalid_params", field="a\nb", detail="x" * 500)
    assert (str(refusal), refusal.reason, refusal.field) == ("invalid_params", "invalid_params", "a b")
    assert len(refusal.detail) == 160
    assert AuthorityRefusal("x", field="\x00\x01").field is None
    stale = FenceStale(StaleFenceKind.STALE_GENERATION, "admission", "a1")
    assert (stale.reason, stale.kind, stale.entity, stale.entity_id) == ("stale_generation", StaleFenceKind.STALE_GENERATION, "admission", "a1")
    assert UnknownExecution("session", "s").reason == "unknown_execution"
    assert ProofRejected("worker_execution", "w").reason == "permission_denied"
    assert RetiredSessionError(SessionId("s")).reason == "session_deleted"
    assert RetiredSessionError(SessionId("s"), "not_found").reason == "not_found"
    for refusal in (stale, UnknownExecution("s", "s"), ProofRejected("s", "s"), RetiredSessionError(SessionId("s"))):
        assert isinstance(refusal, AuthorityRefusal) and isinstance(refusal, ValueError)


def test_store_failures_retry_store_errors_and_never_a_refusal():
    for failure in (StoreContention(), DiskFull(), StoreQuarantined(), StoreUnavailable(), DeletedWalGeneration()):
        assert isinstance(failure, STORE_FAILURES)
    for refusal in (AuthorityRefusal("session_busy"), TurnLeaseLost(SessionId("s")), NotFound("session", "s")):
        assert not isinstance(refusal, STORE_FAILURES)


def test_mutation_guard_refusals_map_to_session_busy_naming_a_held_compression():
    assert MUTATION_GUARD_REFUSALS == (TurnLeaseLost, CompressionInProgress)
    turn = busy_refusal(TurnLeaseLost(SessionId("s")))
    compressing = busy_refusal(CompressionInProgress(SessionId("s")))
    assert (turn.reason, turn.field) == ("session_busy", None)
    assert (compressing.reason, compressing.field) == ("session_busy", "compression")


# ── Fixture contract: real backend, two scopes, one provider instance ────────


def test_backend_declares_capabilities_honestly(backend, scope_a):
    assert isinstance(backend.capabilities, PersistenceCapabilities)
    # Earned flags require a registered repository executing the behavior: a
    # bare store (no search/coordination registered) and a store
    # whose search/coordination repositories are not registered must both
    # declare them False.
    store = backend.open(scope_a)
    assert store.capabilities == backend.capabilities
    unfolded = getattr(backend, "unfolded_capabilities", frozenset())
    if not hasattr(store, "search") or "search" in unfolded:
        assert backend.capabilities.full_text_search is False
        assert backend.capabilities.substring_search is False
    if not hasattr(store, "coordination") or "coordination" in unfolded:
        assert backend.capabilities.transactional_claims is False


def test_two_scopes_route_through_one_instance_and_stay_isolated(
    backend, scope_a, scope_b
):
    # The two scopes are two distinct profiles on every backend.
    assert scope_a != scope_b

    scope_a_twin = profile_scope(scope_a.name)

    # One backend instance serves both profiles through handles bound at open.
    meta_a = backend.open(scope_a).state_meta
    meta_b = backend.open(scope_b).state_meta

    # Colliding key, distinct values per profile: each visible in its own
    # profile, the other profile's value invisible, and a key written only
    # in profile A absent from profile B entirely.
    meta_a.set("smoke:shared-id", "sentinel-a")
    meta_a.set("smoke:only-in-a", "private-a")
    meta_b.set("smoke:shared-id", "sentinel-b")
    assert meta_a.get("smoke:shared-id") == "sentinel-a"
    assert meta_b.get("smoke:shared-id") == "sentinel-b"
    assert meta_b.get("smoke:only-in-a") is None

    # Writing profile B's value cannot reach profile A's.
    meta_b.set("smoke:shared-id", "sentinel-b2")
    assert meta_a.get("smoke:shared-id") == "sentinel-a"

    # A scope on the same folder under another label opens the same store.
    assert backend.open(scope_a_twin).state_meta.get("smoke:only-in-a") == "private-a"


def test_backend_factory_yields_fresh_state_and_tears_down(backend_factory, scope_a):
    with backend_factory() as first:
        first.open(scope_a).state_meta.set("smoke:marker", "left-over")
    with backend_factory() as second:
        assert second.open(scope_a).state_meta.get("smoke:marker") is None
    with pytest.raises(RuntimeError):
        second.open(scope_a)


@pytest.mark.design_new
def test_reopen_matches_declared_capability(backend, scope_a):
    # A backend declaring reopen_durability completes reopen() and stays usable. A non-declaring backend raises CapabilityUnsupported.
    if not backend.capabilities.reopen_durability:
        with pytest.raises(CapabilityUnsupported):
            backend.reopen()
        return
    backend.reopen()
    assert backend.capabilities.reopen_durability is True


def test_a_profile_opened_twice_is_one_store(backend, scope_a):
    # CONTRACT: a handle is bound to its profile when opened; a second open of the same profile (another label, same folder) sees the same store.
    from persistence import SessionId
    from persistence.conversation.sessions import NewSession

    created = backend.open(scope_a).sessions.create(NewSession(session_id=SessionId("eager-open"), source="cli"))
    relabelled = ProfileScope(profile_dir=scope_a.profile_dir, name="another-label")
    assert backend.open(relabelled).sessions.get(created) is not None
