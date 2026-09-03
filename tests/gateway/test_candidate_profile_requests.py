"""Safety contracts for inert, local specialist-candidate requests."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sqlite3

import pytest

from gateway.candidate_profile_requests import (
    CandidateProfileRequests,
    DEFAULT_POLICY_DIGEST,
    OpaqueEvidenceReference,
    SanitizedTaskEnvelope,
)
from gateway.capability_registry import CapabilityRegistry, CapabilitySignature, RegistryResolution
from gateway.specialist_handoff import HandoffSource, create_specialist_handoff
from gateway.specialist_routing import RouteKind, SpecialistRouteDecision
from hermes_cli import kanban_db as kb


NO_MATCH = CapabilitySignature(
    domain="market-data",
    actions=("audit", "read"),
    evidence_class="diagnostic-only",
    requested_permissions=("market-data:read",),
)
WRITE_CAPABILITY = CapabilitySignature(
    domain="market-data",
    actions=("read", "write"),
    evidence_class="diagnostic-only",
    requested_permissions=("market-data:read", "market-data:write"),
)
NO_MATCH_RESOLUTION = RegistryResolution(
    status="no_match", profile=None, reason="no active profile"
)


def _opaque_reference(label: str) -> OpaqueEvidenceReference:
    return OpaqueEvidenceReference(digest=hashlib.sha256(label.encode("utf-8")).hexdigest())


@pytest.fixture
def requests(tmp_path):
    return CandidateProfileRequests(db_path=tmp_path / "candidate-requests.db")


def test_no_match_creates_one_inert_candidate_and_orchestrator_fallback(requests):
    result = requests.open_or_reuse(
        NO_MATCH,
        source_key="discord:1",
        resolution=NO_MATCH_RESOLUTION,
        envelope=SanitizedTaskEnvelope(evidence_refs=(_opaque_reference("task-1"),)),
    )

    assert result.status == "candidate"
    assert result.profile_id is None


def test_forged_no_match_cannot_create_candidate_when_local_registry_matches(tmp_path):
    db_path = tmp_path / "candidate-requests.db"
    CapabilityRegistry(db_path=db_path).register_fixed_baseline(
        profile_id="market-data-authority-auditor", signature=NO_MATCH
    )
    requests = CandidateProfileRequests(db_path=db_path)
    forged = RegistryResolution(
        status="no_match", profile=None, reason="untrusted caller data"
    )

    result = requests.open_or_reuse(
        NO_MATCH,
        source_key="discord:forged-no-match",
        resolution=forged,
        envelope=SanitizedTaskEnvelope(evidence_refs=(_opaque_reference("forged-no-match"),)),
    )

    assert result.status == "rejected"
    assert result.request_id == ""
    with kb.connect_closing(db_path) as conn:
        rows = conn.execute("SELECT request_id FROM candidate_profile_requests").fetchall()
    assert rows == []


def test_duplicate_source_and_scope_reuses_candidate_request(requests):
    first = requests.open_or_reuse(
        NO_MATCH, source_key="discord:1", resolution=NO_MATCH_RESOLUTION
    )
    repeated = requests.open_or_reuse(
        NO_MATCH, source_key="discord:1", resolution=NO_MATCH_RESOLUTION
    )

    assert first.status == "candidate"
    assert repeated.status == "duplicate"
    assert repeated.request_id == first.request_id
    assert repeated.profile_id is None


def test_candidate_rejects_non_read_only_permission_delta(requests):
    result = requests.open_or_reuse(
        WRITE_CAPABILITY, source_key="discord:2", resolution=NO_MATCH_RESOLUTION
    )

    assert result.status == "rejected"
    assert result.profile_id is None
    assert "read-only" in result.reason


def test_rejected_candidate_enters_finite_cooldown(requests):
    first = requests.open_or_reuse(
        WRITE_CAPABILITY, source_key="discord:2", resolution=NO_MATCH_RESOLUTION
    )
    repeated = requests.open_or_reuse(
        WRITE_CAPABILITY, source_key="discord:2", resolution=NO_MATCH_RESOLUTION
    )

    assert first.status == "rejected"
    assert repeated.status == "cooldown"
    assert repeated.request_id == first.request_id


def test_corrected_envelope_cannot_bypass_latest_rejection_cooldown(tmp_path):
    now = [1_000]
    requests = CandidateProfileRequests(
        db_path=tmp_path / "candidate-requests.db", cooldown_seconds=10, clock=lambda: now[0]
    )
    rejected = requests.open_or_reuse(
        NO_MATCH,
        source_key="discord:fixed-state",
        resolution=NO_MATCH_RESOLUTION,
        envelope=SanitizedTaskEnvelope(evidence_refs=("api_key=not-safe",)),
    )
    corrected = requests.open_or_reuse(
        NO_MATCH,
        source_key="discord:fixed-state",
        resolution=NO_MATCH_RESOLUTION,
        envelope=SanitizedTaskEnvelope(evidence_refs=(_opaque_reference("corrected"),)),
    )

    assert rejected.status == "rejected"
    assert corrected.status == "cooldown"
    assert corrected.request_id == rejected.request_id


def test_later_terminal_state_supersedes_an_older_candidate(tmp_path):
    now = [1_000]
    requests = CandidateProfileRequests(
        db_path=tmp_path / "candidate-requests.db", cooldown_seconds=10, clock=lambda: now[0]
    )
    candidate = requests.open_or_reuse(
        NO_MATCH, source_key="discord:terminal-state", resolution=NO_MATCH_RESOLUTION
    )
    with kb.connect_closing(requests._db_path) as conn:
        request_hash = conn.execute(
            "SELECT request_hash FROM candidate_profile_requests WHERE request_id = ?",
            (candidate.request_id,),
        ).fetchone()["request_hash"]
        with kb.write_txn(conn):
            CandidateProfileRequests._insert(
                conn,
                request_hash=request_hash,
                signature=NO_MATCH,
                source_key="discord:terminal-state",
                policy_digest=DEFAULT_POLICY_DIGEST,
                evidence_ref_hashes=(),
                lifecycle_status="rejected",
                reason_code="test_terminal",
                cooldown_until=now[0] + 10,
                now=now[0],
            )

    retried = requests.open_or_reuse(
        NO_MATCH, source_key="discord:terminal-state", resolution=NO_MATCH_RESOLUTION
    )

    assert retried.status == "cooldown"


def test_rejected_candidate_can_be_reconsidered_after_its_bounded_cooldown(tmp_path):
    now = [1_000]
    requests = CandidateProfileRequests(
        db_path=tmp_path / "candidate-requests.db", cooldown_seconds=10, clock=lambda: now[0]
    )
    first = requests.open_or_reuse(
        WRITE_CAPABILITY, source_key="discord:2", resolution=NO_MATCH_RESOLUTION
    )
    now[0] += 11
    reconsidered = requests.open_or_reuse(
        WRITE_CAPABILITY, source_key="discord:2", resolution=NO_MATCH_RESOLUTION
    )

    assert first.status == "rejected"
    assert reconsidered.status == "rejected"
    assert reconsidered.request_id != first.request_id


@pytest.mark.parametrize(
    "permission",
    ("network:read", "credential:read", "task:concurrency", "operator:authority"),
)
def test_candidate_rejects_egress_credential_concurrency_and_authority_deltas(requests, permission):
    signature = CapabilitySignature(
        domain="diagnostics",
        actions=("read",),
        evidence_class="diagnostic-only",
        requested_permissions=(permission,),
    )

    result = requests.open_or_reuse(
        signature, source_key=f"discord:{permission}", resolution=NO_MATCH_RESOLUTION
    )

    assert result.status == "rejected"


@pytest.mark.parametrize("permission", (".env:read", "ssh-private-key:read", "s3:read", "tcp:read"))
def test_candidate_rejects_unknown_read_namespaces(requests, permission):
    signature = CapabilitySignature(
        domain="financial-analysis",
        actions=("read",),
        evidence_class="diagnostic-only",
        requested_permissions=(permission,),
    )

    assert requests.open_or_reuse(
        signature, source_key=f"discord:unknown:{permission}", resolution=NO_MATCH_RESOLUTION
    ).status == "rejected"


def test_candidate_allows_explicit_safe_financial_analysis_scope(requests):
    signature = CapabilitySignature(
        domain="financial-analysis",
        actions=("audit", "read"),
        evidence_class="diagnostic-only",
        requested_permissions=("financial-analysis:read",),
    )

    assert requests.open_or_reuse(
        signature, source_key="discord:financial-analysis", resolution=NO_MATCH_RESOLUTION
    ).status == "candidate"


def test_unsanitized_evidence_is_rejected_and_not_persisted(requests):
    result = requests.open_or_reuse(
        NO_MATCH,
        source_key="discord:3",
        resolution=NO_MATCH_RESOLUTION,
        envelope=SanitizedTaskEnvelope(evidence_refs=("api_key=not-safe",)),
    )

    assert result.status == "rejected"
    assert "sanitized" in result.reason
    with kb.connect_closing(requests._db_path) as conn:
        stored = conn.execute(
            "SELECT evidence_ref_hashes_json FROM candidate_profile_requests"
        ).fetchone()
    assert stored is not None
    assert "api_key" not in stored["evidence_ref_hashes_json"]


def test_candidate_ledger_never_persists_rejected_private_scope_or_plaintext_evidence(requests):
    unsafe_signature = CapabilitySignature(
        domain="/Users/alice/private-research",
        actions=("read",),
        evidence_class="diagnostic-only",
        requested_permissions=("ssh-private-key:read",),
    )
    requests.open_or_reuse(
        unsafe_signature,
        source_key="discord:Bearer-very-private-value",
        resolution=NO_MATCH_RESOLUTION,
        envelope=SanitizedTaskEnvelope(evidence_refs=("https://private.example/evidence",)),
    )

    with kb.connect_closing(requests._db_path) as conn:
        stored = conn.execute("SELECT * FROM candidate_profile_requests").fetchone()
    persisted = " ".join(str(stored[column]) for column in stored.keys())
    for forbidden in (
        "/Users/alice/private-research",
        "ssh-private-key",
        "https://private.example/evidence",
        "Bearer-very-private-value",
    ):
        assert forbidden not in persisted


def test_candidate_request_rows_are_append_only(requests):
    result = requests.open_or_reuse(
        NO_MATCH, source_key="discord:append-only", resolution=NO_MATCH_RESOLUTION
    )

    with kb.connect_closing(requests._db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE candidate_profile_requests SET lifecycle_status = 'active' WHERE request_id = ?",
                (result.request_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "DELETE FROM candidate_profile_requests WHERE request_id = ?", (result.request_id,)
            )


def test_candidate_request_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "candidate-requests.db"

    kb.init_db(db_path)
    kb.init_db(db_path)

    with kb.connect_closing(db_path) as conn:
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(candidate_profile_requests)")
        }
        triggers = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'candidate_profile_requests'"
            )
        }
    assert {"request_hash", "source_key_hash", "policy_digest", "lifecycle_status"} <= columns
    assert triggers == {
        "candidate_profile_requests_no_delete",
        "candidate_profile_requests_no_update",
    }


def test_legacy_plaintext_candidate_ledger_is_replaced_before_opening_requests(tmp_path):
    db_path = tmp_path / "legacy-candidate-requests.db"
    kb.init_db(db_path)
    with kb.connect_closing(db_path) as conn:
        conn.execute("DROP TABLE candidate_profile_requests")
        conn.execute(
            """
            CREATE TABLE candidate_profile_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL UNIQUE,
                request_hash TEXT NOT NULL,
                signature_hash TEXT NOT NULL,
                permissions_hash TEXT NOT NULL,
                source_key_hash TEXT NOT NULL,
                policy_digest TEXT NOT NULL,
                domain TEXT NOT NULL,
                actions_json TEXT NOT NULL,
                evidence_class TEXT NOT NULL,
                requested_permissions_json TEXT NOT NULL,
                evidence_refs_json TEXT NOT NULL,
                lifecycle_status TEXT NOT NULL,
                reason TEXT NOT NULL,
                cooldown_until INTEGER,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO candidate_profile_requests (
                request_id, request_hash, signature_hash, permissions_hash, source_key_hash,
                policy_digest, domain, actions_json, evidence_class, requested_permissions_json,
                evidence_refs_json, lifecycle_status, reason, cooldown_until, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-raw-request",
                "a" * 64,
                "b" * 64,
                "c" * 64,
                "d" * 64,
                "e" * 64,
                "/Users/alice/private-scope",
                '["read"]',
                "diagnostic-only",
                '["ssh-private-key:read"]',
                '["https://private.example/evidence"]',
                "candidate",
                "legacy raw reason",
                None,
                1,
            ),
        )

    kb.init_db(db_path)
    requests = CandidateProfileRequests(db_path=db_path)
    result = requests.open_or_reuse(
        NO_MATCH,
        source_key="discord:legacy-upgrade",
        resolution=NO_MATCH_RESOLUTION,
        envelope=SanitizedTaskEnvelope(evidence_refs=(_opaque_reference("legacy-upgrade"),)),
    )

    assert result.status == "candidate"
    kb.init_db(db_path)
    with kb.connect_closing(db_path) as conn:
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(candidate_profile_requests)")
        }
        rows = conn.execute("SELECT * FROM candidate_profile_requests").fetchall()
        indexes = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'candidate_profile_requests'"
            )
        }
        triggers = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'candidate_profile_requests'"
            )
        }
    assert {"evidence_ref_hashes_json", "reason_code"} <= columns
    assert not {"domain", "actions_json", "evidence_refs_json", "reason"} & columns
    assert "idx_candidate_profile_requests_hash_state" in indexes
    assert triggers == {
        "candidate_profile_requests_no_delete",
        "candidate_profile_requests_no_update",
    }
    persisted = " ".join(str(row[column]) for row in rows for column in row.keys())
    for raw_legacy_value in (
        "/Users/alice/private-scope",
        "ssh-private-key:read",
        "https://private.example/evidence",
        "legacy raw reason",
    ):
        assert raw_legacy_value not in persisted


def test_hybrid_candidate_ledger_is_replaced_before_opening_requests(tmp_path):
    db_path = tmp_path / "hybrid-candidate-requests.db"
    kb.init_db(db_path)
    with kb.connect_closing(db_path) as conn:
        conn.execute("ALTER TABLE candidate_profile_requests ADD COLUMN domain TEXT")
        conn.execute(
            """
            INSERT INTO candidate_profile_requests (
                request_id, request_hash, signature_hash, permissions_hash, source_key_hash,
                policy_digest, evidence_ref_hashes_json, lifecycle_status, reason_code,
                cooldown_until, created_at, domain
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "hybrid-raw-request",
                "a" * 64,
                "b" * 64,
                "c" * 64,
                "d" * 64,
                "e" * 64,
                '["' + "f" * 64 + '"]',
                "candidate",
                "missing_active_specialist",
                None,
                1,
                "/Users/alice/private-scope",
            ),
        )

    kb.init_db(db_path)
    requests = CandidateProfileRequests(db_path=db_path)
    result = requests.open_or_reuse(
        NO_MATCH,
        source_key="discord:hybrid-upgrade",
        resolution=NO_MATCH_RESOLUTION,
        envelope=SanitizedTaskEnvelope(evidence_refs=(_opaque_reference("hybrid-upgrade"),)),
    )

    assert result.status == "candidate"
    with kb.connect_closing(db_path) as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(candidate_profile_requests)")}
        rows = conn.execute("SELECT * FROM candidate_profile_requests").fetchall()
    assert "domain" not in columns
    assert "/Users/alice/private-scope" not in " ".join(
        str(row[column]) for row in rows for column in row.keys()
    )


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    for profile in ("task-orchestrator", "market-data-authority-auditor"):
        (home / "profiles" / profile).mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_no_match_handoff_uses_existing_orchestrator_and_preserves_source_idempotency(kanban_home):
    source = HandoffSource(
        platform="discord",
        chat_id="channel-1",
        chat_type="group",
        user_id="user-1",
        message_id="message-no-match",
    )
    decision = SpecialistRouteDecision(
        kind=RouteKind.SPECIALIST,
        profile="generated-market-data-candidate",
        confidence=1.0,
        reason="request requires a local capability",
        title="Audit market data",
    )
    kwargs = {
        "decision": decision,
        "source": source,
        "request": "Audit the supplied market-data evidence.",
        "signature": NO_MATCH,
        "registry": CapabilityRegistry(board=kb.DEFAULT_BOARD),
        "board": kb.DEFAULT_BOARD,
    }

    first = create_specialist_handoff(**kwargs)
    repeated = create_specialist_handoff(**kwargs)

    assert first.ok, first.reason
    assert repeated.ok, repeated.reason
    assert first.created is True
    assert repeated.created is False
    assert first.task_id == repeated.task_id
    with kb.connect() as conn:
        task = kb.get_task(conn, first.task_id)
        subscriptions = conn.execute(
            "SELECT COUNT(*) AS count FROM kanban_notify_subs WHERE task_id = ?", (first.task_id,)
        ).fetchone()
        candidate_rows = conn.execute(
            "SELECT request_hash, lifecycle_status FROM candidate_profile_requests"
        ).fetchall()
    assert task is not None
    assert task.assignee == "task-orchestrator"
    assert json.loads(task.body)["candidate_request_id"] == first.candidate_request_id
    assert subscriptions["count"] == 1
    assert len(candidate_rows) == 1
    assert candidate_rows[0]["lifecycle_status"] == "candidate"
