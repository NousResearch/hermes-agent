from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

import pytest

from gateway import canonical_writer_foundation as foundation
from gateway.canonical_writer_db import (
    ManagedCloudSQLAdminHBAReceipt,
    QueryResult,
)


REVISION = "a" * 40
ADMIN = "muncho_canary_admin_1111111111111111"
TLS_PEER = "b" * 64
NOW = 2_000_000_000
SECRET = b"A" * 64
LEGACY_ARCHIVE_OWNER = "legacy_archive_source_owner"


def _credential(state: str) -> dict[str, object]:
    absent = {
        "state": "absent",
        "path": str(foundation.DATABASE_CREDENTIAL_PATH),
        "stage_path": None,
        "device": None,
        "inode": None,
        "owner_uid": None,
        "group_gid": None,
        "mode": None,
        "link_count": None,
        "modification_time_ns": None,
        "change_time_ns": None,
        "content_or_digest_recorded": False,
    }
    if state == "absent":
        return absent
    value = {
        **absent,
        "state": state,
        "device": 11,
        "inode": 12,
        "owner_uid": foundation.WRITER_UID,
        "group_gid": foundation.WRITER_GID,
        "mode": "0400",
        "link_count": 1,
        "modification_time_ns": 13,
        "change_time_ns": 14,
    }
    if state in {"allocated", "staged"}:
        value["stage_path"] = str(
            foundation.DATABASE_CREDENTIAL_PATH.with_name(
                ".canonical-writer-db-password.foundation." + "f" * 24
            )
        )
    return value


class _SecretStore:
    def __init__(self, state: str = "absent") -> None:
        self.state = state

    def observe(self, _plan_sha256: str | None):
        return _credential(self.state)

    def allocate(self, _plan_sha256: str):
        self.state = "allocated"
        return _credential(self.state)

    def materialize(self, _plan_sha256, _expected, factory):
        if self.state == "allocated":
            assert factory() == SECRET
        self.state = "staged"
        return bytearray(SECRET), _credential(self.state)

    def publish(self, _plan_sha256, _expected):
        assert self.state == "staged"
        self.state = "installed"
        return _credential(self.state)

    def read(self, _expected):
        assert self.state == "installed"
        return bytearray(SECRET)

    def remove(self, _expected):
        assert self.state == "installed"
        self.state = "absent"


def _roles(*, login: bool) -> list[dict[str, object]]:
    rows = []
    for name, (expected_login, inherits) in foundation._EXPECTED_ROLE_SHAPES.items():
        rows.append(
            {
                "name": name,
                "can_login": login if name == foundation.SQL_USER else expected_login,
                "inherits": inherits,
                "superuser": False,
                "create_database": False,
                "create_role": False,
                "replication": False,
                "bypass_row_security": False,
                "connection_limit": -1,
                "validity_is_unbounded": True,
                "configuration_is_empty": True,
            }
        )
    return rows


def _memberships(*, writer: bool) -> list[dict[str, object]]:
    pairs = [
        (
            foundation.CANARY_BOOTSTRAP_ROLE,
            foundation.CANARY_BOOTSTRAP_LOGIN,
        )
    ]
    if writer:
        pairs.append((foundation.WRITER_ROLE, foundation.SQL_USER))
    return [
        {
            "granted_role": granted,
            "member_role": member,
            "grantor": foundation.PERSISTENT_MEMBERSHIP_GRANTOR,
            "admin_option": False,
            "inherit_option": True,
            "set_option": False,
        }
        for granted, member in pairs
    ]


def _observation(
    store: _SecretStore,
    *,
    login: bool,
    ping: bool,
    writer: bool,
    session_user: str = ADMIN,
    legacy_archive_present: bool = False,
    legacy_archive_oid: str = "424242",
    legacy_archive_owner: str = LEGACY_ARCHIVE_OWNER,
) -> foundation.FoundationObservation:
    unsigned = {
        "schema": foundation.FOUNDATION_OBSERVATION_SCHEMA,
        "database": foundation.SQL_DATABASE,
        "database_owner": foundation.DATABASE_OWNER_ROLE,
        "postgres_version_num": 180004,
        "session_user": session_user,
        "current_user": session_user,
        "temporary_admin_roles": (
            [] if session_user == foundation.SQL_USER else [session_user]
        ),
        "tls_peer_certificate_sha256": TLS_PEER,
        "roles": _roles(login=login),
        "memberships": _memberships(writer=writer),
        "event_log_shape": "canonical14",
        "event_log_owner": foundation.MIGRATION_OWNER_ROLE,
        "legacy_archive_present": legacy_archive_present,
        "legacy_archive_identity": (
            {
                "oid": legacy_archive_oid,
                "owner": legacy_archive_owner,
                "relation_kind": "r",
                "persistence": "p",
                "owner_superuser": False,
                "owner_create_database": False,
                "owner_create_role": False,
                "owner_replication": False,
                "owner_bypass_row_security": False,
                "owner_connection_limit": -1,
                "owner_validity_is_unbounded": True,
                "owner_configuration_is_empty": True,
            }
            if legacy_archive_present
            else None
        ),
        "canonical_schema_owner": foundation.MIGRATION_OWNER_ROLE if ping else None,
        "writer_ping_present": ping,
        "database_acl": [],
        "public_schema_acl": [],
        "legacy_truth": None,
        "credential": dict(store.observe(None)),
    }
    return foundation.FoundationObservation.from_mapping(
        {
            **unsigned,
            "observation_sha256": foundation._sha256_json(unsigned),
        }
    )


class _AdminSession:
    tls_peer_certificate_sha256 = TLS_PEER

    def __init__(
        self,
        store: _SecretStore,
        artifacts,
        *,
        username: str = ADMIN,
        legacy_archive_present: bool = False,
        legacy_archive_oid: str = "424242",
        legacy_archive_owner: str = LEGACY_ARCHIVE_OWNER,
    ) -> None:
        self.store = store
        self.artifacts = artifacts
        self.username = username
        self.legacy_archive_present = legacy_archive_present
        self.legacy_archive_oid = legacy_archive_oid
        self.legacy_archive_owner = legacy_archive_owner
        self.login = False
        self.ping = False
        self.writer = False
        self.mutations: list[str] = []
        self.setting_sql: list[str] = []
        self.password_statements = 0

    def _db_observation(self) -> dict[str, object]:
        value = _observation(
            self.store,
            login=self.login,
            ping=self.ping,
            writer=self.writer,
            session_user=self.username,
            legacy_archive_present=self.legacy_archive_present,
            legacy_archive_oid=self.legacy_archive_oid,
            legacy_archive_owner=self.legacy_archive_owner,
        ).to_mapping()
        for key in (
            "schema",
            "tls_peer_certificate_sha256",
            "legacy_truth",
            "credential",
            "observation_sha256",
        ):
            value.pop(key)
        return value

    def query(self, sql: str, *, maximum_rows: int):
        if "AS foundation_observation" in sql:
            assert maximum_rows == 1
            return QueryResult(
                ("foundation_observation",),
                ((json.dumps(self._db_observation()),),),
                "SELECT 1",
            )
        if sql.startswith("SET "):
            self.setting_sql.append(sql)
            return QueryResult((), (), "SET")
        if sql.startswith("ROLLBACK;"):
            return QueryResult((), (), "RESET")
        for name, artifact in self.artifacts.items():
            if sql == artifact.payload.decode("utf-8"):
                self.mutations.append(name)
                if name == "base_migration":
                    self.ping = True
                elif name == "writer_membership":
                    self.writer = True
                return QueryResult((), (), "COMMIT")
        raise AssertionError("unexpected SQL")

    def execute_password_copy(self, boundary, *, password):
        assert boundary is foundation.WRITER_PASSWORD_BOUNDARY
        assert isinstance(password, bytearray)
        assert bytes(password) == SECRET
        self.password_statements += 1
        self.login = True
        return {
            "boundary": boundary.name,
            "database": foundation.SQL_DATABASE,
            "role": foundation.SQL_USER,
            "password_encryption": "scram-sha-256",
            "login_enabled": True,
            "secret_transport": "postgres-copy-data-v1",
            "copy_data_completed": True,
            "temporary_security_definer_removed": True,
            "temporary_admin_delete_required": True,
            "secret_material_recorded": False,
        }


def _hba() -> ManagedCloudSQLAdminHBAReceipt:
    return ManagedCloudSQLAdminHBAReceipt(
        version="managed-cloudsqladmin-hba-rejection-v2",
        host=foundation.SQL_HOST,
        tls_server_name=foundation.SQL_TLS_SERVER_NAME,
        port=foundation.SQL_PORT,
        server_certificate_sha256=TLS_PEER,
        database="cloudsqladmin",
        user=foundation.SQL_USER,
        observed_at_unix=NOW,
        expires_at_unix=NOW + 300,
        sqlstate="28000",
        server_message=(
            'no pg_hba.conf entry for host "10.0.0.8", user '
            f'"{foundation.SQL_USER}", database "cloudsqladmin", SSL encryption'
        ),
        result="pg_hba_rejected",
        tls_peer_verified=True,
    )


def _attestation(hba: ManagedCloudSQLAdminHBAReceipt | None = None):
    receipt = _hba() if hba is None else hba
    return {
        "catalog_sha256": foundation.PRODUCTION_CATALOG_SHA256,
        "privilege_attestation_sha256": "c" * 64,
        "private_schema_identity_sha256": "d" * 64,
        "managed_hba_receipt_sha256": receipt.sha256,
        "writer_ping_ok": True,
        "writer_ping_service": "canonical_writer",
        "writer_ping_protocol": "v1",
        "table_grants": [],
        "general_sql_available": False,
        "cross_database_connect_available": False,
        "dangerous_role_attributes": [],
        "migration_admin_membership_present": False,
    }


def _auth(session: _AdminSession):
    def probe(enabled: bool):
        if enabled and session.login:
            return {"authenticated": True, "user": foundation.SQL_USER}
        return {
            "authenticated": False,
            "user": foundation.SQL_USER,
            "sqlstate": "28000",
        }

    return probe


def _phase_b_plan_for_journal_tests(
    observation: foundation.FoundationObservation,
    artifacts,
    *,
    revision: str = REVISION,
) -> foundation.FoundationPlan:
    artifact_sha = foundation._artifact_digest_mapping(artifacts)
    unsigned = {
        "schema": foundation.FOUNDATION_PLAN_SCHEMA,
        "mode": "create_pristine",
        "release_revision": revision,
        "target": dict(foundation._fixed_target()),
        "initial_observation": observation.to_mapping(),
        "initial_observation_sha256": observation.sha256,
        "artifact_sha256": artifact_sha,
        "artifact_set_sha256": foundation._sha256_json(artifact_sha),
        "states": list(foundation.FOUNDATION_STATES),
        "legacy_reconciliation_required": False,
        "legacy_source_owner": None,
        "credential_contract": {
            "generated_on_vm": True,
            "bytes": 64,
            "alphabet": "base64url-no-padding",
            "owner_uid": foundation.WRITER_UID,
            "group_gid": foundation.WRITER_GID,
            "mode": "0400",
            "scram_mechanism": "SCRAM-SHA-256",
            "password_transport": "postgres-copy-data-v1",
            "server_generated_scram_salt": True,
            "copy_data_only": True,
            "statement_logging_safe_without_privileged_guc": True,
            "temporary_security_definer": True,
            "password_or_verifier_serialized": False,
            "content_or_digest_recorded": False,
        },
        "terminal_attestation_catalog_sha256": (
            foundation.PRODUCTION_CATALOG_SHA256
        ),
        "adoption_terminal_attestation": None,
    }
    return foundation.FoundationPlan.from_mapping(
        {**unsigned, "plan_sha256": foundation._sha256_json(unsigned)}
    )


def _creation_fixture(tmp_path: Path):
    artifacts = foundation._load_source_artifacts_for_tests()
    store = _SecretStore()
    session = _AdminSession(store, artifacts)
    observed = _observation(store, login=False, ping=False, writer=False)
    plan = _phase_b_plan_for_journal_tests(observed, artifacts)
    journal = foundation._AppendOnlyFoundationJournal(
        tmp_path / "evidence",
        strict_root=False,
    )
    return artifacts, store, session, plan, journal


def test_phase_a_creation_is_rejected_before_every_mutation(
    tmp_path: Path,
):
    artifacts, store, session, plan, journal = _creation_fixture(tmp_path)
    with pytest.raises(
        foundation.CanonicalWriterFoundationError,
        match="foundation_creation_requires_admin_delete_integration",
    ):
        foundation.build_foundation_plan(
            REVISION,
            plan.initial_observation,
            _artifacts=artifacts,
        )
    with pytest.raises(
        foundation.CanonicalWriterFoundationError,
        match="foundation_creation_requires_admin_delete_integration",
    ):
        foundation.apply_approved_foundation(
            plan,
            approved_plan_sha256=plan.sha256,
            admin_session=session,
            _journal=journal,
            _secret_store=store,
            _artifacts=artifacts,
            _hba_collector=_hba,
            _authentication_probe=_auth(session),
            _terminal_attestor=_attestation,
            _secret_factory=lambda: SECRET,
            _clock=lambda: NOW,
        )
    writer_session = _AdminSession(
        store,
        artifacts,
        username=foundation.SQL_USER,
    )
    with pytest.raises(
        foundation.CanonicalWriterFoundationError,
        match="foundation_creation_requires_admin_delete_integration",
    ):
        foundation.apply_approved_foundation(
            plan,
            approved_plan_sha256=plan.sha256,
            admin_session=writer_session,
            _journal=journal,
            _secret_store=store,
            _artifacts=artifacts,
        )
    assert session.mutations == []
    assert session.password_statements == 0
    assert store.state == "absent"
    assert not journal.root.exists()


def test_exact_existing_terminal_is_adopted_without_mutation_or_rotation(
    tmp_path: Path,
):
    artifacts = foundation._load_source_artifacts_for_tests()
    store = _SecretStore("installed")
    session = _AdminSession(
        store,
        artifacts,
        username=foundation.SQL_USER,
        legacy_archive_present=True,
    )
    session.login = session.ping = session.writer = True
    observed = _observation(
        store,
        login=True,
        ping=True,
        writer=True,
        session_user=foundation.SQL_USER,
        legacy_archive_present=True,
    )
    plan = foundation.build_foundation_adoption_plan(
        REVISION,
        observed,
        _artifacts=artifacts,
        _hba_collector=_hba,
        _authentication_probe=_auth(session),
        _terminal_attestor=_attestation,
        _clock=lambda: NOW,
    )
    journal = foundation._AppendOnlyFoundationJournal(
        tmp_path / "evidence",
        strict_root=False,
    )
    receipt = foundation.apply_approved_foundation(
        plan,
        approved_plan_sha256=plan.sha256,
        admin_session=session,
        _journal=journal,
        _secret_store=store,
        _artifacts=artifacts,
        _hba_collector=_hba,
        _authentication_probe=_auth(session),
        _terminal_attestor=_attestation,
        _clock=lambda: NOW,
    )
    assert receipt["ok"] is True
    assert receipt["mode"] == "adopted_existing_terminal"
    assert receipt["retirement_covered"] is False
    assert session.mutations == []
    assert session.legacy_archive_present is True
    assert store.state == "installed"
    assert [entry.state for entry in journal.load(plan) if not entry.prepared] == [
        "adopted_existing_terminal",
        "terminal",
    ]


def test_legacy_archive_disappearance_blocks_terminal_receipt(tmp_path: Path):
    artifacts = foundation._load_source_artifacts_for_tests()
    store = _SecretStore("installed")
    session = _AdminSession(
        store,
        artifacts,
        username=foundation.SQL_USER,
        legacy_archive_present=True,
    )
    session.login = session.ping = session.writer = True
    observed = _observation(
        store,
        login=True,
        ping=True,
        writer=True,
        session_user=foundation.SQL_USER,
        legacy_archive_present=True,
    )
    plan = foundation.build_foundation_adoption_plan(
        REVISION,
        observed,
        _artifacts=artifacts,
        _hba_collector=_hba,
        _authentication_probe=_auth(session),
        _terminal_attestor=_attestation,
        _clock=lambda: NOW,
    )
    journal = foundation._AppendOnlyFoundationJournal(
        tmp_path / "archive-drift-evidence",
        strict_root=False,
    )

    def remove_archive_after_initial_observation(state: str, point: str) -> None:
        if state == "adopted_existing_terminal" and point == "after_transition":
            session.legacy_archive_present = False

    with pytest.raises(
        foundation.CanonicalWriterFoundationError,
        match="foundation_legacy_archive_changed_after_approval",
    ):
        foundation.apply_approved_foundation(
            plan,
            approved_plan_sha256=plan.sha256,
            admin_session=session,
            _journal=journal,
            _secret_store=store,
            _artifacts=artifacts,
            _hba_collector=_hba,
            _authentication_probe=_auth(session),
            _terminal_attestor=_attestation,
            _clock=lambda: NOW,
            _fault_injector=remove_archive_after_initial_observation,
        )
    assert session.mutations == []
    assert not journal._terminal_root(plan).exists()


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    [
        ("legacy_archive_oid", "424243"),
        ("legacy_archive_owner", "legacy_archive_source_owner_2"),
    ],
)
def test_legacy_archive_identity_change_blocks_terminal_receipt(
    tmp_path: Path,
    attribute: str,
    replacement: str,
):
    artifacts = foundation._load_source_artifacts_for_tests()
    store = _SecretStore("installed")
    session = _AdminSession(
        store,
        artifacts,
        username=foundation.SQL_USER,
        legacy_archive_present=True,
        legacy_archive_oid="424242",
    )
    session.login = session.ping = session.writer = True
    observed = _observation(
        store,
        login=True,
        ping=True,
        writer=True,
        session_user=foundation.SQL_USER,
        legacy_archive_present=True,
        legacy_archive_oid="424242",
    )
    plan = foundation.build_foundation_adoption_plan(
        REVISION,
        observed,
        _artifacts=artifacts,
        _hba_collector=_hba,
        _authentication_probe=_auth(session),
        _terminal_attestor=_attestation,
        _clock=lambda: NOW,
    )
    journal = foundation._AppendOnlyFoundationJournal(
        tmp_path / "archive-replacement-evidence",
        strict_root=False,
    )

    def replace_archive_after_initial_observation(state: str, point: str) -> None:
        if state == "adopted_existing_terminal" and point == "after_transition":
            setattr(session, attribute, replacement)

    with pytest.raises(
        foundation.CanonicalWriterFoundationError,
        match="foundation_legacy_archive_changed_after_approval",
    ):
        foundation.apply_approved_foundation(
            plan,
            approved_plan_sha256=plan.sha256,
            admin_session=session,
            _journal=journal,
            _secret_store=store,
            _artifacts=artifacts,
            _hba_collector=_hba,
            _authentication_probe=_auth(session),
            _terminal_attestor=_attestation,
            _clock=lambda: NOW,
            _fault_injector=replace_archive_after_initial_observation,
        )
    assert session.mutations == []
    assert not journal._terminal_root(plan).exists()


def test_temporary_admin_cannot_author_terminal_adoption():
    artifacts = foundation._load_source_artifacts_for_tests()
    store = _SecretStore("installed")
    observed = _observation(store, login=True, ping=True, writer=True)
    with pytest.raises(
        foundation.CanonicalWriterFoundationError,
        match="foundation_existing_authority_not_exact_terminal",
    ):
        foundation.build_foundation_adoption_plan(
            REVISION,
            observed,
            _artifacts=artifacts,
            _hba_collector=_hba,
            _authentication_probe=lambda _enabled: {
                "authenticated": True,
                "user": foundation.SQL_USER,
            },
            _terminal_attestor=_attestation,
            _clock=lambda: NOW,
        )


@pytest.mark.parametrize(
    ("login", "ping", "writer", "credential_state"),
    [
        (True, False, False, "installed"),
        (True, True, False, "installed"),
        (True, True, True, "absent"),
    ],
)
def test_partial_existing_authority_cannot_be_created_or_adopted(
    login: bool,
    ping: bool,
    writer: bool,
    credential_state: str,
):
    artifacts = foundation._load_source_artifacts_for_tests()
    store = _SecretStore(credential_state)
    observed = _observation(store, login=login, ping=ping, writer=writer)
    with pytest.raises(foundation.CanonicalWriterFoundationError):
        foundation.build_foundation_plan(REVISION, observed, _artifacts=artifacts)
    with pytest.raises(
        foundation.CanonicalWriterFoundationError,
        match="foundation_existing_authority_not_exact_terminal",
    ):
        foundation.build_foundation_adoption_plan(
            REVISION,
            observed,
            _artifacts=artifacts,
            _hba_collector=_hba,
            _authentication_probe=lambda _enabled: {
                "authenticated": True,
                "user": foundation.SQL_USER,
            },
            _terminal_attestor=_attestation,
            _clock=lambda: NOW,
        )


def _rebuild_observation(
    observation: foundation.FoundationObservation,
    **changes: object,
) -> foundation.FoundationObservation:
    unsigned = observation.to_mapping()
    unsigned.pop("observation_sha256")
    unsigned.update(changes)
    return foundation.FoundationObservation.from_mapping(
        {
            **unsigned,
            "observation_sha256": foundation._sha256_json(unsigned),
        }
    )


@pytest.mark.parametrize("grantor", [ADMIN, "postgres"])
def test_persistent_membership_foreign_grantor_fails_closed(grantor: str):
    store = _SecretStore("installed")
    observed = _observation(
        store,
        login=True,
        ping=True,
        writer=True,
        session_user=foundation.SQL_USER,
    )
    memberships = [dict(row) for row in observed.value["memberships"]]
    memberships[0]["grantor"] = grantor
    with pytest.raises(
        foundation.CanonicalWriterFoundationError,
        match="foundation_membership_authority_drifted",
    ):
        _rebuild_observation(observed, memberships=memberships)


def test_persistent_membership_set_role_authority_fails_closed():
    store = _SecretStore("installed")
    observed = _observation(
        store,
        login=True,
        ping=True,
        writer=True,
        session_user=foundation.SQL_USER,
    )
    memberships = [dict(row) for row in observed.value["memberships"]]
    memberships[1]["set_option"] = True
    with pytest.raises(
        foundation.CanonicalWriterFoundationError,
        match="foundation_membership_authority_drifted",
    ):
        _rebuild_observation(observed, memberships=memberships)


def test_wrong_database_owner_fails_closed():
    store = _SecretStore("installed")
    observed = _observation(
        store,
        login=True,
        ping=True,
        writer=True,
        session_user=foundation.SQL_USER,
    )
    with pytest.raises(
        foundation.CanonicalWriterFoundationError,
        match="foundation_database_owner_drifted",
    ):
        _rebuild_observation(observed, database_owner=ADMIN)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("oid", "0"),
        ("owner", "invalid.owner"),
        ("owner", ADMIN),
        ("owner", "postgres"),
        ("owner", "cloudsqladmin"),
        ("owner", "cloudsqlsuperuser"),
        ("owner", foundation.MIGRATION_OWNER_ROLE),
        ("owner", foundation.WRITER_ROLE),
        ("owner", foundation.CANARY_BOOTSTRAP_ROLE),
        ("owner", foundation.CANARY_BOOTSTRAP_LOGIN),
        ("owner", foundation.SQL_USER),
        ("relation_kind", "v"),
        ("persistence", "t"),
        ("owner_superuser", True),
        ("owner_create_database", True),
        ("owner_create_role", True),
        ("owner_replication", True),
        ("owner_bypass_row_security", True),
        ("owner_connection_limit", 0),
        ("owner_validity_is_unbounded", False),
        ("owner_configuration_is_empty", False),
    ],
)
def test_invalid_legacy_archive_identity_fails_closed(field: str, value: object):
    store = _SecretStore("installed")
    observed = _observation(
        store,
        login=True,
        ping=True,
        writer=True,
        session_user=foundation.SQL_USER,
        legacy_archive_present=True,
    )
    archive_identity = dict(observed.value["legacy_archive_identity"])
    archive_identity[field] = value
    with pytest.raises(
        foundation.CanonicalWriterFoundationError,
        match="foundation_legacy_archive_identity_invalid",
    ):
        _rebuild_observation(
            observed,
            legacy_archive_identity=archive_identity,
        )


def test_admissible_durable_legacy_source_owner_is_accepted():
    store = _SecretStore("installed")
    observed = _observation(
        store,
        login=True,
        ping=True,
        writer=True,
        session_user=foundation.SQL_USER,
        legacy_archive_present=True,
        legacy_archive_owner="durable-source_owner-1",
    )
    assert observed.value["legacy_archive_identity"]["owner"] == (
        "durable-source_owner-1"
    )


def test_permission_sensitive_writer_ping_lookup_drift_fails_closed():
    store = _SecretStore("installed")
    observed = _observation(
        store,
        login=True,
        ping=True,
        writer=True,
        session_user=foundation.SQL_USER,
    )
    with pytest.raises(
        foundation.CanonicalWriterFoundationError,
        match="foundation_writer_observation_not_terminal",
    ):
        _rebuild_observation(observed, writer_ping_present=False)


def test_writer_terminal_rejects_any_surviving_temporary_admin_role():
    store = _SecretStore("installed")
    observed = _observation(
        store,
        login=True,
        ping=True,
        writer=True,
        session_user=foundation.SQL_USER,
    )
    with pytest.raises(
        foundation.CanonicalWriterFoundationError,
        match="foundation_temporary_admin_authority_drifted",
    ):
        _rebuild_observation(observed, temporary_admin_roles=[ADMIN])


def test_observation_sql_uses_permission_independent_catalog_identity():
    sql = foundation._load_source_artifacts_for_tests()["observe"].payload.decode()
    for permission_sensitive_lookup in (
        "to_regclass",
        "::regclass",
        "to_regprocedure",
        "::regprocedure",
    ):
        assert permission_sensitive_lookup not in sql
    assert "pg_catalog.pg_namespace" in sql
    assert "pg_catalog.pg_class" in sql
    assert "pg_catalog.pg_proc" in sql


class _InjectedCrash(RuntimeError):
    pass


@pytest.mark.parametrize(
    "point",
    [
        "after_mkdir",
        "after_write",
        "after_fsync",
        "after_publish",
        "after_publish_fsync",
        "after_unlink",
    ],
)
def test_journal_publication_recovers_each_durable_boundary(
    tmp_path: Path,
    point: str,
):
    _artifacts, _store, _session, plan, _journal = _creation_fixture(tmp_path)
    fired = False

    def fail_once(kind: str, observed_point: str):
        nonlocal fired
        if not fired and kind == "journal" and observed_point == point:
            fired = True
            raise _InjectedCrash(point)

    root = tmp_path / "crash-evidence"
    crashing = foundation._AppendOnlyFoundationJournal(
        root,
        strict_root=False,
        publication_fault_injector=fail_once,
    )
    kwargs = {
        "state": "intent",
        "transition_phase": "complete",
        "transition_to": None,
        "database_observation_sha256": plan.initial_observation.sha256,
        "credential": plan.initial_observation.value["credential"],
        "scram_salt_base64": None,
        "managed_hba_receipt_sha256": None,
        "terminal_attestation": None,
        "now_unix": NOW,
    }
    with pytest.raises(_InjectedCrash):
        crashing.append(plan, **kwargs)
    recovered = foundation._AppendOnlyFoundationJournal(root, strict_root=False)
    entries = recovered.load(plan)
    if not entries:
        recovered.append(plan, **{**kwargs, "now_unix": NOW + 1})
    assert len(recovered.load(plan)) == 1
    staging = recovered._staging_root(plan)
    assert not list(staging.glob("journal-*.stage"))


def _adoption_journal(tmp_path: Path):
    artifacts = foundation._load_source_artifacts_for_tests()
    store = _SecretStore("installed")
    observed = _observation(
        store,
        login=True,
        ping=True,
        writer=True,
        session_user=foundation.SQL_USER,
    )
    plan = foundation.build_foundation_adoption_plan(
        REVISION,
        observed,
        _artifacts=artifacts,
        _hba_collector=_hba,
        _authentication_probe=lambda _enabled: {
            "authenticated": True,
            "user": foundation.SQL_USER,
        },
        _terminal_attestor=_attestation,
        _clock=lambda: NOW,
    )
    journal = foundation._AppendOnlyFoundationJournal(
        tmp_path / "terminal-evidence",
        strict_root=False,
    )
    common = {
        "database_observation_sha256": observed.sha256,
        "credential": observed.value["credential"],
        "scram_salt_base64": None,
        "managed_hba_receipt_sha256": _hba().sha256,
        "terminal_attestation": _attestation(),
        "now_unix": NOW,
    }
    journal.append(
        plan,
        state="adopted_existing_terminal",
        transition_phase="complete",
        transition_to=None,
        **common,
    )
    journal.append(
        plan,
        state="adopted_existing_terminal",
        transition_phase="prepared",
        transition_to="terminal",
        **common,
    )
    terminal = journal.append(
        plan,
        state="terminal",
        transition_phase="complete",
        transition_to=None,
        **common,
    )
    return plan, terminal, journal, observed


@pytest.mark.parametrize(
    "point",
    [
        "after_mkdir",
        "after_write",
        "after_fsync",
        "after_publish",
        "after_publish_fsync",
        "after_unlink",
    ],
)
def test_terminal_receipt_publication_recovers_each_durable_boundary(
    tmp_path: Path,
    point: str,
):
    plan, terminal, journal, observed = _adoption_journal(tmp_path)
    fired = False

    def fail_once(kind: str, observed_point: str):
        nonlocal fired
        if not fired and kind == "terminal" and observed_point == point:
            fired = True
            raise _InjectedCrash(point)

    crashing = foundation._AppendOnlyFoundationJournal(
        journal.root,
        strict_root=False,
        publication_fault_injector=fail_once,
    )
    with pytest.raises(_InjectedCrash):
        crashing.publish_terminal_receipt(
            plan,
            terminal,
            observation=observed,
            terminal_attestation=_attestation(),
            hba_receipt=_hba(),
            now_unix=NOW,
        )
    receipt = journal.publish_terminal_receipt(
        plan,
        terminal,
        observation=observed,
        terminal_attestation=_attestation(),
        hba_receipt=_hba(),
        now_unix=NOW + 1,
    )
    assert receipt["terminal_at_unix"] == NOW
    terminal_files = tuple(journal._terminal_root(plan).iterdir())
    assert 1 <= len(terminal_files) <= 2
    assert (
        journal._terminal_root(plan) / f"{receipt['receipt_sha256']}.json"
    ).exists()
    assert not list(journal._staging_root(plan).glob("terminal-*.stage"))


def test_partial_or_tampered_final_journal_blocks(tmp_path: Path):
    _artifacts, _store, _session, plan, journal = _creation_fixture(tmp_path)
    root = journal._journal_root(plan)
    root.mkdir(mode=0o700, parents=True)
    bad = root / ("00000000-" + "0" * 64 + ".json")
    bad.write_bytes(b"{")
    bad.chmod(0o400)
    with pytest.raises(foundation.CanonicalWriterFoundationError):
        journal.load(plan)


def test_cross_plan_evidence_residue_blocks(tmp_path: Path):
    artifacts, _store, _session, plan, journal = _creation_fixture(tmp_path)
    with journal.lock():
        journal.assert_no_cross_plan(plan)
        journal.append(
            plan,
            state="intent",
            transition_phase="complete",
            transition_to=None,
            database_observation_sha256=plan.initial_observation.sha256,
            credential=plan.initial_observation.value["credential"],
            scram_salt_base64=None,
            managed_hba_receipt_sha256=None,
            terminal_attestation=None,
            now_unix=NOW,
        )
    other = _phase_b_plan_for_journal_tests(
        plan.initial_observation,
        artifacts,
        revision="e" * 40,
    )
    with journal.lock(), pytest.raises(
        foundation.CanonicalWriterFoundationError,
        match="foundation_cross_plan_residue",
    ):
        journal.assert_no_cross_plan(other)


def test_foundation_has_no_runtime_target_or_production_scope_selector():
    source = Path(foundation.__file__).read_text(encoding="utf-8")
    sql = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(
            (Path(foundation.__file__).parents[1] / "scripts" / "sql").glob(
                "canonical_writer_foundation_*_v1.sql"
            )
        )
    )
    assert "owner_approved_cutover" not in source
    assert "owner_approved_cutover" not in sql
    assert "os.environ" not in source
    assert "os.getenv" not in source


def _sanitized_subprocess(
    command: list[str],
    *,
    stdin: bytes = b"",
    accepted_returncodes: frozenset[int] = frozenset({0}),
    redactions: tuple[bytes, ...] = (),
) -> bytes:
    completed = subprocess.run(
        command,
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout
    if completed.returncode not in accepted_returncodes:
        sanitized = output
        for value in redactions:
            sanitized = sanitized.replace(value, b"<redacted>")
        sanitized = re.sub(
            rb"SCRAM-SHA-256\$[^\s'\"]+",
            b"<redacted-scram-verifier>",
            sanitized,
        )
        pytest.fail(
            "sanitized subprocess failed with exit "
            f"{completed.returncode}: "
            + sanitized[-4000:].decode("utf-8", errors="replace")
        )
    return output


def _docker_psql(
    container: str,
    sql: bytes,
    *,
    user: str = "postgres",
    database: str = "muncho_canary_brain",
    password: str | None = None,
    tuples_only: bool = False,
    accepted_returncodes: frozenset[int] = frozenset({0}),
    redactions: tuple[bytes, ...] = (),
) -> bytes:
    command = ["docker", "exec"]
    if password is not None:
        command.extend(["-e", f"PGPASSWORD={password}"])
    command.extend(
        [
            "-i",
            container,
            "psql",
            "-X",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            user,
            "-d",
            database,
        ]
    )
    if password is not None:
        command.extend(["-h", "127.0.0.1"])
    if tuples_only:
        command.append("-qAt")
    return _sanitized_subprocess(
        command,
        stdin=sql,
        accepted_returncodes=accepted_returncodes,
        redactions=redactions,
    )


def test_real_postgres18_managed_nonsuper_foundation_and_password_boundary():
    """Exercise real PG18 role, CopyData logging, and fail-closed rules."""

    if shutil.which("docker") is None:
        pytest.skip("docker is unavailable")
    image_probe = subprocess.run(
        ["docker", "image", "inspect", "postgres:18"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if image_probe.returncode != 0:
        pytest.skip("the local postgres:18 image is unavailable")

    container = "cw-foundation-pg18-" + uuid.uuid4().hex[:12]
    admin = ADMIN
    admin_password = "fixture-only-admin-password"
    writer_secret = b"S" * 64
    rejected_secret = b"T" * 64
    redactions = (
        admin_password.encode("ascii"),
        writer_secret,
        rejected_secret,
    )
    try:
        _sanitized_subprocess(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container,
                "-e",
                "POSTGRES_PASSWORD=fixture-only-root-password",
                "postgres:18",
                "-c",
                "log_statement=all",
                "-c",
                "log_min_duration_statement=0",
                "-c",
                "log_duration=on",
                "-c",
                "log_connections=on",
                "-c",
                "log_disconnections=on",
                "-c",
                "log_parameter_max_length=-1",
                "-c",
                "log_parameter_max_length_on_error=-1",
                "-c",
                "allow_system_table_mods=on",
            ],
            redactions=redactions,
        )
        consecutive_ready = 0
        for _attempt in range(80):
            ready = subprocess.run(
                [
                    "docker",
                    "exec",
                    container,
                    "psql",
                    "-U",
                    "postgres",
                    "-d",
                    "postgres",
                    "-Atqc",
                    "SELECT 1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if ready.returncode == 0 and ready.stdout == b"1\n":
                consecutive_ready += 1
                if consecutive_ready == 3:
                    break
            else:
                consecutive_ready = 0
            time.sleep(0.1)
        else:
            pytest.fail("postgres:18 fixture did not become ready")

        bootstrap = f"""
CREATE ROLE cloudsqladmin
    NOLOGIN CREATEROLE NOSUPERUSER NOCREATEDB
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
CREATE ROLE cloudsqlsuperuser
    NOLOGIN CREATEROLE CREATEDB NOSUPERUSER
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
CREATE ROLE {admin}
    LOGIN CREATEROLE CREATEDB NOSUPERUSER
    PASSWORD '{admin_password}';
GRANT cloudsqlsuperuser TO {admin}
    WITH ADMIN FALSE, INHERIT TRUE, SET TRUE;
CREATE ROLE canonical_brain_migration_owner
    NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
CREATE ROLE canonical_brain_writer
    NOLOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
CREATE ROLE canonical_brain_canary_bootstrap
    NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
CREATE ROLE canonical_brain_canary_bootstrap_login
    LOGIN INHERIT PASSWORD NULL NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
CREATE ROLE muncho_canary_writer_login
    NOLOGIN INHERIT PASSWORD NULL NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
CREATE ROLE {LEGACY_ARCHIVE_OWNER}
    NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
    NOREPLICATION NOBYPASSRLS CONNECTION LIMIT -1;
GRANT canonical_brain_canary_bootstrap
    TO canonical_brain_canary_bootstrap_login
    WITH ADMIN FALSE, INHERIT TRUE, SET FALSE;
UPDATE pg_catalog.pg_auth_members
   SET grantor = (SELECT oid FROM pg_catalog.pg_roles
                   WHERE rolname = 'cloudsqladmin')
 WHERE roleid = (SELECT oid FROM pg_catalog.pg_roles
                  WHERE rolname = 'canonical_brain_canary_bootstrap')
   AND member = (SELECT oid FROM pg_catalog.pg_roles
                  WHERE rolname = 'canonical_brain_canary_bootstrap_login');
REVOKE CONNECT, TEMPORARY ON DATABASE postgres FROM PUBLIC;
REVOKE CONNECT, TEMPORARY ON DATABASE template1 FROM PUBLIC;
CREATE DATABASE muncho_canary_brain OWNER cloudsqlsuperuser;
""".encode()
        _docker_psql(
            container,
            bootstrap,
            database="postgres",
            redactions=redactions,
        )
        canonical_truth = b"""
GRANT CREATE ON SCHEMA public TO canonical_brain_migration_owner;
SET ROLE canonical_brain_migration_owner;
CREATE TABLE public.canonical_event_log (
    event_id uuid NOT NULL,
    schema_version text NOT NULL,
    event_type text NOT NULL,
    occurred_at timestamptz NOT NULL,
    case_id text NOT NULL,
    source jsonb NOT NULL,
    actor jsonb NOT NULL,
    subject jsonb NOT NULL,
    evidence jsonb NOT NULL,
    decision jsonb NOT NULL,
    status jsonb NOT NULL,
    next_action jsonb NOT NULL,
    safety jsonb NOT NULL,
    payload jsonb NOT NULL,
    PRIMARY KEY (event_id)
);
REVOKE ALL ON TABLE public.canonical_event_log FROM PUBLIC;
RESET ROLE;
REVOKE CREATE ON SCHEMA public FROM canonical_brain_migration_owner;
"""
        _docker_psql(container, canonical_truth, redactions=redactions)

        artifacts = foundation._load_source_artifacts_for_tests()
        _docker_psql(
            container,
            artifacts["prerequisites"].payload,
            user=admin,
            password=admin_password,
            redactions=redactions,
        )
        least_privilege = _docker_psql(
            container,
            f"""
SELECT role.rolsuper, role.rolcreaterole, role.rolcreatedb,
       pg_catalog.count(membership.*)
  FROM pg_catalog.pg_roles AS role
  LEFT JOIN pg_catalog.pg_auth_members AS membership
    ON membership.member = role.oid
 WHERE role.rolname = '{admin}'
 GROUP BY role.rolsuper, role.rolcreaterole, role.rolcreatedb;
SELECT pg_catalog.count(*)
  FROM pg_catalog.pg_auth_members AS membership
  JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
  JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
 WHERE granted.rolname LIKE 'canonical_brain_%'
    OR member.rolname LIKE 'canonical_brain_%'
    OR granted.rolname = 'muncho_canary_writer_login'
    OR member.rolname = 'muncho_canary_writer_login';
SELECT pg_catalog.has_schema_privilege(
           'canonical_brain_migration_owner', 'public', 'USAGE'
       ), pg_catalog.has_schema_privilege(
           'canonical_brain_migration_owner', 'public', 'CREATE'
       ), pg_catalog.has_database_privilege(
           'canonical_brain_writer', 'postgres', 'CONNECT'
       );
""".encode(),
            tuples_only=True,
            redactions=redactions,
        )
        if least_privilege != b"f|t|t|1\n1\nt|f|f\n":
            pytest.fail("managed-nonsuper least-privilege contract drifted")

        # Phase-B-only authority: the external role creator grants the exact
        # target ADMIN bridge.  The COPY primitive may use it, but Phase A can
        # never emit terminal/ok until Cloud deletes this administrator and a
        # separate session re-observes zero residual authority.
        _docker_psql(
            container,
            f"""
GRANT muncho_canary_writer_login TO {admin}
    WITH ADMIN TRUE, INHERIT FALSE, SET FALSE;
""".encode(),
            redactions=redactions,
        )

        statement = foundation.WRITER_PASSWORD_BOUNDARY
        success_input = (
            statement.setup_sql.encode()
            + b"\n"
            + statement.copy_sql.encode()
            + b";\n"
            + writer_secret
            + b"\n\\.\n"
            + statement.apply_sql.encode()
            + b";\n"
            + statement.cleanup_sql.encode()
            + b"\nSELECT pg_catalog.to_regprocedure("
            + b"'pg_temp._muncho_install_canonical_writer_password()'"
            + b") IS NULL;\n"
        )
        success_output = _docker_psql(
            container,
            success_input,
            user=admin,
            password=admin_password,
            tuples_only=True,
            redactions=redactions,
        )
        if writer_secret in success_output or b"SCRAM-SHA-256$" in success_output:
            pytest.fail("password statement leaked secret material to client output")
        if not success_output.endswith(b"t\n"):
            pytest.fail("password copy logging/cleanup proof was incomplete")

        verifier_proof = _docker_psql(
            container,
            b"""
SELECT rolcanlogin,
       rolpassword LIKE pg_catalog.concat('SCRAM', '-SHA-256$4096:%'),
       rolpassword IS NOT NULL
  FROM pg_catalog.pg_authid
 WHERE rolname = 'muncho_canary_writer_login';
""",
            tuples_only=True,
            redactions=redactions,
        )
        if verifier_proof != b"t|t|t\n":
            pytest.fail("server-generated SCRAM verifier was not installed")

        _docker_psql(
            container,
            b"""
GRANT canonical_brain_writer TO muncho_canary_writer_login
    WITH ADMIN FALSE, INHERIT TRUE, SET FALSE;
UPDATE pg_catalog.pg_auth_members
   SET grantor = (SELECT oid FROM pg_catalog.pg_roles
                   WHERE rolname = 'cloudsqladmin')
 WHERE roleid = (SELECT oid FROM pg_catalog.pg_roles
                  WHERE rolname = 'canonical_brain_writer')
   AND member = (SELECT oid FROM pg_catalog.pg_roles
                  WHERE rolname = 'muncho_canary_writer_login');
""",
            redactions=redactions,
        )
        writer_auth = _sanitized_subprocess(
            [
                "docker",
                "exec",
                "-i",
                container,
                "sh",
                "-c",
                "IFS= read -r PGPASSWORD; export PGPASSWORD; "
                "exec psql -X -qAt -v ON_ERROR_STOP=1 -h 127.0.0.1 "
                "-U muncho_canary_writer_login -d muncho_canary_brain",
            ],
            stdin=(
                writer_secret
                + b"\nSELECT CURRENT_USER::text;\n"
            ),
            redactions=redactions,
        )
        if writer_auth != b"muncho_canary_writer_login\n":
            pytest.fail("writer authentication proof failed")

        _docker_psql(
            container,
            b"""
UPDATE pg_catalog.pg_auth_members
   SET grantor = (SELECT oid FROM pg_catalog.pg_roles
                   WHERE rolname = 'postgres')
 WHERE roleid = (SELECT oid FROM pg_catalog.pg_roles
                  WHERE rolname = 'canonical_brain_writer')
   AND member = (SELECT oid FROM pg_catalog.pg_roles
                  WHERE rolname = 'muncho_canary_writer_login');
REVOKE canonical_brain_writer FROM muncho_canary_writer_login;
ALTER ROLE muncho_canary_writer_login NOLOGIN PASSWORD NULL;
GRANT muncho_canary_writer_login TO canonical_brain_writer
    WITH ADMIN FALSE, INHERIT TRUE, SET TRUE;
""",
            redactions=redactions,
        )
        forced_error_input = (
            b"\\set ON_ERROR_STOP off\n"
            + statement.setup_sql.encode()
            + b"\n"
            + statement.copy_sql.encode()
            + b";\n"
            + rejected_secret
            + b"\n\\.\n"
            + statement.apply_sql.encode()
            + b";\nROLLBACK;\n"
            + b"SELECT pg_catalog.to_regprocedure("
            + b"'pg_temp._muncho_install_canonical_writer_password()'"
            + b") IS NULL;\n"
        )
        forced_error_output = _docker_psql(
            container,
            forced_error_input,
            user=admin,
            password=admin_password,
            tuples_only=True,
            accepted_returncodes=frozenset({0, 3}),
            redactions=redactions,
        )
        if rejected_secret in forced_error_output or b"SCRAM-SHA-256$" in forced_error_output:
            pytest.fail("forced-error path leaked secret material to client output")
        if b"canonical writer password statement prerequisite failed" not in forced_error_output:
            pytest.fail("forced-error path did not fail at the fixed prerequisite")
        if not forced_error_output.endswith(b"t\n"):
            pytest.fail("forced-error transaction/temp-function cleanup was incomplete")

        rejected_state = _docker_psql(
            container,
            b"""
SELECT NOT rolcanlogin, rolpassword IS NULL
  FROM pg_catalog.pg_authid
 WHERE rolname = 'muncho_canary_writer_login';
""",
            tuples_only=True,
            redactions=redactions,
        )
        if rejected_state != b"t|t\n":
            pytest.fail("forced-error path changed writer login/password state")

        _docker_psql(
            container,
            b"""
REVOKE muncho_canary_writer_login FROM canonical_brain_writer;
""",
            redactions=redactions,
        )
        restored_output = _docker_psql(
            container,
            success_input,
            user=admin,
            password=admin_password,
            tuples_only=True,
            redactions=redactions,
        )
        if not restored_output.endswith(b"t\n"):
            pytest.fail("writer credential restoration proof was incomplete")
        _docker_psql(
            container,
            b"""
GRANT canonical_brain_writer TO muncho_canary_writer_login
    WITH ADMIN FALSE, INHERIT TRUE, SET FALSE;
UPDATE pg_catalog.pg_auth_members
   SET grantor = (SELECT oid FROM pg_catalog.pg_roles
                   WHERE rolname = 'cloudsqladmin')
 WHERE roleid = (SELECT oid FROM pg_catalog.pg_roles
                  WHERE rolname = 'canonical_brain_writer')
   AND member = (SELECT oid FROM pg_catalog.pg_roles
                  WHERE rolname = 'muncho_canary_writer_login');

CREATE SCHEMA canonical_brain
    AUTHORIZATION canonical_brain_migration_owner;
CREATE SCHEMA canonical_brain_legacy_quarantine
    AUTHORIZATION canonical_brain_migration_owner;
SET ROLE canonical_brain_migration_owner;
CREATE FUNCTION canonical_brain.writer_ping(jsonb, jsonb)
RETURNS jsonb
LANGUAGE sql
STABLE
AS $writer_ping$
    SELECT pg_catalog.jsonb_build_object('service', 'canonical_writer',
                                         'protocol', 'v1')
$writer_ping$;
CREATE TABLE canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1 (
    event_id uuid PRIMARY KEY
);
RESET ROLE;
ALTER TABLE canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1
    OWNER TO legacy_archive_source_owner;
REVOKE ALL ON SCHEMA public FROM PUBLIC, canonical_brain_writer;
REVOKE ALL ON SCHEMA canonical_brain_legacy_quarantine
    FROM PUBLIC, canonical_brain_writer;
REVOKE ALL ON TABLE public.canonical_event_log
    FROM PUBLIC, canonical_brain_writer;
REVOKE ALL ON TABLE
    canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1
    FROM PUBLIC, canonical_brain_writer;
""",
            redactions=redactions,
        )

        logs = _sanitized_subprocess(
            ["docker", "logs", container],
            redactions=redactions,
        )
        for secret in (writer_secret, rejected_secret):
            if secret in logs:
                pytest.fail("postgres logging leaked writer password CopyData")
        if b"SCRAM-SHA-256$" in logs:
            pytest.fail("postgres logging leaked a SCRAM verifier")

        # PG18 records creator/foreign grants under their real grantor.  A
        # different CREATEROLE admin cannot delete that row, proving why the
        # missing-role/legacy path must remain fail-closed until Cloud deletes
        # the temporary admin and a separate post-delete observation succeeds.
        _docker_psql(
            container,
            f"""
CREATE ROLE legacy_source_owner_proof
    NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE;
GRANT legacy_source_owner_proof TO {admin}
    WITH ADMIN TRUE, INHERIT TRUE, SET TRUE;
""".encode(),
            redactions=redactions,
        )
        _docker_psql(
            container,
            f"REVOKE legacy_source_owner_proof FROM {admin};\n".encode(),
            user=admin,
            password=admin_password,
            redactions=redactions,
        )
        bridge_survived = _docker_psql(
            container,
            f"""
SELECT pg_catalog.count(*), grantor.rolname
  FROM pg_catalog.pg_auth_members AS membership
  JOIN pg_catalog.pg_roles AS granted ON granted.oid = membership.roleid
  JOIN pg_catalog.pg_roles AS member ON member.oid = membership.member
  JOIN pg_catalog.pg_roles AS grantor ON grantor.oid = membership.grantor
 WHERE granted.rolname = 'legacy_source_owner_proof'
   AND member.rolname = '{admin}'
 GROUP BY grantor.rolname;
""".encode(),
            tuples_only=True,
            redactions=redactions,
        )
        if bridge_survived != b"1|postgres\n":
            pytest.fail("different-admin PG18 grantor proof changed")

        _docker_psql(
            container,
            f"REVOKE muncho_canary_writer_login FROM {admin};\n".encode(),
            redactions=redactions,
        )

        def writer_observation() -> dict[str, object]:
            output = _docker_psql(
                container,
                artifacts["observe"].payload,
                user=foundation.SQL_USER,
                password=writer_secret.decode("ascii"),
                tuples_only=True,
                redactions=redactions,
            )
            lines = [line for line in output.splitlines() if line.strip()]
            if not lines:
                pytest.fail("writer observation returned no JSON")
            value = json.loads(lines[-1])
            if not isinstance(value, dict):
                pytest.fail("writer observation JSON was not an object")
            return value

        def bind_observation(
            database_value: dict[str, object],
        ) -> foundation.FoundationObservation:
            unsigned = {
                "schema": foundation.FOUNDATION_OBSERVATION_SCHEMA,
                **database_value,
                "tls_peer_certificate_sha256": TLS_PEER,
                "legacy_truth": None,
                "credential": _credential("installed"),
            }
            return foundation.FoundationObservation.from_mapping(
                {
                    **unsigned,
                    "observation_sha256": foundation._sha256_json(unsigned),
                }
            )

        false_terminal = writer_observation()
        if false_terminal["temporary_admin_roles"] != [admin]:
            pytest.fail("pre-delete observation lost the temporary administrator")
        with pytest.raises(
            foundation.CanonicalWriterFoundationError,
            match="foundation_temporary_admin_authority_drifted",
        ):
            bind_observation(false_terminal)

        archive_oid_before = _docker_psql(
            container,
            b"""
SELECT class.oid
  FROM pg_catalog.pg_class AS class
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = class.relnamespace
 WHERE namespace.nspname = 'canonical_brain_legacy_quarantine'
   AND class.relname = 'canonical_event_log_legacy_v1';
""",
            tuples_only=True,
            redactions=redactions,
        )
        _docker_psql(
            container,
            f"DROP OWNED BY {admin};\n".encode(),
            redactions=redactions,
        )
        _docker_psql(
            container,
            f"DROP OWNED BY {admin};\nDROP ROLE {admin};\n".encode(),
            database="postgres",
            redactions=redactions,
        )
        _docker_psql(
            container,
            b"""
REVOKE ALL PRIVILEGES ON DATABASE muncho_canary_brain FROM PUBLIC;
REVOKE ALL PRIVILEGES ON DATABASE muncho_canary_brain FROM
    canonical_brain_migration_owner,
    canonical_brain_writer,
    canonical_brain_canary_bootstrap,
    canonical_brain_canary_bootstrap_login,
    muncho_canary_writer_login;
GRANT CONNECT ON DATABASE muncho_canary_brain TO
    canonical_brain_writer,
    canonical_brain_canary_bootstrap;
REVOKE ALL ON SCHEMA public FROM
    PUBLIC,
    canonical_brain_migration_owner,
    canonical_brain_writer,
    canonical_brain_canary_bootstrap,
    canonical_brain_canary_bootstrap_login,
    muncho_canary_writer_login;
GRANT USAGE ON SCHEMA public TO canonical_brain_migration_owner;
REVOKE ALL ON SCHEMA canonical_brain_legacy_quarantine FROM
    PUBLIC,
    canonical_brain_writer;
""",
            redactions=redactions,
        )

        restricted = _docker_psql(
            container,
            b"""
SELECT pg_catalog.has_schema_privilege(
           'canonical_brain_writer', 'public', 'USAGE'
       ), pg_catalog.has_schema_privilege(
           'canonical_brain_writer',
           'canonical_brain_legacy_quarantine', 'USAGE'
       ), pg_catalog.has_table_privilege(
           'canonical_brain_writer', 'public.canonical_event_log', 'SELECT'
       ), pg_catalog.has_table_privilege(
           'canonical_brain_writer',
           'canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1',
           'SELECT'
       );
""",
            tuples_only=True,
            redactions=redactions,
        )
        if restricted != b"f|f|f|f\n":
            pytest.fail("writer schema/table observation boundary widened")

        terminal_database_observation = writer_observation()
        observation = bind_observation(terminal_database_observation)
        if (
            terminal_database_observation["database_owner"]
            != foundation.DATABASE_OWNER_ROLE
            or terminal_database_observation["temporary_admin_roles"] != []
            or terminal_database_observation["event_log_shape"] != "canonical14"
            or terminal_database_observation["event_log_owner"]
            != foundation.MIGRATION_OWNER_ROLE
            or terminal_database_observation["canonical_schema_owner"]
            != foundation.MIGRATION_OWNER_ROLE
            or terminal_database_observation["writer_ping_present"] is not True
            or terminal_database_observation["legacy_archive_present"] is not True
            or terminal_database_observation["legacy_archive_identity"]
            != {
                "oid": archive_oid_before.strip().decode("ascii"),
                "owner": LEGACY_ARCHIVE_OWNER,
                "relation_kind": "r",
                "persistence": "p",
                "owner_superuser": False,
                "owner_create_database": False,
                "owner_create_role": False,
                "owner_replication": False,
                "owner_bypass_row_security": False,
                "owner_connection_limit": -1,
                "owner_validity_is_unbounded": True,
                "owner_configuration_is_empty": True,
            }
            or not observation.membership_ready
        ):
            pytest.fail("post-delete writer observation contract drifted")
        membership_contract = {
            (
                row["granted_role"],
                row["member_role"],
                row["grantor"],
                row["admin_option"],
                row["inherit_option"],
                row["set_option"],
            )
            for row in terminal_database_observation["memberships"]
        }
        if membership_contract != {
            (
                foundation.CANARY_BOOTSTRAP_ROLE,
                foundation.CANARY_BOOTSTRAP_LOGIN,
                foundation.PERSISTENT_MEMBERSHIP_GRANTOR,
                False,
                True,
                False,
            ),
            (
                foundation.WRITER_ROLE,
                foundation.SQL_USER,
                foundation.PERSISTENT_MEMBERSHIP_GRANTOR,
                False,
                True,
                False,
            ),
        }:
            pytest.fail("provider-native membership observation drifted")
        archive_oid_after = _docker_psql(
            container,
            b"""
SELECT class.oid
  FROM pg_catalog.pg_class AS class
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = class.relnamespace
 WHERE namespace.nspname = 'canonical_brain_legacy_quarantine'
   AND class.relname = 'canonical_event_log_legacy_v1';
""",
            tuples_only=True,
            redactions=redactions,
        )
        if not archive_oid_before.strip() or archive_oid_after != archive_oid_before:
            pytest.fail("legacy archive identity was not preserved")

        class _DockerWriterObservationSession:
            username = foundation.SQL_USER
            tls_peer_certificate_sha256 = TLS_PEER

            def query(self, sql: str, *, maximum_rows: int) -> QueryResult:
                assert maximum_rows == 1
                output = _docker_psql(
                    container,
                    sql.encode("utf-8"),
                    user=foundation.SQL_USER,
                    password=writer_secret.decode("ascii"),
                    tuples_only=True,
                    redactions=redactions,
                )
                rows = [line for line in output.splitlines() if line.strip()]
                if not rows:
                    pytest.fail("writer re-observation returned no JSON")
                return QueryResult(
                    ("foundation_observation",),
                    ((rows[-1].decode("utf-8"),),),
                    "SELECT 1",
                )

        real_plan = foundation.build_foundation_adoption_plan(
            REVISION,
            observation,
            _artifacts=artifacts,
            _hba_collector=_hba,
            _authentication_probe=lambda _enabled: {
                "authenticated": True,
                "user": foundation.SQL_USER,
            },
            _terminal_attestor=_attestation,
            _clock=lambda: NOW,
        )
        real_store = _SecretStore("installed")
        real_session = _DockerWriterObservationSession()
        foundation._observe_for_plan(
            real_plan,
            real_session,
            secret_store=real_store,
            artifacts=artifacts,
            expected_credential=observation.value["credential"],
        )
        _docker_psql(
            container,
            b"""
DROP TABLE
    canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1;
SET ROLE canonical_brain_migration_owner;
CREATE TABLE canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1 (
    event_id uuid PRIMARY KEY
);
RESET ROLE;
ALTER TABLE canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1
    OWNER TO legacy_archive_source_owner;
REVOKE ALL ON TABLE
    canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1
    FROM PUBLIC, canonical_brain_writer;
""",
            redactions=redactions,
        )
        replaced_archive_oid = _docker_psql(
            container,
            b"""
SELECT class.oid
  FROM pg_catalog.pg_class AS class
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = class.relnamespace
 WHERE namespace.nspname = 'canonical_brain_legacy_quarantine'
   AND class.relname = 'canonical_event_log_legacy_v1';
""",
            tuples_only=True,
            redactions=redactions,
        )
        if replaced_archive_oid == archive_oid_after:
            pytest.fail("same-name archive replacement did not change identity")
        with pytest.raises(
            foundation.CanonicalWriterFoundationError,
            match="foundation_legacy_archive_changed_after_approval",
        ):
            foundation._observe_for_plan(
                real_plan,
                real_session,
                secret_store=real_store,
                artifacts=artifacts,
                expected_credential=observation.value["credential"],
            )
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
