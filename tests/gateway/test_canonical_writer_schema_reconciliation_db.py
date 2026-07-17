from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from gateway import canonical_writer_schema_reconciliation as reconciliation
from gateway import canonical_writer_schema_reconciliation_db as reconciliation_db
from gateway.canonical_writer_db import (
    CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY,
    CredentialSource,
    ManagedCloudSQLAdminHBAReceipt,
    PrivilegeAttestationError,
    QueryResult,
    WriterDBConfig,
    _collect_privilege_attestation,
)
from gateway.canonical_writer_foundation import _load_source_artifacts_for_tests


REVISION = "a" * 40
ASSET = (
    Path(__file__).parents[2]
    / "gateway"
    / "assets"
    / "canonical_writer_schema_contract_v1.json"
)


def _target_and_plan() -> tuple[
    reconciliation.SchemaContract,
    reconciliation.SchemaReconciliationPlan,
]:
    asset = reconciliation.SchemaContractAsset.from_bytes(ASSET.read_bytes())
    artifact = _load_source_artifacts_for_tests()["base_migration"]
    plan = reconciliation._build_plan_from_artifact(
        REVISION,
        asset.contract,
        artifact,
        target_asset_sha256=asset.sha256,
    )
    return asset.contract, plan


def _config(user: str) -> WriterDBConfig:
    return WriterDBConfig(
        host="127.0.0.1",
        tls_server_name="localhost",
        port=5432,
        database="muncho_canary_brain",
        user=user,
        ca_file=Path("/tmp/muncho-canary-test-ca.pem"),
        credential=CredentialSource(fd=0, expected_uid=os.getuid()),
        application_name="muncho-schema-reconciliation-test",
    )


def _receipt(config: WriterDBConfig) -> ManagedCloudSQLAdminHBAReceipt:
    now = int(time.time())
    return ManagedCloudSQLAdminHBAReceipt(
        version="managed-cloudsqladmin-hba-rejection-v2",
        host=config.host,
        tls_server_name=config.tls_server_name,
        port=config.port,
        server_certificate_sha256="e" * 64,
        database="cloudsqladmin",
        user=config.user,
        observed_at_unix=now,
        expires_at_unix=now + 300,
        sqlstate="28000",
        server_message=(
            f'no pg_hba.conf entry for host "127.0.0.1", user "{config.user}", '
            'database "cloudsqladmin", SSL encryption'
        ),
        result="pg_hba_rejected",
        tls_peer_verified=True,
    )


def _truth_receipt(
    row_count: int = 3,
    canonical14_sha256: str = "d" * 64,
) -> reconciliation.CanonicalTruthReceipt:
    return reconciliation.CanonicalTruthReceipt(
        row_count=row_count,
        canonical14_sha256=canonical14_sha256,
        relation_receipts=tuple(
            reconciliation.CanonicalRelationTruthReceipt(
                relation=relation,
                row_count=row_count if index == 0 else 0,
                chunk_count=1 if index == 0 and row_count else 0,
                chunk_manifest_sha256=hashlib.sha256(
                    f"{relation}:{row_count}:{canonical14_sha256}".encode()
                ).hexdigest(),
            )
            for index, relation in enumerate(
                reconciliation.CANONICAL_TRUTH_RELATIONS
            )
        ),
        quarantine_anchor_receipts=tuple(
            reconciliation.CanonicalQuarantineAnchorReceipt(
                anchor=anchor,
                object_oid=9000 + index,
                owner="postgres",
                kind="n" if index == 1 else "r",
                persistence="" if index == 1 else "p",
                acl_sha256=hashlib.sha256(anchor.encode()).hexdigest(),
            )
            for index, anchor in enumerate(
                reconciliation.CANONICAL_QUARANTINE_ANCHORS,
                start=1,
            )
        ),
    )


class _FakeSession:
    def __init__(
        self,
        plan: reconciliation.SchemaReconciliationPlan,
        *,
        username: str = "muncho_canary_admin_aaaaaaaaaaaaaaaa",
        open_receipt_valid: bool = True,
        close_receipt_valid: bool = True,
        open_preflight_failure: str | None = None,
        open_postflight_failure: str | None = None,
        authority_open_server_preflight_failure: bool = False,
    ) -> None:
        self.plan = plan
        self.segments = reconciliation_db._split_sealed_mutation_sql(plan)
        self.username = username
        self.tls_peer_certificate_sha256 = "e" * 64
        self.queries: list[tuple[str, int]] = []
        self.closed = False
        self.transaction_open = False
        self.authority_present = True
        self.body_present = False
        self.committed_with_authority = False
        self.open_receipt_valid = open_receipt_valid
        self.close_receipt_valid = close_receipt_valid
        self.open_preflight_failure = open_preflight_failure
        self.open_postflight_failure = open_postflight_failure
        self.authority_open_server_preflight_failure = (
            authority_open_server_preflight_failure
        )
        self.quarantine_flags = ("t", "t", "t")
        self.quarantine_receipts: object = _truth_receipt().value[
            "quarantine_anchors"
        ]

    def query(self, sql: str, *, maximum_rows: int) -> QueryResult:
        self.queries.append((sql, maximum_rows))
        if sql.startswith("SELECT pg_catalog.pg_advisory_lock("):
            return QueryResult(("pg_advisory_lock",), (("",),), "SELECT 1")
        if sql.startswith("SELECT pg_catalog.pg_advisory_xact_lock("):
            return QueryResult(
                ("pg_advisory_xact_lock",),
                (("",),),
                "SELECT 1",
            )
        if sql.startswith("SELECT pg_catalog.pg_advisory_unlock("):
            return QueryResult(("pg_advisory_unlock",), (("t",),), "SELECT 1")
        if sql == "BEGIN ISOLATION LEVEL SERIALIZABLE":
            self.transaction_open = True
            return QueryResult((), (), "BEGIN")
        if sql == self.segments.authority_open:
            assert self.transaction_open
            assert self.authority_present
            if self.authority_open_server_preflight_failure:
                raise reconciliation_db.PostgresServerError(
                    sqlstate="P0001",
                    server_message=(
                        reconciliation_db._AUTHORITY_OPEN_PREFLIGHT_SERVER_MESSAGE
                    ),
                )
            return QueryResult((), (), "DO")
        if sql == reconciliation_db._AUTHORITY_OPEN_RECEIPT_SQL:
            receipt_count = sum(
                statement == reconciliation_db._AUTHORITY_OPEN_RECEIPT_SQL
                for statement, _maximum_rows in self.queries
            )
            values = [
                "t" if self.authority_present else "f"
                for _column in reconciliation_db._AUTHORITY_OPEN_RECEIPT_COLUMNS
            ]
            if receipt_count == 1 and self.open_preflight_failure is not None:
                values[
                    reconciliation_db._AUTHORITY_OPEN_RECEIPT_COLUMNS.index(
                        self.open_preflight_failure
                    )
                ] = "f"
            if receipt_count > 1 and self.open_postflight_failure is not None:
                values[
                    reconciliation_db._AUTHORITY_OPEN_RECEIPT_COLUMNS.index(
                        self.open_postflight_failure
                    )
                ] = "f"
            if receipt_count > 1 and not self.open_receipt_valid:
                values = ["f" for _value in values]
            return QueryResult(
                reconciliation_db._AUTHORITY_OPEN_RECEIPT_COLUMNS,
                (tuple(values),),
                "SELECT 1",
            )
        if sql == self.segments.body:
            assert self.authority_present
            self.body_present = True
            return QueryResult((), (), "DO")
        if sql == self.segments.authority_close:
            assert self.authority_present
            return QueryResult((), (), "DO")
        if sql == reconciliation_db._AUTHORITY_CLOSE_RECEIPT_SQL:
            value = (
                "t"
                if self.close_receipt_valid and self.authority_present
                else "f"
            )
            return QueryResult(
                reconciliation_db._AUTHORITY_CLOSE_RECEIPT_COLUMNS,
                ((value,) * len(reconciliation_db._AUTHORITY_CLOSE_RECEIPT_COLUMNS),),
                "SELECT 1",
            )
        if sql.startswith("SET LOCAL "):
            return QueryResult((), (), "SET")
        if sql == reconciliation_db._CANONICAL_DATA_LOCK_SQL:
            return QueryResult((), (), "LOCK TABLE")
        if sql == reconciliation_db._CANONICAL_TRUTH_SQL:
            return QueryResult(
                ("row_count", "canonical14_sha256", "relation_receipts"),
                (
                    (
                        "3",
                        "d" * 64,
                        json.dumps(_truth_receipt().value["relation_receipts"]),
                    ),
                ),
                "SELECT 1",
            )
        if sql == reconciliation_db._QUARANTINE_ANCHOR_SQL:
            return QueryResult(
                reconciliation_db._QUARANTINE_ANCHOR_COLUMNS,
                ((
                    *self.quarantine_flags,
                    json.dumps(self.quarantine_receipts),
                ),),
                "SELECT 1",
            )
        if sql == "COMMIT":
            self.committed_with_authority = self.authority_present
            self.transaction_open = False
            return QueryResult((), (), "COMMIT")
        if sql == "ROLLBACK":
            self.transaction_open = False
            self.body_present = False
            return QueryResult((), (), "ROLLBACK")
        raise AssertionError(f"unexpected SQL: {sql[:80]}")

    def close(self) -> None:
        if self.transaction_open:
            self.transaction_open = False
            self.body_present = False
        self.closed = True


class _FreshWriterSession:
    def __init__(
        self,
        *,
        ping_response: object | None = None,
        username: str = reconciliation_db.WRITER_LOGIN,
        tls_peer_certificate_sha256: str = "e" * 64,
        authority_flags: tuple[str, ...] | None = None,
    ) -> None:
        self.username = username
        self.tls_peer_certificate_sha256 = tls_peer_certificate_sha256
        self.ping_response = (
            reconciliation_db._POST_DELETE_WRITER_PING_RESPONSE
            if ping_response is None
            else ping_response
        )
        self.queries: list[tuple[str, int]] = []
        self.closed = False
        self.transaction_open = False
        self.authority_flags = authority_flags or (
            ("t",) * len(reconciliation_db._POST_DELETE_AUTHORITY_COLUMNS)
        )

    def query(self, sql: str, *, maximum_rows: int) -> QueryResult:
        self.queries.append((sql, maximum_rows))
        if sql == "BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY":
            self.transaction_open = True
            return QueryResult((), (), "BEGIN")
        if sql.startswith("SET LOCAL "):
            return QueryResult((), (), "SET")
        if sql.startswith("SELECT pg_catalog.pg_advisory_xact_lock_shared("):
            return QueryResult(
                ("pg_advisory_xact_lock_shared",),
                (("",),),
                "SELECT 1",
            )
        if sql.startswith(
            "SELECT CURRENT_USER = 'muncho_canary_writer_login'"
        ):
            return QueryResult(
                reconciliation_db._POST_DELETE_AUTHORITY_COLUMNS,
                (self.authority_flags,),
                "SELECT 1",
            )
        if sql == reconciliation_db._POST_DELETE_WRITER_PING_SQL:
            return QueryResult(
                ("writer_ping_response",),
                ((json.dumps(self.ping_response),),),
                "SELECT 1",
            )
        if sql == "COMMIT":
            self.transaction_open = False
            return QueryResult((), (), "COMMIT")
        if sql == "ROLLBACK":
            self.transaction_open = False
            return QueryResult((), (), "ROLLBACK")
        raise AssertionError(f"unexpected SQL: {sql[:80]}")

    def close(self) -> None:
        self.transaction_open = False
        self.closed = True


def _database(
    monkeypatch: pytest.MonkeyPatch,
    *,
    session: _FakeSession | None = None,
) -> tuple[
    reconciliation_db.PostgresSchemaReconciliationDatabase,
    _FakeSession,
    reconciliation.SchemaContract,
    reconciliation.SchemaReconciliationPlan,
    list[dict[str, object]],
]:
    target, plan = _target_and_plan()
    writer = _config(reconciliation_db.WRITER_LOGIN)
    admin = _config("muncho_canary_admin_aaaaaaaaaaaaaaaa")
    active_session = session or _FakeSession(plan)
    observations: list[dict[str, object]] = []

    def collect(_session, **kwargs):
        observations.append(kwargs)
        return target

    monkeypatch.setattr(reconciliation_db, "collect_schema_contract", collect)
    database = reconciliation_db.PostgresSchemaReconciliationDatabase(
        plan=plan,
        target=target,
        admin_config=admin,
        writer_config=writer,
        managed_hba_receipt=_receipt(writer),
        _session_factory=lambda config: (
            active_session
            if config == admin
            else (_ for _ in ()).throw(AssertionError("wrong config"))
        ),
    )
    return database, active_session, target, plan, observations


def _plan_with_mutation(
    plan: reconciliation.SchemaReconciliationPlan,
    mutation_sql: str,
) -> reconciliation.SchemaReconciliationPlan:
    value = dict(plan.value)
    value["mutation_sql_sha256"] = hashlib.sha256(
        mutation_sql.encode("utf-8")
    ).hexdigest()
    unsigned = {key: item for key, item in value.items() if key != "plan_sha256"}
    value["plan_sha256"] = reconciliation._sha256_json(unsigned)
    return reconciliation.SchemaReconciliationPlan.from_mapping(
        value,
        mutation_sql=mutation_sql,
    )


def test_transaction_proves_api_authority_and_restored_trampoline_before_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, session, target, plan, observations = _database(monkeypatch)

    with database.transaction(
        advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
    ) as transaction:
        transaction.lock_canonical_truth()
        assert transaction.observe_contract() == target
        assert transaction.observe_canonical_truth() == _truth_receipt()
        transaction.execute_sql(plan.mutation_sql)
        assert transaction.observe_contract() == target
        assert transaction.observe_canonical_truth() == _truth_receipt()

    sql = [item[0] for item in session.queries]
    session_lock = next(
        index
        for index, statement in enumerate(sql)
        if statement.startswith("SELECT pg_catalog.pg_advisory_lock(")
    )
    begin = sql.index("BEGIN ISOLATION LEVEL SERIALIZABLE")
    authority_open_receipts = [
        index
        for index, statement in enumerate(sql)
        if statement == reconciliation_db._AUTHORITY_OPEN_RECEIPT_SQL
    ]
    assert len(authority_open_receipts) == 2
    authority_preflight, authority_open_receipt = authority_open_receipts
    authority_open = sql.index(session.segments.authority_open)
    table_lock = sql.index(reconciliation_db._CANONICAL_DATA_LOCK_SQL)
    xact_lock = next(
        index
        for index, statement in enumerate(sql)
        if statement.startswith("SELECT pg_catalog.pg_advisory_xact_lock(")
    )
    truths = [
        index
        for index, statement in enumerate(sql)
        if statement == reconciliation_db._CANONICAL_TRUTH_SQL
    ]
    mutation_body = sql.index(session.segments.body)
    authority_close = sql.index(session.segments.authority_close)
    authority_close_receipt = sql.index(
        reconciliation_db._AUTHORITY_CLOSE_RECEIPT_SQL
    )
    commit = sql.index("COMMIT")
    unlock = next(
        index
        for index, statement in enumerate(sql)
        if statement.startswith("SELECT pg_catalog.pg_advisory_unlock(")
    )
    assert len(truths) == 2
    assert (
        session_lock
        < begin
        < authority_preflight
        < authority_open
        < authority_open_receipt
        < table_lock
        < xact_lock
        < truths[0]
        < mutation_body
        < truths[1]
        < authority_close
        < authority_close_receipt
        < commit
        < unlock
    )
    assert session.closed
    assert session.authority_present is True
    assert session.committed_with_authority is True
    assert observations[0]["config"].user == reconciliation_db.WRITER_LOGIN
    assert observations[0]["subject_user"] == reconciliation_db.WRITER_LOGIN
    assert observations[0]["allow_missing_helper"] is True
    assert dataclasses.asdict(observations[0]["owner_membership_projection"]) == {
        "owner_role": "canonical_brain_migration_owner",
        "writer_role": "canonical_brain_writer",
        "session_user": "muncho_canary_admin_aaaaaaaaaaaaaaaa",
    }


@pytest.mark.parametrize("failed_index", (0, 1, 2))
def test_quarantine_identity_acl_inventory_and_reachability_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    failed_index: int,
) -> None:
    _, plan = _target_and_plan()
    session = _FakeSession(plan)
    flags = ["t", "t", "t"]
    flags[failed_index] = "f"
    session.quarantine_flags = tuple(flags)
    database, _, _, _, _ = _database(monkeypatch, session=session)

    with pytest.raises(
        reconciliation_db.PostgresProtocolError,
        match="schema_reconciliation_quarantine_anchor_invalid",
    ):
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ) as transaction:
            transaction.lock_canonical_truth()
            transaction.observe_canonical_truth()

    assert session.closed is True
    assert any(statement == "ROLLBACK" for statement, _ in session.queries)


def test_quarantine_anchor_oid_and_acl_are_bound_into_canonical_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, session, _, _, _ = _database(monkeypatch)

    with database.transaction(
        advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
    ) as transaction:
        transaction.lock_canonical_truth()
        before = transaction.observe_canonical_truth()
        changed = [dict(item) for item in session.quarantine_receipts]
        changed[1]["object_oid"] += 1
        changed[2]["acl_sha256"] = "0" * 64
        session.quarantine_receipts = changed
        after = transaction.observe_canonical_truth()

    assert after != before
    assert after.quarantine_anchors_sha256 != before.quarantine_anchors_sha256
    assert after.sha256 != before.sha256


def test_quarantine_anchor_projection_rejects_missing_or_reordered_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, plan = _target_and_plan()
    session = _FakeSession(plan)
    session.quarantine_receipts = list(
        reversed(session.quarantine_receipts)
    )[:-1]
    database, _, _, _, _ = _database(monkeypatch, session=session)

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_canonical_truth_invalid",
    ):
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ) as transaction:
            transaction.lock_canonical_truth()
            transaction.observe_canonical_truth()


def test_transaction_executes_only_the_byte_identical_plan_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, session, _, plan, _ = _database(monkeypatch)

    with database.transaction(
        advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
    ) as transaction:
        transaction.lock_canonical_truth()
        with pytest.raises(
            reconciliation.SchemaReconciliationError,
            match="schema_reconciliation_database_sql_not_sealed",
        ):
            transaction.execute_sql(plan.mutation_sql + "\n")
        transaction.execute_sql(plan.mutation_sql)
        with pytest.raises(
            reconciliation.SchemaReconciliationError,
            match="schema_reconciliation_database_apply_repeated",
        ):
            transaction.execute_sql(plan.mutation_sql)

    sql = [statement for statement, _ in session.queries]
    assert plan.mutation_sql not in sql
    assert sql.count(session.segments.authority_open) == 1
    assert sql.count(session.segments.body) == 1
    assert sql.count(session.segments.authority_close) == 1
    assert session.segments.mutation_sql == plan.mutation_sql
    assert hashlib.sha256(session.segments.mutation_sql.encode()).hexdigest() == (
        plan.value["mutation_sql_sha256"]
    )
    assert session.authority_present is True
    assert session.committed_with_authority is True


def test_transaction_failure_never_commits_and_releases_by_rollback_or_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, session, _, _, _ = _database(monkeypatch)

    with pytest.raises(RuntimeError, match="injected failure"):
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ) as transaction:
            transaction.lock_canonical_truth()
            raise RuntimeError("injected failure")

    sql = [item[0] for item in session.queries]
    assert "COMMIT" not in sql
    assert "ROLLBACK" in sql
    assert any(
        statement.startswith("SELECT pg_catalog.pg_advisory_unlock(")
        for statement in sql
    )
    assert session.closed
    assert session.authority_present is True
    assert session.body_present is False
    assert session.committed_with_authority is False


def test_transaction_rolls_back_sealed_body_without_mutating_api_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, session, _, plan, _ = _database(monkeypatch)

    with pytest.raises(RuntimeError, match="post-apply failure"):
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ) as transaction:
            transaction.lock_canonical_truth()
            transaction.execute_sql(plan.mutation_sql)
            raise RuntimeError("post-apply failure")

    sql = [item[0] for item in session.queries]
    assert session.segments.authority_open in sql
    assert session.segments.body in sql
    assert session.segments.authority_close not in sql
    assert "COMMIT" not in sql
    assert "ROLLBACK" in sql
    assert session.authority_present is True
    assert session.body_present is False
    assert session.committed_with_authority is False
    assert session.closed


def test_exact_target_scope_commits_no_helper_or_role_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, session, target, _, _ = _database(monkeypatch)

    with database.transaction(
        advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
    ) as transaction:
        transaction.lock_canonical_truth()
        assert transaction.observe_contract() == target
        assert transaction.observe_canonical_truth() == _truth_receipt()

    sql = [item[0] for item in session.queries]
    assert session.segments.authority_open in sql
    assert session.segments.body not in sql
    assert session.segments.authority_close in sql
    assert "COMMIT" in sql
    assert "ROLLBACK" not in sql
    assert session.authority_present is True
    assert session.committed_with_authority is True
    assert session.body_present is False
    assert session.closed


def test_exact_target_authority_close_failure_rolls_back_net_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, plan = _target_and_plan()
    session = _FakeSession(plan, close_receipt_valid=False)
    database, _, target, _, _ = _database(monkeypatch, session=session)

    with pytest.raises(
        reconciliation_db.PostgresProtocolError,
        match="schema_reconciliation_database_authority_survived",
    ):
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ) as transaction:
            transaction.lock_canonical_truth()
            assert transaction.observe_contract() == target

    sql = [item[0] for item in session.queries]
    assert session.segments.authority_open in sql
    assert session.segments.body not in sql
    assert session.segments.authority_close in sql
    assert "COMMIT" not in sql
    assert "ROLLBACK" in sql
    assert session.authority_present is True
    assert session.body_present is False
    assert session.committed_with_authority is False
    assert session.closed


@pytest.mark.parametrize("receipt", ("open", "close"))
def test_transaction_rejects_unverified_api_authority_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    receipt: str,
) -> None:
    target, plan = _target_and_plan()
    session = _FakeSession(
        plan,
        open_receipt_valid=receipt != "open",
        close_receipt_valid=receipt != "close",
    )
    database, _, _, _, _ = _database(monkeypatch, session=session)

    expected_error = (
        reconciliation.SchemaReconciliationError
        if receipt == "open"
        else reconciliation_db.PostgresProtocolError
    )
    expected_code = (
        reconciliation_db._AUTHORITY_PREFLIGHT_FAILURE_CODES[
            "current_user_is_session_user"
        ]
        if receipt == "open"
        else "schema_reconciliation_database_authority_survived"
    )
    with pytest.raises(expected_error, match=expected_code):
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ) as transaction:
            if receipt == "open":
                raise AssertionError("scope opened with invalid authority")
            transaction.lock_canonical_truth()
            transaction.execute_sql(plan.mutation_sql)

    assert session.committed_with_authority is False
    assert session.authority_present is True
    assert session.closed


@pytest.mark.parametrize(
    ("column", "error_code"),
    tuple(reconciliation_db._AUTHORITY_PREFLIGHT_FAILURE_CODES.items()),
)
def test_transaction_names_first_failed_authority_preflight_invariant(
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    error_code: str,
) -> None:
    target, plan = _target_and_plan()
    session = _FakeSession(plan, open_preflight_failure=column)
    database, _, _, _, _ = _database(monkeypatch, session=session)

    with pytest.raises(reconciliation.SchemaReconciliationError, match=error_code):
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ):
            raise AssertionError("scope opened with failed authority preflight")

    sql = [item[0] for item in session.queries]
    assert session.segments.authority_open not in sql
    assert "COMMIT" not in sql
    assert "ROLLBACK" in sql
    assert session.closed


def test_transaction_names_post_open_authority_drift_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, plan = _target_and_plan()
    column = "no_foreign_database_client_sessions"
    session = _FakeSession(plan, open_postflight_failure=column)
    database, _, _, _, _ = _database(monkeypatch, session=session)

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match=reconciliation_db._AUTHORITY_PREFLIGHT_FAILURE_CODES[column],
    ):
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ):
            raise AssertionError("scope opened with post-open authority drift")

    sql = [item[0] for item in session.queries]
    assert session.segments.authority_open in sql
    assert "COMMIT" not in sql
    assert "ROLLBACK" in sql
    assert session.closed


def test_transaction_maps_only_fixed_authority_server_preflight_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, plan = _target_and_plan()
    session = _FakeSession(
        plan,
        authority_open_server_preflight_failure=True,
    )
    database, _, _, _, _ = _database(monkeypatch, session=session)

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_authority_preflight_changed",
    ) as raised:
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ):
            raise AssertionError("scope opened after server preflight failure")

    assert raised.value.__cause__ is None
    assert reconciliation_db._AUTHORITY_OPEN_PREFLIGHT_SERVER_MESSAGE not in str(
        raised.value
    )
    sql = [item[0] for item in session.queries]
    assert "COMMIT" not in sql
    assert "ROLLBACK" in sql
    assert session.closed


def test_authority_preflight_rejects_non_boolean_receipt_without_reflection() -> None:
    result = QueryResult(
        reconciliation_db._AUTHORITY_OPEN_RECEIPT_COLUMNS,
        (
            tuple(
                "secret-server-value"
                for _column in reconciliation_db._AUTHORITY_OPEN_RECEIPT_COLUMNS
            ),
        ),
        "SELECT 1",
    )

    with pytest.raises(
        reconciliation_db.PostgresProtocolError,
        match="schema_reconciliation_database_authority_preflight_invalid",
    ) as raised:
        reconciliation_db._require_authority_preflight_receipt(result)

    assert "secret-server-value" not in str(raised.value)


@pytest.mark.parametrize("tamper", ("reordered", "split"))
def test_constructor_rejects_reordered_or_split_tampered_plan_sql(
    tamper: str,
) -> None:
    target, plan = _target_and_plan()
    segments = reconciliation_db._split_sealed_mutation_sql(plan)
    if tamper == "reordered":
        mutation_sql = (
            segments.body
            + "\n\n"
            + segments.authority_open
            + "\n\n"
            + segments.authority_close
        )
    else:
        mutation_sql = plan.mutation_sql.replace(
            "\n\n" + reconciliation_db._MUTATION_BODY_START,
            "\n" + reconciliation_db._MUTATION_BODY_START,
            1,
        )
    tampered = _plan_with_mutation(plan, mutation_sql)
    writer = _config(reconciliation_db.WRITER_LOGIN)
    admin = _config("muncho_canary_admin_aaaaaaaaaaaaaaaa")

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_database_sql_split_invalid",
    ):
        reconciliation_db.PostgresSchemaReconciliationDatabase(
            plan=tampered,
            target=target,
            admin_config=admin,
            writer_config=writer,
            managed_hba_receipt=_receipt(writer),
        )


def test_constructor_and_session_reject_wrong_authority_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, plan = _target_and_plan()
    writer = _config(reconciliation_db.WRITER_LOGIN)
    admin = _config("muncho_canary_admin_aaaaaaaaaaaaaaaa")
    receipt = _receipt(writer)

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_database_authority_invalid",
    ):
        reconciliation_db.PostgresSchemaReconciliationDatabase(
            plan=plan,
            target=target,
            admin_config=dataclasses.replace(admin, port=5433),
            writer_config=writer,
            managed_hba_receipt=receipt,
        )

    wrong = _FakeSession(plan, username="muncho_canary_admin_bbbbbbbbbbbbbbbb")
    database, _, _, _, _ = _database(monkeypatch, session=wrong)
    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_database_session_identity_invalid",
    ):
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ):
            pass
    assert wrong.closed


def test_attestation_subject_and_hba_receipt_bind_to_writer_config() -> None:
    writer = _config(reconciliation_db.WRITER_LOGIN)
    other = dataclasses.replace(writer, user="different_writer")

    with pytest.raises(
        PrivilegeAttestationError,
        match="attestation_subject_config_mismatch",
    ):
        _collect_privilege_attestation(
            object(),
            config=other,
            policy=reconciliation._target_policy(_target_and_plan()[0].attestation),
            managed_hba_receipt=_receipt(writer),
            subject_user=reconciliation_db.WRITER_LOGIN,
        )


def test_post_delete_terminal_uses_fresh_writer_contract_and_exact_ping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, plan = _target_and_plan()
    writer = _config(reconciliation_db.WRITER_LOGIN)
    managed_hba = _receipt(writer)
    truth = _truth_receipt()
    session = _FreshWriterSession()
    observations: list[dict[str, object]] = []

    def collect(_session, **kwargs):
        observations.append(kwargs)
        return target

    monkeypatch.setattr(reconciliation_db, "collect_schema_contract", collect)
    receipt = reconciliation_db.collect_post_delete_terminal_receipt(
        plan=plan,
        target=target,
        temporary_login="muncho_canary_admin_aaaaaaaaaaaaaaaa",
        writer_config=writer,
        managed_hba_receipt=managed_hba,
        pre_delete_canonical_truth=truth,
        observed_at_unix=managed_hba.observed_at_unix,
        _session_factory=lambda config: session,
    )

    assert receipt.writer_ping_verified is True
    assert receipt.writer_ping_request_id == (
        "schema-reconciliation-post-delete-terminal-v1"
    )
    assert receipt.writer_ping_response_sha256 == reconciliation._sha256_json(
        reconciliation_db._POST_DELETE_WRITER_PING_RESPONSE
    )
    assert receipt.canonical_truth_observed is False
    assert receipt.pre_delete_canonical_truth_receipt_sha256 == truth.sha256
    assert session.closed is True
    sql = [statement for statement, _ in session.queries]
    assert sql.index(reconciliation_db._POST_DELETE_WRITER_PING_SQL) < sql.index(
        "COMMIT"
    )
    assert observations == [
        {
            "config": writer,
            "policy": reconciliation._target_policy(target.attestation),
            "managed_hba_receipt": managed_hba,
            "subject_user": reconciliation_db.WRITER_LOGIN,
            "allow_missing_helper": False,
        }
    ]
    assert reconciliation_db.parse_post_delete_terminal_receipt(
        receipt.value
    ) == receipt
    assert reconciliation_db.validate_post_delete_terminal_receipt(
        receipt.value,
        plan=plan,
        target=target,
        temporary_login="muncho_canary_admin_aaaaaaaaaaaaaaaa",
        managed_hba_receipt=managed_hba,
        pre_delete_canonical_truth=truth,
    ) == receipt


def test_post_delete_terminal_rejects_ping_drift_and_closes_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, plan = _target_and_plan()
    writer = _config(reconciliation_db.WRITER_LOGIN)
    managed_hba = _receipt(writer)
    session = _FreshWriterSession(
        ping_response={
            **reconciliation_db._POST_DELETE_WRITER_PING_RESPONSE,
            "unexpected": True,
        }
    )
    monkeypatch.setattr(
        reconciliation_db,
        "collect_schema_contract",
        lambda *_args, **_kwargs: target,
    )

    with pytest.raises(
        reconciliation_db.PostgresProtocolError,
        match="schema_reconciliation_post_delete_writer_ping_invalid",
    ):
        reconciliation_db.collect_post_delete_terminal_receipt(
            plan=plan,
            target=target,
            temporary_login="muncho_canary_admin_aaaaaaaaaaaaaaaa",
            writer_config=writer,
            managed_hba_receipt=managed_hba,
            pre_delete_canonical_truth=_truth_receipt(),
            observed_at_unix=managed_hba.observed_at_unix,
            _session_factory=lambda config: session,
        )

    assert session.closed is True
    assert any(statement == "ROLLBACK" for statement, _ in session.queries)


@pytest.mark.parametrize(
    "failed_column",
    reconciliation_db._POST_DELETE_AUTHORITY_COLUMNS,
)
def test_post_delete_terminal_rejects_each_authority_absence_drift(
    monkeypatch: pytest.MonkeyPatch,
    failed_column: str,
) -> None:
    target, plan = _target_and_plan()
    writer = _config(reconciliation_db.WRITER_LOGIN)
    managed_hba = _receipt(writer)
    flags = ["t"] * len(reconciliation_db._POST_DELETE_AUTHORITY_COLUMNS)
    flags[
        reconciliation_db._POST_DELETE_AUTHORITY_COLUMNS.index(failed_column)
    ] = "f"
    session = _FreshWriterSession(authority_flags=tuple(flags))

    with pytest.raises(
        reconciliation_db.PostgresProtocolError,
        match="schema_reconciliation_post_delete_authority_present",
    ):
        reconciliation_db.collect_post_delete_terminal_receipt(
            plan=plan,
            target=target,
            temporary_login="muncho_canary_admin_aaaaaaaaaaaaaaaa",
            writer_config=writer,
            managed_hba_receipt=managed_hba,
            pre_delete_canonical_truth=_truth_receipt(),
            observed_at_unix=managed_hba.observed_at_unix,
            _session_factory=lambda config: session,
        )

    assert session.closed is True
    assert any(statement == "ROLLBACK" for statement, _ in session.queries)


@pytest.mark.parametrize("drift", ("username", "tls"))
def test_post_delete_terminal_rejects_fresh_session_identity_or_tls_drift(
    drift: str,
) -> None:
    target, plan = _target_and_plan()
    writer = _config(reconciliation_db.WRITER_LOGIN)
    managed_hba = _receipt(writer)
    session = _FreshWriterSession(
        username=("wrong_writer" if drift == "username" else writer.user),
        tls_peer_certificate_sha256=(
            "0" * 64
            if drift == "tls"
            else managed_hba.server_certificate_sha256
        ),
    )

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_post_delete_terminal_session_invalid",
    ):
        reconciliation_db.collect_post_delete_terminal_receipt(
            plan=plan,
            target=target,
            temporary_login="muncho_canary_admin_aaaaaaaaaaaaaaaa",
            writer_config=writer,
            managed_hba_receipt=managed_hba,
            pre_delete_canonical_truth=_truth_receipt(),
            observed_at_unix=managed_hba.observed_at_unix,
            _session_factory=lambda config: session,
        )

    assert session.closed is True
    assert session.queries == []


@pytest.mark.parametrize("failure", ("collector_exception", "contract_drift"))
def test_post_delete_terminal_rolls_back_and_closes_on_contract_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    target, plan = _target_and_plan()
    writer = _config(reconciliation_db.WRITER_LOGIN)
    managed_hba = _receipt(writer)
    session = _FreshWriterSession()
    old_contract = reconciliation.SchemaContract.from_mapping(
        reconciliation._old_contract_value(target)
    )

    def collect(*_args, **_kwargs):
        if failure == "collector_exception":
            raise RuntimeError("collector failed")
        return old_contract

    monkeypatch.setattr(reconciliation_db, "collect_schema_contract", collect)
    expected = (
        "collector failed"
        if failure == "collector_exception"
        else "schema_reconciliation_post_delete_contract_invalid"
    )
    error_type = (
        RuntimeError
        if failure == "collector_exception"
        else reconciliation.SchemaReconciliationError
    )
    with pytest.raises(error_type, match=expected):
        reconciliation_db.collect_post_delete_terminal_receipt(
            plan=plan,
            target=target,
            temporary_login="muncho_canary_admin_aaaaaaaaaaaaaaaa",
            writer_config=writer,
            managed_hba_receipt=managed_hba,
            pre_delete_canonical_truth=_truth_receipt(),
            observed_at_unix=managed_hba.observed_at_unix,
            _session_factory=lambda config: session,
        )

    assert session.closed is True
    assert any(statement == "ROLLBACK" for statement, _ in session.queries)


def test_post_delete_terminal_parser_rejects_resigned_ping_or_type_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, plan = _target_and_plan()
    writer = _config(reconciliation_db.WRITER_LOGIN)
    managed_hba = _receipt(writer)
    truth = _truth_receipt()
    session = _FreshWriterSession()
    monkeypatch.setattr(
        reconciliation_db,
        "collect_schema_contract",
        lambda *_args, **_kwargs: target,
    )
    receipt = reconciliation_db.collect_post_delete_terminal_receipt(
        plan=plan,
        target=target,
        temporary_login="muncho_canary_admin_aaaaaaaaaaaaaaaa",
        writer_config=writer,
        managed_hba_receipt=managed_hba,
        pre_delete_canonical_truth=truth,
        observed_at_unix=managed_hba.observed_at_unix,
        _session_factory=lambda config: session,
    )

    tampered = dict(receipt.value)
    tampered["writer_ping_response_sha256"] = "0" * 64
    unsigned = {
        key: value for key, value in tampered.items() if key != "receipt_sha256"
    }
    tampered["receipt_sha256"] = reconciliation._sha256_json(unsigned)
    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_post_delete_terminal_invalid",
    ):
        reconciliation_db.parse_post_delete_terminal_receipt(tampered)

    wrong_type = dict(receipt.value)
    wrong_type["release_revision"] = None
    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_post_delete_terminal_invalid",
    ):
        reconciliation_db.parse_post_delete_terminal_receipt(wrong_type)
