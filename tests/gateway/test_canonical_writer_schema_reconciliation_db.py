from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gateway import canonical_writer_schema_reconciliation as reconciliation
from gateway import canonical_writer_schema_reconciliation_db as database_module
from gateway import canonical_writer_schema_reconciliation_control as control
from gateway.canonical_writer_db import (
    CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY,
    CredentialSource,
    ManagedCloudSQLAdminHBAReceipt,
    QueryResult,
    WriterDBConfig,
)
from gateway.canonical_writer_foundation import _load_source_artifacts_for_tests


REVISION = "a" * 40
EXECUTOR_LOGIN = "muncho_canary_reconciler_" + "a" * 16
CONTROL_INSTALL_SHA256 = (
    "5519c3510c6211b995aec4ca39057c552869501474fab4dea88e6bb21f090ac5"
)
CONTROL_RETIRE_SHA256 = (
    "ec3ebe8713a378f7c721a20a303074ae99211855a9925ba108f57e8c84606de8"
)
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
        control_install_artifact_sha256=CONTROL_INSTALL_SHA256,
        control_retire_artifact_sha256=CONTROL_RETIRE_SHA256,
    )
    return asset.contract, plan


def _old(target: reconciliation.SchemaContract) -> reconciliation.SchemaContract:
    return reconciliation.SchemaContract.from_mapping(
        reconciliation._old_contract_value(target)
    )


def _config(user: str) -> WriterDBConfig:
    return WriterDBConfig(
        host="127.0.0.1",
        tls_server_name="localhost",
        port=5432,
        database=reconciliation.DATABASE,
        user=user,
        ca_file=Path("/tmp/muncho-canary-test-ca.pem"),
        credential=CredentialSource(fd=0, expected_uid=os.getuid()),
        application_name="muncho-schema-reconciliation-test",
    )


def _managed_hba(
    config: WriterDBConfig,
    *,
    certificate_sha256: str = "e" * 64,
) -> ManagedCloudSQLAdminHBAReceipt:
    now = int(time.time())
    return ManagedCloudSQLAdminHBAReceipt(
        version="managed-cloudsqladmin-hba-rejection-v2",
        host=config.host,
        tls_server_name=config.tls_server_name,
        port=config.port,
        server_certificate_sha256=certificate_sha256,
        database="cloudsqladmin",
        user=config.user,
        observed_at_unix=now,
        expires_at_unix=now + 300,
        sqlstate="28000",
        server_message=(
            f'no pg_hba.conf entry for host "{config.host}", user '
            f'"{config.user}", database "cloudsqladmin", SSL encryption'
        ),
        result="pg_hba_rejected",
        tls_peer_verified=True,
    )


def _truth() -> reconciliation.CanonicalTruthReceipt:
    return reconciliation.CanonicalTruthReceipt(
        row_count=3,
        canonical14_sha256="d" * 64,
        relation_receipts=tuple(
            reconciliation.CanonicalRelationTruthReceipt(
                relation=relation,
                row_count=3 if index == 0 else 0,
                chunk_count=1 if index == 0 else 0,
                chunk_manifest_sha256=hashlib.sha256(
                    f"{relation}:{index}".encode()
                ).hexdigest(),
            )
            for index, relation in enumerate(
                reconciliation.CANONICAL_TRUTH_RELATIONS
            )
        ),
        quarantine_anchor_receipts=tuple(
            reconciliation.CanonicalQuarantineAnchorReceipt(
                anchor=anchor,
                object_oid=9_000 + index,
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


def _observer_result(truth: reconciliation.CanonicalTruthReceipt) -> QueryResult:
    relations = json.dumps(
        truth.value["relation_receipts"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    quarantine = json.dumps(
        truth.value["quarantine_anchors"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    observation = hashlib.sha256(
        "\n".join(
            (
                "canonical-writer-schema-reconciliation-control-observation-v1",
                str(truth.row_count),
                truth.canonical14_sha256,
                relations,
                quarantine,
            )
        ).encode()
    ).hexdigest()
    return QueryResult(
        (
            "row_count",
            "canonical14_sha256",
            "relation_receipts",
            "quarantine_anchor_receipts",
            "observation_sha256",
        ),
        ((str(truth.row_count), truth.canonical14_sha256, relations, quarantine, observation),),
        "SELECT 1",
    )


class _ExecutorSession:
    def __init__(
        self,
        plan: reconciliation.SchemaReconciliationPlan,
        *,
        authority_failure: str | None = None,
        postflight_failure: str | None = None,
        stabilization_failures: tuple[tuple[str, ...], ...] = (),
        apply_applied: bool = True,
        username: str = EXECUTOR_LOGIN,
    ) -> None:
        self.plan = plan
        self.username = username
        self.tls_peer_certificate_sha256 = "e" * 64
        self.authority_failure = authority_failure
        self.postflight_failure = postflight_failure
        self.stabilization_failures = stabilization_failures
        self.apply_applied = apply_applied
        self.queries: list[str] = []
        self.closed = False
        self.transaction_open = False
        self.session_lock_held = False
        self.stabilization_count = 0
        self.transaction_receipt_count = 0
        self.applied = False
        self.bindings: dict[str, str] = {}
        self.observer_result = _observer_result(_truth())

    def _authority_result(self) -> QueryResult:
        values = ["t"] * len(database_module._AUTHORITY_OPEN_RECEIPT_COLUMNS)
        failures: tuple[str, ...] = ()
        if not self.transaction_open:
            self.stabilization_count += 1
            if self.stabilization_failures:
                failures = self.stabilization_failures[
                    min(
                        self.stabilization_count - 1,
                        len(self.stabilization_failures) - 1,
                    )
                ]
        else:
            self.transaction_receipt_count += 1
            if self.transaction_receipt_count == 1 and self.authority_failure:
                failures = (self.authority_failure,)
            elif self.transaction_receipt_count > 1 and self.postflight_failure:
                failures = (self.postflight_failure,)
        for failure in failures:
            values[
                database_module._AUTHORITY_OPEN_RECEIPT_COLUMNS.index(failure)
            ] = "f"
        return QueryResult(
            database_module._AUTHORITY_OPEN_RECEIPT_COLUMNS,
            (tuple(values),),
            "SELECT 1",
        )

    def _apply_result(self) -> QueryResult:
        observation = self.observer_result.rows[0][-1]
        plan = self.bindings[control.PLAN_GUC]
        intent = self.bindings[control.AUTHORIZED_INTENT_GUC]
        truth = self.bindings[control.TRUTH_RECEIPT_GUC]
        applied_text = "true" if self.apply_applied else "false"
        helper = (
            reconciliation.EXPECTED_MISSING_HELPER_CATALOG_IDENTITY
            .definition_sha256
        )
        receipt = hashlib.sha256(
            "\n".join(
                (
                    "canonical-writer-schema-reconciliation-control-apply-v1",
                    applied_text,
                    plan,
                    intent,
                    truth,
                    observation,
                    helper,
                )
            ).encode()
        ).hexdigest()
        self.applied = self.apply_applied
        return QueryResult(
            (
                "applied",
                "plan_sha256",
                "authorized_intent_sha256",
                "canonical_truth_receipt_sha256",
                "observation_sha256",
                "helper_definition_sha256",
                "receipt_sha256",
            ),
            ((applied_text, plan, intent, truth, observation, helper, receipt),),
            "SELECT 1",
        )

    def query(self, sql: str, *, maximum_rows: int) -> QueryResult:
        self.queries.append(sql)
        if sql.startswith("SELECT pg_catalog.pg_advisory_lock("):
            return QueryResult(("pg_advisory_lock",), (("",),), "SELECT 1")
        if sql.startswith("SELECT pg_catalog.pg_advisory_unlock("):
            return QueryResult(("pg_advisory_unlock",), (("t",),), "SELECT 1")
        if sql == "BEGIN ISOLATION LEVEL SERIALIZABLE":
            self.transaction_open = True
            return QueryResult((), (), "BEGIN")
        if sql == database_module._AUTHORITY_OPEN_RECEIPT_SQL:
            return self._authority_result()
        if sql == control.OBSERVER_CALL_SQL:
            return self.observer_result
        if sql.startswith("SET LOCAL muncho.schema_reconciliation_"):
            name, value = sql.removeprefix("SET LOCAL ").split(" = ", 1)
            self.bindings[name] = value.strip("'")
            return QueryResult((), (), "SET")
        if sql == control.APPLY_CALL_SQL:
            return self._apply_result()
        if sql.startswith("SET "):
            return QueryResult((), (), "SET")
        if sql == "COMMIT":
            self.transaction_open = False
            return QueryResult((), (), "COMMIT")
        if sql == "ROLLBACK":
            self.transaction_open = False
            self.applied = False
            return QueryResult((), (), "ROLLBACK")
        raise AssertionError(f"unexpected SQL: {sql[:120]}")

    def close(self) -> None:
        self.transaction_open = False
        self.closed = True


def _database(
    monkeypatch: pytest.MonkeyPatch,
    *,
    initial_target: bool = False,
    session: _ExecutorSession | None = None,
    sleep: Callable[[float], None] = time.sleep,
    admission: Callable[[], None] = lambda: None,
) -> tuple[
    database_module.PostgresSchemaReconciliationDatabase,
    _ExecutorSession,
    reconciliation.SchemaContract,
    reconciliation.SchemaReconciliationPlan,
]:
    target, plan = _target_and_plan()
    executor_config = _config(EXECUTOR_LOGIN)
    writer_config = _config(database_module.WRITER_LOGIN)
    active = session or _ExecutorSession(plan)

    def collect(*_args: Any, **_kwargs: Any) -> reconciliation.SchemaContract:
        if initial_target or active.applied:
            return target
        return _old(target)

    monkeypatch.setattr(database_module, "collect_schema_contract", collect)
    database = database_module.PostgresSchemaReconciliationDatabase(
        plan=plan,
        target=target,
        executor_config=executor_config,
        writer_config=writer_config,
        managed_hba_receipt=_managed_hba(writer_config),
        executor_managed_hba_receipt=_managed_hba(executor_config),
        pre_begin_admission=admission,
        _session_factory=lambda config: (
            active
            if config == executor_config
            else (_ for _ in ()).throw(AssertionError("wrong config"))
        ),
        _stabilization_sleep=sleep,
    )
    return database, active, target, plan


def test_transaction_calls_only_fixed_observer_and_apply_routines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, session, target, plan = _database(monkeypatch)
    intent = "f" * 64

    with database.transaction(
        advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
    ) as transaction:
        transaction.lock_canonical_truth()
        assert transaction.observe_contract() == _old(target)
        assert transaction.observe_canonical_truth() == _truth()
        transaction.apply_missing_helper(authorized_intent_sha256=intent)
        assert transaction.observe_contract() == target
        assert transaction.observe_canonical_truth() == _truth()

    assert session.queries.count(control.OBSERVER_CALL_SQL) == 2
    assert session.queries.count(control.APPLY_CALL_SQL) == 1
    assert session.bindings == {
        control.PLAN_GUC: plan.sha256,
        control.AUTHORIZED_INTENT_GUC: intent,
        control.TRUTH_RECEIPT_GUC: _truth().sha256,
    }
    assert session.queries.count(database_module._AUTHORITY_OPEN_RECEIPT_SQL) == 3
    assert "COMMIT" in session.queries
    assert session.closed is True
    assert all("CREATE FUNCTION" not in sql for sql in session.queries)


def test_exact_target_commits_without_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, session, target, _plan = _database(
        monkeypatch,
        initial_target=True,
    )
    with database.transaction(
        advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
    ) as transaction:
        transaction.lock_canonical_truth()
        assert transaction.observe_contract() == target
        assert transaction.observe_canonical_truth() == _truth()

    assert control.APPLY_CALL_SQL not in session.queries
    assert "COMMIT" in session.queries


def test_apply_false_receipt_fails_closed_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _target, plan = _target_and_plan()
    session = _ExecutorSession(plan, apply_applied=False)
    database, _, _, _ = _database(monkeypatch, session=session)

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_control_apply_not_performed",
    ):
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ) as transaction:
            transaction.lock_canonical_truth()
            transaction.apply_missing_helper(authorized_intent_sha256="f" * 64)

    assert "ROLLBACK" in session.queries
    assert "COMMIT" not in session.queries


def test_transaction_exception_rolls_back_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, session, _target, _plan = _database(monkeypatch)
    with pytest.raises(RuntimeError, match="injected"):
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ) as transaction:
            transaction.lock_canonical_truth()
            raise RuntimeError("injected")
    assert "ROLLBACK" in session.queries
    assert "COMMIT" not in session.queries
    assert session.closed is True


def test_session_lock_wait_is_bounded_before_admission_or_begin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _target, plan = _target_and_plan()

    class _BlockedExecutorSession(_ExecutorSession):
        def query(self, sql: str, *, maximum_rows: int) -> QueryResult:
            if sql.startswith("SELECT pg_catalog.pg_advisory_lock("):
                self.queries.append(sql)
                raise database_module.PostgresProtocolError(
                    "database_error_sqlstate:55P03"
                )
            return super().query(sql, maximum_rows=maximum_rows)

    session = _BlockedExecutorSession(plan)
    admissions: list[str] = []
    database, _, _, _ = _database(
        monkeypatch,
        session=session,
        admission=lambda: admissions.append("admitted"),
    )

    with pytest.raises(
        database_module.PostgresProtocolError,
        match="database_error_sqlstate:55P03",
    ):
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ):
            pytest.fail("scope opened")

    assert session.queries[:2] == [
        "SET lock_timeout = '15s'",
        "SET statement_timeout = '2min'",
    ]
    assert admissions == []
    assert "BEGIN ISOLATION LEVEL SERIALIZABLE" not in session.queries
    assert control.APPLY_CALL_SQL not in session.queries
    assert "COMMIT" not in session.queries
    assert session.closed is True


def test_post_lock_admission_is_immediately_before_begin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _target, plan = _target_and_plan()
    session = _ExecutorSession(plan)

    def admit() -> None:
        assert session.queries[-1] == database_module._AUTHORITY_OPEN_RECEIPT_SQL
        assert "BEGIN ISOLATION LEVEL SERIALIZABLE" not in session.queries
        session.queries.append("trusted-pre-begin-admission")

    database, _, _, _ = _database(
        monkeypatch,
        initial_target=True,
        session=session,
        admission=admit,
    )
    with database.transaction(
        advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
    ) as transaction:
        transaction.lock_canonical_truth()

    marker = session.queries.index("trusted-pre-begin-admission")
    assert session.queries[marker - 1] == database_module._AUTHORITY_OPEN_RECEIPT_SQL
    assert session.queries[marker + 1] == "BEGIN ISOLATION LEVEL SERIALIZABLE"


def test_post_lock_admission_failure_unlocks_without_begin_or_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = "schema_reconciliation_runtime_pre_begin_stopped_boundary_drifted"

    def reject() -> None:
        raise reconciliation.SchemaReconciliationError(code)

    database, session, _, _ = _database(
        monkeypatch,
        admission=reject,
    )
    with pytest.raises(reconciliation.SchemaReconciliationError, match=code):
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ):
            pytest.fail("scope opened")

    assert "BEGIN ISOLATION LEVEL SERIALIZABLE" not in session.queries
    assert control.APPLY_CALL_SQL not in session.queries
    assert "COMMIT" not in session.queries
    assert any(
        sql.startswith("SELECT pg_catalog.pg_advisory_unlock(")
        for sql in session.queries
    )
    assert session.closed is True


@pytest.mark.parametrize("awaitable", (False, True))
def test_post_lock_admission_rejects_non_exact_none_return(
    monkeypatch: pytest.MonkeyPatch,
    awaitable: bool,
) -> None:
    async def async_result() -> None:
        return None

    callback = async_result if awaitable else lambda: object()
    database, session, _, _ = _database(
        monkeypatch,
        admission=callback,
    )
    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_database_admission_invalid",
    ):
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ):
            pytest.fail("scope opened")
    assert "BEGIN ISOLATION LEVEL SERIALIZABLE" not in session.queries
    assert session.closed is True


@pytest.mark.parametrize(
    ("column", "error_code"),
    tuple(database_module._AUTHORITY_PREFLIGHT_FAILURE_CODES.items()),
)
def test_each_authority_invariant_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    error_code: str,
) -> None:
    _target, plan = _target_and_plan()
    session = _ExecutorSession(plan, authority_failure=column)
    database, _, _, _ = _database(monkeypatch, session=session)
    with pytest.raises(reconciliation.SchemaReconciliationError, match=error_code):
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ):
            pytest.fail("scope opened")
    assert "ROLLBACK" in session.queries


def test_only_missing_provider_edge_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _target, plan = _target_and_plan()
    session = _ExecutorSession(
        plan,
        stabilization_failures=(("provider_executor_edge_exact",), ()),
    )
    sleeps: list[float] = []
    database, _, _, _ = _database(
        monkeypatch,
        initial_target=True,
        session=session,
        sleep=sleeps.append,
    )
    with database.transaction(
        advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
    ) as transaction:
        transaction.lock_canonical_truth()
    assert sleeps == [database_module._ROLE_GRAPH_STABILIZATION_INTERVAL_SECONDS]


def test_unsafe_authority_drift_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _target, plan = _target_and_plan()
    session = _ExecutorSession(
        plan,
        stabilization_failures=(("control_schema_acl_surface_exact",),),
    )
    database, _, _, _ = _database(
        monkeypatch,
        session=session,
        sleep=lambda _seconds: pytest.fail("unsafe drift retried"),
    )
    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_authority_control_schema_acl_surface_exact",
    ):
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ):
            pytest.fail("scope opened")


def test_postflight_authority_drift_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _target, plan = _target_and_plan()
    session = _ExecutorSession(
        plan,
        postflight_failure="temporary_executor_has_zero_shared_dependencies",
    )
    database, _, _, _ = _database(monkeypatch, session=session)
    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_authority_preflight_changed",
    ):
        with database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ) as transaction:
            transaction.lock_canonical_truth()
    assert "ROLLBACK" in session.queries


def test_non_boolean_authority_receipt_does_not_reflect_server_value() -> None:
    result = QueryResult(
        database_module._AUTHORITY_OPEN_RECEIPT_COLUMNS,
        (("secret-server-value",) * len(database_module._AUTHORITY_OPEN_RECEIPT_COLUMNS),),
        "SELECT 1",
    )
    with pytest.raises(
        database_module.PostgresProtocolError,
        match="schema_reconciliation_database_authority_preflight_invalid",
    ) as raised:
        database_module._require_authority_preflight_receipt(result)
    assert "secret-server-value" not in str(raised.value)


@pytest.mark.parametrize(
    "required_fragment",
    (
        "owner_role.rolconnlimit = -1",
        "writer_role.rolvaliduntil IS NULL",
        "objsubid = 0",
        "deptype = 'a' AND objsubid = 0",
        "executor_shared_dependencies",
        "dbid = 0",
        "has_database_privilege",
        "managed_actual_database_acl",
        "managed_expected_database_acl",
        "managed_cloudsqladmin_exception",
        "database.datname = 'cloudsqladmin'",
        "cloudsqladmin,muncho_canary_brain,postgres",
        "cloudsqladmin,muncho_canary_brain,postgres,template1",
        "connectable_database_inventory_exact",
        "connectable_non_template_database_inventory_exact",
        "actual_control_schema_acl",
        "expected_control_schema_acl",
        "actual_control_routine_acl",
        "expected_control_routine_acl",
        "pg_catalog.pg_constraint",
        "pg_catalog.pg_extension",
        "pg_catalog.pg_default_acl",
        "pg_catalog.pg_publication_namespace",
        "control_namespace_other_object_inventory_empty",
        "temporary_executor_has_zero_shared_dependencies",
        "routeback_helper_name_inventory_bounded",
        "proargtypes[0]",
    ),
)
def test_active_authority_sql_binds_hardened_surface(
    required_fragment: str,
) -> None:
    assert required_fragment in database_module._AUTHORITY_OPEN_RECEIPT_SQL


def test_active_authority_requires_prepared_transactions_disabled_and_empty() -> None:
    sql = database_module._AUTHORITY_OPEN_RECEIPT_SQL
    assert "current_setting('max_prepared_transactions')::integer = 0" in sql
    assert "NOT EXISTS (SELECT 1 FROM pg_catalog.pg_prepared_xacts)" in sql


def test_authority_helper_identity_does_not_require_schema_usage() -> None:
    active_sql = database_module._AUTHORITY_OPEN_RECEIPT_SQL
    post_delete_sql = database_module._post_delete_authority_absence_sql(
        EXECUTOR_LOGIN
    )
    for sql in (active_sql, post_delete_sql):
        assert "to_regprocedure" not in sql
        assert "prokind = 'f' AND pronargs = 1" in sql
        assert "proargtypes[0]" in sql
        assert "argument_type.typname = 'jsonb'" in sql


def test_active_authority_rejects_executor_sessions_cross_database() -> None:
    sql = database_module._AUTHORITY_OPEN_RECEIPT_SQL
    assert "activity.usesysid = temporary_login.oid" in sql
    assert "activity.pid <> pg_catalog.pg_backend_pid()" in sql


def test_verified_cloud_shape_database_inventory_is_exact_and_fail_closed() -> None:
    non_template_shape = (
        "cloudsqladmin",
        "muncho_canary_brain",
        "postgres",
    )
    all_connectable_shape = (*non_template_shape, "template1")
    active_sql = database_module._AUTHORITY_OPEN_RECEIPT_SQL
    post_delete_sql = database_module._post_delete_authority_absence_sql(
        EXECUTOR_LOGIN
    )
    for expected_inventory in (
        ",".join(sorted(non_template_shape)),
        ",".join(sorted(all_connectable_shape)),
    ):
        assert expected_inventory in active_sql
        assert expected_inventory in post_delete_sql
    for sql, end_marker in (
        (active_sql, "AS executor_schema_acl_exact"),
        (post_delete_sql, "AS routeback_helper_name_inventory_exact"),
    ):
        effective_scan = sql.split(
            "AS connectable_non_template_database_inventory_exact,", 1
        )[1].split(end_marker, 1)[0]
        assert "WHERE database.datallowconn\n" in effective_scan
        assert "WHERE database.datallowconn AND NOT database.datistemplate" not in (
            effective_scan
        )

    values = ["t"] * len(database_module._AUTHORITY_OPEN_RECEIPT_COLUMNS)
    values[
        database_module._AUTHORITY_OPEN_RECEIPT_COLUMNS.index(
            "connectable_database_inventory_exact"
        )
    ] = "f"
    result = QueryResult(
        database_module._AUTHORITY_OPEN_RECEIPT_COLUMNS,
        (tuple(values),),
        "SELECT 1",
    )
    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match=(
            "schema_reconciliation_authority_"
            "connectable_database_inventory_exact"
        ),
    ):
        database_module._require_authority_preflight_receipt(result)


def test_constructor_rejects_old_admin_or_owner_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, plan = _target_and_plan()
    writer = _config(database_module.WRITER_LOGIN)
    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_database_authority_invalid",
    ):
        database_module.PostgresSchemaReconciliationDatabase(
            plan=plan,
            target=target,
            executor_config=_config("muncho_canary_admin_" + "a" * 16),
            writer_config=writer,
            managed_hba_receipt=_managed_hba(writer),
            executor_managed_hba_receipt=_managed_hba(
                _config("muncho_canary_admin_" + "a" * 16)
            ),
            pre_begin_admission=lambda: None,
            _session_factory=lambda _config: pytest.fail("opened"),
        )


@pytest.mark.parametrize("drift", ("user", "certificate"))
def test_constructor_requires_executor_specific_managed_hba_proof(
    drift: str,
) -> None:
    target, plan = _target_and_plan()
    executor = _config(EXECUTOR_LOGIN)
    writer = _config(database_module.WRITER_LOGIN)
    executor_hba = (
        _managed_hba(writer)
        if drift == "user"
        else _managed_hba(executor, certificate_sha256="f" * 64)
    )

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_database_hba_receipt_invalid",
    ):
        database_module.PostgresSchemaReconciliationDatabase(
            plan=plan,
            target=target,
            executor_config=executor,
            writer_config=writer,
            managed_hba_receipt=_managed_hba(writer),
            executor_managed_hba_receipt=executor_hba,
            pre_begin_admission=lambda: None,
            _session_factory=lambda _config: pytest.fail("opened"),
        )


class _WriterSession:
    def __init__(
        self,
        *,
        authority_flags: tuple[str, ...] | None = None,
        ping_response: Any | None = None,
        username: str = database_module.WRITER_LOGIN,
        tls_peer: str = "e" * 64,
    ) -> None:
        self.username = username
        self.tls_peer_certificate_sha256 = tls_peer
        self.authority_flags = authority_flags or (
            ("t",) * len(database_module._POST_DELETE_AUTHORITY_COLUMNS)
        )
        self.ping_response = (
            database_module._POST_DELETE_WRITER_PING_RESPONSE
            if ping_response is None
            else ping_response
        )
        self.queries: list[str] = []
        self.closed = False
        self.transaction_open = False

    def query(self, sql: str, *, maximum_rows: int) -> QueryResult:
        self.queries.append(sql)
        if sql == "BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY":
            self.transaction_open = True
            return QueryResult((), (), "BEGIN")
        if sql.startswith("SET "):
            return QueryResult((), (), "SET")
        if sql.startswith("SELECT pg_catalog.pg_advisory_lock_shared("):
            self.session_lock_held = True
            return QueryResult(
                ("pg_advisory_lock_shared",), (("",),), "SELECT 1"
            )
        if sql.startswith("SELECT pg_catalog.pg_advisory_unlock_shared("):
            was_held = self.session_lock_held
            self.session_lock_held = False
            return QueryResult(
                ("pg_advisory_unlock_shared",),
                (("t" if was_held else "f",),),
                "SELECT 1",
            )
        if sql.startswith("WITH executor AS ("):
            return QueryResult(
                database_module._POST_DELETE_AUTHORITY_COLUMNS,
                (self.authority_flags,),
                "SELECT 1",
            )
        if sql == database_module._POST_DELETE_WRITER_PING_SQL:
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
        raise AssertionError(f"unexpected SQL: {sql[:120]}")

    def close(self) -> None:
        self.transaction_open = False
        self.closed = True


def _collect_post_delete(
    monkeypatch: pytest.MonkeyPatch,
    *,
    session: _WriterSession | None = None,
) -> tuple[
    database_module.PostDeleteTerminalReceipt,
    _WriterSession,
    reconciliation.SchemaReconciliationPlan,
]:
    target, plan = _target_and_plan()
    writer = _config(database_module.WRITER_LOGIN)
    hba = _managed_hba(writer)
    active = session or _WriterSession()
    monkeypatch.setattr(
        database_module,
        "collect_schema_contract",
        lambda *_args, **_kwargs: target,
    )
    receipt = database_module.collect_post_delete_terminal_receipt(
        plan=plan,
        target=target,
        temporary_executor_login=EXECUTOR_LOGIN,
        writer_config=writer,
        managed_hba_receipt=hba,
        pre_delete_canonical_truth=_truth(),
        observed_at_unix=hba.observed_at_unix,
        _session_factory=lambda config: (
            active
            if config == writer
            else (_ for _ in ()).throw(AssertionError("wrong config"))
        ),
    )
    return receipt, active, plan


def test_post_delete_terminal_proves_dormant_executor_and_writer_ping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, session, plan = _collect_post_delete(monkeypatch)
    assert receipt.temporary_executor_absent is True
    assert receipt.temporary_executor_inventory_empty is True
    assert receipt.prepared_transactions_disabled_and_empty is True
    assert receipt.persistent_executor_memberships_empty is True
    assert receipt.persistent_executor_owns_zero_objects_clusterwide is True
    assert receipt.connectable_database_inventory_exact is True
    assert receipt.connectable_non_template_database_inventory_exact is True
    assert receipt.persistent_executor_database_effective_privileges_exact is True
    assert receipt.routeback_helper_name_inventory_exact is True
    assert receipt.control_schema_identity_acl_exact is True
    assert receipt.control_namespace_other_object_inventory_empty is True
    assert receipt.control_routine_identity_acl_exact is True
    assert receipt.control_foundation_contract_sha256 == plan.value[
        "control_foundation_contract_sha256"
    ]
    assert session.closed is True
    assert "COMMIT" in session.queries
    lock_index = next(
        index
        for index, sql in enumerate(session.queries)
        if sql.startswith("SELECT pg_catalog.pg_advisory_lock_shared(")
    )
    begin_index = session.queries.index(
        "BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY"
    )
    commit_index = session.queries.index("COMMIT")
    unlock_index = next(
        index
        for index, sql in enumerate(session.queries)
        if sql.startswith("SELECT pg_catalog.pg_advisory_unlock_shared(")
    )
    assert lock_index < begin_index < commit_index < unlock_index
    assert session.session_lock_held is False


def test_post_delete_shared_lock_wait_is_bounded_before_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BlockedWriterSession(_WriterSession):
        def query(self, sql: str, *, maximum_rows: int) -> QueryResult:
            if sql.startswith("SELECT pg_catalog.pg_advisory_lock_shared("):
                self.queries.append(sql)
                raise database_module.PostgresProtocolError(
                    "database_error_sqlstate:55P03"
                )
            return super().query(sql, maximum_rows=maximum_rows)

    session = _BlockedWriterSession()
    with pytest.raises(
        database_module.PostgresProtocolError,
        match="database_error_sqlstate:55P03",
    ):
        _collect_post_delete(monkeypatch, session=session)

    assert session.queries[:2] == [
        "SET lock_timeout = '15s'",
        "SET statement_timeout = '2min'",
    ]
    assert "BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY" not in session.queries
    assert "COMMIT" not in session.queries
    assert session.closed is True


@pytest.mark.parametrize(
    "failed_index",
    range(len(database_module._POST_DELETE_AUTHORITY_COLUMNS)),
)
def test_post_delete_rejects_each_dormant_authority_drift(
    monkeypatch: pytest.MonkeyPatch,
    failed_index: int,
) -> None:
    flags = ["t"] * len(database_module._POST_DELETE_AUTHORITY_COLUMNS)
    flags[failed_index] = "f"
    session = _WriterSession(authority_flags=tuple(flags))
    with pytest.raises(
        database_module.PostgresProtocolError,
        match="schema_reconciliation_post_delete_authority_present",
    ):
        _collect_post_delete(monkeypatch, session=session)
    assert "ROLLBACK" in session.queries
    assert session.closed is True


def test_post_delete_rejects_writer_ping_drift_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _WriterSession(ping_response={"ok": False})
    with pytest.raises(
        database_module.PostgresProtocolError,
        match="schema_reconciliation_post_delete_writer_ping_invalid",
    ):
        _collect_post_delete(monkeypatch, session=session)
    assert "ROLLBACK" in session.queries
    assert session.closed is True


@pytest.mark.parametrize("drift", ("username", "tls"))
def test_post_delete_rejects_writer_session_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    session = _WriterSession(
        username="wrong" if drift == "username" else database_module.WRITER_LOGIN,
        tls_peer="f" * 64 if drift == "tls" else "e" * 64,
    )
    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_post_delete_terminal_session_invalid",
    ):
        _collect_post_delete(monkeypatch, session=session)
    assert session.closed is True


def test_post_delete_parser_rejects_resigned_boolean_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, _session, _plan = _collect_post_delete(monkeypatch)
    tampered = copy.deepcopy(dict(receipt.value))
    tampered["persistent_executor_memberships_empty"] = 1
    unsigned = {
        key: value for key, value in tampered.items() if key != "receipt_sha256"
    }
    tampered["receipt_sha256"] = reconciliation._sha256_json(unsigned)
    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_post_delete_terminal_invalid",
    ):
        database_module.parse_post_delete_terminal_receipt(tampered)


@pytest.mark.parametrize(
    "required_fragment",
    (
        "temporary_executor_inventory_empty",
        "prepared_transactions_disabled_and_empty",
        "current_setting('max_prepared_transactions')::integer = 0",
        "persistent_executor_memberships_empty",
        "persistent_executor_owns_zero_objects_clusterwide",
        "persistent_executor_acl_dependencies_exact",
        "deptype = 'a' AND objsubid = 0",
        "executor_shared_dependencies",
        "persistent_executor_database_effective_privileges_exact",
        "cloudsqladmin,muncho_canary_brain,postgres",
        "cloudsqladmin,muncho_canary_brain,postgres,template1",
        "connectable_database_inventory_exact",
        "connectable_non_template_database_inventory_exact",
        "managed_actual_database_acl",
        "managed_expected_database_acl",
        "managed_cloudsqladmin_exception",
        "routeback_helper_name_inventory_exact",
        "proargtypes[0]",
        "control_schema_identity_acl_exact",
        "pg_catalog.pg_constraint",
        "pg_catalog.pg_extension",
        "pg_catalog.pg_default_acl",
        "pg_catalog.pg_publication_namespace",
        "pg_catalog.pg_event_trigger",
        "control_namespace_other_object_inventory_empty",
        "control_routine_identity_acl_exact",
        control.OBSERVER_PROSRC_SHA256,
        control.OBSERVER_DEFINITION_SHA256,
        control.APPLY_PROSRC_SHA256,
        control.APPLY_DEFINITION_SHA256,
    ),
)
def test_post_delete_sql_binds_complete_dormant_control_surface(
    required_fragment: str,
) -> None:
    sql = database_module._post_delete_authority_absence_sql(EXECUTOR_LOGIN)
    assert required_fragment in sql
