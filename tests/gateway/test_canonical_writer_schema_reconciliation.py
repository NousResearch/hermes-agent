from __future__ import annotations

import dataclasses
import copy
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from gateway import canonical_writer_schema_reconciliation as reconciliation
from gateway.canonical_writer_db import (
    CANONICAL_EVENT_LOG_COLUMNS,
    CANONICAL_EVENT_LOG_OWNER,
    CANONICAL_EVENT_LOG_TABLE,
    CANONICAL_PRIVATE_WRITER_TABLES,
    CanonicalEventLogIdentity,
    CanonicalPrivateRelationIdentity,
    CanonicalPrivateSchemaIdentity,
    PrivilegeAttestation,
    RoutineIdentity,
    WriterPrivilegePolicy,
    _expected_canonical_non_owner_acl_grants,
)
from gateway.canonical_writer_foundation import _load_source_artifacts_for_tests
from gateway.canonical_writer_postgres_backend import (
    CANONICAL_WRITER_MIGRATION_OWNER,
    CANONICAL_WRITER_ROLE,
    EXPECTED_HELPER_ROUTINE_SIGNATURES,
    EXPECTED_ROUTINE_SIGNATURES,
)


REVISION = "a" * 40


@pytest.fixture(autouse=True)
def _tmp_path_uses_process_primary_group(tmp_path: Path) -> None:
    os.chown(tmp_path, -1, os.getgid())


def _identity(signature: str, *, public: bool) -> RoutineIdentity:
    marker = "a" if public else "b"
    definition_sha256 = marker * 64
    if signature == reconciliation.MISSING_HELPER_SIGNATURE:
        definition_sha256 = (
            reconciliation.EXPECTED_MISSING_HELPER_CATALOG_IDENTITY
            .definition_sha256
        )
    return RoutineIdentity(
        signature=signature,
        owner=CANONICAL_WRITER_MIGRATION_OWNER,
        security_definer=public,
        language="plpgsql" if public else "sql",
        configuration=("search_path=pg_catalog, canonical_brain",),
        definition_sha256=definition_sha256,
    )


def _event_identity() -> CanonicalEventLogIdentity:
    return CanonicalEventLogIdentity(
        table=CANONICAL_EVENT_LOG_TABLE,
        owner=CANONICAL_EVENT_LOG_OWNER,
        owner_dangerous=False,
        relation_kind="r",
        persistence="p",
        is_partition=False,
        access_method="heap",
        tablespace_oid=0,
        row_security=False,
        force_row_security=False,
        replica_identity="d",
        relation_options=(),
        columns=CANONICAL_EVENT_LOG_COLUMNS,
        constraints=("PRIMARY KEY (event_id)",),
        user_triggers=(),
        rewrite_rules=(),
        policies=(),
        inheritance=False,
        non_owner_acl_grants=(),
        index_count=1,
        primary_index_exact=True,
    )


def _private_identity() -> CanonicalPrivateSchemaIdentity:
    return CanonicalPrivateSchemaIdentity(
        schema="canonical_brain",
        owner=CANONICAL_WRITER_MIGRATION_OWNER,
        owner_dangerous=False,
        relations=tuple(
            CanonicalPrivateRelationIdentity(
                name=name,
                owner=CANONICAL_WRITER_MIGRATION_OWNER,
                owner_dangerous=False,
                relation_kind="r",
                persistence="p",
                is_partition=False,
                access_method="heap",
                tablespace_oid=0,
                row_security=False,
                force_row_security=False,
                replica_identity="d",
                relation_options=(),
                columns=(json.dumps({"name": "identity"}, separators=(",", ":")),),
                constraints=(json.dumps({"type": "p"}, separators=(",", ":")),),
                indexes=(json.dumps({"primary": True}, separators=(",", ":")),),
                index_owners=(f"{CANONICAL_WRITER_MIGRATION_OWNER}:f",),
                user_triggers=(),
                rewrite_rules=(),
                policies=(),
                inheritance=False,
            )
            for name in CANONICAL_PRIVATE_WRITER_TABLES
        ),
    )


def _attestation(*, include_missing_helper: bool = True) -> PrivilegeAttestation:
    public = tuple(
        _identity(signature, public=True) for signature in EXPECTED_ROUTINE_SIGNATURES
    )
    helpers = tuple(
        _identity(signature, public=False)
        for signature in EXPECTED_HELPER_ROUTINE_SIGNATURES
        if include_missing_helper
        or signature != reconciliation.MISSING_HELPER_SIGNATURE
    )
    private = _private_identity()
    policy = WriterPrivilegePolicy(
        schema="canonical_brain",
        executable_routines=EXPECTED_ROUTINE_SIGNATURES,
        routine_identities=public,
        dependency_routine_identities=helpers,
        schema_privileges=("USAGE",),
        database_privileges=("CONNECT",),
        role_memberships=(CANONICAL_WRITER_ROLE,),
        private_schema_identity_sha256=private.sha256,
    )
    return PrivilegeAttestation(
        role="muncho_canary_writer_login",
        executable_routines=EXPECTED_ROUTINE_SIGNATURES,
        routine_identities=public,
        dependency_routine_identities=helpers,
        schema_privileges=("USAGE",),
        database_privileges=("CONNECT",),
        role_memberships=(CANONICAL_WRITER_ROLE,),
        canonical_non_owner_acl_grants=_expected_canonical_non_owner_acl_grants(
            policy
        ),
        canonical_writer_role_inheritors=("muncho_canary_writer_login:1:t:f",),
        canonical_event_log_identity=_event_identity(),
        canonical_private_schema_identity=private,
    )


def _target() -> reconciliation.SchemaContract:
    return reconciliation.SchemaContract(
        _attestation(),
        helper_catalog_identity=(
            reconciliation.EXPECTED_MISSING_HELPER_CATALOG_IDENTITY
        ),
    )


def _old() -> reconciliation.SchemaContract:
    return reconciliation.SchemaContract(_attestation(include_missing_helper=False))


def _plan() -> reconciliation.SchemaReconciliationPlan:
    artifact = _load_source_artifacts_for_tests()["base_migration"]
    return reconciliation._build_plan_from_artifact(REVISION, _target(), artifact)


def _truth_receipt(
    row_count: int = 3,
    canonical14_sha256: str = "d" * 64,
) -> reconciliation.CanonicalTruthReceipt:
    relations = tuple(
        reconciliation.CanonicalRelationTruthReceipt(
            relation=relation,
            row_count=row_count if index == 0 else 0,
            chunk_count=1 if index == 0 and row_count else 0,
            chunk_manifest_sha256=hashlib.sha256(
                f"{relation}:{row_count}:{canonical14_sha256}".encode()
            ).hexdigest(),
        )
        for index, relation in enumerate(reconciliation.CANONICAL_TRUTH_RELATIONS)
    )
    return reconciliation.CanonicalTruthReceipt(
        row_count=row_count,
        canonical14_sha256=canonical14_sha256,
        relation_receipts=relations,
        quarantine_anchor_receipts=tuple(
            reconciliation.CanonicalQuarantineAnchorReceipt(
                anchor=anchor,
                object_oid=8000 + index,
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


@pytest.mark.parametrize(
    ("field", "substitution"),
    (
        ("relation_count", 10.0),
        ("canonical_data_row_count", 3.0),
        ("canonical_data_row_count", False),
    ),
)
@pytest.mark.parametrize("rehash", (False, True))
def test_canonical_truth_parser_rejects_equal_comparing_type_substitution(
    field: str,
    substitution: object,
    rehash: bool,
) -> None:
    tampered = copy.deepcopy(dict(_truth_receipt().value))
    tampered[field] = substitution
    if rehash:
        unsigned = {
            key: value
            for key, value in tampered.items()
            if key != "receipt_sha256"
        }
        tampered["receipt_sha256"] = reconciliation._sha256_json(unsigned)

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_canonical_truth_invalid",
    ):
        reconciliation.CanonicalTruthReceipt.from_mapping(tampered)


def test_api_projection_rejects_duplicate_writer_inheritor_path() -> None:
    base = _attestation()
    owner = CANONICAL_EVENT_LOG_OWNER
    private = base.canonical_private_schema_identity
    assert private is not None
    dangerous = dataclasses.replace(
        base,
        routine_identities=tuple(
            dataclasses.replace(identity, owner_dangerous=True)
            for identity in base.routine_identities
        ),
        dependency_routine_identities=tuple(
            dataclasses.replace(identity, owner_dangerous=True)
            for identity in base.dependency_routine_identities
        ),
        canonical_writer_role_inheritors=(
            "muncho_canary_writer_login:1:t:f",
            "muncho_canary_writer_login:1:t:f",
            "muncho_canary_admin_aaaaaaaaaaaaaaaa:1:t:f",
        ),
        canonical_event_log_identity=dataclasses.replace(
            base.canonical_event_log_identity,
            owner_dangerous=True,
        ),
        canonical_private_schema_identity=dataclasses.replace(
            private,
            owner_dangerous=True,
            relations=tuple(
                dataclasses.replace(
                    relation,
                    owner_dangerous=True,
                    index_owners=tuple(
                        f"{owner}:t" for _ in relation.index_owners
                    ),
                )
                for relation in private.relations
            ),
        ),
    )
    projection = (
        reconciliation._ExactApiAssignedReconciliationMembershipProjection(
            owner_role=owner,
            writer_role="canonical_brain_writer",
            session_user="muncho_canary_admin_aaaaaaaaaaaaaaaa",
        )
    )

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_membership_projection_invalid",
    ):
        reconciliation._project_exact_api_assigned_owner_membership(
            dangerous,
            projection,
        )


def test_plan_binds_the_exact_ten_relation_lock_manifest() -> None:
    plan = _plan()

    assert plan.value["canonical_truth_lock"] == (
        reconciliation.CANONICAL_TRUTH_LOCK_SQL
    )
    assert all(
        relation in plan.value["canonical_truth_lock"]
        for relation in reconciliation.CANONICAL_TRUTH_RELATIONS
    )
    tampered = copy.deepcopy(dict(plan.value))
    tampered["canonical_truth_lock"] = (
        "LOCK TABLE public.canonical_event_log IN SHARE MODE"
    )
    unsigned = {
        key: value for key, value in tampered.items() if key != "plan_sha256"
    }
    tampered["plan_sha256"] = reconciliation._sha256_json(unsigned)

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_plan_invalid",
    ):
        reconciliation.SchemaReconciliationPlan.from_mapping(
            tampered,
            mutation_sql=plan.mutation_sql,
        )


class _Transaction:
    def __init__(
        self,
        database: "_Database",
        current: reconciliation.SchemaContract,
    ) -> None:
        self.database = database
        self.current = current
        self.truth_locked = False

    def lock_canonical_truth(self) -> None:
        self.truth_locked = True
        self.database.truth_lock_calls += 1

    def observe_contract(self) -> reconciliation.SchemaContract:
        assert self.truth_locked
        if self.database.observation_override is not None:
            return self.database.observation_override
        return self.current

    def observe_canonical_truth(self) -> reconciliation.CanonicalTruthReceipt:
        assert self.truth_locked
        if self.database.execute_calls:
            return self.database.after_execute_truth
        return self.database.truth

    def execute_sql(self, sql: str) -> None:
        assert self.truth_locked
        assert sql == self.database.plan.mutation_sql
        self.database.execute_calls += 1
        self.current = self.database.after_execute


class _Database:
    def __init__(
        self,
        plan: reconciliation.SchemaReconciliationPlan,
        current: reconciliation.SchemaContract,
        target: reconciliation.SchemaContract,
    ) -> None:
        self.plan = plan
        self.current = current
        self.after_execute = target
        self.observation_override: reconciliation.SchemaContract | None = None
        self.execute_calls = 0
        self.lock_keys: list[int] = []
        self.truth_lock_calls = 0
        self.truth = _truth_receipt()
        self.after_execute_truth = self.truth

    @contextmanager
    def transaction(self, *, advisory_lock_key: int) -> Iterator[_Transaction]:
        self.lock_keys.append(advisory_lock_key)
        transaction = _Transaction(self, self.current)
        try:
            yield transaction
        except BaseException:
            raise
        else:
            self.current = transaction.current


def _journal(
    tmp_path: Path,
    fault=None,
) -> reconciliation.AppendOnlySchemaReconciliationJournal:
    return reconciliation.AppendOnlySchemaReconciliationJournal(
        tmp_path / "evidence",
        strict_root=False,
        publication_fault_injector=fault,
    )


def _write_unpublished_stage(
    journal: reconciliation.AppendOnlySchemaReconciliationJournal,
    plan: reconciliation.SchemaReconciliationPlan,
    name: str,
    payload: bytes,
) -> tuple[Path, Path]:
    stage, final = journal._paths(plan, name)
    journal._ensure_directory(stage.parent, kind=name)
    journal._ensure_directory(final.parent, kind=name)
    stage.write_bytes(payload)
    os.chown(stage, -1, os.getgid())
    os.chmod(stage, 0o400)
    return stage, final


def _approval(
    plan: reconciliation.SchemaReconciliationPlan,
    target: reconciliation.SchemaContract,
    observed: reconciliation.SchemaContract,
    truth: reconciliation.CanonicalTruthReceipt,
    *,
    observed_at_unix: int = 90,
    issued_at_unix: int = 100,
    expires_at_unix: int = 1_000,
    nonce: str = "4" * 64,
):
    preflight = reconciliation.preflight_schema_reconciliation(
        plan,
        target=target,
        observed=observed,
        truth=truth,
        observed_at_unix=observed_at_unix,
    )
    authorization = reconciliation.SchemaReconciliationAuthorization.build(
        plan=plan,
        preflight=preflight,
        truth=truth,
        owner_frame_sha256="1" * 64,
        owner_subject_sha256="2" * 64,
        owner_key_id="3" * 64,
        issued_at_unix=issued_at_unix,
        expires_at_unix=expires_at_unix,
        nonce=nonce,
    )
    owner_frame = reconciliation.build_schema_reconciliation_owner_frame_receipt(
        plan=plan,
        preflight=preflight,
        truth=truth,
        authorization=authorization,
        signed_frame_sha256=authorization.value["owner_frame_sha256"],
        signature_sshsig_sha256="6" * 64,
    )
    return preflight, authorization, owner_frame


def test_plan_seals_exact_api_authority_trampoline_helper_and_restore() -> None:
    plan = _plan()
    sql = plan.mutation_sql

    assert plan.value["base_artifact_sha256"] == _load_source_artifacts_for_tests()[
        "base_migration"
    ].sha256
    assert sql.count("CREATE FUNCTION") == 1
    assert sql.count(
        "CREATE OR REPLACE FUNCTION "
        "canonical_brain._deterministic_uuid(value text)"
    ) == 2
    assert "ALTER FUNCTION" not in sql
    assert "_discord_guild_routeback_target_valid(jsonb)" in sql
    assert "SET LOCAL search_path = pg_catalog" in sql
    assert "current_database() <> 'muncho_canary_brain'" in sql
    assert "current_setting('server_version_num')::integer / 10000 <> 18" in sql
    assert ") <> 'cloudsqlsuperuser'" in sql
    assert "grantor.rolname = 'cloudsqladmin'" in sql
    assert "admin_option IS FALSE" in sql
    assert "inherit_option IS TRUE" in sql
    assert "set_option IS FALSE" in sql
    assert "'canonical_brain_migration_owner', 'USAGE'" in sql
    assert "'canonical_brain_migration_owner', 'SET'" in sql
    assert "pg_catalog.pg_event_trigger" in sql
    assert "max_prepared_transactions" in sql
    assert "pg_catalog.pg_prepared_xacts" in sql
    assert "pg_catalog.pg_shdepend" in sql
    assert "pg_catalog.pg_seclabel" in sql
    assert "pg_catalog.obj_description" in sql
    assert "muncho.schema_reconciliation_trampoline_oid" in sql
    assert "muncho.schema_reconciliation_trampoline_semantic_sha256" in sql
    assert "REVOKE ALL PRIVILEGES ON FUNCTION" in sql
    open_index = sql.index(
        "DO $reconcile_discord_routeback_helper_authority_open$"
    )
    trampoline_index = sql.index(
        "CREATE OR REPLACE FUNCTION "
        "canonical_brain._deterministic_uuid(value text)"
    )
    helper_index = sql.index(
        "CREATE FUNCTION canonical_brain."
        "_discord_guild_routeback_target_valid("
    )
    restore_index = sql.index(
        "CREATE OR REPLACE FUNCTION "
        "canonical_brain._deterministic_uuid(value text)",
        trampoline_index + 1,
    )
    terminal_index = sql.index(
        "DO $reconcile_discord_routeback_helper_terminal_validation$"
    )
    close_index = sql.index(
        "DO $reconcile_discord_routeback_helper_authority_close$"
    )
    assert (
        open_index
        < trampoline_index
        < helper_index
        < restore_index
        < terminal_index
        < close_index
    )
    for forbidden in (
        "SET ROLE",
        "RESET ROLE",
        "GRANT canonical_brain_migration_owner",
        "REVOKE canonical_brain_migration_owner",
        "GRANT EXECUTE",
        "GRANT ALL",
        "CREATE TABLE",
        "ALTER ROLE",
        "DROP ",
        "CASCADE",
        "writer_routeback_claim(request",
    ):
        assert forbidden not in sql


def test_public_plan_builder_uses_only_revision_bound_sealed_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _load_source_artifacts_for_tests()["base_migration"]
    target_asset = reconciliation.SchemaContractAsset(
        base_artifact_sha256=artifact.sha256,
        contract=_target(),
    )
    seen: list[tuple[str, str]] = []

    def load(revision: str):
        seen.append(("sql", revision))
        return {"base_migration": artifact}

    def load_target(revision: str):
        seen.append(("target", revision))
        return target_asset

    monkeypatch.setattr(reconciliation, "_load_sealed_artifacts", load)
    monkeypatch.setattr(
        reconciliation,
        "load_release_schema_contract_asset",
        load_target,
    )
    plan = reconciliation.build_schema_reconciliation_plan(REVISION)

    assert seen == [("target", REVISION), ("sql", REVISION)]
    assert plan.value["base_artifact_sha256"] == artifact.sha256
    assert plan.value["target_asset_sha256"] == target_asset.sha256
    assert plan.value["postgresql_major"] == 18


def test_target_contract_and_asset_require_exact_canonical_types() -> None:
    artifact = _load_source_artifacts_for_tests()["base_migration"]
    target = _target()
    target_value = json.loads(json.dumps(target.value))

    parsed = reconciliation.SchemaContract.from_mapping(target_value)
    assert parsed.sha256 == target.sha256

    asset = reconciliation.SchemaContractAsset(
        base_artifact_sha256=artifact.sha256,
        contract=target,
    )
    parsed_asset = reconciliation.SchemaContractAsset.from_bytes(
        asset.canonical_bytes
    )
    assert parsed_asset.value == asset.value

    bad_boolean = json.loads(json.dumps(target_value))
    bad_boolean["attestation"]["dangerous_attributes"]["superuser"] = 0
    with pytest.raises(reconciliation.SchemaReconciliationError):
        reconciliation.SchemaContract.from_mapping(bad_boolean)

    bad_integer = json.loads(json.dumps(target_value))
    bad_integer["attestation"]["canonical_event_log_identity"][
        "tablespace_oid"
    ] = False
    with pytest.raises(reconciliation.SchemaReconciliationError):
        reconciliation.SchemaContract.from_mapping(bad_integer)

    extra = json.loads(json.dumps(target_value))
    extra["attestation"]["unexpected_field"] = None
    with pytest.raises(reconciliation.SchemaReconciliationError):
        reconciliation.SchemaContract.from_mapping(extra)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.update({"postgresql_major": 17}),
        lambda value: value.update({"base_artifact_sha256": "0" * 64}),
        lambda value: value.update({"contract_sha256": "0" * 64}),
        lambda value: value.update({"asset_sha256": "0" * 64}),
        lambda value: value.update({"unexpected": True}),
    ),
)
def test_target_asset_rejects_every_unbound_or_extra_field(mutate) -> None:
    artifact = _load_source_artifacts_for_tests()["base_migration"]
    asset = reconciliation.SchemaContractAsset(
        base_artifact_sha256=artifact.sha256,
        contract=_target(),
    )
    value = json.loads(json.dumps(asset.value))
    mutate(value)

    with pytest.raises(reconciliation.SchemaReconciliationError):
        reconciliation.SchemaContractAsset.from_mapping(value)

    with pytest.raises(reconciliation.SchemaReconciliationError):
        reconciliation.SchemaContractAsset.from_bytes(asset.canonical_bytes[:-1])


@pytest.mark.parametrize("kind", ("second_missing", "owner", "acl", "schema"))
def test_old_contract_rejects_every_non_helper_delta(kind: str) -> None:
    attestation = _attestation(include_missing_helper=False)
    kwargs = {}
    if kind == "second_missing":
        attestation = dataclasses.replace(
            attestation,
            dependency_routine_identities=(
                attestation.dependency_routine_identities[1:]
            ),
        )
    elif kind == "owner":
        helpers = list(attestation.dependency_routine_identities)
        helpers[0] = dataclasses.replace(helpers[0], owner="unexpected_owner")
        attestation = dataclasses.replace(
            attestation,
            dependency_routine_identities=tuple(helpers),
        )
    elif kind == "acl":
        attestation = dataclasses.replace(
            attestation,
            public_acl_grants=("function:canonical_brain._unexpected:PUBLIC",),
        )
    else:
        kwargs["schema_owner"] = "unexpected_owner"

    with pytest.raises(reconciliation.SchemaReconciliationError):
        reconciliation.SchemaContract(attestation, **kwargs)


def test_read_only_preflight_proves_exact_single_helper_delta() -> None:
    plan = _plan()
    target = _target()
    truth = _truth_receipt()

    old = reconciliation.preflight_schema_reconciliation(
        plan,
        target=target,
        observed=_old(),
        truth=truth,
        observed_at_unix=80,
    )
    terminal = reconciliation.preflight_schema_reconciliation(
        plan,
        target=target,
        observed=target,
        truth=truth,
        observed_at_unix=81,
    )

    assert old["state"] == "exact_old_missing_one_helper"
    assert old["mutation_required"] is True
    assert old["observed_contract_sha256"] == plan.value[
        "expected_old_contract_sha256"
    ]
    assert terminal["state"] == "exact_target"
    assert terminal["mutation_required"] is False
    assert terminal["observed_contract_sha256"] == target.sha256
    assert old["truth_receipt_sha256"] == truth.sha256
    assert old["preflight_sha256"] != terminal["preflight_sha256"]


def test_exact_old_contract_reconciles_once_and_terminal_replay_reattests(
    tmp_path: Path,
) -> None:
    plan = _plan()
    target = _target()
    database = _Database(plan, _old(), target)
    journal = _journal(tmp_path)
    preflight, authorization, owner_frame = _approval(
        plan, target, _old(), database.truth
    )
    ticks = iter((100, 102))

    first = reconciliation.execute_schema_reconciliation(
        plan,
        target=target,
        preflight=preflight,
        authorization=authorization,
        owner_authorization_frame=owner_frame,
        database=database,
        journal=journal,
        now=lambda: next(ticks),
    )
    second = reconciliation.execute_schema_reconciliation(
        plan,
        target=target,
        preflight=preflight,
        authorization=authorization,
        owner_authorization_frame=owner_frame,
        database=database,
        journal=journal,
        now=lambda: 10_000,
    )

    assert first == second
    assert first["ok"] is True
    assert first["mode"] == "reconcile_missing_helper"
    assert first["mutation_applied"] is True
    assert first["final_contract_sha256"] == target.sha256
    assert database.execute_calls == 1
    assert len(database.lock_keys) == 2
    assert database.truth_lock_calls == 2
    assert first["initial_canonical_truth"] == first["final_canonical_truth"]
    assert first["authorization_sha256"] == authorization.sha256
    assert first["preflight_sha256"] == preflight["preflight_sha256"]
    assert first["owner_frame_receipt_sha256"] == owner_frame[
        "receipt_sha256"
    ]
    assert first["truth_receipt_sha256"] == database.truth.sha256
    with journal.lock():
        authorized_intent = journal.load_authorized_intent(plan)
    assert authorized_intent is not None
    assert set(authorized_intent) == {
        "schema",
        "release_revision",
        "plan_sha256",
        "base_artifact_sha256",
        "target_asset_sha256",
        "postgresql_major",
        "mutation_sql_sha256",
        "expected_old_contract_sha256",
        "target_contract_sha256",
        "preflight",
        "owner_authorization_frame",
        "authorization",
        "initial_contract_sha256",
        "initial_canonical_truth",
        "authorization_sha256",
        "preflight_sha256",
        "owner_frame_receipt_sha256",
        "truth_receipt_sha256",
        "mode",
        "mutation_required",
        "admitted_at_unix",
        "authorized_intent_sha256",
    }
    assert authorized_intent["preflight"] == preflight
    assert authorized_intent["authorization"] == authorization.value
    assert authorized_intent["owner_authorization_frame"] == owner_frame


def test_exact_target_is_adopted_without_mutation(tmp_path: Path) -> None:
    plan = _plan()
    target = _target()
    database = _Database(plan, target, target)
    preflight, authorization, owner_frame = _approval(
        plan, target, target, database.truth
    )

    receipt = reconciliation.execute_schema_reconciliation(
        plan,
        target=target,
        preflight=preflight,
        authorization=authorization,
        owner_authorization_frame=owner_frame,
        database=database,
        journal=_journal(tmp_path),
        now=lambda: 200,
    )

    assert receipt["mode"] == "adopt_existing_target"
    assert receipt["mutation_applied"] is False
    assert database.execute_calls == 0


def test_any_other_routine_identity_drift_fails_before_mutation(tmp_path: Path) -> None:
    plan = _plan()
    target = _target()
    old = _old()
    identities = list(old.attestation.dependency_routine_identities)
    identities[0] = dataclasses.replace(identities[0], definition_sha256="f" * 64)
    drift = reconciliation.SchemaContract(
        dataclasses.replace(
            old.attestation,
            dependency_routine_identities=tuple(identities),
        )
    )
    database = _Database(plan, drift, target)
    preflight, authorization, owner_frame = _approval(
        plan, target, old, database.truth
    )

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_unreviewed_database_drift",
    ):
        reconciliation.execute_schema_reconciliation(
            plan,
            target=target,
            preflight=preflight,
            authorization=authorization,
            owner_authorization_frame=owner_frame,
            database=database,
            journal=_journal(tmp_path),
            now=lambda: 300,
        )

    assert database.execute_calls == 0
    assert database.current == drift


def test_invalid_post_apply_contract_rolls_back_and_retry_is_safe(
    tmp_path: Path,
) -> None:
    plan = _plan()
    target = _target()
    old = _old()
    identities = list(target.attestation.dependency_routine_identities)
    changed_index = next(
        index
        for index, identity in enumerate(identities)
        if identity.signature != reconciliation.MISSING_HELPER_SIGNATURE
    )
    identities[changed_index] = dataclasses.replace(
        identities[changed_index],
        definition_sha256="e" * 64,
    )
    drift = reconciliation.SchemaContract(
        dataclasses.replace(
            target.attestation,
            dependency_routine_identities=tuple(identities),
        ),
        helper_catalog_identity=(
            reconciliation.EXPECTED_MISSING_HELPER_CATALOG_IDENTITY
        ),
    )
    database = _Database(plan, old, target)
    database.after_execute = drift
    journal = _journal(tmp_path)
    preflight, authorization, owner_frame = _approval(
        plan, target, old, database.truth
    )

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_post_apply_contract_invalid",
    ):
        reconciliation.execute_schema_reconciliation(
            plan,
            target=target,
            preflight=preflight,
            authorization=authorization,
            owner_authorization_frame=owner_frame,
            database=database,
            journal=journal,
            now=lambda: 400,
        )
    assert database.current == old
    assert database.execute_calls == 1

    database.after_execute = target
    receipt = reconciliation.execute_schema_reconciliation(
        plan,
        target=target,
        preflight=preflight,
        authorization=authorization,
        owner_authorization_frame=owner_frame,
        database=database,
        journal=journal,
        now=lambda: 401,
    )
    assert receipt["ok"] is True
    assert database.execute_calls == 2


def test_canonical_truth_change_rolls_back_before_terminal(tmp_path: Path) -> None:
    plan = _plan()
    target = _target()
    old = _old()
    database = _Database(plan, old, target)
    database.after_execute_truth = _truth_receipt(4, "e" * 64)
    preflight, authorization, owner_frame = _approval(
        plan, target, old, database.truth
    )

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_canonical_truth_changed",
    ):
        reconciliation.execute_schema_reconciliation(
            plan,
            target=target,
            preflight=preflight,
            authorization=authorization,
            owner_authorization_frame=owner_frame,
            database=database,
            journal=_journal(tmp_path),
            now=lambda: 450,
        )

    assert database.current == old


def test_crash_after_terminal_stage_recovers_without_reapplying(
    tmp_path: Path,
) -> None:
    plan = _plan()
    target = _target()
    database = _Database(plan, _old(), target)
    preflight, authorization, owner_frame = _approval(
        plan, target, _old(), database.truth
    )
    fired = False

    def fault(kind: str, point: str) -> None:
        nonlocal fired
        if kind == "terminal" and point == "after_write" and not fired:
            fired = True
            raise RuntimeError("simulated process loss")

    with pytest.raises(RuntimeError, match="simulated process loss"):
        reconciliation.execute_schema_reconciliation(
            plan,
            target=target,
            preflight=preflight,
            authorization=authorization,
            owner_authorization_frame=owner_frame,
            database=database,
            journal=_journal(tmp_path, fault),
            now=lambda: 500,
        )
    assert database.current == target
    assert database.execute_calls == 1

    receipt = reconciliation.execute_schema_reconciliation(
        plan,
        target=target,
        preflight=preflight,
        authorization=authorization,
        owner_authorization_frame=owner_frame,
        database=database,
        journal=_journal(tmp_path),
        now=lambda: 501,
    )
    assert receipt["ok"] is True
    assert receipt["mutation_applied"] is True
    assert database.execute_calls == 1


def test_crash_after_authorized_intent_stage_recovers_then_applies_once(
    tmp_path: Path,
) -> None:
    plan = _plan()
    target = _target()
    database = _Database(plan, _old(), target)
    preflight, authorization, owner_frame = _approval(
        plan, target, _old(), database.truth
    )
    fired = False

    def fault(kind: str, point: str) -> None:
        nonlocal fired
        if kind == "authorized_intent" and point == "after_write" and not fired:
            fired = True
            raise RuntimeError("simulated process loss")

    with pytest.raises(RuntimeError, match="simulated process loss"):
        reconciliation.execute_schema_reconciliation(
            plan,
            target=target,
            preflight=preflight,
            authorization=authorization,
            owner_authorization_frame=owner_frame,
            database=database,
            journal=_journal(tmp_path, fault),
            now=lambda: 600,
        )
    assert database.current == _old()
    assert database.execute_calls == 0

    receipt = reconciliation.execute_schema_reconciliation(
        plan,
        target=target,
        preflight=preflight,
        authorization=authorization,
        owner_authorization_frame=owner_frame,
        database=database,
        journal=_journal(tmp_path),
        now=lambda: 601,
    )
    assert receipt["ok"] is True
    assert database.execute_calls == 1


def test_terminal_receipt_never_masks_later_contract_drift(tmp_path: Path) -> None:
    plan = _plan()
    target = _target()
    database = _Database(plan, _old(), target)
    journal = _journal(tmp_path)
    preflight, authorization, owner_frame = _approval(
        plan, target, _old(), database.truth
    )
    reconciliation.execute_schema_reconciliation(
        plan,
        target=target,
        preflight=preflight,
        authorization=authorization,
        owner_authorization_frame=owner_frame,
        database=database,
        journal=journal,
        now=lambda: 700,
    )
    database.current = _old()

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_terminal_contract_drifted",
    ):
        reconciliation.execute_schema_reconciliation(
            plan,
            target=target,
            preflight=preflight,
            authorization=authorization,
            owner_authorization_frame=owner_frame,
            database=database,
            journal=journal,
            now=lambda: 701,
        )


def test_authorization_is_exact_canonical_and_bound_to_preflight_truth() -> None:
    plan = _plan()
    target = _target()
    truth = _truth_receipt()
    preflight, authorization, owner_frame = _approval(
        plan, target, _old(), truth
    )

    assert set(authorization.value) == {
        "schema",
        "release_revision",
        "plan_sha256",
        "preflight_sha256",
        "preflight_state",
        "observed_contract_sha256",
        "truth_receipt_sha256",
        "owner_frame_sha256",
        "owner_subject_sha256",
        "owner_key_id",
        "issued_at_unix",
        "expires_at_unix",
        "nonce",
        "authorization_sha256",
    }
    assert authorization.value["plan_sha256"] == plan.sha256
    assert authorization.value["preflight_sha256"] == preflight[
        "preflight_sha256"
    ]
    assert authorization.value["preflight_state"] == preflight["state"]
    assert authorization.value["observed_contract_sha256"] == preflight[
        "observed_contract_sha256"
    ]
    assert authorization.value["truth_receipt_sha256"] == truth.sha256
    assert (
        reconciliation.SchemaReconciliationAuthorization.from_mapping(
            authorization.value
        )
        == authorization
    )
    assert set(owner_frame) == {
        "schema",
        "frame_schema",
        "action",
        "approved",
        "signed_frame_sha256",
        "signature_sshsig_sha256",
        "signature_namespace",
        "signature_verified",
        "release_revision",
        "plan_sha256",
        "preflight_sha256",
        "observed_contract_sha256",
        "truth_receipt_sha256",
        "authorization_sha256",
        "owner_subject_sha256",
        "owner_key_id",
        "issued_at_unix",
        "expires_at_unix",
        "nonce",
        "receipt_sha256",
    }
    assert owner_frame["signed_frame_sha256"] == authorization.value[
        "owner_frame_sha256"
    ]
    assert owner_frame["authorization_sha256"] == authorization.sha256
    assert owner_frame["frame_schema"] == (
        "MSP2-u32be-canonical-json-no-secret.v1"
    )
    assert owner_frame["signature_namespace"] == (
        "muncho-canonical-writer-schema-reconciliation-"
        "preflight-authorization-owner-v2"
    )


def test_expired_authorization_cannot_create_authorized_intent_or_mutate(
    tmp_path: Path,
) -> None:
    plan = _plan()
    target = _target()
    database = _Database(plan, _old(), target)
    journal = _journal(tmp_path)
    preflight, authorization, owner_frame = _approval(
        plan,
        target,
        _old(),
        database.truth,
        expires_at_unix=101,
    )

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_authorization_expired",
    ):
        reconciliation.execute_schema_reconciliation(
            plan,
            target=target,
            preflight=preflight,
            authorization=authorization,
            owner_authorization_frame=owner_frame,
            database=database,
            journal=journal,
            now=lambda: 101,
        )

    assert database.execute_calls == 0
    assert database.current == _old()
    with journal.lock():
        assert journal.load_authorized_intent(plan) is None


def test_mutated_authorization_object_is_revalidated_before_use(
    tmp_path: Path,
) -> None:
    plan = _plan()
    target = _target()
    database = _Database(plan, _old(), target)
    preflight, authorization, owner_frame = _approval(
        plan,
        target,
        _old(),
        database.truth,
    )
    authorization.value["expires_at_unix"] = 10_000

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_authorization_invalid",
    ):
        reconciliation.execute_schema_reconciliation(
            plan,
            target=target,
            preflight=preflight,
            authorization=authorization,
            owner_authorization_frame=owner_frame,
            database=database,
            journal=_journal(tmp_path),
            now=lambda: 1_001,
        )

    assert database.execute_calls == 0


def test_stale_contract_or_truth_authorization_fails_before_journal(
    tmp_path: Path,
) -> None:
    plan = _plan()
    target = _target()
    old = _old()

    stale_contract_db = _Database(plan, target, target)
    preflight, authorization, owner_frame = _approval(
        plan,
        target,
        old,
        stale_contract_db.truth,
    )
    contract_journal = _journal(tmp_path / "contract")
    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_authorization_stale",
    ):
        reconciliation.execute_schema_reconciliation(
            plan,
            target=target,
            preflight=preflight,
            authorization=authorization,
            owner_authorization_frame=owner_frame,
            database=stale_contract_db,
            journal=contract_journal,
            now=lambda: 100,
        )
    with contract_journal.lock():
        assert contract_journal.load_authorized_intent(plan) is None

    stale_truth_db = _Database(plan, old, target)
    original_truth = stale_truth_db.truth
    truth_preflight, truth_authorization, truth_owner_frame = _approval(
        plan,
        target,
        old,
        original_truth,
    )
    stale_truth_db.truth = _truth_receipt(4, "e" * 64)
    truth_journal = _journal(tmp_path / "truth")
    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_authorization_binding_invalid",
    ):
        reconciliation.execute_schema_reconciliation(
            plan,
            target=target,
            preflight=truth_preflight,
            authorization=truth_authorization,
            owner_authorization_frame=truth_owner_frame,
            database=stale_truth_db,
            journal=truth_journal,
            now=lambda: 100,
        )
    with truth_journal.lock():
        assert truth_journal.load_authorized_intent(plan) is None


def test_atomic_authorized_intent_has_no_authorization_only_crash_state(
    tmp_path: Path,
) -> None:
    plan = _plan()
    target = _target()
    database = _Database(plan, _old(), target)
    fired = False

    def fault(kind: str, point: str) -> None:
        nonlocal fired
        if kind == "authorized_intent" and point == "after_write" and not fired:
            fired = True
            raise RuntimeError("simulated process loss")

    journal = _journal(tmp_path, fault)
    preflight, authorization, owner_frame = _approval(
        plan,
        target,
        _old(),
        database.truth,
    )
    with pytest.raises(RuntimeError, match="simulated process loss"):
        reconciliation.execute_schema_reconciliation(
            plan,
            target=target,
            preflight=preflight,
            authorization=authorization,
            owner_authorization_frame=owner_frame,
            database=database,
            journal=journal,
            now=lambda: 100,
        )

    with journal.lock():
        stored = journal.load_authorized_intent(plan)
        assert stored is not None
        assert stored["authorization"] == authorization.value
        assert stored["preflight"] == preflight
        assert stored["owner_authorization_frame"] == owner_frame
        for legacy_name in ("authorization", "intent"):
            stage, final = journal._paths(plan, legacy_name)
            assert not os.path.lexists(stage)
            assert not os.path.lexists(final)

    _, replay, replay_owner_frame = _approval(
        plan,
        target,
        _old(),
        database.truth,
        nonce="5" * 64,
    )
    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_authorization_replayed",
    ):
        reconciliation.execute_schema_reconciliation(
            plan,
            target=target,
            preflight=preflight,
            authorization=replay,
            owner_authorization_frame=replay_owner_frame,
            database=database,
            journal=journal,
            now=lambda: 101,
        )

    assert database.execute_calls == 0


def test_durable_authorized_intent_retries_old_database_after_expiry(
    tmp_path: Path,
) -> None:
    plan = _plan()
    target = _target()
    database = _Database(plan, _old(), target)
    fired = False

    def fault(kind: str, point: str) -> None:
        nonlocal fired
        if kind == "authorized_intent" and point == "after_write" and not fired:
            fired = True
            raise RuntimeError("simulated process loss")

    journal = _journal(tmp_path, fault)
    preflight, authorization, owner_frame = _approval(
        plan,
        target,
        _old(),
        database.truth,
    )
    with pytest.raises(RuntimeError, match="simulated process loss"):
        reconciliation.execute_schema_reconciliation(
            plan,
            target=target,
            preflight=preflight,
            authorization=authorization,
            owner_authorization_frame=owner_frame,
            database=database,
            journal=journal,
            now=lambda: 600,
        )

    receipt = reconciliation.execute_schema_reconciliation(
        plan,
        target=target,
        preflight=preflight,
        authorization=authorization,
        owner_authorization_frame=owner_frame,
        database=database,
        journal=_journal(tmp_path),
        now=lambda: 1_000,
    )

    assert receipt["ok"] is True
    assert database.execute_calls == 1
    assert database.current == target


def test_committed_target_can_terminalize_after_authorization_expiry(
    tmp_path: Path,
) -> None:
    plan = _plan()
    target = _target()
    old = _old()
    database = _Database(plan, target, target)
    journal = _journal(tmp_path)
    preflight, authorization, owner_frame = _approval(
        plan,
        target,
        old,
        database.truth,
    )
    with journal.lock():
        journal.append_authorized_intent(
            plan,
            initial_contract_sha256=old.sha256,
            initial_canonical_truth=database.truth,
            authorization=authorization,
            preflight=preflight,
            owner_authorization_frame=owner_frame,
            admitted_at_unix=100,
        )

    receipt = reconciliation.execute_schema_reconciliation(
        plan,
        target=target,
        preflight=preflight,
        authorization=authorization,
        owner_authorization_frame=owner_frame,
        database=database,
        journal=journal,
        now=lambda: 1_000,
    )

    assert receipt["ok"] is True
    assert receipt["mode"] == "reconcile_missing_helper"
    assert receipt["mutation_applied"] is True
    assert database.execute_calls == 0


def test_owner_frame_tamper_is_rejected_before_atomic_admission(
    tmp_path: Path,
) -> None:
    plan = _plan()
    target = _target()
    database = _Database(plan, _old(), target)
    journal = _journal(tmp_path)
    preflight, authorization, owner_frame = _approval(
        plan,
        target,
        _old(),
        database.truth,
    )
    tampered = copy.deepcopy(owner_frame)
    tampered["signature_sshsig_sha256"] = "7" * 64

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_owner_frame_invalid",
    ):
        reconciliation.execute_schema_reconciliation(
            plan,
            target=target,
            preflight=preflight,
            authorization=authorization,
            owner_authorization_frame=tampered,
            database=database,
            journal=journal,
            now=lambda: 100,
        )

    assert database.execute_calls == 0
    with journal.lock():
        assert journal.load_authorized_intent(plan) is None


def test_durable_retry_requires_byte_identical_owner_frame_receipt(
    tmp_path: Path,
) -> None:
    plan = _plan()
    target = _target()
    old = _old()
    database = _Database(plan, old, target)
    journal = _journal(tmp_path)
    preflight, authorization, owner_frame = _approval(
        plan,
        target,
        old,
        database.truth,
    )
    with journal.lock():
        journal.append_authorized_intent(
            plan,
            initial_contract_sha256=old.sha256,
            initial_canonical_truth=database.truth,
            authorization=authorization,
            preflight=preflight,
            owner_authorization_frame=owner_frame,
            admitted_at_unix=100,
        )

    different_frame = copy.deepcopy(owner_frame)
    different_frame["signature_sshsig_sha256"] = "7" * 64
    unsigned = {
        key: item
        for key, item in different_frame.items()
        if key != "receipt_sha256"
    }
    different_frame["receipt_sha256"] = reconciliation._sha256_json(
        unsigned
    )
    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_authorization_replayed",
    ):
        reconciliation.execute_schema_reconciliation(
            plan,
            target=target,
            preflight=preflight,
            authorization=authorization,
            owner_authorization_frame=different_frame,
            database=database,
            journal=journal,
            now=lambda: 1_000,
        )

    assert database.execute_calls == 0
    assert database.current == old


def test_authorized_intent_nested_tamper_fails_even_with_new_outer_digest(
    tmp_path: Path,
) -> None:
    plan = _plan()
    target = _target()
    old = _old()
    database = _Database(plan, old, target)
    journal = _journal(tmp_path)
    preflight, authorization, owner_frame = _approval(
        plan,
        target,
        old,
        database.truth,
    )
    with journal.lock():
        original = journal.append_authorized_intent(
            plan,
            initial_contract_sha256=old.sha256,
            initial_canonical_truth=database.truth,
            authorization=authorization,
            preflight=preflight,
            owner_authorization_frame=owner_frame,
            admitted_at_unix=100,
        )

    tampered = copy.deepcopy(original)
    tampered["owner_authorization_frame"]["signature_verified"] = False
    outer = {
        key: item
        for key, item in tampered.items()
        if key != "authorized_intent_sha256"
    }
    tampered["authorized_intent_sha256"] = reconciliation._sha256_json(outer)

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_authorized_intent_invalid",
    ):
        reconciliation._validate_authorized_intent(plan, tampered)


def test_mid_write_authorized_intent_stage_is_discarded_and_rebuilt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    target = _target()
    database = _Database(plan, _old(), target)
    journal = _journal(tmp_path)
    preflight, authorization, owner_frame = _approval(
        plan,
        target,
        _old(),
        database.truth,
    )
    real_write = reconciliation.os.write
    writes = 0

    def crash_mid_write(descriptor: int, payload: bytes) -> int:
        nonlocal writes
        writes += 1
        if writes == 1:
            return real_write(descriptor, payload[:1])
        raise OSError("simulated mid-write process loss")

    monkeypatch.setattr(reconciliation.os, "write", crash_mid_write)
    with pytest.raises(OSError, match="simulated mid-write process loss"):
        reconciliation.execute_schema_reconciliation(
            plan,
            target=target,
            preflight=preflight,
            authorization=authorization,
            owner_authorization_frame=owner_frame,
            database=database,
            journal=journal,
            now=lambda: 100,
        )
    monkeypatch.setattr(reconciliation.os, "write", real_write)

    stage, final = journal._paths(plan, "authorized_intent")
    assert stage.read_bytes() == b"{"
    assert not os.path.lexists(final)
    assert database.current == _old()
    assert database.execute_calls == 0

    receipt = reconciliation.execute_schema_reconciliation(
        plan,
        target=target,
        preflight=preflight,
        authorization=authorization,
        owner_authorization_frame=owner_frame,
        database=database,
        journal=journal,
        now=lambda: 101,
    )

    assert receipt["ok"] is True
    assert not os.path.lexists(stage)
    assert os.path.lexists(final)
    assert database.execute_calls == 1


def test_mid_write_terminal_stage_is_discarded_and_rebuilt_after_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    target = _target()
    database = _Database(plan, _old(), target)
    journal = _journal(tmp_path)
    preflight, authorization, owner_frame = _approval(
        plan,
        target,
        _old(),
        database.truth,
    )
    real_write = reconciliation.os.write
    terminal_started = False
    terminal_marker = reconciliation.RECONCILIATION_RECEIPT_SCHEMA.encode(
        "ascii"
    )

    def crash_terminal_write(descriptor: int, payload: bytes) -> int:
        nonlocal terminal_started
        if terminal_started:
            raise OSError("simulated terminal mid-write process loss")
        if terminal_marker in payload:
            terminal_started = True
            return real_write(descriptor, payload[:1])
        return real_write(descriptor, payload)

    monkeypatch.setattr(reconciliation.os, "write", crash_terminal_write)
    with pytest.raises(
        OSError,
        match="simulated terminal mid-write process loss",
    ):
        reconciliation.execute_schema_reconciliation(
            plan,
            target=target,
            preflight=preflight,
            authorization=authorization,
            owner_authorization_frame=owner_frame,
            database=database,
            journal=journal,
            now=lambda: 100,
        )
    monkeypatch.setattr(reconciliation.os, "write", real_write)

    stage, final = journal._paths(plan, "terminal")
    assert stage.read_bytes() == b"{"
    assert not os.path.lexists(final)
    assert database.current == target
    assert database.execute_calls == 1

    receipt = reconciliation.execute_schema_reconciliation(
        plan,
        target=target,
        preflight=preflight,
        authorization=authorization,
        owner_authorization_frame=owner_frame,
        database=database,
        journal=journal,
        now=lambda: 1_000,
    )

    assert receipt["ok"] is True
    assert not os.path.lexists(stage)
    assert os.path.lexists(final)
    assert database.execute_calls == 1


def test_partial_stage_is_never_discarded_without_root_journal_lock(
    tmp_path: Path,
) -> None:
    plan = _plan()
    journal = _journal(tmp_path)
    stage, final = _write_unpublished_stage(
        journal,
        plan,
        "authorized_intent",
        b"{",
    )

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_staged_receipt_invalid",
    ):
        journal.load_authorized_intent(plan)

    assert stage.read_bytes() == b"{"
    assert not os.path.lexists(final)
    with journal.lock():
        assert journal.load_authorized_intent(plan) is None
    assert not os.path.lexists(stage)


@pytest.mark.parametrize(
    "hostile_kind",
    ("mode", "xattr", "symlink", "hardlink"),
)
def test_hostile_unpublished_stage_is_never_discarded(
    tmp_path: Path,
    hostile_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    journal = _journal(tmp_path)
    stage, final = journal._paths(plan, "authorized_intent")
    journal._ensure_directory(stage.parent, kind="authorized_intent")
    journal._ensure_directory(final.parent, kind="authorized_intent")
    if hostile_kind == "symlink":
        target = stage.parent / "attacker-target"
        target.write_bytes(b"{")
        os.chmod(target, 0o400)
        stage.symlink_to(target)
    else:
        source = (
            stage
            if hostile_kind in {"mode", "xattr"}
            else stage.parent / "source"
        )
        source.write_bytes(b"{")
        os.chown(source, -1, os.getgid())
        os.chmod(source, 0o600 if hostile_kind == "mode" else 0o400)
        if hostile_kind == "hardlink":
            os.link(source, stage)
        elif hostile_kind == "xattr":
            real_list_xattrs = reconciliation._list_xattrs
            monkeypatch.setattr(
                reconciliation,
                "_list_xattrs",
                lambda path: (
                    ("user.muncho-hostile",)
                    if path == stage
                    else real_list_xattrs(path)
                ),
            )

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_staged_receipt_invalid",
    ):
        with journal.lock():
            journal.load_authorized_intent(plan)

    assert os.path.lexists(stage)
    assert not os.path.lexists(final)


def test_complete_canonical_stage_is_published_not_discarded_as_partial(
    tmp_path: Path,
) -> None:
    plan = _plan()
    journal = _journal(tmp_path)
    stage, final = _write_unpublished_stage(
        journal,
        plan,
        "authorized_intent",
        b"{}",
    )

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_receipt_collision",
    ):
        with journal.lock():
            journal._publish(
                plan,
                "authorized_intent",
                b'{"different":true}',
            )

    assert not os.path.lexists(stage)
    assert final.read_bytes() == b"{}"


def test_partial_stage_is_never_discarded_when_any_final_exists(
    tmp_path: Path,
) -> None:
    plan = _plan()
    journal = _journal(tmp_path)
    stage, final = _write_unpublished_stage(
        journal,
        plan,
        "authorized_intent",
        b"{",
    )
    final.write_bytes(b"{}")
    os.chown(final, -1, os.getgid())
    os.chmod(final, 0o400)

    with pytest.raises(
        reconciliation.SchemaReconciliationError,
        match="schema_reconciliation_staged_receipt_invalid",
    ):
        with journal.lock():
            journal.load_authorized_intent(plan)

    assert stage.read_bytes() == b"{"
    assert final.read_bytes() == b"{}"
