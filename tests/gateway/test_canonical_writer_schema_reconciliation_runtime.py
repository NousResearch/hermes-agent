from __future__ import annotations

import copy
import hashlib
import os
import stat
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

import pytest

from gateway import canonical_writer_foundation as foundation
from gateway import canonical_writer_foundation_phase_b as phase_b
from gateway import canonical_writer_schema_reconciliation as reconciliation
from gateway import canonical_writer_schema_reconciliation_bootstrap as bootstrap
from gateway import canonical_writer_schema_reconciliation_runtime as runtime
from gateway.canonical_writer_db import (
    CredentialSource,
    ManagedCloudSQLAdminHBAReceipt,
    QueryResult,
    WriterDBConfig,
)
from gateway.canonical_writer_schema_reconciliation_db import (
    PostDeleteTerminalReceipt,
    WRITER_LOGIN,
)
from gateway.canonical_writer_foundation import _load_source_artifacts_for_tests


REVISION = "a" * 40
ASSET = (
    Path(__file__).parents[2]
    / "gateway"
    / "assets"
    / "canonical_writer_schema_contract_v1.json"
)


def _sha(value: Any) -> str:
    return hashlib.sha256(runtime._canonical_bytes(value)).hexdigest()


def _truth() -> reconciliation.CanonicalTruthReceipt:
    relations = tuple(
        reconciliation.CanonicalRelationTruthReceipt(
            relation=relation,
            row_count=3 if index == 0 else 0,
            chunk_count=1 if index == 0 else 0,
            chunk_manifest_sha256=hashlib.sha256(
                f"canonical-chunks-{index}".encode("ascii")
            ).hexdigest(),
        )
        for index, relation in enumerate(
            reconciliation.CANONICAL_TRUTH_RELATIONS
        )
    )
    quarantine_anchors = tuple(
        reconciliation.CanonicalQuarantineAnchorReceipt(
            anchor=anchor,
            object_oid=10_000 + index,
            owner=owner,
            kind=kind,
            persistence=persistence,
            acl_sha256=hashlib.sha256(
                f"quarantine-acl-{index}".encode("ascii")
            ).hexdigest(),
        )
        for index, (anchor, owner, kind, persistence) in enumerate(
            reconciliation._CANONICAL_QUARANTINE_ANCHOR_EXPECTATIONS
        )
    )
    return reconciliation.CanonicalTruthReceipt(
        row_count=3,
        canonical14_sha256="d" * 64,
        relation_receipts=relations,
        quarantine_anchor_receipts=quarantine_anchors,
    )


def _target_plan_asset():
    asset = reconciliation.SchemaContractAsset.from_bytes(ASSET.read_bytes())
    artifact = _load_source_artifacts_for_tests()["base_migration"]
    plan = reconciliation._build_plan_from_artifact(
        REVISION,
        asset.contract,
        artifact,
        target_asset_sha256=asset.sha256,
    )
    return asset.contract, plan, asset


def _host(*, observed_at_unix: int = 100, boot: str = "b" * 64) -> dict[str, Any]:
    unsigned = {
        "schema": "muncho-full-canary-host-identity.v1",
        "collector_authority": "trusted_root_read_only_host_collector",
        "project_id": foundation.PROJECT,
        "project_number": "39589465056",
        "zone": "europe-west3-a",
        "instance_name": "muncho-canary-v2-01",
        "instance_id": "9153645328899914617",
        "service_account_email": (
            "muncho-canary-v2-runtime@adventico-ai-platform.iam.gserviceaccount.com"
        ),
        "gce_identity_sha256": "1" * 64,
        "machine_id_sha256": "2" * 64,
        "hostname_sha256": "3" * 64,
        "host_identity_sha256": "4" * 64,
        "boot_id_sha256": boot,
        "observed_at_unix": observed_at_unix,
    }
    return {**unsigned, "receipt_sha256": _sha(unsigned)}


def _services(*, observed_at_unix: int = 100) -> dict[str, Any]:
    rows = [
        {
            "name": name,
            "load_state": "not-found",
            "active_state": "inactive",
            "sub_state": "dead",
            "unit_file_state": "not-found",
            "main_pid": 0,
            "fragment_path": None,
            "drop_in_paths": [],
            "triggered_by": [],
            "triggers": [],
            "next_elapse_unix_usec": None,
        }
        for name in phase_b.SERVICE_UNITS
    ]
    unsigned = {
        "schema": "muncho-canonical-writer-phase-b-services-stopped.v1",
        "release_revision": REVISION,
        "services": rows,
        "services_stopped_and_disabled": True,
        "observed_at_unix": observed_at_unix,
    }
    return {**unsigned, "attestation_sha256": _sha(unsigned)}


def _manifest(artifact_sha256: str = "9" * 64):
    site_packages = "venv/lib/python3.11/site-packages"
    required = {
        **{
            f"{site_packages}/{path}": mode
            for path, mode in runtime._PACKAGED_RELEASE_FILES.items()
        },
        **runtime._ROOT_RELEASE_FILES,
    }
    entries = tuple(
        SimpleNamespace(
            path=path,
            kind="file",
            mode=mode,
            size=1,
            sha256="8" * 64,
        )
        for path, mode in required.items()
    )
    return SimpleNamespace(
        revision=REVISION,
        artifact_root=f"/opt/muncho-canary-releases/{REVISION}",
        python_version="3.11.15",
        interpreter=f"/opt/muncho-canary-releases/{REVISION}/venv/bin/python",
        artifact_sha256=artifact_sha256,
        entries=entries,
    )


def _stopped(manifest_raw: bytes, *, artifact_sha256: str = "9" * 64):
    release = f"/opt/muncho-canary-releases/{REVISION}"
    return {
        "release_revision": REVISION,
        "release_root": release,
        "release_manifest_path": f"{release}/release-manifest.json",
        "release_manifest_file_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "release_artifact_sha256": artifact_sha256,
        "python_version": "3.11.15",
        "interpreter": f"{release}/venv/bin/python",
        "interpreter_sha256": "7" * 64,
        "activation_inventory": [
            {
                "path": "/etc/muncho/writer-activation/staged/writer.json",
                "state": "absent",
            },
            {
                "path": "/etc/systemd/system/hermes-cloud-gateway.service",
                "state": "absent",
            },
        ],
        "receipt_sha256": "6" * 64,
    }


def _dependencies(tmp_path: Path, **changes):
    _target, plan, asset = _target_plan_asset()
    manifest = _manifest()
    manifest_raw = b"sealed-manifest\n"
    stopped = _stopped(manifest_raw)
    historical_host = _host(observed_at_unix=90)
    journal = reconciliation.AppendOnlySchemaReconciliationJournal(
        tmp_path / "journal",
        strict_root=False,
    )
    values = {
        "current_revision": lambda: REVISION,
        "load_manifest": lambda _revision: (manifest, manifest_raw),
        "load_stopped": lambda _revision: (copy.deepcopy(stopped), b"stopped\n"),
        "load_historical_host": lambda _stopped: (
            copy.deepcopy(historical_host),
            b"host",
        ),
        "collect_host": lambda *, observed_at_unix: _host(
            observed_at_unix=observed_at_unix
        ),
        "collect_services": lambda _revision, observed_at: _services(
            observed_at_unix=observed_at
        ),
        "build_plan": lambda _revision: plan,
        "load_target_asset": lambda _revision: asset,
        "journal_factory": lambda: journal,
        "random_bytes": lambda size: b"n" * size,
        "now": lambda: 100,
        "path_exists": lambda _path: False,
        "interpreter_sha256": lambda _path: "7" * 64,
        "read_ca": lambda: b"sealed-ca",
        "harden": lambda: None,
        "protocol_runner": lambda *_args, **_kwargs: {"ok": True},
    }
    values.update(changes)
    return runtime._RuntimeDependencies(**values)


def _drifted_services(*, observed_at_unix: int) -> dict[str, Any]:
    value = _services(observed_at_unix=observed_at_unix)
    value["services"][0] = {
        **value["services"][0],
        "load_state": "loaded",
        "unit_file_state": "disabled",
        "fragment_path": (
            "/etc/systemd/system/" + str(value["services"][0]["name"])
        ),
    }
    unsigned = {
        name: item
        for name, item in value.items()
        if name != "attestation_sha256"
    }
    return {**unsigned, "attestation_sha256": _sha(unsigned)}


def _driftable_dependencies(
    tmp_path: Path,
    state: Mapping[str, str | None],
    **changes: Any,
) -> runtime._RuntimeDependencies:
    initial_manifest_raw = b"sealed-manifest\n"
    drifted_manifest_raw = b"different-sealed-manifest\n"
    initial_manifest = _manifest()
    drifted_manifest = _manifest("a" * 64)
    initial_stopped = _stopped(initial_manifest_raw)
    drifted_stopped = _stopped(
        drifted_manifest_raw,
        artifact_sha256="a" * 64,
    )

    def load_manifest(_revision: str):
        if state["kind"] == "release":
            return drifted_manifest, drifted_manifest_raw
        return initial_manifest, initial_manifest_raw

    def load_stopped(_revision: str):
        if state["kind"] == "release":
            return copy.deepcopy(drifted_stopped), b"different-stopped\n"
        return copy.deepcopy(initial_stopped), b"stopped\n"

    def collect_host(*, observed_at_unix: int):
        return _host(
            observed_at_unix=observed_at_unix,
            boot="c" * 64 if state["kind"] == "host" else "b" * 64,
        )

    def collect_services(_revision: str, observed_at_unix: int):
        if state["kind"] == "services":
            return _drifted_services(observed_at_unix=observed_at_unix)
        return _services(observed_at_unix=observed_at_unix)

    overrides = {
        "load_manifest": load_manifest,
        "load_stopped": load_stopped,
        "collect_host": collect_host,
        "collect_services": collect_services,
    }
    overrides.update(changes)
    return _dependencies(tmp_path, **overrides)


def test_owner_pin_derives_exact_key_id_and_openssh_fingerprint():
    assert runtime._derived_owner_identity() == {
        "owner_public_key_ed25519_hex": runtime.OWNER_PUBLIC_KEY_ED25519_HEX,
        "owner_key_id": runtime.OWNER_KEY_ID,
        "owner_public_fingerprint": runtime.OWNER_PUBLIC_FINGERPRINT,
        "owner_subject_sha256": runtime.OWNER_SUBJECT_SHA256,
    }


def test_owner_pin_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(runtime, "OWNER_KEY_ID", "0" * 64)
    with pytest.raises(
        runtime.SchemaReconciliationRuntimeError,
        match="schema_reconciliation_runtime_owner_pin_invalid",
    ):
        runtime._derived_owner_identity()


def test_fixed_ca_read_allows_only_writer_group_parent_chain(monkeypatch):
    calls = []

    def read_trusted(path, **kwargs):
        calls.append((path, kwargs))
        return b"trusted-ca"

    monkeypatch.setattr(runtime, "_read_trusted_file", read_trusted)

    assert runtime._read_fixed_ca() == b"trusted-ca"
    assert calls == [
        (
            foundation.DATABASE_CA_PATH,
            {
                "expected_uid": 0,
                "expected_gid": runtime.CANARY_WRITER_GID,
                "allowed_modes": frozenset({0o440}),
                "maximum": 2 * 1024 * 1024,
                "allowed_parent_gids": frozenset(
                    {0, runtime.CANARY_WRITER_GID}
                ),
            },
        )
    ]


def test_prepare_runtime_builds_secret_free_pre_database_gate(tmp_path: Path):
    database_calls: list[str] = []
    dependencies = _dependencies(
        tmp_path,
        collect_hba=lambda *_args, **_kwargs: database_calls.append("hba"),
        open_session=lambda *_args, **_kwargs: database_calls.append("db"),
    )

    context = runtime._prepare_runtime(dependencies)

    assert database_calls == []
    assert context.gate["state"] == "stopped_release_admin_preflight_ready"
    assert context.gate["temporary_admin_username"] == (
        runtime.TEMPORARY_ADMIN_PREFIX + context.plan.sha256[:16]
    )
    assert context.gate["journal_head"]["state"] == "empty"
    assert (
        context.gate["host_identity_sha256"]
        == context.initial_host_state["state_sha256"]
    )
    assert (
        context.gate["services_stopped_sha256"]
        == (context.initial_services_state["state_sha256"])
    )
    assert "mutation_required" not in context.gate
    assert "database_identity_sha256" not in context.gate
    assert b"password" not in runtime._canonical_bytes(context.gate).lower()
    assert (
        context.gate["expires_at_unix"] - context.gate["issued_at_unix"]
        == bootstrap.MAX_GATE_TTL_SECONDS
        == 1_800
    )
    assert context.gate["release_manifest_sha256"] == hashlib.sha256(
        b"sealed-manifest\n"
    ).hexdigest()
    assert context.gate["stopped_release_receipt_file_sha256"] == hashlib.sha256(
        b"stopped\n"
    ).hexdigest()
    assert context.gate["stopped_release_receipt_sha256"] == "6" * 64
    assert context.gate["python_version"] == "3.11.15"
    assert (
        bootstrap.validate_gate(
            context.gate,
            owner_public_key_ed25519_hex=runtime.OWNER_PUBLIC_KEY_ED25519_HEX,
            owner_public_fingerprint=runtime.OWNER_PUBLIC_FINGERPRINT,
            now_unix=100,
        )
        == context.gate
    )


def test_release_file_contract_matches_real_installed_layout() -> None:
    manifest = _manifest()

    runtime._validate_release_files(manifest)

    assert {
        entry.path for entry in manifest.entries
    } == {
        "venv/lib/python3.11/site-packages/"
        "gateway/canonical_writer_schema_reconciliation.py",
        "venv/lib/python3.11/site-packages/"
        "gateway/canonical_writer_schema_reconciliation_db.py",
        "venv/lib/python3.11/site-packages/"
        "gateway/canonical_writer_schema_reconciliation_bootstrap.py",
        "venv/lib/python3.11/site-packages/"
        "gateway/canonical_writer_schema_reconciliation_runtime.py",
        "venv/lib/python3.11/site-packages/"
        "gateway/assets/canonical_writer_schema_contract_v1.json",
        "gateway/assets/canonical_writer_schema_contract_v1.json",
        "scripts/sql/canonical_writer_v1.sql",
    }


def test_release_file_contract_rejects_source_tree_module_paths() -> None:
    manifest = _manifest()
    manifest.entries = tuple(
        SimpleNamespace(
            path=entry.path.removeprefix(
                "venv/lib/python3.11/site-packages/"
            ),
            kind=entry.kind,
            mode=entry.mode,
            size=entry.size,
            sha256=entry.sha256,
        )
        for entry in manifest.entries
    )

    with pytest.raises(
        runtime.SchemaReconciliationRuntimeError,
        match="schema_reconciliation_runtime_release_invalid",
    ):
        runtime._validate_release_files(manifest)


def test_prepare_runtime_rejects_any_activation_inventory_collision(tmp_path: Path):
    dependencies = _dependencies(
        tmp_path,
        path_exists=lambda path: path.endswith("writer.json"),
    )
    with pytest.raises(
        runtime.SchemaReconciliationRuntimeError,
        match="schema_reconciliation_runtime_activation_inventory_invalid",
    ):
        runtime._prepare_runtime(dependencies)


def test_prepare_runtime_rejects_reboot_since_stopped_receipt(tmp_path: Path):
    dependencies = _dependencies(
        tmp_path,
        collect_host=lambda *, observed_at_unix: _host(
            observed_at_unix=observed_at_unix,
            boot="c" * 64,
        ),
    )
    with pytest.raises(
        runtime.SchemaReconciliationRuntimeError,
        match="schema_reconciliation_runtime_release_binding_invalid",
    ):
        runtime._prepare_runtime(dependencies)


@pytest.mark.parametrize("drift", ("release", "host", "services"))
def test_preflight_revalidates_stopped_boundary_before_admin_database_open(
    tmp_path: Path,
    drift: str,
) -> None:
    state: dict[str, str | None] = {"kind": None}
    writer_config = foundation._fixed_writer_config()
    database_opens: list[WriterDBConfig] = []
    dependencies = _driftable_dependencies(
        tmp_path,
        state,
        writer_config=lambda: writer_config,
        collect_hba=lambda *_args, **_kwargs: _managed_hba(writer_config),
        open_session=lambda config: database_opens.append(config),
    )
    context = runtime._prepare_runtime(dependencies)
    state["kind"] = drift
    credential = bytearray(b"A" * 64)

    with pytest.raises(
        runtime.SchemaReconciliationRuntimeError,
        match=(
            "schema_reconciliation_runtime_preflight_"
            "stopped_boundary_drifted"
        ),
    ):
        runtime._preflight_callback(
            context,
            context.gate,
            {},
            credential,
        )

    assert database_opens == []
    assert credential == bytearray(64)


def test_prepare_runtime_rejects_release_missing_runtime_module(tmp_path: Path):
    manifest = _manifest()
    manifest.entries = tuple(
        entry
        for entry in manifest.entries
        if entry.path
        != (
            "venv/lib/python3.11/site-packages/"
            "gateway/canonical_writer_schema_reconciliation_runtime.py"
        )
    )
    dependencies = _dependencies(
        tmp_path,
        load_manifest=lambda _revision: (manifest, b"sealed-manifest\n"),
    )
    with pytest.raises(
        runtime.SchemaReconciliationRuntimeError,
        match="schema_reconciliation_runtime_release_invalid",
    ):
        runtime._prepare_runtime(dependencies)


def test_host_and_service_state_digests_exclude_observation_time():
    assert runtime._host_state(_host(observed_at_unix=10)) == runtime._host_state(
        _host(observed_at_unix=20)
    )
    assert runtime._services_state(
        _services(observed_at_unix=10), revision=REVISION
    ) == runtime._services_state(_services(observed_at_unix=20), revision=REVISION)


def test_host_state_binds_boot_id_separately_from_machine_identity():
    initial = runtime._host_state(_host(boot="b" * 64))
    rebooted = runtime._host_state(_host(boot="c" * 64))
    assert initial["host_identity_sha256"] == rebooted["host_identity_sha256"]
    assert initial["boot_id_sha256"] != rebooted["boot_id_sha256"]
    assert initial["state_sha256"] != rebooted["state_sha256"]


class _IdentitySession:
    def __init__(self, *, username: str = "muncho_canary_admin_aaaaaaaaaaaaaaaa"):
        self.username = username
        self.tls_peer_certificate_sha256 = "e" * 64
        self.closed = False
        self.query_calls = 0

    def query(self, sql: str, *, maximum_rows: int) -> QueryResult:
        self.query_calls += 1
        assert sql == runtime._DATABASE_IDENTITY_SQL
        assert maximum_rows == 1
        return QueryResult(
            ("database_name", "version_num", "database_owner", "postmaster_started"),
            (
                (
                    foundation.SQL_DATABASE,
                    "180004",
                    foundation.DATABASE_OWNER_ROLE,
                    "2026-07-16 00:00:00+00",
                ),
            ),
            "SELECT 1",
        )

    def close(self) -> None:
        self.closed = True


def test_database_identity_is_stable_and_excludes_session_or_backend_identity():
    first = runtime._database_identity(_IdentitySession(username="admin_a"))
    second = runtime._database_identity(_IdentitySession(username="admin_b"))
    assert first == second
    assert set(first) == {
        "schema",
        "project",
        "instance",
        "host",
        "port",
        "database",
        "database_owner",
        "postgresql_major",
        "tls_server_name",
        "tls_peer_certificate_sha256",
        "postmaster_started",
        "identity_sha256",
    }
    assert "session_user" not in first
    assert "current_user" not in first
    assert "backend_pid" not in first


def test_authenticated_session_lease_allows_only_sequential_borrows():
    session = _IdentitySession()
    config = object()
    lease = runtime._AuthenticatedSessionLease(session, config)
    first = lease.borrow(config)
    with pytest.raises(runtime.SchemaReconciliationRuntimeError):
        lease.borrow(config)
    first.close()
    with pytest.raises(runtime.SchemaReconciliationRuntimeError):
        lease.borrow(object())
    second = lease.borrow(config)
    second.close()
    lease.close()
    lease.close()
    assert session.closed is True


def test_durable_replay_uses_byte_identical_stored_core_preflight(tmp_path: Path):
    dependencies = _dependencies(tmp_path)
    journal = dependencies.journal_factory()
    target, plan, _asset = _target_plan_asset()
    truth = _truth()
    durable_preflight = reconciliation.preflight_schema_reconciliation(
        plan,
        target=target,
        observed=target,
        truth=truth,
        observed_at_unix=80,
    )
    authorization = reconciliation.SchemaReconciliationAuthorization.build(
        plan=plan,
        preflight=durable_preflight,
        truth=truth,
        owner_frame_sha256="1" * 64,
        owner_subject_sha256=runtime.OWNER_SUBJECT_SHA256,
        owner_key_id=runtime.OWNER_KEY_ID,
        issued_at_unix=90,
        expires_at_unix=190,
        nonce="2" * 64,
    )
    owner_frame = reconciliation.build_schema_reconciliation_owner_frame_receipt(
        plan=plan,
        preflight=durable_preflight,
        truth=truth,
        authorization=authorization,
        signed_frame_sha256="1" * 64,
        signature_sshsig_sha256="3" * 64,
    )
    with journal.lock():
        journal.append_authorized_intent(
            plan,
            initial_contract_sha256=target.sha256,
            initial_canonical_truth=truth,
            authorization=authorization,
            preflight=durable_preflight,
            owner_authorization_frame=owner_frame,
            admitted_at_unix=90,
        )
    context = runtime._prepare_runtime(dependencies)
    fresh_preflight = reconciliation.preflight_schema_reconciliation(
        plan,
        target=target,
        observed=target,
        truth=truth,
        observed_at_unix=100,
    )
    assert fresh_preflight != durable_preflight
    context.preflight = fresh_preflight
    context.truth = truth

    stored_authorization, stored_frame, core_preflight = runtime._admission_for_apply(
        context, None, None
    )

    assert stored_authorization.value == authorization.value
    assert stored_frame == owner_frame
    assert core_preflight == durable_preflight
    assert core_preflight != fresh_preflight


def test_open_admin_config_uses_0400_memfd_and_zeroizes_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    observed: dict[str, Any] = {}

    @contextmanager
    def descriptor(secret: bytearray) -> Iterator[int]:
        path = tmp_path / "credential"
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o400)
        os.write(fd, secret)
        os.fchmod(fd, 0o400)
        try:
            yield fd
        finally:
            os.close(fd)

    def open_session(config: WriterDBConfig):
        source = config.credential
        assert isinstance(source, CredentialSource)
        assert source.fd is not None
        observed["secret"] = os.pread(source.fd, 1024, 0)
        observed["mode"] = stat.S_IMODE(os.fstat(source.fd).st_mode)
        observed["source"] = source
        return _IdentitySession(username=config.user)

    monkeypatch.setattr(runtime.phase_b_runtime, "_secret_descriptor", descriptor)
    context = SimpleNamespace(
        gate={"temporary_admin_username": "muncho_canary_admin_aaaaaaaaaaaaaaaa"},
        dependencies=SimpleNamespace(open_session=open_session),
    )
    secret = bytearray(b"A" * 64)
    config, session = runtime._open_admin_config(context, secret)

    assert config.user == "muncho_canary_admin_aaaaaaaaaaaaaaaa"
    assert session.username == config.user
    assert observed["secret"] == b"A" * 64
    assert observed["mode"] == 0o400
    assert observed["source"].path is None
    assert secret == bytearray(64)


def test_preflight_apply_and_reattest_reuse_one_authenticated_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target, plan, _asset = _target_plan_asset()
    truth = _truth()
    sessions: list[_IdentitySession] = []
    transaction_count = 0

    @contextmanager
    def descriptor(secret: bytearray) -> Iterator[int]:
        path = tmp_path / "one-credential"
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o400)
        os.write(fd, secret)
        os.fchmod(fd, 0o400)
        try:
            yield fd
        finally:
            os.close(fd)

    def open_session(config: WriterDBConfig):
        assert config.credential.fd is not None
        assert os.pread(config.credential.fd, 64, 0) == b"A" * 64
        session = _IdentitySession(username=config.user)
        sessions.append(session)
        return session

    def hba(config: WriterDBConfig, *, now_unix: int, ttl_seconds: int):
        assert config.user == "muncho_canary_writer_login"
        return ManagedCloudSQLAdminHBAReceipt(
            version="managed-cloudsqladmin-hba-rejection-v2",
            host=config.host,
            tls_server_name=config.tls_server_name,
            port=config.port,
            server_certificate_sha256="e" * 64,
            database="cloudsqladmin",
            user=config.user,
            observed_at_unix=now_unix,
            expires_at_unix=now_unix + ttl_seconds,
            sqlstate="28000",
            server_message=(
                f'no pg_hba.conf entry for host "{config.host}", user "{config.user}", '
                'database "cloudsqladmin", SSL encryption'
            ),
            result="pg_hba_rejected",
            tls_peer_verified=True,
        )

    class Transaction:
        def lock_canonical_truth(self):
            return None

        def observe_contract(self):
            return target

        def observe_canonical_truth(self):
            return truth

    class Database:
        def __init__(self, *, _session_factory, admin_config, **_kwargs):
            self.session_factory = _session_factory
            self.admin_config = admin_config

        @contextmanager
        def transaction(self, *, advisory_lock_key: int):
            nonlocal transaction_count
            assert advisory_lock_key == plan.value["advisory_lock_key"]
            transaction_count += 1
            borrowed = self.session_factory(self.admin_config)
            try:
                yield Transaction()
            finally:
                borrowed.close()

    def execute(_plan, *, database, **_kwargs):
        with database.transaction(
            advisory_lock_key=plan.value["advisory_lock_key"]
        ) as transaction:
            transaction.lock_canonical_truth()
            assert transaction.observe_contract() is target
            assert transaction.observe_canonical_truth() == truth
        return {"authorized_intent_sha256": "f" * 64}

    monkeypatch.setattr(runtime.phase_b_runtime, "_secret_descriptor", descriptor)
    monkeypatch.setattr(runtime, "PostgresSchemaReconciliationDatabase", Database)
    monkeypatch.setattr(runtime, "execute_schema_reconciliation", execute)
    dependencies = _dependencies(
        tmp_path,
        writer_config=foundation._fixed_writer_config,
        collect_hba=hba,
        open_session=open_session,
    )
    context = runtime._prepare_runtime(dependencies)
    secret = bytearray(b"A" * 64)
    collection = runtime._preflight_callback(context, context.gate, {}, secret)
    assert secret == bytearray(64)
    assert len(sessions) == 1
    assert transaction_count == 1
    authorization = reconciliation.SchemaReconciliationAuthorization.build(
        plan=context.plan,
        preflight=collection["preflight"],
        truth=truth,
        owner_frame_sha256="1" * 64,
        owner_subject_sha256=runtime.OWNER_SUBJECT_SHA256,
        owner_key_id=runtime.OWNER_KEY_ID,
        issued_at_unix=100,
        expires_at_unix=200,
        nonce="2" * 64,
    )
    owner_frame = reconciliation.build_schema_reconciliation_owner_frame_receipt(
        plan=context.plan,
        preflight=collection["preflight"],
        truth=truth,
        authorization=authorization,
        signed_frame_sha256="1" * 64,
        signature_sshsig_sha256="3" * 64,
    )
    challenge = {
        "preflight": collection["preflight"],
        "canonical_truth_receipt": truth.value,
    }

    result = runtime._apply_callback(
        context,
        context.gate,
        {},
        challenge,
        {},
        authorization,
        owner_frame,
    )

    assert result["authorized_intent_sha256"] == "f" * 64
    assert transaction_count == 3
    assert len(sessions) == 1
    assert sessions[0].closed is True
    assert context.temporary_admin_database_closed_before_cleanup is True
    assert result["database_commit_attestation"]["database_session_closed"] is True


@pytest.mark.parametrize("drift", ("release", "host", "services"))
def test_apply_revalidates_stopped_boundary_before_schema_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    target, plan, _asset = _target_plan_asset()
    truth = _truth()
    state: dict[str, str | None] = {"kind": None}
    writer_config = foundation._fixed_writer_config()
    sessions: list[_IdentitySession] = []
    execute_calls: list[str] = []

    @contextmanager
    def descriptor(secret: bytearray) -> Iterator[int]:
        path = tmp_path / "apply-boundary-credential"
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o400)
        os.write(fd, secret)
        os.fchmod(fd, 0o400)
        try:
            yield fd
        finally:
            os.close(fd)

    def open_session(config: WriterDBConfig) -> _IdentitySession:
        session = _IdentitySession(username=config.user)
        sessions.append(session)
        return session

    class Transaction:
        def lock_canonical_truth(self) -> None:
            return None

        def observe_contract(self) -> reconciliation.SchemaContract:
            return target

        def observe_canonical_truth(
            self,
        ) -> reconciliation.CanonicalTruthReceipt:
            return truth

    class Database:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        @contextmanager
        def transaction(self, *, advisory_lock_key: int):
            assert advisory_lock_key == plan.value["advisory_lock_key"]
            yield Transaction()

    monkeypatch.setattr(runtime.phase_b_runtime, "_secret_descriptor", descriptor)
    monkeypatch.setattr(runtime, "PostgresSchemaReconciliationDatabase", Database)
    monkeypatch.setattr(
        runtime,
        "execute_schema_reconciliation",
        lambda *_args, **_kwargs: execute_calls.append("execute"),
    )
    dependencies = _driftable_dependencies(
        tmp_path,
        state,
        writer_config=lambda: writer_config,
        collect_hba=lambda *_args, **_kwargs: _managed_hba(writer_config),
        open_session=open_session,
    )
    context = runtime._prepare_runtime(dependencies)
    collection = runtime._preflight_callback(
        context,
        context.gate,
        {},
        bytearray(b"A" * 64),
    )
    authorization = reconciliation.SchemaReconciliationAuthorization.build(
        plan=context.plan,
        preflight=collection["preflight"],
        truth=truth,
        owner_frame_sha256="1" * 64,
        owner_subject_sha256=runtime.OWNER_SUBJECT_SHA256,
        owner_key_id=runtime.OWNER_KEY_ID,
        issued_at_unix=100,
        expires_at_unix=200,
        nonce="2" * 64,
    )
    owner_frame = reconciliation.build_schema_reconciliation_owner_frame_receipt(
        plan=context.plan,
        preflight=collection["preflight"],
        truth=truth,
        authorization=authorization,
        signed_frame_sha256="1" * 64,
        signature_sshsig_sha256="3" * 64,
    )
    challenge = {
        "preflight": collection["preflight"],
        "canonical_truth_receipt": truth.value,
    }
    state["kind"] = drift

    with pytest.raises(
        runtime.SchemaReconciliationRuntimeError,
        match="schema_reconciliation_runtime_apply_stopped_boundary_drifted",
    ):
        runtime._apply_callback(
            context,
            context.gate,
            {},
            challenge,
            {},
            authorization,
            owner_frame,
        )

    assert execute_calls == []
    assert len(sessions) == 1
    assert sessions[0].closed is True
    assert context.temporary_admin_database_closed_before_cleanup is True


@pytest.mark.parametrize(
    "secret",
    [bytearray(b"A" * 63), bytearray(b"+" * 64), bytearray(b"A" * 65)],
)
def test_open_admin_config_rejects_non_exact_urlsafe_64_bytes(secret: bytearray):
    context = SimpleNamespace(
        gate={"temporary_admin_username": "muncho_canary_admin_aaaaaaaaaaaaaaaa"},
        dependencies=SimpleNamespace(open_session=lambda _config: None),
    )
    with pytest.raises(
        runtime.SchemaReconciliationRuntimeError,
        match="schema_reconciliation_runtime_credential_invalid",
    ):
        runtime._open_admin_config(context, secret)


def test_bridge_receipt_truthfully_describes_owner_collection_then_cleanup():
    receipt = runtime._temporary_owner_bridge_receipt(123)
    assert receipt["owner_authority_active_during_locked_collection"] is True
    assert receipt["exact_provider_memberships_during_locked_collection"] is True
    assert receipt["current_user_remained_temporary_login"] is True
    assert receipt["memberships_remain_until_cloud_user_cleanup"] is True
    assert receipt["transaction_committed"] is True
    assert receipt["secret_material_recorded"] is False
    assert receipt["receipt_sha256"] == _sha({
        name: item for name, item in receipt.items() if name != "receipt_sha256"
    })


def _managed_hba(
    config: WriterDBConfig,
    *,
    observed_at_unix: int = 100,
    server_certificate_sha256: str = "e" * 64,
) -> ManagedCloudSQLAdminHBAReceipt:
    return ManagedCloudSQLAdminHBAReceipt(
        version="managed-cloudsqladmin-hba-rejection-v2",
        host=config.host,
        tls_server_name=config.tls_server_name,
        port=config.port,
        server_certificate_sha256=server_certificate_sha256,
        database="cloudsqladmin",
        user=config.user,
        observed_at_unix=observed_at_unix,
        expires_at_unix=observed_at_unix + 300,
        sqlstate="28000",
        server_message=(
            f'no pg_hba.conf entry for host "{config.host}", user '
            f'"{config.user}", database "cloudsqladmin", SSL encryption'
        ),
        result="pg_hba_rejected",
        tls_peer_verified=True,
    )


def _post_delete_terminal(
    context: runtime._RuntimeContext,
    hba: ManagedCloudSQLAdminHBAReceipt,
    *,
    observed_at_unix: int = 100,
) -> PostDeleteTerminalReceipt:
    request_id = "schema-reconciliation-post-delete-terminal-v1"
    response = {
        "ok": True,
        "result": {
            "service": "canonical_writer",
            "protocol": "v1",
            "database_identity": "canonical_brain_migration_owner",
            "request_id": request_id,
        },
    }
    assert context.truth is not None
    return PostDeleteTerminalReceipt(
        release_revision=context.revision,
        plan_sha256=context.plan.sha256,
        database=foundation.SQL_DATABASE,
        writer_login=WRITER_LOGIN,
        temporary_login=context.gate["temporary_admin_username"],
        temporary_login_sha256=context.gate[
            "temporary_admin_username_sha256"
        ],
        target_contract_sha256=context.target.sha256,
        observed_contract_sha256=context.target.sha256,
        writer_session_identity_exact=True,
        temporary_login_absent=True,
        temporary_login_inventory_empty=True,
        migration_owner_memberships_absent=True,
        writer_authority_exact=True,
        writer_ping_verified=True,
        writer_ping_request_id=request_id,
        writer_ping_response_sha256=_sha(response),
        fresh_writer_session_closed=True,
        tls_peer_certificate_sha256="e" * 64,
        managed_hba_receipt_sha256=hba.sha256,
        pre_delete_canonical_truth_receipt_sha256=context.truth.sha256,
        canonical_truth_observed=False,
        canonical_truth_limitation=(
            "writer_principal_has_no_direct_canonical_data_read_and_no_"
            "fixed_security_definer_full_truth_export"
        ),
        observed_at_unix=observed_at_unix,
    )


def _prepare_post_cleanup_context(
    context: runtime._RuntimeContext,
    writer_config: WriterDBConfig,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], _IdentitySession]:
    context.truth = _truth()
    context.database_identity = runtime._database_identity(_IdentitySession())
    context.writer_config = writer_config
    admin_session = _IdentitySession()
    context.lease = runtime._AuthenticatedSessionLease(admin_session, object())
    context.close_temporary_admin_database()
    challenge = {
        "database_identity_sha256": context.database_identity[
            "identity_sha256"
        ],
        "tls_peer_certificate_sha256": context.database_identity[
            "tls_peer_certificate_sha256"
        ],
    }
    intermediate = {"final_canonical_truth": context.truth.value}
    cleanup = {
        "temporary_admin_username_sha256": context.gate[
            "temporary_admin_username_sha256"
        ],
        "cloud_sql_absence_receipt": {"temporary_admin_absent": True},
        "issued_at_unix": 100,
    }
    return challenge, intermediate, cleanup, admin_session


def test_post_cleanup_uses_one_distinct_fresh_writer_session_and_rechecks_stopped_state(
    tmp_path: Path,
):
    writer_config = foundation._fixed_writer_config()
    hba = _managed_hba(writer_config)
    opened: list[WriterDBConfig] = []

    def open_session(config: WriterDBConfig):
        opened.append(config)
        return _IdentitySession(username=config.user)

    collector_calls: list[str] = []

    def collect_post_delete_terminal(**kwargs):
        collector_calls.append(kwargs["temporary_login"])
        assert kwargs["writer_config"] == writer_config
        assert kwargs["managed_hba_receipt"] == hba
        assert kwargs["_session_factory"] is open_session
        session = kwargs["_session_factory"](kwargs["writer_config"])
        session.close()
        return _post_delete_terminal(context, hba)

    dependencies = _dependencies(
        tmp_path,
        writer_config=lambda: writer_config,
        collect_hba=lambda *_args, **_kwargs: hba,
        open_session=open_session,
        collect_post_delete_terminal=collect_post_delete_terminal,
    )
    context = runtime._prepare_runtime(dependencies)
    challenge, intermediate, cleanup, admin_session = (
        _prepare_post_cleanup_context(context, writer_config)
    )

    receipt = runtime._post_cleanup_callback(
        context,
        context.gate,
        challenge,
        intermediate,
        cleanup,
    )

    assert admin_session.query_calls == 0
    assert collector_calls == [context.gate["temporary_admin_username"]]
    assert opened == [writer_config]
    assert receipt["host_identity_sha256"] == context.gate["host_identity_sha256"]
    assert receipt["services_stopped_sha256"] == context.gate["services_stopped_sha256"]
    assert receipt["fresh_managed_hba_receipt"] == hba.as_dict()
    assert receipt["fresh_managed_hba_receipt_sha256"] == hba.sha256
    assert receipt["post_delete_terminal_receipt"][
        "managed_hba_receipt_sha256"
    ] == receipt["fresh_managed_hba_receipt_sha256"]
    assert receipt["post_delete_terminal_receipt"][
        "tls_peer_certificate_sha256"
    ] == receipt["fresh_managed_hba_receipt"][
        "server_certificate_sha256"
    ]
    assert receipt["post_delete_terminal_receipt"][
        "fresh_writer_session_closed"
    ] is True
    assert receipt["post_delete_terminal_receipt"][
        "canonical_truth_observed"
    ] is False
    assert receipt["secret_material_recorded"] is False
    assert receipt["release_manifest_sha256"] == context.gate[
        "release_manifest_sha256"
    ]
    assert receipt["stopped_release_receipt_file_sha256"] == context.gate[
        "stopped_release_receipt_file_sha256"
    ]


@pytest.mark.parametrize(
    "drift",
    ["writer_config", "hba_time", "hba_host", "tls_peer"],
)
def test_post_cleanup_rejects_config_or_hba_drift_before_writer_collector(
    tmp_path: Path,
    drift: str,
):
    writer_config = foundation._fixed_writer_config()
    returned_config = (
        replace(writer_config, application_name="drifted-writer")
        if drift == "writer_config"
        else writer_config
    )
    hba = _managed_hba(
        writer_config,
        observed_at_unix=99 if drift == "hba_time" else 100,
        server_certificate_sha256=(
            "f" * 64 if drift == "tls_peer" else "e" * 64
        ),
    )
    if drift == "hba_host":
        hba = replace(hba, host="10.91.0.4")
    collector_calls: list[str] = []
    dependencies = _dependencies(
        tmp_path,
        writer_config=lambda: returned_config,
        collect_hba=lambda *_args, **_kwargs: hba,
        collect_post_delete_terminal=lambda **_kwargs: collector_calls.append(
            "called"
        ),
    )
    context = runtime._prepare_runtime(dependencies)
    challenge, intermediate, cleanup, _admin_session = (
        _prepare_post_cleanup_context(context, writer_config)
    )

    with pytest.raises(
        runtime.SchemaReconciliationRuntimeError,
        match="schema_reconciliation_runtime_post_cleanup_invalid",
    ):
        runtime._post_cleanup_callback(
            context,
            context.gate,
            challenge,
            intermediate,
            cleanup,
        )
    assert collector_calls == []


def test_post_cleanup_rejects_terminal_receipt_not_bound_to_fresh_hba(
    tmp_path: Path,
):
    writer_config = foundation._fixed_writer_config()
    hba = _managed_hba(writer_config)
    collector_calls: list[str] = []

    def collect_post_delete_terminal(**_kwargs):
        collector_calls.append("called")
        return replace(
            _post_delete_terminal(context, hba),
            managed_hba_receipt_sha256="f" * 64,
        )

    dependencies = _dependencies(
        tmp_path,
        writer_config=lambda: writer_config,
        collect_hba=lambda *_args, **_kwargs: hba,
        collect_post_delete_terminal=collect_post_delete_terminal,
    )
    context = runtime._prepare_runtime(dependencies)
    challenge, intermediate, cleanup, _admin_session = (
        _prepare_post_cleanup_context(context, writer_config)
    )

    with pytest.raises(
        runtime.SchemaReconciliationRuntimeError,
        match="schema_reconciliation_runtime_post_cleanup_invalid",
    ):
        runtime._post_cleanup_callback(
            context,
            context.gate,
            challenge,
            intermediate,
            cleanup,
        )
    assert collector_calls == ["called"]


def test_post_cleanup_reloads_and_rejects_stopped_release_receipt_drift(
    tmp_path: Path,
):
    initial_loader = _dependencies(tmp_path).load_stopped
    load_count = 0

    def load_stopped(revision: str):
        nonlocal load_count
        load_count += 1
        stopped, raw = initial_loader(revision)
        if load_count == 2:
            raw = b"different-sealed-stopped-receipt\n"
        return stopped, raw

    writer_config = foundation._fixed_writer_config()
    collector_calls: list[str] = []
    dependencies = _dependencies(
        tmp_path,
        load_stopped=load_stopped,
        writer_config=lambda: writer_config,
        collect_post_delete_terminal=lambda **_kwargs: collector_calls.append(
            "called"
        ),
    )
    context = runtime._prepare_runtime(dependencies)
    challenge, intermediate, cleanup, admin_session = (
        _prepare_post_cleanup_context(context, writer_config)
    )

    with pytest.raises(
        runtime.SchemaReconciliationRuntimeError,
        match="schema_reconciliation_runtime_post_cleanup_drifted",
    ):
        runtime._post_cleanup_callback(
            context,
            context.gate,
            challenge,
            intermediate,
            cleanup,
        )

    assert load_count == 2
    assert admin_session.query_calls == 0
    assert collector_calls == []


def test_post_cleanup_fails_if_database_was_not_closed(tmp_path: Path):
    writer_config = foundation._fixed_writer_config()
    collector_calls: list[str] = []
    context = runtime._prepare_runtime(
        _dependencies(
            tmp_path,
            writer_config=lambda: writer_config,
            collect_post_delete_terminal=lambda **_kwargs: collector_calls.append(
                "called"
            ),
        )
    )
    context.truth = _truth()
    context.database_identity = runtime._database_identity(_IdentitySession())
    context.writer_config = writer_config
    context.lease = runtime._AuthenticatedSessionLease(_IdentitySession(), object())
    with pytest.raises(
        runtime.SchemaReconciliationRuntimeError,
        match="schema_reconciliation_runtime_cleanup_order_invalid",
    ):
        runtime._post_cleanup_callback(context, context.gate, {}, {}, {})
    assert collector_calls == []


def test_main_emits_only_generic_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        runtime,
        "run",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("secret detail")),
    )
    assert runtime.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "schema reconciliation runtime failed closed\n"
    assert "secret detail" not in captured.err
