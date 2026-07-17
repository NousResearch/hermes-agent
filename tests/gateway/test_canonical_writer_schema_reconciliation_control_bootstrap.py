from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import struct
import types
from typing import Any, Mapping

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway import canonical_writer_schema_reconciliation_control_bootstrap as bootstrap
from gateway.canonical_writer_schema_reconciliation import (
    _control_foundation_contract_sha256,
)
from gateway.canonical_writer_db import QueryResult


NOW = 1_000
REVISION = "a" * 40
CREDENTIAL = b"C" * bootstrap.OPAQUE_CREDENTIAL_BYTES


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _hashed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {**copy.deepcopy(dict(value)), field: hashlib.sha256(_canonical(value)).hexdigest()}


def _ssh_string(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def _public_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _fingerprint(public: bytes) -> str:
    wire = _ssh_string(b"ssh-ed25519") + _ssh_string(public)
    return "SHA256:" + base64.b64encode(hashlib.sha256(wire).digest()).decode(
        "ascii"
    ).rstrip("=")


def _sshsig(key: Ed25519PrivateKey, message: bytes, *, namespace: str) -> str:
    public = _public_bytes(key)
    namespace_bytes = namespace.encode("ascii")
    signed = (
        b"SSHSIG"
        + _ssh_string(namespace_bytes)
        + _ssh_string(b"")
        + _ssh_string(b"sha512")
        + _ssh_string(hashlib.sha512(message).digest())
    )
    public_blob = _ssh_string(b"ssh-ed25519") + _ssh_string(public)
    signature_blob = _ssh_string(b"ssh-ed25519") + _ssh_string(key.sign(signed))
    envelope = (
        b"SSHSIG"
        + struct.pack(">I", 1)
        + _ssh_string(public_blob)
        + _ssh_string(namespace_bytes)
        + _ssh_string(b"")
        + _ssh_string(b"sha512")
        + _ssh_string(signature_blob)
    )
    encoded = base64.b64encode(envelope).decode("ascii")
    lines = [encoded[index : index + 70] for index in range(0, len(encoded), 70)]
    return (
        "-----BEGIN SSH SIGNATURE-----\n"
        + "\n".join(lines)
        + "\n-----END SSH SIGNATURE-----\n"
    )


def _frame(magic: bytes, value: Mapping[str, Any]) -> bytes:
    payload = _canonical(value)
    return magic + struct.pack(">I", len(payload)) + payload


def _gate(key: Ed25519PrivateKey) -> dict[str, Any]:
    public = _public_bytes(key)
    install_sha256 = _digest("control-install")
    retire_sha256 = _digest("control-retire")
    plan_sha256 = _digest("plan")
    username = bootstrap.CONTROL_ADMIN_PREFIX + plan_sha256[:16]
    unsigned = {
        "schema": bootstrap.GATE_SCHEMA,
        "ok": True,
        "state": "stopped_release_control_bootstrap_ready",
        "release_revision": REVISION,
        "release_manifest_sha256": _digest("manifest"),
        "stopped_release_receipt_file_sha256": _digest("stopped-file"),
        "stopped_release_receipt_sha256": _digest("stopped"),
        "release_artifact_sha256": _digest("release-artifact"),
        "python_version": bootstrap.EXPECTED_PYTHON_VERSION,
        "interpreter_sha256": _digest("interpreter"),
        "activation_inventory_sha256": _digest("activation"),
        "plan_sha256": plan_sha256,
        "control_install_artifact_sha256": install_sha256,
        "control_retire_artifact_sha256": retire_sha256,
        "control_foundation_contract_sha256": _control_foundation_contract_sha256(
            install_sha256, retire_sha256
        ),
        "advisory_lock_key": bootstrap.CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY,
        "host_identity_sha256": _digest("host"),
        "services_stopped_sha256": _digest("services"),
        "project": bootstrap.foundation.PROJECT,
        "sql_instance": bootstrap.foundation.SQL_INSTANCE,
        "database": bootstrap.foundation.SQL_DATABASE,
        "postgresql_major": 18,
        "tls_server_name": bootstrap.foundation.SQL_TLS_SERVER_NAME,
        "ca_file_sha256": _digest("ca"),
        "temporary_control_admin_username": username,
        "temporary_control_admin_username_sha256": hashlib.sha256(
            username.encode("ascii")
        ).hexdigest(),
        "owner_subject_sha256": _digest("owner"),
        "owner_public_key_ed25519_hex": public.hex(),
        "owner_key_id": hashlib.sha256(public).hexdigest(),
        "owner_public_fingerprint": _fingerprint(public),
        "run_nonce_sha256": _digest("run-nonce"),
        "issued_at_unix": 900,
        "expires_at_unix": 1_100,
        "services_stopped": True,
        "secret_material_recorded": False,
    }
    return _hashed(unsigned, "gate_sha256")


def _validate_gate(gate: Mapping[str, Any]) -> Mapping[str, Any]:
    return bootstrap.validate_gate_for_owner(
        gate,
        expected_release_revision=REVISION,
        expected_owner_subject_sha256=gate["owner_subject_sha256"],
        owner_public_key_ed25519_hex=gate["owner_public_key_ed25519_hex"],
        owner_public_fingerprint=gate["owner_public_fingerprint"],
        now_unix=NOW,
    )


def _cloud_authority(gate: Mapping[str, Any]) -> dict[str, Any]:
    authority = [
        "a-authority",
        "CREATE_USER",
        "DONE",
        gate["owner_subject_sha256"],
        True,
    ]
    return _hashed(
        {
            "schema": bootstrap.CLOUD_AUTHORITY_SCHEMA,
            "project": gate["project"],
            "instance": gate["sql_instance"],
            "username_sha256": gate[
                "temporary_control_admin_username_sha256"
            ],
            "host": "",
            "type": "BUILT_IN",
            "user_present": True,
            "owner_subject_sha256": gate["owner_subject_sha256"],
            "mutation_context_sha256": gate["gate_sha256"],
            "baseline_operation_names": [],
            "baseline_user_operations": [],
            "authority_operation": authority,
            "broad_bootstrap_authority": True,
            "database_roles_requested": [],
            "normal_reconciliation_executor": False,
        },
        "receipt_sha256",
    )


def _signed_install(
    key: Ed25519PrivateKey, gate: Mapping[str, Any]
) -> dict[str, Any]:
    unsigned = bootstrap.build_owner_install_claim(
        gate=gate,
        cloud_sql_authority_receipt=_cloud_authority(gate),
        credential_length=bootstrap.OPAQUE_CREDENTIAL_BYTES,
        issued_at_unix=950,
        expires_at_unix=1_050,
        nonce_sha256=_digest("install-nonce"),
    )
    template = {
        **unsigned,
        "install_claim_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
        "signature_sshsig": "",
    }
    return {
        **template,
        "signature_sshsig": _sshsig(
            key,
            bootstrap.owner_install_signature_payload(template),
            namespace=bootstrap.CONTROL_BOOTSTRAP_INSTALL_OWNER_SSHSIG_NAMESPACE,
        ),
    }


def _observation(
    *,
    phase: str,
    state: str,
    session_user_sha256: str,
    control_admin_count: int,
    membership_count: int,
    observed_at_unix: int,
) -> dict[str, Any]:
    exact = state == "exact_installed"
    return _hashed(
        {
            "schema": bootstrap.FOUNDATION_OBSERVATION_SCHEMA,
            "phase": phase,
            "state": state,
            "database": bootstrap.foundation.SQL_DATABASE,
            "postgresql_major": 18,
            "session_user_sha256": session_user_sha256,
            "control_admin_count": control_admin_count,
            "control_admin_role_exact": control_admin_count == 1,
            "control_admin_forward_role_count": (
                1 + membership_count if control_admin_count == 1 else 0
            ),
            "control_admin_owned_object_count": 0,
            "control_admin_shared_dependency_count": 0,
            "foreign_client_session_count": 0,
            "max_prepared_transactions": 0,
            "cluster_prepared_xact_count": 0,
            "non_template_database_inventory_exact": True,
            "all_connectable_database_inventory_exact": True,
            "latent_provider_exception_exact": True,
            "executor_database_effective_privileges_exact": True,
            "migration_owner_role_exact": True,
            "current_database_owner_exact": True,
            "executor_membership_count": membership_count,
            "executor_owned_object_count": 0,
            "executor_acl_dependency_count": 4 if exact else 0,
            "observer_prosrc_sha256": (
                bootstrap.OBSERVER_PROSRC_SHA256 if exact else None
            ),
            "observer_definition_sha256": (
                bootstrap.OBSERVER_DEFINITION_SHA256 if exact else None
            ),
            "apply_prosrc_sha256": (
                bootstrap.APPLY_PROSRC_SHA256 if exact else None
            ),
            "apply_definition_sha256": (
                bootstrap.APPLY_DEFINITION_SHA256 if exact else None
            ),
            "foundation_exact": exact,
            "helper_absent": True,
            "helper_same_name_count": 0,
            "observed_at_unix": observed_at_unix,
        },
        "observation_sha256",
    )


def _intermediate(
    gate: Mapping[str, Any],
    install: Mapping[str, Any],
    *,
    initial_state: str = "absent",
) -> dict[str, Any]:
    session_hash = gate["temporary_control_admin_username_sha256"]
    before = _observation(
        phase="before_install",
        state=initial_state,
        session_user_sha256=session_hash,
        control_admin_count=1,
        membership_count=0,
        observed_at_unix=960,
    )
    after = _observation(
        phase="after_install",
        state="exact_installed",
        session_user_sha256=session_hash,
        control_admin_count=1,
        membership_count=1 if initial_state == "absent" else 0,
        observed_at_unix=970,
    )
    return _hashed(
        {
            "schema": bootstrap.INTERMEDIATE_SCHEMA,
            "ok": True,
            "state": "database_session_closed_awaiting_cloud_cleanup",
            "gate_sha256": gate["gate_sha256"],
            "release_revision": gate["release_revision"],
            "plan_sha256": gate["plan_sha256"],
            "install_claim_sha256": install["install_claim_sha256"],
            "control_install_artifact_sha256": gate[
                "control_install_artifact_sha256"
            ],
            "control_foundation_contract_sha256": gate[
                "control_foundation_contract_sha256"
            ],
            "initial_foundation_state": initial_state,
            "mutation_applied": initial_state == "absent",
            "before_observation": before,
            "before_observation_sha256": before["observation_sha256"],
            "after_observation": after,
            "after_observation_sha256": after["observation_sha256"],
            "database_capability_terminated": True,
            "database_session_closed": True,
            "services_stopped_sha256": gate["services_stopped_sha256"],
            "observed_at_unix": 980,
            "secret_material_recorded": False,
        },
        "intermediate_sha256",
    )


def _cloud_absence(
    gate: Mapping[str, Any], install: Mapping[str, Any]
) -> dict[str, Any]:
    authority = install["cloud_sql_authority_receipt"]["authority_operation"]
    delete = [
        "z-delete",
        "DELETE_USER",
        "DONE",
        gate["owner_subject_sha256"],
        True,
    ]
    return _hashed(
        {
            "schema": bootstrap.CLOUD_ABSENCE_SCHEMA,
            "temporary_control_admin_absent": True,
            "project": gate["project"],
            "instance": gate["sql_instance"],
            "username_sha256": gate[
                "temporary_control_admin_username_sha256"
            ],
            "owner_subject_sha256": gate["owner_subject_sha256"],
            "mutation_context_sha256": gate["gate_sha256"],
            "user_absent": True,
            "baseline_operation_names": [],
            "baseline_user_operations": [],
            "known_operation_names": ["a-authority", "z-delete"],
            "response_known_authority_operation_names": ["a-authority"],
            "response_known_delete_operation_names": ["z-delete"],
            "post_baseline_authority_operations": [authority],
            "response_known_candidate_observed": True,
            "post_baseline_authority_operation_count": 1,
            "terminal_user_operations": [authority, delete],
            "mutation_ambiguity_observed": False,
            "quiet_window_seconds": 180,
        },
        "evidence_sha256",
    )


def _signed_cleanup(
    key: Ed25519PrivateKey,
    gate: Mapping[str, Any],
    install: Mapping[str, Any],
    intermediate: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = bootstrap.build_owner_cleanup_claim(
        gate=gate,
        install_claim_sha256=install["install_claim_sha256"],
        intermediate=intermediate,
        cloud_sql_absence_receipt=_cloud_absence(gate, install),
        issued_at_unix=985,
        expires_at_unix=1_050,
        nonce_sha256=_digest("cleanup-nonce"),
    )
    template = {
        **unsigned,
        "cleanup_claim_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest(),
        "signature_sshsig": "",
    }
    return {
        **template,
        "signature_sshsig": _sshsig(
            key,
            bootstrap.owner_cleanup_signature_payload(template),
            namespace=bootstrap.CONTROL_BOOTSTRAP_CLEANUP_OWNER_SSHSIG_NAMESPACE,
        ),
    }


def _terminal(
    gate: Mapping[str, Any],
    install: Mapping[str, Any],
    intermediate: Mapping[str, Any],
    cleanup: Mapping[str, Any],
) -> dict[str, Any]:
    post = _observation(
        phase="post_cleanup",
        state="exact_installed",
        session_user_sha256=hashlib.sha256(
            bootstrap.foundation.SQL_USER.encode("ascii")
        ).hexdigest(),
        control_admin_count=0,
        membership_count=0,
        observed_at_unix=990,
    )
    return _hashed(
        {
            "schema": bootstrap.TERMINAL_SCHEMA,
            "ok": True,
            "state": "control_installed_admin_absent_stopped",
            "gate_sha256": gate["gate_sha256"],
            "release_revision": gate["release_revision"],
            "plan_sha256": gate["plan_sha256"],
            "install_claim_sha256": install["install_claim_sha256"],
            "intermediate_sha256": intermediate["intermediate_sha256"],
            "cleanup_claim_sha256": cleanup["cleanup_claim_sha256"],
            "control_install_artifact_sha256": gate[
                "control_install_artifact_sha256"
            ],
            "control_retire_artifact_sha256": gate[
                "control_retire_artifact_sha256"
            ],
            "control_foundation_contract_sha256": gate[
                "control_foundation_contract_sha256"
            ],
            "post_cleanup_observation": post,
            "post_cleanup_observation_sha256": post["observation_sha256"],
            "temporary_control_admin_absent": True,
            "executor_memberships_absent": True,
            "executor_owns_zero_objects": True,
            "fixed_routines_exact": True,
            "services_stopped_sha256": gate["services_stopped_sha256"],
            "completed_at_unix": 995,
            "secret_material_recorded": False,
        },
        "terminal_sha256",
    )


def _protocol_fixture() -> tuple[
    Ed25519PrivateKey,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    key = Ed25519PrivateKey.generate()
    gate = _gate(key)
    install = _signed_install(key, gate)
    intermediate = _intermediate(gate, install)
    cleanup = _signed_cleanup(key, gate, install, intermediate)
    terminal = _terminal(gate, install, intermediate, cleanup)
    return key, gate, install, intermediate, cleanup, terminal


def test_gate_binds_exact_runtime_constants_and_control_contract() -> None:
    key = Ed25519PrivateKey.generate()
    gate = _gate(key)
    assert _validate_gate(gate) == gate
    for field, replacement in (
        ("python_version", "3.11.14"),
        ("advisory_lock_key", 1),
        ("tls_server_name", "wrong.invalid"),
        ("control_foundation_contract_sha256", _digest("wrong-contract")),
    ):
        changed = {**gate, field: replacement}
        changed = _hashed(
            {name: value for name, value in changed.items() if name != "gate_sha256"},
            "gate_sha256",
        )
        with pytest.raises(bootstrap.ControlBootstrapError):
            _validate_gate(changed)


def test_fixed_observation_rejects_third_routine_and_nonroutine_objects() -> None:
    sql = bootstrap._FOUNDATION_OBSERVATION_SQL
    assert "pg_advisory_xact_lock" not in sql
    assert sql.startswith(
        "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE READ ONLY;"
    )
    routine_cte = sql.split("routine_facts AS MATERIALIZED (", 1)[1].split(
        "), facts AS MATERIALIZED (", 1
    )[0]
    assert "routine.pronamespace" in routine_cte
    assert "routine.proname =" not in routine_cte
    assert "count(*) FROM routine_facts) = 2" in sql
    for catalog in (
        "pg_catalog.pg_class",
        "pg_catalog.pg_type",
        "pg_catalog.pg_constraint",
        "pg_catalog.pg_operator",
        "pg_catalog.pg_opclass",
        "pg_catalog.pg_opfamily",
        "pg_catalog.pg_default_acl",
    ):
        assert catalog in sql


@pytest.mark.parametrize(
    ("phase", "field", "replacement"),
    (
        ("before_install", "control_admin_count", 2),
        ("before_install", "control_admin_role_exact", False),
        ("before_install", "control_admin_forward_role_count", 99),
        ("before_install", "control_admin_owned_object_count", 1),
        ("before_install", "control_admin_shared_dependency_count", 1),
        ("before_install", "foreign_client_session_count", 1),
        ("before_install", "max_prepared_transactions", 1),
        ("before_install", "cluster_prepared_xact_count", 1),
        ("before_install", "non_template_database_inventory_exact", False),
        ("before_install", "all_connectable_database_inventory_exact", False),
        (
            "before_install",
            "latent_provider_exception_exact",
            False,
        ),
        (
            "before_install",
            "executor_database_effective_privileges_exact",
            False,
        ),
        ("before_install", "migration_owner_role_exact", False),
        ("before_install", "current_database_owner_exact", False),
        ("post_cleanup", "control_admin_count", 1),
        ("post_cleanup", "control_admin_role_exact", True),
        ("post_cleanup", "control_admin_forward_role_count", 1),
        ("post_cleanup", "control_admin_owned_object_count", 1),
        ("post_cleanup", "control_admin_shared_dependency_count", 1),
        ("post_cleanup", "foreign_client_session_count", 1),
        ("post_cleanup", "max_prepared_transactions", 1),
        ("post_cleanup", "cluster_prepared_xact_count", 1),
        ("post_cleanup", "non_template_database_inventory_exact", False),
        ("post_cleanup", "all_connectable_database_inventory_exact", False),
        (
            "post_cleanup",
            "latent_provider_exception_exact",
            False,
        ),
        (
            "post_cleanup",
            "executor_database_effective_privileges_exact",
            False,
        ),
        ("post_cleanup", "migration_owner_role_exact", False),
        ("post_cleanup", "current_database_owner_exact", False),
        ("post_cleanup", "executor_membership_count", 1),
    ),
)
def test_observation_validator_rejects_rehashed_session_boundary_tamper(
    phase: str,
    field: str,
    replacement: Any,
) -> None:
    observation = _observation(
        phase=phase,
        state="exact_installed" if phase == "post_cleanup" else "absent",
        session_user_sha256=_digest("session"),
        control_admin_count=0 if phase == "post_cleanup" else 1,
        membership_count=0,
        observed_at_unix=NOW,
    )
    observation[field] = replacement
    observation = _hashed(
        {
            name: value
            for name, value in observation.items()
            if name != "observation_sha256"
        },
        "observation_sha256",
    )
    with pytest.raises(
        bootstrap.ControlBootstrapError,
        match="schema_reconciliation_control_observation_invalid",
    ):
        bootstrap._validate_observation(observation, phase=phase)


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("before_observation", "session_user_sha256"), _digest("wrong-user")),
        (("after_observation", "control_admin_count"), 0),
        (("after_observation", "executor_membership_count"), 0),
    ),
)
def test_intermediate_binds_admin_session_and_creator_edge(
    path: tuple[str, str], replacement: Any
) -> None:
    _key, gate, install, intermediate, _cleanup, _terminal_value = (
        _protocol_fixture()
    )
    changed = copy.deepcopy(intermediate)
    nested, field = path
    observation = changed[nested]
    observation[field] = replacement
    observation.update(
        _hashed(
            {
                name: value
                for name, value in observation.items()
                if name != "observation_sha256"
            },
            "observation_sha256",
        )
    )
    changed[f"{nested.removesuffix('_observation')}_observation_sha256"] = (
        observation["observation_sha256"]
    )
    changed = _hashed(
        {
            name: value
            for name, value in changed.items()
            if name != "intermediate_sha256"
        },
        "intermediate_sha256",
    )
    with pytest.raises(bootstrap.ControlBootstrapError):
        bootstrap.validate_intermediate_for_owner(
            changed,
            gate=gate,
            install_claim=install,
            now_unix=NOW,
        )


def test_terminal_binds_fresh_writer_observation_after_cleanup() -> None:
    _key, gate, install, intermediate, cleanup, terminal = _protocol_fixture()
    assert bootstrap.validate_terminal_for_owner(
        terminal,
        gate=gate,
        install_claim=install,
        intermediate=intermediate,
        cleanup_claim=cleanup,
        now_unix=NOW,
    ) == terminal
    for field, replacement in (
        ("session_user_sha256", _digest("wrong-writer")),
        ("control_admin_count", 1),
        ("executor_membership_count", 1),
        ("observed_at_unix", cleanup["issued_at_unix"] - 1),
    ):
        changed = copy.deepcopy(terminal)
        post = changed["post_cleanup_observation"]
        post[field] = replacement
        post = _hashed(
            {
                name: value
                for name, value in post.items()
                if name != "observation_sha256"
            },
            "observation_sha256",
        )
        changed["post_cleanup_observation"] = post
        changed["post_cleanup_observation_sha256"] = post["observation_sha256"]
        changed = _hashed(
            {
                name: value
                for name, value in changed.items()
                if name != "terminal_sha256"
            },
            "terminal_sha256",
        )
        with pytest.raises(bootstrap.ControlBootstrapError):
            bootstrap.validate_terminal_for_owner(
                changed,
                gate=gate,
                install_claim=install,
                intermediate=intermediate,
                cleanup_claim=cleanup,
                now_unix=NOW,
            )


def test_protocol_success_uses_two_frames_and_zeroizes_callback_credential() -> None:
    _key, gate, install, intermediate, cleanup, terminal = _protocol_fixture()
    callback_credential: bytearray | None = None

    def install_callback(*args: Any) -> Mapping[str, Any]:
        nonlocal callback_credential
        callback_credential = args[2]
        assert bytes(callback_credential) == CREDENTIAL
        return intermediate

    output = io.BytesIO()
    result = bootstrap.run_protocol(
        gate,
        install_callback=install_callback,
        post_cleanup_callback=lambda *_args: terminal,
        input_stream=io.BytesIO(
            _frame(bootstrap.INSTALL_MAGIC, install)
            + CREDENTIAL
            + _frame(bootstrap.CLEANUP_MAGIC, cleanup)
        ),
        output_stream=output,
        now=lambda: NOW,
    )
    assert result == terminal
    assert callback_credential == bytearray(len(CREDENTIAL))
    assert [json.loads(line) for line in output.getvalue().splitlines()] == [
        gate,
        intermediate,
        terminal,
    ]


def test_fixed_observation_parser_binds_authenticated_session_and_exact_shape() -> None:
    class Session:
        username = bootstrap.foundation.SQL_USER

        def __init__(self) -> None:
            self.queries: list[str] = []
            self.closed = False

        def query(self, sql: str, *, maximum_rows: int) -> QueryResult:
            self.queries.append(sql)
            if sql == bootstrap._FOUNDATION_OBSERVATION_SET_LOCK_TIMEOUT_SQL:
                assert maximum_rows == 0
                return QueryResult((), (), "SET")
            if sql == bootstrap._FOUNDATION_OBSERVATION_LOCK_SQL:
                assert maximum_rows == 1
                return QueryResult(
                    ("pg_advisory_lock_shared",), (("",),), "SELECT 1"
                )
            if sql == bootstrap._FOUNDATION_OBSERVATION_RESET_LOCK_TIMEOUT_SQL:
                assert maximum_rows == 0
                return QueryResult((), (), "RESET")
            if sql == bootstrap._FOUNDATION_OBSERVATION_UNLOCK_SQL:
                assert maximum_rows == 1
                return QueryResult(
                    ("pg_advisory_unlock_shared",), (("t",),), "SELECT 1"
                )
            assert sql is bootstrap._FOUNDATION_OBSERVATION_SQL
            assert maximum_rows == 1
            return QueryResult(
                bootstrap._FOUNDATION_OBSERVATION_COLUMNS,
                ((
                    bootstrap.foundation.SQL_DATABASE,
                    "180002",
                    self.username,
                    "0",
                    "false",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "true",
                    "true",
                    "true",
                    "true",
                    "true",
                    "true",
                    "0",
                    "0",
                    "4",
                    bootstrap.OBSERVER_PROSRC_SHA256,
                    bootstrap.OBSERVER_DEFINITION_SHA256,
                    bootstrap.APPLY_PROSRC_SHA256,
                    bootstrap.APPLY_DEFINITION_SHA256,
                    "exact_installed",
                    "true",
                    "true",
                    "0",
                ),),
                "COMMIT",
            )

        def close(self) -> None:
            self.closed = True

    session = Session()
    receipt = bootstrap._observe_foundation(
        session,
        phase="post_cleanup",
        observed_at_unix=NOW,
    )
    assert receipt["state"] == "exact_installed"
    assert receipt["control_admin_count"] == 0
    assert receipt["executor_membership_count"] == 0
    assert receipt["session_user_sha256"] == hashlib.sha256(
        bootstrap.foundation.SQL_USER.encode("ascii")
    ).hexdigest()
    assert session.queries == [
        bootstrap._FOUNDATION_OBSERVATION_SET_LOCK_TIMEOUT_SQL,
        bootstrap._FOUNDATION_OBSERVATION_LOCK_SQL,
        bootstrap._FOUNDATION_OBSERVATION_RESET_LOCK_TIMEOUT_SQL,
        bootstrap._FOUNDATION_OBSERVATION_SQL,
        bootstrap._FOUNDATION_OBSERVATION_UNLOCK_SQL,
    ]
    assert session.closed is False


def test_fixed_observation_rolls_back_and_unlocks_on_query_failure() -> None:
    class Session:
        username = bootstrap.foundation.SQL_USER

        def __init__(self) -> None:
            self.queries: list[str] = []
            self.closed = False

        def query(self, sql: str, *, maximum_rows: int) -> QueryResult:
            self.queries.append(sql)
            if sql == bootstrap._FOUNDATION_OBSERVATION_SET_LOCK_TIMEOUT_SQL:
                return QueryResult((), (), "SET")
            if sql == bootstrap._FOUNDATION_OBSERVATION_LOCK_SQL:
                return QueryResult(
                    ("pg_advisory_lock_shared",), (("",),), "SELECT 1"
                )
            if sql == bootstrap._FOUNDATION_OBSERVATION_RESET_LOCK_TIMEOUT_SQL:
                return QueryResult((), (), "RESET")
            if sql == bootstrap._FOUNDATION_OBSERVATION_SQL:
                raise RuntimeError("fixture observation failure")
            if sql == "ROLLBACK":
                assert maximum_rows == 0
                return QueryResult((), (), "ROLLBACK")
            if sql == bootstrap._FOUNDATION_OBSERVATION_UNLOCK_SQL:
                return QueryResult(
                    ("pg_advisory_unlock_shared",), (("t",),), "SELECT 1"
                )
            raise AssertionError(sql)

        def close(self) -> None:
            self.closed = True

    session = Session()
    with pytest.raises(
        bootstrap.ControlBootstrapError,
        match="schema_reconciliation_control_database_observation_invalid",
    ):
        bootstrap._observe_foundation(
            session,
            phase="post_cleanup",
            observed_at_unix=NOW,
        )
    assert session.queries == [
        bootstrap._FOUNDATION_OBSERVATION_SET_LOCK_TIMEOUT_SQL,
        bootstrap._FOUNDATION_OBSERVATION_LOCK_SQL,
        bootstrap._FOUNDATION_OBSERVATION_RESET_LOCK_TIMEOUT_SQL,
        bootstrap._FOUNDATION_OBSERVATION_SQL,
        "ROLLBACK",
        bootstrap._FOUNDATION_OBSERVATION_UNLOCK_SQL,
    ]
    assert session.closed is False


def test_fixed_observation_closes_when_lock_outcome_or_unlock_is_unknown() -> None:
    class Session:
        username = bootstrap.foundation.SQL_USER

        def __init__(self, *, fail_lock: bool) -> None:
            self.fail_lock = fail_lock
            self.closed = False

        def query(self, sql: str, *, maximum_rows: int) -> QueryResult:
            if sql == bootstrap._FOUNDATION_OBSERVATION_SET_LOCK_TIMEOUT_SQL:
                return QueryResult((), (), "SET")
            if sql == bootstrap._FOUNDATION_OBSERVATION_LOCK_SQL:
                if self.fail_lock:
                    raise RuntimeError("fixture lock transport failure")
                return QueryResult(
                    ("pg_advisory_lock_shared",), (("",),), "SELECT 1"
                )
            if sql == bootstrap._FOUNDATION_OBSERVATION_RESET_LOCK_TIMEOUT_SQL:
                return QueryResult((), (), "RESET")
            if sql == bootstrap._FOUNDATION_OBSERVATION_SQL:
                return QueryResult((), (), "COMMIT")
            if sql == "ROLLBACK":
                return QueryResult((), (), "ROLLBACK")
            if sql == bootstrap._FOUNDATION_OBSERVATION_UNLOCK_SQL:
                raise RuntimeError("fixture unlock transport failure")
            raise AssertionError(sql)

        def close(self) -> None:
            self.closed = True

    for fail_lock in (True, False):
        session = Session(fail_lock=fail_lock)
        with pytest.raises(bootstrap.ControlBootstrapError):
            bootstrap._observe_foundation(
                session,
                phase="post_cleanup",
                observed_at_unix=NOW,
            )
        assert session.closed is True


@pytest.mark.parametrize("initial_state", ("absent", "exact_installed"))
def test_runtime_install_closes_broad_session_before_intermediate(
    initial_state: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _key, gate, install, _intermediate_value, _cleanup, _terminal_value = (
        _protocol_fixture()
    )
    events: list[str] = []
    times = iter((960, 970, 980, 990, 1_000))
    base = types.SimpleNamespace(
        dependencies=types.SimpleNamespace(now=lambda: next(times))
    )
    context = bootstrap._ControlRuntimeContext(
        base=base,
        gate=gate,
        install_artifact=types.SimpleNamespace(),
    )

    class Session:
        def close(self) -> None:
            events.append("close")

    session = Session()
    credential = bytearray(CREDENTIAL)
    monkeypatch.setattr(
        bootstrap,
        "_revalidate_stopped",
        lambda *_args: events.append("stopped"),
    )
    monkeypatch.setattr(
        bootstrap,
        "_open_control_session",
        lambda *_args: session,
    )

    def observe(
        _session: Any,
        *,
        phase: str,
        observed_at_unix: Any,
    ) -> Mapping[str, Any]:
        events.append(phase)
        captured_at = (
            observed_at_unix()
            if callable(observed_at_unix)
            else observed_at_unix
        )
        return _observation(
            phase=phase,
            state=initial_state if phase == "before_install" else "exact_installed",
            session_user_sha256=gate[
                "temporary_control_admin_username_sha256"
            ],
            control_admin_count=1,
            membership_count=(
                0
                if phase == "before_install" or initial_state == "exact_installed"
                else 1
            ),
            observed_at_unix=captured_at,
        )

    monkeypatch.setattr(bootstrap, "_observe_foundation", observe)
    monkeypatch.setattr(
        bootstrap,
        "_execute_install_artifact",
        lambda *_args: events.append("execute"),
    )
    intermediate = bootstrap._runtime_install_callback(
        context,
        gate,
        install,
        credential,
    )
    assert intermediate["database_capability_terminated"] is True
    assert credential == bytearray(len(CREDENTIAL))
    assert events.index("close") < len(events) - 1
    assert events[-1] == "stopped"
    assert ("execute" in events) is (initial_state == "absent")
    assert bootstrap.validate_intermediate_for_owner(
        intermediate,
        gate=gate,
        install_claim=install,
        now_unix=NOW,
    ) == intermediate


def test_runtime_install_rejects_claim_expired_during_before_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _key, gate, install, _intermediate_value, _cleanup, _terminal_value = (
        _protocol_fixture()
    )
    events: list[str] = []
    times = iter((1_049, 1_051))
    base = types.SimpleNamespace(
        dependencies=types.SimpleNamespace(now=lambda: next(times))
    )
    context = bootstrap._ControlRuntimeContext(
        base=base,
        gate=gate,
        install_artifact=types.SimpleNamespace(),
    )

    class Session:
        def close(self) -> None:
            events.append("close")

    credential = bytearray(CREDENTIAL)
    monkeypatch.setattr(bootstrap, "_revalidate_stopped", lambda *_: None)
    monkeypatch.setattr(
        bootstrap,
        "_open_control_session",
        lambda *_args: Session(),
    )

    def delayed_observe(
        _session: Any,
        *,
        phase: str,
        observed_at_unix: Any,
    ) -> Mapping[str, Any]:
        assert phase == "before_install"
        captured_at = observed_at_unix()
        return _observation(
            phase=phase,
            state="absent",
            session_user_sha256=gate[
                "temporary_control_admin_username_sha256"
            ],
            control_admin_count=1,
            membership_count=0,
            observed_at_unix=captured_at,
        )

    monkeypatch.setattr(bootstrap, "_observe_foundation", delayed_observe)
    monkeypatch.setattr(
        bootstrap,
        "_execute_install_artifact",
        lambda *_args: events.append("execute"),
    )
    with pytest.raises(
        bootstrap.ControlBootstrapError,
        match="schema_reconciliation_control_install_authorization_expired",
    ):
        bootstrap._runtime_install_callback(
            context,
            gate,
            install,
            credential,
        )
    assert events == ["close"]
    assert credential == bytearray(len(CREDENTIAL))


def test_runtime_post_cleanup_uses_fresh_fixed_writer_then_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _key, gate, install, intermediate, cleanup, _terminal_value = _protocol_fixture()
    events: list[str] = []

    class Session:
        username = bootstrap.foundation.SQL_USER

        def close(self) -> None:
            events.append("writer-close")

    session = Session()
    config = types.SimpleNamespace(user=bootstrap.foundation.SQL_USER)
    times = iter((990, 995))
    base = types.SimpleNamespace(
        dependencies=types.SimpleNamespace(
            now=lambda: next(times),
            writer_config=lambda: config,
            open_session=lambda value: (
                events.append("writer-open") or session
                if value is config
                else (_ for _ in ()).throw(AssertionError("wrong config"))
            ),
        )
    )
    context = bootstrap._ControlRuntimeContext(
        base=base,
        gate=gate,
        install_artifact=types.SimpleNamespace(),
        temporary_database_session_closed=True,
        install_callback_used=True,
    )
    monkeypatch.setattr(
        bootstrap,
        "_revalidate_stopped",
        lambda *_args: events.append("stopped"),
    )

    def observe(
        _session: Any,
        *,
        phase: str,
        observed_at_unix: Any,
    ) -> Mapping[str, Any]:
        events.append("writer-observe")
        captured_at = (
            observed_at_unix()
            if callable(observed_at_unix)
            else observed_at_unix
        )
        return _observation(
            phase=phase,
            state="exact_installed",
            session_user_sha256=hashlib.sha256(
                bootstrap.foundation.SQL_USER.encode("ascii")
            ).hexdigest(),
            control_admin_count=0,
            membership_count=0,
            observed_at_unix=captured_at,
        )

    monkeypatch.setattr(bootstrap, "_observe_foundation", observe)
    terminal = bootstrap._runtime_post_cleanup_callback(
        context,
        gate,
        install,
        intermediate,
        cleanup,
    )
    assert events == [
        "stopped",
        "writer-open",
        "writer-observe",
        "writer-close",
        "stopped",
    ]
    assert bootstrap.validate_terminal_for_owner(
        terminal,
        gate=gate,
        install_claim=install,
        intermediate=intermediate,
        cleanup_claim=cleanup,
        now_unix=NOW,
    ) == terminal


def test_install_failure_receipt_binds_accepted_install_claim() -> None:
    _key, gate, install, _intermediate_value, _cleanup, _terminal_value = (
        _protocol_fixture()
    )
    output = io.BytesIO()
    with pytest.raises(RuntimeError, match="boom"):
        bootstrap.run_protocol(
            gate,
            install_callback=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
            post_cleanup_callback=lambda *_args: {},
            input_stream=io.BytesIO(
                _frame(bootstrap.INSTALL_MAGIC, install) + CREDENTIAL
            ),
            output_stream=output,
            now=lambda: NOW,
        )
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(records) == 2
    assert records[-1]["schema"] == bootstrap.FAILURE_SCHEMA
    assert records[-1]["transcript_head_sha256"] == install[
        "install_claim_sha256"
    ]


class _PartialSecondWrite:
    def __init__(self) -> None:
        self.calls = 0
        self.payloads: list[bytes] = []

    def write(self, payload: bytes) -> int:
        self.calls += 1
        self.payloads.append(payload)
        if self.calls == 2:
            return len(payload) - 1
        return len(payload)

    def flush(self) -> None:
        return None


class _SecondFlushFails(_PartialSecondWrite):
    def flush(self) -> None:
        if self.calls == 2:
            raise OSError("closed")


@pytest.mark.parametrize("sink", (_PartialSecondWrite(), _SecondFlushFails()))
def test_partial_intermediate_output_never_appends_failure(sink: Any) -> None:
    _key, gate, install, intermediate, cleanup, terminal = _protocol_fixture()
    with pytest.raises(
        bootstrap.ControlBootstrapError,
        match="schema_reconciliation_control_output_failed",
    ):
        bootstrap.run_protocol(
            gate,
            install_callback=lambda *_args: intermediate,
            post_cleanup_callback=lambda *_args: terminal,
            input_stream=io.BytesIO(
                _frame(bootstrap.INSTALL_MAGIC, install)
                + CREDENTIAL
                + _frame(bootstrap.CLEANUP_MAGIC, cleanup)
            ),
            output_stream=sink,
            now=lambda: NOW,
        )
    assert sink.calls == 2
    assert all(bootstrap.FAILURE_SCHEMA.encode() not in item for item in sink.payloads)


def test_main_is_root_only_and_accepts_only_install(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(bootstrap.os, "geteuid", lambda: 0)
    monkeypatch.setattr(bootstrap, "run", lambda: called.append(True))
    assert bootstrap.main(["install"]) == 0
    assert called == [True]
    assert bootstrap.main([]) == 2
    monkeypatch.setattr(bootstrap.os, "geteuid", lambda: 501)
    assert bootstrap.main(["install"]) == 2


def test_main_fails_closed_without_posix_effective_uid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[bool] = []
    monkeypatch.delattr(bootstrap.os, "geteuid", raising=False)
    monkeypatch.setattr(bootstrap, "run", lambda: called.append(True))

    assert bootstrap.main(["install"]) == 2
    assert called == []
