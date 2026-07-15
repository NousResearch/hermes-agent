from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gateway import production_owner_runtime as runtime


REVISION = "a" * 40


def _attestation(revision: str = REVISION) -> dict:
    unsigned = {
        "schema": runtime.ATTESTATION_SCHEMA,
        "revision": revision,
        "manifest_sha256": "1" * 64,
        "tree_sha256": "2" * 64,
        "interpreter_sha256": "3" * 64,
        "pyvenv_cfg_sha256": "4" * 64,
        "sys_path_sha256": "5" * 64,
        "required_modules_sha256": "6" * 64,
        "module_origins_release_local": True,
        "ambient_python_environment_present": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {
        **unsigned,
        "attestation_sha256": runtime._sha256_bytes(
            runtime._canonical(unsigned)
        ),
    }


def test_portable_attestation_is_exact_revision_bound_and_self_digesting() -> None:
    attestation = _attestation()

    assert runtime.validate_owner_runtime_attestation(
        attestation,
        revision=REVISION,
    ) == attestation

    for mutate in (
        lambda value: value.update(extra=True),
        lambda value: value.update(revision="b" * 40),
        lambda value: value.update(module_origins_release_local=False),
        lambda value: value.update(tree_sha256="f" * 64),
    ):
        drifted = copy.deepcopy(attestation)
        mutate(drifted)
        with pytest.raises(
            runtime.ProductionOwnerRuntimeError,
            match="attestation_invalid",
        ):
            runtime.validate_owner_runtime_attestation(
                drifted,
                revision=REVISION,
            )


def test_active_runtime_gate_returns_a_copy_and_rejects_ambient_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attestation = _attestation()
    monkeypatch.setattr(runtime, "_ACTIVE_ATTESTATION", None)
    runtime._set_active_attestation(attestation)

    observed = runtime.require_active_owner_runtime(REVISION)
    observed["tree_sha256"] = "f" * 64
    assert runtime.require_active_owner_runtime(REVISION) == attestation

    monkeypatch.setattr(runtime, "_ACTIVE_ATTESTATION", None)
    with pytest.raises(
        runtime.ProductionOwnerRuntimeError,
        match="not_active",
    ):
        runtime.require_active_owner_runtime(REVISION)


def test_tree_manifest_rejects_any_writable_release_entry(tmp_path: Path) -> None:
    root = tmp_path / "sealed-runtime"
    root.mkdir()
    entry = root / "module.py"
    entry.write_text("VALUE = 1\n", encoding="utf-8")
    entry.chmod(0o644)
    root.chmod(0o555)
    try:
        with pytest.raises(
            runtime.ProductionOwnerRuntimeError,
            match="tree_writable",
        ):
            runtime.collect_tree_entries(root)
    finally:
        root.chmod(0o755)


def test_manifest_decoder_rejects_duplicate_keys_and_noncanonical_bytes() -> None:
    with pytest.raises(
        runtime.ProductionOwnerRuntimeError,
        match="manifest_invalid",
    ):
        runtime._decode_manifest(b'{"a":1,"a":1}\n')

    value = {"schema": "example", "value": 1}
    pretty = json.dumps(value, indent=2).encode("ascii") + b"\n"
    with pytest.raises(
        runtime.ProductionOwnerRuntimeError,
        match="manifest_invalid",
    ):
        runtime._decode_manifest(pretty)


def test_required_module_set_imports_crypto_and_every_mutating_launcher() -> None:
    assert "cryptography" in runtime.REQUIRED_MODULES
    assert "scripts.canary.production_cutover_owner_launcher" in (
        runtime.REQUIRED_MODULES
    )
    assert "scripts.canary.production_os_login_metadata_migration" in (
        runtime.REQUIRED_MODULES
    )
    assert "gateway.production_cron_cutover_runtime" in runtime.REQUIRED_MODULES
    assert "gateway.operational_edge_readiness" in runtime.REQUIRED_MODULES
