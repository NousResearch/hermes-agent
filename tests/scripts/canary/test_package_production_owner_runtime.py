from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import pytest

from scripts.canary import package_production_owner_runtime as package


REVISION = "a" * 40
ROOT = Path(__file__).parents[3]


def _spec(tmp_path: Path) -> package.OwnerRuntimeBuildSpec:
    source = tmp_path / "source"
    release = tmp_path / "releases"
    source.mkdir()
    release.mkdir()
    uv = tmp_path / "uv"
    git = tmp_path / "git"
    uv.touch()
    git.touch()
    return package.OwnerRuntimeBuildSpec(
        revision=REVISION,
        source_root=source,
        release_base=release,
        uv_executable=uv,
        git_executable=git,
    )


def test_build_argv_is_frozen_noneditable_and_hash_constrained(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    commands = package.build_commands(
        spec,
        managed_python=spec.python_root / "cpython/bin/python3.11",
    )

    assert commands[0][1:3] == ("-I", "-B")
    assert "-S" not in commands[0]
    assert "--copies" in commands[0]
    assert "--frozen" in commands[1]
    assert "--no-editable" in commands[1]
    assert "--no-dev" in commands[1]
    assert "--no-install-project" in commands[1]
    assert "--require-hashes" in commands[2]
    assert "--force-pep517" in commands[2]


def test_runtime_argv_uses_site_imports_but_rejects_ambient_python(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    command = package.runtime_command(
        spec,
        "run",
        "author-freeze",
        "--revision",
        REVISION,
    )

    assert command[:4] == (
        str(spec.interpreter),
        "-I",
        "-B",
        "-m",
    )
    assert command[4:8] == (
        "gateway.production_owner_runtime",
        "run",
        "--revision",
        REVISION,
    )
    assert "-S" not in command
    assert command[8:] == (
        "--",
        "author-freeze",
        "--revision",
        REVISION,
    )


def test_package_metadata_includes_owner_launchers_and_operational_rails() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    included = set(metadata["tool"]["setuptools"]["packages"]["find"]["include"])

    assert {"scripts", "scripts.canary", "scripts.canary.*"} <= included
    assert {"ops", "ops.muncho", "ops.muncho.*"} <= included


def test_spec_rejects_release_nested_inside_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    spec = package.OwnerRuntimeBuildSpec(
        revision=REVISION,
        source_root=source,
        release_base=source / "releases",
        uv_executable=tmp_path / "uv",
        git_executable=tmp_path / "git",
    )

    with pytest.raises(
        package.ProductionOwnerRuntimePackagingError,
        match="spec_invalid",
    ):
        spec.validate()


def _publication_receipt(
    spec: package.OwnerRuntimeBuildSpec,
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema": package.RECEIPT_SCHEMA,
        "release_revision": spec.revision,
        "release_root": str(spec.release_root),
        "manifest_sha256": "1" * 64,
        "attestation_sha256": "2" * 64,
        "interpreter_sha256": "3" * 64,
        "pyvenv_cfg_sha256": "4" * 64,
        "wheel_sha256": "5" * 64,
        "runtime_reused": False,
        "non_editable_install": True,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    canonical = json.dumps(
        unsigned,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return {
        **unsigned,
        "receipt_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def test_publication_receipt_has_one_exact_shape_for_build_and_reuse(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    receipt = _publication_receipt(spec)

    assert package.validate_publication_receipt(receipt, spec=spec) == receipt

    reused = _publication_receipt(spec)
    reused["runtime_reused"] = True
    unsigned = {
        name: value
        for name, value in reused.items()
        if name != "receipt_sha256"
    }
    canonical = json.dumps(
        unsigned,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    reused["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()

    assert package.validate_publication_receipt(reused, spec=spec) == reused


@pytest.mark.parametrize("field", ["wheel_sha256", "receipt_sha256"])
def test_publication_receipt_rejects_missing_or_unbound_digest(
    tmp_path: Path,
    field: str,
) -> None:
    spec = _spec(tmp_path)
    receipt = _publication_receipt(spec)
    if field == "wheel_sha256":
        del receipt[field]
    else:
        receipt[field] = "f" * 64

    with pytest.raises(
        package.ProductionOwnerRuntimePackagingError,
        match="receipt_invalid",
    ):
        package.validate_publication_receipt(receipt, spec=spec)


def test_retained_wheel_requires_one_regular_non_symlink_artifact(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    spec.artifact_root.mkdir(parents=True)
    wheel = spec.artifact_root / "hermes_agent-1.0-py3-none-any.whl"
    wheel.write_bytes(b"sealed-wheel")

    assert package._retained_wheel(spec) == wheel

    (spec.artifact_root / "unexpected.txt").write_text("not a wheel")
    with pytest.raises(
        package.ProductionOwnerRuntimePackagingError,
        match="wheel_invalid",
    ):
        package._retained_wheel(spec)
