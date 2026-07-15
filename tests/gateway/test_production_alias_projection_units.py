from __future__ import annotations

import copy
import json
import stat
from pathlib import Path

import pytest

from gateway import production_alias_projection_units as units
from gateway.production_alias_projection_cutover import load_runtime_context


REVISION = "a" * 40


def _bundle():
    return units.render_production_alias_projection_units(
        revision=REVISION,
        database_ip="10.20.30.40",
        writer_user="muncho-canonical-writer",
        writer_group="muncho-canonical-writer",
        writer_uid=2000,
        writer_gid=2000,
        projector_user="muncho-projector",
        projector_group="muncho-projector",
        projector_uid=2004,
        projector_gid=2004,
        gateway_user="ai-platform-brain",
        gateway_group="ai-platform-brain",
        gateway_uid=1000,
        gateway_gid=1000,
        interpreter_sha256="1" * 64,
        writer_module_sha256="2" * 64,
        projector_module_sha256="3" * 64,
        projection_reader_sha256="4" * 64,
        team_registry_sha256="5" * 64,
        cutover_runtime_sha256="6" * 64,
        cutover_entrypoint_sha256="7" * 64,
    )


def test_units_separate_writer_credential_from_credential_free_projector():
    bundle = _bundle()
    manifest = bundle.manifest()
    exporter = bundle.exporter_service
    projector = bundle.projector_service
    timer = bundle.projector_timer

    assert b"canonical-writer-db-password" in exporter
    assert b"canonical-writer-db-password" not in projector
    assert b"LoadCredential=" not in projector
    assert b"EnvironmentFile=" not in projector
    assert b"PrivateNetwork=yes" in projector
    assert f"Requires={units.EXPORTER_UNIT}".encode() in projector
    assert f"Unit={units.PROJECTOR_UNIT}".encode() in timer
    assert b"[Install]" not in exporter
    assert b"[Install]" not in projector
    assert manifest["ordering"]["timer_enabled_before_activation"] is False
    assert manifest["credential_boundary"]["projector_credential_paths"] == []
    assert manifest["package_sha256"] == bundle.package_sha256


def test_package_manifest_tamper_and_identity_alias_fail_closed():
    bundle = _bundle()
    tampered = copy.deepcopy(bundle.manifest())
    tampered["ordering"]["interval_seconds"] = 1
    with pytest.raises(
        units.ProductionAliasProjectionUnitError,
        match="package_identity_invalid",
    ):
        units.validate_package_manifest(tampered)

    kwargs = dict(
        revision=REVISION,
        database_ip="10.20.30.40",
        writer_user="muncho-canonical-writer",
        writer_group="muncho-canonical-writer",
        writer_uid=2000,
        writer_gid=2000,
        projector_user="muncho-projector",
        projector_group="muncho-projector",
        projector_uid=2000,
        projector_gid=2004,
        gateway_user="ai-platform-brain",
        gateway_group="ai-platform-brain",
        gateway_uid=1000,
        gateway_gid=1000,
        interpreter_sha256="1" * 64,
        writer_module_sha256="2" * 64,
        projector_module_sha256="3" * 64,
        projection_reader_sha256="4" * 64,
        team_registry_sha256="5" * 64,
        cutover_runtime_sha256="6" * 64,
        cutover_entrypoint_sha256="7" * 64,
    )
    with pytest.raises(
        units.ProductionAliasProjectionUnitError,
        match="identity_invalid",
    ):
        units.render_production_alias_projection_units(**kwargs)


def test_release_package_is_atomic_read_only_and_loader_rejects_symlink(tmp_path):
    bundle = _bundle()
    release = (tmp_path / f"hermes-agent-{REVISION[:12]}").resolve()
    release.mkdir()
    (release / ".codex-source-commit").write_text(REVISION + "\n", encoding="ascii")

    manifest = units.write_release_alias_projection_package(release, bundle)
    package_root = release / units.PACKAGE_RELATIVE_ROOT
    assert stat.S_IMODE((package_root / "manifest.json").stat().st_mode) == 0o444
    assert not list(package_root.glob(".*.tmp"))
    context = load_runtime_context(
        package_root=package_root,
        expected_revision=REVISION,
        expected_package_sha256=manifest["package_sha256"],
        enforce_production_address=False,
        enforce_package_metadata=False,
    )
    assert context.package_sha256 == manifest["package_sha256"]

    target = package_root / units.PROJECTOR_UNIT
    payload = target.read_bytes()
    target.unlink()
    real = package_root / "real-projector.service"
    real.write_bytes(payload)
    target.symlink_to(real)
    with pytest.raises(Exception, match="file_unavailable|identity_invalid"):
        load_runtime_context(
            package_root=package_root,
            expected_revision=REVISION,
            expected_package_sha256=manifest["package_sha256"],
            enforce_production_address=False,
            enforce_package_metadata=False,
        )


def test_manifest_is_canonical_json_round_trip():
    manifest = _bundle().manifest()
    encoded = json.dumps(
        manifest,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    assert units.validate_package_manifest(json.loads(encoded)) == manifest


@pytest.mark.parametrize("symlink_target", ("marker", "package_directory"))
def test_release_package_writer_rejects_symlink_redirection(
    tmp_path, symlink_target
):
    bundle = _bundle()
    release = (tmp_path / f"hermes-agent-{REVISION[:12]}").resolve()
    release.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    if symlink_target == "marker":
        marker = real / ".codex-source-commit"
        marker.write_text(REVISION + "\n", encoding="ascii")
        (release / ".codex-source-commit").symlink_to(marker)
    else:
        (release / ".codex-source-commit").write_text(
            REVISION + "\n", encoding="ascii"
        )
        (release / "ops").symlink_to(real)

    with pytest.raises(
        units.ProductionAliasProjectionUnitError,
        match="release_invalid|directory_untrusted",
    ):
        units.write_release_alias_projection_package(release, bundle)
