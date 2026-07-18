from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.canary import owner_gate_outer_stage0 as outer


ROOT = Path(__file__).parents[3]
REVISION = "a" * 40
TREE = "b" * 40


class SimulatedCrash(RuntimeError):
    pass


def _crash() -> None:
    raise SimulatedCrash("simulated_power_loss")


def _owner_kwargs(path: Path) -> dict[str, int]:
    state = path.stat()
    return {"owner_uid": state.st_uid, "owner_gid": state.st_gid}


def _incoming_kit(tmp_path: Path) -> tuple[Path, str, Path]:
    manifest = outer.build_manifest(
        ROOT,
        release_revision=REVISION,
        source_tree_oid=TREE,
    )
    manifest_raw = outer.canonical_json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    incoming_base = tmp_path / "incoming"
    incoming_base.mkdir()
    incoming = incoming_base / manifest_sha256
    outer.materialize_kit(
        ROOT,
        incoming,
        release_revision=REVISION,
        source_tree_oid=TREE,
    )
    sealer = tmp_path / "root-owned-sealer.py"
    shutil.copyfile(
        incoming / "scripts/canary/owner_gate_outer_stage0.py",
        sealer,
    )
    sealer.chmod(0o400)
    assert manifest["files"][2]["path"] == (
        "scripts/canary/owner_gate_outer_stage0.py"
    )
    return incoming, manifest_sha256, sealer


def _receiver_sealer(tmp_path: Path) -> tuple[Path, str]:
    sealer = tmp_path / "root-receiver.py"
    shutil.copyfile(ROOT / "scripts/canary/owner_gate_outer_stage0.py", sealer)
    sealer.chmod(0o400)
    return sealer, hashlib.sha256(sealer.read_bytes()).hexdigest()


def _tree_stream(
    tmp_path: Path,
    *,
    purpose: str,
    release_id: str,
) -> tuple[bytes, dict, Path, str]:
    source = tmp_path / f"source-{purpose}"
    (source / "bin").mkdir(parents=True)
    (source / "data/nested").mkdir(parents=True)
    executable = source / "bin/run"
    executable.write_bytes(b"#!/bin/false\ntrusted receiver fixture\n")
    executable.chmod(0o755)
    payload = source / "data/nested/payload.json"
    payload.write_bytes(b'{"safe":true}\n')
    payload.chmod(0o644)
    output = tmp_path / f"{purpose}.stream"
    build = outer.write_tree_stream(
        source,
        output,
        purpose=purpose,
        release_id=release_id,
    )
    sealer, sealer_sha256 = _receiver_sealer(tmp_path)
    return output.read_bytes(), build, sealer, sealer_sha256


def test_outer_stage0_seals_and_replays_exact_root_owned_closure(
    tmp_path: Path,
) -> None:
    incoming, manifest_sha256, sealer = _incoming_kit(tmp_path)
    release_base = tmp_path / "sealed" / "releases"
    receipt_base = tmp_path / "state" / "receipts"
    kwargs = {
        "expected_manifest_sha256": manifest_sha256,
        "incoming_base": incoming.parent,
        "release_base": release_base,
        "receipt_base": receipt_base,
        **_owner_kwargs(incoming),
        "require_root": False,
        "sealer_path": sealer,
    }

    first = outer.seal_incoming_kit(incoming, **kwargs)
    second = outer.seal_incoming_kit(incoming, **kwargs)

    release = release_base / manifest_sha256
    assert first == second
    assert first["incoming_payload_code_executed"] is False
    assert first["incoming_payload_imported"] is False
    assert first["release"]["path"] == str(release)
    assert (release / outer.TRUSTED_RUNNER).is_file()
    assert (release / "scripts/canary/owner_gate_stage0.py").is_file()
    assert (release / "scripts/canary/trusted_signer_stage0.py").is_file()
    assert release.stat().st_mode & 0o777 == 0o555
    assert not any("incoming" in str(path) for path in release.rglob("*.py"))
    assert (receipt_base / f"{manifest_sha256}.json").is_file()


def test_outer_stage0_rejects_tampered_incoming_module(tmp_path: Path) -> None:
    incoming, manifest_sha256, sealer = _incoming_kit(tmp_path)
    target = incoming / "scripts/canary/owner_gate_stage0.py"
    target.chmod(0o644)
    target.write_bytes(target.read_bytes() + b"\n# tampered\n")
    target.chmod(0o444)

    with pytest.raises(
        outer.OwnerGateOuterStage0Error,
        match="owner_gate_outer_stage0_incoming_invalid",
    ):
        outer.seal_incoming_kit(
            incoming,
            expected_manifest_sha256=manifest_sha256,
            incoming_base=incoming.parent,
            release_base=tmp_path / "sealed/releases",
            receipt_base=tmp_path / "state/receipts",
            **_owner_kwargs(incoming),
            require_root=False,
            sealer_path=sealer,
        )


def test_outer_stage0_rejects_extra_incoming_projection(tmp_path: Path) -> None:
    incoming, manifest_sha256, sealer = _incoming_kit(tmp_path)
    incoming.chmod(0o755)
    canary = incoming / "scripts/canary"
    canary.chmod(0o755)
    extra = canary / "unreviewed.py"
    extra.write_text("raise RuntimeError('must never execute')\n", encoding="utf-8")
    extra.chmod(0o444)
    canary.chmod(0o555)
    incoming.chmod(0o555)

    with pytest.raises(
        outer.OwnerGateOuterStage0Error,
        match="owner_gate_outer_stage0_incoming_invalid",
    ):
        outer.seal_incoming_kit(
            incoming,
            expected_manifest_sha256=manifest_sha256,
            incoming_base=incoming.parent,
            release_base=tmp_path / "sealed/releases",
            receipt_base=tmp_path / "state/receipts",
            **_owner_kwargs(incoming),
            require_root=False,
            sealer_path=sealer,
        )


def test_outer_stage0_rejects_symlinked_transient_sealer(tmp_path: Path) -> None:
    incoming, manifest_sha256, sealer = _incoming_kit(tmp_path)
    symlink = tmp_path / "sealer-link.py"
    symlink.symlink_to(sealer)

    with pytest.raises(
        outer.OwnerGateOuterStage0Error,
        match="owner_gate_outer_stage0_file_invalid",
    ):
        outer.seal_incoming_kit(
            incoming,
            expected_manifest_sha256=manifest_sha256,
            incoming_base=incoming.parent,
            release_base=tmp_path / "sealed/releases",
            receipt_base=tmp_path / "state/receipts",
            **_owner_kwargs(incoming),
            require_root=False,
            sealer_path=symlink,
        )


def test_outer_stage0_state_parent_is_private(tmp_path: Path) -> None:
    incoming, manifest_sha256, sealer = _incoming_kit(tmp_path)
    state_parent = tmp_path / "private-state"
    outer.seal_incoming_kit(
        incoming,
        expected_manifest_sha256=manifest_sha256,
        incoming_base=incoming.parent,
        release_base=tmp_path / "sealed/releases",
        receipt_base=state_parent / "receipts",
        **_owner_kwargs(incoming),
        require_root=False,
        sealer_path=sealer,
    )

    assert stat.S_IMODE(state_parent.stat().st_mode) == 0o700
    assert stat.S_IMODE((state_parent / "receipts").stat().st_mode) == 0o700


@pytest.mark.parametrize(
    ("purpose", "release_id", "base_name"),
    (
        ("outer-stage0-kit", "c" * 64, "kit-incoming"),
        ("owner-gate-bundle", REVISION, "bundle-incoming"),
    ),
)
def test_exact_stream_receiver_publishes_and_replays_root_owned_tree(
    tmp_path: Path,
    purpose: str,
    release_id: str,
    base_name: str,
) -> None:
    stream, build, sealer, sealer_sha256 = _tree_stream(
        tmp_path,
        purpose=purpose,
        release_id=release_id,
    )
    kit_base = tmp_path / "kit-incoming"
    bundle_base = tmp_path / "bundle-incoming"
    kwargs = {
        "purpose": purpose,
        "release_id": release_id,
        "expected_stream_manifest_sha256": build[
            "stream_manifest_sha256"
        ],
        "expected_self_sha256": sealer_sha256,
        "kit_base": kit_base,
        "bundle_base": bundle_base,
        "receipt_base": tmp_path / "private/transport-receipts",
        **_owner_kwargs(sealer),
        "require_root": False,
        "sealer_path": sealer,
    }

    first = outer.receive_exact_tree_stream(io.BytesIO(stream), **kwargs)
    second = outer.receive_exact_tree_stream(io.BytesIO(stream), **kwargs)

    received = tmp_path / base_name / release_id
    assert first == second
    assert first["input_code_executed"] is False
    assert first["input_code_imported"] is False
    assert first["extra_paths_received"] is False
    assert first["received_tree"]["path"] == str(received)
    assert (received / "bin/run").read_bytes().startswith(b"#!/bin/false")
    assert stat.S_IMODE((received / "bin/run").stat().st_mode) == 0o555
    assert stat.S_IMODE(
        (received / "data/nested/payload.json").stat().st_mode
    ) == 0o444
    assert stat.S_IMODE(received.stat().st_mode) == 0o555
    assert stat.S_IMODE(received.parent.stat().st_mode) == 0o700


def test_exact_stream_receiver_replays_truncated_transfer(tmp_path: Path) -> None:
    stream, build, sealer, sealer_sha256 = _tree_stream(
        tmp_path,
        purpose="owner-gate-bundle",
        release_id=REVISION,
    )
    kwargs = {
        "purpose": "owner-gate-bundle",
        "release_id": REVISION,
        "expected_stream_manifest_sha256": build[
            "stream_manifest_sha256"
        ],
        "expected_self_sha256": sealer_sha256,
        "kit_base": tmp_path / "kit-incoming",
        "bundle_base": tmp_path / "bundle-incoming",
        "receipt_base": tmp_path / "private/transport-receipts",
        **_owner_kwargs(sealer),
        "require_root": False,
        "sealer_path": sealer,
    }
    with pytest.raises(
        outer.OwnerGateOuterStage0Error,
        match="owner_gate_outer_stage0_stream_invalid",
    ):
        outer.receive_exact_tree_stream(io.BytesIO(stream[:-5]), **kwargs)
    assert not (tmp_path / "bundle-incoming" / REVISION).exists()

    receipt = outer.receive_exact_tree_stream(io.BytesIO(stream), **kwargs)
    assert receipt["release_id"] == REVISION
    assert (tmp_path / "bundle-incoming" / REVISION).is_dir()


def test_exact_stream_receiver_rejects_trailing_or_tampered_bytes(
    tmp_path: Path,
) -> None:
    stream, build, sealer, sealer_sha256 = _tree_stream(
        tmp_path,
        purpose="owner-gate-bundle",
        release_id=REVISION,
    )
    common = {
        "purpose": "owner-gate-bundle",
        "release_id": REVISION,
        "expected_stream_manifest_sha256": build[
            "stream_manifest_sha256"
        ],
        "expected_self_sha256": sealer_sha256,
        "kit_base": tmp_path / "kit-incoming",
        "receipt_base": tmp_path / "private/transport-receipts",
        **_owner_kwargs(sealer),
        "require_root": False,
        "sealer_path": sealer,
    }
    for suffix, payload in (
        ("trailing", stream + b"x"),
        ("tampered", stream[:-1] + bytes([stream[-1] ^ 1])),
    ):
        with pytest.raises(
            outer.OwnerGateOuterStage0Error,
            match="owner_gate_outer_stage0_stream_invalid",
        ):
            outer.receive_exact_tree_stream(
                io.BytesIO(payload),
                bundle_base=tmp_path / f"bundle-{suffix}",
                **common,
            )


def test_tree_stream_builder_rejects_symlink_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = source / "target"
    target.write_bytes(b"safe")
    (source / "alias").symlink_to(target.name)

    with pytest.raises(
        outer.OwnerGateOuterStage0Error,
        match="owner_gate_outer_stage0_tree_source_invalid",
    ):
        outer.build_tree_stream_manifest(
            source,
            purpose="owner-gate-bundle",
            release_id=REVISION,
        )


@pytest.mark.parametrize(
    "hook",
    ("after_open", "after_write_chunk", "after_fsync", "after_link"),
)
def test_outer_stage0_exact_publisher_replays_all_crash_windows(
    tmp_path: Path,
    hook: str,
) -> None:
    parent = tmp_path / "root-only"
    parent.mkdir(mode=0o700)
    destination = parent / "stage0.py"
    payload = b"trusted-stage-zero" * 8
    owner_uid = parent.stat().st_uid
    owner_gid = parent.stat().st_gid
    arguments = {
        "uid": owner_uid,
        "gid": owner_gid,
        "mode": 0o400,
        hook: _crash,
    }
    with pytest.raises(SimulatedCrash):
        outer._publish_exact_bytes(destination, payload, **arguments)

    outer._publish_exact_bytes(
        destination,
        payload,
        uid=owner_uid,
        gid=owner_gid,
        mode=0o400,
    )
    assert destination.read_bytes() == payload
    assert destination.stat().st_nlink == 1
    assert not (parent / ".stage0.py.outer-staged").exists()


@pytest.mark.parametrize(
    ("name", "error"),
    (
        (
            "muncho-owner-gate-bootstrap",
            "owner_gate_trusted_outer_stage0_required",
        ),
        (
            "muncho-host-offline-runtime-bootstrap",
            "owner_gate_trusted_outer_stage0_required",
        ),
        (
            "muncho-owner-gate-trusted-stage0",
            "owner_gate_trusted_stage0_entrypoint_invalid",
        ),
    ),
)
def test_uploaded_bundle_wrappers_fail_before_importing_attacker_code(
    tmp_path: Path,
    name: str,
    error: str,
) -> None:
    attacker = tmp_path / "attacker"
    package = attacker / "scripts"
    package.mkdir(parents=True)
    marker = tmp_path / "attacker-imported"
    (package / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported')\n",
        encoding="utf-8",
    )
    wrapper = ROOT / "ops/muncho/owner-gate/bin" / name
    environment = {**os.environ, "PYTHONPATH": str(attacker)}

    completed = subprocess.run(
        [sys.executable, str(wrapper)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert error in completed.stderr
    assert not marker.exists()
