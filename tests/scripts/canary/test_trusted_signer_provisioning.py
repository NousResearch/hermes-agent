from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.canary import trusted_signer_provisioning as provisioning


class SimulatedCrash(RuntimeError):
    pass


def _crash() -> None:
    raise SimulatedCrash("simulated_power_loss")


@pytest.mark.parametrize(
    "hook",
    (
        "after_open",
        "after_write_chunk",
        "after_fsync",
        "after_publish_link",
    ),
)
def test_exclusive_install_replays_every_crash_window(
    tmp_path: Path,
    hook: str,
) -> None:
    parent = tmp_path / "protected"
    parent.mkdir(mode=0o700)
    destination = parent / "signer.key"
    payload = bytes(range(64))

    with pytest.raises(SimulatedCrash):
        provisioning._install_exclusive(
            destination,
            payload,
            uid=os.getuid(),
            gid=os.getgid(),
            mode=0o400,
            include_digest=False,
            after_open=_crash if hook == "after_open" else None,
            after_write_chunk=(
                _crash if hook == "after_write_chunk" else None
            ),
            after_fsync=_crash if hook == "after_fsync" else None,
            after_publish_link=(
                _crash if hook == "after_publish_link" else None
            ),
        )

    evidence = provisioning._install_exclusive(
        destination,
        payload,
        uid=os.getuid(),
        gid=os.getgid(),
        mode=0o400,
        include_digest=False,
    )
    assert destination.read_bytes() == payload
    assert destination.stat().st_nlink == 1
    assert not (parent / ".signer.key.muncho-staged").exists()
    assert "sha256" not in evidence
    assert evidence["size"] == 64


def test_exclusive_install_rejects_nonprefix_staged_bytes(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "protected"
    parent.mkdir(mode=0o700)
    destination = parent / "signer.key"
    staged = parent / ".signer.key.muncho-staged"
    staged.write_bytes(b"not-a-prefix")
    staged.chmod(0o400)
    staged_identity = staged.stat()

    with pytest.raises(
        provisioning.TrustedSignerProvisioningError,
        match="trusted_signer_staging_invalid",
    ):
        provisioning._install_exclusive(
            destination,
            b"expected-secret-seed-material",
            uid=staged_identity.st_uid,
            gid=staged_identity.st_gid,
            mode=0o400,
            include_digest=False,
        )
    assert staged.read_bytes() == b"not-a-prefix"
    assert not destination.exists()


def test_exclusive_install_never_replaces_conflicting_final(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "protected"
    parent.mkdir(mode=0o700)
    destination = parent / "signer.key"
    destination.write_bytes(b"existing")
    destination.chmod(0o400)
    destination_identity = destination.stat()

    with pytest.raises(
        provisioning.TrustedSignerProvisioningError,
        match="trusted_signer_install_conflict",
    ):
        provisioning._install_exclusive(
            destination,
            b"replacement",
            uid=destination_identity.st_uid,
            gid=destination_identity.st_gid,
            mode=0o400,
            include_digest=False,
        )
    assert destination.read_bytes() == b"existing"


def test_exclusive_install_rejects_symlink_final(tmp_path: Path) -> None:
    parent = tmp_path / "protected"
    parent.mkdir(mode=0o700)
    target = parent / "target"
    target.write_bytes(b"target")
    target.chmod(0o400)
    destination = parent / "signer.key"
    destination.symlink_to(target)

    with pytest.raises(provisioning.TrustedSignerProvisioningError):
        provisioning._install_exclusive(
            destination,
            b"target",
            uid=os.getuid(),
            gid=os.getgid(),
            mode=0o400,
            include_digest=False,
        )
    assert destination.is_symlink()


def test_selected_release_rejects_non_root_symlink_owner(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    release = tmp_path / "releases" / revision
    release.mkdir(parents=True)
    current = tmp_path / "current"
    current.symlink_to(release)
    layout = provisioning.cloud_layout(revision)
    object.__setattr__(layout, "release_base", release.parent)
    object.__setattr__(layout, "release", release)
    object.__setattr__(layout, "current_link", current)

    with pytest.raises(
        provisioning.TrustedSignerProvisioningError,
        match="trusted_signer_current_release_invalid",
    ):
        provisioning._selected_release_evidence(layout)
