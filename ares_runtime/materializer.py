"""Ares-owned boundary around the existing verified release extraction core."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

from plugins.context_engine._context_governor import release_identity

from .contracts import ActivationGrant
from .errors import AresRuntimeError
from .image import verify_release_manifest
from .layout import AresRuntimeLayout


@dataclass(frozen=True)
class MaterializedRelease:
    """Verified immutable release returned to the activation transaction."""

    sealed_candidate_id: str
    release_root: Path
    observed_identity: dict[str, str]


def _seal_release_tree(root: Path) -> None:
    """Remove write permission from every verified release object."""

    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not (
            stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)
        ):
            raise AresRuntimeError("UNSAFE_RELEASE_OBJECT", str(path))
        if stat.S_ISDIR(info.st_mode):
            os.chmod(path, 0o555)
        else:
            os.chmod(path, 0o555 if info.st_mode & 0o111 else 0o444)
    os.chmod(root, 0o555)


def materialize_candidate_release(
    *,
    archive: Path,
    candidate_core: Path,
    certification_set: Path,
    sealed_candidate: Path,
    post_seal_evidence: Path,
    authorization: Path,
    layout: AresRuntimeLayout,
    grant: ActivationGrant,
) -> MaterializedRelease:
    """Materialize exactly the CandidateStore-authorized sealed release."""

    if Path(grant.target_release_root) != layout.releases_dir:
        raise AresRuntimeError("ACTIVATION_GRANT_TARGET_MISMATCH")
    try:
        verified = release_identity.materialize_verified_release(
            archive=archive,
            candidate_core=candidate_core,
            certification_set=certification_set,
            sealed_candidate=sealed_candidate,
            post_seal_evidence=post_seal_evidence,
            authorization=authorization,
            release_parent=layout.releases_dir,
        )
    except release_identity.ReleaseIdentityError as exc:
        raise AresRuntimeError("MATERIALIZATION_FAILED", exc.code) from exc
    if verified.sealed_candidate_id != grant.sealed_candidate_id:
        raise AresRuntimeError("MATERIALIZATION_IDENTITY_MISMATCH")
    if verified.release_root != layout.release_dir(grant.sealed_candidate_id):
        raise AresRuntimeError("MATERIALIZATION_TARGET_MISMATCH")
    try:
        verify_release_manifest(
            verified.release_root / "payload",
            expected_manifest_sha256=grant.release_manifest_sha256,
            expected_runtime_tree_sha256=grant.runtime_tree_sha256,
        )
    except AresRuntimeError as exc:
        raise AresRuntimeError(
            "MATERIALIZATION_RUNTIME_IDENTITY_MISMATCH", exc.code
        ) from exc
    _seal_release_tree(verified.release_root)
    return MaterializedRelease(
        sealed_candidate_id=verified.sealed_candidate_id,
        release_root=verified.release_root,
        observed_identity={
            **verified.observed_identity,
            "release_manifest_sha256": grant.release_manifest_sha256,
            "runtime_tree_sha256": grant.runtime_tree_sha256,
        },
    )


def materialize_from_candidate_store(
    store, layout: AresRuntimeLayout, sealed_candidate_id: str, grant: ActivationGrant
) -> MaterializedRelease:
    """Resolve materialization inputs only from already-verified custody."""

    snapshot = store.verify(sealed_candidate_id)
    if snapshot["sealed_candidate_id"] != grant.sealed_candidate_id:
        raise AresRuntimeError("ACTIVATION_GRANT_MISMATCH")
    candidate_root = store.root / "candidates" / sealed_candidate_id
    return materialize_candidate_release(
        archive=candidate_root / snapshot["archive_relative_path"],
        candidate_core=candidate_root / snapshot["candidate_core"]["relative_path"],
        certification_set=candidate_root
        / snapshot["certification_set_manifest"]["relative_path"],
        sealed_candidate=candidate_root
        / snapshot["sealed_candidate_manifest"]["relative_path"],
        post_seal_evidence=candidate_root
        / snapshot["post_seal_evidence_set"]["relative_path"],
        authorization=candidate_root
        / snapshot["activation_authorization"]["relative_path"],
        layout=layout,
        grant=grant,
    )
