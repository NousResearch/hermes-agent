#!/usr/bin/env python3
"""Publish exact release source and author unsigned owner-gate trust.

Both operations are intentionally owner-side edge commands.  They accept only
public, immutable evidence and never accept a private key or a caller-selected
publication destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Never, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from gateway import canonical_writer_production_cutover as cutover
from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_bootstrap as bootstrap
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_inert_input_preparer as inert_inputs
from scripts.canary import owner_gate_outer_stage0 as outer
from scripts.canary import owner_gate_package as package
from scripts.canary import owner_gate_pre_foundation as pre_foundation
from scripts.canary import owner_gate_trust as trust
from scripts.canary import owner_gate_trust_author as trust_author


SCHEMA = "muncho-owner-gate-release-authoring-receipt.v1"
FORK_ORIGIN = "https://github.com/lomliev/hermes-agent.git"
OWNER_HOME = Path(pwd.getpwuid(os.geteuid()).pw_dir)  # POSIX owner boundary
TRUSTED_ROOT = OWNER_HOME / ".hermes" / "trusted"
RELEASE_SOURCE_BASE = TRUSTED_ROOT / "owner-gate-release-sources"
OWNER_CUTOVER_AUTHORITY_ROOT = (
    OWNER_HOME / ".hermes" / "owner-gate-production-cutover"
)
ISOLATED_CANARY_PREREQUISITE_ROOT = (
    OWNER_CUTOVER_AUTHORITY_ROOT / "isolated-canary-prerequisites"
)
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON = 16 * 1024 * 1024
_GIT = "/usr/bin/git"


class OwnerGateReleaseAuthorError(RuntimeError):
    """Stable, secret-free release authoring failure."""


def _error(code: str, _cause: BaseException | None = None) -> Never:
    raise OwnerGateReleaseAuthorError(code) from None


def _canonical(value: Any) -> bytes:
    try:
        return foundation.canonical_json_bytes(value)
    except foundation.OwnerGateFoundationError as exc:
        _error("owner_gate_release_author_json_invalid", exc)


def _cutover_canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        _error("owner_gate_release_author_json_invalid", exc)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_output(source: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            (_GIT, "-C", str(source), *arguments),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
            env={
                "PATH": "/usr/bin:/bin",
                "LC_ALL": "C",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "HOME": str(OWNER_HOME),
                "TMPDIR": "/tmp",
            },
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _error("owner_gate_release_author_git_invalid", exc)
    if (
        completed.returncode != 0
        or completed.stderr
        or len(completed.stdout) > 4 * 1024 * 1024
    ):
        _error("owner_gate_release_author_git_invalid")
    try:
        return completed.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeError as exc:
        _error("owner_gate_release_author_git_invalid", exc)


def _run_git(*arguments: str) -> None:
    try:
        completed = subprocess.run(
            (_GIT, *arguments),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=300,
            env={
                "PATH": "/usr/bin:/bin",
                "LC_ALL": "C",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "HOME": str(OWNER_HOME),
                "TMPDIR": "/tmp",
            },
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _error("owner_gate_release_author_git_invalid", exc)
    if completed.returncode != 0:
        _error("owner_gate_release_author_git_invalid")


def _verify_outer_source_files(
    source: Path,
    *,
    revision: str,
    tree_oid: str,
) -> None:
    for relative in outer.SOURCE_FILES.values():
        selected = package._git_blob(source, revision, relative, required=True)
        assert selected is not None
        raw, git_mode = selected
        package._require_worktree_matches_git_blob(
            source / relative,
            expected=raw,
            git_mode=git_mode,
        )
    manifest = outer.build_manifest(
        source,
        release_revision=revision,
        source_tree_oid=tree_oid,
    )
    outer.validate_manifest(manifest)


def _verify_release_source(
    source: Path,
    *,
    revision: str,
    require_main: bool,
) -> str:
    if (
        not source.is_absolute()
        or ".." in source.parts
        or _REVISION.fullmatch(revision or "") is None
    ):
        _error("owner_gate_release_author_source_invalid")
    try:
        head, tree_oid = package._verify_clean_git_source(source)
        top = Path(_git_output(source, "rev-parse", "--show-toplevel")).resolve(
            strict=True
        )
        exact = source.resolve(strict=True)
        remotes = _git_output(source, "remote").splitlines()
        origins = _git_output(source, "remote", "get-url", "--all", "origin").splitlines()
        branch = _git_output(source, "branch", "--show-current")
        origin_main = _git_output(
            source, "rev-parse", "--verify", "refs/remotes/origin/main"
        )
    except (
        OSError,
        package.OwnerGatePackageError,
        outer.OwnerGateOuterStage0Error,
    ) as exc:
        _error("owner_gate_release_author_source_invalid", exc)
    if (
        exact != top
        or head != revision
        or remotes != ["origin"]
        or origins != [FORK_ORIGIN]
        or origin_main != revision
        or (require_main and branch != "main")
        or (not require_main and branch)
    ):
        _error("owner_gate_release_author_source_invalid")
    try:
        _verify_outer_source_files(source, revision=revision, tree_oid=tree_oid)
    except (
        package.OwnerGatePackageError,
        outer.OwnerGateOuterStage0Error,
    ) as exc:
        _error("owner_gate_release_author_source_invalid", exc)
    return tree_oid


def _ensure_release_source_roots() -> None:
    try:
        hermes = inert_inputs._hermes_root()
        inert_inputs._ensure_directory(
            TRUSTED_ROOT,
            parent=hermes,
            mode=0o700,
            code="owner_gate_release_author_trusted_root_invalid",
        )
        inert_inputs._ensure_directory(
            RELEASE_SOURCE_BASE,
            parent=TRUSTED_ROOT,
            mode=0o700,
            code="owner_gate_release_author_source_root_invalid",
        )
    except launcher.OwnerLauncherError as exc:
        _error("owner_gate_release_author_source_root_invalid", exc)


def _ensure_cutover_authority_roots() -> None:
    try:
        hermes = inert_inputs._hermes_root()
        inert_inputs._ensure_directory(
            OWNER_CUTOVER_AUTHORITY_ROOT,
            parent=hermes,
            mode=0o700,
            code="owner_gate_release_author_cutover_root_invalid",
        )
        inert_inputs._ensure_directory(
            ISOLATED_CANARY_PREREQUISITE_ROOT,
            parent=OWNER_CUTOVER_AUTHORITY_ROOT,
            mode=0o700,
            code="owner_gate_release_author_cutover_root_invalid",
        )
    except launcher.OwnerLauncherError as exc:
        _error("owner_gate_release_author_cutover_root_invalid", exc)


def publish_release_source(
    *,
    source_root: Path,
    release_revision: str,
) -> Mapping[str, Any]:
    """Publish one exact fork-main checkout below the fixed trusted root."""

    source_tree_oid = _verify_release_source(
        source_root,
        revision=release_revision,
        require_main=True,
    )
    _ensure_release_source_roots()
    destination = RELEASE_SOURCE_BASE / release_revision
    pending = RELEASE_SOURCE_BASE / f".{release_revision}.pending"
    if os.path.lexists(destination):
        try:
            inert_inputs._require_directory(
                destination,
                parent=RELEASE_SOURCE_BASE,
                mode=0o700,
                code="owner_gate_release_author_destination_invalid",
            )
        except launcher.OwnerLauncherError as exc:
            _error("owner_gate_release_author_destination_invalid", exc)
        observed_tree = _verify_release_source(
            destination,
            revision=release_revision,
            require_main=False,
        )
        if observed_tree != source_tree_oid:
            _error("owner_gate_release_author_destination_conflict")
        created = False
    else:
        if os.path.lexists(pending):
            _error("owner_gate_release_author_manual_reconciliation_required")
        pending_created = False
        try:
            _run_git(
                "clone",
                "--no-hardlinks",
                "--no-checkout",
                "--",
                str(source_root),
                str(pending),
            )
            pending_created = True
            os.chmod(pending, 0o700)
            _run_git("-C", str(pending), "remote", "set-url", "origin", FORK_ORIGIN)
            _run_git(
                "-C",
                str(pending),
                "checkout",
                "--detach",
                "--force",
                release_revision,
            )
            observed_tree = _verify_release_source(
                pending,
                revision=release_revision,
                require_main=False,
            )
            if observed_tree != source_tree_oid:
                _error("owner_gate_release_author_source_changed")
            inert_inputs._fsync_directory(
                pending,
                code="owner_gate_release_author_publish_failed",
            )
            launcher._atomic_rename_no_replace(
                str(pending),
                str(destination),
                exists_code="owner_gate_release_author_destination_exists",
                failed_code="owner_gate_release_author_publish_failed",
            )
            pending_created = False
            inert_inputs._fsync_directory(
                RELEASE_SOURCE_BASE,
                code="owner_gate_release_author_publish_failed",
            )
            observed_tree = _verify_release_source(
                destination,
                revision=release_revision,
                require_main=False,
            )
            if observed_tree != source_tree_oid:
                _error("owner_gate_release_author_postwrite_invalid")
            created = True
        except BaseException:
            if pending_created and os.path.lexists(pending):
                shutil.rmtree(pending, ignore_errors=True)
            raise
    return {
        "schema": SCHEMA,
        "action": "publish-release-source",
        "release_revision": release_revision,
        "source_tree_oid": source_tree_oid,
        "publication_path": str(destination),
        "origin": FORK_ORIGIN,
        "detached_head": True,
        "created": created,
        "network_fetch_performed": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }


def _read_immutable(path: Path, *, maximum: int) -> bytes:
    try:
        return trust._read_immutable(
            path,
            maximum=maximum,
            expected_uid=os.geteuid(),
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
        )
    except trust.OwnerGateTrustError as exc:
        _error("owner_gate_release_author_input_invalid", exc)


def _read_canonical_json(path: Path, *, maximum: int = _MAX_JSON) -> Mapping[str, Any]:
    raw = _read_immutable(path, maximum=maximum)
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _error("owner_gate_release_author_input_invalid", exc)
    if not isinstance(value, Mapping) or _canonical(value) != raw:
        _error("owner_gate_release_author_input_invalid")
    return value


def _read_cutover_json(path: Path) -> Mapping[str, Any]:
    raw = _read_immutable(path, maximum=_MAX_JSON)
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _error("owner_gate_release_author_cutover_input_invalid", exc)
    if not isinstance(value, Mapping) or _cutover_canonical(value) != raw:
        _error("owner_gate_release_author_cutover_input_invalid")
    return value


def build_isolated_canary_goal_prerequisite(
    *,
    release_revision: str,
    fixture: Mapping[str, Any],
    workspace_gateway: Mapping[str, Any],
    cleanup_receipt: Mapping[str, Any],
    production_diff: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Derive the exact cutover prerequisite from four public receipts."""

    if _REVISION.fullmatch(release_revision or "") is None:
        _error("owner_gate_release_author_cutover_input_invalid")
    value = cutover.build_isolated_canary_goal_prerequisite(
        fixture=fixture,
        fixture_sha256=_sha256(_cutover_canonical(fixture)),
        workspace_gateway=workspace_gateway,
        cleanup_receipt=cleanup_receipt,
        production_diff=production_diff,
    )
    if value.get("release_revision") != release_revision:
        _error("owner_gate_release_author_cutover_revision_mismatch")
    return value


def author_isolated_canary_goal_prerequisite(
    *,
    release_revision: str,
    fixture_path: Path,
    workspace_gateway_path: Path,
    cleanup_receipt_path: Path,
    production_diff_path: Path,
) -> Mapping[str, Any]:
    """Publish the derived prerequisite at its fixed owner-only pathname."""

    inputs = (
        fixture_path,
        workspace_gateway_path,
        cleanup_receipt_path,
        production_diff_path,
    )
    if any(not path.is_absolute() or ".." in path.parts for path in inputs):
        _error("owner_gate_release_author_cutover_input_invalid")
    value = build_isolated_canary_goal_prerequisite(
        release_revision=release_revision,
        fixture=_read_cutover_json(fixture_path),
        workspace_gateway=_read_cutover_json(workspace_gateway_path),
        cleanup_receipt=_read_cutover_json(cleanup_receipt_path),
        production_diff=_read_cutover_json(production_diff_path),
    )
    _ensure_cutover_authority_roots()
    output = ISOLATED_CANARY_PREREQUISITE_ROOT / f"{release_revision}.json"
    raw = _cutover_canonical(value)
    existed = os.path.lexists(output)
    trust_author._publish_exclusive(
        output,
        raw,
        mode=0o444,
        code="owner_gate_release_author_cutover_write_failed",
    )
    return {
        "schema": SCHEMA,
        "action": "author-isolated-canary-prerequisite",
        "release_revision": release_revision,
        "publication_path": str(output),
        "publication_sha256": _sha256(raw),
        "evidence_sha256": value["evidence_sha256"],
        "created": not existed,
        "private_key_loaded": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }


def _collector_key(path: Path) -> tuple[Ed25519PublicKey, str]:
    raw = _read_immutable(path, maximum=32)
    if len(raw) != 32:
        _error("owner_gate_release_author_collector_key_invalid")
    try:
        key = Ed25519PublicKey.from_public_bytes(raw)
    except ValueError as exc:
        _error("owner_gate_release_author_collector_key_invalid", exc)
    return key, _sha256(raw)


def author_unsigned_trust(
    *,
    source_root: Path,
    release_revision: str,
    wheelhouse_root: Path,
    wheelhouse_manifest_path: Path,
    interpreter_sha256: str,
    foundation_source_revision: str,
    foundation_source_tree_oid: str,
    pre_foundation_authority_path: Path,
    owner_reauthentication_receipt_path: Path,
    network_evidence_path: Path,
    foundation_collector_public_key_path: Path,
    project_ancestry_evidence_path: Path,
    direct_iam_identity_authority_path: Path,
    network_collector_public_key_path: Path,
    cloud_collector_public_key_path: Path,
    host_collector_public_key_path: Path,
    credential_migration_envelope_path: Path,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Build and publish the exact unsigned trust manifest for one release."""

    paths = (
        source_root,
        wheelhouse_root,
        wheelhouse_manifest_path,
        pre_foundation_authority_path,
        owner_reauthentication_receipt_path,
        network_evidence_path,
        foundation_collector_public_key_path,
        project_ancestry_evidence_path,
        direct_iam_identity_authority_path,
        network_collector_public_key_path,
        cloud_collector_public_key_path,
        host_collector_public_key_path,
        credential_migration_envelope_path,
    )
    if (
        any(not item.is_absolute() or ".." in item.parts for item in paths)
        or _REVISION.fullmatch(release_revision or "") is None
        or _REVISION.fullmatch(foundation_source_revision or "") is None
        or _REVISION.fullmatch(foundation_source_tree_oid or "") is None
        or _SHA256.fullmatch(interpreter_sha256 or "") is None
        or type(now_unix) is bool
        or (now_unix is not None and (type(now_unix) is not int or now_unix <= 0))
    ):
        _error("owner_gate_release_author_input_invalid")
    source_tree_oid = _verify_release_source(
        source_root,
        revision=release_revision,
        require_main=True,
    )
    wheelhouse_manifest = _read_canonical_json(wheelhouse_manifest_path)
    try:
        inventory = package.build_inventory(package.PackageSpec(
            source_root=source_root,
            release_revision=release_revision,
            wheelhouse_root=wheelhouse_root,
            wheelhouse_manifest=wheelhouse_manifest,
            interpreter_sha256=interpreter_sha256,
            foundation_source_revision=foundation_source_revision,
            foundation_source_tree_oid=foundation_source_tree_oid,
            direct_iam_identity_authority_path=direct_iam_identity_authority_path,
        ))
    except package.OwnerGatePackageError as exc:
        _error("owner_gate_release_author_inventory_invalid", exc)

    release_public_raw = trust_author._read_exact_regular(
        trust_author.KEY_DIRECTORY / trust_author.PUBLIC_KEY_NAME,
        size=trust_author.PUBLIC_KEY_BYTES,
        modes=frozenset({0o400, 0o440, 0o444}),
        code="owner_gate_release_author_release_key_invalid",
    )
    signer_key_id = _sha256(release_public_raw)
    if signer_key_id != trust.PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256:
        _error("owner_gate_release_author_release_key_invalid")
    release_public_key = Ed25519PublicKey.from_public_bytes(release_public_raw)
    chain = trust_author._validate_foundation_chain_files(
        pre_foundation_authority_path=pre_foundation_authority_path,
        owner_reauthentication_receipt_path=owner_reauthentication_receipt_path,
        network_evidence_path=network_evidence_path,
        network_collector_public_key_path=foundation_collector_public_key_path,
        project_ancestry_evidence_path=project_ancestry_evidence_path,
        project_ancestry_collector_public_key_path=(
            foundation_collector_public_key_path
        ),
        direct_iam_identity_authority_path=direct_iam_identity_authority_path,
        release_public_key=release_public_key,
        final_release_revision=release_revision,
    )
    authority_raw = _read_immutable(
        pre_foundation_authority_path,
        maximum=pre_foundation.MAX_JSON_BYTES,
    )
    try:
        authority = json.loads(authority_raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _error("owner_gate_release_author_foundation_invalid", exc)
    if not isinstance(authority, Mapping) or _canonical(authority) != authority_raw:
        _error("owner_gate_release_author_foundation_invalid")
    image = authority.get("interpreter_image")
    if not isinstance(image, Mapping):
        _error("owner_gate_release_author_foundation_invalid")

    _network_key, network_key_id = _collector_key(
        network_collector_public_key_path
    )
    _cloud_key, cloud_key_id = _collector_key(
        cloud_collector_public_key_path
    )
    host_key, host_key_id = _collector_key(
        host_collector_public_key_path
    )
    final_collectors = {
        "network": network_key_id,
        "cloud": cloud_key_id,
        "host": host_key_id,
    }
    migration_raw = _read_immutable(
        credential_migration_envelope_path,
        maximum=bootstrap.MAX_JSON_BYTES,
    )
    try:
        migration = json.loads(migration_raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _error("owner_gate_release_author_migration_invalid", exc)
    if not isinstance(migration, Mapping) or _canonical(migration) != migration_raw:
        _error("owner_gate_release_author_migration_invalid")
    try:
        bootstrap.validate_migration(
            migration,
            release_revision=release_revision,
            host_public_key=host_key,
            host_key_id=host_key_id,
        )
    except bootstrap.OwnerGateBootstrapError as exc:
        _error("owner_gate_release_author_migration_invalid", exc)

    if (
        source_tree_oid != inventory.get("source_tree_oid")
        or foundation_source_revision != chain.foundation_source_revision
        or foundation_source_tree_oid != chain.foundation_source_tree_oid
        or interpreter_sha256 != chain.interpreter_sha256
        or interpreter_sha256 != inventory.get("interpreter_sha256")
        or chain.interpreter_version != package.PYTHON_VERSION
        or image.get("interpreter_sha256") != interpreter_sha256
        or image.get("python_version") != package.PYTHON_VERSION
        or inventory.get("direct_iam_identity_authority_sha256")
        != chain.direct_iam_identity_authority_sha256
        or inventory.get("pre_foundation_authority_sha256")
        != chain.pre_foundation_authority_sha256
        or inventory.get("foundation_apply_receipt_sha256")
        != chain.foundation_apply_receipt_sha256
        or inventory.get("resource_ancestor_chain")
        != list(chain.resource_ancestor_chain)
        or len(set(final_collectors.values())) != 3
        or chain.bootstrap_network_collector_public_key_id
        in set(final_collectors.values())
    ):
        _error("owner_gate_release_author_authority_mismatch")

    unsigned = {
        "schema": trust.TRUST_SCHEMA,
        "approved_for_offline_install": True,
        "fork_repository": trust.FORK_REPOSITORY,
        "release_revision": release_revision,
        "source_tree_oid": source_tree_oid,
        "foundation_source_revision": foundation_source_revision,
        "foundation_source_tree_oid": foundation_source_tree_oid,
        "package_inventory_sha256": foundation.sha256_json(inventory),
        "boot_image_self_link": str(image["image_self_link"]).removeprefix(
            "https://www.googleapis.com/compute/v1/"
        ),
        "collector_public_key_ids": final_collectors,
        "credential_migration_envelope_sha256": _sha256(migration_raw),
        "direct_iam_identity_authority_sha256": (
            chain.direct_iam_identity_authority_sha256
        ),
        "pre_foundation_authority_sha256": (
            chain.pre_foundation_authority_sha256
        ),
        "foundation_apply_receipt_sha256": chain.foundation_apply_receipt_sha256,
        "project_ancestry_evidence_sha256": (
            chain.project_ancestry_evidence_sha256
        ),
        "project_ancestry_chain_sha256": chain.project_ancestry_chain_sha256,
        "resource_ancestor_chain": list(chain.resource_ancestor_chain),
        "interpreter_image": dict(image),
        "release_attestation": {
            "purpose": trust.ATTESTATION_PURPOSE,
            "attested_at_unix": int(time.time()) if now_unix is None else now_unix,
        },
        "signer_key_id": signer_key_id,
    }
    try:
        trust._validate_unsigned(unsigned)
    except trust.OwnerGateTrustError as exc:
        _error("owner_gate_release_author_unsigned_invalid", exc)
    if (
        _verify_release_source(
            source_root,
            revision=release_revision,
            require_main=True,
        )
        != source_tree_oid
    ):
        _error("owner_gate_release_author_source_changed")
    trust_author._require_owner_directory(
        trust_author.KEY_DIRECTORY,
        expected=trust_author.KEY_DIRECTORY,
        parent=trust_author.AUTHORITY_PARENT,
        create=False,
    )
    trust_author._require_owner_directory(
        trust_author.MANIFEST_DIRECTORY,
        expected=trust_author.MANIFEST_DIRECTORY,
        parent=trust_author.KEY_DIRECTORY,
        create=False,
    )
    output = trust_author.MANIFEST_DIRECTORY / (
        f"{release_revision}.trust.unsigned.json"
    )
    raw = _canonical(unsigned)
    trust_author._publish_exclusive(
        output,
        raw,
        mode=0o444,
        code="owner_gate_release_author_unsigned_write_failed",
    )
    return {
        "schema": SCHEMA,
        "action": "author-unsigned-trust",
        "release_revision": release_revision,
        "source_tree_oid": source_tree_oid,
        "unsigned_manifest_path": str(output),
        "unsigned_manifest_sha256": _sha256(raw),
        "package_inventory_sha256": unsigned["package_inventory_sha256"],
        "private_key_loaded": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish-release-source")
    publish.add_argument("--source-root", type=Path, required=True)
    publish.add_argument("--release-revision", required=True)
    isolated = commands.add_parser("author-isolated-canary-prerequisite")
    isolated.add_argument("--release-revision", required=True)
    isolated.add_argument("--fixture", type=Path, required=True)
    isolated.add_argument("--workspace-gateway", type=Path, required=True)
    isolated.add_argument("--cleanup-receipt", type=Path, required=True)
    isolated.add_argument("--production-diff", type=Path, required=True)
    author = commands.add_parser("author-unsigned-trust")
    author.add_argument("--source-root", type=Path, required=True)
    author.add_argument("--release-revision", required=True)
    author.add_argument("--wheelhouse-root", type=Path, required=True)
    author.add_argument("--wheelhouse-manifest", type=Path, required=True)
    author.add_argument("--interpreter-sha256", required=True)
    author.add_argument("--foundation-source-revision", required=True)
    author.add_argument("--foundation-source-tree-oid", required=True)
    author.add_argument("--pre-foundation-authority", type=Path, required=True)
    author.add_argument("--owner-reauth-receipt", type=Path, required=True)
    author.add_argument("--network-evidence", type=Path, required=True)
    author.add_argument(
        "--foundation-collector-public-key", type=Path, required=True
    )
    author.add_argument("--project-ancestry-evidence", type=Path, required=True)
    author.add_argument(
        "--direct-iam-identity-authority", type=Path, required=True
    )
    author.add_argument("--network-collector-public-key", type=Path, required=True)
    author.add_argument("--cloud-collector-public-key", type=Path, required=True)
    author.add_argument("--host-collector-public-key", type=Path, required=True)
    author.add_argument("--credential-migration-envelope", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "publish-release-source":
            receipt = publish_release_source(
                source_root=arguments.source_root,
                release_revision=arguments.release_revision,
            )
        elif arguments.command == "author-isolated-canary-prerequisite":
            receipt = author_isolated_canary_goal_prerequisite(
                release_revision=arguments.release_revision,
                fixture_path=arguments.fixture,
                workspace_gateway_path=arguments.workspace_gateway,
                cleanup_receipt_path=arguments.cleanup_receipt,
                production_diff_path=arguments.production_diff,
            )
        else:
            receipt = author_unsigned_trust(
                source_root=arguments.source_root,
                release_revision=arguments.release_revision,
                wheelhouse_root=arguments.wheelhouse_root,
                wheelhouse_manifest_path=arguments.wheelhouse_manifest,
                interpreter_sha256=arguments.interpreter_sha256,
                foundation_source_revision=arguments.foundation_source_revision,
                foundation_source_tree_oid=arguments.foundation_source_tree_oid,
                pre_foundation_authority_path=arguments.pre_foundation_authority,
                owner_reauthentication_receipt_path=arguments.owner_reauth_receipt,
                network_evidence_path=arguments.network_evidence,
                foundation_collector_public_key_path=(
                    arguments.foundation_collector_public_key
                ),
                project_ancestry_evidence_path=(
                    arguments.project_ancestry_evidence
                ),
                direct_iam_identity_authority_path=(
                    arguments.direct_iam_identity_authority
                ),
                network_collector_public_key_path=(
                    arguments.network_collector_public_key
                ),
                cloud_collector_public_key_path=(
                    arguments.cloud_collector_public_key
                ),
                host_collector_public_key_path=(
                    arguments.host_collector_public_key
                ),
                credential_migration_envelope_path=(
                    arguments.credential_migration_envelope
                ),
            )
    except (
        OSError,
        OwnerGateReleaseAuthorError,
        launcher.OwnerLauncherError,
        trust_author.OwnerGateTrustAuthorError,
        bootstrap.OwnerGateBootstrapError,
        ValueError,
    ):
        print(
            '{"error_code":"owner_gate_release_authoring_failed","ok":false}',
            file=sys.stderr,
        )
        return 2
    print(_canonical(receipt).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORK_ORIGIN",
    "OwnerGateReleaseAuthorError",
    "RELEASE_SOURCE_BASE",
    "author_isolated_canary_goal_prerequisite",
    "author_unsigned_trust",
    "build_isolated_canary_goal_prerequisite",
    "publish_release_source",
]
