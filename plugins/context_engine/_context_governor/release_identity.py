"""Immutable sealed-release identity for the certified Context Governor path.

This module is deliberately independent of the candidate builder.  The builder
creates the five documents below; this module validates them and materializes a
release without consulting the development checkout, ``PATH``, or ambient
binary overrides.  It is safe to use in an activation *dry run* but performs no
live configuration change itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CANONICALIZATION_VERSION = "canonical-json-utf8-v1"
CANDIDATE_CORE_SCHEMA = "CandidateCoreV2"
CERTIFICATION_SET_SCHEMA = "CertificationSetV2"
SEALED_CANDIDATE_SCHEMA = "SealedCandidateV2"
POST_SEAL_EVIDENCE_SCHEMA = "PostSealEvidenceSetV1"
ACTIVATION_AUTHORIZATION_SCHEMA = "ActivationAuthorizationV1"
NON_AUTHORIZING = "NON_AUTHORIZING"


class ReleaseIdentityError(RuntimeError):
    """Stable, typed identity failure used by activation and hostile tests."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def object_id(value: dict[str, Any], field: str) -> str:
    projection = dict(value)
    projection.pop(field, None)
    return sha256_bytes(canonical_json(projection))


def _require_schema(value: dict[str, Any], expected: str) -> None:
    if (
        value.get("schema") != expected
        or value.get("canonicalization_version") != CANONICALIZATION_VERSION
    ):
        raise ReleaseIdentityError("WrongSchemaVersion", f"expected={expected}")


def _require_id(value: dict[str, Any], field: str, code: str) -> str:
    observed = value.get(field)
    if not isinstance(observed, str) or observed != object_id(value, field):
        raise ReleaseIdentityError(code)
    return observed


def _read_object(path: Path, schema: str, id_field: str, code: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        raw = path.read_bytes()
        if not raw.endswith(b"\n"):
            raise ValueError("missing canonical newline")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates)
        if not isinstance(value, dict) or canonical_json(value) + b"\n" != raw:
            raise ValueError("noncanonical authority JSON")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReleaseIdentityError("RuntimeIdentityUnavailable", str(path)) from exc
    _require_schema(value, schema)
    _require_id(value, id_field, code)
    return value


@dataclass(frozen=True)
class VerifiedRelease:
    sealed_candidate_id: str
    release_root: Path
    binary: Path
    adapter_root: Path
    rendered_config: Path
    observed_identity: dict[str, str]


def _safe_member_name(name: str) -> str:
    if not name or "\x00" in name or name.startswith("/") or "\\" in name:
        raise ReleaseIdentityError("UnsafeReleasePath", repr(name))
    path = Path(name)
    if any(part in ("", ".", "..") for part in path.parts):
        raise ReleaseIdentityError("UnsafeReleasePath", name)
    return name


def _validate_archive(archive: Path) -> list[tarfile.TarInfo]:
    seen: set[str] = set()
    normalized: set[str] = set()
    try:
        with tarfile.open(archive, "r:") as bundle:
            members = bundle.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise ReleaseIdentityError("WrongArchiveDigest", str(archive)) from exc
    for member in members:
        name = _safe_member_name(member.name)
        if name in seen:
            raise ReleaseIdentityError(
                "UnsafeReleasePath", f"duplicate archive path {name}"
            )
        seen.add(name)
        norm = unicodedata.normalize("NFC", name).casefold()
        if norm in normalized:
            raise ReleaseIdentityError(
                "UnsafeReleasePath", f"normalization collision {name}"
            )
        normalized.add(norm)
        if (
            member.issym()
            or member.islnk()
            or member.isdev()
            or member.isfifo()
            or not (member.isdir() or member.isfile())
        ):
            raise ReleaseIdentityError(
                "UnsafeReleasePath", f"forbidden archive member {name}"
            )
    return members


def _extract_safe(archive: Path, destination: Path) -> None:
    members = _validate_archive(archive)
    with tarfile.open(archive, "r:") as bundle:
        directories = sorted(
            (member for member in members if member.isdir()),
            key=lambda member: len(Path(member.name).parts),
        )
        files = [member for member in members if member.isfile()]
        for member in directories:
            target = destination / member.name
            target.mkdir(parents=True, exist_ok=False)
        for member in files:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise ReleaseIdentityError("UnsafeReleasePath", member.name)
            with source, open(target, "xb") as stream:
                shutil.copyfileobj(source, stream)
            os.chmod(target, stat.S_IMODE(member.mode) or 0o600)


def _file_digest(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ReleaseIdentityError("RuntimeIdentityMismatch", str(path))
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino, metadata.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise ReleaseIdentityError(
                "RuntimeIdentityMismatch", f"changed during read: {path}"
            )
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _payload_digest(entries: Iterable[dict[str, Any]]) -> str:
    return sha256_bytes(
        canonical_json(sorted(entries, key=lambda entry: entry["path"]))
    )


def _verify_payload(root: Path, core: dict[str, Any]) -> set[str]:
    entries = core.get("payload_files")
    if not isinstance(entries, list) or not entries:
        raise ReleaseIdentityError(
            "WrongSchemaVersion", "CandidateCoreV2 payload_files"
        )
    actual: list[dict[str, Any]] = []
    expected_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ReleaseIdentityError(
                "WrongSchemaVersion", "invalid payload file entry"
            )
        relative = _safe_member_name(entry["path"])
        expected_paths.add(relative)
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ReleaseIdentityError("RuntimeIdentityMismatch", relative)
        observed = _file_digest(path)
        if observed != entry.get("sha256"):
            code = (
                "WrongBinaryDigest"
                if relative == core.get("binary_path")
                else "RuntimeIdentityMismatch"
            )
            raise ReleaseIdentityError(code, relative)
        actual.append({
            "path": relative,
            "sha256": observed,
            "size": path.stat().st_size,
            "mode": stat.S_IMODE(path.stat().st_mode),
        })
    if _payload_digest(actual) != core.get("payload_tree_sha256"):
        raise ReleaseIdentityError("RuntimeIdentityMismatch", "payload tree")
    return expected_paths


def _component_digest(root: Path, relative: str, expected: str, code: str) -> Path:
    path = root / _safe_member_name(relative)
    if path.is_dir():
        entries = [
            {
                "path": item.relative_to(path).as_posix(),
                "mode": stat.S_IMODE(item.stat().st_mode),
                "size": item.stat().st_size,
                "sha256": _file_digest(item),
            }
            for item in sorted(child for child in path.rglob("*") if child.is_file())
        ]
        actual = sha256_bytes(canonical_json(entries))
    elif path.is_file():
        actual = _file_digest(path)
    else:
        actual = ""
    if actual != expected:
        raise ReleaseIdentityError(code, relative)
    return path


def materialize_verified_release(
    *,
    archive: Path,
    candidate_core: Path,
    certification_set: Path,
    sealed_candidate: Path,
    post_seal_evidence: Path,
    authorization: Path,
    release_parent: Path,
) -> VerifiedRelease:
    """Verify the complete staged identity chain and publish a sealed release.

    ``release_parent`` is caller-supplied only as a parent directory; the final
    release name is always the verified seal ID.  Existing releases are
    reverified, never silently reused.
    """
    core = _read_object(
        candidate_core, CANDIDATE_CORE_SCHEMA, "candidate_id", "WrongCandidateId"
    )
    cert = _read_object(
        certification_set,
        CERTIFICATION_SET_SCHEMA,
        "certification_set_id",
        "WrongCertificationSetId",
    )
    sealed = _read_object(
        sealed_candidate,
        SEALED_CANDIDATE_SCHEMA,
        "sealed_candidate_id",
        "WrongSealedCandidateId",
    )
    post = _read_object(
        post_seal_evidence,
        POST_SEAL_EVIDENCE_SCHEMA,
        "post_seal_evidence_set_id",
        "WrongSealedCandidateId",
    )
    auth = _read_object(
        authorization,
        ACTIVATION_AUTHORIZATION_SCHEMA,
        "activation_authorization_id",
        "WrongSealedCandidateId",
    )
    # The candidate-bundled document is deliberately an identity input for a
    # temporary dry-run only.  It is never an activation grant; live authority
    # must come from the governed CandidateStore transition.
    required_auth = {
        "schema",
        "canonicalization_version",
        "candidate_id",
        "certification_set_id",
        "sealed_candidate_id",
        "post_seal_evidence_set_id",
        "archive_sha256",
        "rendered_config_path",
        "rendered_config_sha256",
        "authorization_state",
        "non_authorizing",
        "approved_release_root",
        "governed_key_policy",
        "activation_authorization_id",
    }
    if (
        set(auth) != required_auth
        or auth.get("authorization_state") != NON_AUTHORIZING
        or auth.get("non_authorizing") is not True
    ):
        raise ReleaseIdentityError("NonAuthorizingArtifact", "activation input")
    if (
        cert.get("candidate_id") != core["candidate_id"]
        or sealed.get("candidate_id") != core["candidate_id"]
        or auth.get("candidate_id") != core["candidate_id"]
    ):
        raise ReleaseIdentityError("WrongCandidateId")
    if (
        sealed.get("certification_set_id") != cert["certification_set_id"]
        or auth.get("certification_set_id") != cert["certification_set_id"]
    ):
        raise ReleaseIdentityError("WrongCertificationSetId")
    if (
        post.get("sealed_candidate_id") != sealed["sealed_candidate_id"]
        or auth.get("sealed_candidate_id") != sealed["sealed_candidate_id"]
    ):
        raise ReleaseIdentityError("WrongSealedCandidateId")
    if auth.get("post_seal_evidence_set_id") != post["post_seal_evidence_set_id"]:
        raise ReleaseIdentityError("WrongSealedCandidateId")
    archive_digest = _file_digest(archive)
    if archive_digest != sealed.get("archive_sha256") or archive_digest != auth.get(
        "archive_sha256"
    ):
        raise ReleaseIdentityError("WrongArchiveDigest")
    expected_root = release_parent / sealed["sealed_candidate_id"]
    configured_root = auth.get("approved_release_root")
    if configured_root not in {
        str(expected_root),
        f"content-addressed/{sealed['sealed_candidate_id']}",
    }:
        raise ReleaseIdentityError("UnsafeReleasePath", "authorization release root")
    if expected_root.exists():
        raise ReleaseIdentityError(
            "RuntimeIdentityUnavailable", "refusing to reuse a release root"
        )
    release_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".cg-release-", dir=release_parent
    ) as temporary:
        staging = Path(temporary) / "release"
        staging.mkdir()
        _extract_safe(archive, staging)
        expected_paths = _verify_payload(staging, core)
        artifacts = cert.get("artifacts")
        if not isinstance(artifacts, list):
            raise ReleaseIdentityError(
                "WrongSchemaVersion", "CertificationSetV2 artifacts"
            )
        allowed = expected_paths | {
            "candidate-core-manifest.json",
            "certification-set-manifest.json",
        }
        for artifact in artifacts:
            if not isinstance(artifact, dict) or not isinstance(
                artifact.get("name"), str
            ):
                raise ReleaseIdentityError(
                    "WrongSchemaVersion", "CertificationSetV2 artifact"
                )
            name = _safe_member_name(artifact["name"])
            allowed.add(name)
            if _file_digest(staging / name) != artifact.get("sha256"):
                raise ReleaseIdentityError(
                    "RuntimeIdentityMismatch", f"certification artifact {name}"
                )
        observed_paths = {
            item.relative_to(staging).as_posix()
            for item in staging.rglob("*")
            if item.is_file()
        }
        if observed_paths != allowed:
            raise ReleaseIdentityError(
                "RuntimeIdentityMismatch", "unexpected or missing archive payload"
            )
        try:
            archive_core = json.loads(
                (staging / "candidate-core-manifest.json").read_text(encoding="utf-8")
            )
            archive_cert = json.loads(
                (staging / "certification-set-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ReleaseIdentityError(
                "RuntimeIdentityUnavailable", "archive manifests"
            ) from exc
        if archive_core != core or archive_cert != cert:
            raise ReleaseIdentityError("RuntimeIdentityMismatch", "archive manifests")
        binary = _component_digest(
            staging, core["binary_path"], core["binary_sha256"], "WrongBinaryDigest"
        )
        adapter = _component_digest(
            staging,
            core["adapter_bundle_path"],
            core["adapter_bundle_sha256"],
            "WrongAdapterDigest",
        )
        _component_digest(
            staging,
            core["activation_bootstrap_path"],
            core["activation_bootstrap_sha256"],
            "RuntimeIdentityMismatch",
        )
        template = _component_digest(
            staging,
            core["config_template_path"],
            core["config_template_sha256"],
            "WrongConfigDigest",
        )
        rendered = _component_digest(
            staging,
            auth["rendered_config_path"],
            auth["rendered_config_sha256"],
            "WrongConfigDigest",
        )
        binary_rel = binary.relative_to(staging)
        adapter_rel = adapter.relative_to(staging)
        rendered_rel = rendered.relative_to(staging)
        os.replace(staging, expected_root)
    return VerifiedRelease(
        sealed_candidate_id=sealed["sealed_candidate_id"],
        release_root=expected_root,
        binary=expected_root / binary_rel,
        adapter_root=expected_root / adapter_rel,
        rendered_config=expected_root / rendered_rel,
        observed_identity={
            "archive_sha256": archive_digest,
            "binary_sha256": core["binary_sha256"],
            "adapter_bundle_sha256": core["adapter_bundle_sha256"],
            "config_template_sha256": core["config_template_sha256"],
            "rendered_config_sha256": auth["rendered_config_sha256"],
        },
    )
