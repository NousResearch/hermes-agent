"""Seal production-owned operational helpers into an immutable release.

The helpers are operational configuration, not credentials.  This packager
copies only the code-owned asset catalog, records stable source identities and
content digests, and fails if any catalog operation lacks an asset.  Secret
files are never opened by this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping

from gateway.operational_edge_catalog import (
    ASSET_ROOT_RELATIVE,
    CANONICAL_BRAIN,
    HERMES_HOME,
    asset_catalog,
    catalog_public_contract,
    operation_catalog,
)


ASSET_MANIFEST_SCHEMA = "muncho-operational-edge-assets.v1"
PACKAGED_ASSET_VERIFICATION_SCHEMA = (
    "muncho-operational-edge-assets-verification.v1"
)
ASSET_MANIFEST_RELATIVE = ASSET_ROOT_RELATIVE / "manifest.json"
MAX_ASSET_BYTES = 16 * 1024 * 1024
PRODUCTION_RELEASE_ROOT = Path(
    "/opt/adventico-ai-platform/hermes-agent-releases"
)

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OperationalEdgeAssetError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise OperationalEdgeAssetError("operational_asset_json_invalid") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_regular(path: Path) -> tuple[bytes, os.stat_result]:
    descriptor = -1
    try:
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= MAX_ASSET_BYTES
            or stat.S_IMODE(before.st_mode) & 0o022
        ):
            raise OperationalEdgeAssetError("operational_asset_source_invalid")
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    except OperationalEdgeAssetError:
        raise
    except OSError as exc:
        raise OperationalEdgeAssetError(
            "operational_asset_source_unavailable"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_gid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    raw = b"".join(chunks)
    if len(raw) != before.st_size or identity(before) != identity(opened) or identity(before) != identity(after):
        raise OperationalEdgeAssetError("operational_asset_source_changed")
    return raw, before


def _source_path(
    asset: Any,
    *,
    hermes_home: Path,
    canonical_brain: Path,
    release_root: Path,
) -> Path:
    roots = {
        "hermes": hermes_home,
        "canonical": canonical_brain,
        "release": release_root,
    }
    root = roots.get(asset.source_root)
    if root is None:
        raise OperationalEdgeAssetError("operational_asset_source_invalid")
    path = root / asset.source_relative
    if (
        not path.is_absolute()
        or ".." in path.parts
        or asset.source_root not in roots
    ):
        raise OperationalEdgeAssetError("operational_asset_source_invalid")
    return path


def build_operational_asset_manifest(
    *,
    revision: str,
    source_facts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if _REVISION.fullmatch(revision or "") is None:
        raise OperationalEdgeAssetError("operational_asset_revision_invalid")
    assets = asset_catalog()
    if not isinstance(source_facts, Mapping) or set(source_facts) != set(assets):
        raise OperationalEdgeAssetError("operational_asset_facts_invalid")
    rows: list[dict[str, Any]] = []
    for asset_id in sorted(assets):
        asset = assets[asset_id]
        fact = source_facts[asset_id]
        if (
            not isinstance(fact, Mapping)
            or set(fact) != {"source_path", "source_uid", "source_gid", "source_mode", "size", "sha256"}
            or not isinstance(fact["source_path"], str)
            or type(fact["source_uid"]) is not int
            or type(fact["source_gid"]) is not int
            or fact["source_mode"] not in {
                "0400", "0440", "0444", "0500", "0540", "0550", "0555",
                "0600", "0640", "0644", "0700", "0740", "0750", "0755",
            }
            or type(fact["size"]) is not int
            or not 0 < fact["size"] <= MAX_ASSET_BYTES
            or _SHA256.fullmatch(str(fact["sha256"])) is None
        ):
            raise OperationalEdgeAssetError("operational_asset_facts_invalid")
        rows.append(
            {
                "asset_id": asset_id,
                "source_path": fact["source_path"],
                "source_uid": fact["source_uid"],
                "source_gid": fact["source_gid"],
                "source_mode": fact["source_mode"],
                "packaged_relative": str(asset.packaged_relative),
                "packaged_mode": "0555",
                "size": fact["size"],
                "sha256": fact["sha256"],
            }
        )
    unsigned = {
        "schema": ASSET_MANIFEST_SCHEMA,
        "release_revision": revision,
        "catalog_sha256": _sha256(_canonical(catalog_public_contract())),
        "assets": rows,
        "asset_count": len(rows),
        "operation_count": len(operation_catalog()),
        "all_operations_implemented": True,
        "credential_material_packaged": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "manifest_sha256": _sha256(_canonical(unsigned))}


def validate_operational_asset_manifest(
    value: Any,
    *,
    revision: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OperationalEdgeAssetError("operational_asset_manifest_invalid")
    expected = {
        "schema", "release_revision", "catalog_sha256", "assets",
        "asset_count", "operation_count", "all_operations_implemented",
        "credential_material_packaged", "secret_digest_recorded",
        "manifest_sha256",
    }
    rows = value.get("assets")
    unsigned = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if (
        set(value) != expected
        or value.get("schema") != ASSET_MANIFEST_SCHEMA
        or value.get("release_revision") != revision
        or value.get("catalog_sha256") != _sha256(_canonical(catalog_public_contract()))
        or not isinstance(rows, list)
        or value.get("asset_count") != len(rows)
        or value.get("operation_count") != len(operation_catalog())
        or value.get("all_operations_implemented") is not True
        or value.get("credential_material_packaged") is not False
        or value.get("secret_digest_recorded") is not False
        or _SHA256.fullmatch(str(value.get("manifest_sha256") or "")) is None
        or value.get("manifest_sha256") != _sha256(_canonical(unsigned))
    ):
        raise OperationalEdgeAssetError("operational_asset_manifest_invalid")
    assets = asset_catalog()
    seen: set[str] = set()
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "asset_id", "source_path", "source_uid", "source_gid",
                "source_mode", "packaged_relative", "packaged_mode", "size", "sha256",
            }
            or row.get("asset_id") not in assets
            or row.get("asset_id") in seen
            or row.get("packaged_relative") != str(assets[row["asset_id"]].packaged_relative)
            or row.get("packaged_mode") != "0555"
            or not isinstance(row.get("source_path"), str)
            or not Path(row["source_path"]).is_absolute()
            or ".." in Path(row["source_path"]).parts
            or type(row.get("source_uid")) is not int
            or type(row.get("source_gid")) is not int
            or row["source_uid"] < 0
            or row["source_gid"] < 0
            or row.get("source_mode") not in {
                "0400", "0440", "0444", "0500", "0540", "0550", "0555",
                "0600", "0640", "0644", "0700", "0740", "0750", "0755",
            }
            or type(row.get("size")) is not int
            or not 0 < row["size"] <= MAX_ASSET_BYTES
            or _SHA256.fullmatch(str(row.get("sha256") or "")) is None
        ):
            raise OperationalEdgeAssetError("operational_asset_manifest_invalid")
        seen.add(row["asset_id"])
    if seen != set(assets):
        raise OperationalEdgeAssetError("operational_asset_manifest_invalid")
    return dict(value)


def _stable_packaged_regular(
    path: Path,
    *,
    expected_uid: int,
    expected_gid: int,
    expected_mode: int,
    maximum: int,
) -> tuple[bytes, os.stat_result]:
    descriptor = -1
    try:
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != expected_uid
            or before.st_gid != expected_gid
            or stat.S_IMODE(before.st_mode) != expected_mode
            or not 0 < before.st_size <= maximum
        ):
            raise OperationalEdgeAssetError(
                "operational_asset_packaged_file_invalid"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    except OperationalEdgeAssetError:
        raise
    except OSError as exc:
        raise OperationalEdgeAssetError(
            "operational_asset_packaged_file_unavailable"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_gid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or identity(before) != identity(opened)
        or identity(before) != identity(after)
    ):
        raise OperationalEdgeAssetError(
            "operational_asset_packaged_file_changed"
        )
    return raw, before


def verify_packaged_operational_assets(
    *,
    release_root: Path,
    revision: str,
    expected_uid: int,
    expected_gid: int,
    expected_manifest_sha256: str | None = None,
    reported_release_root: Path | None = None,
) -> dict[str, Any]:
    """Verify every packaged helper byte, identity and immutable mode.

    The physical build may still have a temporary name.  A trusted packager
    can report the already-authorized final release address while every byte is
    read from ``release_root``.  Exact owner IDs are mandatory inputs because
    production releases are owned by the signed deployment principal, not
    necessarily root.
    """

    reported = release_root if reported_release_root is None else reported_release_root
    if (
        _REVISION.fullmatch(revision or "") is None
        or not release_root.is_absolute()
        or ".." in release_root.parts
        or type(expected_uid) is not int
        or type(expected_gid) is not int
        or expected_uid < 0
        or expected_gid < 0
        or expected_manifest_sha256 is not None
        and _SHA256.fullmatch(expected_manifest_sha256) is None
        or not reported.is_absolute()
        or ".." in reported.parts
        or reported_release_root is not None
        and reported.name != f"hermes-agent-{revision[:12]}"
    ):
        raise OperationalEdgeAssetError(
            "operational_asset_verification_input_invalid"
        )
    manifest_raw, _manifest_metadata = _stable_packaged_regular(
        release_root / ASSET_MANIFEST_RELATIVE,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        expected_mode=0o444,
        maximum=MAX_ASSET_BYTES,
    )
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = item
        return result

    try:
        manifest_value = json.loads(
            manifest_raw.decode("ascii", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise OperationalEdgeAssetError(
            "operational_asset_manifest_invalid"
        ) from exc
    if (
        not isinstance(manifest_value, Mapping)
        or manifest_raw != _canonical(manifest_value) + b"\n"
    ):
        raise OperationalEdgeAssetError(
            "operational_asset_manifest_invalid"
        )
    manifest = validate_operational_asset_manifest(
        manifest_value,
        revision=revision,
    )
    if (
        expected_manifest_sha256 is not None
        and manifest["manifest_sha256"] != expected_manifest_sha256
    ):
        raise OperationalEdgeAssetError(
            "operational_asset_manifest_digest_mismatch"
        )
    assets = asset_catalog()
    files: list[dict[str, Any]] = []
    for row in manifest["assets"]:
        relative = assets[row["asset_id"]].packaged_relative
        physical_path = release_root / relative
        raw, metadata = _stable_packaged_regular(
            physical_path,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            expected_mode=0o555,
            maximum=MAX_ASSET_BYTES,
        )
        if len(raw) != row["size"] or _sha256(raw) != row["sha256"]:
            raise OperationalEdgeAssetError(
                "operational_asset_packaged_payload_mismatch"
            )
        files.append(
            {
                "asset_id": row["asset_id"],
                "path": str(reported / relative),
                "uid": metadata.st_uid,
                "gid": metadata.st_gid,
                "mode": "0555",
                "size": len(raw),
                "sha256": row["sha256"],
            }
        )
    unsigned = {
        "schema": PACKAGED_ASSET_VERIFICATION_SCHEMA,
        "release_revision": revision,
        "manifest_sha256": manifest["manifest_sha256"],
        "expected_uid": expected_uid,
        "expected_gid": expected_gid,
        "files": files,
        "file_count": len(files),
        "all_payloads_verified": len(files) == len(asset_catalog()),
        "credential_values_read": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    result = {
        **unsigned,
        "verification_sha256": _sha256(_canonical(unsigned)),
    }
    return validate_packaged_operational_asset_verification(
        result,
        revision=revision,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_release_root=(
            reported if reported_release_root is not None else None
        ),
        expected_uid=expected_uid,
        expected_gid=expected_gid,
    )


def validate_packaged_operational_asset_verification(
    value: Any,
    *,
    revision: str,
    expected_manifest_sha256: str | None = None,
    expected_release_root: Path | None = None,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "release_revision",
        "manifest_sha256",
        "expected_uid",
        "expected_gid",
        "files",
        "file_count",
        "all_payloads_verified",
        "credential_values_read",
        "secret_material_recorded",
        "secret_digest_recorded",
        "verification_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise OperationalEdgeAssetError(
            "operational_asset_verification_invalid"
        )
    unsigned = {
        key: item for key, item in value.items() if key != "verification_sha256"
    }
    rows = value.get("files")
    assets = asset_catalog()
    if (
        value.get("schema") != PACKAGED_ASSET_VERIFICATION_SCHEMA
        or value.get("release_revision") != revision
        or _SHA256.fullmatch(str(value.get("manifest_sha256") or "")) is None
        or expected_manifest_sha256 is not None
        and value.get("manifest_sha256") != expected_manifest_sha256
        or expected_release_root is not None
        and (
            not expected_release_root.is_absolute()
            or ".." in expected_release_root.parts
            or expected_release_root.name != f"hermes-agent-{revision[:12]}"
        )
        or type(value.get("expected_uid")) is not int
        or type(value.get("expected_gid")) is not int
        or value["expected_uid"] < 0
        or value["expected_gid"] < 0
        or expected_uid is not None
        and value["expected_uid"] != expected_uid
        or expected_gid is not None
        and value["expected_gid"] != expected_gid
        or not isinstance(rows, list)
        or value.get("file_count") != len(assets)
        or len(rows) != len(assets)
        or value.get("all_payloads_verified") is not True
        or value.get("credential_values_read") is not False
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or _SHA256.fullmatch(str(value.get("verification_sha256") or ""))
        is None
        or value.get("verification_sha256") != _sha256(_canonical(unsigned))
    ):
        raise OperationalEdgeAssetError(
            "operational_asset_verification_invalid"
        )
    expected_row_fields = {
        "asset_id",
        "path",
        "uid",
        "gid",
        "mode",
        "size",
        "sha256",
    }
    seen: set[str] = set()
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_row_fields
            or row.get("asset_id") not in assets
            or row.get("asset_id") in seen
            or not isinstance(row.get("path"), str)
            or not Path(row["path"]).is_absolute()
            or ".." in Path(row["path"]).parts
            or expected_release_root is not None
            and Path(row["path"])
            != expected_release_root / assets[row["asset_id"]].packaged_relative
            or row.get("uid") != value["expected_uid"]
            or row.get("gid") != value["expected_gid"]
            or row.get("mode") != "0555"
            or type(row.get("size")) is not int
            or not 0 < row["size"] <= MAX_ASSET_BYTES
            or _SHA256.fullmatch(str(row.get("sha256") or "")) is None
        ):
            raise OperationalEdgeAssetError(
                "operational_asset_verification_invalid"
            )
        seen.add(row["asset_id"])
    if seen != set(assets):
        raise OperationalEdgeAssetError(
            "operational_asset_verification_invalid"
        )
    return dict(value)


def package_operational_assets(
    *,
    release_root: Path,
    revision: str,
    hermes_home: Path = HERMES_HOME,
    canonical_brain: Path = CANONICAL_BRAIN,
) -> dict[str, Any]:
    """Copy every exact non-secret helper into one staging release."""

    marker, _metadata = _stable_regular(release_root / ".codex-source-commit")
    if marker != (revision + "\n").encode("ascii"):
        raise OperationalEdgeAssetError("operational_asset_release_invalid")
    assets = asset_catalog()
    payloads: dict[str, bytes] = {}
    facts: dict[str, dict[str, Any]] = {}
    for asset_id, asset in assets.items():
        source = _source_path(
            asset,
            hermes_home=hermes_home,
            canonical_brain=canonical_brain,
            release_root=release_root,
        )
        raw, metadata = _stable_regular(source)
        payloads[asset_id] = raw
        facts[asset_id] = {
            "source_path": str(source),
            "source_uid": metadata.st_uid,
            "source_gid": metadata.st_gid,
            "source_mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
            "size": len(raw),
            "sha256": _sha256(raw),
        }
    manifest = build_operational_asset_manifest(
        revision=revision, source_facts=facts
    )
    destination_root = release_root / ASSET_ROOT_RELATIVE
    destination_root.mkdir(parents=True, exist_ok=True, mode=0o755)
    for asset_id, raw in payloads.items():
        destination = release_root / assets[asset_id].packaged_relative
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        descriptor, temporary = tempfile.mkstemp(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
        )
        try:
            os.fchmod(descriptor, 0o555)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
    manifest_path = release_root / ASSET_MANIFEST_RELATIVE
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    os.chmod(manifest_path, 0o444)
    verify_packaged_operational_assets(
        release_root=release_root,
        revision=revision,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_uid=os.geteuid(),  # windows-footgun: ok — Linux release-packager boundary
        expected_gid=os.getegid(),  # windows-footgun: ok — Linux release-packager boundary
    )
    return manifest


__all__ = [
    "ASSET_MANIFEST_RELATIVE",
    "ASSET_MANIFEST_SCHEMA",
    "PACKAGED_ASSET_VERIFICATION_SCHEMA",
    "PRODUCTION_RELEASE_ROOT",
    "OperationalEdgeAssetError",
    "build_operational_asset_manifest",
    "package_operational_assets",
    "validate_operational_asset_manifest",
    "validate_packaged_operational_asset_verification",
    "verify_packaged_operational_assets",
]
