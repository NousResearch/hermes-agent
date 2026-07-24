"""Fail-closed production Phase-B readiness for the Canonical Writer.

The production cutover publishes one root-owned, read-only database preflight
receipt.  This module lets the non-root writer consume that exact receipt
without entering the canary-only Phase-B runtime, whose fixed UID/GID and
release topology are intentionally different.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping


PRODUCTION_DATABASE = "ai_platform_brain"
PRODUCTION_PHASE_B_RECEIPT_PATH = Path(
    "/var/lib/muncho/canonical-writer-phase-b/runtime-receipt.json"
)
PRODUCTION_RELEASE_BASE = Path(
    "/opt/adventico-ai-platform/hermes-agent-releases"
)
PRODUCTION_WRITER_CONFIG_PATH = Path(
    "/etc/muncho-canonical-writer/writer.json"
)
PRODUCTION_PREFLIGHT_SCHEMA = (
    "muncho-production-legacy-cutover-preflight.v1"
)

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_RECEIPT_BYTES = 64 * 1024
_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "plan_sha256",
        "artifact_sha256",
        "final_snapshot_sha256",
        "source_row_count",
        "archive_row_count",
        "canonical_row_count",
        "archive_extended19_sha256",
        "canonical14_sha256",
        "relation_identity_sha256",
        "acl_identity_sha256",
        "index_identity_sha256",
        "roles_acl_sha256",
        "zero_canonical_writer_writes",
        "legacy_shape_restored",
        "ok",
        "legacy_truth_mode",
        "legacy_truth_decision_sha256",
        "legacy_truth_decision_event_id",
        "accepted_event_set_sha256",
        "trusted_legacy_event_count",
        "truth_epoch_sha256",
        "secret_material_recorded",
        "receipt_sha256",
    }
)
_DIGEST_FIELDS = frozenset(
    {
        "plan_sha256",
        "artifact_sha256",
        "final_snapshot_sha256",
        "archive_extended19_sha256",
        "canonical14_sha256",
        "relation_identity_sha256",
        "acl_identity_sha256",
        "index_identity_sha256",
        "roles_acl_sha256",
        "legacy_truth_decision_sha256",
        "accepted_event_set_sha256",
        "receipt_sha256",
    }
)


class ProductionWriterReadinessError(RuntimeError):
    """Stable, secret-free production writer readiness failure."""


def _fail(code: str) -> None:
    raise ProductionWriterReadinessError(code)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ProductionWriterReadinessError(
            "production_writer_readiness_json_invalid"
        ) from exc


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            _fail("production_writer_readiness_json_invalid")
        result[name] = value
    return result


def _reject_json_constant(_value: str) -> None:
    _fail("production_writer_readiness_json_invalid")


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _validate_trusted_parent_chain(path: Path, *, owner_uid: int) -> None:
    current = path
    while True:
        try:
            item = os.lstat(current)
        except OSError as exc:
            raise ProductionWriterReadinessError(
                "production_writer_readiness_parent_invalid"
            ) from exc
        if (
            stat.S_ISLNK(item.st_mode)
            or not stat.S_ISDIR(item.st_mode)
            or item.st_uid != owner_uid
            or stat.S_IMODE(item.st_mode) & 0o022
        ):
            _fail("production_writer_readiness_parent_invalid")
        if current == current.parent:
            return
        current = current.parent


def _read_stable_regular(
    path: Path,
    *,
    maximum: int,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
    expected_mode: int | None = None,
    forbidden_uid: int | None = None,
) -> bytes:
    if not path.is_absolute() or ".." in path.parts:
        _fail("production_writer_readiness_path_invalid")
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ProductionWriterReadinessError(
            "production_writer_readiness_file_unavailable"
        ) from exc
    mode = stat.S_IMODE(before.st_mode)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= maximum
        or (expected_uid is not None and before.st_uid != expected_uid)
        or (expected_gid is not None and before.st_gid != expected_gid)
        or (expected_mode is not None and mode != expected_mode)
        or (forbidden_uid is not None and before.st_uid == forbidden_uid)
        or (
            expected_mode is None
            and stat.S_IMODE(before.st_mode) & 0o022
        )
    ):
        _fail("production_writer_readiness_file_invalid")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        _fail("production_writer_readiness_nofollow_unavailable")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
        )
    except OSError as exc:
        raise ProductionWriterReadinessError(
            "production_writer_readiness_file_unavailable"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if _identity(opened) != _identity(before):
            _fail("production_writer_readiness_file_changed")
        chunks: list[bytes] = []
        total = 0
        while total <= maximum:
            chunk = os.read(
                descriptor,
                min(4096, maximum + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        reached = os.lstat(path)
    except OSError as exc:
        raise ProductionWriterReadinessError(
            "production_writer_readiness_file_changed"
        ) from exc
    if (
        not raw
        or len(raw) > maximum
        or _identity(before) != _identity(after)
        or _identity(after) != _identity(reached)
    ):
        _fail("production_writer_readiness_file_changed")
    return raw


def _canonical_uuid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        return False
    return str(parsed) == value


def _validate_receipt(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RECEIPT_FIELDS:
        _fail("production_writer_readiness_receipt_invalid")
    raw = dict(value)
    unsigned = {
        name: item
        for name, item in raw.items()
        if name != "receipt_sha256"
    }
    source_count = raw["source_row_count"]
    truth_mode = raw["legacy_truth_mode"]
    truth_epoch = raw["truth_epoch_sha256"]
    if (
        raw["schema"] != PRODUCTION_PREFLIGHT_SCHEMA
        or any(
            not isinstance(raw[name], str)
            or _SHA256.fullmatch(raw[name]) is None
            for name in _DIGEST_FIELDS
        )
        or raw["receipt_sha256"]
        != hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
        or type(source_count) is not int
        or source_count <= 0
        or raw["archive_row_count"] != source_count
        or type(raw["canonical_row_count"]) is not int
        or raw["canonical_row_count"] != 0
        or type(raw["trusted_legacy_event_count"]) is not int
        or raw["trusted_legacy_event_count"] != 0
        or raw["zero_canonical_writer_writes"] is not True
        or raw["legacy_shape_restored"] is not True
        or raw["ok"] is not True
        or raw["secret_material_recorded"] is not False
        or not isinstance(truth_mode, str)
        or truth_mode not in {
            "reseed_accepted_events",
            "start_new_truth_epoch",
        }
        or not _canonical_uuid(raw["legacy_truth_decision_event_id"])
        or (
            truth_mode == "reseed_accepted_events"
            and truth_epoch is not None
        )
        or (
            truth_mode == "start_new_truth_epoch"
            and (
                not isinstance(truth_epoch, str)
                or _SHA256.fullmatch(truth_epoch) is None
            )
        )
    ):
        _fail("production_writer_readiness_receipt_invalid")
    return raw


def _validate_release_identity(
    *,
    revision: str,
    release_base: Path,
    module_path: Path,
    executable: str,
    writer_uid: int,
) -> None:
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        _fail("production_writer_readiness_revision_invalid")
    release = release_base / f"hermes-agent-{revision[:12]}"
    try:
        release_status = os.lstat(release)
        resolved_release = release.resolve(strict=True)
    except OSError as exc:
        raise ProductionWriterReadinessError(
            "production_writer_readiness_release_unavailable"
        ) from exc
    if (
        stat.S_ISLNK(release_status.st_mode)
        or not stat.S_ISDIR(release_status.st_mode)
        or resolved_release != release
        or release_status.st_uid == writer_uid
        or stat.S_IMODE(release_status.st_mode) & 0o022
    ):
        _fail("production_writer_readiness_release_invalid")
    expected_module = (
        release / "gateway/canonical_writer_production_readiness.py"
    )
    if module_path != expected_module:
        _fail("production_writer_readiness_module_origin_invalid")
    _read_stable_regular(
        expected_module,
        maximum=1024 * 1024,
        forbidden_uid=writer_uid,
    )
    marker = _read_stable_regular(
        release / ".codex-source-commit",
        maximum=128,
        forbidden_uid=writer_uid,
    )
    if marker != f"{revision}\n".encode("ascii"):
        _fail("production_writer_readiness_release_mismatch")
    if executable != str(release / ".venv/bin/python"):
        _fail("production_writer_readiness_executable_invalid")


def _validate_production_phase_b_readiness_at(
    config: Any,
    *,
    release_revision: str,
    receipt_path: Path,
    release_base: Path,
    module_path: Path,
    executable: str,
    expected_receipt_uid: int,
    expected_receipt_gid: int,
    trusted_parent_uid: int | None,
    expected_config_path: Path = PRODUCTION_WRITER_CONFIG_PATH,
    expected_receipt_path: Path = PRODUCTION_PHASE_B_RECEIPT_PATH,
) -> Mapping[str, Any]:
    """Testable implementation; the public entry point pins every host path."""

    if (
        not isinstance(receipt_path, Path)
        or receipt_path != expected_receipt_path
        or getattr(getattr(config, "database", None), "database", None)
        != PRODUCTION_DATABASE
        or getattr(config, "source_config_path", None)
        != expected_config_path
        or _SHA256.fullmatch(
            str(getattr(config, "source_config_sha256", ""))
        )
        is None
        or getattr(
            getattr(config, "discord_edge_authority", None),
            "enabled",
            None,
        )
        is not True
    ):
        _fail("production_writer_readiness_config_invalid")
    writer_uid = getattr(config, "writer_uid", None)
    if type(writer_uid) is not int or writer_uid <= 0:
        _fail("production_writer_readiness_config_invalid")
    _validate_release_identity(
        revision=release_revision,
        release_base=release_base,
        module_path=module_path,
        executable=executable,
        writer_uid=writer_uid,
    )
    if trusted_parent_uid is not None:
        _validate_trusted_parent_chain(
            receipt_path.parent,
            owner_uid=trusted_parent_uid,
        )
    raw = _read_stable_regular(
        receipt_path,
        maximum=_MAX_RECEIPT_BYTES,
        expected_uid=expected_receipt_uid,
        expected_gid=expected_receipt_gid,
        expected_mode=0o444,
    )
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
        ProductionWriterReadinessError,
    ) as exc:
        raise ProductionWriterReadinessError(
            "production_writer_readiness_json_invalid"
        ) from exc
    if not isinstance(value, Mapping) or raw != _canonical_bytes(value):
        _fail("production_writer_readiness_json_invalid")
    return _validate_receipt(value)


def validate_fixed_production_phase_b_readiness(
    config: Any,
    *,
    release_revision: str,
    receipt_path: str | os.PathLike[str],
) -> Mapping[str, Any]:
    """Validate the fixed production preflight before constructing DB state."""

    if not sys.platform.startswith("linux"):
        _fail("production_writer_readiness_requires_linux")
    supplied = Path(receipt_path)
    if (
        not supplied.is_absolute()
        or ".." in supplied.parts
        or str(supplied) != os.fspath(receipt_path)
    ):
        _fail("production_writer_readiness_path_invalid")
    return _validate_production_phase_b_readiness_at(
        config,
        release_revision=release_revision,
        receipt_path=supplied,
        release_base=PRODUCTION_RELEASE_BASE,
        module_path=Path(__file__).resolve(strict=True),
        executable=sys.executable,
        expected_receipt_uid=0,
        expected_receipt_gid=0,
        trusted_parent_uid=0,
    )


__all__ = [
    "PRODUCTION_PHASE_B_RECEIPT_PATH",
    "PRODUCTION_PREFLIGHT_SCHEMA",
    "ProductionWriterReadinessError",
    "validate_fixed_production_phase_b_readiness",
]
