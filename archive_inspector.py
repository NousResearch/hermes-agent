"""Bounded ZIP/OOXML archive inspection helpers.

The helpers in this module validate ZIP central-directory metadata before any
CRC pass or member extraction happens.  Callers can then read individual members
through :func:`read_zip_member`, which enforces the same per-member bounds while
streaming so malformed archives cannot expand unbounded data into memory.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable


_DEFAULT_MAX_ENTRIES = 2_000
_DEFAULT_MAX_COMPRESSED_BYTES = 512 * 1024 * 1024
_DEFAULT_MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
_DEFAULT_MAX_MEMBER_BYTES = 64 * 1024 * 1024
_DEFAULT_MAX_XML_MEMBER_BYTES = 16 * 1024 * 1024
_DEFAULT_MAX_COMPRESSION_RATIO = 100.0
_READ_CHUNK_BYTES = 1024 * 1024
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class ZipInspectionError(ValueError):
    """Raised when a ZIP archive violates configured safety limits."""


@dataclass(frozen=True)
class ZipInspectionLimits:
    max_entries: int = _DEFAULT_MAX_ENTRIES
    max_compressed_bytes: int = _DEFAULT_MAX_COMPRESSED_BYTES
    max_uncompressed_bytes: int = _DEFAULT_MAX_UNCOMPRESSED_BYTES
    max_member_bytes: int = _DEFAULT_MAX_MEMBER_BYTES
    max_xml_member_bytes: int = _DEFAULT_MAX_XML_MEMBER_BYTES
    max_compression_ratio: float | None = _DEFAULT_MAX_COMPRESSION_RATIO


@dataclass(frozen=True)
class ZipMemberInfo:
    name: str
    file_size: int
    compress_size: int
    is_dir: bool


@dataclass(frozen=True)
class ZipInspectionReport:
    members: tuple[ZipMemberInfo, ...]
    total_compressed_bytes: int
    total_uncompressed_bytes: int

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(member.name for member in self.members)

    def has_member(self, name: str) -> bool:
        return any(member.name == name for member in self.members)


def _config_int(config: dict[str, object], key: str, default: int) -> int:
    raw = config.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ZipInspectionError(f"invalid integer archive limit gateway.file_intake.{key}={raw!r}") from exc
    if value <= 0:
        raise ZipInspectionError(f"archive limit gateway.file_intake.{key} must be positive")
    return value


def _config_float(config: dict[str, object], key: str, default: float | None) -> float | None:
    raw = config.get(key, default)
    if raw is None:
        return None
    if isinstance(raw, str) and raw.lower() in {"0", "off", "false", "none", "disabled"}:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ZipInspectionError(f"invalid numeric archive limit gateway.file_intake.{key}={raw!r}") from exc
    if value <= 0:
        raise ZipInspectionError(f"archive limit gateway.file_intake.{key} must be positive")
    return value


def _load_file_intake_config() -> dict[str, object]:
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config() or {}
        section = cfg_get(cfg, "gateway", "file_intake", default={}) or {}
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def file_intake_zip_limits_from_config() -> ZipInspectionLimits:
    """Return ZIP limits used by gateway intake and lightweight OOXML readers."""

    config = _load_file_intake_config()
    return ZipInspectionLimits(
        max_entries=_config_int(config, "max_archive_entries", _DEFAULT_MAX_ENTRIES),
        max_compressed_bytes=_config_int(
            config,
            "max_archive_compressed_bytes",
            _DEFAULT_MAX_COMPRESSED_BYTES,
        ),
        max_uncompressed_bytes=_config_int(
            config,
            "max_archive_uncompressed_bytes",
            _DEFAULT_MAX_UNCOMPRESSED_BYTES,
        ),
        max_member_bytes=_config_int(
            config,
            "max_archive_member_bytes",
            _DEFAULT_MAX_MEMBER_BYTES,
        ),
        max_xml_member_bytes=_config_int(
            config,
            "max_ooxml_xml_member_bytes",
            _DEFAULT_MAX_XML_MEMBER_BYTES,
        ),
        max_compression_ratio=_config_float(
            config,
            "max_compression_ratio",
            _DEFAULT_MAX_COMPRESSION_RATIO,
        ),
    )


def _safe_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    if not normalized or normalized.startswith("/") or _WINDOWS_DRIVE_RE.match(normalized):
        raise ZipInspectionError(f"unsafe archive member path: {name}")
    parts = PurePosixPath(normalized).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ZipInspectionError(f"unsafe archive member path: {name}")
    return normalized


def inspect_zip(zf: zipfile.ZipFile, limits: ZipInspectionLimits | None = None) -> ZipInspectionReport:
    """Validate central-directory metadata without reading member bodies."""

    limits = limits or ZipInspectionLimits()
    try:
        infos = zf.infolist()
    except zipfile.BadZipFile as exc:
        raise ZipInspectionError(f"not a valid zip archive: {exc}") from exc

    if len(infos) > limits.max_entries:
        raise ZipInspectionError(f"archive has too many entries: {len(infos)} > {limits.max_entries}")

    members: list[ZipMemberInfo] = []
    total_compressed = 0
    total_uncompressed = 0
    for info in infos:
        name = _safe_member_name(info.filename)
        file_size = int(info.file_size)
        compress_size = int(info.compress_size)
        total_compressed += compress_size
        total_uncompressed += file_size

        if total_compressed > limits.max_compressed_bytes:
            raise ZipInspectionError(
                f"archive compressed size exceeds limit: {total_compressed} > {limits.max_compressed_bytes}"
            )
        if total_uncompressed > limits.max_uncompressed_bytes:
            raise ZipInspectionError(
                f"archive uncompressed size exceeds limit: {total_uncompressed} > {limits.max_uncompressed_bytes}"
            )
        if not info.is_dir() and file_size > limits.max_member_bytes:
            raise ZipInspectionError(
                f"archive member exceeds limit: {name} is {file_size} bytes > {limits.max_member_bytes}"
            )
        if name.lower().endswith(".xml") and file_size > limits.max_xml_member_bytes:
            raise ZipInspectionError(
                f"XML member exceeds limit: {name} is {file_size} bytes > {limits.max_xml_member_bytes}"
            )
        if limits.max_compression_ratio is not None and file_size > 0:
            ratio = float("inf") if compress_size == 0 else file_size / max(compress_size, 1)
            if ratio > limits.max_compression_ratio:
                raise ZipInspectionError(
                    f"archive member compression ratio exceeds limit: {name} ratio {ratio:.1f} > {limits.max_compression_ratio:g}"
                )
        members.append(ZipMemberInfo(name=name, file_size=file_size, compress_size=compress_size, is_dir=info.is_dir()))

    return ZipInspectionReport(
        members=tuple(members),
        total_compressed_bytes=total_compressed,
        total_uncompressed_bytes=total_uncompressed,
    )


def ensure_zip_crc(zf: zipfile.ZipFile, _report: ZipInspectionReport | None = None) -> None:
    """Run stdlib CRC validation after callers have already inspected bounds."""

    bad = zf.testzip()
    if bad:
        raise ZipInspectionError(f"corrupt zip member: {bad}")


def read_zip_member(zf: zipfile.ZipFile, name: str, *, max_bytes: int) -> bytes:
    """Read one ZIP member with an explicit decompressed-byte ceiling."""

    try:
        info = zf.getinfo(name)
    except KeyError as exc:
        raise ZipInspectionError(f"missing archive member: {name}") from exc
    safe_name = _safe_member_name(info.filename)
    if int(info.file_size) > max_bytes:
        raise ZipInspectionError(f"archive member exceeds read limit: {safe_name} is {info.file_size} bytes > {max_bytes}")

    chunks: list[bytes] = []
    total = 0
    try:
        with zf.open(info, "r") as fh:
            while True:
                chunk = fh.read(min(_READ_CHUNK_BYTES, max_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise ZipInspectionError(f"archive member exceeds read limit: {safe_name} > {max_bytes} bytes")
    except zipfile.BadZipFile as exc:
        raise ZipInspectionError(f"corrupt zip member: {safe_name}") from exc
    except RuntimeError as exc:
        raise ZipInspectionError(f"cannot read zip member {safe_name}: {exc}") from exc
    return b"".join(chunks)


def existing_members(names: Iterable[str], report: ZipInspectionReport) -> list[str]:
    available = set(report.names)
    return [name for name in names if name in available]
