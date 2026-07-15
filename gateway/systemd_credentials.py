"""Strict, TOCTOU-safe reads for service-private systemd credentials.

systemd may expose a credential as either service-owned or root-owned with a
read ACL for the service.  In both cases the credential mount keeps group
ownership at root.  This module recognizes only an exact
``/run/credentials/<unit>/<name>`` binding and validates the mount/file
provenance before and after a real bounded read.

It is a mechanical credential boundary.  It performs no lookup, fallback,
secret discovery, task classification, or policy selection.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping


SYSTEMD_CREDENTIAL_ROOT = Path("/run/credentials")

DISCORD_EDGE_UNIT = "muncho-discord-egress.service"
DISCORD_TOKEN_CREDENTIAL = "discord-bot-token"
DISCORD_PRIVATE_KEY_CREDENTIAL = "discord-edge-receipt-private-key"
MAC_OPS_UNIT = "muncho-mac-ops-edge.service"
MAC_OPS_GITLAB_CREDENTIAL = "mac-ops-gitlab-env"
GATEWAY_API_UNIT = "hermes-cloud-gateway.service"
GATEWAY_API_BEARER_CREDENTIAL = "api-server.key"
GATEWAY_API_APPROVAL_CREDENTIAL = "api-approval-passkey"
GATEWAY_API_BEARER_VERIFIER_CREDENTIAL = "api-server-bearer-sha256"
GATEWAY_API_APPROVAL_VERIFIER_CREDENTIAL = "api-approval-passkey-scrypt"

_UNIT = re.compile(r"^[A-Za-z0-9_.@:-]+\.service$")
_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_MAX_SUPPORTED_BYTES = 1024 * 1024

# Stable aliases make race/ownership behavior testable without replacing the
# process-wide ``os`` module.  Production always uses the real syscalls.
_lstat = os.lstat
_fstat = os.fstat
_open = os.open
_read = os.read
_close = os.close


class SystemdCredentialError(RuntimeError):
    """One stable, secret-free systemd credential failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _binding(unit: Any, name: Any) -> tuple[str, str, Path, Path]:
    if not isinstance(unit, str) or _UNIT.fullmatch(unit) is None:
        raise SystemdCredentialError("systemd_credential_unit_invalid")
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise SystemdCredentialError("systemd_credential_name_invalid")
    root = Path(SYSTEMD_CREDENTIAL_ROOT)
    if (
        not root.is_absolute()
        or root != Path(os.path.normpath(os.fspath(root)))
        or ".." in root.parts
    ):
        raise SystemdCredentialError("systemd_credential_root_invalid")
    directory = root / unit
    return unit, name, directory, directory / name


def _canonical_path(value: Any) -> Path:
    try:
        path = Path(value)
    except TypeError as exc:
        raise SystemdCredentialError("systemd_credential_path_invalid") from exc
    if (
        not path.is_absolute()
        or path != Path(os.path.normpath(os.fspath(path)))
        or ".." in path.parts
    ):
        raise SystemdCredentialError("systemd_credential_path_invalid")
    return path


def is_expected_systemd_credential(
    path: str | os.PathLike[str],
    *,
    unit: str,
    name: str,
) -> bool:
    """Return whether ``path`` is the exact binding; reject namespace drift.

    A path outside the systemd credential root is a legacy fixed-file path and
    returns ``False``.  Any path *inside* that namespace must match the exact
    reviewed unit/name pair or it fails closed rather than falling back to a
    legacy ownership rule.
    """

    _unit, _name, _directory, expected = _binding(unit, name)
    candidate = _canonical_path(path)
    root = Path(SYSTEMD_CREDENTIAL_ROOT)
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    if candidate != expected:
        raise SystemdCredentialError("systemd_credential_binding_invalid")
    return True


def _directory_identity(item: os.stat_result) -> tuple[int, ...]:
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_uid,
        item.st_gid,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _file_identity(item: os.stat_result) -> tuple[int, ...]:
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_uid,
        item.st_gid,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _validate_directory(item: os.stat_result, *, service_uid: int) -> None:
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISDIR(item.st_mode)
        or item.st_uid not in {0, service_uid}
        or item.st_gid != 0
        or stat.S_IMODE(item.st_mode) != 0o500
    ):
        raise SystemdCredentialError(
            "systemd_credential_directory_provenance_invalid"
        )


def _validate_file(
    item: os.stat_result,
    *,
    service_uid: int,
    maximum: int,
) -> None:
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISREG(item.st_mode)
        or item.st_nlink != 1
        or item.st_uid not in {0, service_uid}
        or item.st_gid != 0
        or stat.S_IMODE(item.st_mode) != 0o400
        or not 0 < item.st_size <= maximum
    ):
        raise SystemdCredentialError("systemd_credential_file_provenance_invalid")


def _read_with_provenance(
    path: str | os.PathLike[str],
    *,
    unit: str,
    name: str,
    service_uid: int,
    maximum: int,
    credentials_directory: str | os.PathLike[str] | None,
) -> tuple[bytes, Mapping[str, Any]]:
    if type(service_uid) is not int or not 1 <= service_uid < (1 << 31):
        raise SystemdCredentialError("systemd_credential_service_uid_invalid")
    if type(maximum) is not int or not 1 <= maximum <= _MAX_SUPPORTED_BYTES:
        raise SystemdCredentialError("systemd_credential_size_bound_invalid")
    _unit, _name, directory, expected = _binding(unit, name)
    candidate = _canonical_path(path)
    if not is_expected_systemd_credential(candidate, unit=unit, name=name):
        raise SystemdCredentialError("systemd_credential_binding_invalid")
    if credentials_directory is not None:
        supplied_directory = _canonical_path(credentials_directory)
        if supplied_directory != directory:
            raise SystemdCredentialError("systemd_credential_binding_invalid")

    try:
        # ``resolve`` proves every traversed component, including the unit
        # directory and final file, is symlink-free and reaches the exact path.
        if directory.resolve(strict=True) != directory:
            raise SystemdCredentialError(
                "systemd_credential_directory_provenance_invalid"
            )
        if candidate.resolve(strict=True) != candidate:
            raise SystemdCredentialError(
                "systemd_credential_file_provenance_invalid"
            )
        directory_before = _lstat(directory)
        file_before = _lstat(candidate)
    except SystemdCredentialError:
        raise
    except OSError as exc:
        raise SystemdCredentialError("systemd_credential_unavailable") from exc
    _validate_directory(directory_before, service_uid=service_uid)
    _validate_file(file_before, service_uid=service_uid, maximum=maximum)

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = _open(candidate, flags)
        opened = _fstat(descriptor)
        _validate_file(opened, service_uid=service_uid, maximum=maximum)
        if _file_identity(opened) != _file_identity(file_before):
            raise SystemdCredentialError("systemd_credential_changed")
        chunks: list[bytes] = []
        total = 0
        while total <= maximum:
            chunk = _read(descriptor, min(64 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        payload = b"".join(chunks)
        file_after = _fstat(descriptor)
        reachable_after = _lstat(candidate)
        directory_after = _lstat(directory)
    except SystemdCredentialError:
        raise
    except OSError as exc:
        raise SystemdCredentialError("systemd_credential_read_failed") from exc
    finally:
        if descriptor is not None:
            _close(descriptor)

    _validate_file(file_after, service_uid=service_uid, maximum=maximum)
    _validate_file(reachable_after, service_uid=service_uid, maximum=maximum)
    _validate_directory(directory_after, service_uid=service_uid)
    if (
        total > maximum
        or total != opened.st_size
        or _file_identity(opened) != _file_identity(file_after)
        or _file_identity(opened) != _file_identity(reachable_after)
        or _directory_identity(directory_before)
        != _directory_identity(directory_after)
    ):
        raise SystemdCredentialError("systemd_credential_changed")
    return payload, {
        "path": str(candidate),
        "device": int(opened.st_dev),
        "inode": int(opened.st_ino),
        "uid": int(opened.st_uid),
        "gid": int(opened.st_gid),
        "mode": "0400",
        "size": int(opened.st_size),
    }


def read_systemd_credential(
    path: str | os.PathLike[str],
    *,
    unit: str,
    name: str,
    service_uid: int,
    maximum: int,
    credentials_directory: str | os.PathLike[str] | None = None,
) -> bytes:
    """Read an exact systemd credential after stable provenance validation."""

    payload, _provenance = _read_with_provenance(
        path,
        unit=unit,
        name=name,
        service_uid=service_uid,
        maximum=maximum,
        credentials_directory=credentials_directory,
    )
    return payload


def systemd_credential_provenance(
    path: str | os.PathLike[str],
    *,
    unit: str,
    name: str,
    service_uid: int,
    maximum: int,
    credentials_directory: str | os.PathLike[str] | None = None,
) -> Mapping[str, Any]:
    """Return secret-free provenance after a real validated credential read."""

    _payload, provenance = _read_with_provenance(
        path,
        unit=unit,
        name=name,
        service_uid=service_uid,
        maximum=maximum,
        credentials_directory=credentials_directory,
    )
    return provenance


__all__ = [
    "DISCORD_EDGE_UNIT",
    "DISCORD_PRIVATE_KEY_CREDENTIAL",
    "DISCORD_TOKEN_CREDENTIAL",
    "GATEWAY_API_APPROVAL_CREDENTIAL",
    "GATEWAY_API_BEARER_CREDENTIAL",
    "GATEWAY_API_UNIT",
    "MAC_OPS_GITLAB_CREDENTIAL",
    "MAC_OPS_UNIT",
    "SYSTEMD_CREDENTIAL_ROOT",
    "SystemdCredentialError",
    "is_expected_systemd_credential",
    "read_systemd_credential",
    "systemd_credential_provenance",
]
