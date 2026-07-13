"""Stdlib-only dedicated-canary host identity and receipt primitives.

This module is intentionally importable by the root source-side release CLI
before a project virtual environment exists.  It performs only bounded GCE
metadata and root-controlled local identity reads.  It has no service,
filesystem-write, config, database, credential, or lifecycle authority.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import stat
import time
from pathlib import Path
from typing import Any, Callable, Mapping


FULL_CANARY_HOST_IDENTITY_SCHEMA = "muncho-full-canary-host-identity.v1"
DEDICATED_CANARY_PROJECT_ID = "adventico-ai-platform"
DEDICATED_CANARY_PROJECT_NUMBER = "39589465056"
DEDICATED_CANARY_ZONE = "europe-west3-a"
DEDICATED_CANARY_INSTANCE_NAME = "muncho-canary-v2-01"
DEDICATED_CANARY_INSTANCE_ID = "9153645328899914617"
DEDICATED_CANARY_SERVICE_ACCOUNT = (
    "muncho-canary-v2-runtime@adventico-ai-platform.iam.gserviceaccount.com"
)

_GCE_METADATA_PATHS = {
    "project_id": "/computeMetadata/v1/project/project-id",
    "project_number": "/computeMetadata/v1/project/numeric-project-id",
    "zone": "/computeMetadata/v1/instance/zone",
    "instance_name": "/computeMetadata/v1/instance/name",
    "instance_id": "/computeMetadata/v1/instance/id",
    "service_account_email": (
        "/computeMetadata/v1/instance/service-accounts/default/email"
    ),
}
_LOCAL_HOST_IDENTITY_PATHS = {
    "machine_id": Path("/etc/machine-id"),
    "hostname": Path("/proc/sys/kernel/hostname"),
    "boot_id": Path("/proc/sys/kernel/random/boot_id"),
}
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError) as exc:
        raise ValueError("host identity value is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _dedicated_canary_gce_identity() -> dict[str, str]:
    """Return the immutable-in-code single VM allow-list as fresh data."""

    return {
        "project_id": DEDICATED_CANARY_PROJECT_ID,
        "project_number": DEDICATED_CANARY_PROJECT_NUMBER,
        "zone": DEDICATED_CANARY_ZONE,
        "instance_name": DEDICATED_CANARY_INSTANCE_NAME,
        "instance_id": DEDICATED_CANARY_INSTANCE_ID,
        "service_account_email": DEDICATED_CANARY_SERVICE_ACCOUNT,
    }


def _bounded_identity_text(value: bytes | str, *, label: str) -> str:
    if isinstance(value, str):
        raw = value.encode("utf-8", errors="strict")
    elif isinstance(value, bytes):
        raw = value
    else:
        raise RuntimeError(f"{label} is not bounded text")
    if raw.endswith(b"\n"):
        raw = raw[:-1]
    if not raw or len(raw) > 1024 or raw.endswith(b"\n"):
        raise RuntimeError(f"{label} is empty or exceeds its bound")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{label} is not UTF-8") from exc
    if text != text.strip() or _CONTROL_RE.search(text) is not None:
        raise RuntimeError(f"{label} contains whitespace or control data")
    return text


def _read_gce_metadata_value(path: str) -> bytes:
    """Read one fixed GCE metadata leaf with a strict size and time bound."""

    if path not in _GCE_METADATA_PATHS.values():
        raise RuntimeError("GCE metadata path is not allow-listed")
    connection = http.client.HTTPConnection(
        "169.254.169.254",
        80,
        timeout=1.0,
    )
    try:
        connection.request(
            "GET",
            path,
            headers={
                "Host": "metadata.google.internal",
                "Metadata-Flavor": "Google",
            },
        )
        response = connection.getresponse()
        raw = response.read(1025)
        if (
            response.status != 200
            or response.getheader("Metadata-Flavor") != "Google"
            or len(raw) > 1024
        ):
            raise RuntimeError("GCE metadata response is not trusted and bounded")
        return raw
    except (OSError, http.client.HTTPException) as exc:
        raise RuntimeError("GCE metadata identity is unavailable") from exc
    finally:
        connection.close()


def _read_local_host_identity_value(name: str) -> bytes:
    """Read a fixed local host/boot identity without following a symlink."""

    try:
        path = _LOCAL_HOST_IDENTITY_PATHS[name]
    except KeyError as exc:
        raise RuntimeError("local host identity name is not allow-listed") from exc
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != 0
        or before.st_gid != 0
        or stat.S_IMODE(before.st_mode) & 0o022
    ):
        raise RuntimeError("local host identity source is not root-controlled")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        raw = os.read(descriptor, 1025)
        if len(raw) <= 1024:
            extra = os.read(descriptor, 1)
            if extra:
                raw += extra
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    reachable = path.lstat()

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_uid,
            item.st_gid,
        )

    if (
        len(raw) > 1024
        or identity(before) != identity(opened)
        or identity(before) != identity(after)
        or identity(before) != identity(reachable)
    ):
        raise RuntimeError("local host identity changed during bounded read")
    return raw


def _observe_dedicated_canary_host(
    *,
    metadata_reader: Callable[[str], bytes | str] | None = None,
    local_identity_reader: Callable[[str], bytes | str] | None = None,
) -> dict[str, str]:
    metadata_reader = metadata_reader or _read_gce_metadata_value
    local_identity_reader = local_identity_reader or _read_local_host_identity_value
    expected_gce = _dedicated_canary_gce_identity()
    observed_gce: dict[str, str] = {}
    for name, path in _GCE_METADATA_PATHS.items():
        value = _bounded_identity_text(
            metadata_reader(path),
            label=f"GCE metadata {name}",
        )
        if name == "zone":
            expected_raw_zone = (
                f"projects/{DEDICATED_CANARY_PROJECT_NUMBER}/zones/"
                f"{DEDICATED_CANARY_ZONE}"
            )
            if value != expected_raw_zone:
                raise RuntimeError("GCE metadata zone is not the dedicated canary")
            value = DEDICATED_CANARY_ZONE
        observed_gce[name] = value
    if observed_gce != expected_gce:
        raise RuntimeError("GCE identity is not the dedicated canary allow-list")

    machine_id = _bounded_identity_text(
        local_identity_reader("machine_id"),
        label="machine-id",
    )
    hostname = _bounded_identity_text(
        local_identity_reader("hostname"),
        label="hostname",
    )
    boot_id = _bounded_identity_text(
        local_identity_reader("boot_id"),
        label="boot-id",
    )
    if re.fullmatch(r"[0-9a-f]{32}", machine_id) is None:
        raise RuntimeError("machine-id format is invalid")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,252}", hostname) is None:
        raise RuntimeError("hostname format is invalid")
    if (
        re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            boot_id,
        )
        is None
    ):
        raise RuntimeError("boot-id format is invalid")
    machine_id_sha256 = _sha256_bytes(machine_id.encode("utf-8"))
    hostname_sha256 = _sha256_bytes(hostname.encode("utf-8"))
    return {
        **observed_gce,
        "gce_identity_sha256": _sha256_json(observed_gce),
        "machine_id_sha256": machine_id_sha256,
        "hostname_sha256": hostname_sha256,
        "host_identity_sha256": _sha256_json({
            "machine_id_sha256": machine_id_sha256,
            "hostname_sha256": hostname_sha256,
        }),
        "boot_id_sha256": _sha256_bytes(boot_id.encode("utf-8")),
    }


def collect_dedicated_canary_host_identity_receipt(
    *,
    metadata_reader: Callable[[str], bytes | str] | None = None,
    local_identity_reader: Callable[[str], bytes | str] | None = None,
    observed_at_unix: int | None = None,
) -> Mapping[str, Any]:
    """Collect receipt bytes for a root bootstrap to seal as an artifact."""

    observed_at_unix = (
        int(time.time()) if observed_at_unix is None else observed_at_unix
    )
    if type(observed_at_unix) is not int or observed_at_unix < 0:
        raise ValueError("host identity observation time is invalid")
    unsigned = {
        "schema": FULL_CANARY_HOST_IDENTITY_SCHEMA,
        "collector_authority": "trusted_root_read_only_host_collector",
        **_observe_dedicated_canary_host(
            metadata_reader=metadata_reader,
            local_identity_reader=local_identity_reader,
        ),
        "observed_at_unix": observed_at_unix,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


__all__ = [
    "DEDICATED_CANARY_INSTANCE_ID",
    "DEDICATED_CANARY_INSTANCE_NAME",
    "DEDICATED_CANARY_PROJECT_ID",
    "DEDICATED_CANARY_PROJECT_NUMBER",
    "DEDICATED_CANARY_SERVICE_ACCOUNT",
    "DEDICATED_CANARY_ZONE",
    "FULL_CANARY_HOST_IDENTITY_SCHEMA",
    "collect_dedicated_canary_host_identity_receipt",
]
