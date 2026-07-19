#!/usr/bin/env python3
"""Read-only, host-attested migration of Muncho's enrolled v1 passkey.

The release-pinned source is streamed over the existing pinned production IAP
transport and runs as root on the one production VM.  The remote role accepts
only a release revision; every file, unit, identity, and command is fixed here.
It reads public WebAuthn credential material, never writes production state,
and emits one fresh canonical observation.

On the owner Mac the observation is bound to the exact IAP/OS Login authority
and signed by the release-specific host observation key.  The resulting
envelope is the exact format consumed by the offline owner-gate bootstrap.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


OBSERVATION_SCHEMA = "muncho-owner-gate-v1-credential-observation.v1"
SOURCE_RECEIPT_SCHEMA = "muncho-owner-gate-v1-credential-source-receipt.v1"
MIGRATION_SCHEMA = "muncho-owner-gate-host-attested-credential-migration.v1"
FAILURE_SCHEMA = "muncho-owner-gate-v1-credential-observation-failure.v1"

PROJECT = "adventico-ai-platform"
ZONE = "europe-west3-a"
VM_NAME = "ai-platform-runtime-01"
INSTANCE_ID = "1094477181810932795"
OWNER_DISCORD_USER_ID = "1279454038731264061"
REMOTE_PYTHON = "/opt/adventico-ai-platform/hermes-agent/.venv/bin/python"
SOURCE_SERVICE = Path(
    "/opt/adventico-ai-platform/hermes-home/services/"
    "passkey-stepup/muncho_passkey_service.py"
)
CREDENTIALS_FILE = Path(
    "/opt/adventico-ai-platform/hermes-home/security/"
    "passkey_stepup/credentials.json"
)
UNIT = "muncho-passkey-stepup.service"
UNIT_FRAGMENT = Path("/etc/systemd/system/muncho-passkey-stepup.service")
SYSTEMCTL = "/usr/bin/systemctl"
SOURCE_UID = 999
SOURCE_GID = 994
SOURCE_MODE = 0o600
CREDENTIALS_UID = 999
CREDENTIALS_GID = 994
CREDENTIALS_MODE = 0o600
EXPECTED_SOURCE_SERVICE_SHA256 = (
    "da8fc7823f378b77791c291a8c949d8c7e59d872cedb9e4a43501ff41200b9ff"
)
EXPECTED_CREDENTIAL_ID_SHA256 = (
    "63bbfca0778101d21dddf2b53cc774460565042391b918eb2d1c87b9d6d19860"
)
EXPECTED_PUBLIC_KEY_SHA256 = (
    "478c0bd2ee54f733dbb63acd329ad35188a7f091f9c6bdc4b6e64e7d59d5db89"
)
EXPECTED_USER_HANDLE_SHA256 = (
    "a72512de5fcd7fa3e679fcca570c9b4db6ff1e403b6329586ddad90c093ad983"
)
FRESHNESS_SECONDS = 300
MAX_SIGNING_DELAY_SECONDS = 120
MAX_SOURCE_BYTES = 512 * 1024
MAX_CREDENTIAL_BYTES = 64 * 1024
MAX_REMOTE_SOURCE_BYTES = 1024 * 1024
MAX_REMOTE_OUTPUT_BYTES = 256 * 1024
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOP_FIELDS = frozenset({"schema", "credentials"})
_ROW_FIELDS = frozenset({
    "schema",
    "discord_user_id",
    "credential_id",
    "credential_id_hash",
    "public_key",
    "sign_count",
    "credential_device_type",
    "credential_backed_up",
    "aaguid",
    "enabled",
    "label",
    "created_at",
    "last_used_at",
})


def _effective_uid() -> int:
    """Return the POSIX effective uid, or a non-root sentinel."""

    getter = getattr(os, "geteuid", None)
    return int(getter()) if callable(getter) else -1
_OBSERVATION_FIELDS = frozenset({
    "schema",
    "release_revision",
    "target",
    "source_service",
    "credentials_file",
    "service",
    "credential_public_material",
    "collected_at_unix",
    "completed_at_unix",
    "fresh_through_unix",
    "collector_authority",
    "caller_selected_input_accepted",
    "production_mutation_performed",
    "secret_material_recorded",
    "secret_digest_recorded",
    "report_sha256",
})
_FILE_EVIDENCE_FIELDS = frozenset({
    "path",
    "uid",
    "gid",
    "mode",
    "size",
    "stable_nofollow_read",
})
_SOURCE_EVIDENCE_FIELDS = _FILE_EVIDENCE_FIELDS | {"sha256"}
_SERVICE_FIELDS = frozenset({
    "unit",
    "load_state",
    "active_state",
    "sub_state",
    "unit_file_state",
    "fragment_path",
    "read_only_systemctl_show",
})
_PUBLIC_MATERIAL_FIELDS = frozenset({
    "owner_discord_user_id",
    "credential_id_b64url",
    "credential_id_sha256",
    "public_key_cose_b64url",
    "public_key_cose_sha256",
    "expected_user_handle_b64url",
    "expected_user_handle_sha256",
    "initial_sign_count",
    "initial_credential_backed_up",
    "enabled",
    "credential_device_type",
    "registry_schema",
    "record_schema",
})
_TRANSPORT_FIELDS = frozenset({
    "kind",
    "project",
    "zone",
    "vm",
    "instance_id",
    "known_hosts_file_sha256",
    "collector_source_sha256",
    "instance_authorization_sha256",
    "project_authorization_sha256",
    "oslogin_authorization_sha256",
})
_SOURCE_RECEIPT_FIELDS = frozenset({
    "schema",
    "release_revision",
    "observation",
    "observation_report_sha256",
    "transport_authority",
    "host_collector_public_key_id",
    "signed_at_unix",
    "production_mutation_performed",
    "secret_material_recorded",
    "secret_digest_recorded",
    "receipt_sha256",
})


class V1CredentialMigrationError(RuntimeError):
    """Stable, secret-free credential migration failure."""


def _error(code: str, exc: BaseException | None = None) -> None:
    del exc
    raise V1CredentialMigrationError(code) from None


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        _error("owner_gate_v1_credential_json_invalid", exc)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256(_canonical(value))


def _b64url_decode(value: Any, *, maximum: int) -> bytes:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum * 2
        or "=" in value
    ):
        _error("owner_gate_v1_credential_base64_invalid")
    try:
        raw = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        _error("owner_gate_v1_credential_base64_invalid", exc)
    if (
        not 0 < len(raw) <= maximum
        or base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != value
    ):
        _error("owner_gate_v1_credential_base64_invalid")
    return raw


@dataclass(frozen=True)
class _StableFile:
    raw: bytes
    evidence: Mapping[str, Any]


def _stable_file(
    path: Path,
    *,
    uid: int,
    gid: int,
    mode: int,
    maximum: int,
    include_sha256: bool,
) -> _StableFile:
    descriptor: int | None = None
    if not path.is_absolute() or ".." in path.parts:
        _error("owner_gate_v1_credential_file_invalid")
    try:
        before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_nlink != 1
            or opened.st_uid != uid
            or opened.st_gid != gid
            or stat.S_IMODE(opened.st_mode) != mode
            or not 0 < opened.st_size <= maximum
        ):
            _error("owner_gate_v1_credential_file_invalid")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                _error("owner_gate_v1_credential_file_changed")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        final = path.lstat()
        identity = (
            opened.st_mode,
            opened.st_uid,
            opened.st_gid,
            opened.st_dev,
            opened.st_ino,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if identity != (
            after.st_mode,
            after.st_uid,
            after.st_gid,
            after.st_dev,
            after.st_ino,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or (final.st_dev, final.st_ino) != (opened.st_dev, opened.st_ino):
            _error("owner_gate_v1_credential_file_changed")
        raw = b"".join(chunks)
        evidence: dict[str, Any] = {
            "path": str(path),
            "uid": uid,
            "gid": gid,
            "mode": f"{mode:04o}",
            "size": opened.st_size,
            "stable_nofollow_read": True,
        }
        if include_sha256:
            evidence["sha256"] = _sha256(raw)
        return _StableFile(raw=raw, evidence=evidence)
    except V1CredentialMigrationError:
        raise
    except OSError as exc:
        _error("owner_gate_v1_credential_file_invalid", exc)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                _error("owner_gate_v1_credential_file_invalid", exc)


def _service_projection() -> Mapping[str, Any]:
    argv = (
        SYSTEMCTL,
        "show",
        "--no-pager",
        "--property=LoadState",
        "--property=ActiveState",
        "--property=SubState",
        "--property=UnitFileState",
        "--property=FragmentPath",
        UNIT,
    )
    try:
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _error("owner_gate_v1_credential_service_invalid", exc)
    if (
        completed.returncode != 0
        or completed.stderr
        or not completed.stdout.endswith(b"\n")
        or len(completed.stdout) > 64 * 1024
    ):
        _error("owner_gate_v1_credential_service_invalid")
    values: dict[str, str] = {}
    try:
        for raw_line in completed.stdout.decode("utf-8", errors="strict").splitlines():
            key, separator, value = raw_line.partition("=")
            if not separator or not key or key in values:
                _error("owner_gate_v1_credential_service_invalid")
            values[key] = value
    except UnicodeError as exc:
        _error("owner_gate_v1_credential_service_invalid", exc)
    expected = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "UnitFileState": "enabled",
        "FragmentPath": str(UNIT_FRAGMENT),
    }
    if values != expected:
        _error("owner_gate_v1_credential_service_invalid")
    return {
        "unit": UNIT,
        "load_state": values["LoadState"],
        "active_state": values["ActiveState"],
        "sub_state": values["SubState"],
        "unit_file_state": values["UnitFileState"],
        "fragment_path": values["FragmentPath"],
        "read_only_systemctl_show": True,
    }


def _public_credential(raw: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _error("owner_gate_v1_credential_registry_invalid", exc)
    if (
        not isinstance(value, Mapping)
        or set(value) != _TOP_FIELDS
        or value.get("schema") != "muncho.passkey.credentials.v1"
        or not isinstance(value.get("credentials"), list)
        or len(value["credentials"]) != 1
    ):
        _error("owner_gate_v1_credential_registry_invalid")
    row = value["credentials"][0]
    if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
        _error("owner_gate_v1_credential_registry_invalid")
    credential_id = _b64url_decode(row.get("credential_id"), maximum=4096)
    public_key = _b64url_decode(row.get("public_key"), maximum=4096)
    user_handle = OWNER_DISCORD_USER_ID.encode("ascii")
    credential_digest = _sha256(credential_id)
    public_digest = _sha256(public_key)
    user_handle_digest = _sha256(user_handle)
    if (
        row.get("schema") != "muncho.passkey.credential.v1"
        or row.get("discord_user_id") != OWNER_DISCORD_USER_ID
        or row.get("credential_id_hash") != credential_digest
        or credential_digest != EXPECTED_CREDENTIAL_ID_SHA256
        or public_digest != EXPECTED_PUBLIC_KEY_SHA256
        or user_handle_digest != EXPECTED_USER_HANDLE_SHA256
        or row.get("sign_count") != 0
        or row.get("credential_backed_up") is not True
        or row.get("credential_device_type") != "multi_device"
        or row.get("enabled") is not True
        or any(
            not isinstance(row.get(field), str) or not row[field]
            for field in ("aaguid", "label", "created_at", "last_used_at")
        )
    ):
        _error("owner_gate_v1_credential_registry_invalid")
    return {
        "owner_discord_user_id": OWNER_DISCORD_USER_ID,
        "credential_id_b64url": row["credential_id"],
        "credential_id_sha256": credential_digest,
        "public_key_cose_b64url": row["public_key"],
        "public_key_cose_sha256": public_digest,
        "expected_user_handle_b64url": base64.urlsafe_b64encode(user_handle)
        .rstrip(b"=")
        .decode("ascii"),
        "expected_user_handle_sha256": user_handle_digest,
        "initial_sign_count": 0,
        "initial_credential_backed_up": True,
        "enabled": True,
        "credential_device_type": "multi_device",
        "registry_schema": value["schema"],
        "record_schema": row["schema"],
    }


def validate_observation(
    value: Mapping[str, Any],
    *,
    release_revision: str,
    now_unix: int,
) -> Mapping[str, Any]:
    if (
        _REVISION.fullmatch(release_revision or "") is None
        or type(now_unix) is not int
        or not isinstance(value, Mapping)
        or set(value) != _OBSERVATION_FIELDS
    ):
        _error("owner_gate_v1_credential_observation_invalid")
    unsigned = {key: item for key, item in value.items() if key != "report_sha256"}
    target = value.get("target")
    source = value.get("source_service")
    credentials_file = value.get("credentials_file")
    service = value.get("service")
    credential = value.get("credential_public_material")
    collected = value.get("collected_at_unix")
    completed = value.get("completed_at_unix")
    fresh = value.get("fresh_through_unix")
    if (
        value.get("schema") != OBSERVATION_SCHEMA
        or value.get("release_revision") != release_revision
        or target
        != {
            "project": PROJECT,
            "zone": ZONE,
            "vm": VM_NAME,
            "instance_id": INSTANCE_ID,
        }
        or not isinstance(source, Mapping)
        or set(source) != _SOURCE_EVIDENCE_FIELDS
        or source.get("path") != str(SOURCE_SERVICE)
        or source.get("uid") != SOURCE_UID
        or source.get("gid") != SOURCE_GID
        or source.get("mode") != f"{SOURCE_MODE:04o}"
        or source.get("stable_nofollow_read") is not True
        or source.get("sha256") != EXPECTED_SOURCE_SERVICE_SHA256
        or not isinstance(credentials_file, Mapping)
        or set(credentials_file) != _FILE_EVIDENCE_FIELDS
        or credentials_file.get("path") != str(CREDENTIALS_FILE)
        or credentials_file.get("uid") != CREDENTIALS_UID
        or credentials_file.get("gid") != CREDENTIALS_GID
        or credentials_file.get("mode") != f"{CREDENTIALS_MODE:04o}"
        or credentials_file.get("stable_nofollow_read") is not True
        or "sha256" in credentials_file
        or not isinstance(service, Mapping)
        or set(service) != _SERVICE_FIELDS
        or service.get("unit") != UNIT
        or service.get("load_state") != "loaded"
        or service.get("active_state") != "active"
        or service.get("sub_state") != "running"
        or service.get("unit_file_state") != "enabled"
        or service.get("fragment_path") != str(UNIT_FRAGMENT)
        or service.get("read_only_systemctl_show") is not True
        or not isinstance(credential, Mapping)
        or set(credential) != _PUBLIC_MATERIAL_FIELDS
        or credential.get("owner_discord_user_id") != OWNER_DISCORD_USER_ID
        or credential.get("credential_id_sha256")
        != EXPECTED_CREDENTIAL_ID_SHA256
        or credential.get("public_key_cose_sha256")
        != EXPECTED_PUBLIC_KEY_SHA256
        or credential.get("expected_user_handle_sha256")
        != EXPECTED_USER_HANDLE_SHA256
        or _sha256(
            _b64url_decode(credential.get("credential_id_b64url"), maximum=4096)
        )
        != EXPECTED_CREDENTIAL_ID_SHA256
        or _sha256(
            _b64url_decode(
                credential.get("public_key_cose_b64url"), maximum=4096
            )
        )
        != EXPECTED_PUBLIC_KEY_SHA256
        or _sha256(
            _b64url_decode(
                credential.get("expected_user_handle_b64url"), maximum=256
            )
        )
        != EXPECTED_USER_HANDLE_SHA256
        or credential.get("initial_sign_count") != 0
        or credential.get("initial_credential_backed_up") is not True
        or type(collected) is not int
        or type(completed) is not int
        or type(fresh) is not int
        or not 0 < collected <= completed <= now_unix <= fresh
        or fresh != completed + FRESHNESS_SECONDS
        or value.get("collector_authority")
        != "production_root_read_only_fixed_projection"
        or value.get("caller_selected_input_accepted") is not False
        or value.get("production_mutation_performed") is not False
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or value.get("report_sha256") != _sha256_json(unsigned)
    ):
        _error("owner_gate_v1_credential_observation_invalid")
    return dict(value)


def collect_observation(
    *,
    release_revision: str,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    if _REVISION.fullmatch(release_revision or "") is None or _effective_uid() != 0:
        _error("owner_gate_v1_credential_collector_identity_invalid")
    collected = int(time.time()) if now_unix is None else now_unix
    source = _stable_file(
        SOURCE_SERVICE,
        uid=SOURCE_UID,
        gid=SOURCE_GID,
        mode=SOURCE_MODE,
        maximum=MAX_SOURCE_BYTES,
        include_sha256=True,
    )
    if source.evidence.get("sha256") != EXPECTED_SOURCE_SERVICE_SHA256:
        _error("owner_gate_v1_credential_source_invalid")
    credentials = _stable_file(
        CREDENTIALS_FILE,
        uid=CREDENTIALS_UID,
        gid=CREDENTIALS_GID,
        mode=CREDENTIALS_MODE,
        maximum=MAX_CREDENTIAL_BYTES,
        include_sha256=False,
    )
    public_credential = _public_credential(credentials.raw)
    service = _service_projection()
    # Re-read both files after systemd observation to close the race window.
    final_source = _stable_file(
        SOURCE_SERVICE,
        uid=SOURCE_UID,
        gid=SOURCE_GID,
        mode=SOURCE_MODE,
        maximum=MAX_SOURCE_BYTES,
        include_sha256=True,
    )
    final_credentials = _stable_file(
        CREDENTIALS_FILE,
        uid=CREDENTIALS_UID,
        gid=CREDENTIALS_GID,
        mode=CREDENTIALS_MODE,
        maximum=MAX_CREDENTIAL_BYTES,
        include_sha256=False,
    )
    if (
        source != final_source
        or credentials != final_credentials
        or public_credential != _public_credential(final_credentials.raw)
    ):
        _error("owner_gate_v1_credential_source_changed")
    completed = int(time.time()) if now_unix is None else now_unix
    unsigned = {
        "schema": OBSERVATION_SCHEMA,
        "release_revision": release_revision,
        "target": {
            "project": PROJECT,
            "zone": ZONE,
            "vm": VM_NAME,
            "instance_id": INSTANCE_ID,
        },
        "source_service": dict(source.evidence),
        "credentials_file": dict(credentials.evidence),
        "service": dict(service),
        "credential_public_material": dict(public_credential),
        "collected_at_unix": collected,
        "completed_at_unix": completed,
        "fresh_through_unix": completed + FRESHNESS_SECONDS,
        "collector_authority": "production_root_read_only_fixed_projection",
        "caller_selected_input_accepted": False,
        "production_mutation_performed": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    value = {**unsigned, "report_sha256": _sha256_json(unsigned)}
    return validate_observation(
        value,
        release_revision=release_revision,
        now_unix=completed,
    )


def _collector_source(
    release_revision: str,
    *,
    trusted_runtime: Any,
) -> tuple[bytes, str]:
    if _REVISION.fullmatch(release_revision or "") is None:
        _error("owner_gate_v1_credential_source_invalid")
    from scripts.canary import full_canary_owner_launcher as launcher

    if type(trusted_runtime) is not launcher.TrustedGcloudExecutable:
        _error("owner_gate_v1_credential_source_invalid")
    relative = Path("scripts/canary/owner_gate_v1_credential_migration.py")
    try:
        source_root, _site_root = (
            launcher.require_trusted_owner_support_activation(
                trusted_runtime,
                release_sha=release_revision,
            )
        )
        manifest_before = trusted_runtime.sealed_owner_support_manifest(
            expected_release_sha=release_revision,
        )
        path = Path(__file__)
        expected_path = Path(source_root) / relative
        if (
            not path.is_absolute()
            or path != expected_path
            or Path(os.path.realpath(path, strict=True)) != path
        ):
            _error("owner_gate_v1_credential_source_invalid")
        before = path.lstat()
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            opened_before = os.fstat(descriptor)
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(
                    descriptor,
                    min(64 * 1024, MAX_REMOTE_SOURCE_BYTES + 1 - total),
                )
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_REMOTE_SOURCE_BYTES:
                    _error("owner_gate_v1_credential_source_invalid")
            opened_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        raw = b"".join(chunks)
        after = path.lstat()
        manifest_after = trusted_runtime.sealed_owner_support_manifest(
            expected_release_sha=release_revision,
        )
        launcher.require_trusted_owner_support_activation(
            trusted_runtime,
            release_sha=release_revision,
        )
    except (OSError, launcher.OwnerLauncherError) as exc:
        _error("owner_gate_v1_credential_source_invalid", exc)
    expected = _sha256(raw)
    metadata = (
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before.st_dev,
        before.st_ino,
        before.st_nlink,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_size,
    )
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o400
        or before.st_uid != os.getuid()  # windows-footgun: ok
        or before.st_nlink != 1
        or not 0 < len(raw) <= MAX_REMOTE_SOURCE_BYTES
        or len(raw) != before.st_size
        or metadata
        != (
            opened_before.st_mode,
            opened_before.st_uid,
            opened_before.st_gid,
            opened_before.st_dev,
            opened_before.st_ino,
            opened_before.st_nlink,
            opened_before.st_mtime_ns,
            opened_before.st_ctime_ns,
            opened_before.st_size,
        )
        or metadata
        != (
            opened_after.st_mode,
            opened_after.st_uid,
            opened_after.st_gid,
            opened_after.st_dev,
            opened_after.st_ino,
            opened_after.st_nlink,
            opened_after.st_mtime_ns,
            opened_after.st_ctime_ns,
            opened_after.st_size,
        )
        or metadata
        != (
            after.st_mode,
            after.st_uid,
            after.st_gid,
            after.st_dev,
            after.st_ino,
            after.st_nlink,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_size,
        )
        or manifest_after != manifest_before
    ):
        _error("owner_gate_v1_credential_source_invalid")
    return raw, expected


class V1CredentialMigrationTransport:
    """Pinned owner IAP transport for the sole fixed read-only collector."""

    def __init__(self, transport: Any | None = None, *, revision: str | None = None) -> None:
        trusted_runtime: Any | None = None
        if transport is None:
            from scripts.canary import full_canary_owner_launcher as launcher
            from scripts.canary import production_cutover_owner_launcher as cutover

            if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
                _error("owner_gate_v1_credential_transport_invalid")
            trusted = launcher.require_trusted_owner_runtime(revision)
            trusted_runtime = trusted
            configuration = launcher.PinnedGcloudConfiguration()
            identity = launcher.GcloudOwnerAccessToken(
                gcloud_executable=trusted,
                gcloud_configuration=configuration,
            )
            identity.account_for_read_only_preflight()
            transport = cutover.ProductionCutoverTransport(
                identity,
                gcloud_executable=trusted,
                gcloud_configuration=configuration,
            )
        required = (
            "_owner_identity",
            "_authorization_snapshot",
            "_run_remote_input",
            "_fixed_remote_environment",
            "_known_hosts",
        )
        if any(not hasattr(transport, name) for name in required):
            _error("owner_gate_v1_credential_transport_invalid")
        self._transport = transport
        self._trusted_runtime = trusted_runtime

    def observe(
        self,
        *,
        release_revision: str,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        source, source_sha256 = _collector_source(
            release_revision,
            trusted_runtime=self._trusted_runtime,
        )
        inner = self._transport
        account = inner._owner_identity.account_for_read_only_preflight()
        inner._owner_identity.require_stable()
        authorization = inner._authorization_snapshot(account)
        if (
            not isinstance(authorization, tuple)
            or len(authorization) != 3
            or any(
                not isinstance(item, str) or _SHA256.fullmatch(item) is None
                for item in authorization
            )
        ):
            _error("owner_gate_v1_credential_transport_invalid")
        command = (
            *inner._fixed_remote_environment(chdir="/"),
            REMOTE_PYTHON,
            "-B",
            "-I",
            "-",
            "collect",
            "--release-revision",
            release_revision,
        )
        try:
            completed = inner._run_remote_input(
                command,
                account=account,
                input_bytes=source,
                timeout_seconds=120,
                maximum_input_bytes=MAX_REMOTE_SOURCE_BYTES,
                maximum_output_bytes=MAX_REMOTE_OUTPUT_BYTES,
            )
        except Exception as exc:
            _error("owner_gate_v1_credential_transport_failed", exc)
        inner._owner_identity.require_stable()
        if inner._authorization_snapshot(account) != authorization:
            _error("owner_gate_v1_credential_transport_changed")
        raw = completed.stdout
        if (
            not isinstance(raw, bytes)
            or not raw.endswith(b"\n")
            or b"\n" in raw[:-1]
        ):
            _error("owner_gate_v1_credential_transport_output_invalid")
        try:
            observation = json.loads(raw[:-1].decode("ascii", errors="strict"))
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            _error("owner_gate_v1_credential_transport_output_invalid", exc)
        if _canonical(observation) != raw[:-1]:
            _error("owner_gate_v1_credential_transport_output_invalid")
        checked = validate_observation(
            observation,
            release_revision=release_revision,
            now_unix=int(time.time()),
        )
        from scripts.canary import owner_gate_production_ingress_observation as ingress

        known_hosts_raw = ingress._stable_owner_file(
            Path(inner._known_hosts.absolute_path()),
            maximum=256 * 1024,
        )
        authority = {
            "kind": "pinned_owner_gcloud_iap_ssh_read_only",
            "project": PROJECT,
            "zone": ZONE,
            "vm": VM_NAME,
            "instance_id": INSTANCE_ID,
            "known_hosts_file_sha256": _sha256(known_hosts_raw),
            "collector_source_sha256": source_sha256,
            "instance_authorization_sha256": authorization[0],
            "project_authorization_sha256": authorization[1],
            "oslogin_authorization_sha256": authorization[2],
        }
        return checked, authority


def _validate_transport_authority(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _TRANSPORT_FIELDS
        or value.get("kind") != "pinned_owner_gcloud_iap_ssh_read_only"
        or value.get("project") != PROJECT
        or value.get("zone") != ZONE
        or value.get("vm") != VM_NAME
        or value.get("instance_id") != INSTANCE_ID
        or any(
            _SHA256.fullmatch(str(value.get(field, ""))) is None
            for field in _TRANSPORT_FIELDS
            if field.endswith("sha256")
        )
    ):
        _error("owner_gate_v1_credential_transport_authority_invalid")
    return dict(value)


def validate_source_receipt(
    value: Mapping[str, Any],
    *,
    release_revision: str,
    host_key_id: str,
    now_unix: int,
) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _SOURCE_RECEIPT_FIELDS
        or _REVISION.fullmatch(release_revision or "") is None
        or _SHA256.fullmatch(host_key_id or "") is None
        or type(now_unix) is not int
    ):
        _error("owner_gate_v1_credential_source_receipt_invalid")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    observation = validate_observation(
        value.get("observation"),
        release_revision=release_revision,
        now_unix=now_unix,
    )
    authority = _validate_transport_authority(value.get("transport_authority"))
    signed_at = value.get("signed_at_unix")
    if (
        value.get("schema") != SOURCE_RECEIPT_SCHEMA
        or value.get("release_revision") != release_revision
        or value.get("observation_report_sha256")
        != observation["report_sha256"]
        or value.get("transport_authority") != authority
        or value.get("host_collector_public_key_id") != host_key_id
        or type(signed_at) is not int
        or not observation["completed_at_unix"] <= signed_at <= now_unix
        or signed_at - observation["completed_at_unix"]
        > MAX_SIGNING_DELAY_SECONDS
        or signed_at > observation["fresh_through_unix"]
        or value.get("production_mutation_performed") is not False
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or value.get("receipt_sha256") != _sha256_json(unsigned)
    ):
        _error("owner_gate_v1_credential_source_receipt_invalid")
    return dict(value)


def collect_and_sign_migration(
    transport: V1CredentialMigrationTransport,
    *,
    release_revision: str,
    host_private_key: Any,
    now_unix: int | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return the bootstrap envelope and its retained source receipt."""

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    if (
        not isinstance(transport, V1CredentialMigrationTransport)
        or _REVISION.fullmatch(release_revision or "") is None
        or not isinstance(host_private_key, Ed25519PrivateKey)
    ):
        _error("owner_gate_v1_credential_author_invalid")
    observation, transport_authority = transport.observe(
        release_revision=release_revision,
    )
    signed_at = int(time.time()) if now_unix is None else now_unix
    checked = validate_observation(
        observation,
        release_revision=release_revision,
        now_unix=signed_at,
    )
    if (
        signed_at - checked["completed_at_unix"] > MAX_SIGNING_DELAY_SECONDS
        or signed_at > checked["fresh_through_unix"]
    ):
        _error("owner_gate_v1_credential_signing_time_invalid")
    authority = _validate_transport_authority(transport_authority)
    public_raw = host_private_key.public_key().public_bytes_raw()
    host_key_id = _sha256(public_raw)
    source_unsigned = {
        "schema": SOURCE_RECEIPT_SCHEMA,
        "release_revision": release_revision,
        "observation": checked,
        "observation_report_sha256": checked["report_sha256"],
        "transport_authority": authority,
        "host_collector_public_key_id": host_key_id,
        "signed_at_unix": signed_at,
        "production_mutation_performed": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    source_receipt = {
        **source_unsigned,
        "receipt_sha256": _sha256_json(source_unsigned),
    }
    source_receipt = validate_source_receipt(
        source_receipt,
        release_revision=release_revision,
        host_key_id=host_key_id,
        now_unix=signed_at,
    )
    return (
        sign_migration_from_source_receipt(
            source_receipt,
            release_revision=release_revision,
            host_private_key=host_private_key,
        ),
        source_receipt,
    )


def sign_migration_from_source_receipt(
    source_receipt: Mapping[str, Any],
    *,
    release_revision: str,
    host_private_key: Any,
) -> Mapping[str, Any]:
    """Deterministically recover the envelope after source receipt publication."""

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from scripts.canary import owner_gate_bootstrap as bootstrap

    if (
        _REVISION.fullmatch(release_revision or "") is None
        or not isinstance(host_private_key, Ed25519PrivateKey)
    ):
        _error("owner_gate_v1_credential_author_invalid")
    host_key_id = _sha256(host_private_key.public_key().public_bytes_raw())
    signed_at = source_receipt.get("signed_at_unix")
    if type(signed_at) is not int:
        _error("owner_gate_v1_credential_source_receipt_invalid")
    source_receipt = validate_source_receipt(
        source_receipt,
        release_revision=release_revision,
        host_key_id=host_key_id,
        now_unix=signed_at,
    )
    checked = source_receipt["observation"]
    credential = checked["credential_public_material"]
    unsigned = {
        "schema": MIGRATION_SCHEMA,
        "release_revision": release_revision,
        "source_service_sha256": EXPECTED_SOURCE_SERVICE_SHA256,
        "owner_discord_user_id": OWNER_DISCORD_USER_ID,
        "credential_id_b64url": credential["credential_id_b64url"],
        "credential_id_sha256": credential["credential_id_sha256"],
        "public_key_cose_b64url": credential["public_key_cose_b64url"],
        "public_key_cose_sha256": credential["public_key_cose_sha256"],
        "expected_user_handle_b64url": credential[
            "expected_user_handle_b64url"
        ],
        "expected_user_handle_sha256": credential[
            "expected_user_handle_sha256"
        ],
        "initial_sign_count": credential["initial_sign_count"],
        "initial_credential_backed_up": credential[
            "initial_credential_backed_up"
        ],
        "source_receipt_sha256": source_receipt["receipt_sha256"],
        "collected_at_unix": checked["completed_at_unix"],
        "host_collector_public_key_id": host_key_id,
    }
    envelope_sha256 = _sha256_json(unsigned)
    signed = {**unsigned, "envelope_sha256": envelope_sha256}
    signature = host_private_key.sign(_canonical(signed))
    envelope = {
        **signed,
        "signature_ed25519_b64url": base64.urlsafe_b64encode(signature)
        .rstrip(b"=")
        .decode("ascii"),
    }
    try:
        validated = bootstrap.validate_migration(
            envelope,
            release_revision=release_revision,
            host_public_key=host_private_key.public_key(),
            host_key_id=host_key_id,
        )
    except bootstrap.OwnerGateBootstrapError as exc:
        _error("owner_gate_v1_credential_envelope_invalid", exc)
    return validated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--release-revision", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.action != "collect":
            _error("owner_gate_v1_credential_action_invalid")
        value = collect_observation(release_revision=arguments.release_revision)
    except V1CredentialMigrationError as exc:
        failure = {
            "schema": FAILURE_SCHEMA,
            "ok": False,
            "error_code": str(exc),
            "production_mutation_performed": False,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        print(_canonical(failure).decode("ascii"))
        return 1
    except BaseException:
        failure = {
            "schema": FAILURE_SCHEMA,
            "ok": False,
            "error_code": "owner_gate_v1_credential_unexpected_failure",
            "production_mutation_performed": False,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        print(_canonical(failure).decode("ascii"))
        return 1
    print(_canonical(value).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "V1CredentialMigrationError",
    "V1CredentialMigrationTransport",
    "collect_and_sign_migration",
    "collect_observation",
    "sign_migration_from_source_receipt",
    "validate_observation",
    "validate_source_receipt",
]
