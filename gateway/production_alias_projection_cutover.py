"""Transactional activation boundary for recurring Canonical alias projection.

The package is installed and attested while all three units remain disabled.
Only a coordinator authority bound to exact writer/gateway journal entries may
run the initial projection and enable its timer.  Before that terminal
authority exists, rollback restores the byte-exact unit prestate.

No function in this module interprets text or chooses a person.  The writer
exports the append-only log, and the unprivileged projector folds only the
exact ``person.alias.learned`` and ``channel.alias.learned`` event schemas.
"""

from __future__ import annotations

import argparse
import base64
import copy
import grp
import hashlib
import json
import os
import pwd
import re
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from gateway.production_alias_projection_units import (
    EXPORTER_UNIT,
    PACKAGE_RELATIVE_ROOT,
    PRIVATE_EXPORT_DIRECTORY,
    PROJECTOR_ROOT,
    PROJECTOR_TIMER,
    PROJECTOR_UNIT,
    PUBLIC_PROJECTION_DIRECTORY,
    SYSTEMD_ROOT,
    WRITER_CONFIG_PATH,
    WRITER_CREDENTIAL_PATH,
    validate_package_manifest,
)
from gateway.support_ops_alias_projection import (
    load_alias_projection_document,
)
from gateway.support_ops_team_registry import (
    SKYVISION_GUILD_ID,
    STATIC_ALIAS_CHANNEL_IDS,
    STATIC_ALIAS_MEMBER_KEYS,
    TEAM_MEMBERS_BY_KEY,
    normalize_team_member_alias,
)
from scripts.canonical_brain_alias_projector import (
    PRODUCTION_PUBLIC_PROJECTION_PATH,
    PRODUCTION_RUN_RECEIPT_PATH,
    PRODUCTION_WRITER_EXPORT_PATH,
    validate_run_receipt,
)


PREFLIGHT_SCHEMA = "muncho-production-alias-projection-preflight.v1"
APPLY_SCHEMA = "muncho-production-alias-projection-apply.v1"
POSTFLIGHT_SCHEMA = "muncho-production-alias-projection-postflight.v1"
ACTIVATION_SCHEMA = "muncho-production-alias-projection-activation.v1"
ROLLBACK_SCHEMA = "muncho-production-alias-projection-rollback.v1"
ACTIVATION_AUTHORITY_SCHEMA = (
    "muncho-production-alias-projection-activation-authority.v1"
)
PREPARED_SCHEMA = "muncho-production-alias-projection-prepared.v1"

EVIDENCE_ROOT = Path(
    "/var/lib/muncho-production-legacy-cutover/alias-projection"
)
STAGED_ACTIVATION_AUTHORITY_PATH = Path(
    "/var/lib/muncho-production-legacy-cutover/staged/"
    "alias-projection-activation-authority.json"
)
SYSTEMCTL = "/usr/bin/systemctl"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(
    r"^20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
_MAX_FILE = 8 * 1024 * 1024
_UNITS = (EXPORTER_UNIT, PROJECTOR_UNIT, PROJECTOR_TIMER)
_RECEIPT_SCHEMAS = {
    "preflight": PREFLIGHT_SCHEMA,
    "apply": APPLY_SCHEMA,
    "postflight": POSTFLIGHT_SCHEMA,
    "activation": ACTIVATION_SCHEMA,
    "rollback": ROLLBACK_SCHEMA,
}


class ProductionAliasProjectionCutoverError(RuntimeError):
    """Stable, secret-free alias rail cutover failure."""


@dataclass(frozen=True)
class UnitState:
    load_state: str
    active_state: str
    unit_file_state: str
    fragment_path: str
    result: str

    @property
    def active(self) -> bool:
        return self.active_state in {"active", "activating", "reloading"}

    @property
    def enabled(self) -> bool:
        return self.unit_file_state in {
            "enabled",
            "enabled-runtime",
            "linked",
            "linked-runtime",
        }


class SystemdOperations(Protocol):
    def state(self, unit: str) -> UnitState: ...

    def daemon_reload(self) -> None: ...

    def disable_now(self, unit: str) -> None: ...

    def enable_now(self, unit: str) -> None: ...

    def start(self, unit: str) -> None: ...

    def stop(self, unit: str) -> None: ...


class ProductionSystemdOperations:
    """Fixed-command systemd adapter with no caller-selected unit names."""

    @staticmethod
    def _run(*args: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                (SYSTEMCTL, *args),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="strict",
                env={"LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin"},
                cwd="/",
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired, UnicodeError) as exc:
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_systemd_failed"
            ) from exc

    @staticmethod
    def _fixed(unit: str) -> str:
        if unit not in _UNITS and unit not in {
            "muncho-canonical-writer.service",
            "hermes-cloud-gateway.service",
        }:
            raise ValueError("alias projection systemd unit is not fixed")
        return unit

    def state(self, unit: str) -> UnitState:
        unit = self._fixed(unit)
        result = self._run(
            "show",
            "--no-pager",
            "--property=LoadState,ActiveState,UnitFileState,FragmentPath,Result",
            "--",
            unit,
        )
        if result.returncode != 0 or result.stderr:
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_systemd_observation_failed"
            )
        fields: dict[str, str] = {}
        for line in result.stdout.splitlines():
            key, separator, value = line.partition("=")
            if not separator or key in fields:
                raise ProductionAliasProjectionCutoverError(
                    "alias_projection_systemd_observation_invalid"
                )
            fields[key] = value
        if set(fields) != {
            "LoadState",
            "ActiveState",
            "UnitFileState",
            "FragmentPath",
            "Result",
        }:
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_systemd_observation_invalid"
            )
        return UnitState(
            load_state=fields["LoadState"],
            active_state=fields["ActiveState"],
            unit_file_state=fields["UnitFileState"],
            fragment_path=fields["FragmentPath"],
            result=fields["Result"],
        )

    def _change(self, action: str, unit: str) -> None:
        unit = self._fixed(unit)
        result = self._run(action, "--", unit)
        if result.returncode != 0 or result.stderr:
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_systemd_action_failed"
            )

    def daemon_reload(self) -> None:
        result = self._run("daemon-reload")
        if result.returncode != 0 or result.stderr:
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_systemd_action_failed"
            )

    def disable_now(self, unit: str) -> None:
        self._change("disable", unit)
        self._change("stop", unit)

    def enable_now(self, unit: str) -> None:
        self._change("enable", unit)
        self._change("start", unit)

    def start(self, unit: str) -> None:
        self._change("start", unit)

    def stop(self, unit: str) -> None:
        self._change("stop", unit)


@dataclass(frozen=True)
class Principal:
    name: str
    uid: int
    gid: int
    home: str
    shell: str
    gids: tuple[int, ...]


class IdentityOperations(Protocol):
    def principal(self, name: str) -> Principal: ...

    def group_gid(self, name: str) -> int: ...


class ProductionIdentityOperations:
    def principal(self, name: str) -> Principal:
        try:
            account = pwd.getpwnam(name)
        except KeyError as exc:
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_identity_unavailable"
            ) from exc
        return Principal(
            name=account.pw_name,
            uid=account.pw_uid,
            gid=account.pw_gid,
            home=account.pw_dir,
            shell=account.pw_shell,
            gids=tuple(
                sorted(set(os.getgrouplist(account.pw_name, account.pw_gid)))
            ),
        )

    def group_gid(self, name: str) -> int:
        try:
            return grp.getgrnam(name).gr_gid
        except KeyError as exc:
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_identity_unavailable"
            ) from exc


@dataclass(frozen=True)
class RuntimeContext:
    manifest: Mapping[str, Any]
    unit_payloads: Mapping[str, bytes]

    @property
    def package_sha256(self) -> str:
        return str(self.manifest["package_sha256"])

    @property
    def revision(self) -> str:
        return str(self.manifest["release_revision"])


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
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_cutover_json_invalid"
        ) from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
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


def _stable_read(
    path: Path,
    *,
    maximum: int = _MAX_FILE,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
    expected_mode: int | None = None,
    allow_empty: bool = False,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        path_before = path.lstat()
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_cutover_file_unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            _file_identity(path_before) != _file_identity(before)
            or stat.S_ISLNK(path_before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or (not allow_empty and before.st_size == 0)
            or before.st_size > maximum
            or (expected_uid is not None and before.st_uid != expected_uid)
            or (expected_gid is not None and before.st_gid != expected_gid)
            or (
                expected_mode is not None
                and stat.S_IMODE(before.st_mode) != expected_mode
            )
        ):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_cutover_file_identity_invalid"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_cutover_file_changed"
        ) from exc
    if (
        len(raw) != before.st_size
        or _file_identity(before) != _file_identity(after)
        or _file_identity(before) != _file_identity(path_after)
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_cutover_file_changed"
        )
    return raw


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if not isinstance(key, str) or not key or key in result:
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_cutover_json_keys_invalid"
            )
        result[key] = value
    return result


def _json_file(path: Path, **read_kwargs: Any) -> dict[str, Any]:
    try:
        value = json.loads(
            _stable_read(path, **read_kwargs).decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ProductionAliasProjectionCutoverError(
                    "alias_projection_cutover_json_constant_invalid"
                )
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_cutover_json_file_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_cutover_json_file_invalid"
        )
    return value


def _atomic_write(
    path: Path,
    payload: bytes,
    *,
    mode: int,
    uid: int | None = None,
    gid: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_cutover_symlink_forbidden"
        )
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        os.fchmod(descriptor, mode)
        if uid is not None or gid is not None:
            os.fchown(descriptor, -1 if uid is None else uid, -1 if gid is None else gid)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ProductionAliasProjectionCutoverError(
                    "alias_projection_cutover_write_failed"
                )
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _receipt(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(unsigned))
    return {**payload, "receipt_sha256": _sha256(_canonical(payload))}


def build_activation_authority(
    *,
    cutover_plan_sha256: str,
    package_sha256: str,
    postflight_receipt_sha256: str,
    database_terminal_entry_sha256: str,
    activation_commit_intent_entry_sha256: str,
    writer_ready_entry_sha256: str,
    gateway_started_entry_sha256: str,
) -> dict[str, Any]:
    """Build forward-only authority after writer and gateway readiness."""

    digests = (
        cutover_plan_sha256,
        package_sha256,
        postflight_receipt_sha256,
        database_terminal_entry_sha256,
        activation_commit_intent_entry_sha256,
        writer_ready_entry_sha256,
        gateway_started_entry_sha256,
    )
    if any(_SHA256.fullmatch(value or "") is None for value in digests):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_activation_authority_invalid"
        )
    unsigned = {
        "schema": ACTIVATION_AUTHORITY_SCHEMA,
        "cutover_plan_sha256": cutover_plan_sha256,
        "package_sha256": package_sha256,
        "postflight_receipt_sha256": postflight_receipt_sha256,
        "database_terminal_entry_sha256": database_terminal_entry_sha256,
        "activation_commit_intent_entry_sha256": (
            activation_commit_intent_entry_sha256
        ),
        "writer_ready_entry_sha256": writer_ready_entry_sha256,
        "gateway_started_entry_sha256": gateway_started_entry_sha256,
        "database_terminal_validated": True,
        "activation_commit_intent_recorded": True,
        "writer_ready": True,
        "gateway_started": True,
        "forward_recovery_only": True,
        "secret_material_recorded": False,
    }
    return {**unsigned, "authority_sha256": _sha256(_canonical(unsigned))}


def validate_activation_authority(
    value: Any,
    *,
    cutover_plan_sha256: str,
    package_sha256: str,
    postflight_receipt_sha256: str,
    expected_authority_sha256: str,
) -> dict[str, Any]:
    fields = {
        "schema",
        "cutover_plan_sha256",
        "package_sha256",
        "postflight_receipt_sha256",
        "database_terminal_entry_sha256",
        "activation_commit_intent_entry_sha256",
        "writer_ready_entry_sha256",
        "gateway_started_entry_sha256",
        "database_terminal_validated",
        "activation_commit_intent_recorded",
        "writer_ready",
        "gateway_started",
        "forward_recovery_only",
        "secret_material_recorded",
        "authority_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_activation_authority_invalid"
        )
    result = dict(value)
    if (
        result.get("schema") != ACTIVATION_AUTHORITY_SCHEMA
        or result.get("cutover_plan_sha256") != cutover_plan_sha256
        or result.get("package_sha256") != package_sha256
        or result.get("postflight_receipt_sha256")
        != postflight_receipt_sha256
        or result.get("authority_sha256") != expected_authority_sha256
        or any(
            _SHA256.fullmatch(str(result.get(field) or "")) is None
            for field in (
                "database_terminal_entry_sha256",
                "activation_commit_intent_entry_sha256",
                "writer_ready_entry_sha256",
                "gateway_started_entry_sha256",
                "authority_sha256",
            )
        )
        or any(
            result.get(field) is not True
            for field in (
                "database_terminal_validated",
                "activation_commit_intent_recorded",
                "writer_ready",
                "gateway_started",
                "forward_recovery_only",
            )
        )
        or result.get("secret_material_recorded") is not False
        or _sha256(
            _canonical(
                {
                    key: item
                    for key, item in result.items()
                    if key != "authority_sha256"
                }
            )
        )
        != result.get("authority_sha256")
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_activation_authority_invalid"
        )
    return result


def validate_cutover_receipt(
    value: Any,
    *,
    action: str,
    cutover_plan_sha256: str,
    package_sha256: str,
    expected_sha256: str | None = None,
    expected_prior_sha256: str | None = None,
) -> dict[str, Any]:
    fields = {
        "schema",
        "action",
        "created_at",
        "cutover_plan_sha256",
        "package_sha256",
        "prior_receipt_sha256",
        "evidence",
        "provider_or_model_invoked",
        "discord_delivery_attempted",
        "secret_material_recorded",
        "receipt_sha256",
    }
    schema = _RECEIPT_SCHEMAS.get(action)
    if not isinstance(value, Mapping) or set(value) != fields or schema is None:
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_cutover_receipt_invalid"
        )
    result = dict(value)
    if (
        result.get("schema") != schema
        or result.get("action") != action
        or _UTC.fullmatch(str(result.get("created_at") or "")) is None
        or result.get("cutover_plan_sha256") != cutover_plan_sha256
        or result.get("package_sha256") != package_sha256
        or not isinstance(result.get("evidence"), Mapping)
        or result.get("prior_receipt_sha256") != expected_prior_sha256
        or result.get("provider_or_model_invoked") is not False
        or result.get("discord_delivery_attempted") is not False
        or result.get("secret_material_recorded") is not False
        or _SHA256.fullmatch(str(result.get("receipt_sha256") or "")) is None
        or (expected_sha256 is not None and result["receipt_sha256"] != expected_sha256)
        or _sha256(
            _canonical(
                {
                    key: item
                    for key, item in result.items()
                    if key != "receipt_sha256"
                }
            )
        )
        != result.get("receipt_sha256")
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_cutover_receipt_invalid"
        )
    return result


def _plan_root(cutover_plan_sha256: str, *, evidence_root: Path) -> Path:
    if _SHA256.fullmatch(cutover_plan_sha256 or "") is None:
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_cutover_plan_invalid"
        )
    return evidence_root / cutover_plan_sha256


def _receipt_path(action: str, cutover_plan_sha256: str, *, evidence_root: Path) -> Path:
    if action not in _RECEIPT_SCHEMAS:
        raise ValueError("alias projection receipt action is not fixed")
    return _plan_root(cutover_plan_sha256, evidence_root=evidence_root) / f"{action}.json"


def _publish_receipt(
    *,
    action: str,
    cutover_plan_sha256: str,
    package_sha256: str,
    prior_receipt_sha256: str | None,
    evidence: Mapping[str, Any],
    evidence_root: Path,
    clock: Callable[[], str],
) -> dict[str, Any]:
    result = _receipt(
        {
            "schema": _RECEIPT_SCHEMAS[action],
            "action": action,
            "created_at": clock(),
            "cutover_plan_sha256": cutover_plan_sha256,
            "package_sha256": package_sha256,
            "prior_receipt_sha256": prior_receipt_sha256,
            "evidence": copy.deepcopy(dict(evidence)),
            "provider_or_model_invoked": False,
            "discord_delivery_attempted": False,
            "secret_material_recorded": False,
        }
    )
    validate_cutover_receipt(
        result,
        action=action,
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=package_sha256,
        expected_prior_sha256=prior_receipt_sha256,
    )
    path = _receipt_path(action, cutover_plan_sha256, evidence_root=evidence_root)
    if path.exists() or path.is_symlink():
        current = validate_cutover_receipt(
            _json_file(path),
            action=action,
            cutover_plan_sha256=cutover_plan_sha256,
            package_sha256=package_sha256,
            expected_prior_sha256=prior_receipt_sha256,
        )
        comparable = {
            key: item for key, item in current.items() if key not in {"created_at", "receipt_sha256"}
        }
        proposed = {
            key: item for key, item in result.items() if key not in {"created_at", "receipt_sha256"}
        }
        if comparable != proposed:
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_cutover_receipt_replay_drifted"
            )
        return current
    _atomic_write(path, _canonical(result) + b"\n", mode=0o400)
    return validate_cutover_receipt(
        _json_file(path),
        action=action,
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=package_sha256,
        expected_sha256=result["receipt_sha256"],
        expected_prior_sha256=prior_receipt_sha256,
    )


def load_runtime_context(
    *,
    package_root: Path,
    expected_revision: str,
    expected_package_sha256: str,
    enforce_production_address: bool = True,
    enforce_package_metadata: bool = True,
) -> RuntimeContext:
    """Load one immutable package and bind every unit payload by digest."""

    root = Path(package_root)
    manifest_path = root / "manifest.json"
    read_kwargs: dict[str, Any] = {}
    if enforce_package_metadata:
        read_kwargs = {"expected_uid": 0, "expected_gid": 0, "expected_mode": 0o444}
    manifest = validate_package_manifest(
        _json_file(manifest_path, **read_kwargs),
        expected_revision=expected_revision,
        expected_package_sha256=expected_package_sha256,
    )
    release_root = Path(str(manifest["release_root"]))
    if enforce_production_address and root != release_root / PACKAGE_RELATIVE_ROOT:
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_package_address_invalid"
        )
    units = manifest.get("units")
    if not isinstance(units, Mapping) or set(units) != set(_UNITS):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_package_units_invalid"
        )
    payloads: dict[str, bytes] = {}
    for name in _UNITS:
        item = units.get(name)
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {"path", "artifact_path", "sha256", "uid", "gid", "mode"}
            or item.get("path") != str(SYSTEMD_ROOT / name)
            or item.get("artifact_path")
            != str(release_root / PACKAGE_RELATIVE_ROOT / name)
            or item.get("uid") != 0
            or item.get("gid") != 0
            or item.get("mode") != "0644"
            or _SHA256.fullmatch(str(item.get("sha256") or "")) is None
        ):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_package_units_invalid"
            )
        payload = _stable_read(root / name, **read_kwargs)
        if _sha256(payload) != item["sha256"]:
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_package_unit_digest_mismatch"
            )
        payloads[name] = payload

    identities = manifest.get("identities")
    expected_identity_names = {
        "writer": ("muncho-canonical-writer", "muncho-canonical-writer"),
        "projector": ("muncho-projector", "muncho-projector"),
        "gateway": ("ai-platform-brain", "ai-platform-brain"),
    }
    if not isinstance(identities, Mapping) or set(identities) != set(
        expected_identity_names
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_package_identities_invalid"
        )
    identity_ids: list[int] = []
    group_ids: list[int] = []
    for role, names in expected_identity_names.items():
        item = identities.get(role)
        if (
            not isinstance(item, Mapping)
            or set(item) != {"user", "group", "uid", "gid"}
            or (item.get("user"), item.get("group")) != names
            or type(item.get("uid")) is not int
            or type(item.get("gid")) is not int
            or item["uid"] <= 0
            or item["gid"] <= 0
        ):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_package_identities_invalid"
            )
        identity_ids.append(item["uid"])
        group_ids.append(item["gid"])
    if len(set(identity_ids)) != 3 or len(set(group_ids)) != 3:
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_package_identities_invalid"
        )

    directories = manifest.get("directories")
    expected_directories = {
        str(PRIVATE_EXPORT_DIRECTORY): (
            identities["writer"]["uid"],
            identities["projector"]["gid"],
            "0750",
        ),
        str(PROJECTOR_ROOT): (0, 0, "0751"),
        str(PUBLIC_PROJECTION_DIRECTORY): (
            identities["projector"]["uid"],
            identities["gateway"]["gid"],
            "2750",
        ),
    }
    if not isinstance(directories, Mapping) or set(directories) != set(
        expected_directories
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_package_directories_invalid"
        )
    for path, expected in expected_directories.items():
        item = directories[path]
        if (
            not isinstance(item, Mapping)
            or set(item) != {"uid", "gid", "mode"}
            or (item.get("uid"), item.get("gid"), item.get("mode")) != expected
        ):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_package_directories_invalid"
            )

    expected_files = {
        "writer_export": (
            str(PRODUCTION_WRITER_EXPORT_PATH),
            identities["writer"]["uid"],
            identities["projector"]["gid"],
            EXPORTER_UNIT,
        ),
        "public_projection": (
            str(PRODUCTION_PUBLIC_PROJECTION_PATH),
            identities["projector"]["uid"],
            identities["gateway"]["gid"],
            PROJECTOR_UNIT,
        ),
        "public_run_receipt": (
            str(PRODUCTION_RUN_RECEIPT_PATH),
            identities["projector"]["uid"],
            identities["gateway"]["gid"],
            PROJECTOR_UNIT,
        ),
    }
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(expected_files):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_package_files_invalid"
        )
    for name, expected in expected_files.items():
        item = files[name]
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "uid", "gid", "mode", "created_by"}
            or (
                item.get("path"),
                item.get("uid"),
                item.get("gid"),
                item.get("created_by"),
            )
            != expected
            or item.get("mode") != "0640"
        ):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_package_files_invalid"
            )

    ordering = manifest.get("ordering")
    credential = manifest.get("credential_boundary")
    if (
        not isinstance(ordering, Mapping)
        or ordering
        != {
            "timer_triggers": PROJECTOR_UNIT,
            "projector_requires": EXPORTER_UNIT,
            "exporter_before_projector": True,
            "timer_enabled_before_activation": False,
            "interval_seconds": 300,
        }
        or not isinstance(credential, Mapping)
        or credential
        != {
            "writer_credential_path": str(WRITER_CREDENTIAL_PATH),
            "projector_credential_paths": [],
            "gateway_credential_paths": [],
            "projector_network_private": True,
        }
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_package_boundary_invalid"
        )
    modules = manifest.get("modules")
    interpreter = manifest.get("interpreter")
    if (
        not isinstance(modules, Mapping)
        or set(modules)
        != {
            "writer_bootstrap",
            "alias_projector",
            "projection_reader",
            "team_registry",
            "cutover_runtime",
            "cutover_entrypoint",
        }
        or not isinstance(interpreter, Mapping)
        or set(interpreter) != {"path", "sha256"}
        or interpreter.get("path") != str(release_root / ".venv/bin/python")
        or _SHA256.fullmatch(str(interpreter.get("sha256") or "")) is None
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_package_runtime_invalid"
        )
    for item in modules.values():
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "sha256"}
            or not str(item.get("path") or "").startswith(str(release_root) + "/")
            or _SHA256.fullmatch(str(item.get("sha256") or "")) is None
        ):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_package_runtime_invalid"
            )

    exporter = payloads[EXPORTER_UNIT]
    projector = payloads[PROJECTOR_UNIT]
    timer = payloads[PROJECTOR_TIMER]
    if (
        f"Before={PROJECTOR_UNIT}".encode() not in exporter
        or f"Requires={EXPORTER_UNIT}".encode() not in projector
        or f"After={EXPORTER_UNIT}".encode() not in projector
        or f"Unit={PROJECTOR_UNIT}".encode() not in timer
        or b"[Install]" in exporter
        or b"[Install]" in projector
        or b"[Install]" not in timer
        or b"canonical-writer-db-password" in projector
        or b"PrivateNetwork=yes" not in projector
        or b"person.alias.learned" not in projector
        or b"channel.alias.learned" not in projector
        or any(
            pattern in projector
            for pattern in (b"EnvironmentFile=", b"LoadCredential=", b"PassEnvironment=")
        )
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_package_unit_boundary_invalid"
        )
    return RuntimeContext(manifest=manifest, unit_payloads=payloads)


def _mode_bits(item: os.stat_result, principal: Principal) -> int:
    mode = stat.S_IMODE(item.st_mode)
    if principal.uid == item.st_uid:
        return (mode >> 6) & 0o7
    if item.st_gid in principal.gids:
        return (mode >> 3) & 0o7
    return mode & 0o7


def _path_chain(path: Path) -> tuple[Path, ...]:
    if not path.is_absolute() or ".." in path.parts:
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_dependency_path_invalid"
        )
    current = Path("/")
    result: list[Path] = []
    for part in path.parts[1:]:
        current /= part
        result.append(current)
    return tuple(result)


def _require_access(
    path: Path,
    principal: Principal,
    *,
    read: bool = False,
    write: bool = False,
    execute: bool = False,
    directory: bool = False,
) -> os.stat_result:
    chain = _path_chain(path)
    if not chain:
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_dependency_path_invalid"
        )
    for parent in chain[:-1]:
        item = parent.lstat()
        if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_dependency_path_untrusted"
            )
        if _mode_bits(item, principal) & 0o1 == 0:
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_dependency_not_accessible"
            )
    item = path.lstat()
    if stat.S_ISLNK(item.st_mode) or (
        directory and not stat.S_ISDIR(item.st_mode)
    ) or (not directory and not stat.S_ISREG(item.st_mode)):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_dependency_path_untrusted"
        )
    bits = _mode_bits(item, principal)
    required = (0o4 if read else 0) | (0o2 if write else 0) | (0o1 if execute else 0)
    if bits & required != required:
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_dependency_not_accessible"
        )
    return item


def _access_denied(
    path: Path,
    principal: Principal,
    *,
    read: bool,
    directory: bool = False,
) -> bool:
    try:
        _require_access(path, principal, read=read, directory=directory)
    except (FileNotFoundError, PermissionError, ProductionAliasProjectionCutoverError):
        return True
    return False


def validate_host_dependencies(
    context: RuntimeContext,
    *,
    identities: IdentityOperations,
    require_directories: bool = True,
) -> Mapping[str, Any]:
    """Attest real service-user access without widening any host permission."""

    manifest_identities = context.manifest["identities"]
    writer = identities.principal("muncho-canonical-writer")
    projector = identities.principal("muncho-projector")
    gateway = identities.principal("ai-platform-brain")
    principals = {"writer": writer, "projector": projector, "gateway": gateway}
    for role, principal in principals.items():
        expected = manifest_identities[role]
        if (
            principal.name != expected["user"]
            or principal.uid != expected["uid"]
            or principal.gid != expected["gid"]
            or identities.group_gid(expected["group"]) != expected["gid"]
        ):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_host_identity_drifted"
            )
    if (
        writer.home != "/nonexistent"
        or projector.home != "/nonexistent"
        or writer.shell != "/usr/sbin/nologin"
        or projector.shell != "/usr/sbin/nologin"
        or writer.gids != tuple(sorted({writer.gid, projector.gid}))
        or projector.gids != (projector.gid,)
        or projector.gid in gateway.gids
        or writer.gid in gateway.gids
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_host_identity_drifted"
        )

    interpreter = Path(context.manifest["interpreter"]["path"])
    interpreter_raw = _stable_read(interpreter, maximum=64 * 1024 * 1024)
    if _sha256(interpreter_raw) != context.manifest["interpreter"]["sha256"]:
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_runtime_digest_drifted"
        )
    _require_access(interpreter, writer, read=True, execute=True)
    _require_access(interpreter, projector, read=True, execute=True)
    for name, item in context.manifest["modules"].items():
        path = Path(item["path"])
        raw = _stable_read(path)
        if _sha256(raw) != item["sha256"]:
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_runtime_digest_drifted"
            )
        if name == "writer_bootstrap":
            _require_access(path, writer, read=True)
        elif name in {"alias_projector", "projection_reader", "team_registry"}:
            _require_access(path, projector, read=True)

    _require_access(WRITER_CONFIG_PATH, writer, read=True)
    credential = _require_access(WRITER_CREDENTIAL_PATH, writer, read=True)
    if (
        credential.st_uid != writer.uid
        or credential.st_gid != writer.gid
        or stat.S_IMODE(credential.st_mode) != 0o400
        or not _access_denied(WRITER_CREDENTIAL_PATH, projector, read=True)
        or not _access_denied(WRITER_CREDENTIAL_PATH, gateway, read=True)
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_credential_boundary_invalid"
        )

    directories = context.manifest["directories"]
    for path_text, expected in directories.items():
        path = Path(path_text)
        if not path.exists() and not path.is_symlink() and not require_directories:
            continue
        item = path.lstat()
        if (
            stat.S_ISLNK(item.st_mode)
            or not stat.S_ISDIR(item.st_mode)
            or item.st_uid != expected["uid"]
            or item.st_gid != expected["gid"]
            or stat.S_IMODE(item.st_mode) != int(expected["mode"], 8)
        ):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_directory_identity_drifted"
            )
    if require_directories or PRIVATE_EXPORT_DIRECTORY.exists():
        _require_access(
            PRIVATE_EXPORT_DIRECTORY,
            writer,
            read=True,
            write=True,
            execute=True,
            directory=True,
        )
        _require_access(
            PRIVATE_EXPORT_DIRECTORY,
            projector,
            read=True,
            execute=True,
            directory=True,
        )
        if not _access_denied(
            PRIVATE_EXPORT_DIRECTORY,
            gateway,
            read=True,
            directory=True,
        ):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_private_export_gateway_accessible"
            )
    if require_directories or PUBLIC_PROJECTION_DIRECTORY.exists():
        _require_access(
            PUBLIC_PROJECTION_DIRECTORY,
            projector,
            read=True,
            write=True,
            execute=True,
            directory=True,
        )
        _require_access(
            PUBLIC_PROJECTION_DIRECTORY,
            gateway,
            read=True,
            execute=True,
            directory=True,
        )
    return {
        "identities_exact": True,
        "release_dependencies_executable_and_readable": True,
        "writer_credential_writer_only": True,
        "projector_credential_access": False,
        "gateway_credential_access": False,
        "gateway_private_export_access": False,
        "directory_contract_attested": require_directories,
        "public_projection_gateway_readable": (
            require_directories or PUBLIC_PROJECTION_DIRECTORY.exists()
        ),
    }


def _unit_states(systemd: SystemdOperations) -> dict[str, UnitState]:
    return {name: systemd.state(name) for name in _UNITS}


def _units_safe_disabled(states: Mapping[str, UnitState]) -> bool:
    return all(not item.active and not item.enabled for item in states.values())


def _file_prestate(path: Path) -> dict[str, Any]:
    if not path.exists() and not path.is_symlink():
        return {"state": "absent"}
    item = path.lstat()
    if stat.S_ISLNK(item.st_mode) or not stat.S_ISREG(item.st_mode) or item.st_nlink != 1:
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_prestate_file_untrusted"
        )
    raw = _stable_read(path, allow_empty=True)
    return {
        "state": "present",
        "uid": item.st_uid,
        "gid": item.st_gid,
        "mode": stat.S_IMODE(item.st_mode),
        "sha256": _sha256(raw),
        "bytes_b64": base64.b64encode(raw).decode("ascii"),
    }


def _directory_prestate(path: Path) -> dict[str, Any]:
    if not path.exists() and not path.is_symlink():
        return {"state": "absent"}
    item = path.lstat()
    if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_prestate_directory_untrusted"
        )
    return {
        "state": "present",
        "uid": item.st_uid,
        "gid": item.st_gid,
        "mode": stat.S_IMODE(item.st_mode),
    }


def _data_file_prestate(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists() and not path.is_symlink():
        return {"state": "absent"}
    item = path.lstat()
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISREG(item.st_mode)
        or item.st_nlink != 1
        or item.st_uid != expected["uid"]
        or item.st_gid != expected["gid"]
        or stat.S_IMODE(item.st_mode) != int(expected["mode"], 8)
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_data_file_identity_invalid"
        )
    return {
        "state": "present",
        "device": item.st_dev,
        "inode": item.st_ino,
        "uid": item.st_uid,
        "gid": item.st_gid,
        "mode": stat.S_IMODE(item.st_mode),
        "size": item.st_size,
        "mtime_ns": item.st_mtime_ns,
        "ctime_ns": item.st_ctime_ns,
    }


def _prepared_snapshot(
    context: RuntimeContext,
    *,
    cutover_plan_sha256: str,
    systemd: SystemdOperations,
) -> dict[str, Any]:
    states = _unit_states(systemd)
    unsigned = {
        "schema": PREPARED_SCHEMA,
        "cutover_plan_sha256": cutover_plan_sha256,
        "package_sha256": context.package_sha256,
        "units": {
            name: {
                "file": _file_prestate(SYSTEMD_ROOT / name),
                "load_state": state.load_state,
                "active_state": state.active_state,
                "unit_file_state": state.unit_file_state,
                "fragment_path": state.fragment_path,
                "result": state.result,
            }
            for name, state in states.items()
        },
        "directories": {
            path: _directory_prestate(Path(path))
            for path in context.manifest["directories"]
        },
        "data_files": {
            context.manifest["files"][name]["path"]: _data_file_prestate(
                Path(context.manifest["files"][name]["path"]),
                context.manifest["files"][name],
            )
            for name in (
                "writer_export",
                "public_projection",
                "public_run_receipt",
            )
        },
        "secret_material_recorded": False,
    }
    return {**unsigned, "prepared_sha256": _sha256(_canonical(unsigned))}


def _validate_prepared(
    value: Any,
    *,
    cutover_plan_sha256: str,
    package_sha256: str,
) -> dict[str, Any]:
    fields = {
        "schema",
        "cutover_plan_sha256",
        "package_sha256",
        "units",
        "directories",
        "data_files",
        "secret_material_recorded",
        "prepared_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or value.get("schema") != PREPARED_SCHEMA
        or value.get("cutover_plan_sha256") != cutover_plan_sha256
        or value.get("package_sha256") != package_sha256
        or value.get("secret_material_recorded") is not False
        or not isinstance(value.get("units"), Mapping)
        or set(value["units"]) != set(_UNITS)
        or not isinstance(value.get("directories"), Mapping)
        or not isinstance(value.get("data_files"), Mapping)
        or _SHA256.fullmatch(str(value.get("prepared_sha256") or "")) is None
        or _sha256(
            _canonical(
                {key: item for key, item in value.items() if key != "prepared_sha256"}
            )
        )
        != value.get("prepared_sha256")
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_prepared_invalid"
        )
    return copy.deepcopy(dict(value))


def _prepared_path(cutover_plan_sha256: str, *, evidence_root: Path) -> Path:
    return _plan_root(cutover_plan_sha256, evidence_root=evidence_root) / "prepared.json"


def _load_prepared(
    *,
    cutover_plan_sha256: str,
    package_sha256: str,
    evidence_root: Path,
) -> dict[str, Any]:
    return _validate_prepared(
        _json_file(_prepared_path(cutover_plan_sha256, evidence_root=evidence_root)),
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=package_sha256,
    )


def _require_root(require_root: bool) -> None:
    if require_root and (os.name != "posix" or os.geteuid() != 0):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_cutover_requires_root"
        )


def _core_services_stopped(systemd: SystemdOperations) -> bool:
    return not systemd.state("muncho-canonical-writer.service").active and not systemd.state(
        "hermes-cloud-gateway.service"
    ).active


def _core_services_active(systemd: SystemdOperations) -> bool:
    return systemd.state("muncho-canonical-writer.service").active and systemd.state(
        "hermes-cloud-gateway.service"
    ).active


def _load_prior_with_chain(
    *,
    action: str,
    cutover_plan_sha256: str,
    package_sha256: str,
    expected_sha256: str,
    expected_prior_sha256: str | None,
    evidence_root: Path,
) -> dict[str, Any]:
    return validate_cutover_receipt(
        _json_file(
            _receipt_path(action, cutover_plan_sha256, evidence_root=evidence_root)
        ),
        action=action,
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=package_sha256,
        expected_sha256=expected_sha256,
        expected_prior_sha256=expected_prior_sha256,
    )


def _ensure_directory(path: Path, *, uid: int, gid: int, mode: int) -> None:
    if path.exists() or path.is_symlink():
        item = path.lstat()
        if (
            stat.S_ISLNK(item.st_mode)
            or not stat.S_ISDIR(item.st_mode)
            or item.st_uid != uid
            or item.st_gid != gid
            or stat.S_IMODE(item.st_mode) != mode
        ):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_directory_identity_drifted"
            )
        return
    parent = path.parent.lstat()
    if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_directory_parent_untrusted"
        )
    path.mkdir(mode=mode)
    os.chown(path, uid, gid)
    os.chmod(path, mode)
    item = path.lstat()
    if (
        item.st_uid != uid
        or item.st_gid != gid
        or stat.S_IMODE(item.st_mode) != mode
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_directory_install_failed"
        )


def _install_context(
    context: RuntimeContext,
    *,
    systemd: SystemdOperations,
) -> None:
    for path_text in sorted(
        context.manifest["directories"], key=lambda value: len(Path(value).parts)
    ):
        item = context.manifest["directories"][path_text]
        _ensure_directory(
            Path(path_text),
            uid=item["uid"],
            gid=item["gid"],
            mode=int(item["mode"], 8),
        )
    states = _unit_states(systemd)
    if states[PROJECTOR_TIMER].load_state == "loaded":
        systemd.disable_now(PROJECTOR_TIMER)
    if states[PROJECTOR_UNIT].active:
        systemd.stop(PROJECTOR_UNIT)
    if states[EXPORTER_UNIT].active:
        systemd.stop(EXPORTER_UNIT)
    for name, payload in context.unit_payloads.items():
        _atomic_write(SYSTEMD_ROOT / name, payload, mode=0o644, uid=0, gid=0)
    systemd.daemon_reload()
    systemd.disable_now(PROJECTOR_TIMER)


def _installed_units_exact(
    context: RuntimeContext,
    *,
    systemd: SystemdOperations,
    require_disabled: bool = True,
) -> bool:
    states = _unit_states(systemd)
    if require_disabled and not _units_safe_disabled(states):
        return False
    if states[EXPORTER_UNIT].active or states[PROJECTOR_UNIT].active:
        return False
    for name, payload in context.unit_payloads.items():
        try:
            raw = _stable_read(
                SYSTEMD_ROOT / name,
                expected_uid=0,
                expected_gid=0,
                expected_mode=0o644,
            )
        except ProductionAliasProjectionCutoverError:
            return False
        state = states[name]
        if (
            raw != payload
            or state.load_state != "loaded"
            or state.fragment_path != str(SYSTEMD_ROOT / name)
        ):
            return False
    return True


def _data_files_unchanged(prepared: Mapping[str, Any]) -> bool:
    for path_text, before in prepared["data_files"].items():
        path = Path(path_text)
        if before.get("state") == "absent":
            if path.exists() or path.is_symlink():
                return False
            continue
        if before.get("state") != "present" or not path.exists() or path.is_symlink():
            return False
        item = path.lstat()
        current = {
            "state": "present",
            "device": item.st_dev,
            "inode": item.st_ino,
            "uid": item.st_uid,
            "gid": item.st_gid,
            "mode": stat.S_IMODE(item.st_mode),
            "size": item.st_size,
            "mtime_ns": item.st_mtime_ns,
            "ctime_ns": item.st_ctime_ns,
        }
        if current != before:
            return False
    return True


def preflight(
    *,
    cutover_plan_sha256: str,
    package_root: Path,
    expected_revision: str,
    expected_package_sha256: str,
    evidence_root: Path = EVIDENCE_ROOT,
    activation_authority_path: Path = STAGED_ACTIVATION_AUTHORITY_PATH,
    systemd: SystemdOperations | None = None,
    identities: IdentityOperations | None = None,
    clock: Callable[[], str] = _now,
    require_root: bool = True,
    enforce_production_address: bool = True,
    enforce_package_metadata: bool = True,
) -> dict[str, Any]:
    """Persist rollback evidence while every projection unit is disabled."""

    _require_root(require_root)
    systemd = systemd or ProductionSystemdOperations()
    identities = identities or ProductionIdentityOperations()
    context = load_runtime_context(
        package_root=package_root,
        expected_revision=expected_revision,
        expected_package_sha256=expected_package_sha256,
        enforce_production_address=enforce_production_address,
        enforce_package_metadata=enforce_package_metadata,
    )
    existing_path = _receipt_path(
        "preflight", cutover_plan_sha256, evidence_root=evidence_root
    )
    if existing_path.exists() and not existing_path.is_symlink():
        return validate_cutover_receipt(
            _json_file(existing_path),
            action="preflight",
            cutover_plan_sha256=cutover_plan_sha256,
            package_sha256=context.package_sha256,
            expected_prior_sha256=None,
        )
    if activation_authority_path.exists() or activation_authority_path.is_symlink():
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_activation_authority_premature"
        )
    states = _unit_states(systemd)
    if not _units_safe_disabled(states) or not _core_services_stopped(systemd):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_preflight_services_not_stopped"
        )
    dependency = validate_host_dependencies(
        context,
        identities=identities,
        require_directories=False,
    )
    prepared = _prepared_snapshot(
        context,
        cutover_plan_sha256=cutover_plan_sha256,
        systemd=systemd,
    )
    prepared_path = _prepared_path(cutover_plan_sha256, evidence_root=evidence_root)
    if prepared_path.exists() or prepared_path.is_symlink():
        existing = _validate_prepared(
            _json_file(prepared_path),
            cutover_plan_sha256=cutover_plan_sha256,
            package_sha256=context.package_sha256,
        )
        if existing != prepared:
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_preflight_replay_drifted"
            )
    else:
        _atomic_write(prepared_path, _canonical(prepared) + b"\n", mode=0o400)
    return _publish_receipt(
        action="preflight",
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=context.package_sha256,
        prior_receipt_sha256=None,
        evidence={
            "package_valid": True,
            "core_services_stopped": True,
            "projection_units_disabled": True,
            "host_dependencies": dict(dependency),
            "prepared_sha256": prepared["prepared_sha256"],
            "runtime_target_mutation_performed": False,
        },
        evidence_root=evidence_root,
        clock=clock,
    )


def apply(
    *,
    cutover_plan_sha256: str,
    package_root: Path,
    expected_revision: str,
    expected_package_sha256: str,
    expected_preflight_receipt_sha256: str,
    evidence_root: Path = EVIDENCE_ROOT,
    activation_authority_path: Path = STAGED_ACTIVATION_AUTHORITY_PATH,
    systemd: SystemdOperations | None = None,
    identities: IdentityOperations | None = None,
    clock: Callable[[], str] = _now,
    require_root: bool = True,
    enforce_production_address: bool = True,
    enforce_package_metadata: bool = True,
) -> dict[str, Any]:
    """Install the package byte-exactly without enabling or executing it."""

    _require_root(require_root)
    systemd = systemd or ProductionSystemdOperations()
    identities = identities or ProductionIdentityOperations()
    context = load_runtime_context(
        package_root=package_root,
        expected_revision=expected_revision,
        expected_package_sha256=expected_package_sha256,
        enforce_production_address=enforce_production_address,
        enforce_package_metadata=enforce_package_metadata,
    )
    pre = _load_prior_with_chain(
        action="preflight",
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=context.package_sha256,
        expected_sha256=expected_preflight_receipt_sha256,
        expected_prior_sha256=None,
        evidence_root=evidence_root,
    )
    existing_path = _receipt_path("apply", cutover_plan_sha256, evidence_root=evidence_root)
    if existing_path.exists() and not existing_path.is_symlink():
        if not _installed_units_exact(context, systemd=systemd):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_apply_replay_state_drifted"
            )
        return validate_cutover_receipt(
            _json_file(existing_path),
            action="apply",
            cutover_plan_sha256=cutover_plan_sha256,
            package_sha256=context.package_sha256,
            expected_prior_sha256=pre["receipt_sha256"],
        )
    if activation_authority_path.exists() or activation_authority_path.is_symlink():
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_activation_authority_premature"
        )
    if not _core_services_stopped(systemd):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_apply_services_not_stopped"
        )
    prepared = _load_prepared(
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=context.package_sha256,
        evidence_root=evidence_root,
    )
    if not _data_files_unchanged(prepared):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_apply_data_prestate_changed"
        )
    _install_context(context, systemd=systemd)
    dependency = validate_host_dependencies(
        context,
        identities=identities,
        require_directories=True,
    )
    if not _installed_units_exact(context, systemd=systemd):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_apply_readback_failed"
        )
    return _publish_receipt(
        action="apply",
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=context.package_sha256,
        prior_receipt_sha256=pre["receipt_sha256"],
        evidence={
            "prepared_sha256": prepared["prepared_sha256"],
            "unit_file_count": len(_UNITS),
            "directories_exact": True,
            "units_installed": True,
            "units_disabled": True,
            "jobs_executed": False,
            "host_dependencies": dict(dependency),
        },
        evidence_root=evidence_root,
        clock=clock,
    )


def postflight(
    *,
    cutover_plan_sha256: str,
    package_root: Path,
    expected_revision: str,
    expected_package_sha256: str,
    expected_preflight_receipt_sha256: str,
    expected_apply_receipt_sha256: str,
    evidence_root: Path = EVIDENCE_ROOT,
    activation_authority_path: Path = STAGED_ACTIVATION_AUTHORITY_PATH,
    systemd: SystemdOperations | None = None,
    identities: IdentityOperations | None = None,
    clock: Callable[[], str] = _now,
    require_root: bool = True,
    enforce_production_address: bool = True,
    enforce_package_metadata: bool = True,
) -> dict[str, Any]:
    """Attest exact installed bytes, access, and still-disabled state."""

    _require_root(require_root)
    systemd = systemd or ProductionSystemdOperations()
    identities = identities or ProductionIdentityOperations()
    context = load_runtime_context(
        package_root=package_root,
        expected_revision=expected_revision,
        expected_package_sha256=expected_package_sha256,
        enforce_production_address=enforce_production_address,
        enforce_package_metadata=enforce_package_metadata,
    )
    pre = _load_prior_with_chain(
        action="preflight",
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=context.package_sha256,
        expected_sha256=expected_preflight_receipt_sha256,
        expected_prior_sha256=None,
        evidence_root=evidence_root,
    )
    applied = _load_prior_with_chain(
        action="apply",
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=context.package_sha256,
        expected_sha256=expected_apply_receipt_sha256,
        expected_prior_sha256=pre["receipt_sha256"],
        evidence_root=evidence_root,
    )
    existing_path = _receipt_path(
        "postflight", cutover_plan_sha256, evidence_root=evidence_root
    )
    if existing_path.exists() and not existing_path.is_symlink():
        return validate_cutover_receipt(
            _json_file(existing_path),
            action="postflight",
            cutover_plan_sha256=cutover_plan_sha256,
            package_sha256=context.package_sha256,
            expected_prior_sha256=applied["receipt_sha256"],
        )
    if activation_authority_path.exists() or activation_authority_path.is_symlink():
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_activation_authority_premature"
        )
    if not _core_services_stopped(systemd) or not _installed_units_exact(
        context, systemd=systemd
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_postflight_state_invalid"
        )
    prepared = _load_prepared(
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=context.package_sha256,
        evidence_root=evidence_root,
    )
    if not _data_files_unchanged(prepared):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_postflight_unexpected_execution"
        )
    dependency = validate_host_dependencies(
        context,
        identities=identities,
        require_directories=True,
    )
    return _publish_receipt(
        action="postflight",
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=context.package_sha256,
        prior_receipt_sha256=applied["receipt_sha256"],
        evidence={
            "units_match_package": True,
            "units_disabled": True,
            "units_active": False,
            "projection_not_executed": True,
            "host_dependencies": dict(dependency),
            "rollback_available_before_terminal_authority": True,
        },
        evidence_root=evidence_root,
        clock=clock,
    )


def _projection_run_readback(context: RuntimeContext) -> Mapping[str, Any]:
    identities = context.manifest["identities"]
    export_raw = _stable_read(
        PRODUCTION_WRITER_EXPORT_PATH,
        expected_uid=identities["writer"]["uid"],
        expected_gid=identities["projector"]["gid"],
        expected_mode=0o640,
        maximum=256 * 1024 * 1024,
    )
    projection = load_alias_projection_document(
        PRODUCTION_PUBLIC_PROJECTION_PATH,
        normalize_alias=normalize_team_member_alias,
        valid_member_keys=TEAM_MEMBERS_BY_KEY,
        static_alias_member_keys=STATIC_ALIAS_MEMBER_KEYS,
        expected_channel_guild_id=SKYVISION_GUILD_ID,
        static_channel_alias_ids=STATIC_ALIAS_CHANNEL_IDS,
    )
    run = validate_run_receipt(
        _json_file(
            PRODUCTION_RUN_RECEIPT_PATH,
            expected_uid=identities["projector"]["uid"],
            expected_gid=identities["gateway"]["gid"],
            expected_mode=0o640,
        )
    )
    projection_raw = _stable_read(
        PRODUCTION_PUBLIC_PROJECTION_PATH,
        expected_uid=identities["projector"]["uid"],
        expected_gid=identities["gateway"]["gid"],
        expected_mode=0o640,
    )
    if (
        projection["receipt"]["source_export_sha256"] != _sha256(export_raw)
        or run["source_export_sha256"] != _sha256(export_raw)
        or run["projection_sha256"] != projection["receipt"]["projection_sha256"]
        or run["projection_file_sha256"] != _sha256(projection_raw)
        or run["source_event_count"]
        != projection["receipt"]["source_event_count"]
        or run["alias_event_count"]
        != projection["receipt"]["alias_event_count"]
        or run["alias_count"] != projection["receipt"]["alias_count"]
        or run["last_alias_event_id"]
        != projection["receipt"]["last_alias_event_id"]
        or run["last_alias_event_at"]
        != projection["receipt"]["last_alias_event_at"]
        or run["projection_path"] != str(PRODUCTION_PUBLIC_PROJECTION_PATH)
        or run["source_export_path"] != str(PRODUCTION_WRITER_EXPORT_PATH)
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_activation_readback_invalid"
        )
    return {
        "source_export_sha256": run["source_export_sha256"],
        "projection_sha256": run["projection_sha256"],
        "projection_file_sha256": run["projection_file_sha256"],
        "run_receipt_sha256": run["receipt_sha256"],
        "source_event_count": run["source_event_count"],
        "alias_event_count": run["alias_event_count"],
        "alias_count": run["alias_count"],
    }


def activate(
    *,
    cutover_plan_sha256: str,
    package_root: Path,
    expected_revision: str,
    expected_package_sha256: str,
    expected_preflight_receipt_sha256: str,
    expected_apply_receipt_sha256: str,
    expected_postflight_receipt_sha256: str,
    expected_activation_authority_sha256: str,
    evidence_root: Path = EVIDENCE_ROOT,
    activation_authority_path: Path = STAGED_ACTIVATION_AUTHORITY_PATH,
    systemd: SystemdOperations | None = None,
    identities: IdentityOperations | None = None,
    clock: Callable[[], str] = _now,
    require_root: bool = True,
    enforce_production_address: bool = True,
    enforce_package_metadata: bool = True,
) -> dict[str, Any]:
    """Run one receipt-backed projection, then enable the recurring timer."""

    _require_root(require_root)
    systemd = systemd or ProductionSystemdOperations()
    identities = identities or ProductionIdentityOperations()
    context = load_runtime_context(
        package_root=package_root,
        expected_revision=expected_revision,
        expected_package_sha256=expected_package_sha256,
        enforce_production_address=enforce_production_address,
        enforce_package_metadata=enforce_package_metadata,
    )
    pre = _load_prior_with_chain(
        action="preflight",
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=context.package_sha256,
        expected_sha256=expected_preflight_receipt_sha256,
        expected_prior_sha256=None,
        evidence_root=evidence_root,
    )
    applied = _load_prior_with_chain(
        action="apply",
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=context.package_sha256,
        expected_sha256=expected_apply_receipt_sha256,
        expected_prior_sha256=pre["receipt_sha256"],
        evidence_root=evidence_root,
    )
    post = _load_prior_with_chain(
        action="postflight",
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=context.package_sha256,
        expected_sha256=expected_postflight_receipt_sha256,
        expected_prior_sha256=applied["receipt_sha256"],
        evidence_root=evidence_root,
    )
    authority = validate_activation_authority(
        _json_file(
            activation_authority_path,
            expected_uid=0,
            expected_gid=0,
            expected_mode=0o400,
        ),
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=context.package_sha256,
        postflight_receipt_sha256=post["receipt_sha256"],
        expected_authority_sha256=expected_activation_authority_sha256,
    )
    existing_path = _receipt_path(
        "activation", cutover_plan_sha256, evidence_root=evidence_root
    )
    if existing_path.exists() and not existing_path.is_symlink():
        states = _unit_states(systemd)
        if not states[PROJECTOR_TIMER].active or not states[PROJECTOR_TIMER].enabled:
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_activation_replay_state_drifted"
            )
        _projection_run_readback(context)
        return validate_cutover_receipt(
            _json_file(existing_path),
            action="activation",
            cutover_plan_sha256=cutover_plan_sha256,
            package_sha256=context.package_sha256,
            expected_prior_sha256=post["receipt_sha256"],
        )
    if not _core_services_active(systemd):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_activation_core_not_ready"
        )
    if not _installed_units_exact(
        context,
        systemd=systemd,
        require_disabled=False,
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_activation_unit_state_invalid"
        )
    validate_host_dependencies(
        context,
        identities=identities,
        require_directories=True,
    )
    timer_state = systemd.state(PROJECTOR_TIMER)
    if timer_state.enabled or timer_state.active:
        readback = _projection_run_readback(context)
    else:
        systemd.start(PROJECTOR_UNIT)
        exporter_state = systemd.state(EXPORTER_UNIT)
        projector_state = systemd.state(PROJECTOR_UNIT)
        if (
            exporter_state.active
            or projector_state.active
            or exporter_state.result != "success"
            or projector_state.result != "success"
        ):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_initial_run_failed"
            )
        readback = _projection_run_readback(context)
        systemd.enable_now(PROJECTOR_TIMER)
    final_timer = systemd.state(PROJECTOR_TIMER)
    if (
        not final_timer.active
        or not final_timer.enabled
        or final_timer.fragment_path != str(SYSTEMD_ROOT / PROJECTOR_TIMER)
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_timer_activation_failed"
        )
    return _publish_receipt(
        action="activation",
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=context.package_sha256,
        prior_receipt_sha256=post["receipt_sha256"],
        evidence={
            "activation_authority_sha256": authority["authority_sha256"],
            "writer_ready_entry_sha256": authority["writer_ready_entry_sha256"],
            "gateway_started_entry_sha256": authority["gateway_started_entry_sha256"],
            "initial_projection_readback": dict(readback),
            "timer_enabled": True,
            "timer_active": True,
            "forward_recovery_only": True,
        },
        evidence_root=evidence_root,
        clock=clock,
    )


def _restore_file(path: Path, state: Mapping[str, Any]) -> None:
    if state.get("state") == "absent":
        if path.exists() or path.is_symlink():
            item = path.lstat()
            if stat.S_ISLNK(item.st_mode) or not stat.S_ISREG(item.st_mode):
                raise ProductionAliasProjectionCutoverError(
                    "alias_projection_rollback_file_untrusted"
                )
            path.unlink()
        return
    if state.get("state") != "present":
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_rollback_prestate_invalid"
        )
    try:
        raw = base64.b64decode(state["bytes_b64"], validate=True)
    except (KeyError, ValueError, TypeError) as exc:
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_rollback_prestate_invalid"
        ) from exc
    if _sha256(raw) != state.get("sha256"):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_rollback_prestate_invalid"
        )
    _atomic_write(
        path,
        raw,
        mode=int(state["mode"]),
        uid=int(state["uid"]),
        gid=int(state["gid"]),
    )


def _restore_directories(prepared: Mapping[str, Any]) -> None:
    directories = prepared["directories"]
    for path_text in sorted(
        directories, key=lambda value: len(Path(value).parts), reverse=True
    ):
        path = Path(path_text)
        state = directories[path_text]
        if state.get("state") == "absent":
            if path.exists() or path.is_symlink():
                item = path.lstat()
                if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
                    raise ProductionAliasProjectionCutoverError(
                        "alias_projection_rollback_directory_untrusted"
                    )
                try:
                    path.rmdir()
                except OSError as exc:
                    raise ProductionAliasProjectionCutoverError(
                        "alias_projection_rollback_directory_not_empty"
                    ) from exc
        elif state.get("state") == "present":
            item = path.lstat()
            if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
                raise ProductionAliasProjectionCutoverError(
                    "alias_projection_rollback_directory_untrusted"
                )
            os.chown(path, int(state["uid"]), int(state["gid"]))
            os.chmod(path, int(state["mode"]))
        else:
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_rollback_prestate_invalid"
            )


def rollback(
    *,
    cutover_plan_sha256: str,
    package_root: Path,
    expected_revision: str,
    expected_package_sha256: str,
    expected_preflight_receipt_sha256: str,
    expected_apply_receipt_sha256: str,
    evidence_root: Path = EVIDENCE_ROOT,
    activation_authority_path: Path = STAGED_ACTIVATION_AUTHORITY_PATH,
    systemd: SystemdOperations | None = None,
    clock: Callable[[], str] = _now,
    require_root: bool = True,
    enforce_production_address: bool = True,
    enforce_package_metadata: bool = True,
) -> dict[str, Any]:
    """Restore prestate only before terminal activation authority exists."""

    _require_root(require_root)
    systemd = systemd or ProductionSystemdOperations()
    context = load_runtime_context(
        package_root=package_root,
        expected_revision=expected_revision,
        expected_package_sha256=expected_package_sha256,
        enforce_production_address=enforce_production_address,
        enforce_package_metadata=enforce_package_metadata,
    )
    pre = _load_prior_with_chain(
        action="preflight",
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=context.package_sha256,
        expected_sha256=expected_preflight_receipt_sha256,
        expected_prior_sha256=None,
        evidence_root=evidence_root,
    )
    applied = _load_prior_with_chain(
        action="apply",
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=context.package_sha256,
        expected_sha256=expected_apply_receipt_sha256,
        expected_prior_sha256=pre["receipt_sha256"],
        evidence_root=evidence_root,
    )
    existing_path = _receipt_path(
        "rollback", cutover_plan_sha256, evidence_root=evidence_root
    )
    if existing_path.exists() and not existing_path.is_symlink():
        return validate_cutover_receipt(
            _json_file(existing_path),
            action="rollback",
            cutover_plan_sha256=cutover_plan_sha256,
            package_sha256=context.package_sha256,
            expected_prior_sha256=applied["receipt_sha256"],
        )
    if (
        activation_authority_path.exists()
        or activation_authority_path.is_symlink()
        or _receipt_path(
            "activation", cutover_plan_sha256, evidence_root=evidence_root
        ).exists()
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_rollback_after_terminal_forbidden"
        )
    if not _core_services_stopped(systemd):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_rollback_services_not_stopped"
        )
    prepared = _load_prepared(
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=context.package_sha256,
        evidence_root=evidence_root,
    )
    if not _data_files_unchanged(prepared):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_rollback_unexpected_data_mutation"
        )
    systemd.disable_now(PROJECTOR_TIMER)
    systemd.stop(PROJECTOR_UNIT)
    systemd.stop(EXPORTER_UNIT)
    if not _installed_units_exact(
        context,
        systemd=systemd,
        require_disabled=False,
    ):
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_rollback_unit_state_drifted"
        )
    for name, state in prepared["units"].items():
        _restore_file(SYSTEMD_ROOT / name, state["file"])
    systemd.daemon_reload()
    _restore_directories(prepared)
    states = _unit_states(systemd)
    for name, before in prepared["units"].items():
        current = states[name]
        if (
            current.active_state != before["active_state"]
            or current.unit_file_state != before["unit_file_state"]
        ):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_rollback_readback_failed"
            )
    return _publish_receipt(
        action="rollback",
        cutover_plan_sha256=cutover_plan_sha256,
        package_sha256=context.package_sha256,
        prior_receipt_sha256=applied["receipt_sha256"],
        evidence={
            "prepared_sha256": prepared["prepared_sha256"],
            "unit_prestate_restored": True,
            "directory_prestate_restored": True,
            "data_files_unchanged": True,
            "timer_disabled": True,
            "terminal_authority_consumed": False,
        },
        evidence_root=evidence_root,
        clock=clock,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("preflight", "apply", "postflight", "activate", "rollback")
    )
    parser.add_argument("--expected-cutover-plan-sha256", required=True)
    parser.add_argument("--expected-release-revision", required=True)
    parser.add_argument("--expected-package-sha256", required=True)
    parser.add_argument("--expected-preflight-receipt-sha256")
    parser.add_argument("--expected-apply-receipt-sha256")
    parser.add_argument("--expected-postflight-receipt-sha256")
    parser.add_argument("--expected-activation-authority-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    revision = args.expected_release_revision
    if re.fullmatch(r"[0-9a-f]{40}", revision or "") is None:
        raise ProductionAliasProjectionCutoverError(
            "alias_projection_cutover_revision_invalid"
        )
    package_root = (
        Path("/opt/adventico-ai-platform/hermes-agent-releases")
        / f"hermes-agent-{revision[:12]}"
        / PACKAGE_RELATIVE_ROOT
    )
    common = {
        "cutover_plan_sha256": args.expected_cutover_plan_sha256,
        "package_root": package_root,
        "expected_revision": revision,
        "expected_package_sha256": args.expected_package_sha256,
    }
    if args.action == "preflight":
        if any(
            value is not None
            for value in (
                args.expected_preflight_receipt_sha256,
                args.expected_apply_receipt_sha256,
                args.expected_postflight_receipt_sha256,
                args.expected_activation_authority_sha256,
            )
        ):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_cutover_prior_receipt_unexpected"
            )
        result = preflight(**common)
    elif args.action == "apply":
        if (
            not args.expected_preflight_receipt_sha256
            or args.expected_apply_receipt_sha256 is not None
            or args.expected_postflight_receipt_sha256 is not None
            or args.expected_activation_authority_sha256 is not None
        ):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_cutover_prior_receipt_invalid"
            )
        result = apply(
            **common,
            expected_preflight_receipt_sha256=(
                args.expected_preflight_receipt_sha256
            ),
        )
    elif args.action == "postflight":
        if (
            not args.expected_preflight_receipt_sha256
            or not args.expected_apply_receipt_sha256
            or args.expected_postflight_receipt_sha256 is not None
            or args.expected_activation_authority_sha256 is not None
        ):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_cutover_prior_receipt_invalid"
            )
        result = postflight(
            **common,
            expected_preflight_receipt_sha256=(
                args.expected_preflight_receipt_sha256
            ),
            expected_apply_receipt_sha256=args.expected_apply_receipt_sha256,
        )
    elif args.action == "activate":
        if any(
            not value
            for value in (
                args.expected_preflight_receipt_sha256,
                args.expected_apply_receipt_sha256,
                args.expected_postflight_receipt_sha256,
                args.expected_activation_authority_sha256,
            )
        ):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_cutover_prior_receipt_invalid"
            )
        result = activate(
            **common,
            expected_preflight_receipt_sha256=(
                args.expected_preflight_receipt_sha256
            ),
            expected_apply_receipt_sha256=args.expected_apply_receipt_sha256,
            expected_postflight_receipt_sha256=(
                args.expected_postflight_receipt_sha256
            ),
            expected_activation_authority_sha256=(
                args.expected_activation_authority_sha256
            ),
        )
    else:
        if (
            not args.expected_preflight_receipt_sha256
            or not args.expected_apply_receipt_sha256
            or args.expected_postflight_receipt_sha256 is not None
            or args.expected_activation_authority_sha256 is not None
        ):
            raise ProductionAliasProjectionCutoverError(
                "alias_projection_cutover_prior_receipt_invalid"
            )
        result = rollback(
            **common,
            expected_preflight_receipt_sha256=(
                args.expected_preflight_receipt_sha256
            ),
            expected_apply_receipt_sha256=args.expected_apply_receipt_sha256,
        )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "ACTIVATION_AUTHORITY_SCHEMA",
    "ProductionAliasProjectionCutoverError",
    "ProductionIdentityOperations",
    "ProductionSystemdOperations",
    "RuntimeContext",
    "SystemdOperations",
    "UnitState",
    "activate",
    "apply",
    "build_activation_authority",
    "load_runtime_context",
    "main",
    "postflight",
    "preflight",
    "rollback",
    "validate_activation_authority",
    "validate_cutover_receipt",
    "validate_host_dependencies",
]


if __name__ == "__main__":
    raise SystemExit(main())
