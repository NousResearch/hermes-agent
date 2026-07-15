#!/usr/bin/env python3
"""Stage and attest the stopped writer-only preflight boundary.

This packaged root entry point is deliberately mechanical.  It joins one
sealed stopped-release receipt, fixed canary database coordinates, the trusted
config collector, and the native observation planner.  It never installs a
systemd unit or live config, reloads systemd, starts a service, creates an
approval, grants IAM authority, or interprets user text.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gateway.canonical_canary_host_identity import (
    FULL_CANARY_HOST_IDENTITY_SCHEMA,
)
from gateway.canonical_writer_activation import (
    CANARY_WRITER_GID,
    DEFAULT_GATEWAY_CONFIG_PATH,
    DEFAULT_GATEWAY_CONFIG_SOURCE_PATH,
    DEFAULT_GATEWAY_UNIT_PATH,
    DEFAULT_NATIVE_PLAN_PATH,
    DEFAULT_PLAN_PATH,
    DEFAULT_STAGED_EXTERNAL_IAM_PATH,
    DEFAULT_STAGED_GATEWAY_UNIT_PATH,
    DEFAULT_STAGED_NATIVE_PLAN_PATH,
    DEFAULT_STAGED_OWNER_APPROVAL_PATH,
    DEFAULT_STAGED_PLAN_PATH,
    DEFAULT_STAGED_WRITER_UNIT_PATH,
    DEFAULT_WRITER_CONFIG_PATH,
    DEFAULT_WRITER_CONFIG_SOURCE_PATH,
    DEFAULT_WRITER_UNIT_PATH,
    DISCORD_UNIT,
    EXPORTER_UNIT,
    GATEWAY_UNIT,
    WRITER_UNIT,
    NativeObservationPlan,
    Runner,
    _current_boot_id_sha256,
    _decode_strict_json,
    _ensure_root_directory,
    _fsync_directory,
    _host_activation_lock,
    _install_exact_bytes,
    _list_xattrs,
    _off_state_is_exact,
    _read_trusted_file,
    _require_root_linux,
    _runner,
    _systemd_show,
    _unlink_exact,
    _verify_native_release,
    current_host_identity_sha256,
    native_observation_read_only_preflight,
)
from gateway.canonical_writer_config_collector import (
    DATABASE_CA_PATH,
    DATABASE_CREDENTIAL_PATH,
    EVIDENCE_ROOT as CONFIG_COLLECTOR_EVIDENCE_ROOT,
    SQL_DATABASE,
    SQL_PORT,
    SQL_PRIVATE_IP,
    SQL_USER,
    ConfigCollectorReceipt,
    _credential_identity,
    collect_and_stage,
    load_config_collector_receipt,
)
from gateway.canonical_writer_planner import (
    build_and_stage_native_observation_plan,
    load_release_manifest,
)
from gateway.canonical_writer_release_contract import (
    DEFAULT_RELEASE_BASE,
    WriterOnlyUnitSpec,
    render_systemd_units,
)


PUBLICATION_PLAN_SCHEMA = "muncho-writer-preflight-publication-plan.v2"
PUBLICATION_RECEIPT_SCHEMA = "muncho-writer-preflight-publication.v3"
PUBLICATION_FAILURE_SCHEMA = "muncho-writer-preflight-publication-failure.v2"
STOPPED_RELEASE_RECEIPT_SCHEMA = "muncho-canary-stopped-release-publication.v1"

OWNER_DISCORD_USER_ID = "1279454038731264061"
DATABASE_TLS_SERVER_NAME = (
    "14-0d81ef63-2cac-4a64-84ad-c4f58c0cfd56.europe-west3.sql.goog"
)
STOPPED_RELEASE_EVIDENCE_BASE = Path("/var/lib/muncho-canary-release-evidence")
HOST_IDENTITY_RECEIPT_PATH = Path("/etc/muncho/full-canary/host-identity.json")
PUBLICATION_EVIDENCE_ROOT = Path(
    "/var/lib/muncho-writer-canary-evidence/staged-publication"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MAX_PUBLIC_JSON_BYTES = 4 * 1024 * 1024
_MAX_UNIT_BYTES = 256 * 1024

_SERVICE_UNITS = (
    WRITER_UNIT,
    GATEWAY_UNIT,
    EXPORTER_UNIT,
    DISCORD_UNIT,
)
_SYSTEMD_STATE_FIELDS = frozenset({
    "LoadState",
    "ActiveState",
    "SubState",
    "MainPID",
    "UnitFileState",
    "FragmentPath",
    "DropInPaths",
    "NeedDaemonReload",
})
_NATIVE_PREFLIGHT_REPORT_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "revision",
    "native_observation_plan_sha256",
    "release_artifact_sha256",
    "release_manifest_file_sha256",
    "config_collector_receipt_sha256",
    "external_iam_policy_sha256",
    "host_identity_sha256",
    "boot_id_sha256",
    "collector_hba_observed_at_unix",
    "collector_collected_at_unix",
    "observed_at_unix",
    "collector_hba_expires_at_unix",
    "services_started",
    "units_installed",
    "daemon_reloaded",
    "discord_started",
    "approval_created",
    "credential_content_or_digest_recorded",
    "report_sha256",
})

_STOPPED_RELEASE_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "release_revision",
    "plan_sha256",
    "source",
    "dedicated_host",
    "activation_inventory",
    "service_state_before",
    "service_state_after",
    "services_stopped_and_disabled",
    "tools",
    "release_root",
    "release_manifest_path",
    "release_manifest_file_sha256",
    "release_artifact_sha256",
    "interpreter",
    "interpreter_sha256",
    "python_version",
    "retained_wheel_path",
    "retained_wheel_sha256",
    "build_constraints_sha256",
    "host_identity_receipt_path",
    "host_identity_receipt_file_sha256",
    "host_identity_receipt_sha256",
    "receipt_path",
    "created_at_unix",
    "receipt_sha256",
})
_HOST_RECEIPT_FIELDS = frozenset({
    "schema",
    "collector_authority",
    "project_id",
    "project_number",
    "zone",
    "instance_name",
    "instance_id",
    "service_account_email",
    "gce_identity_sha256",
    "machine_id_sha256",
    "hostname_sha256",
    "host_identity_sha256",
    "boot_id_sha256",
    "observed_at_unix",
    "receipt_sha256",
})
_HOST_OBSERVATION_FIELDS = _HOST_RECEIPT_FIELDS - {
    "schema",
    "collector_authority",
    "observed_at_unix",
    "receipt_sha256",
}
_PUBLICATION_RECEIPT_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "revision",
    "approved_plan_sha256",
    "stopped_release_receipt_sha256",
    "release_artifact_sha256",
    "release_manifest_file_sha256",
    "host_identity_receipt_sha256",
    "config_collector_receipt_path",
    "config_collector_receipt_sha256",
    "config_collector_receipt_file_sha256",
    "native_observation_plan_sha256",
    "external_iam_policy_sha256",
    "preflight_report_path",
    "preflight_report_file_sha256",
    "preflight_report_sha256",
    "preflight_observed_at_unix",
    "preflight_collector_hba_observed_at_unix",
    "preflight_collector_collected_at_unix",
    "preflight_collector_hba_expires_at_unix",
    "preflight_time_envelope_sha256",
    "preflight_fresh_at_seal",
    "service_state_before",
    "service_state_after",
    "artifacts",
    "provenance",
    "invariants",
    "sealed_at_unix",
    "receipt_path",
    "receipt_sha256",
})

_PUBLICATION_PROVENANCE_FIELDS = frozenset({
    "approved_plan_sha256",
    "release_artifact_sha256",
    "release_manifest_file_sha256",
    "database_ca_sha256",
    "config_collector_receipt_sha256",
    "config_collector_receipt_file_sha256",
    "collector_writer_config_sha256",
    "collector_gateway_config_sha256",
    "native_observation_plan_sha256",
    "native_writer_config_sha256",
    "native_gateway_config_sha256",
    "native_writer_unit_sha256",
    "native_gateway_unit_sha256",
    "preflight_report_sha256",
    "preflight_report_file_sha256",
    "preflight_time_envelope_sha256",
})

_ACTIVATION_PATHS = (
    DEFAULT_WRITER_CONFIG_SOURCE_PATH,
    DEFAULT_GATEWAY_CONFIG_SOURCE_PATH,
    DEFAULT_STAGED_NATIVE_PLAN_PATH,
    DEFAULT_STAGED_PLAN_PATH,
    DEFAULT_STAGED_OWNER_APPROVAL_PATH,
    DEFAULT_STAGED_EXTERNAL_IAM_PATH,
    DEFAULT_STAGED_WRITER_UNIT_PATH,
    DEFAULT_STAGED_GATEWAY_UNIT_PATH,
    DEFAULT_NATIVE_PLAN_PATH,
    DEFAULT_PLAN_PATH,
    Path("/etc/muncho/writer-activation/deployment-manifest.json"),
    DEFAULT_WRITER_UNIT_PATH,
    DEFAULT_GATEWAY_UNIT_PATH,
    Path("/etc/systemd/system/muncho-canonical-writer-export.service"),
    Path("/etc/tmpfiles.d/muncho-canonical-writer.conf"),
    DEFAULT_WRITER_CONFIG_PATH,
    DEFAULT_GATEWAY_CONFIG_PATH,
)
_STAGED_OUTPUTS = frozenset({
    DEFAULT_WRITER_CONFIG_SOURCE_PATH,
    DEFAULT_GATEWAY_CONFIG_SOURCE_PATH,
    DEFAULT_STAGED_NATIVE_PLAN_PATH,
    DEFAULT_STAGED_WRITER_UNIT_PATH,
    DEFAULT_STAGED_GATEWAY_UNIT_PATH,
})
_FORBIDDEN_OUTPUTS = tuple(
    path for path in _ACTIVATION_PATHS if path not in _STAGED_OUTPUTS
)


def _canonical_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("publisher value is not canonical JSON") from exc
    return rendered.encode("utf-8", errors="strict")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _strict_mapping(raw: bytes, *, label: str, newline: bool) -> Mapping[str, Any]:
    candidate = raw[:-1] if newline and raw.endswith(b"\n") else raw
    if newline and (not raw.endswith(b"\n") or b"\n" in raw[:-1]):
        raise ValueError(f"{label} framing is invalid")
    try:
        value = json.loads(
            candidate.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-JSON constant:{item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping) or candidate != _canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("publisher JSON contains duplicate keys")
        result[key] = value
    return result


def _stopped_release_receipt_path(revision: str) -> Path:
    if _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("publisher revision is invalid")
    return STOPPED_RELEASE_EVIDENCE_BASE / revision / "stopped-release-publication.json"


def _load_stopped_release_receipt(revision: str) -> tuple[Mapping[str, Any], bytes]:
    path = _stopped_release_receipt_path(revision)
    raw = _read_trusted_file(
        path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_PUBLIC_JSON_BYTES,
    )
    value = _strict_mapping(raw, label="stopped release receipt", newline=True)
    if (
        set(value) != _STOPPED_RELEASE_FIELDS
        or value.get("schema") != STOPPED_RELEASE_RECEIPT_SCHEMA
        or value.get("ok") is not True
        or value.get("state") != "published_services_stopped"
        or value.get("release_revision") != revision
        or value.get("services_stopped_and_disabled") is not True
        or value.get("receipt_path") != str(path)
    ):
        raise ValueError("stopped release receipt identity drifted")
    receipt_sha256 = _digest(
        value.get("receipt_sha256"),
        "stopped release receipt",
    )
    unsigned = {name: copy.deepcopy(item) for name, item in value.items()}
    del unsigned["receipt_sha256"]
    if receipt_sha256 != _sha256_json(unsigned):
        raise ValueError("stopped release receipt digest drifted")
    expected_inventory = [
        {"path": str(path), "state": "absent"} for path in _ACTIVATION_PATHS
    ]
    if value.get("activation_inventory") != expected_inventory:
        raise ValueError("stopped release activation inventory drifted")
    for name in (
        "release_manifest_file_sha256",
        "release_artifact_sha256",
        "host_identity_receipt_file_sha256",
        "host_identity_receipt_sha256",
    ):
        _digest(value.get(name), f"stopped release {name}")
    return value, raw


def _load_host_receipt(
    stopped: Mapping[str, Any],
) -> tuple[Mapping[str, Any], bytes]:
    raw = _read_trusted_file(
        HOST_IDENTITY_RECEIPT_PATH,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=16 * 1024,
    )
    value = _strict_mapping(raw, label="host identity receipt", newline=False)
    if (
        set(value) != _HOST_RECEIPT_FIELDS
        or value.get("schema") != FULL_CANARY_HOST_IDENTITY_SCHEMA
        or value.get("collector_authority") != "trusted_root_read_only_host_collector"
        or stopped.get("host_identity_receipt_path") != str(HOST_IDENTITY_RECEIPT_PATH)
        or stopped.get("host_identity_receipt_file_sha256") != _sha256_bytes(raw)
        or stopped.get("host_identity_receipt_sha256") != value.get("receipt_sha256")
    ):
        raise ValueError("host identity receipt binding drifted")
    unsigned = {name: copy.deepcopy(item) for name, item in value.items()}
    receipt_sha256 = _digest(
        unsigned.pop("receipt_sha256", None),
        "host identity receipt",
    )
    if receipt_sha256 != _sha256_json(unsigned):
        raise ValueError("host identity receipt digest drifted")
    dedicated_host = stopped.get("dedicated_host")
    if (
        not isinstance(dedicated_host, Mapping)
        or set(dedicated_host) != _HOST_OBSERVATION_FIELDS
        or any(
            dedicated_host.get(name) != value.get(name)
            for name in _HOST_OBSERVATION_FIELDS
        )
    ):
        raise ValueError("stopped release and host receipt identity drifted")
    if (
        value.get("host_identity_sha256") != current_host_identity_sha256()
        or value.get("boot_id_sha256") != _current_boot_id_sha256()
    ):
        raise RuntimeError("writer preflight host identity drifted")
    return value, raw


def _validate_service_snapshot(value: Any) -> Mapping[str, Mapping[str, str]]:
    if not isinstance(value, Mapping) or set(value) != set(_SERVICE_UNITS):
        raise RuntimeError("writer preflight service snapshot units are not exact")
    result: dict[str, dict[str, str]] = {}
    for unit in _SERVICE_UNITS:
        state = value.get(unit)
        if (
            not isinstance(state, Mapping)
            or set(state) != _SYSTEMD_STATE_FIELDS
            or any(not isinstance(item, str) for item in state.values())
        ):
            raise RuntimeError("writer preflight service snapshot fields are not exact")
        exact = (
            _off_state_is_exact(unit, state, absent=False)
            or _off_state_is_exact(unit, state, absent=True)
            if unit in {WRITER_UNIT, GATEWAY_UNIT}
            else _off_state_is_exact(unit, state, absent=True)
        )
        if not exact:
            raise RuntimeError(f"{unit} is not exact stopped/disabled or absent state")
        result[unit] = {name: str(state[name]) for name in sorted(state)}
    return json.loads(_canonical_bytes(result).decode("utf-8"))


def _capture_service_snapshot(*, runner: Runner) -> Mapping[str, Mapping[str, str]]:
    return _validate_service_snapshot({
        unit: _systemd_show(unit, runner=runner)
        for unit in _SERVICE_UNITS
    })


def _require_services_stopped(*, runner: Runner) -> None:
    _capture_service_snapshot(runner=runner)


def _require_no_downstream_mutation() -> None:
    collisions = [str(path) for path in _FORBIDDEN_OUTPUTS if os.path.lexists(path)]
    if collisions:
        raise RuntimeError("writer preflight downstream activation collision")


def plan_writer_preflight_publication(
    *,
    revision: str,
    external_iam_policy_sha256: str,
    _service_runner: Runner = _runner,
) -> Mapping[str, Any]:
    """Build the exact owner-reviewable staging envelope without writes."""

    _require_root_linux()
    if _REVISION_RE.fullmatch(revision) is None:
        raise ValueError("publisher revision is invalid")
    external_digest = _digest(
        external_iam_policy_sha256,
        "publisher external IAM policy",
    )
    stopped, stopped_raw = _load_stopped_release_receipt(revision)
    host, _host_raw = _load_host_receipt(stopped)
    release, manifest_raw = load_release_manifest(revision)
    release_root = DEFAULT_RELEASE_BASE / revision
    if (
        stopped.get("release_root") != str(release_root)
        or stopped.get("release_manifest_path")
        != str(release_root / "release-manifest.json")
        or stopped.get("release_manifest_file_sha256") != _sha256_bytes(manifest_raw)
        or stopped.get("release_artifact_sha256") != release.artifact_sha256
    ):
        raise ValueError("publisher stopped release binding drifted")
    ca_raw = _read_trusted_file(
        DATABASE_CA_PATH,
        expected_uid=0,
        expected_gid=CANARY_WRITER_GID,
        allowed_modes=frozenset({0o440}),
        maximum=2 * 1024 * 1024,
    )
    _credential_stat, credential = _credential_identity()
    service_state = _capture_service_snapshot(runner=_service_runner)
    _require_no_downstream_mutation()
    unsigned: dict[str, Any] = {
        "schema": PUBLICATION_PLAN_SCHEMA,
        "revision": revision,
        "stopped_release_receipt_path": str(_stopped_release_receipt_path(revision)),
        "stopped_release_receipt_file_sha256": _sha256_bytes(stopped_raw),
        "stopped_release_receipt_sha256": stopped["receipt_sha256"],
        "release_root": str(release_root),
        "release_artifact_sha256": release.artifact_sha256,
        "release_manifest_path": str(release_root / "release-manifest.json"),
        "release_manifest_file_sha256": _sha256_bytes(manifest_raw),
        "host_identity_receipt_path": str(HOST_IDENTITY_RECEIPT_PATH),
        "host_identity_receipt_file_sha256": stopped[
            "host_identity_receipt_file_sha256"
        ],
        "host_identity_receipt_sha256": host["receipt_sha256"],
        "host_identity_sha256": host["host_identity_sha256"],
        "boot_id_sha256": host["boot_id_sha256"],
        "database": {
            "host": SQL_PRIVATE_IP,
            "port": SQL_PORT,
            "database": SQL_DATABASE,
            "user": SQL_USER,
            "tls_server_name": DATABASE_TLS_SERVER_NAME,
            "ca_path": str(DATABASE_CA_PATH),
            "ca_sha256": _sha256_bytes(ca_raw),
        },
        "credential_provenance": credential,
        "owner_discord_user_ids": [OWNER_DISCORD_USER_ID],
        "external_iam_policy_sha256": external_digest,
        "service_state": service_state,
        "fixed_output_paths": {
            "writer_config": str(DEFAULT_WRITER_CONFIG_SOURCE_PATH),
            "gateway_config": str(DEFAULT_GATEWAY_CONFIG_SOURCE_PATH),
            "writer_unit": str(DEFAULT_STAGED_WRITER_UNIT_PATH),
            "gateway_unit": str(DEFAULT_STAGED_GATEWAY_UNIT_PATH),
            "native_observation_plan": str(DEFAULT_STAGED_NATIVE_PLAN_PATH),
            "publication_evidence_root": str(PUBLICATION_EVIDENCE_ROOT),
        },
        "invariants": {
            "services_started": False,
            "units_installed": False,
            "daemon_reloaded": False,
            "approval_created": False,
            "discord_started": False,
            "credential_content_or_digest_recorded": False,
        },
    }
    return {**unsigned, "plan_sha256": _sha256_json(unsigned)}


def _trusted_staged_bytes(path: Path, *, maximum: int) -> bytes:
    return _read_trusted_file(
        path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=maximum,
    )


def _matching_collector_receipts(
    *,
    plan: Mapping[str, Any],
    require_fresh: bool,
    now_unix: int | None = None,
) -> list[ConfigCollectorReceipt]:
    writer_exists = os.path.lexists(DEFAULT_WRITER_CONFIG_SOURCE_PATH)
    gateway_exists = os.path.lexists(DEFAULT_GATEWAY_CONFIG_SOURCE_PATH)
    if not writer_exists or not gateway_exists:
        return []
    writer_sha = _sha256_bytes(
        _trusted_staged_bytes(
            DEFAULT_WRITER_CONFIG_SOURCE_PATH,
            maximum=2 * 1024 * 1024,
        )
    )
    gateway_sha = _sha256_bytes(
        _trusted_staged_bytes(
            DEFAULT_GATEWAY_CONFIG_SOURCE_PATH,
            maximum=2 * 1024 * 1024,
        )
    )
    directory = CONFIG_COLLECTOR_EVIDENCE_ROOT / str(plan["revision"])
    if not os.path.lexists(directory):
        return []
    candidates: list[ConfigCollectorReceipt] = []
    for name in sorted(os.listdir(directory)):
        match = re.fullmatch(r"([0-9a-f]{64})\.json", name)
        if match is None:
            raise RuntimeError("collector evidence namespace contains an extra entry")
        receipt = load_config_collector_receipt(
            revision=str(plan["revision"]),
            receipt_sha256=match.group(1),
            require_fresh=False,
        )
        if (
            receipt.value["writer_config_sha256"] == writer_sha
            and receipt.value["gateway_config_sha256"] == gateway_sha
            and receipt.value["release_artifact_sha256"]
            == plan["release_artifact_sha256"]
            and receipt.value["release_manifest_file_sha256"]
            == plan["release_manifest_file_sha256"]
            and receipt.value["database"] == plan["database"]
            and receipt.value["credential_provenance"] == plan["credential_provenance"]
        ):
            if require_fresh:
                try:
                    receipt.require_fresh(
                        int(time.time()) if now_unix is None else now_unix
                    )
                except ValueError:
                    continue
            candidates.append(receipt)
    return candidates


def _matching_collector_receipt(
    *,
    plan: Mapping[str, Any],
    require_fresh: bool,
    now_unix: int | None = None,
) -> ConfigCollectorReceipt | None:
    candidates = _matching_collector_receipts(
        plan=plan,
        require_fresh=require_fresh,
        now_unix=now_unix,
    )
    if len(candidates) > 1:
        raise RuntimeError("multiple collector receipts bind the staged configs")
    return candidates[0] if candidates else None


def _collect_or_resume_configs(
    plan: Mapping[str, Any],
    *,
    now_unix: int,
    clock: Callable[[], float],
) -> ConfigCollectorReceipt:
    if type(now_unix) is not int or now_unix < 0:
        raise ValueError("publisher collector freshness time is invalid")
    writer_exists = os.path.lexists(DEFAULT_WRITER_CONFIG_SOURCE_PATH)
    gateway_exists = os.path.lexists(DEFAULT_GATEWAY_CONFIG_SOURCE_PATH)
    if writer_exists != gateway_exists:
        raise RuntimeError("collector staged config residue is partial")
    existing_plan_outputs = any(
        os.path.lexists(path)
        for path in (
            DEFAULT_STAGED_WRITER_UNIT_PATH,
            DEFAULT_STAGED_GATEWAY_UNIT_PATH,
            DEFAULT_STAGED_NATIVE_PLAN_PATH,
        )
    )
    receipt = _matching_collector_receipt(
        plan=plan,
        require_fresh=True,
        now_unix=now_unix,
    )
    if receipt is not None:
        return receipt
    if existing_plan_outputs:
        _rollback_exact_stale_planner_residue(plan)
    result = collect_and_stage(
        revision=str(plan["revision"]),
        release_artifact_sha256=str(plan["release_artifact_sha256"]),
        release_manifest_file_sha256=str(plan["release_manifest_file_sha256"]),
        tls_server_name=DATABASE_TLS_SERVER_NAME,
        owner_discord_user_ids=(OWNER_DISCORD_USER_ID,),
        _clock=clock,
    )
    receipt = load_config_collector_receipt(
        revision=str(plan["revision"]),
        receipt_sha256=str(result["receipt_sha256"]),
        require_fresh=True,
        now_unix=int(clock()),
    )
    matched = _matching_collector_receipt(
        plan=plan,
        require_fresh=True,
        now_unix=int(clock()),
    )
    if matched is None or matched.to_mapping() != receipt.to_mapping():
        raise RuntimeError("collector result did not bind the staged configs")
    return receipt


def _expected_unit_bytes(revision: str) -> Mapping[Path, bytes]:
    release, _manifest_raw = load_release_manifest(revision)
    bundle = render_systemd_units(
        release,
        WriterOnlyUnitSpec(database_ip_allow=(f"{SQL_PRIVATE_IP}/32",)),
    )
    return {
        DEFAULT_STAGED_WRITER_UNIT_PATH: bundle.writer_service.encode("utf-8"),
        DEFAULT_STAGED_GATEWAY_UNIT_PATH: bundle.gateway_service.encode("utf-8"),
    }


def _collector_receipt_path(plan: Mapping[str, Any], receipt_sha256: str) -> Path:
    return (
        CONFIG_COLLECTOR_EVIDENCE_ROOT
        / str(plan["revision"])
        / f"{_digest(receipt_sha256, 'config collector receipt')}.json"
    )


def _load_bound_collector_receipt(
    plan: Mapping[str, Any],
    receipt_sha256: str,
    *,
    require_fresh: bool,
    now_unix: int | None = None,
) -> tuple[ConfigCollectorReceipt, str]:
    receipt = load_config_collector_receipt(
        revision=str(plan["revision"]),
        receipt_sha256=receipt_sha256,
        require_fresh=require_fresh,
        now_unix=now_unix,
    )
    writer_raw = _trusted_staged_bytes(
        DEFAULT_WRITER_CONFIG_SOURCE_PATH,
        maximum=2 * 1024 * 1024,
    )
    gateway_raw = _trusted_staged_bytes(
        DEFAULT_GATEWAY_CONFIG_SOURCE_PATH,
        maximum=2 * 1024 * 1024,
    )
    receipt.require_bindings(
        revision=str(plan["revision"]),
        release_artifact_sha256=str(plan["release_artifact_sha256"]),
        release_manifest_file_sha256=str(plan["release_manifest_file_sha256"]),
        writer_config_sha256=_sha256_bytes(writer_raw),
        gateway_config_sha256=_sha256_bytes(gateway_raw),
        database_ca_sha256=str(plan["database"]["ca_sha256"]),
        sql_private_ip=SQL_PRIVATE_IP,
        sql_tls_server_name=DATABASE_TLS_SERVER_NAME,
    )
    if (
        receipt.value["database"] != plan["database"]
        or receipt.value["credential_provenance"]
        != plan["credential_provenance"]
    ):
        raise RuntimeError("config collector receipt plan binding drifted")
    raw = _read_trusted_file(
        _collector_receipt_path(plan, receipt.sha256),
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_PUBLIC_JSON_BYTES,
    )
    if raw != _canonical_bytes(receipt.to_mapping()):
        raise RuntimeError("config collector receipt rotated during binding")
    return receipt, _sha256_bytes(raw)


def _load_staged_native_plan() -> NativeObservationPlan:
    raw = _trusted_staged_bytes(
        DEFAULT_STAGED_NATIVE_PLAN_PATH,
        maximum=_MAX_PUBLIC_JSON_BYTES,
    )
    native = NativeObservationPlan.from_mapping(
        _decode_strict_json(raw, label="staged native observation plan")
    )
    if raw != _canonical_bytes(native.to_mapping()) or _sha256_bytes(raw) != native.sha256:
        raise RuntimeError("staged native observation plan file digest drifted")
    return native


def _validate_native_binding(
    plan: Mapping[str, Any],
    native: NativeObservationPlan,
    receipt: ConfigCollectorReceipt,
    *,
    require_all_outputs: bool,
) -> None:
    if (
        native.value["revision"] != plan["revision"]
        or native.value["artifact_sha256"] != plan["release_artifact_sha256"]
        or native.value["release_manifest_file_sha256"]
        != plan["release_manifest_file_sha256"]
        or native.value["config_collector_receipt_sha256"] != receipt.sha256
        or native.value["external_iam_policy_sha256"]
        != plan["external_iam_policy_sha256"]
        or native.value["host_identity_sha256"] != plan["host_identity_sha256"]
        or native.value["boot_id_sha256"] != plan["boot_id_sha256"]
        or native.value["writer_config"]["sha256"]
        != receipt.value["writer_config_sha256"]
        or native.value["gateway_config"]["sha256"]
        != receipt.value["gateway_config_sha256"]
    ):
        raise RuntimeError("existing native plan binding drifted")
    expected_units = _expected_unit_bytes(str(plan["revision"]))
    expected_digests = {
        DEFAULT_STAGED_WRITER_UNIT_PATH: native.value["writer_unit"]["sha256"],
        DEFAULT_STAGED_GATEWAY_UNIT_PATH: native.value["gateway_unit"]["sha256"],
    }
    for path, payload in expected_units.items():
        if _sha256_bytes(payload) != expected_digests[path]:
            raise RuntimeError("existing native plan unit digest drifted")
        if os.path.lexists(path):
            if _trusted_staged_bytes(path, maximum=_MAX_UNIT_BYTES) != payload:
                raise RuntimeError("existing native plan unit content drifted")
        elif require_all_outputs:
            raise RuntimeError("existing native plan output is missing")


def _rollback_exact_stale_planner_residue(plan: Mapping[str, Any]) -> None:
    """Remove only exact same-revision planner residue before recollection."""

    native: NativeObservationPlan | None = None
    if os.path.lexists(DEFAULT_STAGED_NATIVE_PLAN_PATH):
        native = _load_staged_native_plan()
        stale, _file_sha256 = _load_bound_collector_receipt(
            plan,
            str(native.value["config_collector_receipt_sha256"]),
            require_fresh=False,
        )
        _validate_native_binding(
            plan,
            native,
            stale,
            require_all_outputs=False,
        )
    else:
        historical = _matching_collector_receipts(
            plan=plan,
            require_fresh=False,
        )
        if not historical:
            raise RuntimeError("planner residue lacks its exact collector receipt")
        for path, payload in _expected_unit_bytes(str(plan["revision"])).items():
            if os.path.lexists(path) and _trusted_staged_bytes(
                path,
                maximum=_MAX_UNIT_BYTES,
            ) != payload:
                raise RuntimeError("partial planner residue is not exact")

    expected_sha256 = {
        path: _sha256_bytes(payload)
        for path, payload in _expected_unit_bytes(str(plan["revision"])).items()
    }
    if native is not None:
        expected_sha256[DEFAULT_STAGED_NATIVE_PLAN_PATH] = native.sha256
    elif os.path.lexists(DEFAULT_STAGED_NATIVE_PLAN_PATH):
        raise RuntimeError("unbound native planner residue is forbidden")
    for path in (
        DEFAULT_STAGED_NATIVE_PLAN_PATH,
        DEFAULT_STAGED_GATEWAY_UNIT_PATH,
        DEFAULT_STAGED_WRITER_UNIT_PATH,
    ):
        if os.path.lexists(path):
            _unlink_exact(
                path,
                uid=0,
                gid=0,
                mode=0o400,
                sha256=expected_sha256[path],
            )


def _load_or_stage_native_plan(
    plan: Mapping[str, Any],
    receipt: ConfigCollectorReceipt,
) -> NativeObservationPlan:
    if os.path.lexists(DEFAULT_STAGED_NATIVE_PLAN_PATH):
        native = _load_staged_native_plan()
        _validate_native_binding(
            plan,
            native,
            receipt,
            require_all_outputs=False,
        )
        for path, payload in _expected_unit_bytes(str(plan["revision"])).items():
            _install_exact_bytes(path, payload, uid=0, gid=0, mode=0o400)
        return native
    build_and_stage_native_observation_plan(
        revision=str(plan["revision"]),
        external_iam_policy_sha256=str(plan["external_iam_policy_sha256"]),
        config_collector_receipt_sha256=receipt.sha256,
    )
    native = _load_staged_native_plan()
    _validate_native_binding(
        plan,
        native,
        receipt,
        require_all_outputs=True,
    )
    return native


def _publication_receipt_path(plan: Mapping[str, Any]) -> Path:
    return (
        PUBLICATION_EVIDENCE_ROOT
        / str(plan["revision"])
        / str(plan["plan_sha256"])
        / "publication.json"
    )


def _preflight_report_directory(plan: Mapping[str, Any]) -> Path:
    return (
        PUBLICATION_EVIDENCE_ROOT
        / str(plan["revision"])
        / str(plan["plan_sha256"])
        / "reports"
    )


def _validate_native_preflight_report(
    value: Any,
    *,
    plan: Mapping[str, Any],
    native: NativeObservationPlan,
    collector: ConfigCollectorReceipt,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _NATIVE_PREFLIGHT_REPORT_FIELDS:
        raise RuntimeError("native preflight report fields are not exact")
    report_sha256 = _digest(value.get("report_sha256"), "native preflight report")
    unsigned = {
        name: copy.deepcopy(item)
        for name, item in value.items()
        if name != "report_sha256"
    }
    expected = {
        "schema": "muncho-writer-native-read-only-preflight.v2",
        "ok": True,
        "state": "staged_inputs_verified_services_stopped",
        "revision": plan["revision"],
        "native_observation_plan_sha256": native.sha256,
        "release_artifact_sha256": plan["release_artifact_sha256"],
        "release_manifest_file_sha256": plan["release_manifest_file_sha256"],
        "config_collector_receipt_sha256": native.value[
            "config_collector_receipt_sha256"
        ],
        "external_iam_policy_sha256": plan["external_iam_policy_sha256"],
        "host_identity_sha256": plan["host_identity_sha256"],
        "boot_id_sha256": plan["boot_id_sha256"],
        "collector_hba_observed_at_unix": collector.value[
            "hba_observed_at_unix"
        ],
        "collector_collected_at_unix": collector.value["collected_at_unix"],
        "observed_at_unix": value.get("observed_at_unix"),
        "collector_hba_expires_at_unix": collector.value[
            "hba_expires_at_unix"
        ],
        "services_started": False,
        "units_installed": False,
        "daemon_reloaded": False,
        "discord_started": False,
        "approval_created": False,
        "credential_content_or_digest_recorded": False,
    }
    observed_at_unix = value.get("observed_at_unix")
    if (
        type(observed_at_unix) is not int
        or observed_at_unix < 0
        or not collector.value["hba_observed_at_unix"]
        <= collector.value["collected_at_unix"]
        <= observed_at_unix
        <= collector.value["hba_expires_at_unix"]
        or collector.value["hba_expires_at_unix"]
        - collector.value["hba_observed_at_unix"]
        != 300
        or unsigned != expected
        or report_sha256 != _sha256_json(unsigned)
    ):
        raise RuntimeError("native preflight report binding drifted")
    return json.loads(_canonical_bytes(dict(value)).decode("utf-8"))


def _recover_one_install_temporary(
    temporary: Path,
    *,
    target: Path,
    maximum: int,
) -> None:
    """Remove only an exact orphan left by ``_install_exact_bytes``.

    The target may already be the second hard link when the process died
    between link publication and temporary-name cleanup.  No authority-bearing
    target is removed here; only the mechanically named temporary link is.
    """

    item = os.lstat(temporary)
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISREG(item.st_mode)
        or item.st_uid != 0
        or item.st_gid != 0
        or item.st_nlink not in {1, 2}
        or stat.S_IMODE(item.st_mode) not in {0o400, 0o600}
        or item.st_size < 0
        or item.st_size > maximum
        or _list_xattrs(temporary)
    ):
        raise RuntimeError("publisher install temporary identity is not exact")
    if item.st_nlink == 2:
        if not os.path.lexists(target):
            raise RuntimeError("publisher linked temporary lacks its target")
        reached = os.lstat(target)
        if (
            (reached.st_dev, reached.st_ino) != (item.st_dev, item.st_ino)
            or reached.st_nlink != 2
            or stat.S_ISLNK(reached.st_mode)
            or not stat.S_ISREG(reached.st_mode)
        ):
            raise RuntimeError("publisher linked temporary target drifted")
    temporary.unlink()


def _recover_target_install_temporaries(target: Path, *, maximum: int) -> None:
    parent = target.parent
    if not os.path.lexists(parent):
        return
    _ensure_root_directory(parent)
    pattern = re.compile(
        rf"\.{re.escape(target.name)}\.activation\.[1-9][0-9]*"
    )
    changed = False
    for name in sorted(os.listdir(parent)):
        if pattern.fullmatch(name) is None:
            continue
        _recover_one_install_temporary(
            parent / name,
            target=target,
            maximum=maximum,
        )
        changed = True
    if changed:
        _fsync_directory(parent)


def _recover_report_install_temporaries(plan: Mapping[str, Any]) -> None:
    directory = _preflight_report_directory(plan)
    if not os.path.lexists(directory):
        return
    _ensure_root_directory(directory)
    pattern = re.compile(
        r"\.([0-9a-f]{64}\.json)\.activation\.[1-9][0-9]*"
    )
    changed = False
    for name in sorted(os.listdir(directory)):
        match = pattern.fullmatch(name)
        if match is None:
            continue
        _recover_one_install_temporary(
            directory / name,
            target=directory / match.group(1),
            maximum=_MAX_PUBLIC_JSON_BYTES,
        )
        changed = True
    if changed:
        _fsync_directory(directory)


def _validated_report_names(directory: Path) -> list[str]:
    item = os.lstat(directory)
    if (
        not stat.S_ISDIR(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_uid != 0
        or item.st_gid != 0
        or stat.S_IMODE(item.st_mode) != 0o700
    ):
        raise RuntimeError("native preflight report directory is not exact")
    names = sorted(os.listdir(directory))
    if any(re.fullmatch(r"[0-9a-f]{64}\.json", name) is None for name in names):
        raise RuntimeError("native preflight report namespace contains residue")
    if len(names) > 1:
        raise RuntimeError("native preflight report namespace is ambiguous")
    return names


def _load_persisted_preflight_report(
    *,
    plan: Mapping[str, Any],
    native: NativeObservationPlan,
    collector: ConfigCollectorReceipt,
) -> tuple[Mapping[str, Any], Path, bytes] | None:
    directory = _preflight_report_directory(plan)
    if not os.path.lexists(directory):
        return None
    names = _validated_report_names(directory)
    if not names:
        return None
    path = directory / names[0]
    raw = _read_trusted_file(
        path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_PUBLIC_JSON_BYTES,
    )
    report = _validate_native_preflight_report(
        _strict_mapping(raw, label="native preflight report", newline=False),
        plan=plan,
        native=native,
        collector=collector,
    )
    if path.name != f"{report['report_sha256']}.json":
        raise RuntimeError("native preflight report path digest drifted")
    return report, path, raw


def _seal_or_resume_preflight_report(
    *,
    plan: Mapping[str, Any],
    native: NativeObservationPlan,
    collector: ConfigCollectorReceipt,
    service_runner: Runner,
    clock: Callable[[], float],
) -> tuple[Mapping[str, Any], Path, str]:
    existing = _load_persisted_preflight_report(
        plan=plan,
        native=native,
        collector=collector,
    )
    if existing is not None:
        report, path, raw = existing
        return report, path, _sha256_bytes(raw)
    report = _validate_native_preflight_report(
        native_observation_read_only_preflight(
            native,
            runner=service_runner,
            _clock=clock,
        ),
        plan=plan,
        native=native,
        collector=collector,
    )
    raw = _canonical_bytes(report)
    path = _preflight_report_directory(plan) / f"{report['report_sha256']}.json"
    _ensure_root_directory(path.parent)
    _install_exact_bytes(path, raw, uid=0, gid=0, mode=0o400)
    loaded = _load_persisted_preflight_report(
        plan=plan,
        native=native,
        collector=collector,
    )
    if loaded is None or loaded[0] != report or loaded[1] != path or loaded[2] != raw:
        raise RuntimeError("native preflight report durable readback drifted")
    return report, path, _sha256_bytes(raw)


def _resume_persisted_report_state(
    plan: Mapping[str, Any],
) -> tuple[
    ConfigCollectorReceipt,
    NativeObservationPlan,
    tuple[Mapping[str, Any], Path, bytes],
] | None:
    directory = _preflight_report_directory(plan)
    if not os.path.lexists(directory):
        return None
    if not os.path.lexists(DEFAULT_STAGED_NATIVE_PLAN_PATH):
        names = _validated_report_names(directory)
        if names:
            raise RuntimeError("native preflight report lacks its staged plan")
        return None
    native = _load_staged_native_plan()
    collector, _collector_file_sha256 = _load_bound_collector_receipt(
        plan,
        str(native.value["config_collector_receipt_sha256"]),
        require_fresh=False,
    )
    _validate_native_binding(
        plan,
        native,
        collector,
        require_all_outputs=True,
    )
    report = _load_persisted_preflight_report(
        plan=plan,
        native=native,
        collector=collector,
    )
    if report is None:
        return None
    return collector, native, report


def _receipt_artifacts(native: NativeObservationPlan) -> Mapping[str, Any]:
    paths = {
        "writer_config": DEFAULT_WRITER_CONFIG_SOURCE_PATH,
        "gateway_config": DEFAULT_GATEWAY_CONFIG_SOURCE_PATH,
        "writer_unit": DEFAULT_STAGED_WRITER_UNIT_PATH,
        "gateway_unit": DEFAULT_STAGED_GATEWAY_UNIT_PATH,
        "native_observation_plan": DEFAULT_STAGED_NATIVE_PLAN_PATH,
    }
    result: dict[str, Any] = {}
    for name, path in paths.items():
        raw = _trusted_staged_bytes(
            path,
            maximum=(
                _MAX_UNIT_BYTES
                if name in {"writer_unit", "gateway_unit"}
                else _MAX_PUBLIC_JSON_BYTES
            ),
        )
        result[name] = {"path": str(path), "sha256": _sha256_bytes(raw)}
    if (
        result["writer_config"]["sha256"] != native.value["writer_config"]["sha256"]
        or result["gateway_config"]["sha256"]
        != native.value["gateway_config"]["sha256"]
        or result["writer_unit"]["sha256"] != native.value["writer_unit"]["sha256"]
        or result["gateway_unit"]["sha256"] != native.value["gateway_unit"]["sha256"]
        or result["native_observation_plan"]["sha256"] != native.sha256
    ):
        raise RuntimeError("publication artifact digest chain drifted")
    return result


def _preflight_time_envelope_sha256(report: Mapping[str, Any]) -> str:
    envelope = {
        "config_collector_receipt_sha256": report[
            "config_collector_receipt_sha256"
        ],
        "native_observation_plan_sha256": report[
            "native_observation_plan_sha256"
        ],
        "preflight_report_sha256": report["report_sha256"],
        "collector_hba_observed_at_unix": report[
            "collector_hba_observed_at_unix"
        ],
        "collector_collected_at_unix": report["collector_collected_at_unix"],
        "observed_at_unix": report["observed_at_unix"],
        "collector_hba_expires_at_unix": report[
            "collector_hba_expires_at_unix"
        ],
    }
    return _sha256_json(envelope)


def _publication_provenance(
    *,
    plan: Mapping[str, Any],
    collector: ConfigCollectorReceipt,
    collector_file_sha256: str,
    native: NativeObservationPlan,
    report: Mapping[str, Any],
    report_file_sha256: str,
) -> Mapping[str, str]:
    projection = {
        "approved_plan_sha256": plan["plan_sha256"],
        "release_artifact_sha256": plan["release_artifact_sha256"],
        "release_manifest_file_sha256": plan["release_manifest_file_sha256"],
        "database_ca_sha256": plan["database"]["ca_sha256"],
        "config_collector_receipt_sha256": collector.sha256,
        "config_collector_receipt_file_sha256": collector_file_sha256,
        "collector_writer_config_sha256": collector.value[
            "writer_config_sha256"
        ],
        "collector_gateway_config_sha256": collector.value[
            "gateway_config_sha256"
        ],
        "native_observation_plan_sha256": native.sha256,
        "native_writer_config_sha256": native.value["writer_config"]["sha256"],
        "native_gateway_config_sha256": native.value["gateway_config"]["sha256"],
        "native_writer_unit_sha256": native.value["writer_unit"]["sha256"],
        "native_gateway_unit_sha256": native.value["gateway_unit"]["sha256"],
        "preflight_report_sha256": report["report_sha256"],
        "preflight_report_file_sha256": report_file_sha256,
        "preflight_time_envelope_sha256": _preflight_time_envelope_sha256(
            report
        ),
    }
    if set(projection) != _PUBLICATION_PROVENANCE_FIELDS:
        raise AssertionError("publisher provenance projection fields drifted")
    return {
        name: _digest(value, f"publisher provenance {name}")
        for name, value in projection.items()
    }


def _revalidate_pre_receipt_truth(
    *,
    plan: Mapping[str, Any],
    expected_collector: ConfigCollectorReceipt,
    expected_native: NativeObservationPlan,
    expected_report: Mapping[str, Any],
    expected_report_path: Path,
    service_runner: Runner,
) -> tuple[
    ConfigCollectorReceipt,
    str,
    NativeObservationPlan,
    Mapping[str, Any],
    Path,
    str,
    Mapping[str, Any],
    Mapping[str, Mapping[str, str]],
]:
    """Re-read every mutable input immediately before terminal sealing."""

    recomputed = plan_writer_preflight_publication(
        revision=str(plan["revision"]),
        external_iam_policy_sha256=str(plan["external_iam_policy_sha256"]),
        _service_runner=service_runner,
    )
    if (
        recomputed != plan
        or _canonical_bytes(recomputed) != _canonical_bytes(plan)
        or recomputed["plan_sha256"] != plan["plan_sha256"]
    ):
        raise RuntimeError("approved publication plan changed before sealing")
    collector, collector_file_sha256 = _load_bound_collector_receipt(
        plan,
        expected_collector.sha256,
        require_fresh=False,
    )
    if collector.to_mapping() != expected_collector.to_mapping():
        raise RuntimeError("publication collector changed before sealing")
    native = _load_staged_native_plan()
    if (
        native.sha256 != expected_native.sha256
        or native.to_mapping() != expected_native.to_mapping()
    ):
        raise RuntimeError("publication native plan changed before sealing")
    _validate_native_binding(
        plan,
        native,
        collector,
        require_all_outputs=True,
    )
    _verify_native_release(native)
    persisted = _load_persisted_preflight_report(
        plan=plan,
        native=native,
        collector=collector,
    )
    if (
        persisted is None
        or persisted[0] != expected_report
        or persisted[1] != expected_report_path
    ):
        raise RuntimeError("publication preflight report changed before sealing")
    report, report_path, report_raw = persisted
    report_file_sha256 = _sha256_bytes(report_raw)
    artifacts = _receipt_artifacts(native)
    service_state = _capture_service_snapshot(runner=service_runner)
    if service_state != _validate_service_snapshot(plan["service_state"]):
        raise RuntimeError("systemd service state changed during writer preflight")
    _require_no_downstream_mutation()
    return (
        collector,
        collector_file_sha256,
        native,
        report,
        report_path,
        report_file_sha256,
        artifacts,
        service_state,
    )


def _validate_terminal_receipt(
    path: Path,
    *,
    plan: Mapping[str, Any],
    service_runner: Runner,
) -> Mapping[str, Any]:
    recomputed_plan = plan_writer_preflight_publication(
        revision=str(plan["revision"]),
        external_iam_policy_sha256=str(plan["external_iam_policy_sha256"]),
        _service_runner=service_runner,
    )
    if (
        recomputed_plan != plan
        or _canonical_bytes(recomputed_plan) != _canonical_bytes(plan)
    ):
        raise RuntimeError("publication plan changed after terminal sealing")
    raw = _read_trusted_file(
        path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_PUBLIC_JSON_BYTES,
    )
    value = _strict_mapping(raw, label="publication receipt", newline=False)
    receipt_sha = _digest(value.get("receipt_sha256"), "publication receipt")
    unsigned = {name: copy.deepcopy(item) for name, item in value.items()}
    del unsigned["receipt_sha256"]
    if set(value) != _PUBLICATION_RECEIPT_FIELDS:
        raise RuntimeError("publication receipt fields are not exact")
    copied_plan_fields = {
        "revision": "revision",
        "approved_plan_sha256": "plan_sha256",
        "stopped_release_receipt_sha256": "stopped_release_receipt_sha256",
        "release_artifact_sha256": "release_artifact_sha256",
        "release_manifest_file_sha256": "release_manifest_file_sha256",
        "host_identity_receipt_sha256": "host_identity_receipt_sha256",
        "external_iam_policy_sha256": "external_iam_policy_sha256",
    }
    if (
        value.get("schema") != PUBLICATION_RECEIPT_SCHEMA
        or value.get("ok") is not True
        or value.get("state") != "staged_preflight_passed_services_stopped"
        or any(
            value.get(receipt_name) != plan[plan_name]
            for receipt_name, plan_name in copied_plan_fields.items()
        )
        or value.get("invariants") != plan["invariants"]
        or value.get("receipt_path") != str(path)
        or type(value.get("sealed_at_unix")) is not int
        or value["sealed_at_unix"] < 0
        or receipt_sha != _sha256_json(unsigned)
    ):
        raise RuntimeError("publication receipt binding drifted")
    before = _validate_service_snapshot(value.get("service_state_before"))
    after = _validate_service_snapshot(value.get("service_state_after"))
    planned = _validate_service_snapshot(plan.get("service_state"))
    current = _capture_service_snapshot(runner=service_runner)
    if before != planned or after != before or current != before:
        raise RuntimeError("publication service state changed across preflight")

    collector_sha256 = _digest(
        value.get("config_collector_receipt_sha256"),
        "publication config collector receipt",
    )
    collector_path = _collector_receipt_path(plan, collector_sha256)
    if value.get("config_collector_receipt_path") != str(collector_path):
        raise RuntimeError("publication collector receipt path drifted")
    collector, collector_file_sha256 = _load_bound_collector_receipt(
        plan,
        collector_sha256,
        require_fresh=False,
    )
    if value.get("config_collector_receipt_file_sha256") != (
        collector_file_sha256
    ):
        raise RuntimeError("publication collector receipt file drifted")
    native = _load_staged_native_plan()
    if value.get("native_observation_plan_sha256") != native.sha256:
        raise RuntimeError("publication native plan digest drifted")
    _validate_native_binding(
        plan,
        native,
        collector,
        require_all_outputs=True,
    )
    _verify_native_release(native)

    report_sha256 = _digest(
        value.get("preflight_report_sha256"),
        "publication native preflight report",
    )
    report_path = (
        _preflight_report_directory(plan) / f"{report_sha256}.json"
    )
    if value.get("preflight_report_path") != str(report_path):
        raise RuntimeError("publication native preflight report path drifted")
    persisted_report = _load_persisted_preflight_report(
        plan=plan,
        native=native,
        collector=collector,
    )
    if persisted_report is None or persisted_report[1] != report_path:
        raise RuntimeError("publication native preflight report is absent")
    report, _persisted_path, report_raw = persisted_report
    if _sha256_bytes(report_raw) != _digest(
        value.get("preflight_report_file_sha256"),
        "publication native preflight report file",
    ):
        raise RuntimeError("publication native preflight report file drifted")
    if report["report_sha256"] != report_sha256:
        raise RuntimeError("publication native preflight report digest drifted")

    artifacts = value.get("artifacts")
    expected_artifacts = _receipt_artifacts(native)
    if artifacts != expected_artifacts:
        raise RuntimeError("publication receipt artifact binding drifted")
    report_file_sha256 = _sha256_bytes(report_raw)
    expected_provenance = _publication_provenance(
        plan=plan,
        collector=collector,
        collector_file_sha256=collector_file_sha256,
        native=native,
        report=report,
        report_file_sha256=report_file_sha256,
    )
    if value.get("provenance") != expected_provenance:
        raise RuntimeError("publication receipt provenance binding drifted")
    hba_observed_at = report["collector_hba_observed_at_unix"]
    collector_collected_at = report["collector_collected_at_unix"]
    preflight_observed_at = report["observed_at_unix"]
    hba_expires_at = report["collector_hba_expires_at_unix"]
    sealed_at = value["sealed_at_unix"]
    time_envelope_sha256 = _preflight_time_envelope_sha256(report)
    if (
        value.get("preflight_collector_hba_observed_at_unix")
        != hba_observed_at
        or value.get("preflight_collector_collected_at_unix")
        != collector_collected_at
        or value.get("preflight_observed_at_unix") != preflight_observed_at
        or value.get("preflight_collector_hba_expires_at_unix")
        != hba_expires_at
        or value.get("preflight_time_envelope_sha256")
        != time_envelope_sha256
        or not hba_observed_at
        <= collector_collected_at
        <= preflight_observed_at
        <= hba_expires_at
        or sealed_at < preflight_observed_at
        or value.get("preflight_fresh_at_seal")
        is not (sealed_at <= hba_expires_at)
    ):
        raise RuntimeError("publication receipt preflight time binding drifted")
    if (
        current_host_identity_sha256() != plan["host_identity_sha256"]
        or _current_boot_id_sha256() != plan["boot_id_sha256"]
    ):
        raise RuntimeError("publication receipt host identity drifted")
    _require_no_downstream_mutation()
    return value


def apply_writer_preflight_publication(
    *,
    revision: str,
    external_iam_policy_sha256: str,
    approved_plan_sha256: str,
    _service_runner: Runner = _runner,
    _clock: Callable[[], float] = time.time,
) -> Mapping[str, Any]:
    """Stage exact inputs and seal proof while all services remain stopped."""

    approved = _digest(approved_plan_sha256, "approved publication plan")
    prelock_plan = plan_writer_preflight_publication(
        revision=revision,
        external_iam_policy_sha256=external_iam_policy_sha256,
        _service_runner=_service_runner,
    )
    if prelock_plan["plan_sha256"] != approved:
        raise PermissionError("approved publisher plan digest does not match")
    with _host_activation_lock():
        return _apply_writer_preflight_publication_locked(
            revision=revision,
            external_iam_policy_sha256=external_iam_policy_sha256,
            approved_plan_sha256=approved,
            service_runner=_service_runner,
            clock=_clock,
        )


def _apply_writer_preflight_publication_locked(
    *,
    revision: str,
    external_iam_policy_sha256: str,
    approved_plan_sha256: str,
    service_runner: Runner,
    clock: Callable[[], float],
) -> Mapping[str, Any]:
    """Apply the publisher contract while holding the shared host lock."""

    plan = plan_writer_preflight_publication(
        revision=revision,
        external_iam_policy_sha256=external_iam_policy_sha256,
        _service_runner=service_runner,
    )
    if plan["plan_sha256"] != approved_plan_sha256:
        raise PermissionError("approved publisher plan digest does not match")
    receipt_path = _publication_receipt_path(plan)
    _recover_target_install_temporaries(
        receipt_path,
        maximum=_MAX_PUBLIC_JSON_BYTES,
    )
    _recover_report_install_temporaries(plan)
    if os.path.lexists(receipt_path):
        return _validate_terminal_receipt(
            receipt_path,
            plan=plan,
            service_runner=service_runner,
        )
    resumed = _resume_persisted_report_state(plan)
    if resumed is None:
        now_unix = int(clock())
        if now_unix < 0:
            raise ValueError("publisher current time is invalid")
        collector = _collect_or_resume_configs(
            plan,
            now_unix=now_unix,
            clock=clock,
        )
        native = _load_or_stage_native_plan(plan, collector)
        report, report_path, report_file_sha256 = (
            _seal_or_resume_preflight_report(
                plan=plan,
                native=native,
                collector=collector,
                service_runner=service_runner,
                clock=clock,
            )
        )
    else:
        collector, native, persisted = resumed
        report, report_path, report_raw = persisted
        report_file_sha256 = _sha256_bytes(report_raw)
    (
        collector,
        collector_file_sha256,
        native,
        report,
        report_path,
        report_file_sha256,
        artifacts,
        service_state_after,
    ) = _revalidate_pre_receipt_truth(
        plan=plan,
        expected_collector=collector,
        expected_native=native,
        expected_report=report,
        expected_report_path=report_path,
        service_runner=service_runner,
    )
    service_state_before = _validate_service_snapshot(plan["service_state"])
    sealed_at = int(clock())
    if sealed_at < 0 or sealed_at < report["observed_at_unix"]:
        raise ValueError("publication receipt time is invalid")
    provenance = _publication_provenance(
        plan=plan,
        collector=collector,
        collector_file_sha256=collector_file_sha256,
        native=native,
        report=report,
        report_file_sha256=report_file_sha256,
    )
    unsigned: dict[str, Any] = {
        "schema": PUBLICATION_RECEIPT_SCHEMA,
        "ok": True,
        "state": "staged_preflight_passed_services_stopped",
        "revision": revision,
        "approved_plan_sha256": approved_plan_sha256,
        "stopped_release_receipt_sha256": plan["stopped_release_receipt_sha256"],
        "release_artifact_sha256": plan["release_artifact_sha256"],
        "release_manifest_file_sha256": plan["release_manifest_file_sha256"],
        "host_identity_receipt_sha256": plan["host_identity_receipt_sha256"],
        "config_collector_receipt_path": str(
            CONFIG_COLLECTOR_EVIDENCE_ROOT / revision / f"{collector.sha256}.json"
        ),
        "config_collector_receipt_sha256": collector.sha256,
        "config_collector_receipt_file_sha256": collector_file_sha256,
        "native_observation_plan_sha256": native.sha256,
        "external_iam_policy_sha256": plan["external_iam_policy_sha256"],
        "preflight_report_path": str(report_path),
        "preflight_report_file_sha256": report_file_sha256,
        "preflight_report_sha256": report["report_sha256"],
        "preflight_observed_at_unix": report["observed_at_unix"],
        "preflight_collector_hba_observed_at_unix": report[
            "collector_hba_observed_at_unix"
        ],
        "preflight_collector_collected_at_unix": report[
            "collector_collected_at_unix"
        ],
        "preflight_collector_hba_expires_at_unix": report[
            "collector_hba_expires_at_unix"
        ],
        "preflight_time_envelope_sha256": _preflight_time_envelope_sha256(
            report
        ),
        "preflight_fresh_at_seal": (
            sealed_at <= report["collector_hba_expires_at_unix"]
        ),
        "service_state_before": service_state_before,
        "service_state_after": service_state_after,
        "artifacts": artifacts,
        "provenance": provenance,
        "invariants": copy.deepcopy(plan["invariants"]),
        "sealed_at_unix": sealed_at,
        "receipt_path": str(receipt_path),
    }
    receipt = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
    _ensure_root_directory(receipt_path.parent)
    _install_exact_bytes(
        receipt_path,
        _canonical_bytes(receipt),
        uid=0,
        gid=0,
        mode=0o400,
    )
    return _validate_terminal_receipt(
        receipt_path,
        plan=plan,
        service_runner=service_runner,
    )


class _CanonicalArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise ValueError("invalid writer preflight publisher CLI arguments")


class _StoreOnce(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        del parser, option_string
        if getattr(namespace, self.dest, None) is not None:
            raise ValueError("writer preflight publisher option was repeated")
        setattr(namespace, self.dest, values)


def _exact_revision(value: str) -> str:
    if _REVISION_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("invalid revision")
    return value


def _exact_sha256(value: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("invalid digest")
    return value


def _cli_parser() -> argparse.ArgumentParser:
    parser = _CanonicalArgumentParser(
        description="Stage one stopped writer-only preflight",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=_CanonicalArgumentParser,
    )
    for command in ("plan", "apply"):
        child = subparsers.add_parser(command, allow_abbrev=False)
        child.add_argument(
            "--revision",
            required=True,
            default=None,
            type=_exact_revision,
            action=_StoreOnce,
        )
        child.add_argument(
            "--external-iam-policy-sha256",
            required=True,
            default=None,
            type=_exact_sha256,
            action=_StoreOnce,
        )
        if command == "apply":
            child.add_argument(
                "--approved-plan-sha256",
                required=True,
                default=None,
                type=_exact_sha256,
                action=_StoreOnce,
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _cli_parser().parse_args(argv)
        if arguments.command == "plan":
            result = plan_writer_preflight_publication(
                revision=arguments.revision,
                external_iam_policy_sha256=(arguments.external_iam_policy_sha256),
            )
        elif arguments.command == "apply":
            result = apply_writer_preflight_publication(
                revision=arguments.revision,
                external_iam_policy_sha256=(arguments.external_iam_policy_sha256),
                approved_plan_sha256=arguments.approved_plan_sha256,
            )
        else:  # pragma: no cover - argparse enforces this set.
            raise RuntimeError("unsupported publisher command")
        print(_canonical_bytes(result).decode("utf-8", errors="strict"))
        return 0
    except Exception as exc:
        failure = {
            "schema": PUBLICATION_FAILURE_SCHEMA,
            "ok": False,
            "error_code": "writer_preflight_publication_failed",
            "error_type": type(exc).__name__,
        }
        print(_canonical_bytes(failure).decode("utf-8", errors="strict"))
        return 2


__all__ = [
    "DATABASE_TLS_SERVER_NAME",
    "OWNER_DISCORD_USER_ID",
    "PUBLICATION_FAILURE_SCHEMA",
    "PUBLICATION_PLAN_SCHEMA",
    "PUBLICATION_RECEIPT_SCHEMA",
    "apply_writer_preflight_publication",
    "main",
    "plan_writer_preflight_publication",
]


if __name__ == "__main__":
    raise SystemExit(main())
