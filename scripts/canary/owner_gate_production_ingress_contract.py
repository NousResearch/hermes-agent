#!/usr/bin/env python3
"""Pure signed contract for the owner-gate production-ingress observation.

This module deliberately contains no subprocess, gcloud, SSH, launcher, or
transport implementation.  It is safe in the installed canary runtime closure
and validates the exact release-signed observation produced by the separate
owner-side collector.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, NoReturn


OBSERVATION_SCHEMA = "muncho-owner-gate-production-ingress-observation.v1"
ENVELOPE_SCHEMA = (
    "muncho-owner-gate-release-signed-production-ingress-observation.v1"
)
SIGNATURE_DOMAIN = b"muncho-owner-gate/production-ingress-observation/v1\x00"
PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256 = (
    "302bd03b449a4f46476d9d2dc8026acedaca17334154ba2cf8ba2a68c72992a0"
)

PROJECT = "adventico-ai-platform"
ZONE = "europe-west3-a"
VM_NAME = "ai-platform-runtime-01"
INSTANCE_ID = "1094477181810932795"

OLD_V1_UNIT = "muncho-passkey-stepup.service"
OLD_V1_MASK_PATH = Path("/etc/systemd/system/muncho-passkey-stepup.service")
OLD_V1_MASK_TARGET = "/dev/null"
CADDY_UNIT = "caddy.service"
CADDY_UNIT_FRAGMENT = "/lib/systemd/system/caddy.service"
CADDY_EXECUTABLE = "/usr/bin/caddy"
CADDYFILE_PATH = Path("/etc/caddy/Caddyfile")
CADDY_ADMIN_PORT = 2019
PUBLIC_ORIGIN = "https://auth.lomliev.com"
PUBLIC_HOST = "auth.lomliev.com"
PRIVATE_V2_UPSTREAM = "10.80.3.2:8080"

EXPECTED_ROOT_UID = 0
EXPECTED_ROOT_GID = 0
CADDYFILE_MODE = 0o644
FRESHNESS_SECONDS = 900
MAX_SIGNING_DELAY_SECONDS = 300
MAX_CADDYFILE_BYTES = 1024 * 1024

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_B64URL_SIGNATURE = re.compile(r"^[A-Za-z0-9_-]{86}$")

_OBSERVATION_FIELDS = frozenset(
    {
        "schema",
        "phase",
        "release_revision",
        "plan_sha256",
        "target",
        "collected_at_unix",
        "completed_at_unix",
        "fresh_through_unix",
        "old_v1",
        "caddy",
        "collector_authority",
        "caller_selected_input_accepted",
        "cloud_mutation_performed",
        "service_mutation_performed",
        "secret_material_recorded",
        "secret_digest_recorded",
        "report_sha256",
    }
)
_TARGET_FIELDS = frozenset({"project", "zone", "vm", "instance_id"})
_OLD_V1_FIELDS = frozenset(
    {
        "unit",
        "load_state",
        "active_state",
        "sub_state",
        "unit_file_state",
        "fragment_path",
        "drop_in_paths",
        "permanent_mask_path",
        "permanent_mask_target",
        "mask_uid",
        "mask_gid",
        "trusted_for_v2",
    }
)
_CADDY_FIELDS = frozenset(
    {
        "unit",
        "load_state",
        "active_state",
        "sub_state",
        "unit_file_state",
        "fragment_path",
        "drop_in_paths",
        "main_pid",
        "exec_start_argv",
        "config_path",
        "config_uid",
        "config_gid",
        "config_mode",
        "config_size",
        "public_origin",
        "auth_host_route_count",
        "reverse_proxy_handler_count",
        "reverse_proxy_upstream_count",
        "still_on_current_host",
        "private_v2_upstream_active",
        "process_executable",
        "process_cmdline",
        "admin_endpoint",
        "live_route_projection_sha256",
        "effective_unit_inventory_closed",
        "active_process_stable",
        "admin_listener_owned_by_main_pid",
        "live_config_matches_adapted_config",
        "double_live_config_projection_identical",
        "config_validated",
        "stable_nofollow_config_verified",
        "double_adapt_projection_identical",
        "rollback_mode",
    }
)
_TRANSPORT_FIELDS = frozenset(
    {
        "kind",
        "project",
        "zone",
        "vm",
        "instance_id",
        "known_hosts_file_sha256",
        "observer_source_sha256",
        "instance_authorization_sha256",
        "project_authorization_sha256",
        "oslogin_authorization_sha256",
    }
)
_ENVELOPE_FIELDS = frozenset(
    {
        "schema",
        "phase",
        "release_revision",
        "plan_sha256",
        "observation",
        "observer_report_sha256",
        "transport_authority",
        "signed_at_unix",
        "fresh_through_unix",
        "signer_key_id",
        "secret_material_recorded",
        "secret_digest_recorded",
        "signature_ed25519_b64url",
        "envelope_sha256",
    }
)


class ProductionIngressObservationError(RuntimeError):
    """Stable, secret-free production-ingress contract failure."""


def _error(code: str, _cause: BaseException | None = None) -> NoReturn:
    del _cause
    raise ProductionIngressObservationError(code) from None


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
        _error("owner_gate_production_ingress_json_invalid", exc)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_mapping(
    value: Any,
    fields: frozenset[str],
    *,
    code: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != fields:
        _error(code)
    return value


def _duplicate_keys(pairs: list[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _error("owner_gate_production_ingress_json_invalid")
        result[key] = value
    return result


def _decode_json(raw: bytes, *, canonical: bool) -> Mapping[str, Any]:
    if not isinstance(raw, bytes) or not raw:
        _error("owner_gate_production_ingress_json_invalid")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_duplicate_keys,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except ProductionIngressObservationError:
        raise
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        _error("owner_gate_production_ingress_json_invalid", exc)
    if not isinstance(value, Mapping) or (canonical and _canonical(value) != raw):
        _error("owner_gate_production_ingress_json_invalid")
    return value


def _phase(value: Any) -> str:
    if value not in {"inert", "post_iam"}:
        _error("owner_gate_production_ingress_binding_invalid")
    return str(value)


def _binding(
    *,
    phase: Any,
    release_revision: Any,
    plan_sha256: Any,
) -> tuple[str, str, str]:
    checked_phase = _phase(phase)
    if (
        not isinstance(release_revision, str)
        or _REVISION.fullmatch(release_revision) is None
        or not isinstance(plan_sha256, str)
        or _SHA256.fullmatch(plan_sha256) is None
    ):
        _error("owner_gate_production_ingress_binding_invalid")
    return checked_phase, release_revision, plan_sha256


def _validate_old_v1(value: Any) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value,
        _OLD_V1_FIELDS,
        code="owner_gate_production_ingress_old_v1_invalid",
    )
    if raw != {
        "unit": OLD_V1_UNIT,
        "load_state": "masked",
        "active_state": "inactive",
        "sub_state": "dead",
        "unit_file_state": "masked",
        "fragment_path": OLD_V1_MASK_TARGET,
        "drop_in_paths": [],
        "permanent_mask_path": str(OLD_V1_MASK_PATH),
        "permanent_mask_target": OLD_V1_MASK_TARGET,
        "mask_uid": EXPECTED_ROOT_UID,
        "mask_gid": EXPECTED_ROOT_GID,
        "trusted_for_v2": False,
    }:
        _error("owner_gate_production_ingress_old_v1_invalid")
    return raw


def _validate_caddy(value: Any) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value,
        _CADDY_FIELDS,
        code="owner_gate_production_ingress_caddy_invalid",
    )
    expected_argv = [
        CADDY_EXECUTABLE,
        "run",
        "--environ",
        "--config",
        str(CADDYFILE_PATH),
    ]
    route_projection = {
        "auth_host_route_count": raw.get("auth_host_route_count"),
        "reverse_proxy_handler_count": raw.get(
            "reverse_proxy_handler_count"
        ),
        "reverse_proxy_upstream_count": raw.get(
            "reverse_proxy_upstream_count"
        ),
        "still_on_current_host": raw.get("still_on_current_host"),
        "private_v2_upstream_active": raw.get(
            "private_v2_upstream_active"
        ),
    }
    if (
        raw.get("unit") != CADDY_UNIT
        or raw.get("load_state") != "loaded"
        or raw.get("active_state") != "active"
        or raw.get("sub_state") != "running"
        or raw.get("unit_file_state") != "enabled"
        or raw.get("fragment_path") != CADDY_UNIT_FRAGMENT
        or raw.get("drop_in_paths") != []
        or type(raw.get("main_pid")) is not int
        or raw["main_pid"] <= 0
        or raw.get("exec_start_argv") != expected_argv
        or raw.get("config_path") != str(CADDYFILE_PATH)
        or raw.get("config_uid") != EXPECTED_ROOT_UID
        or raw.get("config_gid") != EXPECTED_ROOT_GID
        or raw.get("config_mode") != f"{CADDYFILE_MODE:04o}"
        or type(raw.get("config_size")) is not int
        or raw["config_size"] <= 0
        or raw["config_size"] > MAX_CADDYFILE_BYTES
        or raw.get("public_origin") != PUBLIC_ORIGIN
        or raw.get("auth_host_route_count") != 1
        or type(raw.get("reverse_proxy_handler_count")) is not int
        or raw["reverse_proxy_handler_count"] <= 0
        or type(raw.get("reverse_proxy_upstream_count")) is not int
        or raw["reverse_proxy_upstream_count"] <= 0
        or raw.get("still_on_current_host") is not True
        or raw.get("private_v2_upstream_active") is not False
        or raw.get("process_executable") != CADDY_EXECUTABLE
        or raw.get("process_cmdline") != expected_argv
        or raw.get("admin_endpoint")
        not in {
            f"127.0.0.1:{CADDY_ADMIN_PORT}",
            f"[::1]:{CADDY_ADMIN_PORT}",
        }
        or not isinstance(raw.get("live_route_projection_sha256"), str)
        or raw["live_route_projection_sha256"]
        != _sha256(_canonical(route_projection))
        or raw.get("effective_unit_inventory_closed") is not True
        or raw.get("active_process_stable") is not True
        or raw.get("admin_listener_owned_by_main_pid") is not True
        or raw.get("live_config_matches_adapted_config") is not True
        or raw.get("double_live_config_projection_identical") is not True
        or raw.get("config_validated") is not True
        or raw.get("stable_nofollow_config_verified") is not True
        or raw.get("double_adapt_projection_identical") is not True
        or raw.get("rollback_mode") != "pre_migration_v1_only"
    ):
        _error("owner_gate_production_ingress_caddy_invalid")
    return raw


def validate_production_ingress_observation(
    value: Any,
    *,
    phase: str,
    release_revision: str,
    plan_sha256: str,
    now_unix: int,
) -> Mapping[str, Any]:
    """Strictly validate one remote observation before it can be signed."""

    checked_phase, revision, plan_digest = _binding(
        phase=phase,
        release_revision=release_revision,
        plan_sha256=plan_sha256,
    )
    raw = _strict_mapping(
        value,
        _OBSERVATION_FIELDS,
        code="owner_gate_production_ingress_observation_invalid",
    )
    unsigned = {key: item for key, item in raw.items() if key != "report_sha256"}
    collected = raw.get("collected_at_unix")
    completed = raw.get("completed_at_unix")
    fresh = raw.get("fresh_through_unix")
    if (
        type(now_unix) is not int
        or now_unix <= 0
        or raw.get("schema") != OBSERVATION_SCHEMA
        or raw.get("phase") != checked_phase
        or raw.get("release_revision") != revision
        or raw.get("plan_sha256") != plan_digest
        or raw.get("target")
        != {
            "project": PROJECT,
            "zone": ZONE,
            "vm": VM_NAME,
            "instance_id": INSTANCE_ID,
        }
        or type(collected) is not int
        or type(completed) is not int
        or type(fresh) is not int
        or collected <= 0
        or completed < collected
        or fresh != collected + FRESHNESS_SECONDS
        or completed > now_unix
        or now_unix > fresh
        or raw.get("collector_authority")
        != "production_root_read_only_fixed_projection"
        or raw.get("caller_selected_input_accepted") is not False
        or raw.get("cloud_mutation_performed") is not False
        or raw.get("service_mutation_performed") is not False
        or raw.get("secret_material_recorded") is not False
        or raw.get("secret_digest_recorded") is not False
        or not isinstance(raw.get("report_sha256"), str)
        or raw["report_sha256"] != _sha256(_canonical(unsigned))
    ):
        _error("owner_gate_production_ingress_observation_invalid")
    _strict_mapping(
        raw["target"],
        _TARGET_FIELDS,
        code="owner_gate_production_ingress_observation_invalid",
    )
    _validate_old_v1(raw["old_v1"])
    _validate_caddy(raw["caddy"])
    return _decode_json(_canonical(raw), canonical=True)


def _release_key_id(public_key: Any) -> str:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError as exc:
        _error("owner_gate_production_ingress_signer_invalid", exc)
    if not isinstance(public_key, Ed25519PublicKey):
        _error("owner_gate_production_ingress_signer_invalid")
    return _sha256(public_key.public_bytes_raw())


def _require_pinned_release_key(public_key: Any) -> str:
    expected = PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256
    key_id = _release_key_id(public_key)
    if (
        not isinstance(expected, str)
        or _SHA256.fullmatch(expected) is None
        or key_id != expected
    ):
        _error("owner_gate_production_ingress_signer_not_pinned")
    return key_id


def _validate_transport_authority(value: Any) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value,
        _TRANSPORT_FIELDS,
        code="owner_gate_production_ingress_transport_authority_invalid",
    )
    digest_fields = (
        "known_hosts_file_sha256",
        "observer_source_sha256",
        "instance_authorization_sha256",
        "project_authorization_sha256",
        "oslogin_authorization_sha256",
    )
    if (
        raw.get("kind") != "pinned_owner_gcloud_iap_ssh_read_only"
        or raw.get("project") != PROJECT
        or raw.get("zone") != ZONE
        or raw.get("vm") != VM_NAME
        or raw.get("instance_id") != INSTANCE_ID
        or any(
            not isinstance(raw.get(name), str)
            or _SHA256.fullmatch(raw[name]) is None
            for name in digest_fields
        )
    ):
        _error("owner_gate_production_ingress_transport_authority_invalid")
    return raw


def _decode_signature(value: Any) -> bytes:
    if not isinstance(value, str) or _B64URL_SIGNATURE.fullmatch(value) is None:
        _error("owner_gate_production_ingress_signature_invalid")
    try:
        raw = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        _error("owner_gate_production_ingress_signature_invalid", exc)
    if (
        len(raw) != 64
        or base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != value
    ):
        _error("owner_gate_production_ingress_signature_invalid")
    return raw


def validate_signed_production_ingress_observation(
    value: Any,
    *,
    phase: str,
    release_revision: str,
    plan_sha256: str,
    release_public_key: Any,
    now_unix: int,
) -> Mapping[str, Any]:
    """Validate signer pin, signature, transport binding, and freshness."""

    checked_phase, revision, plan_digest = _binding(
        phase=phase,
        release_revision=release_revision,
        plan_sha256=plan_sha256,
    )
    raw = _strict_mapping(
        value,
        _ENVELOPE_FIELDS,
        code="owner_gate_production_ingress_envelope_invalid",
    )
    signed = {key: item for key, item in raw.items() if key != "envelope_sha256"}
    unsigned = {
        key: item
        for key, item in signed.items()
        if key != "signature_ed25519_b64url"
    }
    signed_at = raw.get("signed_at_unix")
    fresh = raw.get("fresh_through_unix")
    if (
        type(now_unix) is not int
        or now_unix <= 0
        or raw.get("schema") != ENVELOPE_SCHEMA
        or raw.get("phase") != checked_phase
        or raw.get("release_revision") != revision
        or raw.get("plan_sha256") != plan_digest
        or type(signed_at) is not int
        or type(fresh) is not int
        or signed_at <= 0
        or signed_at > now_unix
        or now_unix > fresh
        or raw.get("secret_material_recorded") is not False
        or raw.get("secret_digest_recorded") is not False
        or not isinstance(raw.get("envelope_sha256"), str)
        or raw["envelope_sha256"] != _sha256(_canonical(signed))
    ):
        _error("owner_gate_production_ingress_envelope_invalid")
    observation = validate_production_ingress_observation(
        raw.get("observation"),
        phase=checked_phase,
        release_revision=revision,
        plan_sha256=plan_digest,
        now_unix=now_unix,
    )
    if (
        raw.get("observer_report_sha256") != observation["report_sha256"]
        or fresh != observation["fresh_through_unix"]
        or signed_at < observation["completed_at_unix"]
        or signed_at - observation["completed_at_unix"]
        > MAX_SIGNING_DELAY_SECONDS
    ):
        _error("owner_gate_production_ingress_envelope_invalid")
    _validate_transport_authority(raw.get("transport_authority"))
    signer_key_id = _require_pinned_release_key(release_public_key)
    if raw.get("signer_key_id") != signer_key_id:
        _error("owner_gate_production_ingress_signer_invalid")
    try:
        from cryptography.exceptions import InvalidSignature

        release_public_key.verify(
            _decode_signature(raw.get("signature_ed25519_b64url")),
            SIGNATURE_DOMAIN + _canonical(unsigned),
        )
    except InvalidSignature as exc:
        _error("owner_gate_production_ingress_signature_invalid", exc)
    return _decode_json(_canonical(raw), canonical=True)
