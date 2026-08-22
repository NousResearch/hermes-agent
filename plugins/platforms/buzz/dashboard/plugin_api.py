"""Profile-scoped Buzz policy routes for the Hermes Dashboard.

The host mounts this router under ``/api/plugins/buzz-platform``. Its existing
Dashboard middleware authenticates every plugin API request before dispatch.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, StrictBool, field_validator, model_validator
import yaml

from plugins.platforms.buzz import settings


router = APIRouter()
_POLICY_FIELDS = tuple(settings._POLICY_FIELDS)
_CANONICAL_PREFIX = "gateway.platforms.buzz.extra"
_LEGACY_PREFIXES = (
    "gateway.platforms.buzz",
    "platforms.buzz",
    "platforms.buzz.extra",
    "gateway.buzz",
    "gateway.buzz.extra",
    "buzz",
    "buzz.extra",
)


class BuzzPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    allowed_users: list[str] | None = None
    allow_all_users: StrictBool | None = None
    require_mention: StrictBool | None = None
    thread_require_mention: StrictBool | None = None

    @model_validator(mode="before")
    @classmethod
    def require_nonempty_partial_policy(cls, value: Any) -> Any:
        if not isinstance(value, dict) or not value:
            raise ValueError("policy must submit at least one supported field")
        if any(item is None for item in value.values()):
            raise ValueError("submitted policy fields cannot be null")
        return value

    @field_validator("allowed_users")
    @classmethod
    def normalize_allowed_users(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        normalized: list[str] = []
        for value in values:
            identity = settings.normalize_user_ref(value)
            if identity is None:
                raise ValueError("invalid Buzz public identity")
            if identity not in normalized:
                normalized.append(identity)
        return normalized


class BuzzPolicyUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    policy: BuzzPolicy


def _read_user_config_strict() -> dict[str, Any]:
    """Failure-preserving, unexpanded read of the selected user config."""

    from hermes_cli.config import get_config_path

    path = get_config_path()
    try:
        with path.open(encoding="utf-8") as stream:
            parsed = yaml.safe_load(stream)
    except FileNotFoundError:
        return {}
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("user config root must be a mapping")
    return parsed


def _canonical_extra(config: dict[str, Any]) -> dict[str, Any]:
    current = config
    for part in ("gateway", "platforms", "buzz", "extra"):
        value = current.get(part)
        if value is None:
            value = {}
            current[part] = value
        if not isinstance(value, dict):
            raise ValueError("Buzz canonical config path must contain mappings")
        current = value
    return current


def _mapping_at(config: dict[str, Any], dotted: str) -> dict[str, Any]:
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return {}
        current = current.get(part)
    return current if isinstance(current, dict) else {}


def _legacy_fields(config: dict[str, Any]) -> set[str]:
    fields: set[str] = set()
    for prefix in _LEGACY_PREFIXES:
        candidate = _mapping_at(config, prefix)
        fields.update(field for field in _POLICY_FIELDS if field in candidate)
    return fields


def _policy_fields(config: dict[str, Any]) -> set[str]:
    prefixes = (_CANONICAL_PREFIX, *_LEGACY_PREFIXES)
    return {
        field
        for field in _POLICY_FIELDS
        if any(field in _mapping_at(config, prefix) for prefix in prefixes)
    }


def _managed_policy_state() -> tuple[set[str], bool]:
    """Return managed policy fields and whether inspection failed.

    Security-sensitive callers must not use the generic managed cache here:
    that cache deliberately turns read/parse failures into an empty overlay.
    """

    from hermes_cli import managed_scope

    managed_dir = managed_scope.get_managed_dir()
    if managed_dir is None:
        return set(), False
    managed_path = managed_dir / "config.yaml"
    if not managed_path.exists():
        return set(), False
    try:
        parsed = yaml.safe_load(managed_path.read_text(encoding="utf-8")) or {}
        if not isinstance(parsed, dict):
            raise ValueError("managed config root must be a mapping")
    except Exception:
        return set(), True
    return _policy_fields(parsed), False


def _environment_policy_metadata(
    profile: str | None,
) -> tuple[set[str], bool | None]:
    """Inspect only presence/activity, never return environment values."""

    try:
        getenv = settings._environment_getter(profile)
        direct = {
            field
            for field, name in settings._ENV_KEYS.items()
            if getenv(name) is not None
        }
        global_allow_all = str(getenv("GATEWAY_ALLOW_ALL_USERS") or "").strip().lower()
        global_allowed = str(getenv("GATEWAY_ALLOWED_USERS") or "").strip()
        global_active = bool(global_allowed) or global_allow_all in {
            "true", "1", "yes", "on"
        }
        return direct, global_active
    except Exception:
        return set(_POLICY_FIELDS), None


def _pairing_grants_active(profile: str | None) -> bool | None:
    """Detect a Buzz pairing grant without constructing a store or returning IDs."""

    try:
        home = settings._profile_home(profile)
        candidates = (
            home / "platforms" / "pairing" / "buzz-approved.json",
            home / "pairing" / "buzz-approved.json",
        )
        active = False
        for path in candidates:
            if not path.exists():
                continue
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                return None
            active = active or bool(parsed)
        return active
    except Exception:
        return None


def _environment_reference_fields(config: Any) -> set[str]:
    """Return policy fields whose configured value contains an env reference."""

    configured = settings._explicit_policy_values(config)

    def contains_reference(value: Any) -> bool:
        if isinstance(value, str):
            return re.search(r"\${[^}]+}", value) is not None
        if isinstance(value, dict):
            return any(contains_reference(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_reference(item) for item in value)
        return False

    return {
        field for field, value in configured.items() if contains_reference(value)
    }


def _managed_environment_reference_fields() -> set[str]:
    from hermes_cli import managed_scope

    managed_dir = managed_scope.get_managed_dir()
    if managed_dir is None:
        return set()
    path = managed_dir / "config.yaml"
    if not path.exists():
        return set()
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return _environment_reference_fields(parsed)
    except Exception:
        return set(_POLICY_FIELDS)


def _remove_legacy_policy_fields(config: dict[str, Any]) -> None:
    for prefix in _LEGACY_PREFIXES:
        candidate = _mapping_at(config, prefix)
        for field in _POLICY_FIELDS:
            candidate.pop(field, None)


def _policy_payload(profile: str | None) -> dict[str, Any]:
    from hermes_cli import web_server
    from hermes_cli.config import is_managed

    requested = (profile or "").strip()
    selected = requested if requested and requested.lower() != "current" else "current"
    with web_server._config_profile_scope(profile):
        try:
            raw = _read_user_config_strict()
            user_policy_unavailable = False
        except Exception:
            raw = {}
            user_policy_unavailable = True
        coarse_managed = is_managed()
    managed_fields, managed_error = _managed_policy_state()
    locked = (
        user_policy_unavailable
        or coarse_managed
        or managed_error
        or bool(managed_fields)
    )
    environment_fields, global_grants_active = _environment_policy_metadata(profile)
    reference_fields = _environment_reference_fields(raw)
    environment_fields.update(reference_fields)
    environment_fields.update(_managed_environment_reference_fields())
    if user_policy_unavailable or managed_error:
        environment_fields.update(_POLICY_FIELDS)
    if user_policy_unavailable:
        serialized_policy = dict.fromkeys(_POLICY_FIELDS)
    else:
        explicit = settings._explicit_policy_values(raw)
        determinate = {
            field: value
            for field, value in explicit.items()
            if field not in reference_fields
        }
        policy = (
            settings._RUNTIME_POLICY_LOADER.load(profile)
            if locked
            else settings.policy_from_config({"buzz": {"extra": determinate}})
        )
        serialized_policy: dict[str, Any] = deepcopy(policy)
    for field in environment_fields:
        serialized_policy[field] = None
    legacy_fields = _legacy_fields(raw)
    return {
        "profile": selected,
        "policy": serialized_policy,
        "environment_overrides": sorted(environment_fields),
        "indeterminate_fields": sorted(environment_fields),
        "ineffective_fields": sorted(environment_fields),
        "managed_fields": sorted(managed_fields),
        "locked": locked,
        "managed_error": managed_error,
        "user_policy_unavailable": user_policy_unavailable,
        "legacy_fields": sorted(legacy_fields),
        "legacy_cleanup_required": bool(legacy_fields),
        "additional_global_grants_active": global_grants_active,
        "additional_pairing_grants_active": _pairing_grants_active(profile),
    }


def _save_policy(body: BuzzPolicyUpdate, profile: str | None) -> dict[str, Any]:
    from hermes_cli import web_server
    from hermes_cli.config import is_managed, save_config

    policy = body.policy.model_dump(exclude_unset=True)
    with web_server._config_profile_scope(profile):
        with web_server._CONFIG_MUTATION_LOCK:
            managed_fields, managed_error = _managed_policy_state()
            if managed_error:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "managed_policy_unavailable",
                        "managed_fields": [],
                    },
                )
            if managed_fields:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "managed_policy",
                        "managed_fields": sorted(managed_fields),
                    },
                )
            if is_managed():
                raise HTTPException(
                    status_code=409,
                    detail={"error": "managed_install", "managed_fields": []},
                )
            try:
                existing = _read_user_config_strict()
                merged = deepcopy(existing)
                authoritative = settings._explicit_policy_values(existing)
                authoritative.update(policy)
                canonical = _canonical_extra(merged)
                for field, value in authoritative.items():
                    canonical[field] = deepcopy(value)
            except Exception as exc:
                raise HTTPException(
                    status_code=409,
                    detail={"error": "user_policy_unavailable"},
                ) from exc
            _remove_legacy_policy_fields(merged)
            preserve = {
                ("gateway", "platforms", "buzz", "extra", field)
                for field in authoritative
            }
            save_config(merged, preserve_keys=preserve)
    return _policy_payload(profile)


@router.get("/policy")
async def get_policy(profile: str | None = None):
    """Return normalized config plus privacy-minimal precedence metadata."""

    return await asyncio.to_thread(_policy_payload, profile)


@router.put("/policy")
async def put_policy(body: BuzzPolicyUpdate, profile: str | None = None):
    """Persist an explicit policy canonically and report override status."""

    return await asyncio.to_thread(_save_policy, body, profile)
