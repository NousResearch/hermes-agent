"""Atomic, compare-and-swap activation of Hermes' complete main-model route."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from hermes_constants import get_hermes_home

_ROUTE_FIELDS = ("default", "provider", "api_mode")
_STALE_ENDPOINT_FIELDS = ("api_key", "api", "base_url", "context_length")


class ModelActivationError(RuntimeError):
    """Base error for a rejected model activation."""


class ModelActivationCASMismatch(ModelActivationError):
    """The active route changed after the caller observed it."""


@dataclass(frozen=True)
class ModelRoute:
    model: str
    provider: str
    api_mode: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ModelRoute":
        model = str(raw.get("model") or raw.get("default") or "").strip()
        provider = str(raw.get("provider") or "").strip()
        api_mode = str(raw.get("api_mode") or raw.get("transport") or "").strip()
        if not model or not provider or not api_mode:
            raise ValueError("model, provider, and api_mode are required")
        return cls(model=model, provider=provider, api_mode=api_mode)

    def as_config(self) -> dict[str, str]:
        return {"default": self.model, "provider": self.provider, "api_mode": self.api_mode}


@dataclass(frozen=True)
class ActivationResult:
    old_route: ModelRoute
    new_route: ModelRoute
    old_fingerprint: str
    new_fingerprint: str
    generation: int
    runtime_rollover_required: bool


def route_fingerprint(route: ModelRoute | Mapping[str, Any]) -> str:
    value = route if isinstance(route, ModelRoute) else ModelRoute.from_mapping(route)
    payload = json.dumps(value.as_config(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _current_route(config: Mapping[str, Any]) -> ModelRoute:
    model_cfg = config.get("model")
    if not isinstance(model_cfg, Mapping):
        raise ModelActivationError("current model config is missing")
    try:
        return ModelRoute.from_mapping(model_cfg)
    except ValueError as exc:
        raise ModelActivationError("current model route is incomplete") from exc


def _validate_target(route: ModelRoute, config: Mapping[str, Any]) -> None:
    from hermes_cli.providers import (
        TRANSPORT_TO_API_MODE,
        host_mandated_api_mode,
        resolve_provider_full,
    )

    providers = config.get("providers")
    custom_providers = config.get("custom_providers")
    provider = resolve_provider_full(
        route.provider,
        user_providers=providers if isinstance(providers, dict) else None,
        custom_providers=custom_providers if isinstance(custom_providers, list) else None,
    )
    if provider is None:
        raise ModelActivationError(f"unknown provider: {route.provider}")

    supported_modes = set(TRANSPORT_TO_API_MODE.values())
    if route.api_mode not in supported_modes:
        raise ModelActivationError(f"unsupported api_mode: {route.api_mode}")

    mandated = host_mandated_api_mode(provider.base_url)
    declared = TRANSPORT_TO_API_MODE.get(provider.transport)
    required = mandated or declared
    if required and route.api_mode != required:
        raise ModelActivationError(
            f"provider {route.provider} requires api_mode {required}, not {route.api_mode}"
        )


def _lock_path() -> Path:
    return Path(get_hermes_home()) / "runtime" / "model-activation.lock"


class _ActivationLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    def __enter__(self) -> "_ActivationLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                self._handle.write(b"0")
                self._handle.flush()
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except Exception as exc:
            self._handle.close()
            self._handle = None
            raise ModelActivationError("model activation lock unavailable") from exc
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def activate_model_profile(
    profile: ModelRoute | Mapping[str, Any],
    *,
    expected_current: ModelRoute | Mapping[str, Any] | None = None,
    expected_fingerprint: str | None = None,
    expected_generation: int | None = None,
    mirror_legacy_main: bool = True,
) -> ActivationResult:
    """CAS-update model/default/provider/api_mode in one durable YAML write.

    The raw user YAML is read and mutated under both the in-process config lock
    and an inter-process file lock, preserving environment-reference templates.
    Existing endpoint-specific fields are removed when the provider changes.
    """
    from hermes_cli import managed_scope
    from hermes_cli.config import (
        _CONFIG_LOCK,
        _LOAD_CONFIG_CACHE,
        _RAW_CONFIG_CACHE,
        atomic_config_write,
        get_config_path,
        is_managed,
    )
    from utils import fast_safe_load

    if is_managed():
        raise ModelActivationError("configuration is managed and cannot be activated")
    managed = managed_scope.managed_config_keys()
    blocked = sorted({key for key in managed if key in {
        "model.default", "model.provider", "model.api_mode", "model.main",
        "model.routing_generation",
    }})
    if blocked:
        raise ModelActivationError("model route is managed: " + ", ".join(blocked))

    target = profile if isinstance(profile, ModelRoute) else ModelRoute.from_mapping(profile)
    expected = None
    if expected_current is not None:
        expected = (
            expected_current
            if isinstance(expected_current, ModelRoute)
            else ModelRoute.from_mapping(expected_current)
        )

    config_path = get_config_path()
    with _CONFIG_LOCK, _ActivationLock(_lock_path()):
        if not config_path.exists():
            raise ModelActivationError("current config.yaml is missing")
        try:
            loaded = fast_safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            raise ModelActivationError("current config.yaml is unreadable") from exc
        if not isinstance(loaded, dict):
            raise ModelActivationError("current config root must be a mapping")
        raw = copy.deepcopy(loaded)
        current = _current_route(raw)
        current_fingerprint = route_fingerprint(current)
        generation_raw = raw.get("model", {}).get("routing_generation", 0)
        generation = generation_raw if isinstance(generation_raw, int) and not isinstance(generation_raw, bool) else 0

        if expected is not None and current != expected:
            raise ModelActivationCASMismatch("current model route does not match expected_current")
        if expected_fingerprint is not None and current_fingerprint != expected_fingerprint:
            raise ModelActivationCASMismatch("current model route fingerprint changed")
        if expected_generation is not None and generation != expected_generation:
            raise ModelActivationCASMismatch(
                f"expected generation {expected_generation}, found {generation}"
            )

        _validate_target(target, raw)
        model_cfg = raw.setdefault("model", {})
        if not isinstance(model_cfg, dict):
            raise ModelActivationError("model config must be a mapping")
        if current.provider != target.provider:
            for field in _STALE_ENDPOINT_FIELDS:
                model_cfg.pop(field, None)
        model_cfg.update(target.as_config())
        if mirror_legacy_main:
            model_cfg["main"] = target.model
        else:
            model_cfg.pop("main", None)
        new_generation = generation + 1
        model_cfg["routing_generation"] = new_generation

        atomic_config_write(config_path, raw, sort_keys=False)
        path_key = str(config_path)
        _RAW_CONFIG_CACHE.pop(path_key, None)
        _LOAD_CONFIG_CACHE.pop(path_key, None)

    return ActivationResult(
        old_route=current,
        new_route=target,
        old_fingerprint=current_fingerprint,
        new_fingerprint=route_fingerprint(target),
        generation=new_generation,
        runtime_rollover_required=current != target,
    )


__all__ = [
    "ActivationResult",
    "ModelActivationCASMismatch",
    "ModelActivationError",
    "ModelRoute",
    "activate_model_profile",
    "route_fingerprint",
]
