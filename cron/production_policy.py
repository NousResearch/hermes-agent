"""Process-latched cron execution boundary for sealed Cloud Muncho.

Production startup activates this latch after the exact config/environment and
writer-boundary contract is verified, before any inbound adapter can exist.
Release and live-adapter identity are attested separately before READY. Once
active the latch cannot be disabled inside the process. Job management and
execution both reuse the same static validator, so a post-READY create/update
cannot introduce another model/provider route, widen the reviewed tool schema,
or bypass the public-connector delivery path.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping, Sequence


class ProductionCronPolicyError(RuntimeError):
    """A cron job does not satisfy the active production route contract."""


_lock = threading.Lock()
_active = False
_PRODUCTION_CRON_PROVIDER = "openai-codex"
_PRODUCTION_CRON_MODEL = "gpt-5.6-sol"


def activate_production_cron_policy() -> None:
    """Latch the production cron boundary on for this process."""

    global _active
    with _lock:
        _active = True


def production_cron_policy_active() -> bool:
    with _lock:
        return _active


def pin_production_cron_create_route(
    provider: Any,
    model: Any,
) -> tuple[str | None, str | None]:
    """Pin omitted create-time axes without overriding explicit values.

    Generic Hermes cron jobs may inherit either axis from mutable global
    configuration.  The sealed production runtime cannot: when its latch is
    active, an omitted axis is mechanically filled with the exact attested
    route before snapshots or persistence.  A caller-supplied alternate is
    rejected instead of being silently rewritten.
    """

    if not production_cron_policy_active():
        return provider, model
    if provider is not None and not isinstance(provider, str):
        raise ProductionCronPolicyError(
            "production_cron_policy_blocked:production_cron_primary_route_not_exact"
        )
    if model is not None and not isinstance(model, str):
        raise ProductionCronPolicyError(
            "production_cron_policy_blocked:production_cron_primary_route_not_exact"
        )
    normalized_provider = (provider.strip() or None) if provider is not None else None
    normalized_model = (model.strip() or None) if model is not None else None
    if (
        normalized_provider is not None
        and normalized_provider != _PRODUCTION_CRON_PROVIDER
    ):
        raise ProductionCronPolicyError(
            "production_cron_policy_blocked:production_cron_primary_route_not_exact"
        )
    if normalized_model is not None and normalized_model != _PRODUCTION_CRON_MODEL:
        raise ProductionCronPolicyError(
            "production_cron_policy_blocked:production_cron_primary_route_not_exact"
        )
    return (
        normalized_provider or _PRODUCTION_CRON_PROVIDER,
        normalized_model or _PRODUCTION_CRON_MODEL,
    )


def enforce_production_cron_jobs(
    jobs: Sequence[Mapping[str, Any]],
) -> None:
    """Validate an entire post-mutation job set when the latch is active."""

    if not production_cron_policy_active():
        return
    try:
        from gateway.production_model_sovereignty_runtime import (
            validate_production_cron_jobs,
        )

        validate_production_cron_jobs(jobs)
    except Exception as exc:
        code = str(getattr(exc, "code", "") or exc or "invalid")
        raise ProductionCronPolicyError(
            f"production_cron_policy_blocked:{code}"
        ) from exc


def enforce_production_cron_job(job: Mapping[str, Any]) -> None:
    """Validate one job when the process latch is active."""

    enforce_production_cron_jobs([job])


def resolve_production_cron_toolsets(
    job: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[str] | None:
    """Return the exact production tool surface, or ``None`` when inactive.

    The inactive sentinel lets the generic scheduler retain its normal
    per-job/platform resolution behavior.  Once the process latch is active,
    the production resolver either returns a concrete reviewed list or raises;
    it never falls through to AIAgent's full default tool surface.
    """

    if not production_cron_policy_active():
        return None
    try:
        from gateway.production_model_sovereignty_runtime import (
            resolve_production_cron_enabled_toolsets,
        )

        return resolve_production_cron_enabled_toolsets(dict(job), config)
    except Exception as exc:
        code = str(getattr(exc, "code", "") or exc or "invalid")
        raise ProductionCronPolicyError(
            f"production_cron_policy_blocked:{code}"
        ) from exc


__all__ = [
    "ProductionCronPolicyError",
    "activate_production_cron_policy",
    "enforce_production_cron_job",
    "enforce_production_cron_jobs",
    "pin_production_cron_create_route",
    "production_cron_policy_active",
    "resolve_production_cron_toolsets",
]
