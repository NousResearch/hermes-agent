"""Pure, inert crew-route and work-provenance contracts.

The policy in this module is checked against local catalogs only.  Resolving a
route never reads credentials, mutates storage, activates a worker, or contacts
a provider.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterable, Mapping

from agent.reasoning_effort import (
    EFFORT_LADDER,
    OPENAI_COMPAT_WIRE_EFFORTS,
    codex_supported_efforts,
)
from hermes_cli.codex_models import DEFAULT_CODEX_MODELS

ROUTE_POLICY_VERSION = "sunny-route-policy-v0"
TRUSTED_WORK_PROVENANCE = frozenset({"firstmate:approved-plan"})


class RouteContractError(ValueError):
    """An inert route or provenance contract is incomplete or unsupported."""


@dataclass(frozen=True)
class Route:
    provider: str
    model_id: str
    requested_effort: str
    applied_effort: str
    harness: str | None = None
    display_label: str | None = None

    @property
    def effort(self) -> str:
        return self.applied_effort

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.provider, self.model_id, self.applied_effort)


# Policy data, not an assertion about credentials, entitlement, or live reachability.
_LOCAL_MODELS: dict[str, frozenset[str]] = {
    "openai-codex": frozenset(DEFAULT_CODEX_MODELS),
    "anthropic": frozenset({"claude-opus-5"}),
    "openrouter": frozenset(
        {
            "openai/gpt-5.6-sol",
            "google/gemini-3.8-flash",
            "meta/muse-spark-1.3",
            "deepseek/deepseek-v4-flash-0731",
            "z-ai/glm-5.3-flash",
            "qwen/qwen3.8-flash",
            "x-ai/grok-4.6",
        }
    ),
}

_DISPLAY_LABELS = frozenset(
    label.casefold()
    for label in {
        "Hermes",
        "Zoro",
        "Validated-plan worker",
        "Worker/scout",
        "Codex-unavailable fallback",
        "No-mistakes",
        "fm-verify",
        "Sanji",
        "Jinbei",
        "Franky",
        "Usopp",
        "Robin",
        "Brook",
    }
)


def _required_text(value: Mapping[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item.strip():
        raise RouteContractError(f"missing route field: {name}")
    return item.strip()


def resolve_route(value: Mapping[str, Any]) -> Route:
    """Resolve one exact route solely against the checked-in policy catalog.

    An unsupported effort is rejected rather than clamped: a lease capability
    is an identity assertion, so silently substituting a nearby wire level
    would make two different declarations compare equal.
    """
    if not isinstance(value, Mapping):
        raise RouteContractError("route must be a mapping")
    aliases = ("model", "provider_ref", "model_ref", "effort_ref",
               "requested_effort", "applied_effort")
    ambiguous = [name for name in aliases if name in value]
    if ambiguous:
        raise RouteContractError(
            f"ambiguous route identity fields: {', '.join(ambiguous)}"
        )
    provider = _required_text(value, "provider").lower()
    model_id = _required_text(value, "model_id")
    requested = _required_text(value, "effort").lower()
    display_raw = value.get("display_label")
    harness_raw = value.get("harness")
    if display_raw is not None and not isinstance(display_raw, str):
        raise RouteContractError("display_label must be text")
    if harness_raw is not None and not isinstance(harness_raw, str):
        raise RouteContractError("harness must be text")
    display_label = (display_raw or "").strip() or None
    harness = (harness_raw or "").strip() or None

    if provider not in _LOCAL_MODELS:
        raise RouteContractError(f"unknown provider: {provider}")
    if requested not in EFFORT_LADDER:
        raise RouteContractError(f"malformed effort: {requested}")
    if model_id.casefold() in _DISPLAY_LABELS or (
        display_label and model_id.casefold() == display_label.casefold()
    ):
        raise RouteContractError("display labels are not model identifiers")
    if model_id not in _LOCAL_MODELS[provider]:
        raise RouteContractError(f"model is not in local {provider} catalog: {model_id}")

    supported = (
        codex_supported_efforts(model_id)
        if provider == "openai-codex"
        else OPENAI_COMPAT_WIRE_EFFORTS
    )
    if requested not in supported:
        raise RouteContractError(
            f"effort is not supported by {provider}/{model_id}: {requested}"
        )
    return Route(provider, model_id, requested, requested, harness, display_label)


def _route(
    provider: str,
    model_id: str,
    effort: str,
    display_label: str,
    *,
    harness: str | None = None,
) -> Route:
    return resolve_route(
        {
            "provider": provider,
            "model_id": model_id,
            "effort": effort,
            "harness": harness,
            "display_label": display_label,
        }
    )


def sunny_routes() -> dict[str, Route]:
    """Captain-approved, inert operational policy fixtures."""
    return {
        "hermes": _route("openai-codex", "gpt-5.6-luna", "xhigh", "Hermes", harness="codex"),
        "zoro": _route("openai-codex", "gpt-5.6-luna", "xhigh", "Zoro", harness="codex"),
        "validated-plan-worker": _route(
            "openai-codex", "gpt-5.6-sol", "low", "Validated-plan worker", harness="codex"
        ),
        "worker-scout": _route(
            "openai-codex", "gpt-5.6-luna", "xhigh", "Worker/scout", harness="codex"
        ),
        "codex-unavailable": _route(
            "anthropic", "claude-opus-5", "xhigh", "Codex-unavailable fallback"
        ),
        "no-mistakes": _route("anthropic", "claude-opus-5", "xhigh", "No-mistakes"),
        "fm-verify": _route("openrouter", "openai/gpt-5.6-sol", "high", "fm-verify"),
    }


def parliament_routes() -> dict[str, Route]:
    """Parliament fixtures pinned to the OpenRouter/Nous route."""
    models = {
        "sanji": "google/gemini-3.8-flash",
        "jinbei": "meta/muse-spark-1.3",
        "franky": "deepseek/deepseek-v4-flash-0731",
        "usopp": "z-ai/glm-5.3-flash",
        "robin": "qwen/qwen3.8-flash",
        "brook": "x-ai/grok-4.6",
    }
    return {
        name: _route("openrouter", model, "xhigh", name.title())
        for name, model in models.items()
    }


@dataclass(frozen=True)
class WorkIdentity:
    bucket_key: str
    pool_member: str
    business: str
    account: str
    board: str
    project: str
    worktree: str
    idempotency_key: str
    parent_link: str
    route_policy_version: str
    provenance: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkIdentity":
        if not isinstance(value, Mapping):
            raise RouteContractError("work identity must be a mapping")
        names = tuple(field.name for field in fields(cls))
        missing = [
            name
            for name in names
            if not isinstance(value.get(name), str) or not value[name].strip()
        ]
        if missing:
            raise RouteContractError(f"missing work identity fields: {', '.join(missing)}")
        normalized = {name: value[name].strip() for name in names}
        if normalized["route_policy_version"] != ROUTE_POLICY_VERSION:
            raise RouteContractError("unknown route-policy version")
        if normalized["provenance"] not in TRUSTED_WORK_PROVENANCE:
            raise RouteContractError("untrusted work provenance")
        return cls(**normalized)


def validate_work_batch(
    values: Iterable[WorkIdentity], *, allowed_parent_links: set[str] | frozenset[str]
) -> tuple[WorkIdentity, ...]:
    """Validate a batch without consulting or mutating a database."""
    items = tuple(values)
    seen: set[str] = set()
    scope: tuple[str, str, str, str, str] | None = None
    for item in items:
        if not isinstance(item, WorkIdentity):
            raise RouteContractError("work batch entries must be WorkIdentity values")
        if item.idempotency_key in seen:
            raise RouteContractError(f"duplicate idempotency key: {item.idempotency_key}")
        seen.add(item.idempotency_key)
        item_scope = (
            item.bucket_key,
            item.business,
            item.account,
            item.board,
            item.project,
        )
        if scope is None:
            scope = item_scope
        elif item_scope != scope:
            raise RouteContractError("cross-scope work reference")
        if item.parent_link not in allowed_parent_links:
            raise RouteContractError("cross-scope parent reference")
    return items
