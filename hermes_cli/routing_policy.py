"""Explicit, opt-in route-policy checks shared by model-resolution layers.

This module is deliberately pure: it performs no credential lookup, network I/O, or config writes.
Callers carry policy/provenance to the point they admit or dispatch a route.
"""
from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any, Mapping

from hermes_cli.providers import normalize_provider
from utils import base_url_hostname


class RoutingPolicyError(RuntimeError):
    """A terminal rejection of a model route by operator policy."""


def _enabled(policy: Mapping[str, Any] | None) -> bool:
    return bool(isinstance(policy, Mapping) and policy.get("enabled") is True)


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip().lower() for item in value if str(item).strip())


def _deny(policy: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = policy.get("deny")
    return _strings(value.get(key)) if isinstance(value, Mapping) else ()


def _model_forms(model: str) -> tuple[str, ...]:
    raw = model.strip().lower()
    if not raw:
        return ()
    without_tag = raw.split("[", 1)[0]
    parts = without_tag.split(":")
    bare = parts[0]
    variants = tuple(parts[1:])
    tails = tuple(part.rsplit("/", 1)[-1] for part in parts)
    return tuple(dict.fromkeys((raw, without_tag, bare, *variants, *tails)))


def _host_matches(host: str, denied: str) -> bool:
    return host == denied or host.endswith("." + denied)


def _canonical_host(value: str) -> str:
    """Normalize an operator-supplied host or URL before deny matching."""
    return base_url_hostname(str(value or "")).lower().rstrip(".")


def current_routing_policy() -> Mapping[str, Any]:
    """Read the active profile policy at an execution boundary."""
    from hermes_cli.config import load_config

    config = load_config()
    if isinstance(config, Mapping) and isinstance(config.get("routing_policy"), Mapping):
        return config["routing_policy"]
    return {}


def check_outbound_route(*, provider: str, model: str, base_url: str) -> None:
    """Authoritative just-before-send guard used by shared transports."""
    check_route(current_routing_policy(), provider=provider, model=model, base_url=base_url)


def check_requested_route(policy: Mapping[str, Any] | None, *, requested_provider: str, model: str) -> None:
    """Reject credential discovery and silent model resolution when explicitly required."""
    if not isinstance(policy, Mapping) or not _enabled(policy):
        return
    if policy.get("require_explicit") is True and (not requested_provider or requested_provider.strip().lower() == "auto"):
        raise RoutingPolicyError("routing policy requires an explicit provider; credential discovery is forbidden")
    if policy.get("require_explicit") is True and not str(model or "").strip():
        raise RoutingPolicyError("routing policy requires an explicit model; silent defaults are forbidden")


def check_route(policy: Mapping[str, Any] | None, *, provider: str, model: str, base_url: str) -> None:
    """Raise ``RoutingPolicyError`` when a complete effective route is denied."""
    if not isinstance(policy, Mapping) or not _enabled(policy):
        return
    provider_id = normalize_provider(str(provider or ""))
    denied_providers = {normalize_provider(item) for item in _deny(policy, "providers")}
    if provider_id in denied_providers:
        raise RoutingPolicyError(f"routing policy denies provider '{provider_id}'")
    patterns = _deny(policy, "models")
    if any(fnmatchcase(form, pattern) for pattern in patterns for form in _model_forms(str(model or ""))):
        raise RoutingPolicyError("routing policy denies the selected model")
    host = _canonical_host(base_url)
    if any(_host_matches(host, _canonical_host(denied)) for denied in _deny(policy, "base_url_hosts")):
        raise RoutingPolicyError("routing policy denies the selected base-url host")
