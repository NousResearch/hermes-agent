"""Terminal route-policy checks shared by resolution and final wire dispatch."""
from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any, Mapping
from urllib.parse import urlparse

from hermes_cli.providers import normalize_provider
from utils import base_url_hostname


class RoutingPolicyError(RuntimeError):
    """A terminal, safe-to-render rejection of a model route by operator policy."""

    def __init__(self, message: str, *, code: str = "route_denied") -> None:
        self.code = code
        super().__init__(message)


_POLICY_KEYS = frozenset({"enabled", "require_explicit", "deny"})
_DENY_KEYS = frozenset({"providers", "models", "base_url_hosts"})
_DEFAULT_POLICY = {"enabled": False, "require_explicit": False, "deny": {}}


def _invalid(message: str) -> RoutingPolicyError:
    return RoutingPolicyError(f"invalid routing policy: {message}", code="invalid_policy")


def _host_value(value: str) -> str:
    raw = str(value or "").strip()
    if not raw or any(char.isspace() for char in raw):
        raise _invalid("base_url_hosts entries must be host names or absolute URLs")
    try:
        parsed = urlparse(raw if "://" in raw else f"//{raw}")
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError as exc:
        raise _invalid("base_url_hosts entries must contain a valid host") from exc
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise _invalid("base_url_hosts entries must not contain credentials, queries, or fragments")
    if not host or "/" in host or any(part == "" for part in host.split(".")):
        raise _invalid("base_url_hosts entries must contain a valid host")
    return host


def validate_routing_policy(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a normalized validated policy; absent policy retains legacy behavior."""
    if policy is None:
        policy = _DEFAULT_POLICY
    if not isinstance(policy, Mapping):
        raise _invalid("policy must be a mapping")
    unknown = set(policy) - _POLICY_KEYS
    if unknown:
        raise _invalid(f"unsupported key(s): {', '.join(sorted(map(str, unknown)))}")
    enabled = policy.get("enabled", False)
    explicit = policy.get("require_explicit", False)
    if not isinstance(enabled, bool) or not isinstance(explicit, bool):
        raise _invalid("enabled and require_explicit must be booleans")
    deny = policy.get("deny", {})
    if not isinstance(deny, Mapping):
        raise _invalid("deny must be a mapping")
    unknown_deny = set(deny) - _DENY_KEYS
    if unknown_deny:
        raise _invalid(f"unsupported deny key(s): {', '.join(sorted(map(str, unknown_deny)))}")
    normalized: dict[str, tuple[str, ...]] = {}
    for key in _DENY_KEYS:
        values = deny.get(key, [])
        if not isinstance(values, (list, tuple)) or any(not isinstance(item, str) or not item.strip() for item in values):
            raise _invalid(f"deny.{key} must be a list of non-empty strings")
        normalized[key] = (
            tuple(_host_value(item) for item in values)
            if key == "base_url_hosts"
            else tuple(item.strip().lower() for item in values)
        )
    return {"enabled": enabled, "require_explicit": explicit, "deny": normalized}


def current_routing_policy() -> Mapping[str, Any]:
    """Load policy from the currently bound profile scope only."""
    from hermes_cli.config import read_raw_config_readonly

    config = read_raw_config_readonly()
    if not isinstance(config, Mapping) or "routing_policy" not in config:
        return validate_routing_policy(None)
    policy = config["routing_policy"]
    if policy is None:
        raise _invalid("policy must be a mapping")
    return validate_routing_policy(policy)


def effective_wire_model(api_kwargs: Mapping[str, Any], fallback: Any = "") -> Any:
    """Return the model the SDK will send after its ``extra_body`` merge."""
    extra_body = api_kwargs.get("extra_body")
    if isinstance(extra_body, Mapping) and "model" in extra_body:
        return extra_body["model"]
    return api_kwargs.get("model") or fallback


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


def check_requested_route(policy: Mapping[str, Any] | None, *, requested_provider: str, model: str) -> None:
    """Reject implicit route discovery before the resolver can advance its ladder."""
    effective = validate_routing_policy(policy)
    if not effective["enabled"]:
        return
    if effective["require_explicit"] and (not requested_provider or requested_provider.strip().lower() == "auto"):
        raise RoutingPolicyError("routing policy requires an explicit provider; credential discovery is forbidden", code="implicit_provider")
    if effective["require_explicit"] and not str(model or "").strip():
        raise RoutingPolicyError("routing policy requires an explicit model; silent defaults are forbidden", code="implicit_model")


def check_route(policy: Mapping[str, Any] | None, *, provider: str, model: str, base_url: str) -> None:
    """Reject a resolved physical route; callers must not recover from this exception."""
    effective = validate_routing_policy(policy)
    if not effective["enabled"]:
        return
    provider_id = normalize_provider(str(provider or ""))
    denied_providers = {normalize_provider(item) for item in effective["deny"]["providers"]}
    if provider_id in denied_providers:
        raise RoutingPolicyError(f"routing policy denies provider '{provider_id}'", code="denied_provider")
    if any(fnmatchcase(form, pattern) for pattern in effective["deny"]["models"] for form in _model_forms(str(model or ""))):
        raise RoutingPolicyError("routing policy denies the selected model", code="denied_model")
    host = base_url_hostname(str(base_url or "")).lower().rstrip(".")
    if any(_host_matches(host, denied) for denied in effective["deny"]["base_url_hosts"]):
        raise RoutingPolicyError("routing policy denies the selected base-url host", code="denied_base_url")
