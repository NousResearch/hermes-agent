"""Explicit, opt-in route-policy checks shared by model-resolution layers.

This module is deliberately pure: it performs no credential lookup, network I/O, or config writes.
Callers carry policy/provenance to the point they admit or dispatch a route.
"""
from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path
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


def profile_home_for_session_db(session_db: Any) -> Path | None:
    """Return a SessionDB profile home only when its path is under this installation's root."""
    db_path = getattr(session_db, "db_path", None)
    if db_path is None:
        return None
    try:
        from hermes_constants import get_default_hermes_root, named_profile_home

        # Preserve the path SessionDB was opened through before canonicalizing it:
        # ``<root>/profiles/<name>`` may itself be a live profile symlink.
        # Resolving first loses that owner boundary and incorrectly falls back
        # to the ambient profile's policy.
        logical_home = Path(db_path).expanduser().parent.absolute()
        logical_root = get_default_hermes_root().expanduser().absolute()
        logical_owner = named_profile_home(logical_home) or logical_home
        trusted = _trusted_profile_home(logical_owner, logical_root)
        if trusted is not None:
            return trusted

        home = logical_home.resolve(strict=False)
        root = logical_root.resolve(strict=False)
        return _trusted_profile_home(named_profile_home(home) or home, root)
    except (OSError, RuntimeError, ValueError):
        return None


def profile_home_for_config_path(config_path: str | Path) -> Path | None:
    """Return a config file's trusted profile owner, never inferring one for ad-hoc paths."""
    try:
        from hermes_constants import get_default_hermes_root, named_profile_home

        # Validate the config path as supplied before canonicalizing it: a live
        # ``<root>/profiles/<name>`` may point at a profile maintained elsewhere.
        # Resolving it first loses the named-profile authority and would let the
        # caller fall back to the ambient profile's policy.
        logical_path = Path(config_path).expanduser().absolute()
        if logical_path.name != "config.yaml":
            return None
        logical_root = get_default_hermes_root().expanduser().absolute()
        logical_home = logical_path.parent
        logical_owner = named_profile_home(logical_home) or logical_home
        trusted = _trusted_profile_home(logical_owner, logical_root)
        if trusted is not None:
            return trusted

        path = logical_path.resolve(strict=False)
        root = logical_root.resolve(strict=False)
        return _trusted_profile_home(named_profile_home(path.parent) or path.parent, root)
    except (OSError, RuntimeError, ValueError):
        return None


def _trusted_profile_home(home: Path, root: Path) -> Path | None:
    """Return only a repository-valid default or live named profile under *root*."""
    from hermes_constants import named_profile_home, named_profile_is_live
    from hermes_cli.profiles import _PROFILE_ID_RE

    if home == root:
        return home
    named = named_profile_home(home)
    if (
        named == home
        and home.parent == root / "profiles"
        and _PROFILE_ID_RE.fullmatch(home.name)
        and named_profile_is_live(home)
    ):
        return home
    return None


def current_routing_policy(profile_home: str | Path | None = None) -> Mapping[str, Any]:
    """Read routing policy for the explicit owner profile or the active execution scope."""
    from hermes_cli.config import load_config

    if profile_home is None:
        config = load_config()
    else:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        token = set_hermes_home_override(profile_home)
        try:
            config = load_config()
        finally:
            reset_hermes_home_override(token)
    if isinstance(config, Mapping) and isinstance(config.get("routing_policy"), Mapping):
        return config["routing_policy"]
    return {}


def current_routing_policy_for_session_db(session_db: Any) -> Mapping[str, Any]:
    """Resolve policy from a session store's durable owner, not process ambient home."""
    return current_routing_policy(profile_home_for_session_db(session_db))


def _policy_for_profile_home(profile_home: str | Path | None) -> Mapping[str, Any]:
    return current_routing_policy() if profile_home is None else current_routing_policy(profile_home)


def check_outbound_route(*, provider: str, model: str, base_url: str,
                         profile_home: str | Path | None = None) -> None:
    """Authoritative just-before-send guard used by shared transports."""
    check_route(_policy_for_profile_home(profile_home), provider=provider, model=model, base_url=base_url)


def check_persisted_route(*, provider: str, model: str, base_url: str,
                          profile_home: str | Path | None = None) -> None:
    """Reject an incomplete or denied route before it becomes durable state."""
    if not any(str(value or "").strip() for value in (provider, model, base_url)):
        return
    policy = _policy_for_profile_home(profile_home)
    check_requested_route(policy, requested_provider=provider, model=model)
    check_route(policy, provider=provider, model=model, base_url=base_url)


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
