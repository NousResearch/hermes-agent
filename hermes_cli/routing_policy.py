"""Validated terminal route-policy checks shared by resolution and transport layers."""
from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path
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
_model_policy_floor: dict[str, Any] | None = None


def _invalid(message: str) -> RoutingPolicyError:
    return RoutingPolicyError(f"invalid routing policy: {message}", code="invalid_policy")


def _host_value(value: str) -> str:
    raw = str(value or "").strip()
    if not raw or any(char.isspace() for char in raw):
        raise _invalid("base_url_hosts entries must be host names or absolute URLs")
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise _invalid("base_url_hosts entries must not contain credentials, queries, or fragments")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host or "/" in host or any(part == "" for part in host.split(".")):
        raise _invalid("base_url_hosts entries must contain a valid host")
    return host


def validate_routing_policy(policy: Mapping[str, Any] | None, *, selected: bool = False) -> dict[str, Any]:
    """Return a normalized, validated policy; selected floors must be enabled."""
    if policy is None:
        if selected:
            raise _invalid("selected policy must be a mapping")
        return {"enabled": False, "require_explicit": False, "deny": {}}
    if not isinstance(policy, Mapping):
        raise _invalid("policy must be a mapping")
    unknown = set(policy) - _POLICY_KEYS
    if unknown:
        raise _invalid(f"unsupported key(s): {', '.join(sorted(map(str, unknown)))}")
    enabled = policy.get("enabled", False)
    explicit = policy.get("require_explicit", False)
    if not isinstance(enabled, bool) or not isinstance(explicit, bool):
        raise _invalid("enabled and require_explicit must be booleans")
    if selected and not enabled:
        raise _invalid("a selected policy must set enabled: true")
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
        if key == "base_url_hosts":
            normalized[key] = tuple(_host_value(item) for item in values)
        else:
            normalized[key] = tuple(item.strip().lower() for item in values)
    return {"enabled": enabled, "require_explicit": explicit, "deny": normalized}


def select_model_policy_file(path: str | Path) -> None:
    """Activate an explicit process-wide non-weakenable policy floor before work starts."""
    global _model_policy_floor
    candidate = Path(path).expanduser()
    try:
        raw = candidate.read_text(encoding="utf-8")
    except OSError as exc:
        raise RoutingPolicyError(f"routing policy could not read selected file {candidate}", code="policy_file_unreadable") from exc
    try:
        import yaml
        loaded = yaml.safe_load(raw)
    except Exception as exc:
        raise _invalid("selected policy file is not valid YAML") from exc
    _model_policy_floor = validate_routing_policy(loaded, selected=True)


def clear_model_policy_floor() -> None:
    """Test/lifecycle hook: remove the process-local selected floor."""
    global _model_policy_floor
    _model_policy_floor = None


def _effective_policy(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    local = validate_routing_policy(policy)
    floor = _model_policy_floor
    if floor is None:
        return local
    return {
        "enabled": bool(local["enabled"] or floor["enabled"]),
        "require_explicit": bool(local["require_explicit"] or floor["require_explicit"]),
        "deny": {key: tuple(dict.fromkeys((*floor["deny"][key], *local["deny"][key]))) for key in _DENY_KEYS},
    }


def _enabled(policy: Mapping[str, Any] | None) -> bool:
    return bool(_effective_policy(policy)["enabled"])


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
    return base_url_hostname(str(value or "")).lower().rstrip(".")


def profile_home_for_session_db(session_db: Any) -> Path | None:
    """Return a SessionDB profile home only when its path is under this installation's root."""
    db_path = getattr(session_db, "db_path", None)
    if db_path is None:
        return None
    try:
        from hermes_constants import get_default_hermes_root, named_profile_home
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
    from hermes_constants import named_profile_home, named_profile_is_live
    from hermes_cli.profiles import _PROFILE_ID_RE
    if home == root:
        return home
    named = named_profile_home(home)
    if named == home and home.parent == root / "profiles" and _PROFILE_ID_RE.fullmatch(home.name) and named_profile_is_live(home):
        return home
    return None


def current_routing_policy(profile_home: str | Path | None = None) -> Mapping[str, Any]:
    """Read and validate policy for the explicit owner profile or active execution scope."""
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
    local = config.get("routing_policy") if isinstance(config, Mapping) else None
    return _effective_policy(local if isinstance(local, Mapping) else None)


def current_routing_policy_for_session_db(session_db: Any) -> Mapping[str, Any]:
    return current_routing_policy(profile_home_for_session_db(session_db))


def _policy_for_profile_home(profile_home: str | Path | None) -> Mapping[str, Any]:
    return current_routing_policy() if profile_home is None else current_routing_policy(profile_home)


def check_outbound_route(*, provider: str, model: str, base_url: str, profile_home: str | Path | None = None) -> None:
    check_route(_policy_for_profile_home(profile_home), provider=provider, model=model, base_url=base_url)


def check_persisted_route(*, provider: str, model: str, base_url: str, profile_home: str | Path | None = None) -> None:
    if not any(str(value or "").strip() for value in (provider, model, base_url)):
        return
    policy = _policy_for_profile_home(profile_home)
    check_requested_route(policy, requested_provider=provider, model=model)
    check_route(policy, provider=provider, model=model, base_url=base_url)


def check_requested_route(policy: Mapping[str, Any] | None, *, requested_provider: str, model: str) -> None:
    effective = _effective_policy(policy)
    if not effective["enabled"]:
        return
    if effective["require_explicit"] and (not requested_provider or requested_provider.strip().lower() == "auto"):
        raise RoutingPolicyError("routing policy requires an explicit provider; credential discovery is forbidden", code="implicit_provider")
    if effective["require_explicit"] and not str(model or "").strip():
        raise RoutingPolicyError("routing policy requires an explicit model; silent defaults are forbidden", code="implicit_model")


def check_route(policy: Mapping[str, Any] | None, *, provider: str, model: str, base_url: str) -> None:
    effective = _effective_policy(policy)
    if not effective["enabled"]:
        return
    provider_id = normalize_provider(str(provider or ""))
    denied_providers = {normalize_provider(item) for item in effective["deny"]["providers"]}
    if provider_id in denied_providers:
        raise RoutingPolicyError(f"routing policy denies provider '{provider_id}'", code="denied_provider")
    if any(fnmatchcase(form, pattern) for pattern in effective["deny"]["models"] for form in _model_forms(str(model or ""))):
        raise RoutingPolicyError("routing policy denies the selected model", code="denied_model")
    host = _canonical_host(base_url)
    if any(_host_matches(host, denied) for denied in effective["deny"]["base_url_hosts"]):
        raise RoutingPolicyError("routing policy denies the selected base-url host", code="denied_base_url")


def check_config_routes(config: Mapping[str, Any], *, profile_home: str | Path | None = None) -> None:
    """Reject durable model routes in a config candidate before its atomic write."""
    policy = config.get("routing_policy") if isinstance(config.get("routing_policy"), Mapping) else _policy_for_profile_home(profile_home)
    # Validate the candidate policy even if no route is currently set.
    _effective_policy(policy)

    def admit(route: Mapping[str, Any], *, provider_hint: str = "", base_url_hint: str = "") -> tuple[str, str]:
        # v12 provider definitions use ``api``/``default_model``; legacy
        # definitions use ``base_url``/``model``.  A nested model fragment
        # inherits only its parent endpoint/identity, while an explicit child
        # value wins exactly as it will at runtime.
        provider = str(route.get("provider") or route.get("requested_provider") or route.get("name") or provider_hint or "")
        model = str(route.get("model") or route.get("default_model") or route.get("default") or "")
        base_url = str(route.get("base_url") or route.get("api_base") or route.get("api") or base_url_hint or "")
        if any((provider, model, base_url)):
            check_requested_route(policy, requested_provider=provider, model=model)
            check_route(policy, provider=provider, model=model, base_url=base_url)
        return provider, base_url

    model = config.get("model")
    if isinstance(model, Mapping):
        admit(model)
    elif isinstance(model, str) and model:
        admit({"model": model})
    for key in ("fallback_providers", "fallback_model"):
        entries = config.get(key, [])
        for entry in entries if isinstance(entries, (list, tuple)) else [entries]:
            if isinstance(entry, Mapping):
                admit(entry)
    auxiliary = config.get("auxiliary", {})
    if isinstance(auxiliary, Mapping):
        stack = [auxiliary]
        while stack:
            entry = stack.pop()
            if isinstance(entry, Mapping):
                admit(entry)
                stack.extend(entry.values())
            elif isinstance(entry, (list, tuple)):
                stack.extend(entry)
    # Provider definitions are durable dispatch routes too.  ``providers`` is
    # the modern mapping; older configs retain ``custom_providers`` as a list
    # (and importers may preserve tuples).  Walk nested route fragments rather
    # than assuming one schema, because provider-specific options commonly
    # contain fallback/endpoint mappings.
    for key in ("providers", "custom_providers"):
        definitions = config.get(key)
        stack = [(definitions, "", "")]
        while stack:
            entry, provider_hint, base_url_hint = stack.pop()
            if isinstance(entry, Mapping):
                provider, base_url = admit(entry, provider_hint=provider_hint, base_url_hint=base_url_hint)
                for child_key, child in entry.items():
                    # A modern providers mapping names the provider by its key.
                    child_provider = str(child_key) if key == "providers" and not provider else provider
                    stack.append((child, child_provider, base_url))
            elif isinstance(entry, (list, tuple)):
                stack.extend((child, provider_hint, base_url_hint) for child in entry)
