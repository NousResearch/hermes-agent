"""Route identity for auxiliary calls: custom-endpoint health checks and the provider allow-list."""
import contextlib
from typing import Any, Optional

from hermes_cli.route_identity import normalize_route_base_url

def _unhealthy_cache_key(provider: str, base_url: Optional[str] = None, model: Optional[str] = None) -> Any:
    """Provider-wide key, or endpoint-specific key for an explicit custom endpoint — prefixed with the
    active profile home: a 402 on profile A's account must not hide the provider from profile B's
    (differently funded) account in the same multiplexed process."""
    from agent.auxiliary_client import _normalize_chain_label, _normalize_resolved_model
    from hermes_constants import hermes_home_key
    label = _normalize_chain_label(provider)
    endpoint = normalize_route_base_url(_custom_health_base_url(provider, base_url))
    key = (hermes_home_key(), "custom-endpoint", endpoint) if endpoint else (hermes_home_key(), label)
    model = _normalize_resolved_model(model, provider)  # configured ``gpt-x`` and wire ``openai/gpt-x`` share a key
    return (*key, model.strip().lower()) if model and model.strip() else key


def _custom_health_base_url(provider: str, explicit_base_url: Optional[str] = None) -> str:
    """Return the concrete custom endpoint used to scope health and failed-route checks."""
    from agent.auxiliary_client import _current_custom_base_url
    explicit = str(explicit_base_url or "").strip()
    from agent.auxiliary_client import _normalize_chain_label
    label = _normalize_chain_label(provider)
    if label == "local/custom":
        return explicit or _current_custom_base_url()
    if label.startswith("custom:") and explicit:
        return explicit
    with contextlib.suppress(ImportError):
        from hermes_cli.runtime_provider import _get_named_custom_provider, _resolves_to_custom
        if _resolves_to_custom(label):
            return explicit or _current_custom_base_url()
        entry = _get_named_custom_provider(provider)
        if entry:
            return explicit or str(entry.get("base_url") or "").strip()
    return ""




# Quarantine hold per unavailable-reason label. Payment/quota depletion and a dead credential
# last hours, so those keep the long default TTL (None); a per-minute 429, a dropped connection or
# a garbled body clears in seconds — holding the lane for 10 minutes process-wide would hide a
# healthy fallback from every aux task over one transient blip.
_TRANSIENT_CANDIDATE_QUARANTINE_SECONDS = 60.0
_CANDIDATE_QUARANTINE_TTL: dict[str, Optional[float]] = {
    "rate limit": _TRANSIENT_CANDIDATE_QUARANTINE_SECONDS,
    "connection error": _TRANSIENT_CANDIDATE_QUARANTINE_SECONDS,
    "invalid provider response": _TRANSIENT_CANDIDATE_QUARANTINE_SECONDS,
    "provider overloaded": _TRANSIENT_CANDIDATE_QUARANTINE_SECONDS,
    "server error": _TRANSIENT_CANDIDATE_QUARANTINE_SECONDS,
}


def fallback_candidate_quarantine_ttl(reason: Optional[str]) -> Optional[float]:
    """Seconds to hide a fallback candidate for ``reason`` (a ``fallback_label`` label, or None
    for a stale credential); None means the long default TTL."""
    return _CANDIDATE_QUARANTINE_TTL.get(reason or "")


# Provider allow-list (``auxiliary.allowed_providers``). Opt-in: unset or [] keeps today's
# behaviour. Set, auxiliary routing skips a provider that is not listed before resolving or
# calling it (text or vision, primary or fallback), and each resolved client is checked again where
# it is handed over; ``auxiliary.free_only`` only filters OpenRouter SKUs. Identity is the provider name (``custom:<name>`` ==
# ``<name>``). A route with its own ``base_url`` override is ``custom``: its name does not say
# where it goes. A ``custom`` route without one (the live runtime of a ``providers:`` entry)
# counts as that entry when it calls the entry's own endpoint. Set but naming nothing allows
# nothing.


class AuxiliaryProviderNotAllowed(RuntimeError):
    """An auxiliary route names a provider outside ``auxiliary.allowed_providers``."""


def _allowlist_label(provider: Optional[str], base_url: Optional[str] = None) -> str:
    from agent.auxiliary_client import _fallback_provider_from_label, _normalize_aux_provider
    if base_url and str(base_url).strip():
        return "custom"
    label = _normalize_aux_provider(_fallback_provider_from_label(str(provider or "")))
    return "custom" if label == "local/custom" else label


def _allowed_labels(cfg: dict[str, Any]) -> Optional[frozenset]:
    """``auxiliary.allowed_providers`` as route identities; None means no restriction."""
    from agent.skill_utils import parse_config_string_list
    from hermes_cli.config import cfg_get
    raw = cfg_get(cfg, "auxiliary", "allowed_providers")
    if raw is None or raw == "" or raw == []:
        return None
    names = [str(name) for name in parse_config_string_list(raw) if str(name).strip()]
    return frozenset(_allowlist_label(name) for name in names) - {"auto", "api-key"}  # route labels name nothing


def _entry_endpoint(name: str, cfg: dict[str, Any]) -> str:
    """The normalized base_url of the ``providers:`` entry ``name``, or "" when it is not one."""
    from hermes_cli.providers import resolve_user_provider
    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    pdef = resolve_user_provider(name, providers)
    return normalize_route_base_url(pdef.base_url) if pdef is not None and pdef.base_url else ""


def _named_endpoint_label(endpoint: str, allowed: frozenset, cfg: dict[str, Any]) -> str:
    """The listed ``providers:`` entry whose configured base_url is ``endpoint``, or ""."""
    want = normalize_route_base_url(endpoint)
    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    for name in providers:
        label = _allowlist_label(str(name))
        if label in allowed and _entry_endpoint(str(name), cfg) == want:
            return label
    return ""


def aux_route_allowed(
    provider: Optional[str], base_url: Optional[str] = None, *, endpoint: Optional[str] = None,
) -> bool:
    """Whether an auxiliary route may be used. ``base_url`` is the route's own override (always
    ``custom``); ``endpoint`` is where a resolved route actually calls ("" for a client that exposes
    none, None when unknown), which identifies a ``custom`` route that has no override."""
    from hermes_cli.config import load_config_readonly
    cfg = load_config_readonly() or {}
    allowed = _allowed_labels(cfg)
    if allowed is None:
        return True
    label = _allowlist_label(provider, base_url)
    if label == "custom" and endpoint and not (base_url and str(base_url).strip()):
        label = _named_endpoint_label(endpoint, allowed, cfg) or label
    elif endpoint is not None and label in allowed:
        # A runtime can carry another base_url for a named entry: a listed ``providers:`` entry is
        # allowed only at its own endpoint (built-in providers are identified by name alone).
        own = _entry_endpoint(label, cfg)
        if own and own != normalize_route_base_url(endpoint or ""):
            return False
    return label in allowed


def aux_client_route(client: Any, provider: Optional[str]) -> str:
    """The provider an auxiliary client reaches: the backend a discovery lane built it for when the route
    names only the lane (``api-key``) or nothing (``auto``), else the route itself."""
    from agent.auxiliary_client import _fallback_provider_from_label, _normalize_aux_provider
    backend = getattr(client, "_hermes_aux_backend_provider", "")
    route = _normalize_aux_provider(_fallback_provider_from_label(str(provider or "")))
    return backend if isinstance(backend, str) and backend and route in {"auto", "api-key"} else str(provider or "")


def aux_client_allowed(client: Any, provider: Optional[str], base_url: Optional[str] = None) -> bool:
    """Whether a resolved auxiliary client may be used: its route (:func:`aux_client_route`) checked at
    the endpoint the client actually calls; a route that names no provider is refused. Unset, the
    client is not inspected."""
    from hermes_cli.config import load_config_readonly
    if _allowed_labels(load_config_readonly() or {}) is None:
        return True
    from agent.auxiliary_client import _effective_provider_for_client
    route = aux_client_route(client, _effective_provider_for_client(client, str(provider or "")))
    return aux_route_allowed(route, base_url, endpoint=str(getattr(client, "base_url", "") or ""))


def _not_allowed(task: Optional[str], label: str) -> AuxiliaryProviderNotAllowed:
    return AuxiliaryProviderNotAllowed(
        f"auxiliary {task or 'call'}: provider '{label}' is not in auxiliary.allowed_providers — "
        "add it there or route this task to a listed provider")


def require_aux_route_allowed(
    provider: Optional[str], task: Optional[str], base_url: Optional[str] = None,
    *, endpoint: Optional[str] = None,
) -> None:
    """Raise :class:`AuxiliaryProviderNotAllowed` for a route outside the allow-list, before resolution;
    a bare ``auto`` (no ``base_url`` of its own) is checked once resolved (:func:`require_aux_client_allowed`)."""
    bare_auto = str(provider or "").strip().lower() == "auto" and not (base_url and str(base_url).strip())
    if bare_auto or aux_route_allowed(provider, base_url, endpoint=endpoint):
        return
    raise _not_allowed(task, _allowlist_label(provider, base_url))


def require_aux_client_allowed(
    client: Any, provider: Optional[str], task: Optional[str], base_url: Optional[str] = None,
) -> None:
    """Raise :class:`AuxiliaryProviderNotAllowed` unless :func:`aux_client_allowed` admits the resolved client."""
    if not aux_client_allowed(client, provider, base_url):
        raise _not_allowed(task, _allowlist_label(aux_client_route(client, provider), base_url))
