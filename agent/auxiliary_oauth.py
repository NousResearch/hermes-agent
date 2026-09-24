"""Keep a main session's OAuth-proxy capability scoped to its auxiliary route.

``anthropic_oauth_proxy`` is resolved per provider AND per model
(``runtime_provider_custom._lift_model_capabilities`` lets
``providers.<name>.models.<model>.capabilities`` override the provider-level map), so two models
on one relay may legitimately disagree. The value decides Bearer vs ``x-api-key``, the Claude Code
beta/system/tool transforms, thinking-signature replay and the session-affinity header, so a
decision qualified only by provider + endpoint would let a main model's ``true`` lend wire
authority to an auxiliary model declared ``false`` (and lose a model-level ``true`` behind a
provider-level ``false``) immediately before the wire policy is chosen.

Hence: a live main runtime is inherited only when the auxiliary route is the SAME route —
same provider, same endpoint, same effective model. Anything else consumes what that route's own
model-qualified configuration declares, and a route declaring nothing gets nothing.
"""

from typing import Any, Dict, Optional

from hermes_cli.route_identity import normalize_route_base_url


def _bare_model(model: Any) -> str:
    """A model id comparable across routes: no ``vendor/`` prefix, case- and space-insensitive."""
    text = str(model or "").strip().lower()
    return text.rsplit("/", 1)[-1] if text else ""


def declared_route_capabilities(provider: Any, model: Any, base_url: Any = None) -> Dict[str, bool]:
    """The capability map *provider*'s config entry declares for *model* (``{}`` when none).

    Resolved by the canonical owner, so the provider-level map and its per-model override merge
    exactly as the main runtime resolves them — never a second interpretation of the same config.
    A ``vendor/model`` id matches a bare ``models:`` key (and the reverse): aggregator-prefixed and
    native spellings of one model are one route, and the prefix must not silently fall the lookup
    back to the provider-level value.

    A declaration is the entry's statement about its OWN endpoint: with a *base_url* that is not
    that endpoint (``same_provider_endpoint``) the answer is ``{}``. Read-only and credential-free
    (``named_custom_provider_entry``): this runs on every auxiliary call.
    """
    try:
        from hermes_cli.route_identity import same_provider_endpoint
        from hermes_cli.runtime_provider_custom import _lift_model_capabilities, named_custom_provider_entry
        found = named_custom_provider_entry(str(provider or ""))
        if not found:
            return {}
        entry, own_url = found
        if base_url and not same_provider_endpoint(own_url, base_url):
            return {}
        result: Dict[str, Any] = {}
        _lift_model_capabilities(entry, _entry_model_key(entry, model), result)
        capabilities = result.get("capabilities")
        return capabilities if isinstance(capabilities, dict) else {}
    except Exception:  # noqa: BLE001 — a config read must never break client construction
        return {}


def client_route_url(client: Any, base_url: Any = None) -> str:
    """*client*'s endpoint with the query the SDK split off into its default query.

    OpenAI/Anthropic SDK clients keep a query-bearing base URL as a clean ``base_url`` plus
    ``_custom_query``, so ``client.base_url`` alone has lost the tenant choice (``?team=a``). Route
    identity decisions need the URL the request actually goes to, query included."""
    from hermes_cli.route_identity import url_with_query
    base = str((getattr(client, "base_url", "") if base_url is None else base_url) or "").rstrip("/")
    for obj in (client, getattr(client, "_real_client", None)):
        query = getattr(obj, "_custom_query", None)
        if isinstance(query, dict) and query:
            return url_with_query(base, query)
    return base


def routed_client_capabilities(client: Any, provider: Any, model: Any) -> Dict[str, bool]:
    """Capability map for a session that switched onto *client* (``resolve_provider_client``).

    The declared map of *provider*'s entry for *model* at the client's own endpoint — the same map
    ``resolve_runtime_provider`` gives a runtime fallback. The client's own ``capabilities`` holds
    only the OAuth bit and exists only on the Anthropic wrapper, so an OpenAI-wire relay or any other
    declared key would be lost; it is used only when the route declares nothing."""
    declared = declared_route_capabilities(provider, model, client_route_url(client))
    if declared:
        return dict(declared)
    own = vars(client).get("capabilities") if hasattr(client, "__dict__") else None
    return dict(own) if isinstance(own, dict) else {}


def _entry_model_key(entry: Dict[str, Any], model: Any) -> Optional[str]:
    """The ``models:`` key of *entry* naming *model*, else *model* unchanged."""
    name = str(model or "").strip()
    if not name:
        return None
    models = entry.get("models")
    if not isinstance(models, dict) or name in models:
        return name
    bare = _bare_model(name)
    return next((key for key in models if _bare_model(key) == bare), name)


def declared_oauth_proxy(provider: Any, model: Any, base_url: Any = None) -> Optional[bool]:
    """``anthropic_oauth_proxy`` as *provider*'s entry declares it for *model* at *base_url*, else None."""
    value = declared_route_capabilities(provider, model, base_url).get("anthropic_oauth_proxy")
    return value if isinstance(value, bool) else None


def _inherited_oauth_proxy(main_runtime: Any, provider: Any, base_url: Any, model: Any) -> Optional[bool]:
    """The main runtime's value when the auxiliary route is that exact route, else None."""
    if not isinstance(main_runtime, dict):
        return None
    capabilities = main_runtime.get("capabilities")
    if not isinstance(capabilities, dict) or not isinstance(
        capabilities.get("anthropic_oauth_proxy"), bool
    ):
        return None
    runtime_base = main_runtime.get("base_url")
    if not base_url or not runtime_base:
        return None
    if normalize_route_base_url(base_url) != normalize_route_base_url(runtime_base):
        return None
    # A different model on the same endpoint is a different route for this decision: its own
    # declaration owns it. An unknown target model cannot be proven different, so it still
    # inherits — that is the pre-existing shape for callers that resolve no concrete model.
    target_model = _bare_model(model)
    if target_model and target_model != _bare_model(main_runtime.get("model")):
        return None
    target = str(provider or "").lower().removeprefix("custom:")
    source = (
        str(
            main_runtime.get("requested_provider") or main_runtime.get("provider") or ""
        )
        .lower()
        .removeprefix("custom:")
    )
    if target == source or target in {"auto", "main", "custom"}:
        return capabilities["anthropic_oauth_proxy"]
    return None


def runtime_oauth_proxy(
    main_runtime: Any, provider: Any, base_url: Any, model: Any = None,
) -> Optional[bool]:
    """OAuth-proxy policy for one auxiliary route, or None when nothing declares it.

    Inherits the main session's live value only for its own provider + endpoint + model; any other
    route — including a different model on the same relay — answers from its own model-qualified
    declaration, so a pin can neither borrow nor lose wire authority across models.

    A declaration belongs to the provider's OWN endpoint (``same_provider_endpoint``: origin plus
    path modulo ``/v1``). The same name pointed anywhere else (``auxiliary.<task>.base_url``, a
    ``fallback_chain`` entry, a sibling tenant path on the relay's host) is a different server: it
    gets the entry's key if the user composed it so, but never the Claude Code identity and
    Bearer-as-OAuth policy the relay declared for itself. An empty *base_url* means the entry's
    own endpoint. The main runtime carries capabilities only for its own endpoint
    (``resolve_runtime_provider``), so the inherited branch cannot smuggle them to another URL.
    """
    inherited = _inherited_oauth_proxy(main_runtime, provider, base_url, model)
    if inherited is not None:
        return inherited
    return declared_oauth_proxy(provider, model, base_url)


def named_route_identity(prov: Optional[str], base_url: Optional[str]) -> Optional[str]:
    """The named provider *prov* (normalized) when *base_url* is its own endpoint, else None.

    MoA slots, pinned routes and ``auxiliary.<task>`` blocks arrive with the endpoint their
    provider resolved to. That call IS the provider, not an anonymous ``custom`` endpoint: its
    per-provider and per-model declarations (``capabilities.anthropic_oauth_proxy``) are looked up
    by name, so flattening it strips the wire policy. Another endpoint under the same name
    (``same_provider_endpoint``) is a different route and stays ``custom``. A spaced display name
    is dashed like the entry lookup (``My Relay`` → ``my-relay``) — unless the dashed form is a
    built-in id or alias (``Claude Code`` → ``claude-code`` is the ``anthropic`` alias): the
    downstream resolver would then route it to the built-in. Such a name keeps its spaced
    spelling, so it can occupy a second client-cache slot beside the entry key; that costs a
    client, not correctness.
    """
    name = str(prov or "").strip().lower()
    if not name or name in {"auto", "custom"} or not base_url:
        return None
    dashed = name.replace(" ", "-")
    if dashed != name:
        from hermes_cli.auth import known_provider_id
        if known_provider_id(dashed) is None:
            name = dashed
    from hermes_cli.route_identity import named_provider_owns_endpoint
    return name if named_provider_owns_endpoint(name, base_url) else None


def affinity_capabilities(main_runtime: Any, provider: Any, route_url: Any, model: Any) -> Optional[Dict[str, bool]]:
    """Capabilities for the auxiliary conversation-affinity header on this route, or None.

    ``x-claude-code-session-id`` rides only where ``runtime_oauth_proxy`` says this provider,
    endpoint (tenant query included) and model are an OAuth-proxy route, so a model declaring
    itself off the relay's policy sends no such header."""
    if runtime_oauth_proxy(main_runtime, provider, str(route_url or ""), model):
        return {"anthropic_oauth_proxy": True}
    return None
