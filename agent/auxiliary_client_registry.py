"""``auth_type`` dispatch table for the auxiliary registry branch.

``_resolve_registry_branch`` in ``agent/auxiliary_client`` routes every
``PROVIDER_REGISTRY`` provider here on its registered ``auth_type``; this table is the
registry twin of the facade's ``_EXPLICIT_PROVIDER_BRANCHES``. Arms late-import the
facade (siblings never form a module-level cycle) so the seam every test patches —
``agent.auxiliary_client.*`` — is the module production reads.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Dict

if TYPE_CHECKING:
    from agent.auxiliary_client import _ResolveRequest, _ResolveResult

logger = logging.getLogger("agent.auxiliary_client")

# First occurrence surfaces for diagnostics; per-call retries stay silent (the
# contract test_auxiliary_client_resolve_dedup.py pins for every fall-through).
_LOGGED_MINIMAX_ABSENT_KEYS: set = set()
_LOGGED_MINIMAX_UNEXPECTED_KEYS: set = set()


def _resolve_vertex_arm(req: _ResolveRequest) -> _ResolveResult:
    from agent.auxiliary_client import _build_vertex_client, _route_client
    client, final_model = _build_vertex_client(req.provider, req.model)
    return _route_client(req, client, final_model) if client is not None else (None, None)


def _resolve_bedrock_arm(req: _ResolveRequest) -> _ResolveResult:
    from agent.auxiliary_client import _build_bedrock_client, _route_client
    client, final_model = _build_bedrock_client(req.provider, req.model, raw_codex=req.raw_codex)
    return _route_client(req, client, final_model) if client is not None else (None, None)


def _log_once_debug(seen: set, key: str, msg: str, *args: object) -> None:
    if key not in seen:
        seen.add(key)
        logger.debug(msg, *args)


def _resolve_minimax_oauth_arm(req: _ResolveRequest) -> _ResolveResult:
    """``minimax-oauth`` → AnthropicAuxiliaryClient over a callable bearer; (None, None) when absent.

    MiniMax OAuth resolves to an Anthropic-compatible inference endpoint with a callable
    bearer: tokens live ~15 minutes and the Anthropic SDK re-invokes the provider on each
    request, so a static string would 401 mid-session. ``is_oauth`` comes from
    ``anthropic_route_is_oauth`` — the MiniMax host is a third-party Anthropic-protocol
    endpoint, so the wrapper must NOT carry the native Claude Code OAuth identity (mcp__
    tool-name wire transforms, identity rewrites, response prefix stripping); those are
    api.anthropic.com-only (#114967).
    """
    from hermes_cli.auth_constants import AuthError

    from agent.auxiliary_client import (
        AnthropicAuxiliaryClient, _AuxProbeClientStub, _aux_probe_active,
        _get_aux_model_for_provider, _normalize_resolved_model, _route_client,
    )
    from agent.anthropic_credentials import anthropic_route_is_oauth
    from hermes_cli.auth import get_provider_auth_state, resolve_minimax_oauth_runtime_credentials

    # Probe mode answers "resolvable?" for availability gates and must not touch the
    # network: read the raw persisted state (access_token + inference_base_url present →
    # stub) and skip the refresh an expired token would otherwise trigger (mirrors
    # credential_pool.py::_seed_minimax_singleton).
    if _aux_probe_active():
        state = get_provider_auth_state("minimax-oauth")
        base_url = str((state or {}).get("inference_base_url") or "").strip().rstrip("/")
        if not ((state or {}).get("access_token") and base_url):
            _log_once_debug(_LOGGED_MINIMAX_ABSENT_KEYS, "minimax-oauth",
                            "resolve_provider_client: minimax-oauth not logged in")
            return None, None
        final_model = _normalize_resolved_model(
            req.model or _get_aux_model_for_provider(req.provider) or "MiniMax-M3", req.provider,
        )
        return _route_client(req, _AuxProbeClientStub(api_key="", base_url=base_url), final_model)

    try:
        credentials = resolve_minimax_oauth_runtime_credentials(as_token_provider=True)
    except AuthError as exc:
        # Expected absent/not-logged-in/quarantined states: the resolver contract is
        # "absent → (None, None), never an exception" (the aux ladder then falls through
        # to its Step-2 chain). Debug-once, like every other fall-through — a per-call
        # WARNING would recreate the #21521 log spam once a quarantine wipes the tokens.
        _log_once_debug(_LOGGED_MINIMAX_ABSENT_KEYS, "minimax-oauth",
                        "resolve_provider_client: minimax-oauth unavailable: %s", exc)
        return None, None
    except Exception as exc:
        # Deliberate boundary (BLE001): a genuinely unexpected failure still ends the arm
        # with (None, None) — never an exception into the aux ladder — but keeps the
        # traceback, logged once.
        if "minimax-oauth" not in _LOGGED_MINIMAX_UNEXPECTED_KEYS:
            _LOGGED_MINIMAX_UNEXPECTED_KEYS.add("minimax-oauth")
            logger.warning(
                "resolve_provider_client: unexpected minimax-oauth resolution failure: %s", exc,
                exc_info=True)
        return None, None
    token_provider = credentials.get("api_key")
    base_url = str(credentials.get("base_url") or "").strip().rstrip("/")
    if not callable(token_provider) or not base_url:
        _log_once_debug(_LOGGED_MINIMAX_ABSENT_KEYS, "minimax-oauth",
                        "resolve_provider_client: minimax-oauth credentials incomplete")
        return None, None
    final_model = _normalize_resolved_model(
        req.model or _get_aux_model_for_provider(req.provider) or "MiniMax-M3", req.provider,
    )
    from agent.anthropic_adapter import build_anthropic_client  # SDK-absent boundary, like _try_anthropic
    try:
        real_client = build_anthropic_client(token_provider, base_url)
    except ImportError:
        return None, None
    client = AnthropicAuxiliaryClient(
        real_client, final_model, token_provider, base_url,
        is_oauth=anthropic_route_is_oauth(base_url, token_provider))
    return _route_client(req, client, final_model)


# ``api_key`` / ``external_process`` arms stay on the facade (they share its credential
# resolvers); the table maps the auth types whose arms are build-and-route one-liners
# or live here as topical arms.
REGISTRY_AUTHTYPE_ARMS: Dict[str, Callable[[_ResolveRequest], _ResolveResult]] = {
    "vertex": _resolve_vertex_arm,
    "aws_sdk": _resolve_bedrock_arm,
    "oauth_minimax": _resolve_minimax_oauth_arm,
}
