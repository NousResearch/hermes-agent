"""Account limits of the exact credential retained by an authorized session."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
import hashlib

from agent.account_usage_presentation import SUPPORTED_PROVIDERS
from gateway.session_busy_controls import authorize
from hermes_state_runtime import RuntimeStoreError


@dataclass(frozen=True)
class _Credential:
    provider: str
    base_url: str
    api_key: str = field(repr=False)
    api_mode: str


def _credential(provider, base_url, api_key, api_mode):
    return _Credential(
        provider.strip().lower() if isinstance(provider, str) else "",
        base_url.strip() if isinstance(base_url, str) else "",
        api_key.strip() if isinstance(api_key, str) else "",
        api_mode.strip().lower() if isinstance(api_mode, str) else "",
    )


async def _session_credential(authority, ref):
    agent = authority.agent(ref)
    if agent is not None:
        # * Callable keys may resolve another account; a read never invokes a credential source.
        return agent, _credential(getattr(agent, "provider", None), getattr(agent, "base_url", None),
                                  getattr(agent, "api_key", None), getattr(agent, "api_mode", None))
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    if policy is None:
        return None, None
    return policy, await asyncio.to_thread(_policy_credential, authority, policy)


def _policy_credential(authority, policy):
    from gateway.session_policy import profile_read_scope, route_launch_key
    from hermes_cli.runtime_provider import _maybe_apply_codex_app_server_runtime
    # * A launch key belongs only to its bound route, even when the session follows profile edits.
    with profile_read_scope(policy, authority):
        config, _ = policy.effective(authority, secrets=False)
        model = config.get("model") or {}
        try:
            key = route_launch_key(authority, policy, config)
        except RuntimeStoreError as exc:
            if exc.reason != "launch_credentials_unavailable":
                raise
            key = None
    credential = _credential(model.get("provider"), model.get("base_url"), key, model.get("api_mode"))
    mode = _maybe_apply_codex_app_server_runtime(provider=credential.provider, api_mode=credential.api_mode,
                                                model_cfg=model)
    return replace(credential, api_mode=mode)


def _reply(status, credential=None, *, reason=None, snapshot=None):
    provider = credential.provider if credential and credential.provider in SUPPORTED_PROVIDERS else None
    return {"status": status, "provider": provider, "reason": reason, "snapshot": snapshot}


def _fetch(credential):
    from agent.account_usage import fetch_account_usage
    from agent.account_usage_presentation import serialize_account_usage_snapshot
    # * A per-credential cache slot cannot replace the legacy provider-wide gauge of another login.
    identity = "fp:" + hashlib.sha256(credential.api_key.encode("utf-8")).hexdigest()[:16]
    snapshot = fetch_account_usage(credential.provider, base_url=credential.base_url, api_key=credential.api_key,
                                   read_only=True, identity_id=identity)
    if snapshot is None:
        return _reply("unavailable", credential, reason="fetch_failed")
    if snapshot.provider != credential.provider:
        return _reply("unavailable", credential, reason="usage_unavailable")
    view = serialize_account_usage_snapshot(snapshot)
    if view is None:
        return _reply("unavailable", credential, reason="usage_unavailable")
    return _reply("ok", credential, snapshot=view)


async def account_usage(connection, ref, params):
    """Read provider limits off-loop without constructing an agent, rotating auth or starting a turn."""
    if (set(params) - {"session_id", "profile"} or not isinstance(ref.session_id, str)
            or not ref.session_id):
        raise RuntimeStoreError("invalid_params")
    authority = connection.authority
    authorize(connection, ref, params, "session:read")
    owner, credential = await _session_credential(authority, ref)
    if credential is None:
        return _reply("unavailable", reason="credentials_unavailable")
    if credential.provider not in SUPPORTED_PROVIDERS:
        return _reply("unsupported", reason="provider_unsupported")
    if credential.api_mode == "codex_app_server":
        # * App-server authenticates independently; the retained HTTP key does not identify its account.
        return _reply("unsupported", credential, reason="route_unsupported")
    if not credential.api_key:
        return _reply("unavailable", credential, reason="credentials_unavailable")
    if credential.provider == "anthropic":
        from agent.anthropic_credentials import _is_oauth_token, is_native_anthropic_host
        if not is_native_anthropic_host(credential.base_url):
            return _reply("unsupported", credential, reason="route_unsupported")
        if not _is_oauth_token(credential.api_key):
            return _reply("unsupported", credential, reason="oauth_required")
    if credential.provider == "openrouter" and not credential.base_url:
        # * An unresolved endpoint must not borrow today's profile route for a historical key.
        return _reply("unavailable", credential, reason="credentials_unavailable")
    result = await asyncio.to_thread(_fetch, credential)
    authorize(connection, ref, params, "session:read")
    current_owner, current_credential = await _session_credential(authority, ref)
    if current_owner is not owner or current_credential != credential:
        return _reply("unavailable", reason="credentials_changed")
    return result
