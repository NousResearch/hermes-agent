"""Keep Anthropic credential declarations attached to the exact runtime key."""

import re
from typing import Any, Optional


def is_oauth_env_source(source: str) -> bool:
    """OAuth environment sources include their numbered rotation slots."""
    return bool(re.fullmatch(r"env:(ANTHROPIC_TOKEN|CLAUDE_CODE_OAUTH_TOKEN)(?:_\d+)?", source))


def pool_api_key_declaration(pool: Any, token: Any) -> Optional[bool]:
    """Read metadata for this key without advancing the credential pool."""
    if not isinstance(token, str) or not pool or pool.provider != "anthropic":
        return None
    entry_id = pool.entry_id_for_api_key(token)
    for entry in pool.entries():
        if entry.id == entry_id and entry.runtime_api_key == token:
            return entry.auth_type == "api_key" if entry.auth_type is not None else None
    return None


def resolve_aux_credential(explicit_key: Any, main_runtime: Any) -> tuple[Any, Any, Optional[bool]]:
    """An explicit config key declares API auth; a forwarded key retains its origin."""
    from agent import auxiliary_client as aux
    from agent.anthropic_credentials import resolve_anthropic_token

    if explicit_key:
        declaration = None if callable(explicit_key) else True
        runtime = aux._normalize_main_runtime(main_runtime) if main_runtime is not None else {}
        if runtime.get("provider") == "anthropic" and runtime.get("api_key") == explicit_key:
            declaration = runtime.get("anthropic_api_key_declared")
            if declaration is None:
                declaration = pool_api_key_declaration(aux._load_pool_with_credentials("anthropic"), explicit_key)
        return explicit_key, None, declaration
    _, entry = aux._select_pool_entry("anthropic")
    if entry is None:
        return resolve_anthropic_token(), None, None
    token = aux._pool_runtime_api_key(entry)
    kind = getattr(entry, "auth_type", None)
    declaration = kind == "api_key" if kind is not None and not callable(token) else None
    return token, entry, declaration


def _build_anthropic_client_from_runtime(agent: Any, rt: dict[str, Any]) -> None:
    """Rebuild the native Anthropic client from a ``_primary_runtime`` snapshot."""
    from agent.anthropic_adapter import build_anthropic_client
    from hermes_cli.timeouts import get_provider_request_timeout

    agent._anthropic_api_key = rt["anthropic_api_key"]
    agent._anthropic_base_url = rt["anthropic_base_url"]
    agent._anthropic_api_key_declared = rt.get("anthropic_api_key_declared")
    agent._anthropic_client = build_anthropic_client(
        rt["anthropic_api_key"], rt["anthropic_base_url"],
        timeout=get_provider_request_timeout(agent.provider, agent.model),
        declared_api_key=agent._anthropic_api_key_declared,
    )
    agent._is_anthropic_oauth = rt["is_anthropic_oauth"]
    agent.client = None


def propagate_refreshed_key(agent: Any, old_token: str, new_token: str) -> None:
    """Move every holder of a revoked token without keeping its source declaration."""
    from agent.auxiliary_key_rotation import rotate_runtime_main_api_key

    agent._anthropic_api_key_declared = None  # resolver output: shape + env declaration decide
    agent._anthropic_api_key, agent._is_anthropic_oauth = new_token, agent._anthropic_oauth_flag(new_token)
    # Claude Code revokes the old token on refresh; compression and fallback restore must move too.
    if agent.api_key == old_token:
        agent.api_key = new_token
    cc = getattr(agent, "context_compressor", None)
    if cc is not None and getattr(cc, "api_key", None) == old_token:
        cc.api_key = new_token
    rt = getattr(agent, "_primary_runtime", None) or {}
    if rt.get("anthropic_api_key") == old_token:
        rt["is_anthropic_oauth"] = agent._is_anthropic_oauth
        rt.pop("anthropic_api_key_declared", None)
    for k in ("api_key", "anthropic_api_key", "compressor_api_key"):
        if rt.get(k) == old_token:
            rt[k] = new_token
    # The turn may have published its runtime before a request worker refreshed this token.
    rotate_runtime_main_api_key(old_token, new_token)
