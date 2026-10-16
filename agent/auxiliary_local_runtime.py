"""Managed llama.cpp endpoint for auxiliary and fallback client resolution.

A bare ``llamacpp`` / ``llama.cpp`` / ``llama-cpp`` alias (no base_url, no configured provider of
that name) means "the local server Hermes manages for this profile" — exactly what the main ladder
resolves in ``hermes_cli.runtime_provider_custom._resolve_llamacpp_runtime``. With no server
running the answer is "unavailable", never the generic custom/API-key discovery: that handed the
local model slug to whatever cloud provider held a key (#119227) or to the primary's own endpoint.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from hermes_cli.local_runtime.endpoint import LLAMACPP_ALIASES


def bare_llamacpp_endpoint(provider: Optional[str], base_url: Optional[str],
                           api_key: Any) -> Optional[tuple[str, Any]]:
    """``(base_url, api_key)`` of the profile's local llama.cpp server for a bare alias.

    ``base_url`` is ``""`` when nothing is serving. ``None`` when the request is not a bare alias:
    an explicit base_url or a configured provider entry under the alias name resolves as before.
    Read on every call so a restarted server's new port/key is picked up.
    """
    alias = (provider or "").strip().lower()
    if alias not in LLAMACPP_ALIASES or base_url:
        return None
    from hermes_cli.config import load_config
    from hermes_cli.local_runtime.endpoint import resolve_llamacpp_endpoint
    from hermes_cli.runtime_provider import _get_named_custom_provider
    if _get_named_custom_provider(alias):
        return None
    endpoint = resolve_llamacpp_endpoint(config=load_config()) or {}
    return (str(endpoint.get("base_url") or "").strip(),
            api_key or endpoint.get("api_key") or "no-key-required")
