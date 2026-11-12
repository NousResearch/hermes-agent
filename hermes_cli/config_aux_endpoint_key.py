"""Storage for an auxiliary task's custom-endpoint API key: ``.env``, referenced from config.yaml.

Same storage as the main custom-endpoint flow (#69449): the key lives in ``.env`` and
``auxiliary.<task>.api_key`` holds a ``${VAR}`` reference the config loader expands.
"""
from __future__ import annotations

import urllib.parse


def store_aux_endpoint_key(slot: str, base_url: str, api_key: str) -> str:
    """Save an auxiliary endpoint key to ``.env`` and return the ``${VAR}`` config.yaml stores.

    The aux prefix and *slot* keep it from overwriting the main endpoint's key for the same host.
    An existing ``${VAR}`` reference passes through unchanged. Raises ``ValueError`` when the
    ``.env`` write lock refuses the variable: ``save_env_value`` refuses silently, and a reference
    to an unwritten variable would drop the typed key without telling anyone."""
    from hermes_cli.config import custom_endpoint_key_env, require_env_writable, save_env_value
    key = str(api_key or "").strip()
    if not key or (key.startswith("${") and key.endswith("}")):
        return key
    parsed = urllib.parse.urlparse(str(base_url or ""))
    host = f"{parsed.hostname or ''}_{parsed.port}" if parsed.port else (parsed.hostname or "")
    var = custom_endpoint_key_env(f"aux_{slot}_{host}")
    require_env_writable(var, "set")
    save_env_value(var, key)
    return f"${{{var}}}"
