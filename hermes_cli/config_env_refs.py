"""Expand ``${VAR}`` / ``${env:VAR}`` refs in config values.

Split out of ``hermes_cli.config`` (which re-exports every name here) so the
config facade stays under its size cap. MCP stdio ``args`` keep an escaped
``$${VAR}`` for the child process instead of expanding it at load time.
"""

import logging
import os
import re
from typing import Any, Optional

# Same logger name as before the split, so log filters and caplog keep working.
logger = logging.getLogger("hermes_cli.config")

_ENV_REF_RE = re.compile(r"\${([^}]+)}")


def _env_ref_lookup(name: str) -> Optional[str]:
    """Resolve the env var behind a ``${VAR}`` / ``${env:VAR}`` ref — plain ``os.environ`` outside
    a profile secret scope (legacy behavior for the default profile).

    Inside a scope (a multiplexed gateway turn, a secondary profile's config load, a cron job) the read goes
    through ``agent.secret_scope.get_secret`` so the ref resolves against *that* profile's ``.env``: under
    multiplexing a miss is a miss, never another profile's ``os.environ`` value (#84079 — every profile
    "had" the default profile's ``${MATRIX_ACCESS_TOKEN}`` and fanned out). Same policy as
    ``gateway.config._getenv`` and ``get_env_value``.
    """
    try:
        from agent.secret_scope import current_secret_scope, get_secret as _get_secret
    except Exception:
        return os.environ.get(name)
    if current_secret_scope() is None:
        return os.environ.get(name)
    return _get_secret(name)


def _env_expand_match(m: re.Match) -> str:
    """Expand one ``${VAR}`` (legacy bare name) or ``${env:VAR}`` (Cursor-style SecretRef).
    Other SecretRef sources (``file:``, ``bitwarden:``, ``vault:``...) are NOT resolved here:
    external backends inject their values into the environment at startup (the ``secrets:``
    block), so a config ref only ever needs the env shape. Unresolved refs stay verbatim so
    callers can detect them."""
    raw = m.group(0)
    inner = m.group(1).strip()
    name = _env_ref_var_name(inner)
    if name is None:
        if not inner.startswith("env:") and _is_non_env_secret_ref(inner):
            logger.warning(
                "Config ref %r uses source %r which is not resolvable in "
                "config.yaml — external secret sources inject env vars at "
                "startup, so reference the variable as ${env:NAME} instead",
                raw, inner.split(":", 1)[0])
        return raw  # non-env source, or empty ``${env:}``
    val = _env_ref_lookup(name)
    if val is not None:
        return val
    if inner.startswith("env:"):
        logger.warning(
            "Config ref %r: %s is not set (check ~/.hermes/.env); "
            "keeping the literal placeholder", raw, name)
    return raw


def _is_non_env_secret_ref(ref: str) -> bool:
    """True for a SecretRef body with a non-``env`` source (``bitwarden:FOO``, ``vault:...``)."""
    return ":" in ref and re.match(r"^[a-z][a-z0-9_-]*:", ref) is not None


def _env_ref_var_name(ref: str) -> Optional[str]:
    """Env-var name a ``${...}`` body reads, or None for a non-env source / empty ``env:``."""
    ref = ref.strip()
    if ref.startswith("env:"):
        return ref[len("env:"):].strip() or None
    if _is_non_env_secret_ref(ref):
        return None
    return ref


# ``${VAR}`` not preceded by ``$``: ``$${VAR}`` is the MCP stdio-args deferral escape.
_UNESCAPED_ENV_REF_RE = re.compile(r"(?<!\$)\${([^}]+)}")


def _expand_mcp_server_env_vars(server: dict[str, Any]) -> dict[str, Any]:
    """Expand one MCP server config, deferring escaped ``$${VAR}`` refs in ``args``."""
    return {
        key: _expand_env_vars(value, preserve_escaped=key == "args")
        for key, value in server.items()
    }


def _expand_env_vars(obj, *, preserve_escaped: bool = False):
    """Recursively expand ``${VAR}`` / ``${env:VAR}`` in string values (keys/non-strings untouched).
    MCP stdio ``args`` preserve ``$${VAR}`` for the child process; elsewhere it keeps legacy
    interpolation."""
    if isinstance(obj, str):
        pattern = _UNESCAPED_ENV_REF_RE if preserve_escaped else _ENV_REF_RE
        return pattern.sub(_env_expand_match, obj)
    if isinstance(obj, dict):
        expanded = {}
        for key, value in obj.items():
            if key == "mcp_servers" and isinstance(value, dict):
                expanded[key] = {
                    name: _expand_mcp_server_env_vars(server)
                    if isinstance(server, dict)
                    else _expand_env_vars(server)
                    for name, server in value.items()
                }
            else:
                expanded[key] = _expand_env_vars(
                    value, preserve_escaped=preserve_escaped)
        return expanded
    if isinstance(obj, list):
        return [
            _expand_env_vars(item, preserve_escaped=preserve_escaped)
            for item in obj
        ]
    return obj
