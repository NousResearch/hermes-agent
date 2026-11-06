"""Credential ladder for a named custom provider, split out of ``auxiliary_client``.

Aux resolves named custom providers here rather than through ``_resolve_named_custom_runtime``,
so a ``key_cmd`` has to be honoured at the same precedence or every aux call 401s. Kept in its
own module so the ladder can grow (#86446) without inflating the ``auxiliary_client`` facade.
"""

from __future__ import annotations

from typing import Any


def profile_dotenv_key(name: str) -> str:
    """``name``'s value in the served profile's ``.env``, or ``""``.

    ``_scoped_key_env`` treats a bound empty/foreign scope as authoritative and refuses the
    ``os.environ`` fallback (``serves_routed_profile``), so a ``key_env`` that only the profile
    ``.env`` defines reads as absent — aux then sends the ``no-key-required`` placeholder and
    401s on auth-required endpoints (#86446).

    The ``.env`` path comes from the bound scope's own stamp (``current_secret_scope_home()``),
    never from the ambient home: under multiplexing a kanban spawn-env or MCP owner scope binds
    ``profile_home=<served profile>`` while ``get_hermes_home()`` still resolves to the launch
    home, and reading that would hand a foreign-profile turn the launch profile's credential —
    the same leak class the scope refuses by design. The fallback to ``get_hermes_home()`` only
    runs when the scope is unstamped, which is also when ``serves_routed_profile()`` is False
    and the ambient home *is* the served profile. ``load_env_file`` is the tokenizer that scope
    is built from: memoised on the file's bytes, and total (an absent file yields ``{}``).
    """
    from pathlib import Path

    from agent.secret_scope import current_secret_scope_home, load_env_file
    from hermes_constants import get_hermes_home

    stamped = current_secret_scope_home()
    home = Path(stamped) if stamped else Path(get_hermes_home())
    return (load_env_file(home / ".env").get(name) or "").strip()


def _named_custom_api_key(custom_entry: dict[str, Any], provider: str, custom_base: str) -> Any:
    """Credential for a named custom provider: inline api_key → key_env → key_cmd → credential pool → placeholder."""
    from agent.auxiliary_client import _scoped_key_env

    custom_key: Any = (custom_entry.get("api_key") or "").strip()
    custom_key_env = (custom_entry.get("key_env") or custom_entry.get("api_key_env") or "").strip()
    if not custom_key and custom_key_env:
        custom_key = _scoped_key_env(custom_key_env) or profile_dotenv_key(custom_key_env)
    custom_key_cmd = str(custom_entry.get("key_cmd", "") or "").strip()
    if custom_key_cmd:
        from agent.command_token_source import build_command_token_provider
        custom_key = build_command_token_provider(custom_key_cmd, custom_entry.get("name") or provider) or custom_key
    if not custom_key:
        import contextlib

        with contextlib.suppress(Exception):
            from agent.credential_pool import custom_provider_pool_key_candidates, load_pool
            from agent.custom_entry_pool import entry_scoped_pool
            pool_name = custom_entry.get("provider_key") or custom_entry.get("name") or provider
            for pool_key in custom_provider_pool_key_candidates(custom_base, pool_name):
                try:
                    pool = load_pool(pool_key)
                except Exception:
                    continue
                if (pool := entry_scoped_pool(pool, pool_key, custom_entry)) is None or not pool.has_credentials():
                    continue
                if (pool_entry := pool.select()) is None:
                    continue
                pool_api_key = getattr(pool_entry, "runtime_api_key", None) or getattr(pool_entry, "access_token", "") or ""
                if str(pool_api_key).strip():
                    custom_key = str(pool_api_key).strip()
                    break
    return custom_key or "no-key-required"
