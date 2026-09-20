"""Move a user from a memory provider that left core onto its catalog plugin.

A bundled ``plugins/memory/<name>`` that becomes a standalone catalog plugin keeps the same provider
name, config section (``memory.<name>``), data directory and tool names, so the migration is only
"the code now lives under ``HERMES_HOME/plugins/<name>``". Two hooks call :func:`migrate_home`:

* ``hermes update`` — for every profile home that shares the venv (primary; runs where the venv was
  just rebuilt anyway).
* agent init — when the configured provider cannot be found at all, once per home/provider per process (Desktop
  users update through the app and never run ``hermes update`` by hand).

Both install the catalog entry at its reviewed pin through the normal plugin install path (kill
list, dependency constraints, enable), never a custom source. Offline or absent from the catalog:
the user gets the exact one-liner instead of silently running without memory.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Lock
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_attempted: set[tuple[str, str]] = set()
_attempted_lock = Lock()


@contextmanager
def _migration_scope(home: Path):
    """Bind ``home``'s config, secrets and terminal policy; keep the caller's own scope for its own home."""
    from agent.secret_scope import current_secret_scope, serves_routed_profile
    from hermes_constants import hermes_home_key
    from tools.terminal_scope import get_terminal_scope

    same_home = hermes_home_key(home) == hermes_home_key()
    secrets = current_secret_scope() if same_home else None
    # Unscoped single-profile startup already resolved config from the launch environment.
    if same_home and secrets is None and not serves_routed_profile():
        yield
    elif secrets is not None and get_terminal_scope() is not None:
        yield
    else:
        # Not gateway.run's scope helper: importing it boots the gateway's process env (dotenv included).
        from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
        from hermes_cli.env_loader import hydrate_profile_secret_sources
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from tools.terminal_scope import install_and_reset_profile_terminal_scope

        with ExitStack() as stack:
            stack.callback(reset_hermes_home_override, set_hermes_home_override(home))
            if secrets is None:
                hydrate_profile_secret_sources(home)
                secrets = build_profile_secret_scope(home)
            stack.callback(reset_secret_scope, set_secret_scope(secrets))
            stack.enter_context(install_and_reset_profile_terminal_scope(home))
            yield


def configured_provider(home: Path) -> str:
    """``memory.provider`` of *home*'s effective config, or ``""``."""
    from hermes_cli.plugin_python_deps import _read_home_config
    with _migration_scope(home):
        memory = _read_home_config(home).get("memory") or {}
    return str(memory.get("provider") or "").strip()


def provider_present(name: str, home: Path) -> bool:
    """True when the provider resolves anywhere Hermes looks for *home* (bundled, that home's user
    plugins, entry point). The lookup reads the active home, so it is bound explicitly: the update
    hook walks several profile homes from one process."""
    from plugins.memory import find_provider_dir
    with _migration_scope(home):
        return find_provider_dir(name) is not None


def catalog_source(name: str) -> Optional[str]:
    """The catalog entry that ships provider *name*, or None when the catalog has no such plugin."""
    from hermes_cli.plugin_catalog import get_live_catalog_entry
    entry = get_live_catalog_entry(name)
    return entry.name if entry is not None else None


def migrate_home(home: Path, *, install: Callable[[str], dict], say: Callable[[str], None] = print) -> Optional[str]:
    """Install the configured provider's catalog plugin into *home* when the provider is gone.

    Returns the installed plugin name, or None when nothing needed doing or the install could not
    happen (already reported through *say*). Never raises: memory being down must not take the
    update or the agent down with it.
    """
    name = configured_provider(home)
    if not name or provider_present(name, home):
        return None
    if catalog_source(name) is None:
        say(f"  ⚠ Memory provider '{name}' is configured but not installed and not in the plugin catalog. "
            f"Install it with `hermes plugins install <source>` or change memory.provider.")
        return None
    try:
        result = install(name)
    except Exception as exc:  # network, uv, kill list — report, do not raise
        result = {"ok": False, "error": str(exc)}
    if result.get("ok"):
        say(f"  ✓ Memory provider '{name}' moved out of core — installed its plugin from the catalog "
            f"(your memory.{name} settings and data are unchanged).")
        return name
    say(f"  ⚠ Memory provider '{name}' moved out of core and could not be installed automatically: "
        f"{result.get('error') or 'unknown error'}. Run `hermes plugins install {name}`.")
    return None


def _install_into(home: Path) -> Callable[[str], dict]:
    def _install(name: str) -> dict:
        from hermes_cli.plugins_cmd import dashboard_install_plugin
        with _migration_scope(home):
            return dashboard_install_plugin("", force=False, enable=True, catalog_name=name)
    return _install


def migrate_all_homes(*, say: Callable[[str], None] = print) -> list[str]:
    """``hermes update`` hook: every profile home sharing this venv. Returns installed plugin names."""
    from hermes_cli.plugin_python_deps import dependency_homes
    installed: list[str] = []
    for home in dependency_homes():
        try:
            name = migrate_home(home, install=_install_into(home), say=say)
        except Exception as exc:
            logger.debug("memory provider migration skipped for %s: %s", home, exc)
            continue
        if name:
            installed.append(name)
    return installed


def recover_at_startup(name: str) -> bool:
    """One automatic attempt per resolved home/provider per process; explicit installs stay available.

    Honours ``security.allow_lazy_installs`` because it installs code. True when installed.
    """
    from hermes_constants import get_hermes_home, hermes_home_key
    from tools.lazy_deps import _allow_lazy_installs

    home = Path(get_hermes_home())
    key = (hermes_home_key(home), name)
    with _attempted_lock:  # Claim before the slow install so a second startup does not wait on it.
        attempted = key in _attempted
        _attempted.add(key)
    if attempted:
        logger.warning("Memory provider '%s' automatic recovery already attempted for %s; "
                       "run `hermes plugins install %s` in that profile.", name, home, name)
        return False
    with _migration_scope(home):
        if not _allow_lazy_installs():
            logger.warning("Memory provider '%s' is not installed; security.allow_lazy_installs is off — "
                           "run `hermes plugins install %s`.", name, name)
            return False
        return migrate_home(home, install=_install_into(home), say=logger.warning) == name
