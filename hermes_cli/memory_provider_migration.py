"""Move a user from a memory provider that left core onto its catalog plugin.

A bundled ``plugins/memory/<name>`` that becomes a standalone catalog plugin keeps the same provider
name, the settings it already read, its data directory and tool names, so the migration is only
"the code now lives under ``HERMES_HOME/plugins/<name>``". Two hooks call :func:`migrate_home`:

* ``hermes update`` — for every profile home that shares the venv (primary; runs where the venv was
  just rebuilt anyway).
* agent init — when the configured provider cannot be found at all, once per process (Desktop
  users update through the app and never run ``hermes update`` by hand).

Both install the catalog entry at its reviewed pin through the normal plugin install path (kill
list, dependency constraints, enable), never a custom source. Every outcome — installed, refused,
failed, absent from the catalog — reaches the user (terminal, Desktop, chat platform), never only
``agent.log``: a provider that silently stays missing is lost memory.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_attempted: set[tuple[str, str]] = set()

# Providers that shipped bundled in core. Unattended consent covers only these: their users already
# accepted the deps when they picked the built-in. Any other catalog plugin named in memory.provider
# (a synced config, a cloned profile) still needs an explicit install.
_LEFT_CORE = frozenset({"hindsight", "honcho", "mem0", "supermemory", "openviking", "retaindb", "byterover", "holographic"})


def configured_provider(home: Path) -> str:
    """``memory.provider`` of *home*'s effective config, or ``""``."""
    from pm.plugins_state import read_home_selection
    memory = (read_home_selection(home) or {}).get("memory") or {}
    return str(memory.get("provider") or "").strip()


def provider_present(name: str, home: Path) -> bool:
    """True when the provider resolves anywhere Hermes looks for *home* (bundled, that home's user
    plugins, entry point). The lookup reads the active home, so it is bound explicitly: the update
    hook walks several profile homes from one process."""
    from plugins.memory import find_provider_dir
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(home)
    try:
        return find_provider_dir(name) is not None
    finally:
        reset_hermes_home_override(token)


def catalog_source(name: str) -> Optional[str]:
    """The catalog entry that ships provider *name*, or None when the catalog has no such plugin."""
    from hermes_cli.plugin_catalog import get_live_catalog_entry
    entry = get_live_catalog_entry(name)
    return entry.name if entry is not None else None


def _pending_provider(home: Path, *, say: Callable[[str], None]) -> Optional[str]:
    """The provider *home* needs from the catalog, or None (nothing to do, or a catalog miss already
    reported through *say*). Read-only."""
    name = configured_provider(home)
    from agent.memory_provider import is_core_memory_provider
    if is_core_memory_provider(name) or provider_present(name, home):
        return None
    if catalog_source(name) is None:
        say(f"  ⚠ Memory provider '{name}' is configured but not installed and not in the plugin catalog. "
            f"Install it with `hermes plugins install <source>` or change memory.provider.")
        return None
    return name


def _install_command(name: str, home: Path) -> str:
    """The exact command that installs *name* into *home*. ``-p`` is dropped only for the default
    home while no sticky profile is set: a bare command run from a shell targets the sticky
    profile, never the home of the agent (Desktop, gateway, ``hermes -p``) that printed it."""
    from hermes_cli.profiles import get_active_profile
    from hermes_constants import profile_name_for_home
    profile = profile_name_for_home(home)
    if profile is None or (profile == "default" and get_active_profile() == "default"):
        return f"hermes plugins install {name}"
    return f"hermes -p {profile} plugins install {name}"


def _install_pending(home: Path, name: str, *, install: Callable[[str], dict],
                     say: Callable[[str], None]) -> bool:
    try:
        result = install(name)
    except Exception as exc:  # network, uv, kill list — report, do not raise
        result = {"ok": False, "error": str(exc)}
    if result.get("ok"):
        say(f"  ✓ Memory provider '{name}' moved out of core — installed its plugin from the catalog "
            f"(memory.provider and your stored memories are unchanged; check its settings with "
            f"`hermes memory status`).")
        return True
    error = str(result.get("error") or "unknown error").rstrip(". ")
    say(f"  ⚠ Memory provider '{name}' moved out of core and could not be installed automatically: "
        f"{error}. Run `{_install_command(name, home)}`.")
    return False


def migrate_home(home: Path, *, install: Callable[[str], dict], say: Callable[[str], None] = print) -> Optional[str]:
    """Install the configured provider's catalog plugin into *home* when the provider is gone.

    Returns the installed plugin name, or None when nothing needed doing or the install could not
    happen (already reported through *say*). Never raises: memory being down must not take the
    update or the agent down with it.
    """
    name = _pending_provider(home, say=say)
    if name is None:
        return None
    return name if _install_pending(home, name, install=install, say=say) else None


def _interactive() -> bool:
    return sys.stdin is not None and sys.stdout is not None and sys.stdin.isatty() and sys.stdout.isatty()


def _unattended_consent() -> bool:
    """Without a terminal (Desktop, gateway, ``hermes update`` from a script) nobody can answer the
    dependency prompt, so every provider that declares Python deps would fail to migrate. The
    configured ``memory.provider`` plus ``security.allow_lazy_installs`` (read for the active home,
    i.e. the one being migrated) is the same consent that let the bundled provider install its deps
    on demand; with a terminal the user is still asked."""
    from pm.install import lazy_installs_allowed

    return not _interactive() and lazy_installs_allowed()


def _home_consent(home: Path) -> bool:
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(home)
    try:
        return _unattended_consent()
    finally:
        reset_hermes_home_override(token)


def _install_into(home: Path) -> Callable[[str], dict]:
    def _install(name: str) -> dict:
        from hermes_cli.plugins_cmd import dashboard_install_plugin
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        token = set_hermes_home_override(home)
        try:
            return dashboard_install_plugin("", force=False, enable=True, catalog_name=name,
                                            assume_deps_consent=name in _LEFT_CORE and _unattended_consent())
        finally:
            reset_hermes_home_override(token)
    return _install


def _home_label(home: Path) -> str:
    from hermes_constants import profile_name_for_home
    profile = profile_name_for_home(home)
    return f"profile '{profile}'" if profile else str(home)


def migrate_all_homes(*, say: Callable[[str], None] = print) -> list[str]:
    """``hermes update`` hook: every profile home sharing this venv. Returns installed plugin names.

    Each line names the profile it is about. Homes missing the same provider share one environment,
    so they share one set of dependency answers (#125794): the first install asks, the rest reuse
    its answers. Homes are grouped by provider AND by their unattended consent (each home's own
    ``security.allow_lazy_installs``), so a refusal only speaks for homes that would be refused the
    same way: when an install in a group fails, the rest of that group are named in one line
    instead of failing (or re-prompting) one by one, and the other groups still migrate. Ctrl-C
    ends the migration with a message, not a traceback into the updater.
    """
    from hermes_cli.plugins_cmd_install import shared_dependency_answers
    from pm.plugins_state import dependency_homes

    def labelled(home: Path) -> Callable[[str], None]:
        return lambda message: say(f"  [{_home_label(home)}] {message.lstrip()}")

    pending: dict[tuple[str, bool], list[Path]] = {}
    for home in dependency_homes():
        try:
            name = _pending_provider(home, say=labelled(home))
            consent = bool(name) and _home_consent(home)
        except Exception as exc:
            logger.debug("memory provider migration skipped for %s: %s", home, exc)
            continue
        if name:
            pending.setdefault((name, consent), []).append(home)

    installed: list[str] = []
    try:
        for (name, _consent), homes in pending.items():
            if len(homes) > 1 and _interactive():
                say(f"  Memory provider '{name}' is configured in {len(homes)} profiles "
                    f"({', '.join(_home_label(h) for h in homes)}); your answers to its dependency "
                    f"questions apply to all of them.")
            with shared_dependency_answers():
                for index, home in enumerate(homes):
                    if _install_pending(home, name, install=_install_into(home), say=labelled(home)):
                        installed.append(name)
                        continue
                    rest = homes[index + 1:]
                    if rest:
                        say(f"  ⚠ Memory provider '{name}' was not installed for "
                            f"{', '.join(_home_label(h) for h in rest)} either. Run "
                            + ", ".join(f"`{_install_command(name, h)}`" for h in rest) + ".")
                    break
    except KeyboardInterrupt:
        say("  ⚠ Memory provider migration cancelled. Profiles already migrated keep their plugin; "
            "run `hermes plugins install <name>` (with `-p <profile>`) for the rest.")
    return installed


def recover_at_startup(name: str, *, say: Optional[Callable[[str], None]] = None) -> bool:
    """Agent-init hook for a configured provider that resolved nowhere. One attempt per process per
    home and name; honours ``security.allow_lazy_installs`` because it installs code. True when installed."""
    from hermes_constants import get_hermes_home, hermes_home_key

    home = get_hermes_home()
    key = (hermes_home_key(home), name)
    if key in _attempted:
        return False
    _attempted.add(key)

    def report(message: str) -> None:
        logger.warning(message)
        if say is not None:
            try:
                say(message)
            except Exception:
                logger.debug("Memory migration notification failed", exc_info=True)

    from pm.install import lazy_installs_allowed
    if not lazy_installs_allowed():
        report(f"⚠ Memory provider '{name}' is not installed, so external memory is off for this session. "
               f"security.allow_lazy_installs is off, so Hermes did not fetch it: "
               f"run `{_install_command(name, home)}`.")
        return False
    return migrate_home(home, install=_install_into(home), say=report) == name
