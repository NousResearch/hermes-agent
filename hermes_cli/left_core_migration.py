"""Move a home onto the catalog plugin of a feature that left core.

A feature (gateway platform, toolset) that moves from core into a standalone catalog plugin keeps its
names, config keys and env vars, so migrating is only "install the plugin". Each row of
:data:`LEFT_CORE` names the catalog plugin and a read-only predicate "does this home use it". The
contract is the one memory providers established (``memory_provider_migration``, whose helpers this
reuses; memory is not a row because its plugin name comes from ``memory.provider``, not a table):

* ``hermes update`` installs the plugin for every profile home sharing the venv that uses the feature.
* Agent start and gateway start retry once per process for the active home (Desktop users update
  through the app and never run ``hermes update``), honouring ``security.allow_lazy_installs``.

Installs go through the normal catalog install path at the reviewed pin (kill list, dependency
constraints, enable). Unattended dependency consent covers only these rows: the feature shipped in
core, so its users already accepted its dependencies. Every outcome reaches the user (terminal,
Desktop, chat); a gateway-start outcome waits for the home's first agent to deliver it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from hermes_cli.memory_provider_migration import (
    _home_consent, _home_label, _install_command, _interactive, _unattended_consent,
)

logger = logging.getLogger(__name__)


def _read_config(home: Path) -> dict:
    import utils
    try:
        data = utils.fast_safe_load((home / "config.yaml").read_text(encoding="utf-8-sig"))
    except Exception:  # missing/unreadable/invalid YAML: nothing we can see is in use
        return {}
    return data if isinstance(data, dict) else {}


def _platform_block(config: dict, name: str) -> dict:
    """``platforms.<name>`` (either nesting, like gateway/config_loader.py)."""
    gateway = config.get("gateway")
    gateway = gateway if isinstance(gateway, dict) else {}
    for section in (config.get("platforms"), gateway.get("platforms")):
        block = section.get(name) if isinstance(section, dict) else None
        if isinstance(block, dict):
            return block
    return {}


def _toolset_listed(config: dict, names: frozenset[str]) -> bool:
    """A toolset in *names* is selected in ``platform_toolsets`` or the top-level ``toolsets`` list."""
    selections = list((config.get("platform_toolsets") or {}).values()) if isinstance(
        config.get("platform_toolsets"), dict) else []
    selections.append(config.get("toolsets"))
    return any(isinstance(sel, list) and any(str(item) in names for item in sel) for sel in selections)


def homeassistant_in_use(home: Path, *, process_env: bool = False) -> bool:
    """What made core run Home Assistant for *home*: ``HASS_TOKEN`` in its ``.env`` (it enabled both
    the gateway platform and the tools), ``platforms.homeassistant`` enabled or holding a token in
    config.yaml, or the ``homeassistant`` / ``hermes-homeassistant`` toolset selected for a
    platform. *process_env* (the active home at startup only) also counts a ``HASS_TOKEN`` the
    process received from its environment (systemd unit, Docker, shell export)."""
    from agent.secret_scope import load_env_file
    if (load_env_file(home / ".env").get("HASS_TOKEN") or "").strip():
        return True
    if process_env:
        try:
            from agent.secret_scope import get_secret
            if (get_secret("HASS_TOKEN", "") or "").strip():
                return True
        except Exception:
            pass
    config = _read_config(home)
    block = _platform_block(config, "homeassistant")
    if block.get("enabled") is True or (block.get("enabled") is not False and str(block.get("token") or "").strip()):
        return True
    return _toolset_listed(config, frozenset({"homeassistant", "hermes-homeassistant"}))


@dataclass(frozen=True)
class LeftCoreFeature:
    plugin: str                               # catalog entry name == installed plugin name
    label: str                                # user-facing feature name
    in_use: Callable[..., bool]               # (home, *, process_env=False) -> bool, read-only
    unchanged: str                            # what migrated users keep, shown on success
    # Credentials core stripped from every child process while it shipped the feature. Core keeps
    # stripping them (tools/environments/local_env_policy.py): with the plugin absent (migration
    # pending, declined or failed) no manifest declares them, and they would reach every child.
    secret_env: tuple[str, ...] = ()
    # Non-secret settings core kept out of children by default (provider blocklist, Tier 2).
    private_env: tuple[str, ...] = ()


LEFT_CORE: tuple[LeftCoreFeature, ...] = (
    LeftCoreFeature(
        plugin="homeassistant", label="Home Assistant", in_use=homeassistant_in_use,
        unchanged="HASS_TOKEN/HASS_URL, platforms.homeassistant and the ha_* tool names are unchanged",
        secret_env=("HASS_TOKEN",), private_env=("HASS_URL",),
    ),
)

_attempted: set[str] = set()
_undelivered: dict[str, list[str]] = {}


def plugin_present(plugin: str, home: Path) -> bool:
    """Installed in *home* (enabled or not: a user who disabled it chose to). Read-only."""
    return (home / "plugins" / plugin).is_dir()


def _pending(home: Path, *, say: Callable[[str], None], process_env: bool = False) -> list[LeftCoreFeature]:
    """Rows *home* uses whose plugin is not installed and that the catalog ships (a catalog miss is
    reported through *say*). Read-only."""
    from hermes_cli.memory_provider_migration import catalog_source
    out = []
    for feature in LEFT_CORE:
        if plugin_present(feature.plugin, home) or not feature.in_use(home, process_env=process_env):
            continue
        if catalog_source(feature.plugin) is None:
            say(f"  ⚠ {feature.label} moved out of core into the '{feature.plugin}' plugin, which this "
                f"Hermes cannot find in the plugin catalog yet. Run `{_install_command(feature.plugin, home)}` "
                f"once it is listed.")
            continue
        out.append(feature)
    return out


def _install_into(home: Path) -> Callable[[str], dict]:
    def _install(name: str) -> dict:
        from hermes_cli.plugins_cmd import dashboard_install_plugin
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        token = set_hermes_home_override(home)
        try:
            return dashboard_install_plugin("", force=False, enable=True, catalog_name=name,
                                            assume_deps_consent=_unattended_consent())
        finally:
            reset_hermes_home_override(token)
    return _install


def _install_one(home: Path, feature: LeftCoreFeature, *, install: Callable[[str], dict],
                 say: Callable[[str], None]) -> bool:
    try:
        result = install(feature.plugin)
    except Exception as exc:  # network, uv, kill list — report, do not raise
        result = {"ok": False, "error": str(exc)}
    if result.get("ok"):
        say(f"  ✓ {feature.label} moved out of core — installed the '{feature.plugin}' plugin from the "
            f"catalog ({feature.unchanged}).")
        return True
    error = str(result.get("error") or "unknown error").rstrip(". ")
    say(f"  ⚠ {feature.label} moved out of core and its '{feature.plugin}' plugin could not be installed "
        f"automatically: {error}. Run `{_install_command(feature.plugin, home)}`.")
    return False


def migrate_home(home: Path, *, install: Callable[[str], dict], say: Callable[[str], None] = print,
                 process_env: bool = False) -> list[str]:
    """Install every left-core plugin *home* uses and lacks. Returns installed plugin names; never raises."""
    installed = []
    for feature in _pending(home, say=say, process_env=process_env):
        if _install_one(home, feature, install=install, say=say):
            installed.append(feature.plugin)
    return installed


def migrate_all_homes(*, say: Callable[[str], None] = print) -> list[str]:
    """``hermes update`` hook: every profile home sharing this venv. Grouped by plugin and each home's
    unattended consent like the memory migration: homes of one group share dependency answers, and a
    failure names the rest of its group in one line instead of failing them one by one."""
    from hermes_cli.plugins_cmd_install import shared_dependency_answers
    from pm.plugins_state import dependency_homes

    def labelled(home: Path) -> Callable[[str], None]:
        return lambda message: say(f"  [{_home_label(home)}] {message.lstrip()}")

    pending: dict[tuple[str, bool], tuple[LeftCoreFeature, list[Path]]] = {}
    for home in dependency_homes():
        try:
            features = _pending(home, say=labelled(home))
            consent = bool(features) and _home_consent(home)
        except Exception as exc:
            logger.debug("left-core migration skipped for %s: %s", home, exc)
            continue
        for feature in features:
            pending.setdefault((feature.plugin, consent), (feature, []))[1].append(home)

    installed: list[str] = []
    try:
        for feature, homes in pending.values():
            if len(homes) > 1 and _interactive():
                say(f"  {feature.label} is used in {len(homes)} profiles "
                    f"({', '.join(_home_label(h) for h in homes)}); your answers to its dependency "
                    f"questions apply to all of them.")
            with shared_dependency_answers():
                for index, home in enumerate(homes):
                    if _install_one(home, feature, install=_install_into(home), say=labelled(home)):
                        installed.append(feature.plugin)
                        continue
                    rest = homes[index + 1:]
                    if rest:
                        say(f"  ⚠ The '{feature.plugin}' plugin was not installed for "
                            f"{', '.join(_home_label(h) for h in rest)} either. Run "
                            + ", ".join(f"`{_install_command(feature.plugin, h)}`" for h in rest) + ".")
                    break
    except KeyboardInterrupt:
        say("  ⚠ Plugin migration cancelled. Profiles already migrated keep their plugin; run "
            "`hermes plugins install <name>` (with `-p <profile>`) for the rest.")
    return installed


def recover_at_startup(*, say: Optional[Callable[[str], None]] = None) -> list[str]:
    """Agent/gateway start hook for the active home: one attempt per process per home. With *say*
    (an agent's startup-warning sink) outcomes are delivered now, together with any a gateway-start
    attempt queued for this home; without it they are logged and queued for the home's first agent.
    Returns installed plugin names."""
    from hermes_constants import get_hermes_home, hermes_home_key

    home = Path(get_hermes_home())
    key = hermes_home_key(home)

    def deliver(message: str) -> None:
        if say is None:
            _undelivered.setdefault(key, []).append(message)
            return
        try:
            say(message)
        except Exception:
            logger.debug("left-core migration notification failed", exc_info=True)

    if say is not None:
        for message in _undelivered.pop(key, []):
            deliver(message)
    if key in _attempted:
        return []
    _attempted.add(key)

    def report(message: str) -> None:
        message = message.strip()
        logger.warning(message)
        deliver(message)

    try:
        features = _pending(home, say=report, process_env=True)
        if not features:
            return []
        from pm.install import lazy_installs_allowed
        if not lazy_installs_allowed():
            for feature in features:
                report(f"⚠ {feature.label} moved out of core and its '{feature.plugin}' plugin is not "
                       f"installed, so it is off. security.allow_lazy_installs is off, so Hermes did not "
                       f"fetch it: run `{_install_command(feature.plugin, home)}`.")
            return []
        installed = []
        for feature in features:
            if _install_one(home, feature, install=_install_into(home), say=report):
                installed.append(feature.plugin)
        return installed
    except Exception as exc:  # never take agent/gateway start down
        logger.warning("left-core plugin migration failed: %s", exc, exc_info=True)
        return []
