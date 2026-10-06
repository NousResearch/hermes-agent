"""Move a home onto the catalog plugin of a feature that left core.

A feature (gateway platform, toolset) that moves from core into a standalone catalog plugin keeps its
names, config keys and env vars, so migrating is only "install the plugin". Each row of
:data:`LEFT_CORE` names the catalog plugin and a read-only predicate "does this home use it". The
contract is the one memory providers established (``memory_provider_migration``, whose helpers this
reuses; memory is not a row because its plugin name comes from ``memory.provider``, not a table):

* ``hermes update`` installs the plugin for every profile home sharing the venv that uses the feature.
* Agent start and gateway start retry once per process for the active home (Desktop users update
  through the app and never run ``hermes update``), honouring ``security.allow_lazy_installs``.
* After a failed attempt (catalog miss or unreachable, failed install) starts skip the row for
  :data:`STARTUP_RETRY_SECONDS`, so an offline machine pays the network round trip once, not on every
  ``hermes chat``; ``hermes update`` always retries.
* A home gets the plugin automatically at most once (``_left_core_installed`` in its config.yaml):
  ``hermes plugins remove`` afterwards is the user's choice and sticks.

Installs go through the normal catalog install path at the reviewed pin (kill list, dependency
constraints, enable). Unattended dependency consent covers only these rows: the feature shipped in
core, so its users already accepted its dependencies. Every outcome reaches the user (terminal,
Desktop, chat); a gateway-start outcome waits for the home's first agent to deliver it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from hermes_cli.memory_provider_migration import (
    _home_consent, _home_label, _install_command, _interactive, _unattended_consent,
)

logger = logging.getLogger(__name__)


def _read_config(home: Path) -> dict:
    from hermes_cli.config_backend import read_config_doc
    try:
        data = read_config_doc(home / "config.yaml")
    except Exception:  # missing/unreadable/invalid YAML: nothing we can see is in use
        return {}
    return data if isinstance(data, dict) else {}


def platform_configured_on(home: Path, name: str) -> bool:
    """Platform *name* is on in *home*'s config files as the gateway loader reads them: legacy
    ``gateway.json`` base layer, then config.yaml's ``gateway.platforms`` / ``platforms`` /
    ``gateway.<name>`` blocks merged in the loader's order (managed overlay and ``${VAR}`` applied),
    so the last explicit ``enabled`` wins. A stored token with no ``enabled`` anywhere counts too.
    Env credentials are the caller's to check. Read-only; works before the platform's plugin loads."""
    from gateway import config_loader
    from gateway.config import PlatformConfig
    gw_data = config_loader.load_legacy_gateway_json(home)
    gw_data = gw_data if isinstance(gw_data, dict) else {}
    try:
        yaml_cfg = config_loader.read_yaml_layers(home)
    except Exception:  # malformed config.yaml: the gateway falls back to gateway.json alone too
        yaml_cfg = {}
    yaml_cfg = yaml_cfg if isinstance(yaml_cfg, dict) else {}
    block = config_loader.merge_platform_sections(
        yaml_cfg, yaml_cfg.get("gateway"), gw_data, also=frozenset({name})).get(name)
    if not isinstance(block, dict):
        return False
    platform = PlatformConfig.from_dict(block)
    return platform.enabled or ("enabled" not in block and bool(str(platform.token or "").strip()))


def _toolset_listed(config: dict, names: frozenset[str]) -> bool:
    """A toolset in *names* is selected in ``platform_toolsets`` or the top-level ``toolsets`` list."""
    selections = list((config.get("platform_toolsets") or {}).values()) if isinstance(
        config.get("platform_toolsets"), dict) else []
    selections.append(config.get("toolsets"))
    return any(isinstance(sel, list) and any(str(item) in names for item in sel) for sel in selections)


def homeassistant_in_use(home: Path, *, process_env: bool = False) -> bool:
    """What made core run Home Assistant for *home*: ``HASS_TOKEN`` in its ``.env`` (it enabled both
    the gateway platform and the tools), the platform on in its gateway config
    (:func:`platform_configured_on`), or the ``homeassistant`` / ``hermes-homeassistant`` toolset selected for a
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
    if platform_configured_on(home, "homeassistant"):
        return True
    return _toolset_listed(_read_config(home), frozenset({"homeassistant", "hermes-homeassistant"}))


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
    # Toolsets the plugin registers. Core resolved them like any built-in toolset (on only where the
    # platform's saved list or default composite carried them); a plugin toolset is on for every
    # platform whose known_plugin_toolsets does not name it, so :func:`_record_migration` records
    # them there wherever core had them off.
    toolsets: tuple[str, ...] = ()
    # Platforms whose core default composite (``hermes-<platform>``) never carried those toolsets.
    off_platforms: tuple[str, ...] = ()
    # Gateway platform the plugin ships: a running gateway serves it only after a restart.
    platform: str = ""
    # Channel ownership core declared for that platform, kept while the plugin is absent (migration
    # pending, declined, failed) so a channel-less profile clone still strips the source's identity
    # (hermes_cli/profile_channels.py): the env whose presence enabled the adapter, and the env
    # prefixes it shared with the feature's tools (stripped only when the source runs the adapter).
    enable_env: tuple[str, ...] = ()
    channel_env_prefixes: tuple[str, ...] = ()


LEFT_CORE: tuple[LeftCoreFeature, ...] = (
    LeftCoreFeature(
        plugin="homeassistant", label="Home Assistant", in_use=homeassistant_in_use,
        unchanged="HASS_TOKEN/HASS_URL, platforms.homeassistant and the ha_* tool names are unchanged",
        secret_env=("HASS_TOKEN",), private_env=("HASS_URL",),
        toolsets=("homeassistant",), off_platforms=("acp", "webhook"), platform="homeassistant",
        enable_env=("HASS_TOKEN",), channel_env_prefixes=("HASS_",),
    ),
)

def platform_install_hint(platform: str) -> str:
    """For an unknown platform name that left core: the sentence that names its install command for
    the active home, else ``""``. Appended to "unknown platform" errors (``send_message``)."""
    from hermes_constants import get_hermes_home
    feature = next((f for f in LEFT_CORE if f.platform and f.platform == platform), None)
    if feature is None:
        return ""
    return (f". {feature.label} moved out of core into the '{feature.plugin}' plugin: install it with "
            f"`{_install_command(feature.plugin, Path(get_hermes_home()))}`")


_attempted: set[str] = set()
_undelivered: dict[str, list[str]] = {}


def plugin_present(plugin: str, home: Path) -> bool:
    """Installed in *home* (enabled or not: a user who disabled it chose to). Read-only."""
    return (home / "plugins" / plugin).is_dir()


def _core_carried(feature: LeftCoreFeature, selection: list) -> bool:
    """Whether core resolved *feature*'s toolsets on for a platform selecting *selection*: named
    directly, or through a ``hermes`` / ``hermes-<platform>`` composite that included them."""
    def carries(name: str) -> bool:
        if name in feature.toolsets or name == "hermes":
            return True
        return name.startswith("hermes-") and name[len("hermes-"):].replace("-", "_") not in feature.off_platforms
    return any(carries(str(name)) for name in selection)


_SCOPED_KEY = "_left_core_scoped"  # config.yaml: rows whose toolset scope this home already converted
# config.yaml: rows this home has had installed (by the migration, or by the user before it ran).
# Never written before the plugin dir exists, so a home whose install never succeeded keeps retrying.
_INSTALLED_KEY = "_left_core_installed"


def _marked(config: dict, key: str, feature: LeftCoreFeature) -> bool:
    done = config.get(key)
    return isinstance(done, list) and feature.plugin in map(str, done)


def _scope_kept(config: dict, feature: LeftCoreFeature) -> bool:
    return not feature.toolsets or _marked(config, _SCOPED_KEY, feature)


def _mark(config: dict, key: str, feature: LeftCoreFeature) -> None:
    done = config.get(key)
    config[key] = sorted({*(map(str, done) if isinstance(done, list) else ()), feature.plugin})


def _record_migration(home: Path, feature: LeftCoreFeature) -> None:
    """Keep *feature*'s core-era toolset scope, and mark it installed once its plugin is in *home*,
    in one config write.

    Scope: record the toolsets in ``known_plugin_toolsets[platform]`` (= off) for every platform
    where core had them off: a saved ``platform_toolsets`` list that does not carry them, or no list
    on an ``off_platforms`` platform. Platforms core resolved them on for are left alone. A one-time
    conversion per home and row, independent of installation: a plugin installed before the update
    (it stays inert on a core that still ships the feature) still needs it. Completion is marked
    under ``_left_core_scoped`` so a later run never undoes a choice the user made since
    (``hermes tools``). Raises when config.yaml cannot be read or written."""
    from hermes_cli.config import atomic_config_write, read_user_config_raw
    from hermes_cli.toolset_validation import parse_platform_toolsets_value
    path = home / "config.yaml"
    config = read_user_config_raw(path)
    installed = plugin_present(feature.plugin, home) and not _marked(config, _INSTALLED_KEY, feature)
    if _scope_kept(config, feature) and not installed:
        return
    if installed:
        _mark(config, _INSTALLED_KEY, feature)
    if not _scope_kept(config, feature):
        saved = config.get("platform_toolsets")
        saved = saved if isinstance(saved, dict) else {}
        selections = {str(p): parse_platform_toolsets_value(v) for p, v in saved.items()}
        for platform in feature.off_platforms:
            selections.setdefault(platform, None)
        known = config.get("known_plugin_toolsets")
        known = known if isinstance(known, dict) else {}
        for platform, selection in selections.items():
            if selection is None and platform not in feature.off_platforms:
                continue  # no (valid) saved list: core's default composite carried the toolsets
            if selection is not None and _core_carried(feature, selection):
                continue
            current = known.get(platform) if isinstance(known.get(platform), list) else []
            if missing := [ts for ts in feature.toolsets if ts not in current]:
                known[platform] = sorted({*map(str, current), *missing})
        if known:
            config["known_plugin_toolsets"] = known
        _mark(config, _SCOPED_KEY, feature)
    atomic_config_write(path, config)


STARTUP_RETRY_SECONDS = 3600.0


def _failure_stamp(home: Path, feature: LeftCoreFeature) -> Path:
    """Touched on every failed automatic install; its mtime gates the next startup attempt."""
    return home / "cache" / f"left-core-{feature.plugin}.failed"


def _note_failure(home: Path, feature: LeftCoreFeature) -> None:
    stamp = _failure_stamp(home, feature)
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.touch()
    except OSError as exc:
        logger.debug("left-core retry stamp not written for %s: %s", home, exc)


def _failed_recently(home: Path, feature: LeftCoreFeature) -> bool:
    try:
        return time.time() - _failure_stamp(home, feature).stat().st_mtime < STARTUP_RETRY_SECONDS
    except OSError:
        return False


def _pending(home: Path, *, say: Callable[[str], None], process_env: bool = False,
             backoff: bool = False) -> list[LeftCoreFeature]:
    """Rows *home* uses whose plugin it never had and that the catalog ships (a catalog miss is
    reported through *say*). Converts the toolset scope of every row *home* uses once
    (:func:`_record_migration`), installed or not; a row whose scope cannot be recorded is
    reported and skipped, never installed unscoped. A row marked installed is done for good.
    *backoff* (startup) skips a row whose last attempt failed recently, before any network."""
    from hermes_cli.memory_provider_migration import catalog_source
    out = []
    for feature in LEFT_CORE:
        if _marked(_read_config(home), _INSTALLED_KEY, feature):
            continue
        if not feature.in_use(home, process_env=process_env):
            continue
        present = plugin_present(feature.plugin, home)
        try:
            _record_migration(home, feature)
        except Exception as exc:
            say(f"  ⚠ {feature.label} moved out of core into the '{feature.plugin}' plugin: its per-platform "
                f"toolset selection could not be kept ({exc}). Check `hermes tools`"
                + ("." if present else f" and run `{_install_command(feature.plugin, home)}`."))
            continue
        if present:
            continue
        if backoff and _failed_recently(home, feature):
            logger.info("%s plugin install failed recently for %s; retrying after %ds or on `hermes update`",
                        feature.label, home, STARTUP_RETRY_SECONDS)
            continue
        if catalog_source(feature.plugin) is None:
            _note_failure(home, feature)
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


def _first_cause(error: str) -> str:
    """One line for a notice: git's own ``fatal:`` / ``error:`` line when *error* ends with raw git
    stderr (a failed clone), else its first non-empty line."""
    lines = [line.strip() for line in error.splitlines() if line.strip()]
    cause = next((line.split(":", 1)[1].strip() for line in lines if line.startswith(("fatal:", "error:"))), None)
    return (cause or (lines[0] if lines else "unknown error")).rstrip(". ")


def _install_one(home: Path, feature: LeftCoreFeature, *, install: Callable[[str], dict],
                 say: Callable[[str], None]) -> bool:
    try:
        result = install(feature.plugin)
    except Exception as exc:  # network, uv, kill list — report, do not raise
        result = {"ok": False, "error": str(exc)}
    if result.get("ok"):
        try:
            _record_migration(home, feature)
        except Exception as exc:  # the next run sees the plugin dir and records it then
            logger.debug("left-core install marker not written for %s: %s", home, exc)
        say(f"  ✓ {feature.label} moved out of core — installed the '{feature.plugin}' plugin from the "
            f"catalog ({feature.unchanged}).")
        return True
    _note_failure(home, feature)
    error = _first_cause(str(result.get("error") or ""))
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


def _gateway_serves(home: Path) -> bool:
    """A live gateway serves *home*: it loaded its platforms before this agent-start install."""
    try:
        from gateway.status import resolve_gateway_liveness
        return resolve_gateway_liveness(profile_dir=home, use_cache=False).running
    except Exception:
        return False


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
        features = _pending(home, say=report, process_env=True, backoff=True)
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
                if say is not None and feature.platform and _gateway_serves(home):
                    report(f"Restart the gateway (`hermes gateway restart`) so it serves {feature.label}.")
        return installed
    except Exception as exc:  # never take agent/gateway start down
        logger.warning("left-core plugin migration failed: %s", exc, exc_info=True)
        return []
