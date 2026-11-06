"""Plugin discovery: directory scanning, entry-point manifests, and the enable/disable gate.

Split out of :mod:`hermes_cli.plugins`. Names that tests patch on the origin
(``get_bundled_plugins_dir``, ``_env_enabled``) are looked up lazily through it.
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib.metadata
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from hermes_constants import get_hermes_home
from hermes_cli.config import cfg_get
from hermes_cli.plugin_capabilities import VALID_CAPABILITY_IDS
from hermes_cli.plugin_capabilities import parse_declared_capabilities as _parse_declared_capabilities
from hermes_cli.plugins_manifest import (
    PluginManifest, _detect_kind_from_source, manifest_key, _resolve_module_source,
    parse_manifest_file, portable_plugin_manifest,
)
from hermes_cli.relay_plugin_cutover import LEGACY_RELAY_PLUGIN_KEYS, RELAY_PLUGINS_CONFIG_ENV

logger = logging.getLogger("hermes_cli.plugins")

ENTRY_POINTS_GROUP = "hermes_agent.plugins"
ENTRY_POINT_CAPABILITIES_GROUP = "hermes_agent.plugin_capabilities"

# Per-harness manifest directories plugin repos ship for OTHER agent harnesses (e.g. obra/superpowers keeps one
# plugin.json per harness). Their plugin.json is not an Agent Plugins v1 manifest and can never validate, so
# parsing it on every discovery pass only spams warnings (#101962).
_FOREIGN_HARNESS_MANIFEST_DIRS = frozenset({
    ".claude-plugin", ".codex-plugin", ".cursor-plugin", ".devin-plugin", ".kimi-plugin",
})

# Set while a caller reads a profile's config WITHOUT wanting that profile's plugins in this process:
# the multiplex preflight loads a PARKED profile's gateway config for the duplicate-credential guard,
# and ``load_gateway_config`` discovers plugins in the scope it runs under. Importing them runs their
# ``register()`` (threads, DB handles) for a profile the operator took out of the host (#123386).
# A contextvar, not an env flag: it must not leak to other threads or outlive the read.
_discovery_suppressed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "hermes_plugin_discovery_suppressed", default=False)


@contextlib.contextmanager
def suppress_plugin_discovery():
    """``discover_plugins()`` is a no-op inside; the scope's manager stays undiscovered, so the
    first real consumer after the block still loads its plugins."""
    token = _discovery_suppressed.set(True)
    try:
        yield
    finally:
        _discovery_suppressed.reset(token)


def plugin_discovery_suppressed() -> bool:
    return _discovery_suppressed.get()


def _select_entry_point_group(entry_points: Any, group: str) -> list:
    """Return one metadata entry-point group across supported Python APIs."""
    if hasattr(entry_points, "select"):
        return list(entry_points.select(group=group))
    if isinstance(entry_points, dict):
        return list(entry_points.get(group, []))
    return [ep for ep in entry_points if ep.group == group]


def discover_entrypoint_manifests() -> list[PluginManifest]:
    """Return metadata-only manifests for installed entry-point plugins. Kind comes from an import-free source
    scan (memory/model providers route to their own discovery). Capabilities come from the companion
    ``hermes_agent.plugin_capabilities`` group (``<plugin-id>.<capability-id>`` entries pointing at the same
    object), so consent works without importing plugin code. Failures are isolated per entry point."""
    manifests: list[PluginManifest] = []
    try:
        eps = importlib.metadata.entry_points()
        group_eps = _select_entry_point_group(eps, ENTRY_POINTS_GROUP)
        capability_eps = _select_entry_point_group(eps, ENTRY_POINT_CAPABILITIES_GROUP)
    except Exception as exc:
        logger.debug("Entry-point scan failed: %s", exc)
        return manifests
    for ep in group_eps:
        try:
            declared = {d.name for d in capability_eps if d.value == ep.value}
            capabilities = [c for c in VALID_CAPABILITY_IDS if f"{ep.name}.{c}" in declared]
            dist = getattr(ep, "dist", None)
            metadata = getattr(dist, "metadata", None)
            manifests.append(PluginManifest(
                name=ep.name, version=str(getattr(dist, "version", "") or ""),
                description=str(metadata.get("Summary", "") or "") if metadata is not None else "",
                source="entrypoint", path=ep.value, key=ep.name,
                kind=_classify_entrypoint_value_kind(ep.value),
                capabilities=_parse_declared_capabilities(capabilities, ep.name),
            ))
        except Exception as exc:
            logger.debug("Entry-point manifest for %r skipped: %s", getattr(ep, "name", "?"), exc)
    return manifests


def _classify_entrypoint_value_kind(value: str) -> str:
    """Classify an entry-point target by import-free source scan (unresolvable -> standalone)."""
    try:
        module_name = str(value).split(":", 1)[0].strip()
        return (_detect_kind_from_source(_resolve_module_source(module_name)) if module_name else None) or "standalone"
    except Exception:
        return "standalone"


def _get_disabled_plugins() -> set:
    """Read ``plugins.disabled`` — a deny-list that wins over ``plugins.enabled``."""
    try:
        from hermes_cli.config import load_config
        disabled = cfg_get(load_config(), "plugins", "disabled", default=[])
        return set(disabled) if isinstance(disabled, list) else set()
    except Exception:
        return set()


def _get_enabled_plugins() -> Optional[set]:
    """Read the ``plugins.enabled`` allow-list (plugins are opt-in). ``None`` = key missing/malformed ("nothing
    enabled yet"; the first ``migrate_config`` run grandfathers installed user plugins); ``set()`` = explicitly
    empty; else the allow-list."""
    try:
        from hermes_cli.config import load_config
        enabled = cfg_get(load_config(), "plugins", "enabled")
        return set(enabled) if isinstance(enabled, list) else None
    except Exception:
        return None


def is_skipped_plugin_dir(name: str) -> bool:
    """True for directory names discovery never treats as plugins. Cache/dunder dirs (__pycache__,
    __MACOSX__, …) can raise PermissionError when walked and take down every later tool call
    (#86996); foreign-harness dirs hold another tool's manifest convention."""
    return (name.startswith("__") and name.endswith("__")) or name in _FOREIGN_HARNESS_MANIFEST_DIRS


def scan_directory(
    path: Path, source: str, *, skip_names: Optional[set[str]] = None, prefix: str = "", depth: int = 0
) -> list[PluginManifest]:
    """Read manifests under *path*: flat ``<root>/<name>/plugin.yaml`` (key ``name``) or category
    ``<root>/<cat>/<name>/plugin.yaml`` (key ``cat/name``; a manifest-less directory recurses one level, depth
    capped at two). *skip_names* ignores top-level names; portable ``plugin.json`` packages are accepted
    alongside YAML manifests."""
    manifests: list[PluginManifest] = []
    if not path.is_dir():
        return manifests
    try:
        children = sorted(path.iterdir())
    except OSError as exc:
        logger.warning("Skipping unreadable plugin directory %s: %s", path, exc)
        return manifests
    for child in children:
        if is_skipped_plugin_dir(child.name):
            logger.debug("Skipping %s (dunder or foreign-harness manifest dir)", child)
            continue
        # pathlib.Path.is_dir() swallows OSError, but injected Path-likes
        # and test doubles can still raise. Fail closed per child.
        try:
            if not child.is_dir() or (depth == 0 and skip_names and child.name in skip_names):
                continue
            manifest_file = next((f for f in (child / "plugin.yaml", child / "plugin.yml") if f.exists()), None)
            portable_file = child / "plugin.json"
            has_portable = portable_file.exists() or portable_file.is_symlink()
        except OSError as exc:
            # stat() raises (not "False") on an unsearchable directory — Windows ACLs (WinError 5) or a
            # mode-000 dir; one such plugin must not abort discovery for every other plugin (#111804).
            logger.warning("Skipping unreadable plugin directory %s: %s", child, exc)
            continue
        if manifest_file is not None:
            manifest = parse_manifest_file(manifest_file, child, source, prefix)
            if manifest is not None:
                manifests.append(manifest)
        elif has_portable:
            try:
                manifests.append(portable_plugin_manifest(child, source, prefix))
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", portable_file, exc)
        elif depth >= 1:
            logger.debug("Skipping %s (no plugin.yaml, depth cap reached)", child)
        else:
            sub_prefix = f"{prefix}/{child.name}" if prefix else child.name
            manifests.extend(scan_directory(child, source, prefix=sub_prefix, depth=depth + 1))
    return manifests


def collect_directory_manifests() -> list[PluginManifest]:
    """Read directory manifests in full-discovery order (bundled top-level, bundled/platforms, user, opt-in
    project) without loading or mutating anything, so startup probes share the exact precedence/containment
    rules of the real discovery sweep."""
    from hermes_cli import plugins as _origin  # patched names resolve through the origin
    manifests: list[PluginManifest] = []

    def _scan(label: str, directory: Path, source: str, skip_names: Optional[set[str]] = None) -> None:
        found = scan_directory(directory, source, skip_names=skip_names)
        logger.debug("  %s: %d manifest(s)", label, len(found))
        manifests.extend(found)

    # Excluded bundled top-level categories have their own discovery. ``platforms/`` is an ordinary category
    # dir: the recursion keys its adapters ``platforms/<dir>`` like every other category (``web/firecrawl``),
    # which is the key `hermes plugins enable/disable` and the dashboard write (#27548); the manifest name
    # (``photon-platform``) stays an accepted alias through ``gate_manifest``.
    repo_plugins = _origin.get_bundled_plugins_dir()
    logger.debug("Scanning bundled plugins: %s", repo_plugins)
    _scan("bundled (top-level)", repo_plugins, "bundled",
          {"memory", "context_engine", "model-providers", "cron_providers", "computer_use", "config_backends"})
    user_dir = get_hermes_home() / "plugins"
    logger.debug("Scanning user plugins: %s", user_dir)
    _scan("user", user_dir, "user")
    if _origin._env_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
        project_dir = Path.cwd() / ".hermes" / "plugins"
        logger.debug("Scanning project plugins: %s", project_dir)
        _scan("project", project_dir, "project")
    else:
        logger.debug("Project plugins disabled (set HERMES_ENABLE_PROJECT_PLUGINS=1 to enable)")
    return manifests


def discovery_inputs_signature(manifests: list[PluginManifest]) -> tuple:
    """Identity of one sweep's inputs: the directory manifests plus the enabled/disabled gate. Any
    difference means a plugin was added, removed, re-shadowed, bumped, enabled or disabled."""
    sources = tuple(sorted(
        (manifest_key(m), m.source, str(m.path or ""), str(m.version or "")) for m in manifests
    ))
    enabled = _get_enabled_plugins()
    return (sources,
            None if enabled is None else tuple(sorted(enabled)),
            tuple(sorted(_get_disabled_plugins())))


class PluginRediscoveryMixin:
    """Discovery-input identity for ``PluginManager``: remember what a sweep saw so a hot re-serve can
    re-scan only when a later operator change would make the sweep find something new (#144336)."""

    # None until a sweep records it: a manager that has not discovered yet reads the current inputs
    # anyway, so it has nothing to revalidate.
    _inputs_signature: Optional[tuple] = None

    def _collect_directory_manifests(self) -> list[PluginManifest]:
        """Directory manifests in full-discovery order (see :func:`collect_directory_manifests`)."""
        return collect_directory_manifests()

    def _collect_scan_manifests(self) -> list[PluginManifest]:
        """Collect this sweep's directory manifests and record them as the inputs it saw."""
        manifests = self._collect_directory_manifests()
        self._inputs_signature = discovery_inputs_signature(manifests)
        return manifests

    def discovery_inputs_changed(self) -> bool:
        """Whether the discovery inputs differ from what the last sweep saw.

        A plugin added, removed, re-shadowed or version-bumped under any scanned directory, or an
        ``plugins.enabled``/``plugins.disabled`` edit, changes the signature. False for a manager that
        has not discovered yet (its pending sweep reads the current inputs anyway) and for one skipped by
        ``HERMES_SAFE_MODE`` (no sweep to revalidate), so a caller can use this as a "re-scan only if it
        would find something new" probe. Entry-point plugins are out of scope: their install/update path
        reloads explicitly.
        """
        if not self._discovered or self._inputs_signature is None:
            return False
        from hermes_cli.plugins_loader import _plugin_home_scope  # the loader imports this module
        with _plugin_home_scope(self.home_path):
            current = discovery_inputs_signature(self._collect_directory_manifests())
        return current != self._inputs_signature


def rediscover_plugins_if_changed() -> bool:
    """Discovery that re-runs only when the discovery inputs changed since the last sweep (see
    :meth:`PluginRediscoveryMixin.discovery_inputs_changed`); returns True when it forced a rescan.

    The profile hot-serve and reconnect paths use this before rebuilding a profile's adapters: a user
    platform-plugin override placed (or removed) after the profile's first sweep has to be applied by
    ``hermes -p <profile> gateway restart``, while a plain config/.env re-scan keeps the live plugin
    registrations instead of tearing every one of them down (#144336).
    """
    from hermes_cli import plugins as _origin  # patched names resolve through the origin

    manager = _origin.get_plugin_manager()
    if not manager.discovery_inputs_changed():
        # Exactly what the serve path called before: on a manager that has swept already this is a
        # no-op, so an unchanged input set keeps every live registration (and re-imports nothing).
        _origin.discover_plugins()
        return False
    _origin.discover_plugins(force=True)
    return True


def resolve_manifest_winners(manifests: list[PluginManifest]) -> dict[str, PluginManifest]:
    """Later sources win on key collision (project > user > bundled): a same-named copy under
    ``~/.hermes/plugins/<name>`` is the documented way to override a bundled plugin, and is logged. A flat
    user/project manifest that claims a bundled key from a *differently named* directory is an impostor, not
    an override (``impostor_dir/plugin.yaml`` with ``name: kanban``): it is skipped with a warning so
    ``hermes plugins enable kanban`` never activates unrelated code under the bundled name."""
    winners: dict[str, PluginManifest] = {}
    for manifest in manifests:
        key = manifest_key(manifest)
        shadowed = winners.get(key)
        if shadowed is not None and shadowed.source == "bundled" and manifest.source in {"user", "project"}:
            own_dir = Path(manifest.path).name if manifest.path else ""
            bundled_dir = Path(shadowed.path).name if shadowed.path else ""
            if own_dir and bundled_dir and own_dir != bundled_dir:
                logger.warning(
                    "Ignoring %s plugin at %s: its manifest name '%s' is a bundled plugin's key but the "
                    "directory is named '%s'; rename the directory to '%s' to override the bundled plugin",
                    manifest.source, manifest.path, key, own_dir, bundled_dir,
                )
                continue
            logger.info("Plugin '%s' at %s (%s) shadows the bundled copy at %s", key, manifest.path,
                        manifest.source, shadowed.path)
        winners[key] = manifest
    return winners


@dataclass(frozen=True)
class ManifestGate:
    """Routing verdict for one winning manifest (see :func:`gate_manifest`)."""

    action: str  # "load" (dependency-ordered pass) | "placeholder" | "load_now" | "defer"
    enabled: bool = False
    error: Optional[str] = None
    log: Optional[tuple] = None  # (level, message, *args)


IN_BOTH_LISTS = "listed in both plugins.enabled and plugins.disabled — disabled wins"


def plugins_in_both_lists() -> list[str]:
    """Keys of the plugins the loader skips because both lists name them, by the loader's own
    config read and name matching (path key, bare leaf, manifest name)."""
    winners = resolve_manifest_winners(collect_directory_manifests())
    disabled, enabled = _get_disabled_plugins(), _get_enabled_plugins()
    return [key for key, manifest in winners.items() if gate_manifest(manifest, disabled, enabled).error == IN_BOTH_LISTS]


def gate_manifest(
    manifest: PluginManifest, disabled: set[str], enabled: Optional[set[str]]
) -> ManifestGate:
    """Decide how one winning manifest is handled. Gate order matters: legacy relay refusal, explicit disable,
    category-owned kinds (exclusive / model-provider), bundled auto-loads (backend now, platform deferred),
    then ``plugins.enabled`` opt-in (path-derived key or legacy bare name)."""
    lookup_key = manifest_key(manifest)
    from hermes_cli.left_core_migration import legacy_plugin_keys
    # A bundled key that left core (``observability/langfuse``) keeps selecting its catalog plugin.
    names = {lookup_key, manifest.name, *legacy_plugin_keys(lookup_key)}

    def _placeholder(error: Optional[str], level: int, message: str, *args, enabled: bool = False) -> ManifestGate:
        return ManifestGate("placeholder", enabled=enabled, error=error, log=(level, message, lookup_key, *args))

    # Relay lifecycle is core-owned; an old plugin copy would compete for its registries.
    if names & LEGACY_RELAY_PLUGIN_KEYS:
        error = (
            "removed — Relay lifecycle is owned by Hermes core; configure a standard user or system Relay "
            f"plugins.toml, or use {RELAY_PLUGINS_CONFIG_ENV} for an explicit user-file override"
        )
        return _placeholder(error, logging.WARNING, "Refusing to load removed Hermes Relay plugin '%s'; %s", error)
    if names & disabled:
        if enabled and names & enabled:
            # Contradictory config (#131980): the disable wins, but a silent DEBUG skip hides the
            # conflict — the plugin "loads nowhere" while plugins.enabled still lists it. Fail loud.
            return _placeholder(
                IN_BOTH_LISTS,
                logging.WARNING,
                "Plugin '%s' is listed in both plugins.enabled and plugins.disabled; the disable "
                "wins and it will not load — remove it from one of the two lists",
            )
        return _placeholder("disabled via config", logging.DEBUG, "Skipping disabled plugin '%s'")
    # Exclusive plugins (memory providers) have their own activation path; record only.
    if manifest.kind == "exclusive":
        return _placeholder(
            "exclusive plugin — activate via <category>.provider config", logging.DEBUG,
            "Skipping '%s' (exclusive, handled by category discovery)",
        )
    # Model providers load via providers/__init__.py; a second import here would create two ProviderProfile
    # instances and break the bundled-vs-user "last writer wins" override.
    if manifest.kind == "model-provider":
        return _placeholder(
            None, logging.DEBUG, "Skipping '%s' (model-provider, handled by providers/ discovery)", enabled=True)
    if manifest.source == "bundled":
        # Bundled backends auto-load; selection among them is ``<category>.provider`` config.
        if manifest.kind == "backend":
            return ManifestGate("load_now")
        # Bundled platforms register LAZILY: eagerly importing ~20 heavy SDKs added seconds to every `hermes`
        # invocation. A deferred loader keeps every platform available on first use.
        if manifest.kind == "platform":
            return ManifestGate("defer")
    if enabled is None or not names & enabled:
        return _placeholder(
            f"not enabled in config (run `hermes plugins enable {lookup_key}` to activate)", logging.DEBUG,
            "Skipping '%s' (not in plugins.enabled)",
        )
    if manifest.source != "bundled":
        # The catalog kill list is enforced at install; a plugin recalled AFTER it was installed must not keep
        # loading. Offline check (in-tree list + cached live copy), honours an explicit install-time bypass.
        from hermes_cli.plugins_cmd_catalog import installed_plugin_removal
        removed = installed_plugin_removal(manifest.name, manifest.path)
        if removed is not None:
            error = f"removed from the Hermes plugin catalog: {removed.reason or 'no reason recorded'}"
            return _placeholder(error, logging.WARNING, "Refusing to load plugin '%s' — %s; run `hermes plugins remove`",
                                error)
    return ManifestGate("load")
