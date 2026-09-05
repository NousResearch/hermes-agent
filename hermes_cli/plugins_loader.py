"""Plugin loading: directory/entry-point module import, deferred bundled platforms, portable packages,
dependency/config-schema warnings. Mixed into :class:`hermes_cli.plugins.PluginManager`.

Origin-internal names (``PluginContext``, ``LoadedPlugin``, ``_PLUGINS_DEBUG`` …) are imported lazily
through ``hermes_cli.plugins`` so tests that patch them on the origin keep working.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import logging
import re
import sys
import threading
import types
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from registration_lifecycle import replacement_coordinator
from hermes_cli.plugins_discovery import ENTRY_POINTS_GROUP, _select_entry_point_group
from hermes_cli.plugins_manifest import PluginManifest, manifest_key, validate_config_schema
from hermes_cli.plugins_state import _plugin_settings_entry

if TYPE_CHECKING:  # pragma: no cover
    from hermes_cli.plugins import LoadedPlugin

logger = logging.getLogger("hermes_cli.plugins")

_NS_PARENT = "hermes_plugins"
_MODULE_NAMESPACE_LOCK = threading.RLock()
_BARE_MODULE_SCOPE: Dict[str, str] = {}  # bare module name -> owning scope_key


def _plugins_gate_flag(name: str) -> bool:
    """Literal-boolean read of ``plugins.<name>`` (load-time policy flags).

    Startup/discovery-time reads only — these keys have no mid-conversation effect, so a
    plain config read is fine (no deferred-invalidation machinery needed). The gate is
    deliberately literal-boolean: a YAML string (``"false"``/``"no"``/``"true"``) must NOT
    open a fail-closed security gate, hence ``is True`` and never truthiness. Docs show
    literal booleans; a quoted value leaves the gate closed (safe default).
    """
    try:
        from hermes_cli.config import load_config_readonly
        section = (load_config_readonly() or {}).get("plugins")
    except Exception:
        return False
    if not isinstance(section, dict):
        return False
    return section.get(name, False) is True


def _evict_modules(module_name: str) -> None:
    """Drop ``module_name`` and every ``module_name.*`` submodule from ``sys.modules``."""
    prefix = f"{module_name}."
    for name in [n for n in sys.modules if n == module_name or n.startswith(prefix)]:
        del sys.modules[name]


def _serialized_replacement(method):
    """Make snapshot → write → lease attachment one atomic transaction."""
    @wraps(method)
    def wrapped(*args, **kwargs):
        with replacement_coordinator.transaction():
            return method(*args, **kwargs)

    return wrapped


@contextmanager
def _plugin_home_scope(home: Path):
    """Bind discovery and loading to the manager's immutable Hermes home."""
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _dist_installed(req: str) -> Optional[bool]:
    """Best-effort presence probe on a requirement's distribution name; ``None`` when unprobeable."""
    dist = re.split(r"[<>=!~\[;\s]", req, maxsplit=1)[0].strip()
    if not dist:
        return None
    try:
        importlib.metadata.version(dist)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False
    except Exception:
        return None


class PluginLoaderMixin:
    @staticmethod
    def _platform_name_from_manifest(manifest: PluginManifest) -> str:
        """Derive the platform name without importing the adapter: strip a trailing ``-platform`` from the
        manifest name, else the directory basename (the bundled convention)."""
        name = manifest.name or ""
        if name.endswith("-platform"):
            return name[: -len("-platform")]
        return Path(manifest.path).name if manifest.path else name

    @_serialized_replacement
    def _register_deferred_platform(self, manifest: PluginManifest) -> None:
        """Register a lazy loader for a bundled platform: the adapter imports only when the
        ``platform_registry`` is first asked for it; a placeholder ``LoadedPlugin`` keeps it visible in
        ``hermes plugins list`` until then."""
        from hermes_cli.plugins import LoadedPlugin
        lookup_key = manifest_key(manifest)
        platform_name = self._platform_name_from_manifest(manifest)
        loaded = LoadedPlugin(manifest=manifest, enabled=True, deferred=True)
        self._plugins[lookup_key] = loaded
        try:
            from gateway.platform_registry import platform_registry
            scope = self.scope_key

            def _loader(_manifest: PluginManifest = manifest) -> None:
                # Lock before checking cancellation: if an unload won the race it restored the predecessor
                # and this loader must publish nothing; if loading won, unload waits and disposes the set.
                with self._discovery_lock, _plugin_home_scope(self.home_path):
                    if platform_registry.is_deferred_load_cancelled(platform_name, scope=scope):
                        return
                    self._load_plugin_scoped(_manifest)

            previous = platform_registry.snapshot_registration(platform_name, scope=scope)
            platform_registry.register_deferred(platform_name, _loader, scope=scope)
            current = platform_registry.snapshot_registration(platform_name, scope=scope)
            if current[0] is None and current[1] is _loader:
                self._plugin_platform_names.add(platform_name)
                self._track_scoped_registration(
                    manifest, "platform", platform_name, platform_registry, current, previous,
                    finalize=lambda: self._remove_platform_name_if_unowned(platform_name),
                )
            logger.debug("Registered deferred platform loader: %s (plugin=%s)", platform_name, lookup_key)
        except Exception:
            # Fall back to eager loading so the platform is never silently lost.
            logger.debug(
                "Deferred platform registration failed for '%s'; eager-loading", lookup_key, exc_info=True)
            self._load_plugin(manifest)
            return
        self._register_deferred_platform_tools(manifest, loaded)

    def _register_deferred_platform_tools(self, manifest: PluginManifest, loaded: LoadedPlugin) -> None:
        """Register a deferred platform's *client* tools without its adapter. Deferring the plugin would
        otherwise defer its outbound tools too, so CLI/TUI processes (which never materialize platforms)
        would miss them in ``hermes tools`` / ``platform_toolsets``. Opt-in is explicit via ``provides_tools``;
        tools live in a ``tools`` submodule so ``__init__`` stays import-light.

        A platform plugin can ship two independent things: an inbound adapter (heavy — it imports the
        platform SDK) and outbound client tools the agent calls like any other tool. Deferring the plugin
        defers both, so in a CLI/TUI process the client tools never register at all: ``resolve_toolset()``
        returns ``[]``, the toolset is missing from the ``hermes tools`` checklist, and even an explicit
        ``platform_toolsets`` entry is dropped because the key is unknown. The same tools work in
        gateway/web processes only because those materialize every platform at startup (issue #78050).
        Opting in is explicit: the manifest must declare ``provides_tools`` (the field the plugin list and
        web server already read to name a plugin's tools, per #78538). Keying off the mere presence of a
        ``tools.py`` would opt a plugin in by accident — a platform is free to put internal helpers there —
        and would leave the contract invisible to anyone reading the manifest. ``tools.py`` remains where
        the code is imported from; ``provides_tools`` is what asks for it. A platform that does not declare
        the field is untouched and stays fully deferred.
        """
        from hermes_cli.plugins import PluginContext, _PLUGINS_DEBUG
        if not manifest.provides_tools:
            return
        lookup_key = manifest_key(manifest)
        # Never let a client-tool import break discovery — the platform stays deferred and behaves exactly
        # as it did before. But a broken tools.py produces the #78050 symptom itself (declared tools missing
        # from the session), so this has to be visible without turning on debug logging to find it. Where it
        # failed is the first thing an operator needs: nothing registered points at the import or the module
        # body, a partial run points at one tool's definition, and a full run that still raised points past
        # the registrations entirely.
        declared = list(manifest.provides_tools)
        plugin_dir = Path(manifest.path) if manifest.path else None
        if plugin_dir is None or not (plugin_dir / "tools.py").is_file():
            # Declared but undeliverable — staying quiet reproduces the very symptom this fixes.
            logger.warning(
                # Staying quiet here reproduces the exact symptom this path exists to fix — tools the
                # manifest promises, silently absent from the session (#78050) — so say so.
                "Plugin '%s' declares provides_tools %s but has no tools.py; "
                "those tools will not be available in CLI/TUI sessions.", lookup_key, declared,
            )
            return
        before = set(self._plugin_tool_names)  # lets the failure path credit partial registrations

        def _credit() -> List[str]:
            """Attribute every tool registered since ``before`` to this plugin."""
            registered = [t for t in self._plugin_tool_names if t not in before]
            if registered:
                loaded.tools_registered = registered
                self._predeclared_tools[lookup_key] = registered
            return registered

        try:
            module = self._load_directory_module(manifest)
            # Record the module even if nothing registers: the package body has run, so materializing the
            # adapter later must reuse it rather than execute it twice.
            loaded.module = module
            self._predeclared_modules[lookup_key] = module
            tools_module = importlib.import_module(f"{module.__name__}.tools")
            register_tools = getattr(tools_module, "register_tools", None)
            if register_tools is None:
                logger.warning(
                    "Plugin '%s' declares provides_tools %s but its tools.py "
                    "has no register_tools(ctx); those tools will not be "
                    "available in CLI/TUI sessions.", lookup_key, declared,
                )
                return
            register_tools(PluginContext(manifest, self))
            registered = _credit()
            logger.debug(
                "Deferred platform '%s': pre-registered %d client tool(s) %s", lookup_key, len(registered),
                registered,
            )
        except Exception as exc:
            # Tools registered before the raise are live: credit them or `hermes plugins list` under-reports
            # (and _load_plugin's later diff would miss them too). Never break discovery (the platform stays
            # deferred), but a broken tools.py IS the symptom, so warn — and say where it failed first.
            partial, total = _credit(), len(declared)
            complete = len(partial) >= total
            scope = (
                f"before registering any of its {total} declared tool(s)" if not partial
                else f"after registering all {total} declared tool(s)" if complete
                else f"after registering {len(partial)} of {total} declared tool(s)"
            )
            logger.warning(
                "Plugin '%s': client-tool pre-registration failed %s (%s).%s", lookup_key, scope, exc,
                "" if complete else " The remainder will be missing from CLI/TUI sessions.",
                exc_info=_PLUGINS_DEBUG,
            )

    def _warn_python_dependencies(self, manifest: PluginManifest) -> None:
        """Warn about missing declared pip dependencies with an install hint — NEVER auto-install.

        See #64165.
        python_dependencies is a declaration seam ONLY: Hermes validates and prints the requirements with an
        install hint but NEVER auto-installs them. The isolation design (constraints installs vs. vendored
        dirs vs. conflict-detection-and-refusal) is an explicitly deferred follow-up — see the round-2
        review on #64165 and #15220.
        """
        deps = manifest.python_dependencies
        if not deps:
            return
        key = manifest_key(manifest)
        missing = [req for req in deps if _dist_installed(req) is False]
        if missing:
            logger.warning(
                "Plugin %s declares Python dependencies that are not "
                "installed: %s. Hermes does not install plugin dependencies "
                "automatically; install them yourself, e.g.: pip install %s",
                key, ", ".join(missing), " ".join(f"'{m}'" for m in missing),
            )
        else:
            logger.debug("Plugin %s python_dependencies satisfied: %s", key, ", ".join(deps))

    def _validate_plugin_config_schema(self, manifest: PluginManifest) -> None:
        """Warn (never block) on plugins.entries.<id> settings that violate config_schema.

        See #64165.
        """
        if not manifest.config_schema:
            return
        plugin_id = manifest_key(manifest)
        settings: Mapping[str, Any] = {}
        try:
            from hermes_cli.config import load_config
            entry = _plugin_settings_entry(load_config() or {}, plugin_id) or {}
            raw = entry.get("settings")
            if not isinstance(raw, Mapping):
                raw = entry.get("config")  # migration fallback mirroring ctx.get_config
            settings = raw if isinstance(raw, Mapping) else {}
        except Exception:
            settings = {}
        for warning in validate_config_schema(plugin_id, manifest.config_schema, settings):
            logger.warning("Plugin %s config: %s", plugin_id, warning)

    def _load_plugin(self, manifest: PluginManifest) -> None:
        """Import a plugin module and call its ``register(ctx)`` function."""
        with self._discovery_lock, _plugin_home_scope(self.home_path):
            self._load_plugin_scoped(manifest)

    # ── load-time content-drift gate (HookPry G4-3) ────────────────────────────
    # Before importing a manager-managed plugin, compare its CURRENT artifact identity
    # (canonical git tree id for git checkouts / noise-excluded whole-tree sha256 for
    # manual trees — the SAME identity the update transaction records in
    # ``consent.artifact_id``) against that consent baseline. Drift means an out-of-band
    # write — a git pull by any means, a reinstall --force, a same-user write from
    # another profile — reached code that was never reviewed. Fail closed: skip the load
    # with a loud warning. Runs ONLY at discovery/load (once per process) — never in
    # invoke_hook or any per-turn path (per-turn hashing would cost tokens per turn and
    # risk prompt-cache invalidation; see remediation design §4).
    #
    # Scope boundary (upstream-clean): directories with NO install-metadata record and no
    # .git are hand-copied and outside the plugin manager's install/update model — they
    # keep loading unchanged, with a one-time "no consent baseline" log advisory (olympus
    # hand-copies omh, telemetry-hooks, verify_claims_hook; they must keep loading). The
    # gate governs installable plugins — the actual HookPry marketplace channel.
    #
    # Identity residuals (deliberate, documented):
    # * A git checkout's identity is its committed tree (``HEAD^{tree}``): an uncommitted
    #   tracked edit does NOT move it (same semantics as the update owner's consent
    #   match). The byte-level whole-tree hash of a manual tree covers every edit. A
    #   dirty tracked tree is refused at the next update by the transaction's own guard.
    # * Hash-then-import is not atomic: a writer between the hash and the import can slip
    #   content past this process's gate (TOCTOU). Accepted: the window is a same-user
    #   concurrent writer, the gate re-runs every process, and the update transaction
    #   serializes its own commit under the per-plugin lock.
    # * Whole-tree (manual) hashing excludes VCS/cache/env noise (venv, node_modules,
    #   __pycache__, editor temps — mirror of plugin_guard.EXCLUDED_DIRS via
    #   plugin_treehash), so a hand-installed tree with a build venv does not false-drift
    #   on every load.

    def _load_gate_once(self, plugin_key: str) -> set:
        """Per-manager set of plugin keys already given one-time gate advisories."""
        return self.__dict__.setdefault("_load_gate_once_warned", set())

    def _consent_drift_check(self, manifest: PluginManifest) -> Tuple[Optional[str], Optional[str]]:
        """``(block_error, advisory)`` for a manager-managed plugin at load time.

        * block_error — the plugin must NOT be imported: its live artifact identity no
          longer matches ``consent.artifact_id`` (drift) and ``plugins.auto_accept_drift``
          is not a literal ``True``.
        * advisory — the plugin MAY load, but the gate could not verify it (unreadable
          install metadata, or an unhashable tree). Surfaced on the loaded record
          (``hermes plugins list``) so a fail-open check is never invisible.
        * ``(None, None)`` — verified clean, or no consent baseline exists (hand-copied /
          pre-consent installs keep their unchanged load behavior; the one-time "no
          baseline" advisory is logged once per plugin per process).
        """
        if manifest.source not in ("user", "project"):
            return None, None
        plugin_dir = Path(manifest.path) if manifest.path else None
        if plugin_dir is None or not plugin_dir.is_dir():
            return None, None
        plugin_key = manifest_key(manifest)
        try:
            from hermes_cli.plugins_cmd import _read_install_metadata
            record = _read_install_metadata().get(plugin_key) or {}
        except Exception as exc:
            once = self._load_gate_once(plugin_key)
            if plugin_key not in once:
                once.add(plugin_key)
                logger.warning(
                    "Plugin '%s': install metadata unreadable (%s); loading without a "
                    "content-drift check (fail open, loud).", plugin_key, exc)
            return None, (
                "load-time content-drift gate could not read install metadata; "
                "loaded without a drift check (fail open).")
        consent = record.get("consent") if isinstance(record, dict) else None
        baseline = consent.get("artifact_id") if isinstance(consent, dict) else None
        if not isinstance(baseline, str) or not baseline:
            once = self._load_gate_once(plugin_key)
            if plugin_key not in once:
                once.add(plugin_key)
                logger.warning(
                    "Plugin '%s': no content-consent baseline recorded (hand-copied or "
                    "installed before content consent); loading WITHOUT a drift check. "
                    "Record a baseline by reinstalling from git with an explicit --ref "
                    "<SHA> (`hermes plugins install <source> --force --ref <SHA>`).",
                    plugin_key)
            return None, None
        try:
            # Same identity derivation the update transaction's consent match uses.
            if not isinstance(consent, dict):
                return None, None
            from hermes_cli.plugin_update_txn import _consent_artifact_matches
            matches = _consent_artifact_matches(plugin_dir, consent, git_exe=None)
        except Exception as exc:
            once = self._load_gate_once(plugin_key)
            if plugin_key not in once:
                once.add(plugin_key)
                logger.warning(
                    "Plugin '%s': could not verify its tree against the consent baseline "
                    "(%s); loading without a content-drift check (fail open, loud).",
                    plugin_key, exc)
            return None, (
                "load-time content-drift gate could not hash/verify the tree; "
                "loaded without a drift check (fail open).")
        if matches is not False:
            return None, None  # True (verified) or None (baseline unusable) → load clean
        if _plugins_gate_flag("auto_accept_drift"):
            once = self._load_gate_once(plugin_key)
            if plugin_key not in once:
                once.add(plugin_key)
                logger.warning(
                    "Plugin '%s': content drifted from its consented artifact "
                    "(consent %s… → current) and plugins.auto_accept_drift is TRUE — "
                    "loading the changed content without re-consent (DANGEROUS).",
                    plugin_key, baseline[:12])
            return None, None
        return (
            f"content changed since the last consent (consent {baseline[:12]}…); refusing "
            f"to load. Re-consent only through a reviewed `hermes plugins update "
            f"{plugin_key}` (or `plugins install --force --ref <SHA>`), or set "
            f"plugins.auto_accept_drift: true to auto-accept (dangerous)."), None

    def _load_plugin_scoped(self, manifest: PluginManifest) -> None:
        """Load one plugin with the manager's home bound as current."""
        from hermes_cli.plugins import LoadedPlugin, PluginContext, _PLUGINS_DEBUG
        loaded = LoadedPlugin(manifest=manifest)
        plugin_key = manifest_key(manifest)
        logger.debug(
            "Loading plugin '%s' (source=%s, kind=%s, path=%s)",
            plugin_key, manifest.source, manifest.kind, manifest.path,
        )
        if manifest.portable:
            self._load_portable_plugin(manifest, loaded)
            return
        # After the compat-removal date an external plugin that still imports pre-decomposition paths is
        # skipped with a clear reason instead of dying on ImportError mid-register (hermes_cli.plugin_compat).
        from hermes_cli.plugin_compat import disable_reason
        reason = disable_reason(manifest)
        if reason:
            loaded.error = reason
            logger.warning("Plugin '%s' not loaded: %s", manifest.name, reason)
            self._plugins[plugin_key] = loaded
            return
        # Load-time content-drift gate (G4-3): a manager-managed plugin whose current
        # artifact no longer matches its recorded consent.artifact_id is refused before
        # import (fail closed); gate failures that must not block still surface on the
        # record so `hermes plugins list` shows them (never an invisible fail-open).
        drift_error, drift_advisory = self._consent_drift_check(manifest)
        if drift_error:
            loaded.error = drift_error
            logger.warning("Plugin '%s' not loaded: %s", manifest.name, drift_error)
            self._plugins[plugin_key] = loaded
            return
        if drift_advisory:
            loaded.load_advisory = drift_advisory
        registration_start = len(self._registration_order)
        module_name = self._policy_module_name(manifest)
        self._track_tool_override_policy(manifest, module_name)
        try:
            # Reuse a deferred platform's already-imported package so its body doesn't run twice.
            # See #78050.
            module = self._predeclared_modules.pop(plugin_key, None)
            if module is None and manifest.source in {"user", "project", "bundled"}:
                module = self._load_directory_module(manifest, module_name=module_name)
            elif module is None:
                module = self._load_entrypoint_module(manifest)
            loaded.module = module
            register_fn = getattr(module, "register", None)
            if register_fn is None:
                loaded.error = "no register() function"
                logger.warning("Plugin '%s' has no register() function", manifest.name)
            else:
                register_fn(PluginContext(manifest, self))
                self._attribute_registrations(loaded, plugin_key, registration_start)
                loaded.enabled = self._enforce_declared_hooks(loaded, plugin_key)
        except Exception as exc:
            owned = [r for r in self._registration_order if r.plugin_key == plugin_key]
            self._dispose_registrations(owned)
            self._forget_registrations(owned)
            loaded.error = str(exc)
            # register() may have subscribed before raising; a failed plugin must leave no callable reachable
            # from later event dispatch.
            self._remove_plugin_subscriptions(plugin_key)
            logger.warning("Failed to load plugin '%s': %s", manifest.name, exc, exc_info=_PLUGINS_DEBUG)
        # The failure path swept this plugin's whole ledger (not just the registration_start slice), so
        # discovery-time pre-registrations are gone too.
        # There is no live tool left to credit — attribution and the registry agree at zero. Only the
        # success path pops _predeclared_tools, so drop the entry here rather than let the bookkeeping
        # outlive the load attempt (#78050).
        if not loaded.enabled:
            self._predeclared_tools.pop(plugin_key, None)
        self._plugins[plugin_key] = loaded

    def _enforce_declared_hooks(self, loaded: LoadedPlugin, plugin_key: str) -> bool:
        """Warn (default) or refuse (``plugins.strict_hooks: true``) an undeclared hook binding.

        After ``register(ctx)`` returns, compare the hook events actually bound
        (``LoadedPlugin.hooks_registered``) against the manifest's ``provides_hooks``
        (HookPry G2-2a): a plugin that binds an event it never declared did something its
        metadata did not authorize. Default = loud load-log warning + surfaced on
        ``LoadedPlugin.undeclared_hooks`` (``hermes plugins list`` / ``/plugins``); with
        ``plugins.strict_hooks: true`` (literal boolean) the plugin is refused at load
        (fail closed) and its registrations are disposed. Returns False only when the
        load must be refused.
        """
        declared = set(loaded.manifest.provides_hooks or ())
        undeclared = sorted(set(loaded.hooks_registered) - declared)
        if not undeclared:
            return True
        hook_list = ", ".join(undeclared)
        loaded.undeclared_hooks = undeclared
        if _plugins_gate_flag("strict_hooks"):
            owned = [r for r in self._registration_order if r.plugin_key == plugin_key]
            self._dispose_registrations(owned)
            self._forget_registrations(owned)
            # register() may have subscribed before the refusal; nothing may stay reachable
            # from later event dispatch.
            self._remove_plugin_subscriptions(plugin_key)
            loaded.error = (
                f"registered hook(s) not declared in provides_hooks: {hook_list} "
                f"(plugins.strict_hooks: true)")
            logger.warning(
                "Plugin '%s' refused: registered hook(s) not declared in its manifest "
                "provides_hooks: %s", plugin_key, hook_list)
            return False
        logger.warning(
            "Plugin '%s' registered hook(s) not declared in its manifest provides_hooks: "
            "%s — declare them in plugin.yaml to silence this warning (or set "
            "plugins.strict_hooks: true to refuse such plugins at load).",
            plugin_key, hook_list)
        return True

    def _track_tool_override_policy(self, manifest: PluginManifest, module_name: str) -> None:
        """Install the plugin's tool-override policy in tools.registry as a ledger-owned lease."""
        from hermes_cli.plugins import PluginContext
        from tools.registry import registry as _registry
        scope = self.scope_key
        with replacement_coordinator.transaction():
            previous_policy = _registry.snapshot_plugin_override_policy(module_name, scope=scope)
            current_policy = _registry.register_plugin_override_policy(
                module_name, PluginContext(manifest, self)._tool_override_allowed(""), scope=scope,
            )
            policy_lease = replacement_coordinator.acquire(
                ("tool_override_policy", scope, module_name), current=current_policy,
                previous=previous_policy,
                restore=lambda replacement: _registry.restore_plugin_override_policy(
                    module_name, current_policy, replacement, scope=scope,
                ),
            )
            self._track_registration(manifest, "tool_override_policy", module_name, policy_lease.dispose)

    def _attribute_registrations(
        self, loaded: LoadedPlugin, plugin_key: str, registration_start: int
    ) -> None:
        """Fill ``loaded.*_registered`` from the ledger slice this plugin's register() produced."""
        registrations = [
            r for r in self._registration_order[registration_start:]
            if r.plugin_key == plugin_key and r.active
        ]

        def _keys(kind: str) -> List[str]:
            return [r.key for r in registrations if r.kind == kind]

        # Discovery-time tools predate registration_start; credit them back or `hermes plugins list`
        # under-reports once the deferred adapter materializes.
        predeclared = [t for t in self._predeclared_tools.pop(plugin_key, []) if t in self._plugin_tool_names]
        loaded.tools_registered = predeclared + [k for k in _keys("tool") if k not in predeclared]
        loaded.hooks_registered = _keys("hook")
        loaded.middleware_registered = _keys("middleware")
        loaded.commands_registered = _keys("command")
        logger.debug(
            "  registered: %d tool(s), %d hook(s), %d middleware, %d slash command(s), %d CLI command(s)",
            len(loaded.tools_registered), len(loaded.hooks_registered),
            len(loaded.middleware_registered), len(loaded.commands_registered),
            sum(1 for c in self._cli_commands if c in _keys("cli_command")),
        )

    def _load_portable_plugin(self, manifest: PluginManifest, loaded: LoadedPlugin) -> None:
        """Load validated portable components without importing Python code."""
        from hermes_cli.plugins import PluginContext
        lookup_key = manifest_key(manifest)
        try:
            from hermes_cli.agent_plugins import load_agent_plugin
            package = load_agent_plugin(
                Path(manifest.path), get_hermes_home() / "plugin-data" / manifest.skill_namespace)
            ctx = PluginContext(manifest, self)
            for diagnostic in package.diagnostics:
                logger.warning("Agent Plugin '%s' [%s]: %s", lookup_key, diagnostic.scope, diagnostic.message)
            for skill in package.skills:
                try:
                    ctx.register_skill(skill.name, skill.skill_md, skill.description, skill.frontmatter)
                except Exception as exc:
                    logger.warning("Agent Plugin '%s' skill '%s' skipped: %s", lookup_key, skill.name, exc)
            for server_name, config in package.mcp_servers.items():
                internal_name = f"{manifest.skill_namespace}__{server_name}"
                if internal_name in self._portable_mcp_servers:
                    logger.warning("Agent Plugin '%s' MCP server collision: %s", lookup_key, internal_name)
                    continue
                self._portable_mcp_servers[internal_name] = dict(config)
            loaded.enabled = True
        except Exception as exc:
            loaded.error = str(exc)
            logger.warning("Failed to load Agent Plugin '%s': %s", lookup_key, exc)
        self._plugins[lookup_key] = loaded

    def _directory_module_name(self, manifest: PluginManifest) -> str:
        """Profile-safe import namespace for a directory plugin: the bare ``hermes_plugins.<slug>`` for the
        first scope that claims it, a ``__home_<digest>`` suffix for any other scope."""
        slug = manifest_key(manifest).replace("/", "__").replace("-", "_")
        bare_name = f"{_NS_PARENT}.{slug}"
        with _MODULE_NAMESPACE_LOCK:
            if _BARE_MODULE_SCOPE.setdefault(bare_name, self.scope_key) == self.scope_key:
                return bare_name
            digest = hashlib.sha256(self.scope_key.encode("utf-8")).hexdigest()[:12]
            return f"{bare_name}__home_{digest}"

    def _policy_module_name(self, manifest: PluginManifest) -> str:
        """Return the module prefix whose callbacks inherit plugin policy."""
        if manifest.source == "entrypoint" and manifest.path:
            module_name = str(manifest.path).partition(":")[0].strip()
            if module_name:
                return module_name
        return self._directory_module_name(manifest)

    def _load_directory_module(
        self, manifest: PluginManifest, *, module_name: Optional[str] = None,
    ) -> types.ModuleType:
        """Import a directory plugin as ``hermes_plugins.<slug>`` (slug from ``manifest.key`` so
        ``image_gen/openai`` cannot collide with ``tts/openai``)."""
        plugin_dir = Path(manifest.path)  # type: ignore[arg-type]
        init_file = plugin_dir / "__init__.py"
        if not init_file.exists():
            raise FileNotFoundError(f"No __init__.py in {plugin_dir}")
        if _NS_PARENT not in sys.modules:
            ns_pkg = types.ModuleType(_NS_PARENT)
            ns_pkg.__path__ = []  # type: ignore[attr-defined]
            ns_pkg.__package__ = _NS_PARENT
            sys.modules[_NS_PARENT] = ns_pkg
        module_name = module_name or self._directory_module_name(manifest)
        # Evict stale entries for this slug (same slug cached from another Hermes home, or an earlier force
        # reload). Replacing only sys.modules[module_name] is not enough: the plugin's relative imports are
        # cached as "module_name.sub" and resolve from sys.modules first, so a stale submodule would keep
        # serving the previous load's code/state.
        _evict_modules(module_name)
        spec = importlib.util.spec_from_file_location(
            module_name, init_file, submodule_search_locations=[str(plugin_dir)])
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {init_file}")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = module_name
        module.__path__ = [str(plugin_dir)]  # type: ignore[attr-defined]
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            # Don't leave a half-initialized module (or its partially imported relative submodules) cached — a
            # retry or a same-slug plugin in another profile would inherit broken state.
            _evict_modules(module_name)
            raise
        return module

    def _load_entrypoint_module(self, manifest: PluginManifest) -> types.ModuleType:
        """Load a pip-installed plugin via its entry-point reference."""
        for ep in _select_entry_point_group(importlib.metadata.entry_points(), ENTRY_POINTS_GROUP):
            if ep.name == manifest.name:
                return ep.load()
        raise ImportError(f"Entry point '{manifest.name}' not found in group '{ENTRY_POINTS_GROUP}'")
