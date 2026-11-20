"""HTTP-only cold discovery through the ordinary host-isolated profile manager.

One actual worker per resolved home retains capacity until it exits. Deadlines
bound waiters, never pretend to cancel the loader or start replacements for it.
"""
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent.memory_provider import spawn_context_thread
from hermes_constants import hermes_home_key
from hermes_cli.config import load_config_readonly
from hermes_cli.config_backend import ConfigBackendUnavailable
from hermes_cli.plugin_capabilities import plugin_capability_granted
from hermes_cli.plugin_isolation import isolation_mode
from hermes_cli import plugins
from hermes_cli.plugins_discovery import (
    collect_directory_manifests, discover_entrypoint_manifests, gate_manifest, resolve_manifest_winners,
)
from hermes_cli.plugins_loader import _plugin_home_scope
from hermes_cli.plugins_manifest import manifest_key
from hermes_cli.skill_catalog.service import Service
from hermes_cli.skill_catalog.activation import _origin
from hermes_cli.skill_catalog import diagnostics
from hermes_cli.web_server_profiles import _config_profile_scope, _resolve_profile_dir, _is_current_profile

_lock = threading.Lock()
_active = {}


@dataclass
class _Bootstrap:
    done: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


def _eligible(config):
    section = config.get('plugins') or {}
    enabled, disabled = section.get('enabled'), section.get('disabled', [])
    if not isinstance(enabled, list) or not isinstance(disabled, list):
        return False
    manifests = collect_directory_manifests()
    keys = {manifest_key(manifest) for manifest in manifests}
    manifests.extend(manifest for manifest in discover_entrypoint_manifests() if manifest_key(manifest) not in keys)
    return any('skills.catalog' in manifest.capabilities
               and gate_manifest(manifest, set(disabled), set(enabled)).action in ('load', 'load_now')
               and plugin_capability_granted(manifest_key(manifest), 'skills.catalog', config=config)
               for manifest in resolve_manifest_winners(manifests).values())


def _run(profile, task, home):
    try:
        with _config_profile_scope(profile):
            manager = plugins.get_plugin_manager()
            config = load_config_readonly()
            if isolation_mode(config) != 'host' or manager._discovered or not _eligible(config):
                return
            manager._catalog_diagnostics = True
            token = _origin.set((manager, getattr(manager, '_catalog_epoch', 0)))
            diagnostic = diagnostics.enter(manager.home_path)
            try:
                manager.discover_and_load()
            finally:
                diagnostics.leave(diagnostic)
                _origin.reset(token)
    except (Exception, ConfigBackendUnavailable) as exc:  # health: allow BLE001 -- ordinary loader/hydration failures cannot leak to stock responses
        diagnostics.write(home, f'Catalog bootstrap failed: {type(exc).__name__}')
    finally:
        task.done.set()


def _home(profile):
    from hermes_constants import get_hermes_home
    return get_hermes_home() if _is_current_profile(profile) else _resolve_profile_dir(profile.strip())


def discovery_required(profile):
    """A stock-only scope never competes with provider discovery waiters."""
    from hermes_cli.skill_catalog.projection import Projection
    home = _home(profile)
    if Service(home).entries() or not Projection(home)._absent():
        # A foreign process's lease is authority too, even without local entries.
        return True
    try:
        with _plugin_home_scope(Path(home)):
            config = load_config_readonly()
            return isolation_mode(config) == 'host' and _eligible(config)
    except (Exception, ConfigBackendUnavailable):  # health: allow BLE001 -- match cold bootstrap's stock fallback on loader/config failures
        return False


def lookup(profile, *, deadline=None):
    end = min(deadline, time.monotonic() + 5) if deadline is not None else time.monotonic() + 5
    # Resolve without hydration/global mutation locks. Potentially blocking profile
    # secret hydration belongs to the bounded worker, not the request or ASGI loop.
    home = _home(profile)
    service = Service(home)
    entries = service.entries()
    # A local registry miss is not proof of no provider: another serving process
    # may own the actual OS lease. Never send its identifiers to the stock hub.
    from hermes_cli.skill_catalog.projection import Projection
    from hermes_cli.skill_catalog.validation import CatalogError
    registrations = Projection(home).read(deadline=end)['registrations']
    if registrations.keys() - {entry.registration_id for entry in entries}:
        raise CatalogError('provider_unavailable', 'Catalog provider is served by another process.')
    if entries:
        return service
    key = hermes_home_key(home)
    with _lock:
        task = _active.get(key)
        if task is not None and task.thread is not None and not task.thread.is_alive():
            _active.pop(key)
            task = None
        if task is None:
            task = _Bootstrap()
            _active[key] = task
            # Bind even selector-less requests before the worker hops threads.
            def run():
                with _plugin_home_scope(Path(home)):
                    _run(profile, task, home)
            task.thread = spawn_context_thread(run, name='skill-catalog-bootstrap')
            try:
                task.thread.start()
            except RuntimeError:
                _active.pop(key)
                return None
    task.done.wait(max(0, end - time.monotonic()))
    return service if service.entries() else None
