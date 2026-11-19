"""Scope-exact catalog entries. Identity is frozen, never read from a proxy."""

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from hermes_constants import hermes_home_key
from hermes_cli.config import load_config_readonly
from hermes_cli.config_backend import ConfigBackendUnavailable
from hermes_cli.plugin_capabilities import plugin_capability_granted
from hermes_cli.plugins_loader import _plugin_home_scope
from hermes_cli.plugins_discovery import gate_manifest
from hermes_cli.plugins_manifest import PluginManifest
from hermes_cli.skill_catalog.identity import Descriptor, describe, source_id

_lock = threading.RLock()
_entries: dict[str, dict[str, "Entry"]] = {}


@dataclass
class Entry:
    source_id: str
    plugin_key: str
    descriptor: Descriptor
    provider: object
    scope: str
    manifest: PluginManifest
    registration_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    generation: str = field(default_factory=lambda: uuid.uuid4().hex)
    ready: bool = False
    stopped: bool = False
    mutations: object = None
    projection_handle: object = None
    initialization: object = None
    manager: object = None

    def authorized(self) -> bool:
        try:
            with _plugin_home_scope(Path(self.scope)):
                cfg = load_config_readonly()
                section = cfg.get("plugins") or {}
                plugin = (section.get("entries") or {}).get(self.plugin_key) or {}
                enabled, disabled = section.get("enabled"), section.get("disabled", [])
                if not isinstance(disabled, list) or (enabled is not None and not isinstance(enabled, list)):
                    return False
                verdict = gate_manifest(self.manifest, set(disabled), set(enabled) if enabled is not None else None)
                return (verdict.action in {"load", "load_now"} and plugin.get("enabled") is not False
                        and plugin_capability_granted(self.plugin_key, "skills.catalog", config=cfg))
        except (OSError, ValueError, TypeError, AttributeError, ConfigBackendUnavailable):
            return False


def _descriptor(provider) -> Descriptor:
    from hermes_cli.plugin_host import is_hosted_object
    if type(type(provider)) is not type:
        raise ValueError("Catalog must not use a custom metaclass.")
    if is_hosted_object(provider):
        descriptor = object.__getattribute__(provider, "__dict__").get("_skill_catalog_descriptor")
        if type(descriptor) is not Descriptor:
            raise ValueError("Hosted catalog descriptor is missing.")
        return descriptor
    return describe(provider)


def registration_key(ctx, provider) -> str:
    from hermes_cli.plugins_manifest import manifest_key

    descriptor = _descriptor(provider)
    return source_id(manifest_key(ctx.manifest), descriptor.name)


def register_catalog(ctx, provider, *, scope=None):
    from hermes_cli.plugins_manifest import manifest_key

    from hermes_cli.skill_catalog.activation import current
    if not current(getattr(ctx, '_manager', None)):
        raise ValueError("Catalog bootstrap generation changed.")
    key = registration_key(ctx, provider)
    descriptor = _descriptor(provider)
    entry = Entry(key, manifest_key(ctx.manifest), descriptor, provider, hermes_home_key(scope), ctx.manifest,
                  manager=getattr(ctx, '_manager', None))
    if "skills.catalog" not in ctx.manifest.capabilities or not entry.authorized():
        raise ValueError("Catalog capability is not declared and granted.")
    with _lock:
        if key in _entries.get(entry.scope, {}):
            raise ValueError("Catalog source is already registered.")
        _entries.setdefault(entry.scope, {})[key] = entry
    return True


def snapshot_registration(name, *, scope=None):
    with _lock:
        entry = _entries.get(hermes_home_key(scope), {}).get(name)
        return entry.provider if entry else None


def prepare_unload(name, current, *, scope=None):
    """Fence dispatch, then drain without forgetting a busy registration or its leases."""
    with _lock:
        entry = _entries.get(hermes_home_key(scope), {}).get(name)
        if entry is None or entry.provider is not current:
            return
        entry.stopped = True
    if entry.mutations is not None:
        entry.mutations.close()
    if entry.projection_handle is not None:
        entry.projection_handle.close()


def restore_registration(name, current, previous, *, scope=None):
    prepare_unload(name, current, scope=scope)
    with _lock:
        rows = _entries.get(hermes_home_key(scope), {})
        entry = rows.get(name)
        if entry is None or entry.provider is not current:
            return False
        if rows.get(name) is entry:
            rows.pop(name)
    return True


def snapshot(*, scope=None) -> tuple[Entry, ...]:
    with _lock:
        return tuple(row for row in _entries.get(hermes_home_key(scope), {}).values()
                     if row.ready and not row.stopped)


def publish(key, *, scope=None):
    from hermes_cli.skill_catalog.projection import Projection
    from hermes_cli.skill_catalog.mutations import Mutations
    from hermes_cli.skill_catalog.installed import installed_records
    from hermes_cli.skill_catalog.validation import CatalogError

    with _lock:
        row = _entries[hermes_home_key(scope)][key]
    from hermes_cli.skill_catalog.activation import allowed
    if not allowed(row.manager) or not row.authorized():
        raise CatalogError('provider_unavailable', 'Catalog bootstrap authority changed.')
    projection = Projection(Path(row.scope))
    row.projection_handle = projection.start(row)
    row.mutations = Mutations(row, projection)
    row.ready = True
    try:
        projection.publish(row, installed_records(row))
    except CatalogError:
        # Keep the live reconciling lease: unknown ownership must not become local.
        return


def admission_error(manifest) -> str:
    from hermes_cli.plugins_manifest import manifest_key

    if "skills.catalog" not in manifest.capabilities:
        return "Catalog capability is not declared."
    try:
        return "" if plugin_capability_granted(manifest_key(manifest), "skills.catalog",
                                               config=load_config_readonly()) else "Catalog consent required."
    except (OSError, ValueError, TypeError, AttributeError, ConfigBackendUnavailable):
        return "Catalog consent unavailable."
