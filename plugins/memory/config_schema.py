"""Declarative config schema for memory provider plugins: each declares its fields in
a ``config_schema.py`` beside ``__init__.py``; one generic renderer and one
``GET/PUT /api/memory/providers/{name}/config`` pair drive the UI. Schema files are
loaded in a directory-scoped package shell that leaves the plugin's root
``__init__`` unexecuted. They may import this module and relative pure-data helpers;
provider runtime imports belong in ``__init__``. ``web_server`` owns read/write,
dispatching on ``storage``.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import sys
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path

_log = logging.getLogger(__name__)

# Field kinds understood by the generic renderer.
KIND_TEXT = "text"
KIND_SELECT = "select"
KIND_SECRET = "secret"
KIND_BOOL = "bool"
KIND_NUMBER = "number"
KIND_JSON = "json"

# Storage backends understood by web_server (see its read/write dispatch).
STORAGE_FLAT_JSON = "flat_json"
STORAGE_HONCHO_HOST_BLOCK = "honcho_host_block"


@dataclass(frozen=True)
class ProviderFieldOption:
    """A single choice for a ``select`` field."""

    value: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class ProviderField:
    """One configurable field. Stored in exactly one place by ``kind``: non-secret
    kinds under ``key`` in the provider's storage; ``secret`` in the env store under
    ``env_key``, never read back (only ``is_set``). ``aliases``/``env_fallbacks`` read
    legacy values; ``inline`` marks the compact-panel subset (rest: modal, by ``group``)."""

    key: str
    label: str
    kind: str = KIND_TEXT
    default: str = ""
    description: str = ""
    placeholder: str = ""
    options: tuple[ProviderFieldOption, ...] = ()
    env_key: str | None = None
    aliases: tuple[str, ...] = ()
    env_fallbacks: tuple[str, ...] = ()
    inline: bool = False
    group: str = ""
    # Longer help text surfaced as an info tooltip next to the field label.
    info: str = ""
    # Host-block placement: "host" (per-profile) or "root"; flat-json ignores it.
    scope: str = "host"

    @property
    def is_secret(self) -> bool:
        return self.kind == KIND_SECRET

    def allowed_values(self) -> set[str]:
        return {opt.value for opt in self.options}


@dataclass(frozen=True)
class ProviderConfigSchema:
    """A provider plugin's declared config surface."""

    name: str
    label: str
    storage: str = STORAGE_FLAT_JSON
    # Optional link to the provider's config docs, shown in the full-config modal.
    docs_url: str = ""
    fields: tuple[ProviderField, ...] = dataclass_field(default_factory=tuple)

    def inline_fields(self) -> tuple[ProviderField, ...]:
        return tuple(f for f in self.fields if f.inline)


_SCHEMA_CACHE: dict[str, ProviderConfigSchema] = {}


def load_config_schema_file(path: Path) -> ProviderConfigSchema | None:
    """Load schema data with relative imports scoped to this provider's directory."""
    from plugins.plugin_loader import _module_load_lock, register_synthetic_package

    directory = path.resolve().parent
    digest = hashlib.sha256(str(directory).encode()).hexdigest()[:16]
    package = f"_hermes_memory_config_schema_{digest}"
    with _module_load_lock(package):
        # Keep schema helpers separate from the provider's runtime package: its
        # root initializer must still execute when the provider is activated.
        register_synthetic_package(package, [str(directory)])
        schema = None
        try:
            module = importlib.import_module(f"{package}.config_schema")
            schema = getattr(module, "CONFIG_SCHEMA", None)
            return schema
        finally:
            if schema is None:
                # Retry every helper after an incomplete load, including helpers
                # imported successfully before a later dependency failed.
                for name in tuple(sys.modules):
                    if name == package or name.startswith(package + "."):
                        sys.modules.pop(name, None)


def get_provider_config_schema(name: str) -> ProviderConfigSchema | None:
    """``CONFIG_SCHEMA`` of provider ``name``; None (no panel) without a ``config_schema.py``.
    Cached by resolved file path, not name: user plugins are per-profile."""
    from plugins.memory import find_provider_dir

    provider_dir = find_provider_dir(name)
    path = provider_dir / "config_schema.py" if provider_dir else None
    if path is None or not path.is_file():
        return None

    key = str(path)
    if key in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[key]

    try:
        from plugins.memory import _is_bundled
        from hermes_cli.plugin_isolation import user_plugin_host
        host = None if _is_bundled(provider_dir) else user_plugin_host()
        if host is not None:  # plugins.isolation: host — a user schema file is user code too
            schema = _schema_from_record(host.config_schema(path))
        else:
            schema = load_config_schema_file(path)
    except Exception:
        # Never cache a failed load: it would pin an empty panel until restart.
        _log.exception("failed to load config schema for memory provider %r", name)
        return None

    if schema is not None:
        _SCHEMA_CACHE[key] = schema
    return schema


def _schema_from_record(record) -> ProviderConfigSchema | None:
    """Rebuild a schema that crossed the plugin-host boundary as plain data."""
    if not record:
        return None
    fields = tuple(
        ProviderField(**{**f, "options": tuple(ProviderFieldOption(**o) for o in f.get("options") or ()),
                         "aliases": tuple(f.get("aliases") or ()),
                         "env_fallbacks": tuple(f.get("env_fallbacks") or ())})
        for f in record.get("fields") or ())
    return ProviderConfigSchema(**{**record, "fields": fields})
