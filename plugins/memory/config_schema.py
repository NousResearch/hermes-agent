"""Declarative configuration schema for memory provider plugins.

Each memory provider plugin *declares* its configurable surface in a
``config_schema.py`` next to its ``__init__.py`` — the fields, their types,
which values are secrets, and (for selects) the allowed options. A single
generic renderer in the desktop UI drives every declared surface. Simple
providers use the shared ``GET/PUT`` storage path; providers with atomic setup
workflows may register actions handled by their plugin without adding bespoke
UI components.

Schema files are loaded by path (like the provider plugins themselves), never
via package import: plugin ``__init__.py`` files pull in the agent runtime,
which must not load into the web server. A ``config_schema.py`` may only
import from this module.

This module is intentionally pure data: it imports nothing from the
config/env layer. ``web_server`` owns the generic read/write logic that
interprets these declarations, dispatching on ``ProviderConfigSchema.storage``
to the matching backend.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field as dataclass_field

_log = logging.getLogger(__name__)

# Field kinds understood by the generic renderer.
KIND_TEXT = "text"
KIND_SELECT = "select"
KIND_SECRET = "secret"
KIND_BOOL = "bool"
KIND_NUMBER = "number"
KIND_JSON = "json"
KIND_SEGMENTED = "segmented"

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
class ProviderFieldCondition:
    """A declarative condition controlling whether a field or action is shown.

    Conditions are ANDed. ``values`` performs an exact match and ``pattern``
    applies a regular expression to the current string value. Most schemas only
    need ``values``; ``pattern`` covers value shapes such as loopback URLs
    without teaching the shared renderer provider-specific rules.
    """

    key: str
    values: tuple[str, ...] = ()
    pattern: str = ""


@dataclass(frozen=True)
class ProviderConfigAction:
    """A provider-owned operation rendered by the shared Desktop form."""

    name: str
    label: str
    description: str = ""
    after_field: str = ""
    payload_fields: tuple[str, ...] = ()
    visible_when: tuple[ProviderFieldCondition, ...] = ()
    refresh_after: bool = False


@dataclass(frozen=True)
class ProviderField:
    """One configurable field on a memory provider.

    For storage-backed providers, a field is stored in exactly one place,
    decided by ``kind``:

    * non-secret kinds — persisted to the provider's config via its storage
      backend under ``key``.
    * ``secret`` — persisted to the env store under ``env_key`` and never read
      back out over the API (only an ``is_set`` flag is surfaced).

    ``aliases`` and ``env_fallbacks`` let a field read legacy values written by
    earlier CLI/env setup without re-introducing per-provider code. ``inline``
    marks the curated subset shown in the compact panel; the rest surface only
    in the full-config modal. ``group`` buckets fields within that modal.
    Provider-managed forms instead submit the visible field values together to
    their registered action.
    """

    key: str
    label: str
    kind: str = KIND_TEXT
    default: str = ""
    description: str = ""
    placeholder: str = ""
    search_placeholder: str = ""
    options: tuple[ProviderFieldOption, ...] = ()
    env_key: str | None = None
    aliases: tuple[str, ...] = ()
    env_fallbacks: tuple[str, ...] = ()
    inline: bool = False
    group: str = ""
    # Longer help text surfaced as an info tooltip next to the field label.
    info: str = ""
    help_url: str = ""
    help_label: str = ""
    required: bool = False
    read_only: bool = False
    visible_when: tuple[ProviderFieldCondition, ...] = ()
    # Provider-managed selects receive their options from get_desktop_config().
    dynamic_options: bool = False
    # Large closed-world option lists render as a searchable picker in Desktop.
    searchable: bool = False
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
    description: str = ""
    # Provider-managed forms keep persistence and validation in the plugin.
    # Simple providers continue using the storage backend above unchanged.
    submit_action: str = ""
    submit_label: str = "Save changes"
    status_action: str = ""
    actions: tuple[ProviderConfigAction, ...] = dataclass_field(default_factory=tuple)
    fields: tuple[ProviderField, ...] = dataclass_field(default_factory=tuple)

    def inline_fields(self) -> tuple[ProviderField, ...]:
        return tuple(f for f in self.fields if f.inline)


_SCHEMA_CACHE: dict[str, ProviderConfigSchema] = {}


def get_provider_config_schema(name: str) -> ProviderConfigSchema | None:
    """Return the ``CONFIG_SCHEMA`` declared by the provider plugin ``name``.

    Providers without a ``config_schema.py`` (e.g. ``builtin``) return ``None``
    and simply render no config panel. The cache keys on the resolved schema
    file, not the name: user-installed plugins are per-profile, so one
    profile's lookup must never answer for another's.
    """

    from plugins.memory import find_provider_dir

    provider_dir = find_provider_dir(name)
    path = provider_dir / "config_schema.py" if provider_dir else None
    if path is None or not path.is_file():
        return None

    key = str(path)
    if key in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[key]

    try:
        spec = importlib.util.spec_from_file_location(
            f"_hermes_memory_config_schema.{name}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        schema = getattr(module, "CONFIG_SCHEMA", None)
    except Exception:
        # Never cache a failed load: it would pin an empty panel until restart.
        _log.exception("failed to load config schema for memory provider %r", name)
        return None

    if schema is not None:
        _SCHEMA_CACHE[key] = schema
    return schema
