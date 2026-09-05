"""Plugin manifest model and parsing (``plugin.yaml`` v1/v2, portable ``plugin.json``).

Split out of :mod:`hermes_cli.plugins`; validation warns and never fails a load.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Union

from utils import fast_safe_load
from hermes_cli.plugin_capabilities import parse_declared_capabilities as _parse_declared_capabilities

try:
    import yaml
except ImportError:  # pragma: no cover – yaml is optional at import time
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger("hermes_cli.plugins")

_VALID_PLUGIN_KINDS: Set[str] = {"standalone", "backend", "exclusive", "platform", "model-provider"}

# Unknown plugin.yaml fields are forward-compat surface: warn (debug for v1 files, warning for v2+)
# and continue loading. ``capabilities``/``emits``/``listens``/``hermes``/``depends`` are reserved.
# ── Manifest v2 (#64165) parsing helpers ──────────────────────────────────
_KNOWN_MANIFEST_FIELDS: Set[str] = {
    "name", "version", "description", "author", "requires_env", "provides_tools", "provides_hooks",
    "kind", "hooks", "label", "optional_env", "platforms", "external_dependencies",
    "pip_dependencies", "provides_browser_providers", "provides_web_providers",
    "manifest_version", "api_version", "requires_plugins", "python_dependencies", "config_schema",
    "license", "homepage", "tags", "capabilities", "emits", "listens", "hermes", "depends",
}

# Highest manifest schema version this Hermes understands.
SUPPORTED_MANIFEST_VERSION = 2

_CONFIG_SCHEMA_TYPES: Dict[str, tuple] = {
    "str": (str,), "string": (str,), "int": (int,), "integer": (int,), "float": (int, float),
    "number": (int, float), "bool": (bool,), "boolean": (bool,), "list": (list,), "array": (list,),
    "dict": (dict,), "object": (dict,),
}


def _plugins_debug() -> bool:
    from hermes_cli import plugins as _origin
    return _origin._PLUGINS_DEBUG


def _portable_skill_namespace(key: str) -> str:
    """Return a readable, collision-resistant namespace for a portable plugin."""
    slug = "".join(ch if ch.isascii() and (ch.isalnum() or ch in "_-") else "-" for ch in key.lower())
    slug = slug.strip("-_") or "plugin"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"agent-plugin-{slug}-{digest}"


def _display_author(value: object) -> str:
    """Normalize a manifest author value for the string PluginManifest field."""
    if isinstance(value, Mapping):
        return ", ".join(str(value[f]) for f in ("name", "email", "url") if value.get(f))
    return "" if value is None else str(value)


def _manifest_field_of_type(data: Mapping, key: str, field_name: str, typ, what: str):
    """Return ``data[field_name]`` when absent or of ``typ``; warn and return None otherwise."""
    raw = data.get(field_name)
    if raw is not None and not isinstance(raw, typ):
        logger.warning("Plugin %s: %s must be %s; ignoring", key, field_name, what)
        return None
    return raw


def _manifest_list(data: Mapping, key: str, field_name: str, what: str, coerce: Callable, warn: str) -> list:
    """Coerce each item of list field ``field_name``; ``coerce`` returning None warns ``warn`` and skips."""
    out = []
    for item in _manifest_field_of_type(data, key, field_name, list, what) or []:
        coerced = coerce(item)
        if coerced is None:
            logger.warning(warn, key, item)
        else:
            out.append(coerced)
    return out


def _dependency_entry(item: object) -> Optional[Dict[str, Any]]:
    """``{id, version_range}`` from a requires_plugins item (str shorthand ok); None when malformed."""
    if isinstance(item, str):
        return {"id": item, "version_range": None}
    if isinstance(item, Mapping) and isinstance(item.get("id"), str) and item["id"]:
        vr = item.get("version_range")
        return {"id": item["id"], "version_range": str(vr) if vr is not None else None}
    return None


def _manifest_int(raw: object, key: str, warn: str, fallback: Optional[int]) -> Optional[int]:
    """``int(raw)``; a non-integer warns ``warn`` (formatted with key, raw) and yields ``fallback``."""
    try:
        return int(raw)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        logger.warning(warn, key, raw)
        return fallback


def _parse_manifest_v2_fields(data: Mapping, key: str) -> Dict[str, Any]:
    """Validate/normalize manifest v2 fields into PluginManifest kwargs (warnings, never failures).

    See #64165.
    """
    # manifest_version — absent means v1 (supported forever); api_version is the independent API generation.
    mv = _manifest_int(data.get("manifest_version", 1), key,
                       "Plugin %s: manifest_version %r is not an integer; treating as 1", 1)
    if mv > SUPPORTED_MANIFEST_VERSION:
        logger.warning(
            "Plugin %s: manifest_version %d is newer than this Hermes "
            "supports (%d); loading anyway and ignoring unknown fields", key, mv, SUPPORTED_MANIFEST_VERSION,
        )
    raw_api = data.get("api_version")
    api = None if raw_api is None else _manifest_int(
        raw_api, key, "Plugin %s: api_version %r is not an integer; ignoring", None)
    deps = _manifest_list(
        data, key, "requires_plugins", "a list", _dependency_entry,
        "Plugin %s: requires_plugins entry %r must be a plugin id "
        "string or a {id, version_range} mapping; skipping",
    )
    # python_dependencies — validated and surfaced ONLY; never auto-installed.
    pydeps = _manifest_list(
        data, key, "python_dependencies", "a list of requirement strings",
        lambda item: item.strip() if isinstance(item, str) and item.strip() else None,
        "Plugin %s: python_dependencies entry %r must be a non-empty requirement string; skipping",
    )
    # config_schema — mapping of key -> {type?, default?, description?, required?}.
    schema: Dict[str, Any] = {}
    raw_schema = _manifest_field_of_type(data, key, "config_schema", Mapping, "a mapping")
    for skey, spec in (raw_schema or {}).items():
        if not isinstance(spec, Mapping):
            logger.warning(
                "Plugin %s: config_schema entry %r must be a mapping (e.g. {type: str}); skipping", key, skey)
            continue
        stype = spec.get("type")
        if stype is not None and str(stype).lower() not in _CONFIG_SCHEMA_TYPES:
            logger.warning(
                "Plugin %s: config_schema key %r declares unknown type %r "
                "(known: %s); type check will be skipped for it",
                key, skey, stype, ", ".join(sorted(_CONFIG_SCHEMA_TYPES)),
            )
        schema[str(skey)] = dict(spec)
    tags = [str(t) for t in (_manifest_field_of_type(data, key, "tags", list, "a list") or [])]
    # Forward compat: unknown fields warn (never fail); v1 manifests only at debug.
    unknown = sorted(set(data.keys()) - _KNOWN_MANIFEST_FIELDS)
    if unknown:
        (logger.warning if mv >= 2 else logger.debug)(
            "Plugin %s: unknown manifest field(s) ignored: %s "
            "(newer manifest schema or typo; plugin still loads)", key, ", ".join(unknown),
        )
    return {
        "manifest_version": mv, "api_version": api, "requires_plugins": deps, "python_dependencies": pydeps,
        "config_schema": schema, "license": str(data.get("license") or ""),
        "homepage": str(data.get("homepage") or ""), "tags": tags,
    }


def validate_config_schema(plugin_id: str, schema: Mapping, settings: Mapping) -> List[str]:
    """Return actionable warning strings for settings vs config_schema mismatches (never raises).

    Never raises; schema mismatches must not block plugin load (#64165).
    """
    warnings: List[str] = []
    if not isinstance(schema, Mapping) or not isinstance(settings, Mapping):
        return warnings
    for skey, spec in schema.items():
        if not isinstance(spec, Mapping):
            continue
        if skey not in settings:
            if spec.get("required") and "default" not in spec:
                warnings.append(
                    f"plugins.entries.{plugin_id}.settings.{skey} is required "
                    "by the plugin's config_schema but is not set"
                )
            continue
        stype = spec.get("type")
        expected = _CONFIG_SCHEMA_TYPES.get(str(stype).lower()) if stype else None
        if expected is None:
            continue
        value = settings[skey]
        # bool is an int subclass — don't let True satisfy int/float.
        if not isinstance(value, expected) or (isinstance(value, bool) and bool not in expected):
            warnings.append(
                f"plugins.entries.{plugin_id}.settings.{skey} should be "
                f"{stype} (got {type(value).__name__})"
            )
    return warnings


def resolve_plugin_load_order(manifests: Mapping[str, "PluginManifest"]) -> List[str]:
    """Return plugin keys in dependency order: B before A when A requires B; alphabetical ties. A cycle warns
    and falls back to alphabetical order for all; a missing dependency warns once but never removes the
    dependent plugin (loads never hard-fail on advisory deps).

    See #64165.
    """
    import graphlib
    keys = sorted(manifests.keys())
    by_name: Dict[str, str] = {}
    for k in keys:
        if manifests[k].name:
            by_name.setdefault(manifests[k].name, k)
    edges: Dict[str, Set[str]] = {k: set() for k in keys}
    for k in keys:
        for dep in manifests[k].requires_plugins:
            dep_id = dep.get("id") if isinstance(dep, Mapping) else None
            if not dep_id:
                continue
            resolved = dep_id if dep_id in manifests else by_name.get(dep_id)
            if resolved is None:
                logger.warning(
                    "Plugin %s requires plugin '%s' which is not enabled/"
                    "installed; loading anyway (probe availability at runtime "
                    "via ctx.has_plugin). Run `hermes plugins enable %s` if it is installed.",
                    k, dep_id, dep_id,
                )
            elif resolved == k:
                logger.warning("Plugin %s declares a dependency on itself; ignoring", k)
            else:
                edges[k].add(resolved)
    sorter = graphlib.TopologicalSorter(edges)
    try:
        sorter.prepare()
    except graphlib.CycleError as exc:
        cycle = exc.args[1] if len(exc.args) > 1 else []
        logger.warning(
            "Plugin dependency cycle detected (%s); falling back to "
            "alphabetical load order for all plugins", " -> ".join(str(c) for c in cycle),
        )
        return keys
    ordered: List[str] = []
    while sorter.is_active():
        ready = sorted(sorter.get_ready())
        ordered.extend(ready)
        sorter.done(*ready)
    return ordered


def _detect_kind_from_source(source_text: str) -> Optional[str]:
    """Kind implied by source markers (mirrors plugins/memory ``_is_memory_provider_dir``): memory-provider
    markers -> ``exclusive``; ``register_provider`` + ``ProviderProfile`` -> ``model-provider``; else
    ``None``. Keeps both kinds out of the general manager's eager import."""
    if "register_memory_provider" in source_text or "MemoryProvider" in source_text:
        return "exclusive"
    if "register_provider" in source_text and "ProviderProfile" in source_text:
        return "model-provider"
    return None


def _read_source_from_origin(origin: Optional[str], limit: int = 8192) -> str:
    """First ``limit`` chars of a module's source (``.pyc`` mapped back to ``.py``); "" on failure."""
    try:
        if origin and origin.endswith((".pyc", ".pyo")):
            origin = importlib.util.source_from_cache(origin)
        if not origin or not origin.endswith(".py"):
            return ""
        return Path(origin).read_text(encoding="utf-8", errors="replace")[:limit]
    except Exception:
        return ""


def resolve_module_origin(module_name: str) -> Optional[str]:
    """Return a module's source path WITHOUT importing it, or ``None``. ``find_spec`` on a dotted name imports
    the parent package, so only the top-level name uses it; remaining segments are walked through
    ``submodule_search_locations`` by hand. Namespace/zipped/extension modules return ``None``. Shared with
    ``plugins/memory/__init__.py``."""
    parts = [p for p in module_name.split(".") if p]
    if not parts:
        return None
    try:
        spec = importlib.util.find_spec(parts[0])
        if spec is None or not spec.origin:
            return None
        if len(parts) == 1:
            return spec.origin
        search_paths = spec.submodule_search_locations
        if not search_paths:
            return None
        for i, part in enumerate(parts[1:], start=2):
            found_origin = next_paths = None
            for base in map(Path, search_paths):
                pkg_init, mod_file = base / part / "__init__.py", base / (part + ".py")
                if pkg_init.is_file():
                    found_origin, next_paths = str(pkg_init), [base / part]
                    break
                if mod_file.is_file():
                    found_origin = str(mod_file)
                    break
            if found_origin is None:
                return None
            if i == len(parts) or next_paths is None:
                return found_origin
            search_paths = next_paths
        return None
    except Exception:
        return None


def _resolve_module_source(module_name: str, limit: int = 8192) -> str:
    """First ``limit`` chars of a module's source without importing it ("" when unresolvable)."""
    return _read_source_from_origin(resolve_module_origin(module_name), limit)


def manifest_key(manifest: "PluginManifest") -> str:
    """Registry id of a manifest: the path-derived ``key`` when set, else the bare ``name``."""
    return manifest.key or manifest.name


@dataclass
class PluginManifest:
    """Parsed representation of a plugin.yaml manifest."""

    name: str
    version: str = ""
    description: str = ""
    author: str = ""
    requires_env: List[Union[str, Dict[str, Any]]] = field(default_factory=list)
    provides_tools: List[str] = field(default_factory=list)
    provides_hooks: List[str] = field(default_factory=list)
    source: str = ""        # "bundled", "user", "project", or "entrypoint"
    path: Optional[str] = None
    # ``standalone`` (default; opt-in via plugins.enabled) | ``backend`` (pluggable backend for a core tool;
    # bundled auto-load, user-installed gated) | ``exclusive`` (one active provider, selected via
    # <category>.provider; own discovery, general scanner skips) | ``platform`` (gateway adapter; bundled
    # auto-load, user-installed gated as untrusted code).
    kind: str = "standalone"
    # Path-derived registry key used by plugins.enabled/disabled and `hermes plugins list`: ``disk-cleanup``
    # for a flat plugin, ``image_gen/openai`` for a category plugin. Empty -> name.
    key: str = ""
    portable: bool = False
    skill_namespace: str = ""
    # Declared capability ids, normalized to KNOWN ids. Declaration is consent metadata, NOT a grant: live
    # only via plugins.entries.<id>.granted_capabilities or the legacy allow_* key.
    # See #64228.
    capabilities: List[str] = field(default_factory=list)
    # Manifest v2 fields — all optional and additive. manifest_version versions the FILE FORMAT (v1 supported
    # forever); api_version is the runtime plugin API generation (None = current).
    # Absent (v1) manifests are fully supported forever. See #64165.
    manifest_version: int = 1
    api_version: Optional[int] = None
    # Advisory deps [{"id", "version_range"}]: missing ones warn but load; they order the load.
    requires_plugins: List[Dict[str, Any]] = field(default_factory=list)
    # Declared pip deps — VALIDATED AND SURFACED ONLY, never auto-installed.
    # VALIDATED AND SURFACED ONLY — Hermes never auto-installs these (isolation design for the install seam
    # is a deferred follow-up; see #64165 round-2 review and #15220).
    python_dependencies: List[str] = field(default_factory=list)
    # Schema for plugins.entries.<id>.settings; mismatches warn, never fail.
    config_schema: Dict[str, Any] = field(default_factory=dict)
    license: str = ""
    homepage: str = ""
    tags: List[str] = field(default_factory=list)
    # Event-bus declarations, advisory (discoverability only): ``emits`` bare names published under
    # ``<key>:``; ``listens`` fully-qualified ``<plugin>:<event>`` names.
    emits: List[str] = field(default_factory=list)
    listens: List[str] = field(default_factory=list)


def portable_plugin_manifest(child: Path, source: str, prefix: str) -> PluginManifest:
    """Build the manifest for a portable Agent Plugin directory (``plugin.json``); diagnostics warn."""
    from hermes_cli.agent_plugins import read_agent_plugin_manifest
    data, diagnostics = read_agent_plugin_manifest(child)
    for diagnostic in diagnostics:
        logger.warning("Agent Plugin '%s': %s", child, diagnostic.message)
    key = f"{prefix}/{child.name}" if prefix else data["name"]
    return PluginManifest(
        name=data["name"], version=data.get("version", ""), description=data.get("description", ""),
        author=_display_author(data.get("author", "")), source=source, path=str(child), key=key,
        portable=True, skill_namespace=_portable_skill_namespace(key),
    )


def _manifest_kind(data: Mapping, key: str, plugin_dir: Path) -> str:
    """Normalize ``kind``; undeclared memory/model providers are auto-detected from ``__init__.py`` so they
    route to their own discovery instead of the general manager."""
    raw_kind = data.get("kind", "standalone")
    kind = raw_kind.strip().lower() if isinstance(raw_kind, str) else "standalone"
    if kind not in _VALID_PLUGIN_KINDS:
        logger.warning(
            "Plugin %s: unknown kind '%s' (valid: %s); treating as 'standalone'",
            key, raw_kind, ", ".join(sorted(_VALID_PLUGIN_KINDS)),
        )
        kind = "standalone"
    init_file = plugin_dir / "__init__.py"
    if kind == "standalone" and "kind" not in data and init_file.exists():
        with suppress(Exception):
            source_text = init_file.read_text(errors="replace", encoding="utf-8")[:8192]
            detected = _detect_kind_from_source(source_text)
            if detected:
                kind = detected
                logger.debug("Plugin %s: detected %s, treating as kind='%s'", key, detected, detected)
    return kind


def parse_manifest_file(
    manifest_file: Path, plugin_dir: Path, source: str, prefix: str
) -> Optional[PluginManifest]:
    """Parse one ``plugin.yaml`` into a :class:`PluginManifest`; ``None`` (warned) on failure."""
    try:
        if yaml is None:
            logger.warning("PyYAML not installed – cannot load %s", manifest_file)
            return None
        data = fast_safe_load(manifest_file.read_text(encoding="utf-8")) or {}
        name = data.get("name", plugin_dir.name)
        key = f"{prefix}/{plugin_dir.name}" if prefix else name
        kind = _manifest_kind(data, key, plugin_dir)
        logger.debug(
            "Parsed manifest: key=%s name=%s kind=%s source=%s path=%s", key, name, kind, source, plugin_dir)
        # provides_hooks is canonical; the legacy list-form ``hooks:`` (shipped by some
        # first-party bundled plugins) declares the same hook bindings and is normalized
        # into the same field so declared-vs-registered enforcement (G2-2a) and the
        # install scanner see both forms. A config-style MAPPING under hooks: (event ->
        # descriptor for config.yaml shell hooks) is NOT a declaration list and stays
        # ignored here (forward compat).
        provides_hooks = list(data.get("provides_hooks", []) or [])
        legacy_hooks = data.get("hooks")
        if isinstance(legacy_hooks, list):
            provides_hooks += [h for h in legacy_hooks if isinstance(h, str)]
        return PluginManifest(
            name=name, version=str(data.get("version", "")),
            description=data.get("description", ""), author=_display_author(data.get("author", "")),
            requires_env=data.get("requires_env", []),
            provides_tools=data.get("provides_tools", []),
            provides_hooks=provides_hooks, source=source, path=str(plugin_dir),
            kind=kind, key=key,
            capabilities=_parse_declared_capabilities(data.get("capabilities"), name),
            **_parse_manifest_v2_fields(data, key), emits=data.get("emits") or [],
            listens=data.get("listens") or [],
        )
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", manifest_file, exc, exc_info=_plugins_debug())
        return None
