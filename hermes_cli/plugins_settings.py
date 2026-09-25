"""Plugin-declared settings fields for the Desktop/TUI Plugins hub (#46600, #87934).

A ``plugin.yaml`` ``config_schema`` describes the keys under ``plugins.entries.<id>.settings``.
This module turns that schema into renderable form fields (type, current value, choices) and
writes edits back through :func:`hermes_cli.plugins_state.save_plugin_setting` — the same writer
``ctx.set_config`` uses, so the CLI, the plugin and the Desktop never disagree on where a
setting lives. Secrets are declared with ``type: secret`` and never touch ``config.yaml``: the
field carries the ``.env`` name (``env:`` or ``<PLUGIN>_<KEY>``) plus a presence flag, and the
client writes the value through the existing ``PUT /api/env`` credential route.

A ``str`` field becomes a dropdown from static ``choices`` (strings or ``{value, label}`` mappings) or
from ``choices_from: "<module>:<function>"``, a function in the plugin's own package called with a
small context mapping each time fields are built or a save is validated. The function runs only for a
plugin this profile's manager has loaded and enabled, on a worker thread with a
:data:`_CHOICES_TIMEOUT_SECS` deadline; a failure, timeout or malformed return falls back to the
static ``choices`` when declared, else to a free-text field (and a save accepts exactly what the
fallback renders).
"""

from __future__ import annotations

import contextvars
import importlib
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from hermes_cli.plugins_manifest import CHOICES_FROM_RE, normalize_choices
from hermes_cli.plugins_state import _plugin_relative_segments, _plugin_settings_entry, save_plugin_setting

logger = logging.getLogger(__name__)

# manifest ``type`` → wire field type the renderer keys its component table on.
_FIELD_TYPES: Dict[str, str] = {
    "str": "string", "string": "string",
    "int": "number", "integer": "number", "float": "number", "number": "number",
    "bool": "boolean", "boolean": "boolean",
    "list": "json", "array": "json", "dict": "json", "object": "json",
    "secret": "secret",
}
# wire field type → Python types a saved value must have (bool is excluded from number on purpose).
_VALUE_TYPES: Dict[str, tuple] = {
    "string": (str,), "enum": (str,), "number": (int, float), "boolean": (bool,), "json": (list, dict),
}
_ENV_NAME_CLEAN_RE = re.compile(r"[^A-Z0-9]+")
# Deadline for one ``choices_from`` call. Python cannot kill a thread: a call that overruns is abandoned
# as a daemon, and the same field is not called again until it returns (no pile-up per list refresh).
_CHOICES_TIMEOUT_SECS = 2.0
_CHOICES_INFLIGHT: set = set()
_CHOICES_INFLIGHT_LOCK = threading.Lock()


def _manifest_config_schema(plugin_dir: Optional[Path]) -> Mapping[str, Mapping[str, Any]]:
    """``config_schema`` mapping from ``<plugin_dir>/plugin.yaml``; ``{}`` when absent or malformed
    (the loader already warned about malformed entries at load time)."""
    if plugin_dir is None:
        return {}
    manifest = Path(plugin_dir) / "plugin.yaml"
    if not manifest.is_file():
        return {}
    try:
        from utils import fast_safe_load
        data = fast_safe_load(manifest.read_text(encoding="utf-8-sig")) or {}
    except Exception as exc:  # unreadable manifest: no settings surface, never a failed list
        logger.debug("plugin settings: cannot read %s: %s", manifest, exc)
        return {}
    raw = data.get("config_schema") if isinstance(data, Mapping) else None
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, Mapping)}


def secret_env_name(plugin_id: str, key: str, spec: Mapping[str, Any]) -> str:
    """``.env`` variable a ``type: secret`` field is stored under: the manifest's ``env:`` or
    ``<PLUGIN_ID>_<KEY>`` upper-snaked (``image_gen/fal`` + ``api_key`` → ``IMAGE_GEN_FAL_API_KEY``)."""
    declared = str(spec.get("env") or "").strip()
    if declared:
        return declared
    return _ENV_NAME_CLEAN_RE.sub("_", f"{plugin_id}_{key}".upper()).strip("_")


def _base_type(spec: Mapping[str, Any]) -> str:
    if spec.get("secret") is True:
        return "secret"
    return _FIELD_TYPES.get(str(spec.get("type") or "str").lower(), "string")


def _enabled_plugin_package(plugin_id: str) -> Optional[str]:
    """Import name of the package this profile's manager loaded for *plugin_id*, or ``None`` when the
    plugin is not loaded and enabled — plugin code never runs for a disabled or failed plugin."""
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    discover_plugins()  # idempotent; the settings RPC may be the first plugin-state reader in this scope
    loaded = get_plugin_manager()._plugins.get(plugin_id)
    module = getattr(loaded, "module", None)
    if loaded is None or not loaded.enabled or loaded.error or module is None:
        return None
    # A directory plugin is the package itself; an entry point may resolve to a module inside one.
    return module.__name__ if hasattr(module, "__path__") else (module.__package__ or None)


def _call_choices_from(plugin_id: str, key: str, ref: str, context: Dict[str, Any]) -> Optional[List[tuple]]:
    """Run ``choices_from`` under the deadline; normalized choices, or ``None`` on any failure."""
    try:
        package = _enabled_plugin_package(plugin_id)
    except Exception as exc:
        logger.warning("plugin settings: cannot resolve %s for choices_from: %s", plugin_id, exc)
        return None
    if package is None:
        logger.debug("plugin settings: %s is not loaded/enabled; not calling choices_from for %r", plugin_id, key)
        return None
    module_name, _, func_name = ref.partition(":")
    slot = (package, key)
    with _CHOICES_INFLIGHT_LOCK:
        if slot in _CHOICES_INFLIGHT:
            logger.warning("plugin settings: %s choices_from %s is still running; using fallback", plugin_id, ref)
            return None
        _CHOICES_INFLIGHT.add(slot)
    outcome: List[Any] = []
    failure: List[BaseException] = []

    def _worker() -> None:
        try:
            func = getattr(importlib.import_module(f"{package}.{module_name}"), func_name)
            outcome.append(func(context))
        except BaseException as exc:  # a plugin's SystemExit must not end the RPC worker either
            failure.append(exc)
        finally:
            with _CHOICES_INFLIGHT_LOCK:
                _CHOICES_INFLIGHT.discard(slot)

    # The Hermes-home override is a ContextVar: the worker runs in the caller's (profile-scoped) context.
    worker = threading.Thread(target=contextvars.copy_context().run, args=(_worker,),
                              name=f"plugin-choices:{plugin_id}:{key}", daemon=True)
    worker.start()
    worker.join(_CHOICES_TIMEOUT_SECS)
    if worker.is_alive():
        logger.warning("plugin settings: %s choices_from %s timed out after %gs; using fallback",
                       plugin_id, ref, _CHOICES_TIMEOUT_SECS)
        return None
    if failure:
        logger.warning("plugin settings: %s choices_from %s raised %r; using fallback", plugin_id, ref, failure[0])
        return None
    choices = normalize_choices(outcome[0])
    if not choices:
        logger.warning("plugin settings: %s choices_from %s returned %s, not a non-empty list of strings or "
                       "{value, label} mappings; using fallback", plugin_id, ref, type(outcome[0]).__name__)
        return None
    return choices


def _field_choices(plugin_id: str, key: str, spec: Mapping[str, Any],
                   current: Mapping[str, Any]) -> Optional[List[tuple]]:
    """``[(value, label), ...]`` a ``str`` field offers, or ``None`` for a free-text field. The same
    resolution backs rendering and save validation: ``choices_from`` when it succeeds, else static
    ``choices``/``enum`` (malformed static choices count as none)."""
    if _base_type(spec) != "string":
        return None
    ref = spec.get("choices_from")
    if isinstance(ref, str) and CHOICES_FROM_RE.fullmatch(ref):
        from hermes_constants import get_hermes_home
        context = {"plugin_id": plugin_id, "key": key, "settings": dict(current),
                   "hermes_home": str(get_hermes_home())}
        dynamic = _call_choices_from(plugin_id, key, ref, context)
        if dynamic:
            return dynamic
    return normalize_choices(spec.get("choices", spec.get("enum"))) or None


def _current_settings(plugin_id: str) -> Mapping[str, Any]:
    from hermes_cli.config import load_config_readonly
    entry = _plugin_settings_entry(load_config_readonly() or {}, plugin_id) or {}
    raw = entry.get("settings")
    return raw if isinstance(raw, Mapping) else {}


def plugin_settings_fields(plugin_id: str, plugin_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """Renderable settings fields for one plugin: schema + the current value of each key.

    Secret fields never carry the value — only ``env`` (where it lives) and ``has_value``.
    """
    schema = _manifest_config_schema(plugin_dir)
    if not schema:
        return []
    from hermes_cli.config import get_env_value
    current = _current_settings(plugin_id)
    fields: List[Dict[str, Any]] = []
    for key, spec in schema.items():
        try:
            _plugin_relative_segments(key)
        except ValueError:
            continue  # a key the plugin could never read through ctx.get_config
        choices = _field_choices(plugin_id, key, spec, current)
        kind = "enum" if choices else _base_type(spec)
        field: Dict[str, Any] = {
            "key": key, "type": kind,
            "label": str(spec.get("label") or spec.get("title") or key),
            "description": str(spec.get("description") or ""),
            "required": bool(spec.get("required")),
        }
        if kind == "secret":
            env = secret_env_name(plugin_id, key, spec)
            field.update({"env": env, "has_value": get_env_value(env) is not None})
        else:
            if choices:
                # ``choices`` stays the list of values older clients render; labels ride alongside.
                field["choices"] = [value for value, _label in choices]
                if any(value != label for value, label in choices):
                    field["choice_labels"] = [label for _value, label in choices]
            if "default" in spec:
                field["default"] = spec["default"]
            field["value"] = current.get(key, spec.get("default"))
        fields.append(field)
    return fields


def save_plugin_settings(plugin_id: str, plugin_dir: Optional[Path], values: Mapping[str, Any]) -> List[str]:
    """Write ``values`` (``{key: value}``) for the plugin's schema keys; returns the keys written.

    Raises ``ValueError`` on an unknown key, a type mismatch, an enum value outside the field's choices
    (re-resolved exactly as :func:`plugin_settings_fields` renders them) or a
    secret (secrets go to ``.env`` through the credential route, never ``config.yaml``);
    ``PermissionError`` propagates from the shared writer (managed installs / managed keys).
    """
    schema = _manifest_config_schema(plugin_dir)
    current = _current_settings(plugin_id)
    plan: List[tuple] = []
    for key, value in values.items():
        spec = schema.get(str(key))
        if spec is None:
            raise ValueError(f"{key!r} is not declared in the plugin's config_schema")
        choices = _field_choices(plugin_id, str(key), spec, current)
        kind = "enum" if choices else _base_type(spec)
        if kind == "secret":
            raise ValueError(f"{key!r} is a secret; it is stored in .env, not config.yaml")
        expected = _VALUE_TYPES[kind]
        if not isinstance(value, expected) or (isinstance(value, bool) and bool not in expected):
            raise ValueError(f"{key!r} should be {kind} (got {type(value).__name__})")
        if choices and value not in [choice_value for choice_value, _label in choices]:
            raise ValueError(f"{key!r} must be one of the declared choices")
        plan.append((str(key), _plugin_relative_segments(str(key)), value))
    for key, segments, value in plan:
        save_plugin_setting(plugin_id, segments, value)
    return [key for key, _segments, _value in plan]
