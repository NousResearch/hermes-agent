"""Plugin-declared settings fields for the Desktop/TUI Plugins hub (#46600, #87934).

A ``plugin.yaml`` ``config_schema`` describes the keys under ``plugins.entries.<id>.settings``.
This module turns that schema into renderable form fields (type, current value, choices) and
writes edits back through :func:`hermes_cli.plugins_state.save_plugin_setting` — the same writer
``ctx.set_config`` uses, so the CLI, the plugin and the Desktop never disagree on where a
setting lives. Secrets are declared with ``type: secret`` and never touch ``config.yaml``: the
field carries the ``.env`` name (``env:`` or ``<PLUGIN>_<KEY>``) plus a presence flag, and the
client writes the value through the existing ``PUT /api/env`` credential route.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

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


def _field_type(spec: Mapping[str, Any]) -> str:
    if spec.get("secret") is True:
        return "secret"
    kind = _FIELD_TYPES.get(str(spec.get("type") or "str").lower(), "string")
    choices = spec.get("choices", spec.get("enum"))
    if kind == "string" and isinstance(choices, list) and choices:
        return "enum"
    return kind


def plugin_settings_fields(plugin_id: str, plugin_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """Renderable settings fields for one plugin: schema + the current value of each key.

    Secret fields never carry the value — only ``env`` (where it lives) and ``has_value``.
    """
    schema = _manifest_config_schema(plugin_dir)
    if not schema:
        return []
    from hermes_cli.config import get_env_value, load_config_readonly
    entry = _plugin_settings_entry(load_config_readonly() or {}, plugin_id) or {}
    raw_current = entry.get("settings")
    current: Mapping[str, Any] = raw_current if isinstance(raw_current, Mapping) else {}
    fields: List[Dict[str, Any]] = []
    for key, spec in schema.items():
        try:
            _plugin_relative_segments(key)
        except ValueError:
            continue  # a key the plugin could never read through ctx.get_config
        kind = _field_type(spec)
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
            choices = spec.get("choices", spec.get("enum"))
            if kind == "enum":
                field["choices"] = [str(c) for c in choices]
            if "default" in spec:
                field["default"] = spec["default"]
            field["value"] = current.get(key, spec.get("default"))
        fields.append(field)
    return fields


def save_plugin_settings(plugin_id: str, plugin_dir: Optional[Path], values: Mapping[str, Any]) -> List[str]:
    """Write ``values`` (``{key: value}``) for the plugin's schema keys; returns the keys written.

    Raises ``ValueError`` on an unknown key, a type mismatch, an enum value outside ``choices`` or a
    secret (secrets go to ``.env`` through the credential route, never ``config.yaml``);
    ``PermissionError`` propagates from the shared writer (managed installs / managed keys).
    """
    schema = _manifest_config_schema(plugin_dir)
    plan: List[tuple] = []
    for key, value in values.items():
        spec = schema.get(str(key))
        if spec is None:
            raise ValueError(f"{key!r} is not declared in the plugin's config_schema")
        kind = _field_type(spec)
        if kind == "secret":
            raise ValueError(f"{key!r} is a secret; it is stored in .env, not config.yaml")
        expected = _VALUE_TYPES[kind]
        if not isinstance(value, expected) or (isinstance(value, bool) and bool not in expected):
            raise ValueError(f"{key!r} should be {kind} (got {type(value).__name__})")
        if kind == "enum" and value not in [str(c) for c in spec.get("choices", spec.get("enum"))]:
            raise ValueError(f"{key!r} must be one of the declared choices")
        plan.append((str(key), _plugin_relative_segments(str(key)), value))
    for key, segments, value in plan:
        save_plugin_setting(plugin_id, segments, value)
    return [key for key, _segments, _value in plan]
