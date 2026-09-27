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
small context mapping when fields are built (cached briefly, see ``_CHOICES_CACHE_TTL_SECS``) and
fresh when a save is validated. The function runs only for a plugin this profile's manager has loaded
and enabled, on a worker thread under a deadline; a failure, timeout or malformed return falls back to
the static ``choices`` when declared, else to a free-text field (and a save accepts exactly what the
fallback renders, plus the unchanged stored value).
"""

from __future__ import annotations

import contextvars
import importlib
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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
# ``choices_from`` resolution. A slot is ``(hermes_home_key, package, key)``, so profiles and homes served
# by one process never share a result or block each other. Python cannot kill a thread: a call that
# overruns is abandoned as a daemon and its slot is not started again until it returns (no pile-up per
# list refresh). A save re-resolves its field fresh under ``_CHOICES_TIMEOUT_SECS``; a listing runs all
# its uncached calls concurrently under ONE ``_CHOICES_LIST_BUDGET_SECS`` deadline. A finished call caches
# its result per slot (failures for the shorter ``_CHOICES_FAILURE_TTL_SECS``), even when the caller had
# stopped waiting; saving a plugin's settings drops its entries.
_CHOICES_TIMEOUT_SECS = 2.0
_CHOICES_LIST_BUDGET_SECS = 2.0
_CHOICES_CACHE_TTL_SECS = 30.0
_CHOICES_FAILURE_TTL_SECS = 5.0
_CHOICES_LOCK = threading.Lock()  # guards the three tables below
_CHOICES_INFLIGHT: set = set()  # slots with a live worker
_CHOICES_CACHE: Dict[tuple, tuple] = {}  # slot -> (monotonic expiry, choices or None for a failure)
_CHOICES_GENERATION: Dict[tuple, int] = {}  # (home_key, package) -> save count; stale workers don't cache


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


# Waits for one field's dynamic choices until a ``time.monotonic()`` deadline; ``None`` means "use the fallback".
_ChoicesWait = Callable[[float], Optional[List[tuple]]]


def _answer(choices: Optional[List[tuple]]) -> _ChoicesWait:
    return lambda _deadline: choices


def _begin_choices_from(plugin_id: str, key: str, ref: str, current: Mapping[str, Any], *,
                        use_cache: bool) -> _ChoicesWait:
    """Start resolving one ``choices_from`` field and return a waiter for its result.

    Answers at once for a cache hit (when *use_cache*), a plugin that is not loaded/enabled, or a slot whose
    previous call is still running. Otherwise the function starts on a worker thread now, so several fields
    resolve concurrently and the caller decides how long to wait for them."""
    try:
        package = _enabled_plugin_package(plugin_id)
    except Exception as exc:
        logger.warning("plugin settings: cannot resolve %s for choices_from: %s", plugin_id, exc)
        return _answer(None)
    if package is None:
        logger.debug("plugin settings: %s is not loaded/enabled; not calling choices_from for %r", plugin_id, key)
        return _answer(None)
    from hermes_constants import get_hermes_home, hermes_home_key
    home = get_hermes_home()
    scope = (hermes_home_key(home), package)
    slot = (*scope, key)
    with _CHOICES_LOCK:
        cached = _CHOICES_CACHE.get(slot) if use_cache else None
        if cached is not None and cached[0] > time.monotonic():
            return _answer(cached[1])
        if slot in _CHOICES_INFLIGHT:
            logger.warning("plugin settings: %s choices_from %s is still running; using fallback", plugin_id, ref)
            return _answer(None)
        _CHOICES_INFLIGHT.add(slot)
        generation = _CHOICES_GENERATION.get(scope, 0)
    context = {"plugin_id": plugin_id, "key": key, "settings": dict(current), "hermes_home": str(home)}
    module_name, _, func_name = ref.partition(":")
    outcome: List[Optional[List[tuple]]] = []

    def _worker() -> None:
        choices: Optional[List[tuple]] = None
        try:
            raw = getattr(importlib.import_module(f"{package}.{module_name}"), func_name)(context)
            choices = normalize_choices(raw) or None
            if choices is None:
                logger.warning("plugin settings: %s choices_from %s returned %s, not a non-empty list of strings "
                               "or {value, label} mappings; using fallback", plugin_id, ref, type(raw).__name__)
        except BaseException as exc:  # a plugin's SystemExit must not end the RPC worker either
            logger.warning("plugin settings: %s choices_from %s raised %r; using fallback", plugin_id, ref, exc)
        finally:
            with _CHOICES_LOCK:
                _CHOICES_INFLIGHT.discard(slot)
                if _CHOICES_GENERATION.get(scope, 0) == generation:  # not superseded by a save meanwhile
                    ttl = _CHOICES_CACHE_TTL_SECS if choices else _CHOICES_FAILURE_TTL_SECS
                    _CHOICES_CACHE[slot] = (time.monotonic() + ttl, choices)
            outcome.append(choices)

    # Same thread shape as the plugin-load deadline (``run_with_load_deadline`` in
    # hermes_cli/plugins_loader.py): the Hermes-home override is a ContextVar, so the worker runs in the
    # caller's (profile-scoped) context.
    worker = threading.Thread(target=contextvars.copy_context().run, args=(_worker,),
                              name=f"plugin-choices:{plugin_id}:{key}", daemon=True)
    worker.start()

    def _wait(deadline: float) -> Optional[List[tuple]]:
        worker.join(max(0.0, deadline - time.monotonic()))
        if worker.is_alive():
            logger.warning("plugin settings: %s choices_from %s did not return in time; using fallback", plugin_id, ref)
            return None
        # A worker that died before its ``finally`` could record a result (it cannot normally) still
        # means "use the fallback", never an IndexError out of the listing RPC.
        return outcome[0] if outcome else None

    return _wait


def _invalidate_choices(plugin_id: str) -> None:
    """Drop this home's cached ``choices_from`` results for *plugin_id* (its saved settings changed)."""
    try:
        package = _enabled_plugin_package(plugin_id)
    except Exception:
        return
    if package is None:
        return
    from hermes_constants import hermes_home_key
    scope = (hermes_home_key(), package)
    with _CHOICES_LOCK:
        _CHOICES_GENERATION[scope] = _CHOICES_GENERATION.get(scope, 0) + 1
        for slot in [slot for slot in _CHOICES_CACHE if slot[:2] == scope]:
            del _CHOICES_CACHE[slot]


def _choices_from_ref(spec: Mapping[str, Any]) -> Optional[str]:
    ref = spec.get("choices_from")
    if _base_type(spec) == "string" and isinstance(ref, str) and CHOICES_FROM_RE.fullmatch(ref):
        return ref
    return None


def _field_choices(spec: Mapping[str, Any], dynamic: Optional[List[tuple]]) -> Optional[List[tuple]]:
    """``[(value, label), ...]`` a ``str`` field offers, or ``None`` for a free-text field. The same rule backs
    rendering and save validation: the resolved ``choices_from`` result when there is one, else static
    ``choices``/``enum`` (malformed static choices count as none)."""
    if _base_type(spec) != "string":
        return None
    return dynamic or normalize_choices(spec.get("choices", spec.get("enum"))) or None


def _current_settings(plugin_id: str) -> Mapping[str, Any]:
    from hermes_cli.config import load_config_readonly
    entry = _plugin_settings_entry(load_config_readonly() or {}, plugin_id) or {}
    raw = entry.get("settings")
    return raw if isinstance(raw, Mapping) else {}


def _renderable_schema(plugin_dir: Optional[Path]) -> Dict[str, Mapping[str, Any]]:
    schema: Dict[str, Mapping[str, Any]] = {}
    for key, spec in _manifest_config_schema(plugin_dir).items():
        try:
            _plugin_relative_segments(key)
        except ValueError:
            continue  # a key the plugin could never read through ctx.get_config
        schema[key] = spec
    return schema


def _build_fields(plugin_id: str, schema: Mapping[str, Mapping[str, Any]], current: Mapping[str, Any],
                  dynamic: Mapping[str, Optional[List[tuple]]]) -> List[Dict[str, Any]]:
    from hermes_cli.config import get_env_value
    fields: List[Dict[str, Any]] = []
    for key, spec in schema.items():
        choices = _field_choices(spec, dynamic.get(key))
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
            value = current.get(key, spec.get("default"))
            if choices and isinstance(value, str) and value and value not in {v for v, _label in choices}:
                # A runtime option that went away (model unloaded, voice deleted): keep showing the stored
                # value instead of an empty dropdown. Saving it unchanged stays allowed; see save_plugin_settings.
                choices = [*choices, (value, f"{value} (unavailable)")]
            if choices:
                # ``choices`` stays the list of values older clients render; labels ride alongside.
                field["choices"] = [value for value, _label in choices]
                if any(value != label for value, label in choices):
                    field["choice_labels"] = [label for _value, label in choices]
            if "default" in spec:
                field["default"] = spec["default"]
            field["value"] = value
        fields.append(field)
    return fields


def plugin_settings_fields_many(plugins: Sequence[Tuple[str, Optional[Path]]]) -> List[List[Dict[str, Any]]]:
    """:func:`plugin_settings_fields` for each ``(plugin_id, plugin_dir)``, in order.

    Every uncached ``choices_from`` in the batch starts at once and they share one
    ``_CHOICES_LIST_BUDGET_SECS`` deadline, so N slow fields cost about one budget, not N deadlines."""
    staged = []
    for plugin_id, plugin_dir in plugins:
        schema = _renderable_schema(plugin_dir)
        current = _current_settings(plugin_id) if schema else {}
        pending = {key: _begin_choices_from(plugin_id, key, ref, current, use_cache=True)
                   for key, spec in schema.items() if (ref := _choices_from_ref(spec))}
        staged.append((plugin_id, schema, current, pending))
    # The budget covers plugin code only: it starts once every worker is running, after discovery.
    deadline = time.monotonic() + _CHOICES_LIST_BUDGET_SECS
    return [_build_fields(plugin_id, schema, current, {key: wait(deadline) for key, wait in pending.items()})
            for plugin_id, schema, current, pending in staged]


def plugin_settings_fields(plugin_id: str, plugin_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """Renderable settings fields for one plugin: schema + the current value of each key.

    Secret fields never carry the value — only ``env`` (where it lives) and ``has_value``.
    """
    return plugin_settings_fields_many([(plugin_id, plugin_dir)])[0]


def save_plugin_settings(plugin_id: str, plugin_dir: Optional[Path], values: Mapping[str, Any]) -> List[str]:
    """Write ``values`` (``{key: value}``) for the plugin's schema keys; returns the keys written.

    Raises ``ValueError`` on an unknown key, a type mismatch, an enum value outside the field's choices
    (``choices_from`` re-resolved fresh, not from the cache; the unchanged stored value stays accepted even
    when no longer offered) or a secret (secrets go to ``.env`` through the credential route, never
    ``config.yaml``); ``PermissionError`` propagates from the shared writer (managed installs / managed keys).
    """
    schema = _manifest_config_schema(plugin_dir)
    current = _current_settings(plugin_id)
    plan: List[tuple] = []
    for key, value in values.items():
        spec = schema.get(str(key))
        if spec is None:
            raise ValueError(f"{key!r} is not declared in the plugin's config_schema")
        ref = _choices_from_ref(spec)
        dynamic = (_begin_choices_from(plugin_id, str(key), ref, current, use_cache=False)(
            time.monotonic() + _CHOICES_TIMEOUT_SECS) if ref else None)
        choices = _field_choices(spec, dynamic)
        kind = "enum" if choices else _base_type(spec)
        if kind == "secret":
            raise ValueError(f"{key!r} is a secret; it is stored in .env, not config.yaml")
        expected = _VALUE_TYPES[kind]
        if not isinstance(value, expected) or (isinstance(value, bool) and bool not in expected):
            raise ValueError(f"{key!r} should be {kind} (got {type(value).__name__})")
        unchanged = str(key) in current and current[str(key)] == value
        if choices and not unchanged and value not in [choice_value for choice_value, _label in choices]:
            raise ValueError(f"{key!r} must be one of the declared choices")
        plan.append((str(key), _plugin_relative_segments(str(key)), value))
    try:
        for key, segments, value in plan:
            save_plugin_setting(plugin_id, segments, value)
    finally:
        if plan:  # even a partial write changed the settings choices_from sees
            _invalidate_choices(plugin_id)
    return [key for key, _segments, _value in plan]
