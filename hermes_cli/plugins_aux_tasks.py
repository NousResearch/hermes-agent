"""Validation and entry shape for ``PluginContext.register_auxiliary_task``."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes_cli.plugins")


def _core_aux_blocks() -> set[str]:
    """Keys of the core ``auxiliary.<task>`` config blocks (``DEFAULT_CONFIG``)."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    aux = DEFAULT_CONFIG.get("auxiliary")
    return {k for k, v in aux.items() if isinstance(v, dict)} if isinstance(aux, dict) else set()


def build_auxiliary_task_entry(
    me: str, owner_id: str, key: str, *, display_name: str, description: str,
    defaults: Optional[dict[str, Any]], inherit_from: Optional[str], registered: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Registry entry for a plugin auxiliary task, or ``ValueError`` for a bad key.

    ``me`` is the manifest name (for messages); ``owner_id`` is the canonical id ``ctx.llm`` is
    bound to, so agent/plugin_llm.py can match it. ``registered`` is the manager's current
    ``_aux_tasks`` (re-registration by the same owner is allowed).
    """
    if not key or not isinstance(key, str):
        raise ValueError(f"Plugin '{me}' tried to register auxiliary task with invalid key {key!r}")
    if not all(c.isalnum() or c == "_" for c in key):
        raise ValueError(f"Plugin '{me}' auxiliary task key {key!r} "
                         f"must contain only alphanumeric characters and underscores")
    from hermes_cli.main_provider_setup import _AUX_TASKS as _BUILTIN_AUX_TASKS
    builtin_aux_keys = {k for k, _name, _desc in _BUILTIN_AUX_TASKS}
    if key in builtin_aux_keys:
        raise ValueError(f"Plugin '{me}' cannot register auxiliary task {key!r} — that key is reserved "
                         f"for a built-in task. Pick a plugin-namespaced key (e.g. '{me}_{key}').")
    existing = registered.get(key)
    if existing is not None and existing.get("plugin") != owner_id:
        raise ValueError(f"Plugin '{me}' cannot register auxiliary task {key!r} — already registered "
                         f"by plugin '{existing.get('plugin')}'")
    # A base is any built-in auxiliary.<block>: the picker's tasks plus core blocks the picker does
    # not list (background_review, goal_judge, ...). A bad base degrades to "no inheritance" rather
    # than failing the whole plugin load; a self-reference would otherwise only surface at read
    # time, as a cycle.
    if inherit_from is not None and (
            not isinstance(inherit_from, str) or inherit_from == key
            or inherit_from not in (builtin_aux_keys | _core_aux_blocks() | registered.keys())):
        logger.warning("Plugin '%s' auxiliary task %r: ignoring inherit_from=%r — not a built-in "
                       "auxiliary task or one already registered by a plugin", me, key, inherit_from)
        inherit_from = None
    # Plugin owns the schema; routing fields are guaranteed present so consumers don't crash.
    # With inheritance the base supplies the shape, so only the plugin's own overrides go here.
    task_defaults = (dict(defaults or {}) if inherit_from else
                     {"provider": "auto", "model": "", "base_url": "", "api_key": "", "timeout": 60,
                      "extra_body": {}, **(defaults or {})})
    entry = {
        "key": key, "display_name": display_name, "description": description,
        "defaults": task_defaults,
        "inherit_from": inherit_from,
        "plugin": owner_id, "plugin_key": owner_id,
    }
    return entry
