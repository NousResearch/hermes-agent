"""Plugin-contributed Automation Blueprints (``ctx.register_automation_blueprint``).

A plugin hands over the same fields an in-repo :class:`~cron.blueprint_catalog.AutomationBlueprint`
carries; :func:`build_plugin_blueprint` validates them (bad input raises ``ValueError``, which the
registrar logs and ignores) and namespaces the key as ``<plugin>:<key>``, so a plugin can never
shadow a built-in. :func:`plugin_blueprints` reads the ACTIVE profile's plugin manager at call time;
plugins load per profile, so nothing here is cached across profiles.
"""

from __future__ import annotations

import logging
import re
import string
from dataclasses import fields
from typing import Any, Iterable, List, Mapping

from cron.blueprint_catalog import AutomationBlueprint, BlueprintSlot, fill_blueprint

logger = logging.getLogger(__name__)

__all__ = ["build_plugin_blueprint", "plugin_blueprints"]

_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SLOT_NAME_RE = re.compile(r"^[A-Za-z_]\w{0,63}$")
_SLOT_FIELDS = frozenset(f.name for f in fields(BlueprintSlot))
# Placeholders ``_resolve_schedule`` derives itself: minute/hour from the ``time`` slot, dow from a
# ``recurrence``/``day`` slot (or ``*`` when neither exists).
_DERIVED_SCHEDULE_FIELDS = {"minute": "time", "hour": "time", "dow": None}


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _strings(name: str, value: Any) -> tuple:
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValueError(f"{name} must be a list of strings")
    items = tuple(value)
    if not all(isinstance(v, str) and v.strip() for v in items):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return items


def _slot(raw: Any) -> BlueprintSlot:
    """Accept a ``BlueprintSlot`` (in-process) or its field mapping (the plugin-host wire form)."""
    if isinstance(raw, BlueprintSlot):
        slot = raw
    elif isinstance(raw, Mapping):
        unknown = sorted(set(raw) - _SLOT_FIELDS)
        if unknown:
            raise ValueError(f"unknown slot field(s): {', '.join(unknown)}")
        data = dict(raw)
        if "options" in data:
            data["options"] = tuple(data["options"] or ())
        try:
            slot = BlueprintSlot(**data)
        except TypeError as exc:  # missing name/type/label
            raise ValueError(f"slot {data.get('name')!r}: {exc}") from exc
    else:
        raise ValueError(f"slot must be a mapping, got {type(raw).__name__}")
    if not isinstance(slot.name, str) or not _SLOT_NAME_RE.match(slot.name):
        raise ValueError(f"invalid slot name {slot.name!r}")
    _text(f"slot {slot.name} label", slot.label)
    return slot


def _placeholders(template: str, what: str) -> List[str]:
    names = []
    for _literal, name, _spec, _conv in string.Formatter().parse(template):
        if name is None:
            continue
        if not _SLOT_NAME_RE.match(name):
            raise ValueError(f"{what} placeholder {{{name}}} must be a plain slot name")
        names.append(name)
    return names


def _validate(blueprint: AutomationBlueprint) -> None:
    by_name = {s.name: s for s in blueprint.slots}
    if len(by_name) != len(blueprint.slots):
        raise ValueError("slot names must be unique")
    for name in _placeholders(blueprint.prompt_template, "prompt_template"):
        slot = by_name.get(name)
        if slot is None:
            raise ValueError(f"prompt_template uses {{{name}}} but there is no slot named {name!r}")
        if slot.optional and slot.default in (None, ""):
            raise ValueError(f"prompt_template uses optional slot {name!r} that has no default")
    for name in _placeholders(blueprint.schedule_template, "schedule_template"):
        if name in _DERIVED_SCHEDULE_FIELDS:
            needed = _DERIVED_SCHEDULE_FIELDS[name]
            if needed and needed not in by_name:
                raise ValueError(f"schedule_template uses {{{name}}}, which needs a slot named {needed!r}")
        elif name not in by_name:
            raise ValueError(f"schedule_template uses {{{name}}} but there is no slot named {name!r}")
    # When every required slot has a default, fill it now so a broken template or schedule fails at
    # registration instead of on the user's first "Schedule it".
    if all(s.optional or s.default not in (None, "") for s in blueprint.slots):
        from cron.jobs import parse_schedule

        parse_schedule(fill_blueprint(blueprint, {})["schedule"])


def build_plugin_blueprint(
    plugin: str, key: str, *, title: str, description: str, schedule_template: str,
    prompt_template: str, category: str = "general", slots: Iterable[Any] = (),
    deliver_default: str = "origin", skills: Iterable[str] = (), tags: Iterable[str] = (),
) -> AutomationBlueprint:
    """Validated blueprint keyed ``<plugin>:<key>``; raises ``ValueError`` on malformed input."""
    if not isinstance(plugin, str) or not _NAMESPACE_RE.match(plugin):
        raise ValueError(f"plugin name {plugin!r} cannot namespace a blueprint key")
    if not isinstance(key, str) or not _KEY_RE.match(key):
        raise ValueError(
            f"invalid key {key!r}: use lowercase letters, digits, '-' or '_' (the "
            f"'{plugin}:' namespace is added automatically)"
        )
    if isinstance(slots, (str, Mapping)) or not isinstance(slots, Iterable):
        raise ValueError("slots must be a list")
    blueprint = AutomationBlueprint(
        key=f"{plugin}:{key}",
        title=_text("title", title),
        description=_text("description", description),
        category=_text("category", category),
        schedule_template=_text("schedule_template", schedule_template),
        prompt_template=_text("prompt_template", prompt_template),
        slots=[_slot(s) for s in slots],
        deliver_default=_text("deliver_default", deliver_default),
        skills=_strings("skills", skills),
        tags=_strings("tags", tags),
        plugin=plugin,
    )
    _validate(blueprint)
    return blueprint


def plugin_blueprints() -> List[AutomationBlueprint]:
    """Blueprints the active profile's enabled plugins registered (resolved per call)."""
    try:
        from hermes_cli.plugins import discover_plugins, get_plugin_manager

        discover_plugins()  # idempotent; servers that never import model_tools still see plugins
        return get_plugin_manager().list_automation_blueprints()
    except Exception:
        logger.warning("plugin automation blueprints unavailable", exc_info=True)
        return []
