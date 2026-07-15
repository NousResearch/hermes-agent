"""Narrow helper that turns a typed ``/<alias>`` slash command into the
canonical ``/model <alias>`` form so the rest of the gateway can dispatch
it through its existing model CommandDef, busy-path, access-control gate,
and ``command:model`` hook without any per-call duplication.

The function is pure with respect to the gateway:

* No handler is called.
* No session state is mutated.
* The caller-supplied ``self`` reference is never touched.
* No attribute is attached to the inbound event.

Priority Order (Fail Closed):
1. resolve_command() -> built-in / canonical command
2. quick commands
3. get_plugin_command_handler(name.replace("_", "-")) -> plugin command
4. resolve_bundle_command_key(name) -> skill bundle
5. resolve_skill_command_key(name) -> active skill
6. unavailable_skill_fn(name) -> disabled / optional skill

If any resolver raises an exception during checking, the helper FAILS CLOSED:
it logs a sanitized debug message and returns the original unchanged event.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Optional

from gateway.platforms.base import MessageEvent
from hermes_cli.commands import resolve_command as _resolve_cmd
from hermes_cli.model_switch import (
    DIRECT_ALIASES,
    MODEL_ALIASES,
    _ensure_direct_aliases,
)

logger = logging.getLogger(__name__)


def _typed_command(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    raw = text.strip()
    if not raw.startswith("/"):
        return None
    head = raw[1:].split(None, 1)
    if not head:
        return None
    return head[0].strip().lower() or None


def is_name_occupied(
    name: str,
    *,
    config: object = None,
    unavailable_skill_fn: Optional[Callable[[str], Optional[str]]] = None,
) -> bool:
    """Return True if ``name`` is already claimed by a higher-priority
    built-in, quick command, plugin, skill bundle, active skill, or
    unavailable skill.

    Raises Exception if any underlying lookup fails unexpectedly, allowing
    the caller to fail-closed.
    """
    if not name:
        return True

    # 1. Built-in command
    if _resolve_cmd(name) is not None:
        return True

    # 2. Quick command
    qc: object = {}
    if isinstance(config, dict):
        qc = config.get("quick_commands", {}) or {}
    else:
        qc = getattr(config, "quick_commands", {}) or {}
    if isinstance(qc, dict) and name in qc:
        return True

    # 3. Plugin command
    from hermes_cli.plugins import get_plugin_command_handler
    if get_plugin_command_handler(name.replace("_", "-")) is not None:
        return True

    # 4. Skill bundle
    from agent.skill_bundles import resolve_bundle_command_key
    if resolve_bundle_command_key(name) is not None:
        return True

    # 5. Active skill
    from agent.skill_commands import resolve_skill_command_key
    if resolve_skill_command_key(name) is not None:
        return True

    # 6. Known-but-disabled or uninstalled skill
    if unavailable_skill_fn is not None:
        if unavailable_skill_fn(name):
            return True

    return False


def canonicalize_event_for_model_alias(
    event: MessageEvent,
    *,
    config: object = None,
    unavailable_skill_fn: Optional[Callable[[str], Optional[str]]] = None,
) -> MessageEvent:
    """Return either the original event (no rewrite) or a ``dataclasses``
    clone whose ``text`` is the canonical ``/model <alias> <args>`` form.

    The helper bails out (returning ``event`` unchanged) when:
    * the typed name is not a known model alias;
    * the typed name is occupied by a higher-priority command;
    * any resolver raises an unexpected Exception (Fail-Closed).
    """
    name = _typed_command(getattr(event, "text", None))
    if name is None:
        return event

    _ensure_direct_aliases()
    if name not in DIRECT_ALIASES and name not in MODEL_ALIASES:
        return event

    try:
        if is_name_occupied(name, config=config, unavailable_skill_fn=unavailable_skill_fn):
            return event
    except Exception as err:
        logger.debug(
            "Model alias helper failed closed on command check '%s': %s",
            name,
            type(err).__name__,
        )
        return event

    # Build the canonical ``/model <alias> <args>`` text.
    raw = (getattr(event, "text", "") or "").lstrip()
    parts = raw.split(None, 1)
    args = parts[1].strip() if len(parts) > 1 else ""
    new_text = f"/model {name} {args}".strip()
    if new_text == raw.strip():
        return event
    return dataclasses.replace(event, text=new_text)