"""Shared logic for the ``/subagent`` slash command (CLI and gateway call into this module).

``/subagent model`` prints the effective delegation route; ``/subagent model set
<provider>/<model>`` pins ``delegation.provider`` + ``delegation.model`` so every
``delegate_task`` child spawns on that pair. Persistence goes through the CLI config-write
helper (``cli.save_config_value`` -> atomic round-trip YAML update) — YAML is never
hand-edited. Mirrors ``hermes_cli/codex_runtime_switch.py``: pure parse/format/apply logic
here, surface-specific rendering in the CLI handler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

USAGE = "usage: /subagent model [set <provider>/<model>]"

_SUBCOMMAND = "model"
_SET_VERB = "set"
# A provider slug is a single path-safe token; the MODEL half may itself contain "/"
# (e.g. "openrouter/google/gemini-3-flash-preview"), so the split is on the FIRST slash.
_PROVIDER_RE = re.compile(r"[A-Za-z0-9._-]+\Z")


@dataclass
class SubagentModelRequest:
    """Parsed ``/subagent`` invocation. ``action`` is ``"show"`` or ``"set"``; a malformed
    invocation carries ``errors`` and never a target."""

    action: str = "show"
    provider: str = ""
    model: str = ""
    errors: list[str] = field(default_factory=list)


@dataclass
class SubagentModelStatus:
    """Result of committing a pin; callers render ``message`` per surface."""

    success: bool
    provider: str = ""
    model: str = ""
    old_provider: str = ""
    old_model: str = ""
    message: str = ""


def _parse_target(value: str) -> tuple[str, str, Optional[str]]:
    """``<provider>/<model>`` -> ``(provider, model, None)``; ``("", "", error)`` when malformed."""
    if not value:
        return "", "", f"missing <provider>/<model> value. {USAGE}"
    provider, slash, model = value.partition("/")
    provider, model = provider.strip(), model.strip()
    if not slash or not provider or not model:
        return "", "", f"expected <provider>/<model>, got {value!r}. {USAGE}"
    if not _PROVIDER_RE.match(provider):
        return "", "", f"invalid provider {provider!r}. {USAGE}"
    return provider, model, None


def parse_args(arg_string: str) -> SubagentModelRequest:
    """Parse the text after ``/subagent`` into a :class:`SubagentModelRequest`.

    ``""`` / ``"model"`` -> show; ``"model set <provider>/<model>"`` -> set; anything else
    is an error (unknown subcommand, unknown verb, empty or malformed value)."""
    raw = (arg_string or "").strip()
    if not raw:
        return SubagentModelRequest(errors=[f"missing subcommand. {USAGE}"])
    head_parts = raw.split(None, 1)
    head = head_parts[0]
    if head.lower() != _SUBCOMMAND:
        return SubagentModelRequest(errors=[f"unknown subcommand {head!r}. {USAGE}"])
    rest = head_parts[1].strip() if len(head_parts) > 1 else ""
    if not rest:
        return SubagentModelRequest(action="show")
    verb_parts = rest.split(None, 1)
    verb = verb_parts[0]
    if verb.lower() != _SET_VERB:
        return SubagentModelRequest(errors=[f"unknown argument {verb!r}. {USAGE}"])
    value = verb_parts[1].strip() if len(verb_parts) > 1 else ""
    provider, model, err = _parse_target(value)
    if err:
        return SubagentModelRequest(errors=[err])
    return SubagentModelRequest(action="set", provider=provider, model=model)


def get_current(config: dict) -> tuple[str, str]:
    """``(delegation.provider, delegation.model)`` from *config*; ``("", "")`` when unset."""
    delegation = config.get("delegation") if isinstance(config, dict) else None
    if not isinstance(delegation, dict):
        return "", ""
    return str(delegation.get("provider") or "").strip(), str(delegation.get("model") or "").strip()


def format_status(config: dict) -> list[str]:
    """Human-readable lines describing the effective subagent route (bare ``/subagent model``)."""
    provider, model = get_current(config)
    if provider and model:
        effective = f"{provider}/{model}"
    elif model:
        effective = f"<inherited provider>/{model}"
    elif provider:
        effective = f"{provider}/<inherited model>"
    else:
        effective = "inherit (parent agent's provider and model)"
    delegation = config.get("delegation") if isinstance(config, dict) else None
    hot = bool(delegation.get("hot_reload_model", False)) if isinstance(delegation, dict) else False
    return [
        f"Subagent model: {effective}",
        f"  delegation.provider: {provider or '(unset — inherit parent)'}",
        f"  delegation.model: {model or '(unset — inherit parent)'}",
        f"  delegation.hot_reload_model: {str(hot).lower()}",
    ]


def apply(config: dict, provider: str, model: str, *, persist_callback=None) -> SubagentModelStatus:
    """Write the pin into ``config['delegation']`` in place, then persist.

    ``persist_callback(config) -> bool`` is the config-write helper (skipped when None); a
    ``False`` return or a raised exception fails the status without pretending success."""
    old_provider, old_model = get_current(config)
    if not isinstance(config.get("delegation"), dict):
        config["delegation"] = {}
    config["delegation"]["provider"] = provider
    config["delegation"]["model"] = model
    if persist_callback is not None:
        try:
            written = persist_callback(config)
        except Exception as exc:  # surface the write failure, never a silent in-memory-only change
            return SubagentModelStatus(
                False, provider, model, old_provider, old_model,
                message=f"updated the running config but persisting failed: {exc}")
        if not written:
            return SubagentModelStatus(
                False, provider, model, old_provider, old_model,
                message="updated the running config but writing config.yaml failed")
    return SubagentModelStatus(
        True, provider, model, old_provider, old_model,
        message=f"delegation model: {old_provider or '(inherit)'}/{old_model or '(inherit)'} "
                f"→ {provider}/{model}")
