"""Argument-name checks for the tools Hermes ships.

A call that omits a required argument is refused before the handler runs, naming the
field; a call that carries a name the schema does not declare still reaches the handler
untouched, and its result says which names were not parameters. Third-party schemas
(MCP servers, plugins) and open or union-shaped schemas are not judged here: their
``properties`` list is not a promise about what the handler accepts. The exception is a
third-party schema that closes the object itself (``additionalProperties: false``): that IS
the promise, so an undeclared name there is refused before the call.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from hermes_agent.tools.registry import ToolEntry, registry, tool_error

logger = logging.getLogger(__name__)

# Root keywords under which ``properties``/``required`` no longer enumerate the argument set.
_OPEN_ROOT_KEYWORDS = ("anyOf", "oneOf", "allOf", "$ref", "patternProperties")


def _closed_parameters(entry: Optional[ToolEntry]) -> Optional[dict]:
    """The registered parameters when they enumerate a built-in tool's arguments, else None."""
    if entry is None or not registry.is_builtin(entry):
        return None
    params = entry.schema.get("parameters")
    if not isinstance(params, dict) or not isinstance(params.get("properties"), dict) or not params["properties"]:
        return None
    if params.get("additionalProperties") or any(key in params for key in _OPEN_ROOT_KEYWORDS):
        return None
    return params


def _served_parameters(entry: ToolEntry, params: dict) -> Optional[dict]:
    """*params* as the model was served them: the dynamic override decides ``required``, and
    a name declared by either side counts as a parameter. None when the override fails."""
    try:
        live = entry.dynamic_schema_overrides().get("parameters")
    except Exception:
        logger.debug("Dynamic schema unavailable for %s; skipping argument checks", entry.name, exc_info=True)
        return None
    if not isinstance(live, dict):
        return params
    return {**live, "properties": {**params["properties"], **(live.get("properties") or {})}}


def _missing_and_unknown(entry: ToolEntry, params: dict, args: dict) -> tuple[list[str], list[str]]:
    unknown = [key for key in args if key not in params["properties"] and key not in entry.compat_args]
    # A compat argument marks a call shape the schema does not describe; the handler owns it.
    if any(key in entry.compat_args for key in args):
        return [], unknown
    return [key for key in params.get("required") or ()
            if key not in args and key not in entry.self_checked_args], unknown


def _with_choices(key: str, properties: dict) -> str:
    """*key*, followed by its allowed values when the schema enumerates them."""
    spec = properties.get(key)
    choices = spec.get("enum") if isinstance(spec, dict) else None
    return f"{key} (one of: {', '.join(map(str, choices))})" if choices else key


def _closed_schema_refusal(entry: Optional[ToolEntry], args: Any) -> Optional[str]:
    """Refusal for names a third-party schema that closes the object (``additionalProperties:
    false``) does not declare. zod/pydantic-backed MCP servers advertise that form yet strip
    unknown keys at parse time, so a wrapped or misspelled call came back as a default success
    and the model kept the shape all session (port of RooCodeInc/Roomote#2695). Built-in tools
    keep the warn-and-run contract below; the one that closes its schema checks keys itself."""
    if entry is None or not isinstance(args, dict) or registry.is_builtin(entry):
        return None
    params = entry.schema.get("parameters")
    if not isinstance(params, dict):
        return None
    # An omitted ``properties`` declares no arguments; a non-dict one is malformed and left alone.
    properties = params.get("properties", {})
    if (not isinstance(properties, dict) or params.get("additionalProperties") is not False
            or any(key in params for key in _OPEN_ROOT_KEYWORDS)):
        return None
    unknown = [key for key in args if key not in properties]
    if unknown and entry.dynamic_schema_overrides is not None:
        # A name the served (dynamic) schema adds was offered to the model: never refuse it.
        served = _served_parameters(entry, {**params, "properties": properties})
        if served is None:
            return None
        properties = served["properties"]
        unknown = [key for key in unknown if key not in properties]
    if not unknown:
        return None
    valid = ", ".join(properties) or "none"
    unwrap = (" Do not wrap the arguments in an \"arguments\" field (that shape is only for tool_call);"
              " pass each one at the top level." if "arguments" in unknown else "")
    return tool_error(f"{', '.join(unknown)}: not a parameter of {entry.name}. The tool was NOT invoked. "
                      f"Valid parameters: {valid}.{unwrap}")


def check_tool_args(entry: Optional[ToolEntry], args: Any) -> tuple[Optional[str], list[str], str]:
    """``(refusal, unknown, valid)`` for one call: *refusal* is the error result to return
    instead of running the handler, *unknown* the argument names the schema does not declare,
    *valid* the declared ones."""
    if (refusal := _closed_schema_refusal(entry, args)) is not None:
        return refusal, [], ""
    params = _closed_parameters(entry)
    if params is None or not isinstance(args, dict):
        return None, [], ""
    name = entry.name
    missing, unknown = _missing_and_unknown(entry, params, args)
    # The registered schema settles the common case; only a finding pays for the dynamic one.
    if (missing or unknown) and entry.dynamic_schema_overrides is not None:
        params = _served_parameters(entry, params)
        if params is None:
            return None, [], ""
        missing, unknown = _missing_and_unknown(entry, params, args)
    valid = ", ".join(params["properties"])
    if missing:
        return tool_error(f"{name} is missing required argument(s): "
                          f"{', '.join(_with_choices(key, params['properties']) for key in missing)}. "
                          f"The tool was NOT invoked. Valid parameters: {valid}."), [], valid
    return None, unknown, valid


def flag_unknown_args(result: Any, name: str, unknown: list[str], valid: str) -> Any:
    """Name *unknown* in the ``_warning`` of a JSON-object result; any other result is
    returned unchanged."""
    if not unknown or not isinstance(result, str):
        return result
    try:
        payload = json.loads(result)
    except ValueError:
        return result
    if not isinstance(payload, dict):
        return result
    warning = (f"{', '.join(unknown)}: not a parameter of {name} and was probably ignored. "
               f"Valid parameters: {valid}.")
    previous = payload.get("_warning")
    if previous and (not isinstance(previous, str) or warning in previous):
        # A structured warning is the tool's own; a repeat is an inline executor that
        # dispatches through the registry and is checked twice.
        return result
    payload["_warning"] = f"{previous}\n{warning}" if previous else warning
    return json.dumps(payload, ensure_ascii=False)


def call_with_arg_checks(entry: Optional[ToolEntry], args: Any, call: Callable[[Any], Any]) -> Any:
    """Run ``call(args)`` unless argument validation refuses the call; *args* is passed as received."""
    refusal, unknown, valid = check_tool_args(entry, args)
    if refusal is not None:
        return refusal
    result = call(args)
    return flag_unknown_args(result, entry.name, unknown, valid) if unknown else result
