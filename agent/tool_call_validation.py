"""Pre-dispatch validation for model-emitted tool calls.

Provider/tool-call glitches can emit syntactically valid tool calls with empty
argument objects (for example ``terminal {}`` or ``write_file {}``).  Letting
those reach real handlers turns a provider-side malformed response into
side-effect-bearing tool errors that poison the active session history.  These
helpers validate the already-parsed argument dict against the registered tool
schema before any handler, checkpoint, approval prompt, or guardrail execution
runs.
"""

from __future__ import annotations

import json
from typing import Any, Mapping


def _schema_for_tool(function_name: str) -> dict[str, Any] | None:
    try:
        from tools.registry import registry

        entry = registry.get_entry(function_name)
    except Exception:
        return None
    if entry is None or not isinstance(getattr(entry, "schema", None), dict):
        return None
    return entry.schema


def _parameter_schema(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    params = schema.get("parameters")
    return params if isinstance(params, Mapping) else {}


def _properties(params: Mapping[str, Any]) -> Mapping[str, Any]:
    props = params.get("properties")
    return props if isinstance(props, Mapping) else {}


def _required(params: Mapping[str, Any]) -> list[str]:
    req = params.get("required")
    return [str(item) for item in req] if isinstance(req, list) else []


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (isinstance(value, int | float) and not isinstance(value, bool))
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def _field_type(properties: Mapping[str, Any], field: str) -> str | None:
    spec = properties.get(field)
    if not isinstance(spec, Mapping):
        return None
    raw = spec.get("type")
    return raw if isinstance(raw, str) else None


def _field_missing_or_empty(args: Mapping[str, Any], properties: Mapping[str, Any], field: str) -> bool:
    if field not in args or args.get(field) is None:
        return True
    expected = _field_type(properties, field)
    # Empty content is a valid write_file payload, but empty command/path/mode/etc.
    # are provider malformed-output symptoms, not meaningful calls.
    if expected == "string" and field != "content" and isinstance(args.get(field), str) and not args[field].strip():
        return True
    return False


def _conditional_required(function_name: str, args: Mapping[str, Any]) -> list[str]:
    """Return required fields that are conditional and not expressible in the
    current lightweight schemas.
    """
    missing: list[str] = []
    if function_name == "patch":
        mode = args.get("mode", "replace")
        if mode == "patch":
            if _field_missing_or_empty(args, {"patch": {"type": "string"}}, "patch"):
                missing.append("patch")
        else:
            for field in ("path", "old_string", "new_string"):
                if field not in args or args.get(field) is None:
                    missing.append(field)
                elif field != "new_string" and isinstance(args.get(field), str) and not args[field].strip():
                    missing.append(field)
    elif function_name == "cronjob":
        action = str(args.get("action") or "").strip()
        if action in {"update", "pause", "resume", "remove", "run"}:
            if _field_missing_or_empty(args, {"job_id": {"type": "string"}}, "job_id"):
                missing.append("job_id")
        if action == "create":
            for field in ("schedule", "prompt"):
                if _field_missing_or_empty(args, {field: {"type": "string"}}, field):
                    missing.append(field)
        if args.get("no_agent") is True and _field_missing_or_empty(args, {"script": {"type": "string"}}, "script"):
            missing.append("script")
    return missing


def validate_tool_call_arguments(function_name: str, function_args: Any) -> str | None:
    """Return a JSON error result when *function_args* must not dispatch.

    ``None`` means the call is safe to continue to the normal tool path.  The
    validator is intentionally conservative: tools not registered in the normal
    registry (agent-loop/context/memory tools) are allowed unless their own
    special-case checks below apply.
    """
    if not isinstance(function_args, dict):
        return json.dumps(
            {
                "error": "invalid_tool_arguments: tool arguments must be a JSON object",
                "code": "invalid_tool_arguments",
                "tool": function_name,
                "invalid_fields": ["arguments"],
            },
            ensure_ascii=False,
        )

    schema = _schema_for_tool(function_name)
    if schema is None:
        return None

    params = _parameter_schema(schema)
    props = _properties(params)
    missing: list[str] = []
    invalid: list[str] = []

    for field in _required(params):
        if _field_missing_or_empty(function_args, props, field):
            missing.append(field)
            continue
        expected = _field_type(props, field)
        if expected and not _json_type_matches(function_args.get(field), expected):
            invalid.append(field)

    for field, spec in props.items():
        if field not in function_args or function_args.get(field) is None:
            continue
        if not isinstance(spec, Mapping):
            continue
        expected = spec.get("type")
        if isinstance(expected, str) and not _json_type_matches(function_args[field], expected):
            invalid.append(str(field))

    for field in _conditional_required(function_name, function_args):
        if field not in missing:
            missing.append(field)

    if not missing and not invalid:
        return None

    parts: list[str] = []
    if missing:
        parts.append("missing required field(s): " + ", ".join(missing))
    if invalid:
        parts.append("invalid field type(s): " + ", ".join(invalid))
    return json.dumps(
        {
            "error": f"invalid_tool_arguments: {function_name} " + "; ".join(parts),
            "code": "invalid_tool_arguments",
            "tool": function_name,
            "missing_required": missing,
            "invalid_fields": invalid,
        },
        ensure_ascii=False,
    )
