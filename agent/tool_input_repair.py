#!/usr/bin/env python3
"""
Tool Input Repair Layer for Hermes Agent

Validates and repairs tool call arguments when schema validation fails.
This started as a tracer bullet for field alias renaming and now handles a
small set of structural repair rules that coerce_tool_args() does not cover.

Architecture:
    1. validate_tool_args() - Check arguments against schema
    2. repair_tool_args()   - Apply ordered repair rules when validation fails
    3. _rename_aliased_fields() - Rule 1: map common field aliases to canonical names
    4. Rule 2-5: drop nullish/placeholder fields, parse/wrap array strings,
       and wrap a root JSON string into an object

Public entry point: repair_tool_args(tool_name, args, validation_errors)
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Alias Map: common field aliases → canonical Hermes field names
# =============================================================================
#
# Derived from actual Hermes tool schemas, not copied from external projects.
# These are aliases that LLMs (especially GLM) commonly emit instead of the
# canonical field names.
#
FIELD_ALIAS_MAP: Dict[str, Dict[str, str]] = {
    "read_file": {
        "file_path": "path",
        "filePath": "path",
        "filepath": "path",
    },
    "write_file": {
        "file_path": "path",
        "filePath": "path",
        "filepath": "path",
    },
    "patch": {
        "oldValue": "old_string",
        "oldvalue": "old_string",
        "newValue": "new_string",
        "newvalue": "new_string",
    },
    "search_files": {
        "file_path": "path",
        "filePath": "path",
        "filepath": "path",
    },
    "terminal": {
        "command": "command",  # No common aliases for terminal
    },
}


def _get_tool_schema(tool_name: str) -> Optional[Dict[str, Any]]:
    from tools.registry import registry

    return registry.get_schema(tool_name)


def _schema_parameters(schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    params = schema.get("parameters") or {}
    return params if isinstance(params, dict) else {}


def _schema_properties(schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    properties = _schema_parameters(schema).get("properties") or {}
    return properties if isinstance(properties, dict) else {}


def _schema_required(schema: Optional[Dict[str, Any]]) -> Set[str]:
    required = _schema_parameters(schema).get("required") or []
    return set(required) if isinstance(required, list) else set()


def _schema_accepts_kind(schema: Any, kind: str) -> bool:
    """Return True when *schema* permits a value of JSON type *kind*."""
    if not isinstance(schema, dict):
        return False
    t = schema.get("type")
    if t == kind or (isinstance(t, list) and kind in t):
        return True
    for union_key in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(union_key)
        if isinstance(branches, list) and any(
            _schema_accepts_kind(branch, kind) for branch in branches
        ):
            return True
    return False


def _is_nullish_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"null", "undefined"}
    return False


def _is_empty_object_placeholder(value: Any) -> bool:
    return isinstance(value, dict) and not value


def _coerce_root_string_to_object(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    trimmed = value.strip()
    if not trimmed.startswith("{"):
        return value
    try:
        parsed = json.loads(trimmed)
    except (ValueError, TypeError):
        return value
    return parsed if isinstance(parsed, dict) else value


def _parse_json_stringified_array(value: Any) -> Tuple[Any, bool]:
    if not isinstance(value, str):
        return value, False
    trimmed = value.strip()
    if not trimmed.startswith("["):
        return value, False
    try:
        parsed = json.loads(trimmed)
    except (ValueError, TypeError):
        return value, False
    if isinstance(parsed, list):
        return parsed, True
    return value, False


def _wrap_bare_string_as_array(value: Any) -> Tuple[Any, bool]:
    if isinstance(value, str):
        return [value], True
    return value, False


def _validation_issue_codes(validation_errors: List[str]) -> List[str]:
    codes: List[str] = []
    for error in validation_errors:
        lowered = error.lower()
        if lowered.startswith("missing required field:"):
            codes.append("missing_required_field")
        elif "null/undefined" in lowered:
            codes.append("nullish_placeholder")
        elif "empty object placeholder" in lowered:
            codes.append("empty_object_placeholder")
        elif "must be an array" in lowered:
            codes.append("expected_array")
        elif "must be an object" in lowered:
            codes.append("expected_object")
        elif lowered.startswith("tool arguments must be a json object"):
            codes.append("non_object_args")
        else:
            codes.append("validation_error")
    return codes


def validate_tool_args(
    tool_name: str,
    args: Any,
    schema: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Validate tool arguments against their JSON Schema.

    Returns a list of validation error messages. Empty list means valid.

    This is deliberately simple: we only check for missing required fields and
    the structural mismatches that the repair layer can fix. Full JSON Schema
    validation (type checks, format checks, enum validation) is handled
    downstream by coerce_tool_args() and the tool handlers themselves.
    """
    if schema is None:
        schema = _get_tool_schema(tool_name)
        if not schema:
            return []

    properties = _schema_properties(schema)
    required: Set[str] = _schema_required(schema)

    if not isinstance(args, dict):
        return ["Tool arguments must be a JSON object"]

    errors: List[str] = []

    # Check for missing required fields and obvious placeholders.
    for field in required:
        if field not in args:
            errors.append(f"Missing required field: '{field}'")
            continue

    # Check all provided fields for placeholder values and drift that the
    # repair layer knows how to fix.
    for field, value in args.items():
        prop_schema = properties.get(field)
        if _is_nullish_placeholder(value):
            errors.append(f"Field '{field}' must not be null/undefined")
            continue
        if isinstance(prop_schema, dict):
            expects_array = _schema_accepts_kind(prop_schema, "array")
            expects_object = _schema_accepts_kind(prop_schema, "object")
            if _is_empty_object_placeholder(value) and not expects_object:
                errors.append(f"Field '{field}' must not be an empty object placeholder")
                continue
            if expects_array and isinstance(value, str):
                errors.append(f"Field '{field}' must be an array, got string")
            elif expects_object and isinstance(value, str):
                errors.append(f"Field '{field}' must be an object, got string")
            elif expects_array and _is_empty_object_placeholder(value):
                errors.append(f"Field '{field}' must be an array, got empty object placeholder")
            elif expects_object and isinstance(value, list):
                errors.append(f"Field '{field}' must be an object, got array")
            elif expects_array and isinstance(value, dict) and not value:
                errors.append(f"Field '{field}' must be an array, got empty object placeholder")

    return errors


def repair_tool_args(
    tool_name: str,
    args: Any,
    validation_errors: List[str],
) -> Any:
    """Attempt to repair tool arguments using ordered repair rules."""
    if not validation_errors:
        return args

    logger.debug("repair_tool_args: %s has validation errors: %s", tool_name, validation_errors)
    schema = _get_tool_schema(tool_name)
    repaired: Any = args

    # Rule 1: Rename aliased fields.
    if isinstance(repaired, dict):
        repaired = _rename_aliased_fields(tool_name, repaired)

    # Rule 2: Drop null/undefined fields.
    repaired = _drop_null_or_undefined_fields(tool_name, repaired, schema)

    # Rule 3: Drop empty object placeholders.
    repaired = _drop_empty_object_placeholders(tool_name, repaired, schema)

    # Rule 4: Parse JSON-stringified arrays.
    repaired = _parse_json_stringified_arrays(tool_name, repaired, schema)

    # Rule 5: Wrap bare strings as arrays.
    repaired = _wrap_bare_string_as_arrays(tool_name, repaired, schema)

    # Rule 6: Wrap a root JSON string into an object.
    repaired = _wrap_root_string_as_object(tool_name, repaired, schema)

    new_errors = validate_tool_args(tool_name, repaired, schema)
    if new_errors:
        logger.debug(
            "repair_tool_args: %s still has errors after repair: %s",
            tool_name,
            new_errors,
        )

    return repaired


def _drop_null_or_undefined_fields(
    tool_name: str,
    args: Any,
    schema: Optional[Dict[str, Any]],
) -> Any:
    if not isinstance(args, dict):
        return args

    properties = _schema_properties(schema)
    required = _schema_required(schema)
    repaired = dict(args)
    changed = False

    for key, value in list(args.items()):
        prop_schema = properties.get(key)
        if key not in required and _is_nullish_placeholder(value):
            del repaired[key]
            changed = True
            logger.info(
                "_drop_null_or_undefined_fields: dropped nullish optional field %s.%s",
                tool_name,
                key,
            )
            continue
        if isinstance(value, dict):
            nested = _drop_null_or_undefined_fields(tool_name, value, prop_schema)
            if nested is not value:
                repaired[key] = nested
                changed = True
        elif isinstance(value, list):
            nested = _drop_null_or_undefined_fields(tool_name, value, prop_schema)
            if nested is not value:
                repaired[key] = nested
                changed = True

    return repaired if changed else args


def _drop_empty_object_placeholders(
    tool_name: str,
    args: Any,
    schema: Optional[Dict[str, Any]],
) -> Any:
    if not isinstance(args, dict):
        return args

    properties = _schema_properties(schema)
    required = _schema_required(schema)
    repaired = dict(args)
    changed = False

    for key, value in list(args.items()):
        prop_schema = properties.get(key)
        if key not in required and _is_empty_object_placeholder(value):
            del repaired[key]
            changed = True
            logger.info(
                "_drop_empty_object_placeholders: dropped empty object placeholder %s.%s",
                tool_name,
                key,
            )
            continue
        if isinstance(value, dict):
            nested = _drop_empty_object_placeholders(tool_name, value, prop_schema)
            if nested is not value:
                repaired[key] = nested
                changed = True
        elif isinstance(value, list):
            nested = _drop_empty_object_placeholders(tool_name, value, prop_schema)
            if nested is not value:
                repaired[key] = nested
                changed = True

    return repaired if changed else args


def _parse_json_stringified_arrays(
    tool_name: str,
    args: Any,
    schema: Optional[Dict[str, Any]],
) -> Any:
    if not isinstance(args, dict):
        return args

    properties = _schema_properties(schema)
    repaired = dict(args)
    changed = False

    for key, value in list(args.items()):
        prop_schema = properties.get(key)
        if isinstance(value, dict):
            nested = _parse_json_stringified_arrays(tool_name, value, prop_schema)
            if nested is not value:
                repaired[key] = nested
                changed = True
            continue
        if isinstance(value, list):
            item_schema = {}
            if isinstance(prop_schema, dict):
                item_schema = prop_schema.get("items") or {}
                item_schema = item_schema if isinstance(item_schema, dict) else {}
            nested_items = []
            item_changed = False
            for item in value:
                nested = _parse_json_stringified_arrays(tool_name, item, item_schema)
                if nested is not item:
                    item_changed = True
                nested_items.append(nested)
            if item_changed:
                repaired[key] = nested_items
                changed = True
            continue
        if not isinstance(prop_schema, dict) or not _schema_accepts_kind(prop_schema, "array"):
            continue
        parsed, did_parse = _parse_json_stringified_array(value)
        if did_parse:
            repaired[key] = parsed
            changed = True
            logger.info(
                "_parse_json_stringified_arrays: parsed JSON array string for %s.%s",
                tool_name,
                key,
            )

    return repaired if changed else args


def _wrap_bare_string_as_arrays(
    tool_name: str,
    args: Any,
    schema: Optional[Dict[str, Any]],
) -> Any:
    if not isinstance(args, dict):
        return args

    properties = _schema_properties(schema)
    repaired = dict(args)
    changed = False

    for key, value in list(args.items()):
        prop_schema = properties.get(key)
        if isinstance(value, dict):
            nested = _wrap_bare_string_as_arrays(tool_name, value, prop_schema)
            if nested is not value:
                repaired[key] = nested
                changed = True
            continue
        if isinstance(value, list):
            item_schema = {}
            if isinstance(prop_schema, dict):
                item_schema = prop_schema.get("items") or {}
                item_schema = item_schema if isinstance(item_schema, dict) else {}
            nested_items = []
            item_changed = False
            for item in value:
                nested = _wrap_bare_string_as_arrays(tool_name, item, item_schema)
                if nested is not item:
                    item_changed = True
                nested_items.append(nested)
            if item_changed:
                repaired[key] = nested_items
                changed = True
            continue
        if not isinstance(prop_schema, dict) or not _schema_accepts_kind(prop_schema, "array"):
            continue
        wrapped, did_wrap = _wrap_bare_string_as_array(value)
        if did_wrap:
            repaired[key] = wrapped
            changed = True
            logger.info(
                "_wrap_bare_string_as_arrays: wrapped bare string as array for %s.%s",
                tool_name,
                key,
            )

    return repaired if changed else args


def _wrap_root_string_as_object(tool_name: str, args: Any, schema: Optional[Dict[str, Any]]) -> Any:
    if not isinstance(args, str):
        return args
    if not _schema_accepts_kind(_schema_parameters(schema), "object"):
        return args
    parsed = _coerce_root_string_to_object(args)
    if parsed is not args:
        logger.info("_wrap_root_string_as_object: parsed root JSON string for %s", tool_name)
    return parsed


def _rename_aliased_fields(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 1: Rename common field aliases to canonical names."""
    alias_map = FIELD_ALIAS_MAP.get(tool_name)
    if not alias_map:
        return args

    repaired = dict(args)
    renamed_fields = []

    for alias, canonical_name in alias_map.items():
        if alias in args:
            if canonical_name in args:
                logger.debug(
                    "_rename_aliased_fields: %s has both '%s' (alias) and '%s' (canonical). "
                    "Using canonical, dropping alias.",
                    tool_name,
                    alias,
                    canonical_name,
                )
                del repaired[alias]
                renamed_fields.append(f"{alias} → {canonical_name} (redundant alias removed)")
            else:
                repaired[canonical_name] = repaired.pop(alias)
                renamed_fields.append(f"{alias} → {canonical_name}")

    if renamed_fields:
        logger.info(
            "_rename_aliased_fields: repaired %s: %s",
            tool_name,
            ", ".join(renamed_fields),
        )

    return repaired


def format_error_for_model(tool_name: str, validation_errors: List[str]) -> str:
    """Format validation errors as a clean bullet list for the model."""
    if not validation_errors:
        return ""

    lines = [f"{tool_name}:"]
    for error in validation_errors:
        field_name = None
        message = error.strip()
        lowered = error.lower()

        if lowered.startswith("missing required field:"):
            message = "missing required field"
            remainder = error.split(":", 1)[1].strip() if ":" in error else ""
            field_name = remainder.strip("'\"") or None
        elif ":" in error:
            field_name, message = [part.strip() for part in error.split(":", 1)]
            field_name = field_name.strip("'\"") or None

        if field_name:
            lines.append(f"  - {field_name}: {message}")
        else:
            lines.append(f"  - {message}")

    return "\n".join(lines)


def emit_repair_telemetry(
    *,
    event: str,
    tool_name: str,
    model: str,
    rule_fired: str,
    issue_codes: List[str],
    repairs: Optional[List[str]] = None,
) -> None:
    """Emit a local-only structured log line for repair telemetry."""
    payload = {
        "event": event,
        "tool_name": tool_name,
        "model": model,
        "rule_fired": rule_fired,
        "issue_codes": issue_codes,
    }
    if repairs:
        payload["repairs"] = repairs
    logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def append_repair_note(result: str, repairs: List[str]) -> str:
    """Append a note to tool result indicating which repairs were applied."""
    if not repairs:
        return result

    note = f"[Tool input repair applied: {', '.join(repairs)}]"
    return f"{note}\n{result}"
