#!/usr/bin/env python3
"""
Tool Input Repair Layer for Hermes Agent

Validates and repairs tool call arguments when schema validation fails.
This is the tracer bullet implementation focused on field alias renaming.

Architecture:
    1. validate_tool_args() - Check arguments against schema
    2. repair_tool_args()   - Apply ordered repair rules when validation fails
    3. _rename_aliased_fields() - Rule 1: map common field aliases to canonical names

Public entry point: repair_tool_args(tool_name, args, validation_errors)
"""

import logging
from typing import Dict, Any, List, Optional, Set

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


def validate_tool_args(
    tool_name: str,
    args: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Validate tool arguments against their JSON Schema.

    Returns a list of validation error messages. Empty list means valid.

    This is deliberately simple: we only check for missing required fields
    and completely unknown fields. Full JSON Schema validation (type checks,
    format checks, enum validation) is handled downstream by coerce_tool_args()
    and the tool handlers themselves. Our repair layer only needs to detect
    structural problems we can fix (aliases, missing fields).

    Args:
        tool_name: Name of the tool being called
        args: Arguments passed to the tool
        schema: Optional tool schema (if None, fetched from registry)

    Returns:
        List of validation error messages (empty if valid)
    """
    if schema is None:
        from tools.registry import registry
        schema = registry.get_schema(tool_name)
        if not schema:
            return []

    properties = (schema.get("parameters") or {}).get("properties") or {}
    required: Set[str] = set((schema.get("parameters") or {}).get("required") or [])

    errors = []

    # Check for missing required fields
    for field in required:
        if field not in args:
            errors.append(f"Missing required field: '{field}'")

    # Note: we do NOT flag unknown fields here. Some tools accept optional
    # fields that aren't in the schema, and some tools handle extra args
    # gracefully. Only the handler can definitively know what's invalid.

    return errors


def repair_tool_args(
    tool_name: str,
    args: Dict[str, Any],
    validation_errors: List[str],
) -> Dict[str, Any]:
    """Attempt to repair tool arguments using ordered repair rules.

    This is the main entry point called from model_tools.py after
    coerce_tool_args() when validation fails. Each rule runs in order;
    the first successful repair stops further attempts.

    Args:
        tool_name: Name of the tool being called
        args: Original (invalid) arguments
        validation_errors: List of validation errors from validate_tool_args()

    Returns:
        Repaired arguments (possibly unchanged if no rule applied)
    """
    if not validation_errors:
        # No errors = already valid, nothing to do
        return args

    logger.debug("repair_tool_args: %s has validation errors: %s", tool_name, validation_errors)

    repaired = dict(args)

    # Rule 1: Rename aliased fields
    repaired = _rename_aliased_fields(tool_name, repaired)

    # Future rules will go here:
    # - Rule 2: Infer missing fields from context (e.g., path from CWD)
    # - Rule 3: Fix type mismatches (beyond what coerce_tool_args does)
    # - Rule 4: Apply default values for optional fields

    # Re-validate after repair attempts
    new_errors = validate_tool_args(tool_name, repaired)
    if new_errors:
        logger.debug("repair_tool_args: %s still has errors after repair: %s", tool_name, new_errors)

    return repaired


def _rename_aliased_fields(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 1: Rename common field aliases to canonical names.

    LLMs (especially GLM-5.2) often emit non-canonical field names that
    map to valid schema fields. This rule checks the alias map for the
    tool and renames any matching keys.

    Args:
        tool_name: Name of the tool being called
        args: Arguments to repair

    Returns:
        Arguments with aliases renamed (unchanged if no aliases match)
    """
    alias_map = FIELD_ALIAS_MAP.get(tool_name)
    if not alias_map:
        return args

    repaired = dict(args)
    renamed_fields = []

    for alias, canonical_name in alias_map.items():
        if alias in args:
            if canonical_name in args:
                # Both alias AND canonical field present: prefer canonical
                # (this shouldn't happen in practice, but defensive)
                logger.debug(
                    "_rename_aliased_fields: %s has both '%s' (alias) and '%s' (canonical). "
                    "Using canonical, dropping alias.",
                    tool_name, alias, canonical_name
                )
                del repaired[alias]
                renamed_fields.append(f"{alias} → {canonical_name} (redundant alias removed)")
            else:
                # Alias present, canonical not: rename
                repaired[canonical_name] = repaired.pop(alias)
                renamed_fields.append(f"{alias} → {canonical_name}")

    if renamed_fields:
        logger.info(
            "_rename_aliased_fields: repaired %s: %s",
            tool_name, ", ".join(renamed_fields)
        )

    return repaired


def append_repair_note(result: str, repairs: List[str]) -> str:
    """Append a note to tool result indicating which repairs were applied.

    This is called from model_tools.py after a repaired tool call succeeds.

    Args:
        result: Original tool result (JSON string)
        repairs: List of repair descriptions (e.g., ["file_path → path"])

    Returns:
        Result with repair note appended (as a JSON comment before the JSON)
    """
    if not repairs:
        return result

    note = f"[Tool input repair applied: {', '.join(repairs)}]"
    # Prepend the note before the JSON result
    return f"{note}\n{result}"