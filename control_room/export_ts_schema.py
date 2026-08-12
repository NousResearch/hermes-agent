"""Generate the TypeScript mirror of the Control Room contract.

Strategy (CR-005): Python is authoritative. This script walks the pydantic
models in ``control_room/contract.py``, builds their JSON schema, and emits a
deterministic TypeScript types file into ``control_room/schema/control-room-v1.ts``.

Determinism rules:
- No timestamps in the output (byte-for-byte stable across runs).
- Enums emit as literal unions in declaration order.
- Field ordering follows the model's ``model_fields`` order.
- A regeneration test asserts the checked-in file matches ``generate()``.

Usage:  python -m control_room.export_ts_schema [--check]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from control_room.contract import (
    ActionTarget,
    AgentRow,
    AttentionItem,
    AttentionKind,
    AttentionSeverity,
    Capabilities,
    ControlRoomAction,
    ControlRoomActionResult,
    ControlRoomError,
    ControlRoomSnapshot,
    ErrorCode,
    MessageRow,
    SnapshotCounts,
    SourceMeta,
    SystemSummary,
    TaskRow,
)

SCHEMA_PATH = Path(__file__).parent / "schema" / "control-room-v1.ts"

MODELS: List[Tuple[str, type]] = [
    ("AttentionSeverity", AttentionSeverity),
    ("AttentionKind", AttentionKind),
    ("ErrorCode", ErrorCode),
    ("SourceMeta", SourceMeta),
    ("AttentionItem", AttentionItem),
    ("AgentRow", AgentRow),
    ("TaskRow", TaskRow),
    ("MessageRow", MessageRow),
    ("SystemSummary", SystemSummary),
    ("Capabilities", Capabilities),
    ("SnapshotCounts", SnapshotCounts),
    ("ActionTarget", ActionTarget),
    ("ControlRoomAction", ControlRoomAction),
    ("ControlRoomActionResult", ControlRoomActionResult),
    ("ControlRoomError", ControlRoomError),
    ("ControlRoomSnapshot", ControlRoomSnapshot),
]

HEADER = """// Control Room contract — TypeScript mirror (generated, do not edit).
// Source of truth: control_room/contract.py (Python, pydantic v2).
// Regenerate with: python -m control_room.export_ts_schema
// Wire format is snake_case; these types mirror the JSON payloads exactly.
"""


def _py_type_to_ts(schema: dict) -> str:
    """Map a JSON-schema fragment to a TS type expression."""
    if "enum" in schema:
        return " | ".join(json.dumps(v) for v in schema["enum"])
    if "anyOf" in schema:
        return " | ".join(_py_type_to_ts(s) for s in schema["anyOf"])
    t = schema.get("type")
    if t == "integer":
        return "number"
    if t == "number":
        return "number"
    if t == "boolean":
        return "boolean"
    if t == "string":
        return "string"
    if t == "null":
        return "null"
    if t == "array":
        return f"Array<{_py_type_to_ts(schema.get('items', {}))}>"
    if t == "object" or "$ref" in schema:
        ref = schema.get("$ref", "")
        return ref.rsplit("/", 1)[-1] if ref else "Record<string, unknown>"
    return "unknown"


def _interface_for(name: str, schema: dict) -> str:
    lines = [f"export interface {name} {{"]
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    for prop_name, prop_schema in props.items():
        ts_type = _py_type_to_ts(prop_schema)
        opt = "" if prop_name in required else "?"
        lines.append(f"  {prop_name}{opt}: {ts_type}")
    lines.append("}")
    return "\n".join(lines)


def _enum_schema(cls) -> dict:
    """Build a JSON-schema fragment for an Enum class from its members."""
    values = [m.value for m in cls]
    if all(isinstance(v, str) for v in values):
        return {"type": "string", "enum": values}
    return {"type": "integer", "enum": values}


def generate() -> str:
    chunks: List[str] = [HEADER]
    for name, model in MODELS:
        if isinstance(model, type) and issubclass(model, (AttentionSeverity, AttentionKind, ErrorCode)):
            chunks.append(f"export type {name} = {_py_type_to_ts(_enum_schema(model))};")
            chunks.append("")
            continue
        schema = model.model_json_schema()
        chunks.append(_interface_for(name, schema))
        chunks.append("")
    return "\n".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate TS mirror of Control Room contract")
    parser.add_argument("--check", action="store_true", help="exit non-zero if output differs from checked-in file")
    args = parser.parse_args()

    output = generate()
    if args.check:
        if SCHEMA_PATH.exists() and SCHEMA_PATH.read_text() == output:
            print(f"OK: {SCHEMA_PATH} is up to date")
            return 0
        print(f"STALE: {SCHEMA_PATH} differs from generated output")
        return 1

    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(output)
    print(f"wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
