"""Conservative, offline comparison of two MCP tool manifests.

This is a change inspector, not a JSON Schema containment solver. A narrowing
rule is a potential break; unsupported constraints require a human review.
Nothing here imports an MCP client, resolves a reference, or executes a tool.
"""

from __future__ import annotations

from typing import Any

_TYPES = {"null", "boolean", "object", "array", "number", "integer", "string"}
_ANNOTATIONS = {"title", "description", "examples", "default", "deprecated", "readOnly", "writeOnly"}
_RULES = {"type", "enum", "required", "properties", "additionalProperties"}
_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def _value_key(value: Any) -> tuple:
    # JSON booleans are not numbers; Python's True == 1 would miss enum removals.
    if isinstance(value, dict):
        return ("object", tuple((k, _value_key(v)) for k, v in sorted(value.items())))
    if isinstance(value, list):
        return ("array", tuple(_value_key(v) for v in value))
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, (int, float)):
        return ("number", value)
    return (type(value).__name__, value)


def _pointer(path: str, key: str) -> str:
    return path + "/" + key.replace("~", "~0").replace("/", "~1")


def _types(schema: dict) -> set[str]:
    value = schema.get("type", sorted(_TYPES))
    result = {value} if isinstance(value, str) else set(value)
    if "number" in result:
        result.add("integer")
    return result


def _unsupported(schema: Any, path: str) -> list[str]:
    """Return locations we cannot reason about, including malformed supported rules."""
    if isinstance(schema, bool):
        return []
    if not isinstance(schema, dict):
        return [path]
    problems = [_pointer(path, k) for k in sorted(schema.keys() - _RULES - _ANNOTATIONS - {"$schema"})]
    if "$schema" in schema and schema["$schema"] != _DIALECT:
        problems.append(_pointer(path, "$schema"))
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    if (not isinstance(types, list) or any(not isinstance(t, str) or t not in _TYPES for t in types)
            or ("type" in schema and not types)):
        problems.append(_pointer(path, "type"))
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(k, str) for k in required):
        problems.append(_pointer(path, "required"))
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        problems.append(_pointer(path, "enum"))
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        problems.append(_pointer(path, "properties"))
    else:
        for name, child in sorted(properties.items()):
            problems.extend(_unsupported(child, _pointer(_pointer(path, "properties"), name)))
    if "additionalProperties" in schema:
        problems.extend(_unsupported(schema["additionalProperties"], _pointer(path, "additionalProperties")))
    return problems


def _change(changes: list, tool: str, path: str, severity: str, message: str) -> None:
    changes.append({"tool": tool, "path": path, "severity": severity, "message": message})


def _compare_schema(old: Any, new: Any, tool: str, path: str, changes: list) -> None:
    if _value_key(old) == _value_key(new):
        return
    old_metadata = old if isinstance(old, dict) else {}
    new_metadata = new if isinstance(new, dict) else {}
    for key in sorted(_ANNOTATIONS):
        if (_value_key(old_metadata.get(key)) != _value_key(new_metadata.get(key))
                or (key in old_metadata) != (key in new_metadata)):
            _change(changes, tool, _pointer(path, key), "review", "Schema annotation changed; review its meaning.")
    if old is False or new is True or new == {}:
        _change(changes, tool, path, "compatible", "Input constraint relaxed.")
        return
    if new is False:
        _change(changes, tool, path, "breaking", "Previously allowed input is now forbidden.")
        return
    old = {} if old is True else old
    # Malformed or unsupported schemas are handled before entering this walker.
    old_types, new_types = _types(old), _types(new)
    if old_types != new_types:
        removed = old_types - new_types
        _change(changes, tool, _pointer(path, "type"), "breaking" if removed else "compatible",
                f"Accepted input types {'narrowed' if removed else 'widened'}: "
                f"{sorted(old_types)!r} -> {sorted(new_types)!r}.")

    old_enum, new_enum = old.get("enum"), new.get("enum")
    old_values = None if old_enum is None else {_value_key(v) for v in old_enum}
    new_values = None if new_enum is None else {_value_key(v) for v in new_enum}
    if old_values != new_values:
        narrowed = new_values is not None and (old_values is None or not old_values <= new_values)
        _change(changes, tool, _pointer(path, "enum"), "breaking" if narrowed else "compatible",
                "Allowed enum values narrowed." if narrowed else "Allowed enum values widened.")

    # Object keywords have no effect on other types. Keep their structural edits visible,
    # but do not label them as breaking input rules when objects were already excluded.
    if "object" in old_types & new_types:
        _compare_object(old, new, tool, path, changes)
    elif any(old.get(k) != new.get(k) for k in ("properties", "required", "additionalProperties")):
        _change(changes, tool, path, "compatible", "Object rules changed outside the shared input types.")


def _compare_object(old: dict, new: dict, tool: str, path: str, changes: list) -> None:
    old_required, new_required = set(old.get("required", [])), set(new.get("required", []))
    for name in sorted(new_required - old_required):
        _change(changes, tool, _pointer(path, "required"), "breaking", f"New required property: {name!r}.")
    for name in sorted(old_required - new_required):
        _change(changes, tool, _pointer(path, "required"), "compatible", f"Property is no longer required: {name!r}.")

    old_props, new_props = old.get("properties", {}), new.get("properties", {})
    old_extra, new_extra = old.get("additionalProperties", True), new.get("additionalProperties", True)
    for name in sorted(old_props.keys() | new_props.keys()):
        child_path = _pointer(_pointer(path, "properties"), name)
        # Adding an optional property to an OPEN object can narrow its formerly
        # unconstrained values. Removing a property uses the NEW extra-property rule.
        before = old_props.get(name, old_extra)
        after = new_props.get(name, new_extra)
        start = len(changes)
        _compare_schema(before, after, tool, child_path, changes)
        if len(changes) == start and (name in old_props) != (name in new_props):
            _change(changes, tool, child_path, "compatible", "Property declaration changed without narrowing its values.")
    _compare_schema(old_extra, new_extra, tool, _pointer(path, "additionalProperties"), changes)


def validate_manifest(data: Any) -> dict:
    """Reject ambiguous or incomplete snapshot envelopes before producing a verdict."""
    if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Expected an MCP snapshot with version 1.")
    if not isinstance(data.get("server"), str) or not data["server"].strip():
        raise ValueError("Snapshot must identify its server.")
    if not isinstance(data.get("tools"), list):
        raise ValueError("Snapshot tools must be a list.")
    names = set()
    for tool in data["tools"]:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str) or not tool["name"]:
            raise ValueError("Every tool must have a nonempty name.")
        if tool["name"] in names:
            raise ValueError(f"Duplicate tool name: {tool['name']!r}.")
        names.add(tool["name"])
        if not isinstance(tool.get("inputSchema"), dict):
            raise ValueError(f"Tool {tool['name']!r} must have an inputSchema object.")
    return data


def compare_manifests(baseline: dict, current: dict) -> dict:
    """Compare old callers against new tool definitions; never contact either server."""
    validate_manifest(baseline)
    validate_manifest(current)
    if baseline["server"] != current["server"]:
        raise ValueError("Snapshots identify different servers; compare versions of the same server.")
    old_tools = {t["name"]: t for t in baseline["tools"]}
    new_tools = {t["name"]: t for t in current["tools"]}
    changes: list[dict] = []
    for name in sorted(old_tools.keys() - new_tools.keys()):
        _change(changes, name, "", "breaking", "Tool removed.")
    for name in sorted(new_tools.keys() - old_tools.keys()):
        _change(changes, name, "", "compatible", "Tool added.")
    for name in sorted(old_tools.keys() & new_tools.keys()):
        old, new = old_tools[name], new_tools[name]
        before, after = old["inputSchema"], new["inputSchema"]
        if _value_key(before) != _value_key(after):
            unsupported = sorted(set(_unsupported(before, "/inputSchema") + _unsupported(after, "/inputSchema")))
            if unsupported:
                for path in unsupported:
                    _change(changes, name, path, "review", "Changed input schema uses an unsupported or malformed rule.")
            else:
                _compare_schema(before, after, name, "/inputSchema", changes)
        for key in sorted((old.keys() | new.keys()) - {"name", "inputSchema"}):
            if _value_key(old.get(key)) != _value_key(new.get(key)) or (key in old) != (key in new):
                _change(changes, name, _pointer("", key), "review", "Tool metadata or output contract changed.")
    severities = {c["severity"] for c in changes}
    status = next((s for s in ("breaking", "review", "compatible") if s in severities), "unchanged")
    return {"version": 1, "server": baseline["server"], "status": status, "changes": changes}
