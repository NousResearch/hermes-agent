"""Input compatibility is directional: old callers must still fit the new schema."""

from copy import deepcopy

import pytest

from hermes_cli.mcp_schema_diff import compare_manifests


def manifest(schema=None, **metadata):
    return {"version": 1, "server": "demo", "tools": [
        {"name": "search", "inputSchema": schema if schema is not None else {"type": "object"}, **metadata}
    ]}


def object_schema(prop, *, extra=True):
    return {"type": "object", "properties": {"query": prop}, "additionalProperties": extra}


@pytest.mark.parametrize(("before", "after", "status", "path"), [
    ({"type": "integer"}, {"type": "number"}, "compatible", "/type"),
    ({"type": "number"}, {"type": "integer"}, "breaking", "/type"),
    ({"type": ["string", "null"]}, {"type": "string"}, "breaking", "/type"),
    ({"type": "string"}, {"type": ["null", "string"]}, "compatible", "/type"),
    ({"type": ["string", "null"]}, {"type": ["null", "string"]}, "unchanged", None),
    ({"enum": [True, 1]}, {"enum": [1]}, "breaking", "/enum"),
    ({"enum": [1]}, {"enum": [1.0]}, "unchanged", None),
    ({"enum": ["a", "b"]}, {"enum": ["b", "a"]}, "unchanged", None),
    ({"enum": [{"x": True}]}, {"enum": [{"x": 1}]}, "breaking", "/enum"),
    ({"enum": ["a"]}, {"enum": ["a", "b"]}, "compatible", "/enum"),
    ({}, {"enum": ["a"]}, "breaking", "/enum"),
    ({"enum": ["a"]}, {}, "compatible", ""),
    ({"type": "object"}, {"type": "object", "required": ["query"]}, "breaking", "/required"),
    ({"type": "object", "required": ["query"]}, {"type": "object"}, "compatible", "/required"),
    (object_schema({"type": "string"}), {"type": "object"}, "compatible", "/properties/query"),
    (object_schema({"type": "string"}, extra=False), {"type": "object", "additionalProperties": False},
     "breaking", "/properties/query"),
    ({"type": "object", "additionalProperties": False}, object_schema({"type": "string"}, extra=False),
     "compatible", "/properties/query"),
    ({"type": "object"}, object_schema({"type": "string"}), "breaking", "/properties/query/type"),
    ({"type": "object"}, {"type": "object", "additionalProperties": False}, "breaking", "/additionalProperties"),
    (object_schema({"type": "number"}), object_schema({"type": "integer"}), "breaking", "/properties/query/type"),
    (object_schema(True), object_schema(False), "breaking", "/properties/query"),
    (object_schema(False), object_schema({"type": "string"}), "compatible", "/properties/query"),
    ({"type": "object", "additionalProperties": {"type": "number"}},
     {"type": "object", "additionalProperties": {"type": "integer"}}, "breaking", "/additionalProperties/type"),
    ({"type": "object", "properties": {"a/b~c": {"type": "number"}}},
     {"type": "object", "properties": {"a/b~c": {"type": "integer"}}}, "breaking", "/properties/a~1b~0c/type"),
    ({"type": "string", "description": "old"}, {"type": "string", "description": "new"}, "review", "/description"),
    ({"type": "string"}, {"type": "string", "default": None}, "review", "/default"),
    ({"default": "old"}, {}, "review", "/default"),
    ({"type": "string"}, {"type": "string", "minLength": 5}, "review", "/minLength"),
    ({"type": "array", "items": {"type": "string"}},
     {"type": "array", "items": {"type": "integer"}}, "review", "/items"),
    ({"$ref": "https://example.invalid/schema"}, {"$ref": "https://example.invalid/new"}, "review", "/$ref"),
    ({"type": "object", "patternProperties": {"x": {"type": "string"}}},
     {"type": "object", "patternProperties": {"x": {"type": "string"}}, "additionalProperties": False},
     "review", "/patternProperties"),
    ({"type": "object"}, {"type": "object", "allOf": []}, "review", "/allOf"),
    ({"type": "object"}, {"type": "object", "required": True}, "review", "/required"),
    ({"type": "object"}, {"type": [{"invalid": True}]}, "review", "/type"),
    ({"type": "object"}, {"type": "object", "properties": []}, "review", "/properties"),
    ({"type": "object"}, {"type": "object", "enum": []}, "review", "/enum"),
    ({"type": "object"}, {"type": "object", "$schema": "http://json-schema.org/draft-04/schema#"}, "review", "/$schema"),
])
def test_input_contract_changes(before, after, status, path):
    old, new = manifest(before), manifest(after)
    saved_old, saved_new = deepcopy(old), deepcopy(new)
    report = compare_manifests(old, new)
    assert report["status"] == status
    if path is not None:
        assert any(c["path"] == "/inputSchema" + path for c in report["changes"])
    assert old == saved_old and new == saved_new


@pytest.mark.parametrize(("updates", "status"), [
    ({"tools": manifest()["tools"] + [{"name": "count", "inputSchema": {"type": "object"}}]}, "compatible"),
    ({"tools": []}, "breaking"),
    ({"tools": manifest(annotations={"readOnlyHint": False})["tools"]}, "review"),
    ({"tools": manifest(outputSchema={"type": "object"})["tools"]}, "review"),
    ({"server": "another"}, None),
    ({"tools": manifest()["tools"] * 2}, None),
    ({"tools": [{"name": "search"}]}, None),
    ({"version": True}, None),
])
def test_manifest_boundaries(updates, status):
    before, after = manifest(), {**manifest(), **updates}
    if status is None:
        with pytest.raises(ValueError):
            compare_manifests(before, after)
    else:
        assert compare_manifests(before, after)["status"] == status
