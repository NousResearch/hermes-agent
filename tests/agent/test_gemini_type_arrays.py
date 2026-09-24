"""Native tool translation must preserve union alternatives at every depth."""
import copy

import pytest

from agent.gemini_native_adapter import _translate_tools_to_gemini


@pytest.mark.parametrize("types", [["string", "null"], ["integer", "null"], ["string", "integer"], ["string", "integer", "boolean", "null"], ["array", "object", "string"], ["null"], [], "string"])
@pytest.mark.parametrize("location", ["properties", "items", "anyOf"])
@pytest.mark.parametrize("nullable_first", [False, True])
@pytest.mark.parametrize("constrained", [False, True])
def test_native_type_arrays_preserve_alternatives(types, location, nullable_first, constrained):
    node = {"nullable": False, "type": types} if nullable_first else {"type": types, "nullable": False}
    node["description"] = "Keep this guidance"
    if "array" in types:
        node.update(items={"type": "integer"}, minItems=1)
    if "object" in types:
        node.update(properties={"name": {"type": "string"}}, required=["name", "undefined"])
    if constrained:
        node["anyOf"] = [{"enum": ["one"]}, {"enum": ["two"]}]
    wrapper = {"properties": {"value": node}} if location == "properties" else {location: node if location == "items" else [node]}
    params = {"type": "object", "properties": {"nested": wrapper}}
    original = copy.deepcopy(params)
    tools = [{"type": "function", "function": {"name": "probe", "parameters": params}}]
    out = _translate_tools_to_gemini(tools)[0]["functionDeclarations"][0]["parameters"]["properties"]["nested"][location]
    out = out["value"] if location == "properties" else out[0] if location == "anyOf" else out
    expected = {t for t in types if t != "null"} if isinstance(types, list) else {types}
    branches = out["anyOf"] if "type" not in out else [out]
    actual = {branch["type"] for branch in branches}
    for branch in branches:
        if branch["type"] == "array":
            assert branch["items"] == node["items"]
            assert branch["minItems"] == node["minItems"]
        if branch["type"] == "object" and "object" in types:
            assert branch["properties"] == node["properties"]
            assert branch["required"] == ["name"]
        if "array" in types and branch["type"] != "array":
            assert "items" not in branch
    if constrained:
        assert all(branch["anyOf"] == node["anyOf"] for branch in branches)
    assert actual == (expected or {"null" if "null" in types else "object"})
    assert not isinstance(out.get("type"), list)
    if isinstance(types, list) and "null" in types and expected:
        assert out["nullable"] is True
    assert out["description"] == node["description"]
    assert params == original


@pytest.mark.parametrize("types,enum", [(["integer", "null"], [1, 2]), (["string", "integer"], ["one", 2]), (["boolean", "string"], [True, "other"])])
def test_native_type_array_enums_remain_wire_strings(types, enum):
    params = {"type": "object", "properties": {"value": {"type": types, "enum": enum}}}
    tools = [{"type": "function", "function": {"name": "probe", "parameters": params}}]
    node = _translate_tools_to_gemini(tools)[0]["functionDeclarations"][0]["parameters"]["properties"]["value"]
    assert node["enum"] == [str(v).lower() if isinstance(v, bool) else str(v) for v in enum]
