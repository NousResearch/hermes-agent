"""Portable v1 recipes: strict data boundaries, no board or local resolution."""
import copy
import hashlib
import importlib
import json


import pytest


@pytest.fixture
def api():
    class API:
        def __getattr__(self, name):
            try:
                module = importlib.import_module("hermes_cli.kanban_recipes")
            except ModuleNotFoundError as exc:
                if exc.name == "hermes_cli.kanban_recipes":
                    pytest.fail("The pure recipe parser is not implemented")
                raise
            return getattr(module, name)
    return API()


def recipe(**updates):
    value = {"schema_version": 1, "recipe_id": "brief", "roles": ["writer"],
             "nodes": [{"key": "draft", "role": "writer", "title": "Draft"}]}
    value.update(updates)
    return value


def invalid(api, definition, inputs=None, path=None):
    with pytest.raises(api.RecipeError) as caught:
        api.prepare_definition(definition, {} if inputs is None else inputs)
    error = caught.value
    assert error.code == "RECIPE_INVALID"
    assert error.retryable is False
    assert isinstance(error.path, str)
    if path is not None:
        assert error.path == path
    return error


def test_error_contract(api):
    error = api.RecipeError("RECIPE_INVALID", "/nodes/0/title", "Invalid template")
    assert error.as_dict() == {"error": {"code": error.code, "path": error.path,
                                        "message": error.message, "retryable": False}}
    assert str(error) == error.message
    assert api.RecipeError("STORAGE_UNAVAILABLE", "", "Unavailable", True).retryable


def test_original_digest_and_stable_topology(api):
    definition = recipe(inputs={"topic": {"type": "string", "default": "café"}}, nodes=[
        {"key": "join", "role": "writer", "title": "{{input.topic}}", "needs": ["b", "a"]},
        {"key": "a", "role": "writer", "title": "A"},
        {"key": "b", "role": "writer", "title": "B"},
        {"key": "last", "role": "writer", "title": "Last"},
    ])
    original = copy.deepcopy(definition)
    result = api.prepare_definition(definition, {})
    assert result["definition"] == original == definition
    encoded = json.dumps(original, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert api.canonical(original) == encoded
    assert result["definition_digest"] == hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    assert result["inputs"] == {"topic": "café"}
    assert [n["key"] for n in result["nodes"]] == ["join", "a", "b", "last"]
    assert result["order"] == ["a", "b", "join", "last"]
    assert result["nodes"][0]["needs"] == ["b", "a"]
    assert result["nodes"][0]["title"] == "café"
    assert result["nodes"][1]["body"] == ""
    assert result["nodes"][1]["task"]["priority"] == 0
    assert result["nodes"][1]["task"]["goal_mode"] is False
    reordered = copy.deepcopy(original)
    reordered["nodes"][0]["needs"].reverse()
    assert api.prepare_definition(reordered, {})["definition_digest"] != result["definition_digest"]


@pytest.mark.parametrize("slot", ["recipe_id", "input", "role", "key", "role_ref", "needs_ref", "invocation"])
@pytest.mark.parametrize("name", ["", "a" * 65, "A", "a b", " a", "a ", "a.b", "a/b", "é", "a\n", "0a"])
def test_identifiers_are_exact_ascii(api, slot, name):
    definition = recipe()
    inputs = {}
    if slot == "recipe_id":
        definition["recipe_id"] = name
    elif slot == "input":
        definition["inputs"] = {name: {"type": "string"}}
    elif slot == "role":
        definition["roles"] = [name]
    elif slot == "key":
        definition["nodes"][0]["key"] = name
    elif slot == "role_ref":
        definition["nodes"][0]["role"] = name
    elif slot == "needs_ref":
        definition["nodes"][0]["needs"] = [name]
    else:
        inputs = {name: "value"}
    invalid(api, definition, inputs)


@pytest.mark.parametrize("name", ["a", "a" * 64, "a-0_b"])
def test_identifier_endpoints_and_separate_namespaces(api, name):
    value = recipe(recipe_id=name, roles=[name], inputs={name: {"type": "string"}}, nodes=[
        {"key": name, "role": name, "title": "{{input." + name + "}}"}])
    assert api.prepare_definition(value, {name: "yes"})["nodes"][0]["title"] == "yes"


@pytest.mark.parametrize("updates", [
    {"schema_version": True}, {"schema_version": 1.0}, {"schema_version": "1"}, {"schema_version": 2},
    {"roles": []}, {"roles": ["writer", "writer"]}, {"roles": "writer"},
    {"nodes": []}, {"nodes": {}}, {"inputs": None}, {"description": None}, {"description": 1},
    {"gates": []}, {"outputs": {}}, {"cycles": []}, {"joins": {}}, {"include": "elsewhere.json"},
])
def test_definition_shapes_and_unknown_fields(api, updates):
    invalid(api, recipe(**updates))


@pytest.mark.parametrize("field", ["schema_version", "recipe_id", "roles", "nodes"])
def test_missing_definition_fields(api, field):
    value = recipe()
    del value[field]
    invalid(api, value, path="/" + field)


@pytest.mark.parametrize("field", ["key", "role", "title"])
def test_missing_node_fields(api, field):
    value = recipe()
    del value["nodes"][0][field]
    invalid(api, value, path="/nodes/0/" + field)


@pytest.mark.parametrize("patch", [
    {"title": None}, {"body": 1}, {"title": " \n\t"}, {"needs": "draft"}, {"task": []},
    {"role": "unknown"}, {"needs": ["unknown"]}, {"needs": ["draft"]},
    {"status": "ready"}, {"outputs": {}}, {"consumes": {}}, {"reviewer": "salt"},
])
def test_invalid_node_controls(api, patch):
    value = recipe()
    value["nodes"][0].update(patch)
    invalid(api, value)


@pytest.mark.parametrize("kind", ["duplicate-node", "duplicate-edge", "cycle"])
def test_graph_rejections(api, kind):
    nodes = [{"key": "a", "role": "writer", "title": "A"},
             {"key": "b", "role": "writer", "title": "B", "needs": ["a"]}]
    if kind == "duplicate-node":
        nodes[1]["key"] = "a"
    elif kind == "duplicate-edge":
        nodes[1]["needs"] = ["a", "a"]
    else:
        nodes[0]["needs"] = ["b"]
    invalid(api, recipe(nodes=nodes))


@pytest.mark.parametrize("kind,good,bad", [
    ("integer", [-9007199254740991, 0, 9007199254740991], [True, False, 1.0, "1", None, -9007199254740992, 9007199254740992]),
    ("boolean", [True, False], [0, 1, "true", None]),
    ("string", ["", "é"], [1, True, None, []]),
    ("object", [{}, {"x": [None, True, 1.5, 1e20]}], [[], "{}", None]),
    ("array", [[], [None, {"x": 1.5}]], [{}, "[]", None]),
])
def test_input_types_apply_equally_to_defaults_and_invocations(api, kind, good, bad):
    for value in good:
        definition = recipe(inputs={"data": {"type": kind, "default": value}})
        assert api.prepare_definition(definition, {})["inputs"] == {"data": value}
        assert api.prepare_definition(definition, {"data": value})["inputs"] == {"data": value}
    for value in bad:
        invalid(api, recipe(inputs={"data": {"type": kind, "default": value}}))
        invalid(api, recipe(inputs={"data": {"type": kind}}), {"data": value})


@pytest.mark.parametrize("declaration", [None, [], {}, {"type": "null"}, {"type": 1},
    {"type": "string", "required": 1}, {"type": "string", "required": None},
    {"type": "string", "required": True, "default": "x"}, {"type": "string", "other": True}])
def test_input_declarations_are_closed(api, declaration):
    invalid(api, recipe(inputs={"data": declaration}))


def test_required_optional_unknown_and_absent_inputs(api):
    value = recipe(inputs={"required": {"type": "string", "required": True},
                           "optional": {"type": "string"}})
    invalid(api, value)
    invalid(api, value, {"required": "x", "unknown": "x"})
    assert api.prepare_definition(value, {"required": "x"})["inputs"] == {"required": "x"}
    value["nodes"][0]["body"] = "{{input.optional}}"
    invalid(api, value, {"required": "x"}, path="/nodes/0/body")


@pytest.mark.parametrize("text", ["{{", "{{input.data}", "{{input.data", "{{input.missing}}",
    "{{input.Data}}", "{{ input.data }}", "{{input.data.x}}", "{{input.data[0]}}",
    "{{input.data|upper}}", "{{env.HOME}}", "{{{input.data}}", "{{input.{{input.data}}}}"])
def test_only_exact_template_tokens_are_allowed(api, text):
    value = recipe(inputs={"data": {"type": "string", "default": "x"}})
    value["nodes"][0]["title"] = text
    invalid(api, value, path="/nodes/0/title")


def test_literal_escape_and_no_second_pass(api):
    value = recipe(inputs={"data": {"type": "string"}, "flag": {"type": "boolean", "default": False},
                           "count": {"type": "integer", "default": -3}})
    value["nodes"][0].update(title="{{{{input.data}} {{input.data}}",
                            body="{{input.flag}}/{{input.count}} $HOME $(touch nope)")
    result = api.prepare_definition(value, {"data": "{{input.unknown}}"})
    assert result["nodes"][0]["title"] == "{{input.data}} {{input.unknown}}"
    assert result["nodes"][0]["body"] == "false/-3 $HOME $(touch nope)"
    value["nodes"][0]["task"] = {"model": "{{input.data}}"}
    assert api.prepare_definition(value, {"data": "x"})["nodes"][0]["task"]["model_override"] == "{{input.data}}"


@pytest.mark.parametrize("kind,data", [("object", {}), ("array", [])])
def test_structured_inputs_never_interpolate(api, kind, data):
    value = recipe(inputs={"data": {"type": kind}})
    assert api.prepare_definition(value, {"data": data})["inputs"] == {"data": data}
    value["nodes"][0]["title"] = "{{input.data}}"
    invalid(api, value, {"data": data})


@pytest.mark.parametrize("control,low,high", [("priority", -2147483648, 2147483647),
    ("max_runtime_seconds", 1, 604800), ("max_retries", 1, 100), ("goal_max_turns", 1, 1000)])
def test_numeric_control_endpoints_and_strict_types(api, control, low, high):
    value = recipe()
    task = value["nodes"][0]["task"] = {"goal_mode": True}
    for accepted in [low, high]:
        task[control] = accepted
        assert api.prepare_definition(value, {})["nodes"][0]["task"][control] == accepted
    for rejected in [low - 1, high + 1, True, False, 1.0, "1", None]:
        task[control] = rejected
        invalid(api, value, path="/nodes/0/task/" + control)


@pytest.mark.parametrize("mode", [None, 0, 1, "true", "false", 1.0])
def test_goal_mode_is_boolean_only(api, mode):
    value = recipe()
    value["nodes"][0]["task"] = {"goal_mode": mode}
    invalid(api, value, path="/nodes/0/task/goal_mode")


def test_goal_cross_fields_and_deferred_native_default(api):
    value = recipe()
    for task in [{"goal_max_turns": 1}, {"goal_mode": False, "goal_max_turns": 1}]:
        value["nodes"][0]["task"] = task
        invalid(api, value)
    value["nodes"][0]["task"] = {"goal_mode": True}
    result = api.prepare_definition(value, {})["nodes"][0]["task"]
    assert "goal_max_turns" not in result  # Parent resolver owns native default.
    value["nodes"][0]["task"] = {"goal_mode": True, "goal_max_turns": 1000,
                                    "max_runtime_seconds": 1, "max_retries": 1}
    assert api.prepare_definition(value, {})["nodes"][0]["task"]["max_runtime_seconds"] == 1


@pytest.mark.parametrize("control", ["workspace_kind", "priority", "skills", "max_runtime_seconds", "max_retries",
    "model", "provider", "reasoning_effort", "goal_mode", "goal_max_turns", "completion_contract"])
def test_explicit_null_controls_rejected(api, control):
    value = recipe()
    value["nodes"][0]["task"] = {control: None}
    invalid(api, value, path="/nodes/0/task/" + control)


@pytest.mark.parametrize("task", [{"workspace_kind": "dir"}, {"workspace_kind": []}, {"workspace_path": "/tmp"},
    {"branch_name": "main"}, {"project_id": "local"}, {"status": "ready"}, {"triage": True},
    {"initial_status": "blocked"}, {"scheduled_at": 1}, {"reviewer": "salt"}, {"model": []},
    {"provider": 1}, {"provider": "test"}, {"provider": "test", "model": " "},
    {"reasoning_effort": True}, {"reasoning_effort": "nonsense"}, {"skills": "a"}, {"skills": [1]},
    {"skills": ["a,b"]}, {"skills": ["web"]}, {"completion_contract": {}}, {"completion_contract": "not a contract"}])
def test_task_shapes_allowlist_and_native_validation(api, task):
    value = recipe()
    value["nodes"][0]["task"] = task
    error = invalid(api, value)
    assert "not a contract" not in error.message
    assert "nonsense" not in error.message


def test_native_task_normalization_and_no_local_reads(api, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Pure preparation attempted local configuration or database access")
    from hermes_cli import kanban_db_connect, config
    monkeypatch.setattr(kanban_db_connect, "connect", forbidden)
    monkeypatch.setattr(config, "load_config", forbidden)
    value = recipe()
    value["nodes"][0]["task"] = {"workspace_kind": "worktree", "model": " vendor/model ",
        "provider": " provider ", "reasoning_effort": " HIGH ", "skills": [" custom-skill ", "custom-skill", ""],
        "completion_contract": "org/repo"}
    task = api.prepare_definition(value, {})["nodes"][0]["task"]
    assert task["workspace_kind"] == "worktree"
    assert task["model_override"] == "vendor/model"
    assert task["provider_override"] == "provider"
    assert task["reasoning_effort"] == "high"
    assert task["skills"] == ["custom-skill"]
    assert task["completion_contract"] == "org/repo"


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"x":{"a":1,"\\u0061":2}}',
    b'{"a":NaN}', b'[Infinity]', b'[-Infinity]', b'[1e999]', b'[9007199254740992]',
    b'"\\ud800"', b'{"\\udfff":1}', b'"\xff"', b'{} {}', b'//comment\n{}', b'\xef\xbb\xbf{}', b''])
def test_strict_json_read_preserves_files(api, tmp_path, raw):
    path = tmp_path / "input.json"
    path.write_bytes(raw)
    before = path.stat()
    entries = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(api.RecipeError) as caught:
        api.load_json(path)
    assert caught.value.code == "RECIPE_INVALID"
    assert path.read_bytes() == raw
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert sorted(p.name for p in tmp_path.iterdir()) == entries


def test_duplicate_key_error_has_escaped_json_pointer(api, tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"outer":{"a/b~c":1,"a/b~c":2}}')
    with pytest.raises(api.RecipeError) as caught:
        api.load_json(path)
    assert caught.value.path == "/outer/a~1b~0c"


@pytest.mark.parametrize("raw", ["1.0", "1e0", "true", '"1"', "null"])
def test_wire_integer_tokens_do_not_coerce(api, tmp_path, raw):
    path = tmp_path / "input.json"
    path.write_text('{"data":' + raw + '}')
    invalid(api, recipe(inputs={"data": {"type": "integer"}}), api.load_json(path))


def test_json_negative_zero_and_finite_nested_values(api, tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"data":-0,"nested":[1e0,1.5,null,"é","\\ud83d\\ude00"]}')
    result = api.load_json(path)
    assert type(result["data"]) is int and result["data"] == 0
    assert type(result["nested"][0]) is float
    assert result["nested"][-1] == "😀"
    assert api.canonical({"b": "é", "a": 0}) == '{"a":0,"b":"é"}'


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 9007199254740992,
    "\ud800", {"\udfff": 0}, (1,), {1}, {1: "not a string key"}, b"bytes", object()])
def test_direct_python_data_is_validated_globally(api, bad):
    invalid(api, recipe(inputs={"data": {"type": "object"}}), {"data": {"nested": bad}})
    invalid(api, recipe(inputs={"data": {"type": "object", "default": {"nested": bad}}}))
    with pytest.raises(api.RecipeError):
        api.canonical({"nested": bad})


@pytest.mark.parametrize("value", [None, [], (), "recipe", 1])
def test_direct_top_level_must_be_object(api, value):
    invalid(api, value)
    with pytest.raises(api.RecipeError):
        api.prepare_definition(recipe(), value)


def test_json_depth_boundary_and_cycles(api, tmp_path):
    path = tmp_path / "input.json"
    path.write_text("[" * 16 + "0" + "]" * 16)
    assert api.load_json(path)
    path.write_text("[" * 17 + "0" + "]" * 17)
    with pytest.raises(api.RecipeError):
        api.load_json(path)
    path.write_text("[" * 2000 + "0" + "]" * 2000)
    with pytest.raises(api.RecipeError):
        api.load_json(path)
    data = []
    data.append(data)
    invalid(api, recipe(inputs={"data": {"type": "array"}}), {"data": data})
    nested = 0
    for _ in range(16):
        nested = [nested]
    invalid(api, recipe(inputs={"data": {"type": "array"}}), {"data": nested})


def test_definition_and_read_byte_caps(api, tmp_path):
    path = tmp_path / "input.json"
    path.write_bytes(b'{}' + b' ' * (1048576 - 2))
    assert api.load_json(path) == {}
    with path.open("ab") as handle:
        handle.write(b' ')
    with pytest.raises(api.RecipeError):
        api.load_json(path)
    path.write_bytes(b'"xx"')
    assert api.load_json(path, max_bytes=4) == "xx"
    with pytest.raises(api.RecipeError):
        api.load_json(path, max_bytes=3)
    with pytest.raises(api.RecipeError):
        api.load_json(tmp_path / "missing.json")
    invalid(api, recipe(inputs={"data": {"type": "string", "default": "x" * 1048576}}))


@pytest.mark.parametrize("field,cap", [("description", 4096), ("title", 1024), ("body", 65536)])
def test_utf8_text_caps_and_rendered_title(api, field, cap):
    value = recipe()
    target = value if field == "description" else value["nodes"][0]
    target[field] = "é" * (cap // 2)
    api.prepare_definition(value, {})
    target[field] += "x"
    invalid(api, value)
    if field != "description":
        value["inputs"] = {"data": {"type": "string"}}
        target[field] = "{{input.data}}" * 2
        invalid(api, value, {"data": "x" * (cap // 2 + 1)})
    value = recipe(inputs={"data": {"type": "string", "default": " \t"}})
    value["nodes"][0]["title"] = "{{input.data}}"
    invalid(api, value)


def test_input_canonical_byte_cap(api):
    value = recipe(inputs={"data": {"type": "string"}})
    overhead = len(json.dumps({"data": ""}, separators=(",", ":")).encode())
    api.prepare_definition(value, {"data": "x" * (65536 - overhead)})
    invalid(api, value, {"data": "x" * (65537 - overhead)})


def test_graph_count_limits(api):
    nodes = [{"key": f"n{i}", "role": "writer", "title": "N"} for i in range(256)]
    assert len(api.prepare_definition(recipe(nodes=nodes), {})["nodes"]) == 256
    invalid(api, recipe(nodes=nodes + [{"key": "overflow", "role": "writer", "title": "N"}]))
    for count in [64, 65]:
        roles = [f"r{i}" for i in range(count)]
        value = recipe(roles=roles)
        value["nodes"][0]["role"] = roles[0]
        inputs = {f"i{i}": {"type": "string"} for i in range(count)}
        if count == 64:
            api.prepare_definition(value, {})
            api.prepare_definition(recipe(inputs=inputs), {})
        else:
            invalid(api, value)
            invalid(api, recipe(inputs=inputs))
    parents = [f"n{i}" for i in range(32)]
    for i in range(32, 96):
        nodes[i]["needs"] = parents[:]
    api.prepare_definition(recipe(nodes=nodes), {})
    nodes[96]["needs"] = ["n0"]
    invalid(api, recipe(nodes=nodes))


def test_combined_rendered_plan_cap(api):
    value = recipe(inputs={"data": {"type": "string"}}, nodes=[
        {"key": f"n{i}", "role": "writer", "title": "N", "body": "{{input.data}}"}
        for i in range(40)])
    api.prepare_definition(value, {"data": "x" * 1000})
    invalid(api, value, {"data": "x" * 60000})
