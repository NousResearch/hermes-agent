"""Strict, inert v1 recipe preparation; local bindings belong to the resolver.

No board, profile, project or configuration lookup happens here. JSON container
nesting counts the outermost object/array as depth one (scalars add no depth).
"""
from __future__ import annotations

import copy
import hashlib
import heapq
import json
import math
import re
from pathlib import Path
from typing import NoReturn

MAX_DEFINITION_BYTES = 1048576
MAX_INPUT_BYTES = 65536
MAX_PLAN_BYTES = 2097152
MAX_DEPTH = 16
MAX_SAFE_INTEGER = 9007199254740991
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{0,63}", re.ASCII)
_TOKEN = re.compile(r"\{\{input\.([a-z][a-z0-9_-]{0,63})\}\}", re.ASCII)
_INPUT_TYPES = {"string": str, "integer": int, "boolean": bool, "object": dict, "array": list}
_NUMERIC_CONTROLS = {"priority": (-2147483648, 2147483647),
                     "max_runtime_seconds": (1, 604800), "max_retries": (1, 100),
                     "goal_max_turns": (1, 1000)}
_STRING_CONTROLS = {"workspace_kind", "model", "provider", "reasoning_effort", "completion_contract"}
_TASK_FIELDS = _STRING_CONTROLS | _NUMERIC_CONTROLS.keys() | {"skills", "goal_mode"}


class RecipeError(ValueError):
    """A public, value-free diagnostic with a JSON-pointer location."""

    def __init__(self, code, path, message, retryable=False):
        self.code = code
        self.path = path
        self.message = message
        self.retryable = retryable
        super().__init__(message)

    def as_dict(self):
        return {"error": {"code": self.code, "path": self.path,
                          "message": self.message, "retryable": self.retryable}}


def _fail(path, message) -> NoReturn:
    raise RecipeError("RECIPE_INVALID", path, message)


def _pointer(path, key):
    return path + "/" + str(key).replace("~", "~0").replace("/", "~1")


def _string(value, path):
    if type(value) is not str:
        _fail(path, "Expected a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _fail(path, "Lone Unicode surrogates are not allowed")


def _integer(value, path, low=-MAX_SAFE_INTEGER, high=MAX_SAFE_INTEGER):
    if type(value) is not int or not low <= value <= high:
        _fail(path, "Expected an integer token in the allowed range")


def _json_safe(value, path="", depth=0):
    """Also protects direct Python callers from coercions and recursive objects."""
    kind = type(value)
    if kind in (dict, list):
        if depth >= MAX_DEPTH:
            _fail(path, "JSON nesting exceeds 16 containers")
        if kind is dict:
            for key, item in value.items():
                _string(key, path)
                _json_safe(item, _pointer(path, key), depth + 1)
        else:
            for index, item in enumerate(value):
                _json_safe(item, _pointer(path, index), depth + 1)
        return
    if kind is str:
        _string(value, path)
        return
    if kind is int:
        _integer(value, path)
        return
    if kind is float:
        if not math.isfinite(value):
            _fail(path, "Numbers must be finite binary64 values")
        return
    if value is not None and kind is not bool:
        _fail(path, "Expected a JSON value")


def canonical(value) -> str:
    """Canonical UTF-8-safe JSON, with no implicit coercion of Python objects."""
    _json_safe(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


class _ObjectPairs(list):
    """Keep duplicate keys until their full JSON-pointer path is available."""


def _decode_pairs(value, path="", depth=0):
    if isinstance(value, (_ObjectPairs, list)):
        if depth >= MAX_DEPTH:
            _fail(path, "JSON nesting exceeds 16 containers")
        if isinstance(value, _ObjectPairs):
            result = {}
            for key, item in value:
                _string(key, path)
                child_path = _pointer(path, key)
                if key in result:
                    _fail(child_path, "Duplicate JSON object key")
                result[key] = _decode_pairs(item, child_path, depth + 1)
            return result
        return [_decode_pairs(item, _pointer(path, index), depth + 1)
                for index, item in enumerate(value)]
    _json_safe(value, path, depth)
    return value


def load_json(path, max_bytes=MAX_DEFINITION_BYTES):
    """Read at most the byte budget plus one; reject nonstandard JSON safely."""
    if type(max_bytes) is not int or max_bytes < 1:
        _fail("", "Read byte limit must be a positive integer")
    try:
        with Path(path).open("rb") as source:
            raw = source.read(max_bytes + 1)
    except (OSError, ValueError, TypeError):
        _fail("", "Cannot read JSON file")
    if len(raw) > max_bytes:
        _fail("", "JSON file exceeds byte limit")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_ObjectPairs,
                           parse_constant=lambda _: _fail("", "Non-finite JSON constant"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, RecipeError):
            raise
        _fail("", "Invalid UTF-8 JSON document")
    return _decode_pairs(value)


def _object(value, path, allowed=None, required=()):
    if type(value) is not dict:
        _fail(path, "Expected an object")
    if allowed is not None:
        for key in value:
            if key not in allowed:
                _fail(_pointer(path, key), "Unknown field")
    for key in required:
        if key not in value:
            _fail(_pointer(path, key), "Required field is missing")


def _array(value, path, maximum, minimum=0):
    if type(value) is not list:
        _fail(path, "Expected an array")
    if not minimum <= len(value) <= maximum:
        _fail(path, "Array length is outside the allowed range")


def _identifier(value, path):
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        _fail(path, "Expected a 1-64 character lowercase ASCII identifier")


def _bounded_text(value, path, limit):
    _string(value, path)
    if len(value.encode("utf-8")) > limit:
        _fail(path, "Text exceeds UTF-8 byte limit")


def _normalize_inputs(declarations, supplied):
    _object(declarations, "/inputs")
    _object(supplied, "/inputs")
    if len(declarations) > 64:
        _fail("/inputs", "At most 64 inputs are allowed")
    for key in supplied:
        _identifier(key, _pointer("/inputs", key))
        if key not in declarations:
            _fail(_pointer("/inputs", key), "Undeclared invocation input")
    normalized = {}
    for key, declaration in declarations.items():
        path = _pointer("/inputs", key)
        _identifier(key, path)
        _object(declaration, path, {"type", "required", "default"}, ("type",))
        kind = declaration["type"]
        if type(kind) is not str or kind not in _INPUT_TYPES:
            _fail(path + "/type", "Unsupported input type")
        required = declaration.get("required", False)
        if type(required) is not bool:
            _fail(path + "/required", "Expected a boolean")
        if required and "default" in declaration:
            _fail(path + "/default", "Required inputs cannot have defaults")
        expected = _INPUT_TYPES[kind]
        if "default" in declaration and type(declaration["default"]) is not expected:
            _fail(path + "/default", "Default does not match declared input type")
        if key in supplied:
            if type(supplied[key]) is not expected:
                _fail(path, "Input does not match declared type")
            normalized[key] = copy.deepcopy(supplied[key])
        elif "default" in declaration:
            normalized[key] = copy.deepcopy(declaration["default"])
        elif required:
            _fail(path, "Required input is missing")
    if len(canonical(normalized).encode("utf-8")) > MAX_INPUT_BYTES:
        _fail("/inputs", "Canonical inputs exceed 64 KiB")
    return normalized


def _render(text, inputs, path, limit):
    _string(text, path)
    parts = []
    offset = 0
    byte_count = 0
    while offset < len(text):
        start = text.find("{{", offset)
        if start < 0:
            parts.append(text[offset:])
            break
        parts.append(text[offset:start])
        if text.startswith("{{{{", start):
            part, offset = "{{", start + 4
        else:
            token = _TOKEN.match(text, start)
            if token is None:
                _fail(path, "Invalid or unterminated interpolation")
            name = token.group(1)
            if name not in inputs:
                _fail(path, "Interpolation requires a present declared input")
            value = inputs[name]
            if type(value) not in (str, int, bool):
                _fail(path, "Only scalar inputs can interpolate")
            part = value if type(value) is str else canonical(value)
            offset = token.end()
        byte_count += len(parts[-1].encode("utf-8")) + len(part.encode("utf-8"))
        if byte_count > limit:
            _fail(path, "Rendered text exceeds UTF-8 byte limit")
        parts.append(part)
    rendered = "".join(parts)
    _bounded_text(rendered, path, limit)
    return rendered


def _native(validator, path, *args):
    try:
        return validator(*args)
    except (ValueError, TypeError):
        # Native errors may embed input values; the public diagnostic must not.
        _fail(path, "Task control rejected by native validator")


def _task_controls(task, path):
    _object(task, path, _TASK_FIELDS)
    for key, value in task.items():
        location = _pointer(path, key)
        if value is None:
            _fail(location, "Omit optional controls instead of supplying null")
        if key in _STRING_CONTROLS:
            _string(value, location)
        if key in _NUMERIC_CONTROLS:
            _integer(value, location, *_NUMERIC_CONTROLS[key])
    if "skills" in task:
        if type(task["skills"]) is not list:
            _fail(path + "/skills", "Expected an array of strings")
        for index, skill in enumerate(task["skills"]):
            _string(skill, _pointer(path + "/skills", index))
    goal_mode = task.get("goal_mode", False)
    if type(goal_mode) is not bool:
        _fail(path + "/goal_mode", "Expected a boolean")
    if "goal_max_turns" in task and not goal_mode:
        _fail(path + "/goal_max_turns", "Turns require explicit goal_mode=true")
    if "workspace_kind" in task and task["workspace_kind"] not in {"scratch", "worktree"}:
        _fail(path + "/workspace_kind", "Only scratch and worktree workspaces are portable")
    if "provider" in task and not task.get("model", "").strip():
        _fail(path + "/provider", "Provider requires a nonempty model")

    # Import only the existing pure validators, never invoke configuration or DB APIs.
    from hermes_cli.kanban_db import (
        _normalize_task_skills, _validate_model_override, normalize_reasoning_effort,
    )
    from hermes_cli.kanban_pr_acceptance import validate_contract

    result = {key: value for key, value in task.items() if key not in {"model", "provider"}}
    result["priority"] = task.get("priority", 0)
    result["goal_mode"] = goal_mode
    if "model" in task or "provider" in task:
        model, provider = _native(_validate_model_override, path + "/model",
                                  task.get("model"), task.get("provider"))
        result["model_override"] = model
        result["provider_override"] = provider
    validators = {"skills": _normalize_task_skills, "reasoning_effort": normalize_reasoning_effort,
                  "completion_contract": validate_contract}
    for key, validator in validators.items():
        if key in task:
            result[key] = _native(validator, _pointer(path, key), task[key])
    return result


def _nodes(definition, inputs):
    roles = definition["roles"]
    _array(roles, "/roles", 64, 1)
    declared_roles = set()
    for index, role in enumerate(roles):
        path = _pointer("/roles", index)
        _identifier(role, path)
        if role in declared_roles:
            _fail(path, "Duplicate role declaration")
        declared_roles.add(role)
    raw_nodes = definition["nodes"]
    _array(raw_nodes, "/nodes", 256, 1)
    nodes, keys = [], set()
    edge_count = 0
    for index, node in enumerate(raw_nodes):
        path = _pointer("/nodes", index)
        _object(node, path, {"key", "role", "title", "body", "needs", "task"}, ("key", "role", "title"))
        _identifier(node["key"], path + "/key")
        _identifier(node["role"], path + "/role")
        if node["key"] in keys:
            _fail(path + "/key", "Duplicate node declaration")
        keys.add(node["key"])
        if node["role"] not in declared_roles:
            _fail(path + "/role", "Undeclared role reference")
        needs = node.get("needs", [])
        _array(needs, path + "/needs", 2048)
        seen = set()
        for position, parent in enumerate(needs):
            location = _pointer(path + "/needs", position)
            _identifier(parent, location)
            if parent in seen:
                _fail(location, "Duplicate dependency declaration")
            seen.add(parent)
        edge_count += len(needs)
        if edge_count > 2048:
            _fail(path + "/needs", "At most 2048 edges are allowed")
        title = _render(node["title"], inputs, path + "/title", 1024)
        if not title.strip():
            _fail(path + "/title", "Rendered title must contain non-whitespace text")
        nodes.append({"key": node["key"], "role": node["role"], "title": title,
                      "body": _render(node.get("body", ""), inputs, path + "/body", 65536),
                      "needs": list(needs), "task": _task_controls(node.get("task", {}), path + "/task")})
    if len(canonical(nodes).encode("utf-8")) > MAX_PLAN_BYTES:
        _fail("/nodes", "Rendered plan exceeds 2 MiB")
    return nodes


def _topology(nodes):
    indices = {node["key"]: index for index, node in enumerate(nodes)}
    pending = [len(node["needs"]) for node in nodes]
    children = [[] for _ in nodes]
    for index, node in enumerate(nodes):
        for position, parent in enumerate(node["needs"]):
            if parent not in indices:
                _fail(f"/nodes/{index}/needs/{position}", "Unknown dependency reference")
            children[indices[parent]].append(index)
    ready = [index for index, count in enumerate(pending) if not count]
    heapq.heapify(ready)
    order = []
    while ready:
        index = heapq.heappop(ready)
        order.append(nodes[index]["key"])
        for child in children[index]:
            pending[child] -= 1
            if not pending[child]:
                heapq.heappush(ready, child)
    if len(order) != len(nodes):
        _fail("/nodes", "Dependency graph contains a cycle")
    return order


def prepare_definition(definition: dict, inputs: dict) -> dict:
    """Validate and render without mutating either argument or resolving defaults.

    Only portable priority=0 and goal_mode=false are filled here. Omitted local
    controls, notably goal_max_turns for goal mode, stay absent for the resolver.
    The digest always covers the original portable object, never rendered nodes.
    """
    _json_safe(definition)
    _json_safe(inputs, "/inputs")
    _object(definition, "", {"schema_version", "recipe_id", "description", "inputs", "roles", "nodes"},
            ("schema_version", "recipe_id", "roles", "nodes"))
    _integer(definition["schema_version"], "/schema_version", 1, 1)
    _identifier(definition["recipe_id"], "/recipe_id")
    encoded = canonical(definition).encode("utf-8")
    if len(encoded) > MAX_DEFINITION_BYTES:
        _fail("", "Canonical definition exceeds 1 MiB")
    if "description" in definition:
        _bounded_text(definition["description"], "/description", 4096)
    normalized = _normalize_inputs(definition.get("inputs", {}), inputs)
    nodes = _nodes(definition, normalized)
    return {"definition": definition, "definition_digest": hashlib.sha256(encoded).hexdigest(),
            "inputs": normalized, "nodes": nodes, "order": _topology(nodes)}
