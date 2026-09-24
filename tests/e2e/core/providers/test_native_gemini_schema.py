"""Gemini native wire conformance: MCP tool schemas Google's parser rejects arrive sanitized.

An MCP server (a tiny stdio JSON-RPC process) exposes one tool whose ``inputSchema`` uses JSON
Schema keywords that Google's ``FunctionDeclaration`` parser does not accept (``$schema``,
``$ref``/``$defs``, ``additionalProperties``, ``oneOf``, ``const``, list-valued ``type``, integer
``enum``, a ``required`` entry naming no property). A real ``hermes chat -q`` turn declares it to
the fake Google endpoint, which rejects anything Google would (HTTP 400 INVALID_ARGUMENT) and
then drives a call to the tool so the round trip is proven end to end.

Two API surfaces: ``v1beta`` (default; ``parametersJsonSchema``) and ``v1`` (pinned through
``model.base_url``; the proto ``Schema`` subset in ``parameters``, validated field by field).
"""

from __future__ import annotations

import json
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.core.providers import _native_helpers as nh
from tests.fakes.providers.gemini_native import HERMES_ENV, Call, Calls, GeminiFake, Recorded, Text, hermes_model

KNOWN = {
    "ref_dropped_v1": "#99438 legacy `parameters` path drops $ref/$defs instead of inlining (empty schema)",
    "array_items_v1": "#71804 array parameter without `items` is sent as-is; Google 400s 'items: missing field'",
}

TOOL = "mcp__hostile__lookup"
MCP_CANARY = "GEMINI-MCP-CANARY-7731"
DONE = "GEMINI-MCP-DONE"
V1 = "https://generativelanguage.googleapis.com/v1"
_FORBIDDEN_ANYWHERE = ("$ref", "$schema", "$defs")

HOSTILE: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "$defs": {"Mode": {"type": "string", "enum": ["fast", "slow"]}},
    "properties": {
        "query": {"type": "string", "minLength": 1, "examples": ["x"]},
        "mode": {"$ref": "#/$defs/Mode"},
        "limit": {"type": ["integer", "null"], "exclusiveMinimum": 0, "enum": [1, 5, 10]},
        "filters": {"type": "array", "items": {"oneOf": [
            {"type": "string"},
            {"type": "object", "properties": {"k": {"type": "string"}}, "additionalProperties": False}]}},
        "flag": {"const": True},
    },
    "required": ["query", "not_a_property"],
}
ITEMLESS: dict[str, Any] = {"type": "object", "properties": {"tags": {"type": "array", "description": "tags"}}}

_MCP_SERVER = textwrap.dedent('''
    import json, sys
    TOOLS = json.loads(sys.argv[1])
    CANARY = sys.argv[2]
    for line in sys.stdin:
        msg = json.loads(line)
        mid, method = msg.get("id"), msg.get("method")
        if mid is None:
            continue
        if method == "initialize":
            res = {"protocolVersion": msg["params"].get("protocolVersion", "2025-06-18"),
                   "capabilities": {"tools": {}}, "serverInfo": {"name": "hostile", "version": "1"}}
        elif method == "tools/list":
            res = {"tools": TOOLS}
        elif method == "tools/call":
            args = msg["params"].get("arguments") or {}
            res = {"content": [{"type": "text", "text": CANARY + " query=" + str(args.get("query"))}]}
        elif method == "ping":
            res = {}
        else:
            err = {"code": -32601, "message": "method not found: " + str(method)}
            print(json.dumps({"jsonrpc": "2.0", "id": mid, "error": err}), flush=True)
            continue
        print(json.dumps({"jsonrpc": "2.0", "id": mid, "result": res}), flush=True)
''')


@dataclass
class Outcome:
    result: nh.ChatResult
    calls: list[Recorded]
    rejections: list[str]

    def declaration(self) -> dict[str, Any]:
        assert self.calls, self.result.describe()
        decl = self.calls[0].declarations().get(TOOL)
        assert decl is not None, f"{TOOL} not declared: {sorted(self.calls[0].declarations())}"
        return decl

    def function_responses(self) -> list[dict[str, Any]]:
        return [p["functionResponse"] for rec in self.calls for p in rec.parts("functionResponse")]


def _scenario(root: Path, schema: dict[str, Any], base_url: str | None) -> Outcome:
    root.mkdir(parents=True)
    server = root / "hostile_mcp.py"
    server.write_text(_MCP_SERVER, encoding="utf-8")
    tools = [{"name": "lookup", "description": "Look something up.", "inputSchema": schema}]
    extra = {
        # Keep the MCP tool in the model-facing array (not behind the tool_search bridge) and let
        # discovery finish before the one-shot turn is built.
        "tools": {"tool_search": {"enabled": "off"}},
        "mcp_discovery_timeout": 60,
        "mcp_single_query_discovery_timeout": 60,
        "mcp_servers": {"hostile": {"command": sys.executable,
                                    "args": [str(server), json.dumps(tools), MCP_CANARY]}},
    }
    home = nh.make_home(root, hermes_model(base_url), env_file=HERMES_ENV, extra_config=extra)
    script = [Calls([Call(TOOL, {"query": "q1", "mode": "fast"})]), Text(DONE)]
    with GeminiFake(root / "fake", script) as fake:
        result = nh.run_chat(home, "Use the lookup tool for q1.", env=fake.child_env())
    return Outcome(result, fake.generate_calls(), fake.rejections())


SCENARIOS = {"v1beta": (HOSTILE, None), "v1": (HOSTILE, V1), "v1_itemless": (ITEMLESS, V1)}


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Outcome]:
    base = tmp_path_factory.mktemp("gemini_schema")
    with ThreadPoolExecutor(len(SCENARIOS)) as pool:
        futures = {k: pool.submit(_scenario, base / k, *v) for k, v in SCENARIOS.items()}
        return {k: f.result() for k, f in futures.items()}


def _assert_round_trip(o: Outcome, version: str) -> None:
    assert o.rejections == [], o.rejections
    assert [c.version for c in o.calls] == [version, version], [(c.version, c.status) for c in o.calls]
    responses = [r for r in o.function_responses() if r.get("name") == TOOL]
    assert responses and MCP_CANARY in json.dumps(responses[0]["response"]), o.function_responses()
    assert f"{MCP_CANARY} query=q1" in json.dumps(responses[0]["response"]), responses[0]
    assert o.result.returncode == 0 and DONE in o.result.stdout, o.result.describe()


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_v1beta_json_schema_declaration_is_sanitized(outcomes: dict[str, Outcome]) -> None:
    o = outcomes["v1beta"]
    _assert_round_trip(o, "v1beta")
    schema = o.declaration().get("parametersJsonSchema")
    assert isinstance(schema, dict), o.declaration()
    leaked = [k for node in _walk(schema) for k in node if k in _FORBIDDEN_ANYWHERE]
    assert leaked == [], f"reference/meta keywords reached Google: {leaked} in {schema}"
    assert schema["properties"]["mode"].get("enum") == ["fast", "slow"], schema["properties"]["mode"]
    assert set(schema.get("required") or []) <= set(schema["properties"]), schema.get("required")


def test_v1_proto_schema_declaration_is_accepted(outcomes: dict[str, Outcome]) -> None:
    """Every field of ``parameters`` parses as Google's proto ``Schema`` (the fake 400s otherwise)."""
    o = outcomes["v1"]
    _assert_round_trip(o, "v1")
    params = o.declaration().get("parameters")
    assert isinstance(params, dict) and "parametersJsonSchema" not in o.declaration(), o.declaration()
    assert params["required"] == ["query"], params


@pytest.mark.xfail(strict=True, raises=nh.KnownSymptom, reason=KNOWN["ref_dropped_v1"])
def test_v1_ref_parameter_keeps_its_shape(outcomes: dict[str, Outcome]) -> None:
    _assert_round_trip(outcomes["v1"], "v1")
    mode = outcomes["v1"].declaration()["parameters"]["properties"]["mode"]
    if mode == {}:
        raise nh.KnownSymptom(f"$ref-typed parameter lost its shape on the v1 wire: {mode}")
    assert mode.get("enum") == ["fast", "slow"], mode


@pytest.mark.xfail(strict=True, raises=nh.KnownSymptom, reason=KNOWN["array_items_v1"])
def test_v1_array_without_items_is_accepted(outcomes: dict[str, Outcome]) -> None:
    o = outcomes["v1_itemless"]
    assert o.calls, o.result.describe()
    if any("items: missing field" in r for r in o.rejections):
        raise nh.KnownSymptom(f"Google rejected the item-less array: {o.rejections}")
    _assert_round_trip(o, "v1")
