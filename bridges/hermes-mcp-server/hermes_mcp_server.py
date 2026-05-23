#!/usr/bin/env python3
"""Expose selected Hermes services as a small stdio MCP server for jcode."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_VERSION = "0.1.0"
HERMES_MCP_CONTRACT_VERSION = "hermes-mcp.v1"
HERMES_MCP_SCHEMA_RELATIVE_DIR = Path("contracts") / "hermes_mcp" / "v1"
HERMES_MCP_SCHEMA_FILENAMES = (
    "initialize_response.schema.json",
    "tools_list_response.schema.json",
    "tools_call_response.schema.json",
)


def _bootstrap_paths() -> Path:
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("JCODE_BRIDGE_ROOT", str(root))

    candidates = [
        root,
        root / "bridges" / "hermes-plugin-jcode",
        root / "upstreams" / "hermes",
    ]

    manifest = root / "hermes-jcode.manifest.json"
    if manifest.exists():
        try:
            hermes_path = json.loads(manifest.read_text(encoding="utf-8")).get(
                "upstreams", {}
            ).get("hermes", {}).get("path")
        except Exception:
            hermes_path = None
        if isinstance(hermes_path, str):
            candidates.append(Path(hermes_path).expanduser())

    for candidate in candidates:
        if candidate.is_dir():
            resolved = str(candidate.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
    return root


ROOT = _bootstrap_paths()

from plugins.jcode_bridge.hermes_service import (  # noqa: E402
    DEFAULT_ALLOWED_TOOLS,
    HERMES_SERVICE_CONTRACT_VERSION,
    dispatch_service_request,
)


JsonDict = dict[str, Any]


def _dump(payload: JsonDict) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_dir() -> Path:
    return ROOT / "tests" / "fixtures" / "hermes_mcp"


def _schema_dir() -> Path:
    return ROOT / HERMES_MCP_SCHEMA_RELATIVE_DIR


def _check(name: str, ok: bool, **details: Any) -> JsonDict:
    payload: JsonDict = {"name": name, "ok": bool(ok)}
    payload.update(details)
    return payload


def _respond(request_id: Any, result: JsonDict) -> JsonDict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }


def _error(request_id: Any, code: int, message: str, data: Any = None) -> JsonDict:
    payload: JsonDict = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }
    if data is not None:
        payload["error"]["data"] = data
    return payload


def _object_schema(properties: JsonDict, *, required: list[str] | None = None) -> JsonDict:
    schema: JsonDict = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _validate_jsonrpc_response(payload: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["MCP response must be an object"]
    if payload.get("jsonrpc") != "2.0":
        errors.append("MCP response jsonrpc must be '2.0'")
    if "id" not in payload:
        errors.append("MCP response must include id")
    if "result" not in payload and "error" not in payload:
        errors.append("MCP response must include result or error")
    return errors


def validate_initialize_response(payload: Any) -> list[str]:
    errors = _validate_jsonrpc_response(payload)
    if errors:
        return errors
    result = payload.get("result")
    if not isinstance(result, dict):
        return ["initialize response result must be an object"]
    if not isinstance(result.get("protocolVersion"), str):
        errors.append("initialize response protocolVersion must be a string")
    capabilities = result.get("capabilities")
    if not isinstance(capabilities, dict) or not isinstance(capabilities.get("tools"), dict):
        errors.append("initialize response capabilities.tools must be an object")
    server_info = result.get("serverInfo")
    if not isinstance(server_info, dict):
        errors.append("initialize response serverInfo must be an object")
    elif not isinstance(server_info.get("name"), str):
        errors.append("initialize response serverInfo.name must be a string")
    return errors


def validate_tools_list_response(payload: Any) -> list[str]:
    errors = _validate_jsonrpc_response(payload)
    if errors:
        return errors
    result = payload.get("result")
    if not isinstance(result, dict):
        return ["tools/list response result must be an object"]
    tools = result.get("tools")
    if not isinstance(tools, list):
        return ["tools/list response tools must be a list"]
    names: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            errors.append(f"tools/list tool {index} must be an object")
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"tools/list tool {index} must include a non-empty name")
        else:
            names.add(name)
        if not isinstance(tool.get("inputSchema"), dict):
            errors.append(f"tools/list tool {name or index} inputSchema must be an object")
    required = {
        "hermes_tool",
        "hermes_web_search",
        "hermes_web_extract",
        "hermes_session_search",
        "hermes_memory",
    }
    missing = sorted(required - names)
    if missing:
        errors.append(f"tools/list missing required Hermes MCP tools: {', '.join(missing)}")
    return errors


def validate_tools_call_response(payload: Any) -> list[str]:
    errors = _validate_jsonrpc_response(payload)
    if errors:
        return errors
    result = payload.get("result")
    if not isinstance(result, dict):
        return ["tools/call response result must be an object"]
    if not isinstance(result.get("isError"), bool):
        errors.append("tools/call response isError must be a boolean")
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return errors + ["tools/call response content must be a non-empty list"]
    first = content[0]
    if not isinstance(first, dict):
        return errors + ["tools/call first content block must be an object"]
    if first.get("type") != "text" or not isinstance(first.get("text"), str):
        errors.append("tools/call first content block must be text with string text")
        return errors
    try:
        service_response = json.loads(first["text"])
    except json.JSONDecodeError as exc:
        errors.append(f"tools/call text must contain service response JSON: {exc}")
        return errors
    if not isinstance(service_response, dict):
        errors.append("tools/call service response must be an object")
    elif service_response.get("contract_version") != HERMES_SERVICE_CONTRACT_VERSION:
        errors.append(
            f"tools/call service response contract_version must be {HERMES_SERVICE_CONTRACT_VERSION}"
        )
    return errors


def _schema_checks() -> list[JsonDict]:
    checks: list[JsonDict] = []
    for filename in HERMES_MCP_SCHEMA_FILENAMES:
        path = _schema_dir() / filename
        if not path.exists():
            checks.append(_check(f"schema:{filename}", False, errors=["schema file is missing"]))
            continue
        try:
            payload = _load_json(path)
        except Exception as exc:
            checks.append(_check(f"schema:{filename}", False, errors=[str(exc)]))
            continue
        errors: list[str] = []
        if payload.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            errors.append("schema must declare JSON Schema draft 2020-12")
        if payload.get("x-bridge-contract-version") != HERMES_MCP_CONTRACT_VERSION:
            errors.append(
                f"schema x-bridge-contract-version must be {HERMES_MCP_CONTRACT_VERSION}"
            )
        if not isinstance(payload.get("$id"), str) or not payload.get("$id"):
            errors.append("schema must declare a non-empty $id")
        checks.append(_check(f"schema:{filename}", not errors, errors=errors))
    return checks


def _fixture_checks() -> list[JsonDict]:
    fixtures = (
        ("fixture:initialize_response", "initialize_response.json", validate_initialize_response),
        ("fixture:tools_list_response", "tools_list_response.json", validate_tools_list_response),
        (
            "fixture:tools_call_response_success",
            "tools_call_response_success.json",
            validate_tools_call_response,
        ),
    )
    checks: list[JsonDict] = []
    for name, filename, validator in fixtures:
        try:
            payload = _load_json(_fixture_dir() / filename)
        except Exception as exc:
            checks.append(_check(name, False, errors=[str(exc)]))
            continue
        errors = validator(payload)
        checks.append(_check(name, not errors, errors=errors))
    return checks


def _live_mock_check() -> JsonDict:
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "contract-check", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "hermes_web_search",
                "arguments": {"query": "bridge", "limit": 2},
            },
        },
    ]
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--mock"],
        input="\n".join(json.dumps(item) for item in requests) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return _check(
            "live:mock_mcp_roundtrip",
            False,
            returncode=completed.returncode,
            stderr=completed.stderr,
        )
    try:
        responses = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        return _check(
            "live:mock_mcp_roundtrip",
            False,
            errors=[f"failed to parse MCP stdout: {exc}"],
            stdout=completed.stdout,
        )
    errors: list[str] = []
    if len(responses) != 3:
        errors.append(f"expected 3 responses, got {len(responses)}")
    if len(responses) >= 1:
        errors.extend(validate_initialize_response(responses[0]))
    if len(responses) >= 2:
        errors.extend(validate_tools_list_response(responses[1]))
    if len(responses) >= 3:
        errors.extend(validate_tools_call_response(responses[2]))
        call_result = responses[2].get("result", {}) if isinstance(responses[2], dict) else {}
        if isinstance(call_result, dict) and call_result.get("isError") is not False:
            errors.append("mock tools/call should not be an error")
    return _check("live:mock_mcp_roundtrip", not errors, errors=errors)


def mcp_contract_report(*, live: bool = False) -> JsonDict:
    checks = _fixture_checks()
    checks.extend(_schema_checks())
    if live:
        checks.append(_live_mock_check())
    return {
        "success": all(item["ok"] for item in checks),
        "contract_version": HERMES_MCP_CONTRACT_VERSION,
        "fixture_dir": str(_fixture_dir()),
        "schema_dir": str(_schema_dir()),
        "schema_files": list(HERMES_MCP_SCHEMA_FILENAMES),
        "checks": checks,
    }


def _metadata_properties() -> JsonDict:
    return {
        "session_id": {
            "type": "string",
            "description": "Optional Hermes session id for tools that can use one.",
        },
        "task_id": {
            "type": "string",
            "description": "Optional Hermes task id for tool audit context.",
        },
        "confirm_outbound_human_contact": {
            "type": "boolean",
            "description": "Explicit operator confirmation for outbound human contact.",
        },
        "confirm_sensitive_person_data": {
            "type": "boolean",
            "description": "Explicit operator confirmation for sensitive person-data lookup.",
        },
        "safety_override_reason": {
            "type": "string",
            "description": "Short operator reason for an approved safety override.",
        },
    }


def _tool_definitions(allowed_tools: Iterable[str]) -> list[JsonDict]:
    allowed = tuple(dict.fromkeys(allowed_tools))
    metadata = _metadata_properties()
    tools: list[JsonDict] = [
        {
            "name": "hermes_tool",
            "description": (
                "Call an allowlisted Hermes service through the hermes-service.v1 "
                "contract."
            ),
            "inputSchema": _object_schema(
                {
                    "tool": {
                        "type": "string",
                        "enum": list(allowed),
                        "description": "Hermes service tool name.",
                    },
                    "args": {
                        "type": "object",
                        "description": "Arguments passed to the Hermes tool.",
                    },
                    **metadata,
                },
                required=["tool", "args"],
            ),
        }
    ]

    if "web_search" in allowed:
        tools.append({
            "name": "hermes_web_search",
            "description": "Search the web with Hermes' configured web provider.",
            "inputSchema": _object_schema(
                {
                    "query": {"type": "string", "description": "Search query."},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Maximum result count.",
                    },
                    **metadata,
                },
                required=["query"],
            ),
        })

    if "web_extract" in allowed:
        tools.append({
            "name": "hermes_web_extract",
            "description": "Extract page content with Hermes' configured web provider.",
            "inputSchema": _object_schema(
                {
                    "url": {
                        "type": "string",
                        "description": "Single URL to extract.",
                    },
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "One or more URLs to extract.",
                    },
                    **metadata,
                },
            ),
        })

    if "session_search" in allowed:
        tools.append({
            "name": "hermes_session_search",
            "description": "Search Hermes session history.",
            "inputSchema": _object_schema(
                {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    "sort": {
                        "type": "string",
                        "enum": ["relevance", "newest"],
                    },
                    "around_message_id": {"type": "string"},
                    "window": {"type": "integer", "minimum": 1, "maximum": 20},
                    "role_filter": {"type": "string"},
                    **metadata,
                },
                required=["query"],
            ),
        })

    if "memory" in allowed:
        tools.append({
            "name": "hermes_memory",
            "description": "Read or update Hermes persistent memory.",
            "inputSchema": _object_schema(
                {
                    "action": {
                        "type": "string",
                        "enum": ["add", "replace", "remove"],
                        "description": "Memory action.",
                    },
                    "target": {
                        "type": "string",
                        "enum": ["memory", "user"],
                        "description": "Memory target, usually memory or user.",
                    },
                    "content": {"type": "string"},
                    "old_text": {"type": "string"},
                    **metadata,
                },
                required=["action", "target"],
            ),
        })

    return tools


def _pop_metadata(arguments: JsonDict) -> tuple[JsonDict, JsonDict]:
    args = dict(arguments)
    metadata: JsonDict = {}
    for key in (
        "session_id",
        "task_id",
        "confirm_outbound_human_contact",
        "confirm_sensitive_person_data",
        "safety_override_reason",
    ):
        if key in args:
            metadata[key] = args.pop(key)
    return args, metadata


def _service_request_from_call(
    request_id: Any,
    tool_name: str,
    arguments: Any,
) -> JsonDict:
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ValueError("tools/call arguments must be an object")

    args, metadata = _pop_metadata(arguments)

    if tool_name == "hermes_tool":
        service_tool = args.get("tool")
        service_args = args.get("args")
        if not isinstance(service_tool, str) or not service_tool:
            raise ValueError("hermes_tool requires a non-empty string 'tool'")
        if not isinstance(service_args, dict):
            raise ValueError("hermes_tool requires object 'args'")
    elif tool_name == "hermes_web_search":
        service_tool = "web_search"
        service_args = {
            key: value
            for key, value in args.items()
            if key in {"query", "limit"} and value is not None
        }
    elif tool_name == "hermes_web_extract":
        service_tool = "web_extract"
        service_args = {}
        urls = args.get("urls")
        url = args.get("url")
        if isinstance(urls, list):
            service_args["urls"] = urls
        elif isinstance(url, str) and url:
            service_args["urls"] = [url]
    elif tool_name == "hermes_session_search":
        service_tool = "session_search"
        service_args = {
            key: value
            for key, value in args.items()
            if key in {"query", "limit", "sort", "around_message_id", "window", "role_filter"}
            and value is not None
        }
    elif tool_name == "hermes_memory":
        service_tool = "memory"
        service_args = args
    else:
        raise KeyError(f"unknown Hermes MCP tool: {tool_name}")

    return {
        "type": "hermes_service_request",
        "id": f"mcp:{request_id}" if request_id is not None else f"mcp:{time.time_ns()}",
        "tool": service_tool,
        "args": service_args,
        **metadata,
    }


def _mock_dispatch(tool: str, args: JsonDict, request: JsonDict) -> str:
    return json.dumps({
        "mock": True,
        "tool": tool,
        "args": args,
        "request_id": request.get("id"),
    }, ensure_ascii=True)


def _call_tool(
    request_id: Any,
    params: Any,
    *,
    allowed_tools: Iterable[str],
    mock: bool,
) -> JsonDict:
    if not isinstance(params, dict):
        return _error(request_id, -32602, "tools/call params must be an object")
    tool_name = params.get("name")
    if not isinstance(tool_name, str):
        return _error(request_id, -32602, "tools/call params.name must be a string")

    try:
        service_request = _service_request_from_call(
            request_id,
            tool_name,
            params.get("arguments") or {},
        )
    except KeyError as exc:
        return _error(request_id, -32602, str(exc))
    except ValueError as exc:
        return _error(request_id, -32602, str(exc))

    response = dispatch_service_request(
        service_request,
        allowed_tools=tuple(allowed_tools),
        dispatcher=_mock_dispatch if mock else None,
    )
    ok = bool(response.get("ok"))
    return _respond(request_id, {
        "content": [
            {
                "type": "text",
                "text": json.dumps(response, ensure_ascii=True, sort_keys=True),
            }
        ],
        "isError": not ok,
    })


def _handle_request(
    request: Any,
    *,
    allowed_tools: Iterable[str],
    mock: bool,
    server_name: str,
) -> JsonDict | None:
    if not isinstance(request, dict):
        return _error(None, -32600, "JSON-RPC request must be an object")

    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params")
    if not isinstance(method, str):
        return _error(request_id, -32600, "JSON-RPC request method must be a string")

    if method.startswith("notifications/"):
        return None

    if method == "initialize":
        protocol_version = MCP_PROTOCOL_VERSION
        if isinstance(params, dict) and isinstance(params.get("protocolVersion"), str):
            protocol_version = params["protocolVersion"]
        return _respond(request_id, {
            "protocolVersion": protocol_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": server_name,
                "version": SERVER_VERSION,
            },
        })

    if method == "tools/list":
        return _respond(request_id, {"tools": _tool_definitions(allowed_tools)})

    if method == "tools/call":
        return _call_tool(request_id, params, allowed_tools=allowed_tools, mock=mock)

    if method == "shutdown":
        if request_id is None:
            return None
        return _respond(request_id, {})

    return _error(request_id, -32601, f"unknown method: {method}")


def run_stdio_mcp_server(
    *,
    allowed_tools: Iterable[str] = DEFAULT_ALLOWED_TOOLS,
    mock: bool = False,
    server_name: str = "hermes",
) -> int:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        request: Any = None
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            response = _error(None, -32700, f"parse error: {exc}")
        else:
            response = _handle_request(
                request,
                allowed_tools=allowed_tools,
                mock=mock,
                server_name=server_name,
            )
        if response is not None:
            print(_dump(response), flush=True)
        if isinstance(request, dict) and request.get("method") == "shutdown":
            break
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate Hermes MCP bridge fixtures, schemas, and optional live mock roundtrip.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="With --check, run a live mock MCP roundtrip through this server.",
    )
    parser.add_argument(
        "--allow-tool",
        action="append",
        dest="allow_tools",
        help="Allowed Hermes service tool. May be provided multiple times.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Return deterministic mock dispatch payloads for smoke tests.",
    )
    parser.add_argument(
        "--name",
        default="hermes",
        help="MCP server name reported during initialize.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"hermes-mcp-server {SERVER_VERSION} ({HERMES_SERVICE_CONTRACT_VERSION})",
    )
    ns = parser.parse_args(argv)
    if ns.check:
        report = mcp_contract_report(live=bool(ns.live))
        print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))
        return 0 if report.get("success") else 1
    return run_stdio_mcp_server(
        allowed_tools=tuple(ns.allow_tools or DEFAULT_ALLOWED_TOOLS),
        mock=bool(ns.mock),
        server_name=ns.name,
    )


if __name__ == "__main__":
    raise SystemExit(main())
