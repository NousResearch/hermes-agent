#!/usr/bin/env python3
"""Guarded fallback CLI for Rive's local MCP server (Python 3.11+, MCP SDK 2.x).

Prefer native Hermes MCP when available. Install scripts/requirements.txt in a
separate virtual environment; this script never installs or launches anything.
Examples (run from this directory, using that environment's Python):
  python rive_mcp.py list
  python rive_mcp.py schema session_info
  python rive_mcp.py call session_info --args empty-args.json
  python rive_mcp.py call open_file_editor --args current-file-args.json
  python rive_mcp.py call list_artboards --args empty-args.json --expected-file FILE_ID
  python rive_mcp.py call TOOL --args arguments.json --dry-run
  python rive_mcp.py call TOOL --args arguments.json --apply --expected-file FILE_ID

The exact-file check is a preflight, not an atomic editor lock. Inspect/read back
the intended objects after a change; never automatically retry an uncertain call.
"""
# Author: Chris (cygnostik), ProDyn https://prodyn.ai. MIT.
import argparse
import asyncio
from contextlib import asynccontextmanager
from importlib import import_module
from importlib.metadata import version, PackageNotFoundError
import json
import logging
import math
from pathlib import Path
import sys

ENDPOINT = "http://127.0.0.1:9791/mcp"


def require_dependencies():
    # Keep --help and the TCP-only doctor usable in a stock Python installation.
    try:
        for name in ("httpx2", "jsonschema.validators", "referencing", "mcp.client.streamable_http"):
            import_module(name)
        if version("mcp").split(".")[0] != "2":
            raise ImportError("unsupported MCP major version")
    except (ImportError, PackageNotFoundError) as exc:
        raise CliError("MCP SDK 2.x dependencies unavailable or incompatible; install scripts/requirements.txt in a separate Python 3.11+ virtual environment (no automatic install).") from exc


@asynccontextmanager
async def sdk_connection(*, transport=None):
    require_dependencies()
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    # No endpoint/header/auth flags, proxy environment, redirects, or POST retries.
    transport = transport or httpx2.AsyncHTTPTransport(retries=0, trust_env=False)
    async with httpx2.AsyncClient(transport=transport, trust_env=False,
                                 follow_redirects=False, timeout=30.0) as client:
        async with streamable_http_client(ENDPOINT, http_client=client, terminate_on_close=False) as streams:
            async with ClientSession(*streams, read_timeout_seconds=30.0) as session:
                yield session


# Explicit reviewed policy; server readOnlyHint and read-looking names confer no trust.
READ_FIELDS = {
    "session_info": set(),
    "query_property_keys": {"objectIds", "animates", "binds"},
    "query_objects": {"objectIds", "depth"},
    "query_property_values": {"propertyKeys"},
    "find_objects": {"name", "type", "parentId"},
    "get_artboard_hierarchy": {"artboardId", "depth"},
    "list_artboards": set(), "get_selection": set(),
    "script_diagnostics": {"path"}, "get_scripts": set(),
    "grep": {"pattern", "inclusion_set", "case_sensitive", "match_whole_word", "regular_expression"},
    "read_console": {"script_name", "entry_type", "text_search", "limit", "offset", "reverse"},
    "get_scripting_reference": {"topic"},
}
READ_COMMANDS = {
    "animation_editor": {"listStateMachines", "listLinearAnimations", "queryStateMachine", "queryStateMachineLayer", "queryKeyFrames"},
    "assets_tool": {"listAssets", "queryAsset"},
    "tag_editor": {"queryTags"},
    "open_file_editor": {"getCurrentFile", "listArtboards", "getSelectedArtboard"},
    "viewmodel_editor": {"listViewModels", "listViewModelInstances", "listDataBinds", "listConverters", "listDataEnums"},
    "mesh_rigging_tool": {"querySkin"},
    "text_editor": {"view"},
}


def is_read_only(name, arguments):
    if name in READ_FIELDS:
        return set(arguments) <= READ_FIELDS[name]
    command = arguments.get("command")
    if not isinstance(command, str) or command not in READ_COMMANDS.get(name, set()):
        return False
    fields = {"command", "path", "view_range"} if name == "text_editor" else {"command", "data"}
    data = arguments.get("data", {})
    return set(arguments) <= fields and isinstance(data, dict) and set(data) <= {command}


def as_json(value):
    return value.model_dump(by_alias=True, mode="json", exclude_none=True) if hasattr(value, "model_dump") else value


class CliError(Exception):
    """Safe static error text only; never include raw arguments/server messages."""


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def number(value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("non-finite number")
        return value

    def constant(value):
        raise ValueError("non-JSON constant")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)


def read_arguments(path):
    try:
        result = strict_json(Path(path).read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("arguments must be an object")
        return result
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise CliError("arguments must be a readable strict JSON object file") from exc


def response_layers(value):
    """Walk response envelopes, not arbitrary object properties or source text."""
    value = as_json(value)
    if isinstance(value, str):
        try:
            value = strict_json(value)
        except ValueError:
            # Text may be source/prose. But JSON accepted only by Python's lax
            # parser (duplicate keys, NaN, overflow) must not conceal errors.
            try:
                json.loads(value)
            except ValueError:
                return
            raise CliError("server returned ambiguous/non-strict JSON content")
    if isinstance(value, list):
        for item in value:
            yield from response_layers(item)
    elif isinstance(value, dict):
        yield value
        if value.get("type") == "text":
            yield from response_layers(value.get("text"))
        for key in ("structuredContent", "result", "content"):
            if key in value:
                yield from response_layers(value[key])


def check_result(result):
    for layer in response_layers(result):
        if (layer.get("isError") or layer.get("success") is False
                or layer.get("errors") or layer.get("error")):
            raise CliError("server reported an error (details withheld; no automatic retry)")
    return as_json(result)


def local_schema_only(value):
    """Prevent even SDK output validation from resolving a non-loopback $ref."""
    from jsonschema.validators import validator_for
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"$ref", "$dynamicRef", "$recursiveRef"} and (not isinstance(child, str) or not child.startswith("#")):
                raise CliError("schema has non-local references; refusing external resolution")
            if key == "$schema" and validator_for({"$schema": child}, default=None) is None:
                raise CliError("schema uses an unsupported dialect")
            local_schema_only(child)
    elif isinstance(value, list):
        for child in value:
            local_schema_only(child)


def validate_arguments(tool, arguments):
    from jsonschema.validators import validator_for
    from referencing import Registry
    try:
        schema = tool["inputSchema"]
        local_schema_only(schema)
        output = tool.get("outputSchema")
        if output is not None:
            local_schema_only(output)
            validator_for(output).check_schema(output)
        validator = validator_for(schema)
        validator.check_schema(schema)
        validator(schema, registry=Registry()).validate(arguments)
    except Exception as exc:
        raise CliError("arguments or live tool schema invalid (schema validation failed)") from exc


async def active_file(session, tools):
    info = next((t for t in tools if t["name"] == "session_info"), None)
    if info is None:
        raise CliError("session_info unavailable; cannot verify active file")
    validate_arguments(info, {})
    result = check_result(await session.call_tool("session_info", {}))
    ids = [layer["activeFileId"] for layer in response_layers(result) if "activeFileId" in layer]
    if not ids or any(not isinstance(value, str) or not value for value in ids):
        raise CliError("activeFileId missing, null, or invalid; no call made")
    if any(value != ids[0] for value in ids):
        raise CliError("activeFileId inconsistent; no call made")
    return ids[0]


async def all_tools(session):
    from mcp.types import PaginatedRequestParams
    tools, names, cursors = [], set(), set()
    cursor = None
    while True:
        listing = await session.list_tools(**({"params": PaginatedRequestParams(cursor=cursor)} if cursor else {}))
        for value in listing.tools:
            tool = as_json(value)
            if tool["name"] in names:
                raise CliError("duplicate tool names in live catalog; refusing ambiguous schema")
            names.add(tool["name"])
            tools.append(tool)
        cursor = listing.next_cursor
        if cursor is None:
            return tools
        if not cursor or cursor in cursors or len(cursors) >= 100:
            raise CliError("invalid or excessive tool pagination; refusing partial catalog")
        cursors.add(cursor)


async def dispatch(options, connect):
    async with connect() as session:
        await session.initialize()
        tools = await all_tools(session)
        if options.action == "list":
            return {"count": len(tools), "tools": tools}
        tool = next((t for t in tools if t["name"] == options.tool), None)
        if tool is None:
            raise CliError("unknown tool")
        if options.action == "schema":
            return tool
        arguments = options.arguments
        validate_arguments(tool, arguments)
        readonly = is_read_only(options.tool, arguments)
        if options.dry_run:
            return {"dry_run": True, "tool": options.tool,
                    "classification": "read-only" if readonly else "mutation-class",
                    "schema_valid": True, "target_called": False, "active_file_checked": False,
                    "requires_for_call": [] if readonly else ["--apply", "--expected-file matching a fresh activeFileId"]}
        if not readonly:
            if not options.apply or not options.expected_file:
                raise CliError("mutation-class call requires --apply and --expected-file")
        if not readonly or options.expected_file is not None:
            active = await active_file(session, tools)
            if active != options.expected_file:
                raise CliError("activeFileId does not match --expected-file; no call made")
        return check_result(await session.call_tool(options.tool, arguments))


def exception_leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in exception_leaves(child)]
    return [error]


class SafeParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, allow_abbrev=False)

    def error(self, message):
        raise CliError("invalid CLI arguments; use --help (values withheld)")


def main(argv=None, *, connect=None, stdout=None, stderr=None):
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    parser = SafeParser(prog="rive_mcp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("list")
    commands.add_parser("schema").add_argument("tool")
    call = commands.add_parser("call")
    call.add_argument("tool")
    call.add_argument("--args", required=True, help="Path to a JSON object; never inline credentials")
    mode = call.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Explicitly authorize mutation-class calls")
    mode.add_argument("--dry-run", action="store_true", help="Preflight live schema and policy only; never call a tool or check file identity")
    call.add_argument("--expected-file", help="Exact activeFileId; required for mutations, optional file guard for reads")
    logging_disabled = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        options = parser.parse_args(argv)
        if options.action == "call":
            options.arguments = read_arguments(options.args)
        require_dependencies()
        result = asyncio.run(dispatch(options, connect or sdk_connection))
    except CliError as exc:
        print(str(exc), file=stderr)
        return 2
    except Exception as exc:
        leaves = exception_leaves(exc)
        if leaves and all(isinstance(leaf, CliError) for leaf in leaves):
            print(str(leaves[0]), file=stderr)
            return 2
        print("MCP/transport/protocol failure; call not retried; outcome may be unknown. Inspect editor before any manual retry.", file=stderr)
        return 3
    finally:
        logging.disable(logging_disabled)
    print(json.dumps(result, indent=2), file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
