"""``hermes mcp fixtures`` — deterministic record/replay for MCP servers.

The core test-suite for MCP (``tests/tools/test_mcp_tool.py``) mocks the
transport and never starts a real server/process, so protocol framing,
lifecycle, and transport-specific quirks are untested end to end. This
module records a REAL run against a REAL stdio MCP server (initialize,
list_tools, and any requested tool calls) into a fixture file, and can
replay that fixture through a REAL client/server round-trip
(``hermes_cli.mcp_replay_server``) so tests exercise genuine protocol
messages without a live backend.

Secrecy: fixtures pass through the repo's central redactor
(``agent.redact.redact_sensitive_text``) before being written to disk, and
the recording subprocess env is built via ``tools.mcp_tool._build_safe_env``
— the same allowlist-based env filter live MCP servers get, never the raw
host environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

__all__ = ["record_fixture", "cmd_mcp_fixtures"]


def _parse_call_args(raw: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """Parse ``--call NAME=JSON`` entries into ``(name, arguments)`` pairs."""
    calls: List[Tuple[str, Dict[str, Any]]] = []
    for entry in raw:
        if "=" not in entry:
            raise ValueError(f"invalid --call {entry!r} — expected TOOL=JSON_ARGS")
        name, _, raw_args = entry.partition("=")
        name = name.strip()
        raw_args = raw_args.strip() or "{}"
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON arguments for --call {name}: {exc}") from exc
        if not isinstance(arguments, dict):
            raise ValueError(f"--call {name} arguments must be a JSON object")
        calls.append((name, arguments))
    return calls


async def _record_async(
    server_name: str, server_cfg: Dict[str, Any],
    calls: List[Tuple[str, Dict[str, Any]]], timeout: float,
) -> Dict[str, Any]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from tools.mcp_tool import _build_safe_env

    command = server_cfg.get("command")
    if not command:
        raise ValueError(
            f"mcp_servers.{server_name} has no 'command' — only stdio "
            "servers can be recorded"
        )
    args = server_cfg.get("args") or []
    user_env = server_cfg.get("env") or {}
    env = _build_safe_env(user_env if isinstance(user_env, dict) else {})

    params = StdioServerParameters(command=command, args=list(args), env=env)

    fixture: Dict[str, Any] = {
        "schema_version": 1,
        "server_name": server_name,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "initialize": None,
        "tools": [],
        "calls": [],
    }

    async def _run() -> None:
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                init = await session.initialize()
                server_info = getattr(init, "serverInfo", None)
                fixture["initialize"] = {
                    "protocol_version": init.protocolVersion,
                    "server_name": getattr(server_info, "name", None),
                    "server_version": getattr(server_info, "version", None),
                }

                tools_result = await session.list_tools()
                fixture["tools"] = [
                    {
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.inputSchema,
                    }
                    for t in tools_result.tools
                ]

                for name, arguments in calls:
                    try:
                        call_result = await session.call_tool(name, arguments)
                        # The replay stub only ever serves TextContent (see
                        # hermes_cli/mcp_replay_server.py), so non-text results
                        # (images, audio, ...) would record with text=None and
                        # then fail their own replay self-check. Filter them
                        # out here instead of storing content the stub can't
                        # round-trip.
                        content = [
                            {"type": c.type, "text": c.text}
                            for c in call_result.content
                            if c.type == "text"
                        ]
                        fixture["calls"].append(
                            {
                                "name": name,
                                "arguments": arguments,
                                "content": content,
                                "is_error": bool(call_result.isError),
                            }
                        )
                    except Exception as exc:  # noqa: BLE001 — record the failure itself
                        fixture["calls"].append(
                            {
                                "name": name,
                                "arguments": arguments,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )

    await asyncio.wait_for(_run(), timeout=timeout)
    return fixture


def record_fixture(
    server_name: str, server_cfg: Dict[str, Any],
    calls: List[Tuple[str, Dict[str, Any]]], *, timeout: float = 30.0,
) -> Dict[str, Any]:
    """Record a fixture by driving a REAL stdio MCP server. Blocking."""
    return asyncio.run(_record_async(server_name, server_cfg, calls, timeout))


def write_fixture(fixture: Dict[str, Any], output_path: Path) -> None:
    from agent.redact import redact_sensitive_text

    raw = json.dumps(fixture, indent=2, sort_keys=False)
    redacted = redact_sensitive_text(raw, force=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(redacted, encoding="utf-8")


async def _replay_check_async(fixture_path: Path) -> Dict[str, Any]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from hermes_cli.mcp_replay_server import load_fixture

    fixture = load_fixture(fixture_path)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "hermes_cli.mcp_replay_server", str(fixture_path)],
        env={},
    )

    report: Dict[str, Any] = {"initialize_ok": False, "tools_ok": False, "calls": []}

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            report["initialize_ok"] = True

            tools_result = await session.list_tools()
            replayed_names = sorted(t.name for t in tools_result.tools)
            expected_names = sorted(t["name"] for t in fixture.get("tools", []))
            report["tools_ok"] = replayed_names == expected_names

            for call in fixture.get("calls", []):
                name = call["name"]
                arguments = call.get("arguments") or {}
                if "error" in call:
                    try:
                        await session.call_tool(name, arguments)
                        report["calls"].append({"name": name, "ok": False,
                                                 "reason": "expected error, got success"})
                    except Exception:  # noqa: BLE001 — expected path
                        report["calls"].append({"name": name, "ok": True})
                    continue
                result = await session.call_tool(name, arguments)
                replayed_text = [
                    getattr(c, "text", None) for c in result.content
                ]
                expected_text = [c.get("text") for c in call.get("content", [])]
                report["calls"].append(
                    {"name": name, "ok": replayed_text == expected_text}
                )

    return report


def cmd_mcp_fixtures(args: argparse.Namespace) -> int:
    action = getattr(args, "mcp_fixtures_action", None)

    if action == "record":
        from hermes_cli.config import load_config_readonly

        server_name = args.name
        config = load_config_readonly()
        servers = (config.get("mcp_servers") or {})
        server_cfg = servers.get(server_name)
        if not isinstance(server_cfg, dict):
            print(f"error: no mcp_servers.{server_name} in config.yaml", file=sys.stderr)
            return 1
        try:
            calls = _parse_call_args(args.call)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            fixture = record_fixture(
                server_name, server_cfg, calls, timeout=args.timeout
            )
        except Exception as exc:  # noqa: BLE001 — surface any recording failure
            print(f"error: recording failed: {exc}", file=sys.stderr)
            return 1
        write_fixture(fixture, Path(args.output))
        print(f"wrote {args.output} "
              f"({len(fixture['tools'])} tools, {len(fixture['calls'])} calls)")
        return 0

    if action == "replay":
        fixture_path = Path(args.fixture)
        if not fixture_path.exists():
            print(f"error: fixture not found: {fixture_path}", file=sys.stderr)
            return 1
        try:
            report = asyncio.run(_replay_check_async(fixture_path))
        except Exception as exc:  # noqa: BLE001 — surface any replay failure
            print(f"error: replay failed: {exc}", file=sys.stderr)
            return 1
        print(f"initialize: {'ok' if report['initialize_ok'] else 'FAIL'}")
        print(f"list_tools: {'ok' if report['tools_ok'] else 'FAIL'}")
        all_ok = report["initialize_ok"] and report["tools_ok"]
        for call in report["calls"]:
            status = "ok" if call["ok"] else "FAIL"
            print(f"call {call['name']}: {status}")
            all_ok = all_ok and call["ok"]
        return 0 if all_ok else 1

    print(
        "usage: hermes mcp fixtures <record|replay> ...",
        file=sys.stderr,
    )
    return 1
