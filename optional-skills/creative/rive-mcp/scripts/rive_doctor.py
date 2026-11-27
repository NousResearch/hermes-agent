#!/usr/bin/env python3
"""Preflight the official local Rive MCP endpoint; never install or launch it.

Default: stdlib TCP probe only (not proof of MCP). --handshake: explicitly opt
into SDK initialization and paginated tool discovery, never tools/call.
Exit codes: 0 = handshake/catalog verified, 1 = closed/failed, 2 = TCP only.
A verified handshake does not establish file identity, authoring, or export.
"""
# Original doctor: Brooklyn Nicholson. Guarded diagnostics: Chris (cygnostik),
# ProDyn https://prodyn.ai. MIT. No telemetry or third-party launch paths.
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket


RIVE_HOST = "127.0.0.1"
RIVE_PORT = 9791


def _port_open(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def _handshake():
    import rive_mcp

    async with rive_mcp.sdk_connection() as session:
        await session.initialize()
        return await rive_mcp.all_tools(session)


def check(*, handshake: bool = False) -> dict:
    tcp = _port_open(RIVE_HOST, RIVE_PORT)
    official = {
        "url": f"http://{RIVE_HOST}:{RIVE_PORT}/mcp",
        "status": "tcp-only" if tcp else "closed",
        "tcp_reachable": tcp,
        "handshake_verified": False,
        "active_file_checked": False,
        "scope": "TCP connectivity only; MCP, file identity, authoring and export not verified.",
    }
    if not tcp:
        official["detail"] = "No TCP connection; open the authorized Rive desktop editor and enable its MCP server."
    elif not handshake:
        official["detail"] = "A listener exists; use --handshake for an SDK MCP check."
    else:
        import rive_mcp

        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            # Fail before networking with a useful dependency message, not an
            # opaque nested async exception or raw server/credential text.
            rive_mcp.require_dependencies()
            tools = asyncio.run(_handshake())
        except rive_mcp.CliError as exc:
            official.update(status="handshake-failed", detail=str(exc))
        except Exception:
            official.update(status="handshake-failed", detail="MCP/transport/protocol check failed; no tool was called and nothing was launched.")
        else:
            official.update(status="mcp-verified", handshake_verified=True, tool_count=len(tools),
                            scope="MCP initialization and tool catalog verified; file identity, authoring and export not verified.")
        finally:
            logging.disable(previous)
    return {"official_rive": official}


def _summary(status: dict) -> str:
    official = status["official_rive"]
    lines = [f"{official['status']}: {official['url']}", official["scope"]]
    if "detail" in official:
        lines.append(official["detail"])
    if "tool_count" in official:
        lines.append(f"Discovered tools: {official['tool_count']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument("--handshake", action="store_true", help="Opt into SDK initialization and tool listing; no tool calls")
    args = parser.parse_args(argv)
    status = check(handshake=args.handshake)
    print(json.dumps(status, indent=2) if args.json else _summary(status))
    return {"mcp-verified": 0, "tcp-only": 2}.get(status["official_rive"]["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
