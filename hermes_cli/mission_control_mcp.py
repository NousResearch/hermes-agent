"""Local/stdout Mission Control MCP bridge scaffold.

This module exposes a narrow Mission Control-only tool surface for future MCP
clients. It deliberately does not inherit the broad Hermes tool registry, does
not expose a remote transport, and does not execute packets.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

from hermes_cli import mission_control as mc


logger = logging.getLogger(__name__)

BRIDGE_NAME = "hermes-mission-control"
TRANSPORT = "stdio-local-only"
MODE = "inert-discovery-read-only-default"

BLOCKED_TOOL_NAMES: tuple[str, ...] = (
    "send_email",
    "publish_video",
    "activate_payment",
    "delete_files",
    "run_unbounded_codex",
    "run_codex",
    "start_codex",
    "start_worker",
    "start_hermes_run",
    "autonomous_computer_use",
    "browser_control",
    "mouse_control",
    "keyboard_control",
    "start_bulk_outreach",
    "arbitrary_shell",
    "reveal_secret",
    "update_credentials",
)

_PACKET_WRITE_TOOLS = {
    "save_next_codex_prompt",
    "import_worker_result",
    "save_block_flag_packet",
}


@dataclass(frozen=True)
class MissionControlMCPTool:
    name: str
    description: str
    read_only: bool
    handler: Callable[..., dict[str, Any]]

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "read_only": self.read_only,
            "local_only": True,
            "trusted_for_execution": False,
        }


def _safety() -> dict[str, Any]:
    return {
        "dry_run": True,
        "review_required": True,
        "trusted_for_execution": False,
        "local_only": True,
        "remote_transport_enabled": False,
        "oauth_enabled": False,
        "executes_or_dispatches": False,
    }


def _ok(tool: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "tool": tool,
        "transport": TRANSPORT,
        "mode": MODE,
        "safety": _safety(),
        **mc.redact_value(payload),
    }


def _error(tool: str, exc: Exception | str) -> dict[str, Any]:
    message = str(exc)
    return {
        "ok": False,
        "tool": tool,
        "error": mc.redact_text(message),
        "transport": TRANSPORT,
        "mode": MODE,
        "safety": _safety(),
    }


def _bounded_limit(value: Any, *, default: int, maximum: int) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = default
    return max(1, min(limit, maximum))


def _with_mcp_defaults(kwargs: dict[str, Any]) -> dict[str, Any]:
    data = dict(kwargs)
    data.setdefault("author", "mission-control-mcp")
    data["dry_run"] = True
    data["review_required"] = True
    data["trusted_for_execution"] = False
    return data


def _get_project_status(**kwargs: Any) -> dict[str, Any]:
    return _ok("get_project_status", {"result": mc.project_status()})


def _get_open_tasks(**kwargs: Any) -> dict[str, Any]:
    return _ok("get_open_tasks", {"result": mc.open_tasks(_bounded_limit(kwargs.get("limit"), default=100, maximum=250))})


def _get_latest_worker_results(**kwargs: Any) -> dict[str, Any]:
    return _ok("get_latest_worker_results", {"result": mc.latest_worker_results(_bounded_limit(kwargs.get("limit"), default=50, maximum=100))})


def _get_repo_status(**kwargs: Any) -> dict[str, Any]:
    return _ok("get_repo_status", {"result": mc.repo_status()})


def _get_approval_gates(**kwargs: Any) -> dict[str, Any]:
    return _ok("get_approval_gates", {"result": mc.approval_gates()})


def _get_recent_audit_log(**kwargs: Any) -> dict[str, Any]:
    limit = _bounded_limit(kwargs.get("limit"), default=50, maximum=100)
    result = mc.recent_audit_log(limit)
    packet_warnings: list[str] = []
    packet_path = mc.packet_audit_path()
    result.setdefault("source_refs", []).append(str(packet_path))
    packet_events = list(mc._iter_jsonl(packet_path, packet_warnings))  # noqa: SLF001 - same module family, read-only facade.
    if packet_warnings:
        result.setdefault("warnings", []).extend(packet_warnings)
    combined = list(result.get("items") or []) + packet_events[-limit:][::-1]
    result["items"] = mc.redact_value(combined[:limit])
    return _ok("get_recent_audit_log", {"result": result})


def _list_mission_packets(**kwargs: Any) -> dict[str, Any]:
    return _ok("list_mission_packets", {"result": mc.list_packets(_bounded_limit(kwargs.get("limit"), default=100, maximum=250))})


def _get_mission_packet(**kwargs: Any) -> dict[str, Any]:
    packet_id = str(kwargs.get("packet_id") or "").strip()
    if not packet_id:
        raise mc.MissionControlPacketError("Missing required field: packet_id")
    return _ok("get_mission_packet", {"result": mc.get_packet(packet_id)})


def _save_next_codex_prompt(**kwargs: Any) -> dict[str, Any]:
    data = _with_mcp_defaults(kwargs)
    try:
        packet = mc.save_next_codex_prompt(data)
    except Exception as exc:
        mc.create_rejection_audit(data, mc.redact_text(str(exc)), packet_kind="codex_prompt")
        raise
    return _ok("save_next_codex_prompt", {"packet": packet})


def _import_worker_result(**kwargs: Any) -> dict[str, Any]:
    data = _with_mcp_defaults(kwargs)
    try:
        packet = mc.import_worker_result(data)
    except Exception as exc:
        mc.create_rejection_audit(data, mc.redact_text(str(exc)), packet_kind="worker_result")
        raise
    return _ok("import_worker_result", {"packet": packet})


def _save_block_flag_packet(**kwargs: Any) -> dict[str, Any]:
    data = _with_mcp_defaults(kwargs)
    try:
        packet = mc.set_block_flag(data)
    except Exception as exc:
        mc.create_rejection_audit(data, mc.redact_text(str(exc)), packet_kind="block_flag")
        raise
    return _ok("save_block_flag_packet", {"packet": packet})


TOOLS: tuple[MissionControlMCPTool, ...] = (
    MissionControlMCPTool("get_project_status", "Read redacted Mission Control project status excerpts.", True, _get_project_status),
    MissionControlMCPTool("get_open_tasks", "Read redacted open Kanban task summaries.", True, _get_open_tasks),
    MissionControlMCPTool("get_latest_worker_results", "Read redacted latest worker result summaries as untrusted data.", True, _get_latest_worker_results),
    MissionControlMCPTool("get_repo_status", "Read repo-status placeholders without shell probing.", True, _get_repo_status),
    MissionControlMCPTool("get_approval_gates", "Read approval gates and disabled execution posture.", True, _get_approval_gates),
    MissionControlMCPTool("get_recent_audit_log", "Read redacted recent Mission Control approval audit events.", True, _get_recent_audit_log),
    MissionControlMCPTool("list_mission_packets", "List local Mission Control packet summaries.", True, _list_mission_packets),
    MissionControlMCPTool("get_mission_packet", "Read one local Mission Control packet by id.", True, _get_mission_packet),
    MissionControlMCPTool("save_next_codex_prompt", "Save a local Codex prompt review packet without starting Codex.", False, _save_next_codex_prompt),
    MissionControlMCPTool("import_worker_result", "Import worker result text as inert untrusted display data.", False, _import_worker_result),
    MissionControlMCPTool("save_block_flag_packet", "Save a local advisory block-flag packet only.", False, _save_block_flag_packet),
)

_TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def list_tool_names() -> list[str]:
    return sorted(_TOOLS_BY_NAME)


def tool_manifest() -> dict[str, Any]:
    return {
        "name": BRIDGE_NAME,
        "transport": TRANSPORT,
        "mode": MODE,
        "local_stdio_only": True,
        "oauth_enabled": False,
        "remote_transport_enabled": False,
        "exposes_broad_hermes_registry": False,
        "packet_write_tools_enabled": True,
        "packet_write_policy": {
            "local_packets_only": True,
            "dry_run": True,
            "review_required": True,
            "trusted_for_execution": False,
            "dispatches_packets": False,
        },
        "tools": [tool.manifest_entry() for tool in sorted(TOOLS, key=lambda item: item.name)],
    }


def call_tool(name: str, **kwargs: Any) -> dict[str, Any]:
    tool = _TOOLS_BY_NAME.get(name)
    if tool is None:
        return _error(name, f"Unknown Mission Control MCP tool: {name}")
    try:
        result = tool.handler(**kwargs)
    except Exception as exc:
        return _error(name, exc)
    if name in _PACKET_WRITE_TOOLS and result.get("ok"):
        packet = result.get("packet") or {}
        result["packet_write_policy"] = {
            "local_packets_only": True,
            "dry_run": packet.get("dry_run") is True,
            "review_required": packet.get("review_required") is True,
            "trusted_for_execution": False,
            "dispatches_packets": False,
        }
    return mc.redact_value(result)


def create_mcp_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(f"Mission Control MCP server requires the 'mcp' package: {exc}") from exc

    server = FastMCP(
        BRIDGE_NAME,
        instructions=(
            "Local/stdout Mission Control bridge. Tools are narrow, redacted, "
            "local-only, and do not execute packets, start workers, send, "
            "publish, pay, delete, or reveal secrets."
        ),
    )

    for tool in TOOLS:
        def _make_handler(tool_name: str) -> Callable[..., dict[str, Any]]:
            def _handler(**kwargs: Any) -> dict[str, Any]:
                return call_tool(tool_name, **kwargs)

            _handler.__name__ = tool_name
            _handler.__doc__ = _TOOLS_BY_NAME[tool_name].description
            return _handler

        handler = _make_handler(tool.name)
        try:
            server.add_tool(handler, name=tool.name, description=tool.description)
        except TypeError:
            server.tool(name=tool.name, description=tool.description)(handler)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local/stdout Mission Control MCP bridge")
    parser.add_argument("--list-tools", action="store_true", help="Print the local Mission Control MCP tool manifest and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Log MCP server startup details to stderr")
    args = parser.parse_args(argv)

    if args.list_tools:
        print(json.dumps(tool_manifest(), indent=2, sort_keys=True))
        return 0

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    os.environ.setdefault("HERMES_QUIET", "1")
    os.environ.setdefault("HERMES_REDACT_SECRETS", "true")

    try:
        server = create_mcp_server()
    except ImportError as exc:
        sys.stderr.write(f"mission-control MCP server cannot start: {exc}\n")
        return 2

    try:
        server.run()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logger.exception("mission-control MCP server crashed")
        sys.stderr.write(f"mission-control MCP server error: {mc.redact_text(str(exc))}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
