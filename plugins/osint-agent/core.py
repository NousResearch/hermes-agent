"""Tool handlers for osint-agent plugin."""

from __future__ import annotations

import json
from typing import Any

from . import orchestrator
from . import plugin_loader


def check_available() -> bool:
    return True


STATUS_SCHEMA = {
    "name": "osint_agent_status",
    "description": (
        "Unified OSINT agent readiness: SitDeck, World Monitor Free, scrapling-feeds, "
        "shinka-osint stack (no World Monitor Pro MCP)."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

BRIEF_SCHEMA = {
    "name": "osint_agent_brief",
    "description": (
        "Generate integrated OSINT brief: PDB (WM Free + Shinka) + SitDeck crawl + "
        "government RSS + MHLW designated-substances monitor."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "slot": {
                "type": "string",
                "enum": ["morning", "evening"],
                "description": "Briefing slot label (default morning).",
            },
            "topic": {
                "type": "string",
                "description": "PDB topic (default 日本の安全保障と世界情勢).",
            },
            "wm_tier": {
                "type": "string",
                "enum": ["auto", "free", "pro"],
                "description": "World Monitor tier (default free).",
            },
            "include_sitdeck": {"type": "boolean", "default": True},
            "include_mhlw": {"type": "boolean", "default": True},
            "llm_summary": {"type": "boolean", "default": False},
        },
        "required": [],
    },
}


def handle_status(_args: dict[str, Any], **_: Any) -> str:
    try:
        plugin_loader.load_plugin_modules("sitdeck-osint", ("credentials",))
        creds_mod = plugin_loader.get_module(
            "sitdeck-osint", "credentials", stems_chain=("credentials",)
        )
        sitdeck_cred = creds_mod.credential_status()
    except Exception as exc:
        sitdeck_cred = {"error": str(exc)[:120]}
    payload = {
        "success": True,
        "stack": [
            "sitdeck-osint",
            "worldmonitor-osint",
            "scrapling-feeds",
            "shinka-osint",
        ],
        "worldmonitor_mcp": "disabled (use free tier)",
        "sitdeck_credentials": sitdeck_cred,
        "cli": "hermes osint-agent brief | cron install",
        "reports_dir": str(orchestrator._reports_dir()),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def handle_brief(args: dict[str, Any], **_: Any) -> str:
    result = orchestrator.generate_integrated_brief(
        slot=args.get("slot") or "morning",
        topic=args.get("topic") or "日本の安全保障と世界情勢",
        source_mode="real",
        wm_tier=args.get("wm_tier") or "free",
        llm_summary=bool(args.get("llm_summary")),
        include_sitdeck=bool(args.get("include_sitdeck", True)),
        include_mhlw=bool(args.get("include_mhlw", True)),
        save=bool(args.get("save", True)),
    )
    if args.get("markdown_only"):
        return result.get("markdown") or ""
    return json.dumps(
        {k: v for k, v in result.items() if k != "pdb"},
        ensure_ascii=False,
        indent=2,
    )


def handle_slash(args: str) -> str:
    parts = (args or "").strip().split()
    sub = (parts[0] if parts else "status").lower()
    if sub in {"status", "st"}:
        return handle_status({})
    if sub in {"brief", "run", "digest"}:
        return handle_brief({"markdown_only": True})
    return "Usage: /osint-agent [status|brief]"


def run_brief_cli(**kwargs: Any) -> int:
    if kwargs.pop("cron_stdout", False):
        return orchestrator.run_for_cron_stdout(kwargs.pop("slot", "morning"), **kwargs)
    result = orchestrator.generate_integrated_brief(**kwargs)
    print(result.get("markdown") or json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1
