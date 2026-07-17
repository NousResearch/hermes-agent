"""Tool handlers for osint-agent plugin."""

from __future__ import annotations

import json
from typing import Any

from . import computer_use_playbooks
from . import orchestrator
from . import plugin_loader


def check_available() -> bool:
    return True


STATUS_SCHEMA = {
    "name": "osint_agent_status",
    "description": (
        "Unified OSINT agent readiness: SitDeck, World Monitor Free, scrapling-feeds, "
        "Computer Use playbooks, web search stack (no World Monitor Pro MCP required)."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

BRIEF_SCHEMA = {
    "name": "osint_agent_brief",
    "description": (
        "Generate integrated OSINT brief: PDB + WM Free JSON + SitDeck crawl + "
        "government RSS + MHLW + multilayer web/CU collection plan."
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
            "include_wm_free": {"type": "boolean", "default": True},
            "llm_summary": {"type": "boolean", "default": False},
        },
        "required": [],
    },
}

CU_PLAN_SCHEMA = {
    "name": "osint_agent_computer_use_plan",
    "description": (
        "Return Computer Use playbooks to manually browse WorldMonitor UI and SitDeck "
        "(cua-driver). Use with toolset computer_use for live operator-style OSINT."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "enum": ["all", "worldmonitor", "sitdeck"],
                "description": "Which CU playbook(s) to return (default all).",
            },
            "topic": {"type": "string"},
        },
        "required": [],
    },
}

MULTILAYER_SCHEMA = {
    "name": "osint_agent_multilayer_collect",
    "description": (
        "Multi-layer OSINT collection plan + optional Free WM snapshot: "
        "WM Free JSON → gov RSS → SitDeck → Computer Use → web_search queries → Shinka."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "fetch_wm_free": {
                "type": "boolean",
                "default": True,
                "description": "Actually fetch World Monitor Free JSON now.",
            },
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional override web_search queries.",
            },
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
            "computer_use",
            "web",
        ],
        "worldmonitor_mcp": "disabled (use free tier + CU manual UI)",
        "sitdeck_credentials": sitdeck_cred,
        "computer_use": {
            "cli": "hermes computer-use doctor",
            "tool": "osint_agent_computer_use_plan",
        },
        "cli": "hermes osint-agent brief | stack enable | cron install",
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
        include_wm_free=bool(args.get("include_wm_free", True)),
        save=bool(args.get("save", True)),
    )
    if args.get("markdown_only"):
        return result.get("markdown") or ""
    return json.dumps(
        {k: v for k, v in result.items() if k != "pdb"},
        ensure_ascii=False,
        indent=2,
    )


def handle_computer_use_plan(args: dict[str, Any], **_: Any) -> str:
    topic = args.get("topic") or "日本の安全保障と世界情勢"
    target = (args.get("target") or "all").strip().lower()
    if target == "worldmonitor":
        payload = {
            "success": True,
            "playbook": computer_use_playbooks.worldmonitor_manual_playbook(),
        }
    elif target == "sitdeck":
        payload = {
            "success": True,
            "playbook": computer_use_playbooks.sitdeck_computer_use_playbook(),
        }
    else:
        payload = computer_use_playbooks.build_full_osint_playbook(topic=topic)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def handle_multilayer_collect(args: dict[str, Any], **_: Any) -> str:
    topic = args.get("topic") or "日本の安全保障と世界情勢"
    queries = args.get("queries")
    if isinstance(queries, str):
        queries = [q.strip() for q in queries.split("\n") if q.strip()]
    plan = computer_use_playbooks.multilayer_search_plan(
        topic=topic,
        queries=list(queries) if isinstance(queries, list) else None,
    )
    payload: dict[str, Any] = {
        "success": True,
        "plan": plan,
        "next": [
            "Execute L5 via web_search tool for each query",
            "Execute L4 via computer_use following osint_agent_computer_use_plan",
            "Merge notes into osint_agent_brief",
        ],
    }
    if bool(args.get("fetch_wm_free", True)):
        payload["wm_free"] = orchestrator._worldmonitor_free_section(focus="japan_security")
    return json.dumps(payload, ensure_ascii=False, indent=2)


def handle_slash(args: str) -> str:
    parts = (args or "").strip().split()
    sub = (parts[0] if parts else "status").lower()
    if sub in {"status", "st"}:
        return handle_status({})
    if sub in {"brief", "run", "digest"}:
        return handle_brief({"markdown_only": True})
    if sub in {"cu", "computer-use", "playbook"}:
        return handle_computer_use_plan({"target": "all"})
    if sub in {"multi", "multilayer", "layers"}:
        return handle_multilayer_collect({"fetch_wm_free": False})
    return "Usage: /osint-agent [status|brief|cu|multilayer]"


def run_brief_cli(**kwargs: Any) -> int:
    if kwargs.pop("cron_stdout", False):
        return orchestrator.run_for_cron_stdout(kwargs.pop("slot", "morning"), **kwargs)
    result = orchestrator.generate_integrated_brief(**kwargs)
    print(result.get("markdown") or json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") else 1
