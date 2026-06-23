"""Core World Monitor OSINT plugin — real-time feeds + Shinka fusion."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from . import api
from . import auth_setup
from . import free_web

STATUS_SCHEMA = {
    "name": "worldmonitor_status",
    "description": "Show World Monitor API connectivity and OSINT stack readiness.",
    "parameters": {"type": "object", "properties": {}},
}

SNAPSHOT_SCHEMA = {
    "name": "worldmonitor_snapshot",
    "description": (
        "Fetch a real-time Japan-security snapshot from World Monitor "
        "(country risk, regional brief, news digest, risk scores)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "country_code": {
                "type": "string",
                "description": "ISO country code (default JP).",
            },
            "region_id": {
                "type": "string",
                "description": "Regional brief id (default east-asia).",
            },
            "news_lang": {
                "type": "string",
                "description": "News digest language (default en).",
            },
            "news_limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
            },
        },
    },
}

COUNTRY_BRIEF_SCHEMA = {
    "name": "worldmonitor_country_brief",
    "description": "Get World Monitor strategic intel brief for one country.",
    "parameters": {
        "type": "object",
        "properties": {
            "country_code": {
                "type": "string",
                "description": "ISO 3166-1 alpha-2 code (e.g. JP, US, CN).",
            },
            "framework": {
                "type": "string",
                "description": "Optional analytical framework (max 2000 chars).",
            },
        },
        "required": ["country_code"],
    },
}

FREE_CRAWL_SCHEMA = {
    "name": "worldmonitor_free_crawl",
    "description": (
        "Collect World Monitor Free-tier OSINT via public web JSON "
        "(news digest, GPS jamming map, alerts) without Pro OAuth or wm_ key."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "focus": {
                "type": "string",
                "description": "Collection focus (default japan_security).",
            },
            "news_lang": {
                "type": "string",
                "description": "News digest language (default en).",
            },
            "news_limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
            },
            "include_shell": {
                "type": "boolean",
                "description": "Crawl worldmonitor.app HTML metadata (default true).",
            },
        },
    },
}

FUSION_SCHEMA = {
    "name": "worldmonitor_fusion_report",
    "description": (
        "Fusion OSINT report: World Monitor real-time snapshot + ShinkaEvolve MILSPEC "
        "briefing (evolution scoring). Use egov-law MCP tools for Japanese primary law citations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Briefing topic (e.g. 日本の安全保障, 台湾有事, 中東情勢).",
            },
            "domain": {
                "type": "string",
                "description": "Shinka domain shortcut (taiwan, middle_east, cyber_defense, ...).",
            },
            "country_code": {
                "type": "string",
                "description": "World Monitor country focus (default JP).",
            },
            "max_scenarios": {
                "type": "integer",
                "minimum": 1,
                "maximum": 8,
            },
            "source_mode": {
                "type": "string",
                "enum": ["mock", "real"],
                "description": "Shinka source mode; use real for live primary-source retrieval.",
            },
            "save_report": {
                "type": "boolean",
                "description": "Save fusion JSON under ~/.hermes/worldmonitor-osint/reports/.",
            },
            "wm_tier": {
                "type": "string",
                "enum": ["auto", "free", "pro"],
                "description": "World Monitor data tier: auto (sidecar/key else Free web), free, pro.",
            },
            "llm_summary": {
                "type": "boolean",
                "description": (
                    "Add Shinka executive summary via Hermes LLM "
                    "(GPT Auth / NVIDIA / Nous / xAI — not google-generativeai)."
                ),
            },
        },
    },
}

EGOV_MCP_TOOLS = [
    "search_laws",
    "get_law_article",
    "get_law_full_text",
    "keyword_search",
    "list_law_types",
]


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _reports_dir() -> Path:
    path = get_hermes_home() / "worldmonitor-osint" / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_shinka_core():
  """Load shinka-osint core with package context for relative imports."""
  import importlib.util
  import sys

  shinka_dir = Path(__file__).resolve().parent.parent / "shinka-osint"
  pkg = "hermes_fusion_shinka_osint"
  cached = sys.modules.get(f"{pkg}.core")
  if cached is not None:
      return cached

  if pkg not in sys.modules:
      pkg_mod = importlib.util.module_from_spec(
          importlib.util.spec_from_file_location(pkg, shinka_dir / "__init__.py")
      )
      pkg_mod.__path__ = [str(shinka_dir)]  # type: ignore[attr-defined]
      sys.modules[pkg] = pkg_mod

  for sub in ("providers", "bridge", "core"):
      full = f"{pkg}.{sub}"
      if full in sys.modules:
          continue
      spec = importlib.util.spec_from_file_location(
          full,
          shinka_dir / f"{sub}.py",
          submodule_search_locations=[str(shinka_dir)],
      )
      if spec is None or spec.loader is None:
          raise ImportError(f"Cannot load {full}")
      mod = importlib.util.module_from_spec(spec)
      mod.__package__ = pkg
      sys.modules[full] = mod
      spec.loader.exec_module(mod)

  return sys.modules[f"{pkg}.core"]


def check_available() -> bool:
    """Plugin is always loadable; API may still need credentials."""
    return True


def status() -> dict[str, Any]:
    conn = api.connectivity_status()
    sidecar = auth_setup.probe_sidecar()
    dev = {}
    try:
        from . import dev_server

        dev = dev_server.dev_status(probe=True)
    except Exception as exc:
        dev = {"error": str(exc)}
    mcp_oauth = auth_setup._mcp_oauth_configured()
    shinka_ready = False
    shinka_detail: dict[str, Any] = {}
    try:
        shinka = _load_shinka_core()
        shinka_detail = shinka.status()
        shinka_ready = bool(shinka_detail.get("available"))
    except Exception as exc:
        shinka_detail = {"error": str(exc)}

    from hermes_cli.mcp_config import _get_mcp_servers

    mcp_servers = _get_mcp_servers()
    egov_configured = "egov-law" in mcp_servers
    free_probe = free_web.probe_free_tier()

    return {
        "success": True,
        "worldmonitor": conn,
        "sidecar": sidecar,
        "dev_server": dev,
        "mcp_oauth": mcp_oauth,
        "free_web": free_probe,
        "auth_guidance": auth_setup.auth_guidance(),
        "shinka_osint": shinka_detail,
        "shinka_available": shinka_ready,
        "egov_law_mcp_configured": egov_configured,
        "egov_primary_source_tools": EGOV_MCP_TOOLS,
        "fusion_ready": shinka_ready
        and (
            conn.get("api_key_configured")
            or conn.get("local_sidecar")
            or conn.get("local_dev")
            or sidecar.get("running")
            or (dev.get("dev_server") or {}).get("running")
            or mcp_oauth.get("configured")
            or free_probe.get("available")
        ),
    }


def snapshot(
    *,
    country_code: str = "JP",
    region_id: str = "east-asia",
    news_lang: str = "en",
    news_limit: int = 12,
    tier_mode: str = "auto",
) -> dict[str, Any]:
    mode = (tier_mode or "auto").strip().lower()
    if mode == "free":
        return free_web.free_snapshot(
            focus="japan_security" if country_code.upper() == "JP" else "general",
            news_lang=news_lang,
            news_limit=news_limit,
        )

    code = (country_code or "JP").upper()
    if code == "JP" and region_id == "east-asia":
        return api.snapshot_japan_security(news_lang=news_lang, news_limit=news_limit)

    conn = api.connectivity_status()
    has_paid = bool(
        conn.get("api_key_configured") or conn.get("local_sidecar") or conn.get("local_dev")
    )
    if mode == "auto" and not has_paid:
        snap = free_web.free_snapshot(
            focus="general",
            news_lang=news_lang,
            news_limit=news_limit,
            include_shell=False,
        )
        snap["country_code"] = code
        snap["region_id"] = region_id
        return snap

    out: dict[str, Any] = {
        "success": True,
        "country_code": code,
        "region_id": region_id,
        "api": api.connectivity_status(),
        "sections": {},
        "errors": [],
    }
    for key, fn in (
        ("country_risk", lambda: api.get_country_risk(code)),
        ("country_intel_brief", lambda: api.get_country_intel_brief(code)),
        ("regional_brief", lambda: api.get_regional_brief(region_id)),
        ("risk_scores", lambda: api.get_risk_scores(region_id)),
        ("news_digest", lambda: api.list_feed_digest(lang=news_lang)),
    ):
        try:
            out["sections"][key] = fn()
        except Exception as exc:
            out["errors"].append({"section": key, "error": str(exc)})
    out["success"] = bool(out["sections"])
    return out


def free_crawl(
    *,
    focus: str = "japan_security",
    news_lang: str = "en",
    news_limit: int = 20,
    include_shell: bool = True,
) -> dict[str, Any]:
    return free_web.free_snapshot(
        focus=focus or "japan_security",
        news_lang=news_lang or "en",
        news_limit=int(news_limit or 20),
        include_shell=bool(include_shell),
    )


def country_brief(country_code: str, framework: str = "") -> dict[str, Any]:
    code = (country_code or "").strip().upper()
    if not code:
        return {"success": False, "error": "country_code is required"}
    try:
        data = api.get_country_intel_brief(code, framework=framework)
        return {"success": True, "country_code": code, "brief": data}
    except Exception as exc:
        return {"success": False, "country_code": code, "error": str(exc)}


def fusion_report(
    *,
    topic: str = "日本の安全保障と世界情勢",
    domain: str = "",
    country_code: str = "JP",
    max_scenarios: int = 3,
    source_mode: str = "real",
    save_report: bool = False,
    wm_tier: str = "auto",
    llm_summary: bool = False,
) -> dict[str, Any]:
    wm = snapshot(country_code=country_code, tier_mode=wm_tier)
    shinka_block: dict[str, Any] = {"success": False}
    shinka_llm: dict[str, Any] = {}
    try:
        shinka = _load_shinka_core()
        example = shinka.bridge.resolve_default_example()
        shinka_block = shinka.briefing(
            topic=topic,
            domain=domain,
            max_scenarios=max_scenarios,
            example=example,
            source_mode=source_mode,
            save_report=False,
            llm_summary=llm_summary,
        )
        if hasattr(shinka, "providers"):
            shinka_llm = shinka.providers.provider_status()
    except Exception as exc:
        shinka_block = {"success": False, "error": str(exc)}

    payload = {
        "success": wm.get("success") or shinka_block.get("success"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "topic": topic,
        "domain": domain or None,
        "country_code": country_code.upper(),
        "source_mode": source_mode,
        "worldmonitor": wm,
        "shinka_milspec": shinka_block,
        "shinka_llm": shinka_llm,
        "primary_sources": {
            "egov_law_mcp": {
                "configured_hint": "Use MCP server `egov-law` when installed via `hermes mcp install egov-law`.",
                "recommended_tools": EGOV_MCP_TOOLS,
                "usage": (
                    "For Japanese legal primary sources, call egov-law MCP tools "
                    "(search_laws, get_law_article) and cite article numbers in Evidence Blocks."
                ),
            },
            "shinka_source_mode": source_mode,
        },
        "methodology": (
            "Fusion combines World Monitor real-time risk/news (koala73/worldmonitor) with "
            "ShinkaEvolve-OSINT MILSPEC scoring (rule-based). "
            "PDB reports enforce primary-source discipline: government/treaty sources preferred; "
            "media headlines require [出典: URL] and verification. "
            "Optional LLM summary uses Hermes auth with MILSPEC citation rules."
        ),
    }

    if save_report:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = re.sub(r"[^\w\-]+", "_", (topic or "fusion")[:40]).strip("_") or "fusion"
        out = _reports_dir() / f"{stamp}_{slug}.json"
        out.write_text(_json(payload), encoding="utf-8")
        payload["saved_report"] = str(out)

    return payload


def handle_status(args: dict[str, Any], **_: Any) -> str:
    return _json(status())


def handle_snapshot(args: dict[str, Any], **_: Any) -> str:
    return _json(
        snapshot(
            country_code=args.get("country_code") or "JP",
            region_id=args.get("region_id") or "east-asia",
            news_lang=args.get("news_lang") or "en",
            news_limit=int(args.get("news_limit") or 12),
            tier_mode=args.get("tier_mode") or "auto",
        )
    )


def handle_country_brief(args: dict[str, Any], **_: Any) -> str:
    return _json(
        country_brief(
            args.get("country_code") or "",
            framework=args.get("framework") or "",
        )
    )


def handle_free_crawl(args: dict[str, Any], **_: Any) -> str:
    return _json(
        free_crawl(
            focus=args.get("focus") or "japan_security",
            news_lang=args.get("news_lang") or "en",
            news_limit=int(args.get("news_limit") or 20),
            include_shell=bool(args.get("include_shell", True)),
        )
    )


def handle_fusion_report(args: dict[str, Any], **_: Any) -> str:
    return _json(
        fusion_report(
            topic=args.get("topic") or "日本の安全保障と世界情勢",
            domain=args.get("domain") or "",
            country_code=args.get("country_code") or "JP",
            max_scenarios=int(args.get("max_scenarios") or 3),
            source_mode=args.get("source_mode") or "real",
            save_report=bool(args.get("save_report")),
            wm_tier=args.get("wm_tier") or "auto",
            llm_summary=bool(args.get("llm_summary")),
        )
    )


def handle_slash(cmd: str) -> str:
    parts = (cmd or "").strip().split()
    sub = parts[1].lower() if len(parts) > 1 else "status"
    if sub == "status":
        return handle_status({})
    if sub == "snapshot":
        return handle_snapshot({})
    if sub == "free":
        return handle_free_crawl({})
    if sub == "fusion":
        topic = " ".join(parts[2:]).strip() or "日本の安全保障"
        return handle_fusion_report({"topic": topic, "source_mode": "real"})
    return _json(
        {
            "success": False,
            "error": "Usage: /worldmonitor-osint [status|snapshot|free|fusion <topic>]",
        }
    )
