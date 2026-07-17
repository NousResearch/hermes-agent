"""Computer Use playbooks — WorldMonitor manual UI + SitDeck dashboard.

Hermes Agent executes these via the ``computer_use`` toolset (cua-driver).
Automated Playwright crawl remains available for SitDeck; CU is for live
operator-style inspection when SPA state / maps need eyes-on review.
"""

from __future__ import annotations

from typing import Any

WORLDMONITOR_URL = "https://worldmonitor.app/"
WORLDMONITOR_DOCS = "https://www.worldmonitor.app/docs/usage-auth"
SITDECK_APP_URL = "https://app.sitdeck.com/"
SITDECK_LOGIN_URL = "https://app.sitdeck.com/#login"
SITDECK_PULSE_URL = "https://sitdeck.com/global-pulse"


def worldmonitor_manual_playbook(*, focus: str = "japan_security") -> dict[str, Any]:
    """Steps for Computer Use on the public World Monitor web UI (Free tier)."""
    return {
        "target": "worldmonitor",
        "mode": "computer_use_manual",
        "url": WORLDMONITOR_URL,
        "focus": focus,
        "prerequisites": [
            "hermes computer-use install && hermes computer-use doctor",
            "Enable toolset: hermes tools enable computer_use",
            "Prefer Free web JSON via worldmonitor_free_crawl for bulk data; use CU for map/UI verification",
        ],
        "safety": [
            "Do not enter paid checkout unless user explicitly asks",
            "Do not store screenshots with PII in public channels",
            "Prefer read-only navigation; no account creation",
        ],
        "steps": [
            {
                "id": "wm1",
                "action": "open",
                "computer_use": {
                    "action": "open_url",
                    "url": WORLDMONITOR_URL,
                },
                "observe": "Capture window title and primary nav labels",
            },
            {
                "id": "wm2",
                "action": "inspect_digest",
                "hint": "Open news / digest panels; note Japan, Taiwan, Middle East, cyber headlines",
                "observe": "List top 5 visible headlines with timestamps if shown",
            },
            {
                "id": "wm3",
                "action": "inspect_alerts",
                "hint": "Check GPS jam / regional alert widgets if visible on Free UI",
                "observe": "Record any active alert badges and geographies",
            },
            {
                "id": "wm4",
                "action": "cross_check",
                "hint": f"Relate UI findings to topic focus={focus}; flag Pro-only locked panels",
                "observe": "Separate Free-visible vs Pro-gated content",
            },
        ],
        "agent_instruction": (
            "Use computer_use to follow steps wm1–wm4 on worldmonitor.app. "
            "Then call worldmonitor_free_crawl / osint_agent_multilayer_collect "
            "to attach machine-readable Free JSON. Never invent EvidenceBlock IDs."
        ),
    }


def sitdeck_computer_use_playbook(*, use_saved_session: bool = True) -> dict[str, Any]:
    """Steps for Computer Use on SitDeck (login may use env credentials)."""
    return {
        "target": "sitdeck",
        "mode": "computer_use_manual",
        "login_url": SITDECK_LOGIN_URL,
        "app_url": SITDECK_APP_URL,
        "pulse_url": SITDECK_PULSE_URL,
        "use_saved_session": use_saved_session,
        "prerequisites": [
            "SITDECK_EMAIL / SITDECK_PASSWORD in ~/.hermes/.env (or public pulse only)",
            "hermes computer-use doctor",
            "Optional faster path: sitdeck crawl via Playwright (osint_agent_brief)",
        ],
        "safety": [
            "Never echo password into chat or screenshots OCR notes",
            "Prefer storage_state / existing session if already logged in",
            "Stay on sitdeck.com / app.sitdeck.com origins",
        ],
        "steps": [
            {
                "id": "sd0",
                "action": "public_pulse",
                "computer_use": {"action": "open_url", "url": SITDECK_PULSE_URL},
                "observe": "Capture Global Pulse public summary without login",
            },
            {
                "id": "sd1",
                "action": "open_app",
                "computer_use": {"action": "open_url", "url": SITDECK_APP_URL},
                "observe": "Detect login wall vs dashboard widgets",
            },
            {
                "id": "sd2",
                "action": "login_if_needed",
                "hint": (
                    "If login form: type email from SITDECK_EMAIL then password from env "
                    "(do not print values). Submit and wait for dashboard."
                ),
                "observe": "Confirm deck / briefing widgets loaded",
            },
            {
                "id": "sd3",
                "action": "scan_widgets",
                "hint": "Scroll situation deck; note alerts, regions, timestamps",
                "observe": "Bullet list of visible OSINT cards (title + region + time)",
            },
            {
                "id": "sd4",
                "action": "export_notes",
                "hint": "Summarize for MILSPEC brief: KJ candidates + open sources only",
                "observe": "Return structured notes for osint_agent_brief merge",
            },
        ],
        "agent_instruction": (
            "Prefer automated sitdeck-osint crawl when credentials exist. "
            "Use computer_use playbook when SPA widgets fail headless crawl or "
            "operator needs live verification. Redact secrets in all outputs."
        ),
    }


def multilayer_search_plan(*, topic: str, queries: list[str] | None = None) -> dict[str, Any]:
    """Ordered multi-layer OSINT collection plan (API + CU + web_search)."""
    topic = (topic or "日本の安全保障と世界情勢").strip()
    default_queries = [
        f"{topic} 公式発表",
        f"{topic} site:go.jp OR site:mod.go.jp OR site:mofa.go.jp",
        f"{topic} Taiwan OR Hormuz OR cyber OR AI",
        f"{topic} Reuters OR AP OR Kyodo primary",
    ]
    q = [x for x in (queries or default_queries) if x.strip()]
    return {
        "topic": topic,
        "layers": [
            {
                "id": "L1_worldmonitor_free",
                "tool": "worldmonitor_free_crawl / free_snapshot",
                "why": "Machine JSON from worldmonitor.app Free tier",
            },
            {
                "id": "L2_gov_rss",
                "tool": "scrapling-feeds / worldmonitor primary_backfill gov feeds",
                "why": "Allowlisted government primary sources",
            },
            {
                "id": "L3_sitdeck",
                "tool": "sitdeck crawl OR computer_use sitdeck playbook",
                "why": "Dashboard situation cards (authenticated)",
            },
            {
                "id": "L4_computer_use_wm",
                "tool": "computer_use + worldmonitor_manual_playbook",
                "why": "Human-in-the-loop / map UI verification on worldmonitor.app",
            },
            {
                "id": "L5_web_search",
                "tool": "web_search (toolset=web)",
                "queries": q,
                "why": "Diversify open-web corroboration; cite URLs with access date",
            },
            {
                "id": "L6_shinka_fusion",
                "tool": "shinka / MoA deepresearch (optional)",
                "why": "MILSPEC verifier + EvidenceBlock discipline when evolving reports",
            },
        ],
        "agent_instruction": (
            "Execute layers in order. Fail soft on a layer; never fabricate sources. "
            "For L5 call web_search once per query (limit results). "
            "Merge into osint_agent_brief markdown with clear section headers."
        ),
    }


def build_full_osint_playbook(*, topic: str = "日本の安全保障と世界情勢") -> dict[str, Any]:
    return {
        "success": True,
        "role": "Hermes OSINT Agent",
        "topic": topic,
        "computer_use": {
            "worldmonitor": worldmonitor_manual_playbook(focus="japan_security"),
            "sitdeck": sitdeck_computer_use_playbook(),
        },
        "multilayer": multilayer_search_plan(topic=topic),
        "enable": [
            "hermes tools enable osint_agent",
            "hermes tools enable worldmonitor_osint",
            "hermes tools enable sitdeck_osint",
            "hermes tools enable web",
            "hermes tools enable computer_use",
            "hermes osint-agent stack enable",
        ],
    }
