"""Browser UI server for the Hermes Context Cockpit.

Read-only localhost server. Binds only to 127.0.0.1 by default and reuses the
existing metrics/status pipeline. Visual surface is instrument-first (gauges,
heartbeat, state chips) with text demoted to tooltips / expandable details.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from .metrics import collect_metrics
from .status import build_status_payload


REFRESH_MS = 10_000
STREAM_IDLE_MS = 8_000
STREAM_WATCH_MS = 2_500
STREAM_CRITICAL_MS = 1_000
OPERATOR_GUIDE = "docs/hermes-context-visor-operator-guide-v1.md"
FALLBACK_OPERATOR_GUIDE = """# Hermes Context Cockpit Operator Guide\n\nThis local cockpit is read-only.\n\nSafe controls:\n- Copy slash commands only\n- Refresh status\n- Open fixed local read-only pages\n\nBlocked in the browser cockpit:\n- Running /compress or other mutating Hermes commands\n- Broker or gateway execution\n- Hindsight or LCM mutation\n- Shell or arbitrary command input\n"""

CRITICAL_RIBBONS = frozenset(
    {
        "HERMES OFFLINE",
        "OLD NUMBERS",
        "MODEL CHANGED",
        "COST WARNING",
        "MEMORY LINE HIT",
        "SHRINKING NOW",
    }
)
WATCH_RIBBONS = frozenset(
    {
        "GETTING FULL",
        "SHRINK QUEUED",
        "CAN'T SHRINK YET",
        "MEMORY UNKNOWN",
        "QUIET",
    }
)

# Read-only visual fixtures for alert-state screenshots. Never mutates Hermes.
DEMO_SCENARIOS = (
    "healthy",
    "near_threshold",
    "shrink_queued",
    "shrinking",
    "just_shrank",
    "stale",
    "offline",
    "model_warning",
    "cost_warning",
)


def _demo_base_metrics() -> Dict[str, Any]:
    return {
        "profile": "personal-ops",
        "freshness": "fresh",
        "prompt_pct": 11.0,
        "prompt_tokens": 108_900,
        "message_count": 42,
        "model": "deepseek/deepseek-v4-flash",
        "model_alert": None,
        "window": 1_000_000,
        "window_source": "demo",
        "collected_at": time.time(),
        "lcm": {
            "loaded": True,
            "threshold_ratio": 0.25,
            "threshold_tokens": 250_000,
            "threshold_pct": 25.0,
            "compressions": 0,
            "fill_of_lcm": 0.436,
            "cache_state": "hot",
            "turns_since_leaf": 3,
            "last_leaf_compaction_at": None,
            "last_compaction_duration_ms": 420,
            "last_api_call_at": time.time() - 8,
            "last_compression_status": "idle",
            "last_compression_noop_reason": "",
            "fresh_tail_count": 12,
            "pre_tail_message_count": 30,
            "total_message_count": 42,
            "live_snapshot_loaded": True,
        },
        "cost": {
            "estimated_usd": 0.42,
            "actual_usd": 0.38,
            "billing_mode": "payg",
            "billing_provider": "openrouter",
            "cost_status": "estimated",
            "api_calls": 18,
            "burn": {"usd_per_call_recent": 0.02, "tok_per_call": 1200},
        },
        "liveness": {
            "running": True,
            "heartbeat_age_sec": 2.0,
            "heartbeat_source": "demo-fixture",
            "state_db_age_sec": 4.0,
            "gateway_age_sec": 6.0,
            "source": "demo",
        },
    }


def build_demo_metrics(scenario: str) -> Dict[str, Any]:
    """Build read-only metrics for a named visual demo scenario."""
    name = (scenario or "").strip().lower().replace("-", "_")
    m = _demo_base_metrics()
    if name in {"healthy", "all_good"}:
        return m
    if name in {"near_threshold", "getting_full", "near"}:
        m["prompt_tokens"] = 230_000
        m["prompt_pct"] = 23.0
        m["lcm"]["fill_of_lcm"] = 0.92
        m["lcm"]["compressions"] = 0
        m["lcm"]["last_compression_status"] = "idle"
        return m
    if name in {"shrink_queued", "eligible", "waiting", "pending"}:
        m["prompt_tokens"] = 255_000
        m["prompt_pct"] = 25.5
        m["lcm"]["fill_of_lcm"] = 1.02
        m["lcm"]["last_compression_status"] = "pending"
        return m
    if name in {"shrinking", "compacting", "running"}:
        m["prompt_tokens"] = 260_000
        m["prompt_pct"] = 26.0
        m["lcm"]["fill_of_lcm"] = 1.04
        m["lcm"]["last_compression_status"] = "running"
        return m
    if name in {"just_shrank", "recently_compacted"}:
        now = time.time()
        m["prompt_tokens"] = 95_000
        m["prompt_pct"] = 9.5
        m["lcm"]["fill_of_lcm"] = 0.38
        m["lcm"]["compressions"] = 4
        m["lcm"]["turns_since_leaf"] = 0
        m["lcm"]["last_leaf_compaction_at"] = now - 20
        m["lcm"]["last_api_call_at"] = now - 5
        m["lcm"]["last_compression_status"] = "idle"
        return m
    if name in {"stale", "old_numbers"}:
        m["freshness"] = "stale"
        m["liveness"]["heartbeat_age_sec"] = 900.0
        m["liveness"]["state_db_age_sec"] = 920.0
        return m
    if name in {"offline", "hermes_offline"}:
        m["freshness"] = "offline"
        m["liveness"]["running"] = False
        m["liveness"]["heartbeat_age_sec"] = None
        return m
    if name in {"model_warning", "model_changed"}:
        m["model"] = "openai/gpt-5.4"
        m["model_alert"] = "Model changed since last turn — was deepseek-v4-flash, now gpt-5.4."
        return m
    if name in {"cost_warning", "cost"}:
        m["cost"]["estimated_usd"] = 12.5
        m["cost"]["actual_usd"] = 11.8
        m["cost"]["cost_status"] = "warning"
        m["cost"]["burn"] = {"usd_per_call_recent": 0.85, "tok_per_call": 18000}
        return m
    raise KeyError(f"unknown demo scenario: {scenario}")


def build_demo_payload(scenario: str) -> Dict[str, Any]:
    payload = build_status_payload(build_demo_metrics(scenario))
    payload["demo"] = True
    payload["demo_scenario"] = scenario
    return payload


def list_demo_scenarios() -> tuple[str, ...]:
    return DEMO_SCENARIOS


def build_cockpit_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/"


def open_browser(url: str) -> bool:
    try:
        return bool(webbrowser.open(url, new=2, autoraise=True))
    except Exception:
        return False


def stream_interval_ms(ribbon: str) -> int:
    if ribbon in CRITICAL_RIBBONS:
        return STREAM_CRITICAL_MS
    if ribbon in WATCH_RIBBONS:
        return STREAM_WATCH_MS
    return STREAM_IDLE_MS


def _operator_guide_path() -> Path | None:
    try:
        candidate = Path(__file__).resolve().parents[4] / OPERATOR_GUIDE
    except Exception:
        return None
    return candidate if candidate.exists() else None


def render_operator_guide_html(markdown_text: str) -> str:
    safe = escape(markdown_text)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hermes Context Cockpit Operator Guide</title>
  <style>
    body {{ margin: 0; background: #08111f; color: #e7efff; font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif; }}
    .shell {{ max-width: 980px; margin: 0 auto; padding: 28px 20px 56px; }}
    h1 {{ font-size: 32px; margin: 0 0 10px; }}
    .meta {{ color: #97a6c2; margin-bottom: 18px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #111827; border: 1px solid #23304a; border-radius: 18px; padding: 18px; line-height: 1.55; overflow: auto; }}
    a {{ color: #7dd3fc; }}
  </style>
</head>
<body>
  <div class="shell">
    <h1>Context Cockpit Operator Guide</h1>
    <div class="meta">Read-only local doc view served by the cockpit. Source: <code>{escape(OPERATOR_GUIDE)}</code></div>
    <pre>{safe}</pre>
  </div>
</body>
</html>
"""


def render_cockpit_html(
    *,
    profile: str,
    refresh_ms: int = REFRESH_MS,
    demo_scenario: str | None = None,
    static_shot: bool = False,
) -> str:
    demo_payload_json = "null"
    if demo_scenario:
        demo_payload_json = json.dumps(build_demo_payload(demo_scenario), default=str)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hermes Context Cockpit</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #070b14;
      --panel: rgba(12, 18, 32, 0.92);
      --line: rgba(120, 144, 180, 0.28);
      --text: #eef4ff;
      --muted: #8fa0bc;
      --healthy: #22c55e;
      --watch: #facc15;
      --soon: #fb923c;
      --lcm: #f97316;
      --stale: #38bdf8;
      --offline: #64748b;
      --model: #c084fc;
      --cost: #ef4444;
      --critical: #ff4d6d;
      --track: rgba(30, 41, 59, 0.95);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "IBM Plex Sans", "Segoe UI", ui-sans-serif, system-ui, sans-serif;
      color: var(--text);
      background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(56, 189, 248, 0.14), transparent 55%),
        radial-gradient(ellipse 40% 30% at 90% 80%, rgba(249, 115, 22, 0.08), transparent 50%),
        linear-gradient(180deg, #05080f 0%, #0a101c 100%);
    }}
    .shell {{ max-width: 1280px; margin: 0 auto; padding: 18px 18px 28px; }}
    .deck-top {{
      display: flex; align-items: center; justify-content: space-between; gap: 14px;
      margin-bottom: 14px; flex-wrap: wrap;
    }}
    .brand {{
      display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
    }}
    .brand-mark {{
      font-family: "IBM Plex Mono", ui-monospace, monospace;
      font-size: 11px; letter-spacing: 0.22em; text-transform: uppercase;
      color: #7dd3fc; font-weight: 700;
    }}
    .brand-title {{
      font-size: clamp(22px, 3vw, 30px); font-weight: 800; letter-spacing: -0.03em;
    }}
    .live-pill {{
      display: inline-flex; align-items: center; gap: 10px;
      padding: 8px 14px; border-radius: 999px;
      border: 1px solid var(--line); background: rgba(8, 14, 26, 0.85);
      font-size: 14px; font-weight: 700; position: relative; overflow: hidden;
    }}
    .live-pill[data-mode="live"] {{ border-color: rgba(34, 197, 94, 0.45); color: #bbf7d0; }}
    .live-pill[data-mode="stale"] {{ border-color: rgba(56, 189, 248, 0.55); color: #bae6fd; }}
    .live-pill[data-mode="offline"] {{ border-color: rgba(100, 116, 139, 0.55); color: #cbd5e1; }}
    .live-pill[data-mode="critical"] {{ border-color: rgba(239, 68, 68, 0.65); color: #fecaca; animation: alert-flash 1.1s ease-in-out infinite; }}
    .live-pill[data-mode="live"]::before {{
      content: ""; position: absolute; inset: 0; pointer-events: none;
      background: linear-gradient(110deg, transparent 0%, rgba(34,197,94,0.18) 45%, transparent 70%);
      transform: translateX(-120%);
      animation: live-sweep 2.4s ease-in-out infinite;
    }}
    .pulse-dot {{
      width: 11px; height: 11px; border-radius: 50%; background: currentColor;
      box-shadow: 0 0 0 0 currentColor; animation: heartbeat 1.4s ease-out infinite;
      position: relative; z-index: 1; flex: 0 0 auto;
    }}
    .live-pill[data-mode="live"] .pulse-dot {{ animation: heartbeat 1.1s ease-out infinite; }}
    .live-pill[data-mode="stale"] .pulse-dot {{ animation: heartbeat 2.2s ease-out infinite; }}
    .live-pill[data-mode="offline"] .pulse-dot {{ animation: none; opacity: 0.45; }}
    .live-pill #live-label {{ position: relative; z-index: 1; }}
    @keyframes heartbeat {{
      0% {{ box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.55); transform: scale(1); }}
      35% {{ box-shadow: 0 0 0 8px rgba(34, 197, 94, 0.12); transform: scale(1.12); }}
      70% {{ box-shadow: 0 0 0 14px rgba(34, 197, 94, 0); transform: scale(1); }}
      100% {{ box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); transform: scale(1); }}
    }}
    @keyframes live-sweep {{
      0% {{ transform: translateX(-120%); }}
      55% {{ transform: translateX(120%); }}
      100% {{ transform: translateX(120%); }}
    }}
    @keyframes alert-flash {{
      0%, 100% {{ filter: brightness(1); }}
      50% {{ filter: brightness(1.25); }}
    }}
    .banner {{
      border-radius: 20px; padding: 18px 20px; margin-bottom: 16px;
      border: 1px solid var(--line); background: var(--panel);
      position: relative; overflow: hidden;
      transition: box-shadow 180ms ease, border-color 180ms ease;
    }}
    .banner::before {{
      content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 7px;
      background: var(--accent, var(--healthy));
    }}
    .banner.critical {{
      border-color: rgba(239, 68, 68, 0.55);
      box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.25), 0 0 40px rgba(239, 68, 68, 0.18);
      animation: alert-flash 1.4s ease-in-out infinite;
    }}
    .banner.critical::before {{ width: 10px; }}
    .banner-row {{ display: flex; gap: 18px; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; }}
    .banner-state {{
      font-size: clamp(28px, 4.5vw, 44px); font-weight: 900; letter-spacing: -0.04em;
      line-height: 1.05; margin: 2px 0 8px;
    }}
    .banner-summary {{ color: #d7e3f8; font-size: 17px; max-width: 760px; line-height: 1.45; }}
    .next-strip {{
      margin-top: 14px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
      padding: 12px 14px; border-radius: 14px;
      background: rgba(8, 14, 26, 0.72); border: 1px solid rgba(125, 211, 252, 0.18);
    }}
    .next-label {{
      font-size: 13px; letter-spacing: 0.12em; text-transform: uppercase;
      color: #7dd3fc; font-weight: 800;
    }}
    .next-text {{ font-size: 17px; font-weight: 650; }}
    .instrument-deck {{
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .gauge-wrap {{
      width: 250px; height: 118px; position: relative; margin: 8px 0 0;
    }}
    .gauge-wrap svg {{ width: 250px; height: 118px; overflow: visible; }}
    .gauge-track {{ fill: none; stroke: var(--track); stroke-width: 18; stroke-linecap: butt; }}
    .gauge-zone {{
      fill: none; stroke: rgba(249, 115, 22, 0.82); stroke-width: 18; stroke-linecap: butt;
    }}
    .gauge-zone.hot {{ stroke: rgba(239, 68, 68, 0.85); }}
    .gauge-fill {{
      fill: none; stroke: var(--healthy); stroke-width: 18; stroke-linecap: round;
      stroke-dasharray: 100; stroke-dashoffset: 100;
      transition: stroke-dashoffset 320ms ease, stroke 180ms ease;
    }}
    .gauge-tick {{ stroke: #9fb0cc; stroke-width: 2; }}
    .gauge-tick.major {{ stroke: #eef4ff; stroke-width: 3; }}
    .gauge-tick.threshold {{ stroke: #fb923c; stroke-width: 5; }}
    .gauge-end-label {{
      fill: #d7e3f8; font-size: 15px; font-weight: 800;
      font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
    }}
    .gauge-needle {{
      stroke: #eef4ff; stroke-width: 4; stroke-linecap: round;
    }}
    .gauge-hub {{ fill: #eef4ff; }}
    .gauge-readout {{
      width: 100%; text-align: center; margin-top: 10px;
      display: grid; gap: 4px;
    }}
    .gauge-value {{
      font-size: 36px; font-weight: 900; letter-spacing: -0.03em; line-height: 1.05;
    }}
    .gauge-total {{
      font-size: 18px; font-weight: 800; color: #eef4ff; line-height: 1.2;
    }}
    .inst-sub {{
      text-align: center; color: #d7e3f8; font-size: 15px; line-height: 1.35;
      margin-top: 10px; min-height: 2.4em; font-weight: 650;
    }}
    .instrument {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 14px 16px;
      min-height: 310px;
      position: relative;
      display: flex; flex-direction: column; align-items: center;
      transition: border-color 160ms ease, transform 160ms ease;
    }}
    .instrument:hover {{ border-color: rgba(125, 211, 252, 0.35); transform: translateY(-1px); }}
    .span-3 {{ grid-column: span 3; }}
    .span-2 {{ grid-column: span 2; }}
    .instrument.alert {{
      border-color: rgba(239, 68, 68, 0.55);
      box-shadow: inset 0 0 0 1px rgba(239, 68, 68, 0.2), 0 0 28px rgba(239, 68, 68, 0.12);
    }}
    .instrument.dim {{ opacity: 0.55; filter: grayscale(0.35); }}
    .inst-label {{
      align-self: stretch; display: flex; justify-content: space-between; align-items: center;
      font-size: 14px; letter-spacing: 0.1em; text-transform: uppercase;
      color: var(--muted); font-weight: 800; margin-bottom: 6px;
    }}
    .inst-chip {{
      font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase;
      padding: 4px 9px; border-radius: 999px; border: 1px solid var(--line); color: #c7d5ef;
    }}
    .burn-meter {{
      width: 100%; margin-top: 10px; height: 10px; border-radius: 999px;
      background: var(--track); overflow: hidden; border: 1px solid rgba(148,163,184,0.15);
    }}
    .burn-meter > span {{
      display: block; height: 100%; width: 0%; border-radius: inherit;
      background: linear-gradient(90deg, var(--healthy), var(--soon), var(--cost));
      transition: width 280ms ease;
    }}
    .model-orb {{
      width: 88px; height: 88px; border-radius: 50%; margin: 10px 0 8px;
      display: grid; place-items: center;
      border: 3px solid rgba(192, 132, 252, 0.45);
      background: radial-gradient(circle at 35% 30%, rgba(192,132,252,0.35), rgba(12,18,32,0.9));
      font-size: 13px; font-weight: 800; text-align: center; padding: 8px; line-height: 1.2;
    }}
    .model-orb.alert {{
      border-color: var(--critical);
      box-shadow: 0 0 24px rgba(239, 68, 68, 0.35);
      animation: alert-flash 1.1s ease-in-out infinite;
    }}
    .heart-ring {{
      width: 88px; height: 88px; border-radius: 50%; margin: 10px 0 8px;
      display: grid; place-items: center; position: relative;
      border: 3px solid rgba(34, 197, 94, 0.45);
      background: radial-gradient(circle at 35% 30%, rgba(34,197,94,0.22), rgba(12,18,32,0.9));
    }}
    .heart-ring::before {{
      content: ""; position: absolute; inset: 8px; border-radius: 50%;
      border: 2px solid transparent; border-top-color: rgba(187, 247, 208, 0.75);
      animation: heart-spin 2.8s linear infinite;
    }}
    .heart-ring::after {{
      content: ""; position: absolute; inset: -6px; border-radius: 50%;
      border: 2px solid rgba(34, 197, 94, 0.35);
      animation: heartbeat 1.4s ease-out infinite;
    }}
    .heart-ring.stale {{ border-color: var(--stale); }}
    .heart-ring.stale::before {{ border-top-color: rgba(186, 230, 253, 0.7); animation-duration: 4.2s; }}
    .heart-ring.stale::after {{ border-color: rgba(56, 189, 248, 0.4); }}
    .heart-ring.offline {{ border-color: var(--offline); }}
    .heart-ring.offline::before, .heart-ring.offline::after {{ animation: none; opacity: 0.25; }}
    .heart-core {{ font-size: 14px; font-weight: 800; text-align: center; line-height: 1.2; position: relative; z-index: 1; }}
    @keyframes heart-spin {{
      to {{ transform: rotate(360deg); }}
    }}
    .details {{
      border: 1px solid var(--line); border-radius: 16px; background: rgba(8, 14, 26, 0.72);
      margin-bottom: 12px;
    }}
    .details > summary {{
      cursor: pointer; list-style: none; padding: 12px 16px;
      font-size: 14px; letter-spacing: 0.04em;
      color: #c7d5ef; font-weight: 800; user-select: none;
      display: flex; align-items: center; justify-content: space-between; gap: 12px;
      white-space: nowrap; overflow: hidden;
    }}
    .details > summary::-webkit-details-marker {{ display: none; }}
    .details-summary-main {{
      overflow: hidden; text-overflow: ellipsis;
      text-transform: uppercase; letter-spacing: 0.1em;
    }}
    .details-summary-hint {{
      color: #8fa0bc; font-size: 13px; font-weight: 650;
      text-transform: none; letter-spacing: 0; overflow: hidden; text-overflow: ellipsis;
    }}
    .details[open] > summary {{ border-bottom: 1px solid var(--line); color: #eef4ff; }}
    .details[open] .details-summary-hint {{ display: none; }}
    .detail-grid {{
      display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0;
    }}
    .detail-col {{ padding: 14px 16px; border-right: 1px solid rgba(148,163,184,0.12); }}
    .detail-col:last-child {{ border-right: 0; }}
    .detail-title {{ font-size: 13px; letter-spacing: 0.1em; text-transform: uppercase; color: #7dd3fc; margin-bottom: 10px; font-weight: 800; }}
    .kv {{ display: grid; gap: 9px; }}
    .kv-row {{ display: flex; justify-content: space-between; gap: 12px; font-size: 15px; }}
    .kv-label {{ color: #a8b8d4; }}
    .kv-value {{ font-weight: 700; text-align: right; color: #eef4ff; }}
    .action-bar {{
      display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
      padding: 12px 0 4px;
    }}
    .action-btn {{
      border: 1px solid rgba(125, 211, 252, 0.28);
      background: rgba(8, 17, 31, 0.9); color: #eef4ff;
      border-radius: 999px; padding: 10px 16px; font: inherit; font-size: 15px;
      font-weight: 700; cursor: pointer;
    }}
    .action-btn:hover {{ border-color: rgba(125, 211, 252, 0.55); }}
    .action-btn.refresh {{ border-color: rgba(250, 204, 21, 0.4); }}
    .action-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
    .copy-tray {{
      display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px;
    }}
    .cmd-chip {{
      background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(59, 130, 246, 0.25);
      color: #dbeafe; border-radius: 999px; padding: 8px 14px;
      font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 13px; cursor: pointer;
    }}
    .cmd-chip:hover {{ border-color: rgba(125, 211, 252, 0.5); }}
    .footer-note {{
      margin-top: 10px; color: #a8b8d4; font-size: 14px;
      display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap;
    }}
    .tooltip {{
      position: absolute; left: 12px; right: 12px; bottom: 8px;
      font-size: 13px; color: #a8b8d4; text-align: center; opacity: 0;
      transition: opacity 140ms ease; pointer-events: none;
    }}
    .instrument:hover .tooltip {{ opacity: 1; }}
    @media (max-width: 1100px) {{
      .span-3, .span-2 {{ grid-column: span 6; }}
      .detail-grid {{ grid-template-columns: 1fr; }}
      .detail-col {{ border-right: 0; border-bottom: 1px solid rgba(148,163,184,0.12); }}
    }}
    @media (max-width: 700px) {{
      .span-3, .span-2 {{ grid-column: span 12; }}
      .instrument {{ min-height: 0; }}
      .shell {{ padding: 12px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="deck-top">
      <div class="brand">
        <span class="brand-mark">Flight Deck</span>
        <span class="brand-title">Context Cockpit</span>
      </div>
      <div class="live-pill" id="live-pill" data-mode="offline" title="Live update age">
        <span class="pulse-dot" id="live-dot"></span>
        <span id="live-label">Connecting…</span>
      </div>
    </div>

    <section class="banner" id="banner" data-state="ALL GOOD">
      <div class="banner-row">
        <div>
          <div class="inst-chip" id="banner-chip">status</div>
          <div class="banner-state" id="banner-state">Loading…</div>
          <div class="banner-summary" id="banner-summary">Checking chat health.</div>
        </div>
        <div class="meta-side" style="text-align:right;color:var(--muted);font-size:13px;">
          <div id="meta-profile">{profile}</div>
          <div id="meta-mode">read-only · live stream</div>
        </div>
      </div>
      <div class="next-strip" id="next-strip">
        <span class="next-label">What to do</span>
        <span class="next-text" id="next-text">waiting for live data</span>
      </div>
    </section>

    <section class="instrument-deck" id="instrument-deck" aria-label="Telemetry instruments">
      <article class="instrument span-3" id="context-inst" title="How full the chat window is">
        <div class="inst-label"><span>Chat fill</span><span class="inst-chip" id="context-chip">window</span></div>
        <div class="gauge-wrap">
          <svg viewBox="0 0 250 120" aria-hidden="true">
            <path class="gauge-track" d="M35 105 A90 90 0 0 1 215 105" pathLength="100"></path>
            <path class="gauge-zone" id="context-zone" d="M35 105 A90 90 0 0 1 215 105" pathLength="100"></path>
            <path class="gauge-fill" id="context-arc" d="M35 105 A90 90 0 0 1 215 105" pathLength="100"></path>
            <g id="context-ticks"></g>
            <line class="gauge-needle" id="context-needle" x1="125" y1="105" x2="125" y2="48"></line>
            <circle class="gauge-hub" cx="125" cy="105" r="5"></circle>
            <text class="gauge-end-label" id="context-min" x="28" y="118" text-anchor="middle">0</text>
            <text class="gauge-end-label" id="context-max" x="222" y="118" text-anchor="middle">1M</text>
          </svg>
        </div>
        <div class="gauge-readout">
          <div class="gauge-value" id="context-value">—</div>
          <div class="gauge-total" id="context-total">of —</div>
        </div>
        <div class="inst-sub" id="context-sub">tokens used</div>
        <div class="tooltip" id="context-tip">Orange band = suggested shrink range</div>
      </article>

      <article class="instrument span-3" id="lcm-inst" title="Distance to the LCM auto-shrink line (does not delete chat)">
        <div class="inst-label"><span>LCM / Auto-shrink</span><span class="inst-chip" id="lcm-chip">to line</span></div>
        <div class="gauge-wrap">
          <svg viewBox="0 0 250 120" aria-hidden="true">
            <path class="gauge-track" d="M35 105 A90 90 0 0 1 215 105" pathLength="100"></path>
            <path class="gauge-zone hot" id="lcm-zone" d="M35 105 A90 90 0 0 1 215 105" pathLength="100"></path>
            <path class="gauge-fill" id="lcm-arc" d="M35 105 A90 90 0 0 1 215 105" pathLength="100"></path>
            <g id="lcm-ticks"></g>
            <line class="gauge-needle" id="lcm-needle" x1="125" y1="105" x2="125" y2="48"></line>
            <circle class="gauge-hub" cx="125" cy="105" r="5"></circle>
            <text class="gauge-end-label" id="lcm-min" x="28" y="118" text-anchor="middle">0</text>
            <text class="gauge-end-label" id="lcm-max" x="222" y="118" text-anchor="middle">250k</text>
          </svg>
        </div>
        <div class="gauge-readout">
          <div class="gauge-value" id="lcm-value">—</div>
          <div class="gauge-total" id="lcm-total">distance to auto-shrink</div>
        </div>
        <div class="inst-sub" id="lcm-sub">LCM threshold distance</div>
        <div class="tooltip" id="lcm-tip">Needle = progress to auto-shrink line · big number = room left</div>
      </article>

      <article class="instrument span-2" id="cost-inst" title="Session spend">
        <div class="inst-label"><span>Spend</span><span class="inst-chip" id="cost-chip">session</span></div>
        <div class="gauge-value" id="cost-value" style="position:static;margin:18px 0 6px;font-size:32px;">—</div>
        <div class="inst-sub" id="cost-sub">billing</div>
        <div class="burn-meter" title="Recent burn intensity"><span id="cost-burn"></span></div>
        <div class="tooltip" id="cost-tip">estimated session cost</div>
      </article>

      <article class="instrument span-2" id="model-inst" title="Active model">
        <div class="inst-label"><span>Model</span><span class="inst-chip" id="model-chip">active</span></div>
        <div class="model-orb" id="model-orb">—</div>
        <div class="inst-sub" id="model-sub">window</div>
        <div class="tooltip" id="model-tip">silent change shows red</div>
      </article>

      <article class="instrument span-2" id="heart-inst" title="Is Hermes awake?">
        <div class="inst-label"><span>Heartbeat</span><span class="inst-chip" id="heart-chip">live</span></div>
        <div class="heart-ring" id="heart-ring"><div class="heart-core" id="heart-core">—</div></div>
        <div class="inst-sub" id="heart-sub">activity age</div>
        <div class="tooltip" id="heart-tip">pulse = live feed</div>
      </article>
    </section>

    <details class="details" id="details-panel">
      <summary>
        <span class="details-summary-main" id="details-summary-main">Details</span>
        <span class="details-summary-hint" id="details-summary-hint">one-row · expand for numbers</span>
      </summary>
      <div class="detail-grid">
        <div class="detail-col">
          <div class="detail-title">LCM / Auto-shrink</div>
          <div class="kv" id="lcm-kv"></div>
        </div>
        <div class="detail-col">
          <div class="detail-title">Cost · Model</div>
          <div class="kv" id="cost-model-kv"></div>
        </div>
        <div class="detail-col">
          <div class="detail-title">Freshness</div>
          <div class="kv" id="freshness-kv"></div>
        </div>
      </div>
    </details>

    <div class="action-bar" id="action-bar" aria-label="Safe controls"></div>
    <div class="copy-tray" id="copy-tray" aria-label="Copyable commands"></div>
    <div class="footer-note">
      <span>Read-only. Buttons only copy, refresh, or open local pages — they never run Hermes for you.</span>
      <span id="footer-updated">Not updated yet</span>
    </div>
  </div>

  <script>
    const FALLBACK_MS = {refresh_ms};
    const CRITICAL = new Set(['HERMES OFFLINE','OLD NUMBERS','MODEL CHANGED','COST WARNING','MEMORY LINE HIT','SHRINKING NOW']);
    const WATCH = new Set(['GETTING FULL','SHRINK QUEUED',"CAN'T SHRINK YET",'MEMORY UNKNOWN','QUIET']);
    const DEMO_SCENARIO = new URLSearchParams(window.location.search).get('demo') || '';
    const STATIC_SHOT = new URLSearchParams(window.location.search).has('static') || {str(static_shot).lower()};
    const EMBEDDED_DEMO = {demo_payload_json};
    const STATUS_URL = DEMO_SCENARIO
      ? ('/api/demo?scenario=' + encodeURIComponent(DEMO_SCENARIO))
      : '/api/status';
    const STREAM_URL = DEMO_SCENARIO
      ? ('/api/demo-stream?scenario=' + encodeURIComponent(DEMO_SCENARIO))
      : '/api/stream';

    const stateColor = {{
      'ALL GOOD': 'var(--healthy)',
      'QUIET': 'var(--watch)',
      'GETTING FULL': 'var(--soon)',
      'MEMORY LINE HIT': 'var(--lcm)',
      'SHRINK QUEUED': 'var(--soon)',
      'SHRINKING NOW': 'var(--lcm)',
      'JUST SHRANK': 'var(--healthy)',
      "CAN'T SHRINK YET": 'var(--lcm)',
      'MEMORY UNKNOWN': 'var(--offline)',
      'OLD NUMBERS': 'var(--stale)',
      'HERMES OFFLINE': 'var(--offline)',
      'MODEL CHANGED': 'var(--model)',
      'COST WARNING': 'var(--cost)'
    }};

    const freshnessPlain = {{
      fresh: 'Up to date',
      idle: 'Quiet but running',
      quiet: 'Recently used',
      stale: 'Looks old',
      offline: 'Not detected',
      unknown: 'Unknown'
    }};

    let lastPayloadAt = 0;
    let pollTimer = null;
    let ageTimer = null;
    let stream = null;
    let usingStream = false;

    function fmtTokens(value) {{
      if (value === null || value === undefined) return '—';
      const n = Number(value) || 0;
      if (n >= 1000000) {{
        const m = n / 1000000;
        return (Number.isInteger(m) ? m.toFixed(0) : m.toFixed(2).replace(/0+$/, '').replace(/\\.$/, '')) + 'M';
      }}
      if (n >= 1000) {{
        const k = n / 1000;
        return (Number.isInteger(k) ? k.toFixed(0) : k.toFixed(1).replace(/0$/, '').replace(/\\.$/, '')) + 'k';
      }}
      return String(Math.round(n));
    }}

    function fmtDial(value) {{
      // Shorter labels for meter face (avoid 250.0k clutter).
      if (value === null || value === undefined) return '—';
      const n = Number(value) || 0;
      if (n >= 1000000) return (n / 1000000).toFixed(n % 1000000 === 0 ? 0 : 1).replace(/0$/, '').replace(/\\.$/, '') + 'M';
      if (n >= 1000) return Math.round(n / 1000) + 'k';
      return String(Math.round(n));
    }}

    function fmtUsd(value) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
      const n = Number(value);
      return n < 0.01 ? '$' + n.toFixed(4) : '$' + n.toFixed(2);
    }}

    function secondsLabel(value) {{
      if (value === null || value === undefined) return '—';
      const n = Number(value);
      if (!Number.isFinite(n)) return '—';
      if (n >= 3600) return (n / 3600).toFixed(1) + 'h';
      if (n >= 60) return (n / 60).toFixed(1) + 'm';
      return Math.round(n) + 's';
    }}

    function timeLabel(epochSeconds) {{
      if (epochSeconds === null || epochSeconds === undefined) return '—';
      const n = Number(epochSeconds);
      if (!Number.isFinite(n) || n <= 0) return '—';
      return new Date(n * 1000).toLocaleTimeString();
    }}

    function setArc(el, pct, color) {{
      const clamped = Math.max(0, Math.min(100, Number(pct) || 0));
      el.style.strokeDasharray = '100';
      el.style.strokeDashoffset = String(100 - clamped);
      el.style.stroke = color;
    }}

    function polar(cx, cy, r, pct) {{
      const angle = Math.PI - (Math.max(0, Math.min(100, pct)) / 100) * Math.PI;
      return [cx + r * Math.cos(angle), cy - r * Math.sin(angle)];
    }}

    function setNeedle(el, pct) {{
      const clamped = Math.max(0, Math.min(100, Number(pct) || 0));
      const deg = (-90 + clamped * 1.8).toFixed(2);
      el.setAttribute('transform', `rotate(${{deg}} 125 105)`);
    }}

    function setZone(el, startPct, endPct) {{
      const start = Math.max(0, Math.min(100, Number(startPct) || 0));
      const end = Math.max(start, Math.min(100, Number(endPct) || 0));
      const span = Math.max(0, end - start);
      el.setAttribute('stroke-dasharray', `${{span}} ${{100 - span}}`);
      el.style.strokeDashoffset = String(-start);
    }}

    function drawTicks(groupId, {{ thresholdPct = null }} = {{}}) {{
      const g = document.getElementById(groupId);
      if (!g) return;
      g.innerHTML = '';
      // Dial face stays empty: ticks only, no text on the arc.
      const cx = 125, cy = 105, rOuter = 90, rInner = 72;
      for (let p = 0; p <= 100; p += 10) {{
        const isMajor = p === 0 || p === 50 || p === 100;
        const [x1, y1] = polar(cx, cy, rOuter + 1, p);
        const [x2, y2] = polar(cx, cy, isMajor ? rInner : rInner + 8, p);
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1.toFixed(1));
        line.setAttribute('y1', y1.toFixed(1));
        line.setAttribute('x2', x2.toFixed(1));
        line.setAttribute('y2', y2.toFixed(1));
        line.setAttribute('class', isMajor ? 'gauge-tick major' : 'gauge-tick');
        g.appendChild(line);
      }}
      if (thresholdPct !== null && thresholdPct !== undefined) {{
        const t = Math.max(0, Math.min(100, Number(thresholdPct) || 0));
        const [x1, y1] = polar(cx, cy, rOuter + 8, t);
        const [x2, y2] = polar(cx, cy, rInner - 2, t);
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1.toFixed(1));
        line.setAttribute('y1', y1.toFixed(1));
        line.setAttribute('x2', x2.toFixed(1));
        line.setAttribute('y2', y2.toFixed(1));
        line.setAttribute('class', 'gauge-tick threshold');
        g.appendChild(line);
      }}
    }}

    function paintMeter({{
      arcId, needleId, ticksId, zoneId, pct, color,
      zoneStart, zoneEnd, thresholdPct,
    }}) {{
      setArc(document.getElementById(arcId), pct, color);
      setNeedle(document.getElementById(needleId), pct);
      if (zoneId) setZone(document.getElementById(zoneId), zoneStart, zoneEnd);
      drawTicks(ticksId, {{ thresholdPct }});
    }}

    function setRows(id, rows) {{
      const root = document.getElementById(id);
      root.innerHTML = '';
      for (const [label, value] of rows) {{
        const row = document.createElement('div');
        row.className = 'kv-row';
        row.innerHTML = `<span class="kv-label">${{label}}</span><span class="kv-value">${{value}}</span>`;
        root.appendChild(row);
      }}
    }}

    function plainStatusFallback(status) {{
      const low = String(status || '').toLowerCase();
      if (low === 'running') return 'Shrinking the chat right now';
      if (low === 'pending') return 'Ready to shrink on the next chance';
      if (low === 'noop') return 'Checked, but did not shrink';
      if (low === 'idle') return 'Idle — not shrinking right now';
      return status || '—';
    }}

    function intervalFor(ribbon) {{
      if (CRITICAL.has(ribbon)) return 1000;
      if (WATCH.has(ribbon)) return 2500;
      return FALLBACK_MS;
    }}

    function updateLivePill(ribbon) {{
      const ageSec = lastPayloadAt ? (Date.now() - lastPayloadAt) / 1000 : null;
      const pill = document.getElementById('live-pill');
      const label = document.getElementById('live-label');
      let mode = 'live';
      if (ribbon === 'HERMES OFFLINE') mode = 'offline';
      else if (ribbon === 'OLD NUMBERS' || ribbon === 'MODEL CHANGED' || ribbon === 'COST WARNING') mode = 'critical';
      else if (ageSec !== null && ageSec > 20) mode = 'stale';
      else if (CRITICAL.has(ribbon)) mode = 'critical';
      pill.dataset.mode = mode;
      const feed = usingStream ? 'stream' : 'poll';
      if (ageSec === null) label.textContent = 'Connecting…';
      else if (mode === 'offline') label.textContent = `Offline · ${{feed}}`;
      else if (mode === 'stale') label.textContent = `Stale ${{Math.round(ageSec)}}s · ${{feed}}`;
      else label.textContent = `Live ${{Math.round(ageSec)}}s · ${{feed}}`;
    }}

    async function copyText(text) {{
      if (navigator.clipboard && navigator.clipboard.writeText) {{
        await navigator.clipboard.writeText(text);
        return true;
      }}
      const area = document.createElement('textarea');
      area.value = text;
      area.setAttribute('readonly', 'readonly');
      area.style.position = 'absolute';
      area.style.left = '-9999px';
      document.body.appendChild(area);
      area.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(area);
      return ok;
    }}

    function renderActions(payload) {{
      const controls = Array.isArray(payload.controls) ? payload.controls : [];
      const bar = document.getElementById('action-bar');
      const tray = document.getElementById('copy-tray');
      bar.innerHTML = '';
      tray.innerHTML = '';
      for (const control of controls) {{
        if (!control.allowed && control.action_type !== 'copy_command') continue;
        if (control.action_type === 'copy_command') {{
          const chip = document.createElement('button');
          chip.type = 'button';
          chip.className = 'cmd-chip';
          chip.textContent = control.command || control.label;
          chip.title = control.description || 'Copy command';
          if (!control.allowed) {{
            chip.disabled = true;
            chip.title = control.disabled_reason || 'Blocked';
          }} else {{
            chip.addEventListener('click', async () => {{
              await copyText(control.command || '');
              chip.textContent = 'Copied';
              window.setTimeout(() => {{ chip.textContent = control.command || control.label; }}, 1200);
            }});
          }}
          tray.appendChild(chip);
          continue;
        }}
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'action-btn' + (control.action_type === 'refresh_status' ? ' refresh' : '');
        btn.textContent = control.label;
        btn.title = control.description || '';
        if (!control.allowed) {{
          btn.disabled = true;
        }} else {{
          btn.addEventListener('click', async () => {{
            if (control.action_type === 'refresh_status') {{
              await loadOnce();
              return;
            }}
            if (control.action_type === 'open_readonly_url') {{
              window.open(control.url, '_blank', 'noopener,noreferrer');
            }}
          }});
        }}
        bar.appendChild(btn);
      }}
    }}

    function render(payload) {{
      const metrics = payload.metrics || {{}};
      const status = payload.status || {{}};
      const lcm = metrics.lcm || {{}};
      const lcmState = payload.lcm_state || {{}};
      const cost = metrics.cost || {{}};
      const live = metrics.liveness || {{}};
      const ribbon = status.ribbon || 'HERMES OFFLINE';
      const color = stateColor[ribbon] || 'var(--watch)';
      const critical = CRITICAL.has(ribbon);

      lastPayloadAt = Date.now();
      updateLivePill(ribbon);

      const banner = document.getElementById('banner');
      banner.dataset.state = ribbon;
      banner.classList.toggle('critical', critical);
      banner.style.setProperty('--accent', color);
      document.getElementById('banner-chip').textContent = status.severity_label || status.severity || 'status';
      document.getElementById('banner-state').textContent = ribbon;
      document.getElementById('banner-summary').textContent = payload.summary || status.summary || 'No summary';
      document.getElementById('next-text').textContent = status.next_action || 'No recommendation';
      document.getElementById('meta-profile').textContent = metrics.profile || '{profile}';

      const dim = !!status.dim_gauges;
      for (const id of ['context-inst','lcm-inst','cost-inst','model-inst','heart-inst']) {{
        document.getElementById(id).classList.toggle('dim', dim && ribbon !== 'HERMES OFFLINE');
      }}

      const pct = Number(metrics.prompt_pct || 0);
      const thresholdPct = Number(lcm.threshold_pct || 0);
      const windowTokens = Number(metrics.window || 0);
      // Suggested compaction band: from auto-shrink line to full window.
      const zoneStart = thresholdPct > 0 ? thresholdPct : 25;
      const zoneEnd = 100;
      paintMeter({{
        arcId: 'context-arc',
        needleId: 'context-needle',
        ticksId: 'context-ticks',
        zoneId: 'context-zone',
        pct,
        color,
        zoneStart,
        zoneEnd,
        thresholdPct: zoneStart,
      }});
      document.getElementById('context-value').textContent = pct.toFixed(0) + '%';
      document.getElementById('context-total').textContent = `${{fmtTokens(metrics.prompt_tokens)}} of ${{fmtTokens(metrics.window)}}`;
      document.getElementById('context-max').textContent = fmtDial(metrics.window);
      document.getElementById('context-sub').textContent = `Orange = shrink zone · starts ~${{fmtTokens(lcm.threshold_tokens)}} (${{zoneStart.toFixed(0)}}%)`;
      document.getElementById('context-chip').textContent = dim ? 'may be old' : 'window';
      document.getElementById('context-tip').textContent = `Total window ${{fmtTokens(metrics.window)}} · shrink from ${{fmtTokens(lcm.threshold_tokens)}}`;

      const fillPct = Math.max(0, Math.min(100, Math.round(Number(lcm.fill_of_lcm || 0) * 100)));
      const remainPct = lcm.loaded ? Math.max(0, 100 - fillPct) : null;
      const remainTokens = lcm.loaded
        ? Math.max(0, Number(lcm.threshold_tokens || 0) - Number(metrics.prompt_tokens || 0))
        : null;
      const lcmColor = stateColor[lcmState.ribbon || ribbon] || color;
      paintMeter({{
        arcId: 'lcm-arc',
        needleId: 'lcm-needle',
        ticksId: 'lcm-ticks',
        zoneId: 'lcm-zone',
        pct: fillPct,
        color: lcmColor,
        zoneStart: 80,
        zoneEnd: 100,
        thresholdPct: 80,
      }});
      // Big number = room left to auto-shrink (not a second "fill %" competing with Chat fill).
      document.getElementById('lcm-value').textContent = remainPct === null
        ? '—'
        : (remainPct + '% left');
      document.getElementById('lcm-total').textContent = lcm.loaded
        ? `${{fmtTokens(metrics.prompt_tokens)}} now · line ${{fmtTokens(lcm.threshold_tokens)}} · ${{fmtTokens(remainTokens)}} room`
        : 'LCM not loaded';
      document.getElementById('lcm-max').textContent = fmtDial(lcm.threshold_tokens);
      document.getElementById('lcm-sub').textContent = lcm.loaded
        ? (lcmState.label
          ? `Distance to auto-shrink · ${{lcmState.label}}`
          : 'Distance to auto-shrink · needle shows progress to the LCM line')
        : 'LCM not loaded';
      document.getElementById('lcm-chip').textContent = lcmState.chip || lcmState.label || 'to line';
      document.getElementById('lcm-tip').textContent = lcmState.detail
        || 'Needle = progress toward the LCM auto-shrink line. Big number = room left. Shrink keeps recent chat; it does not delete your conversation.';
      document.getElementById('lcm-inst').classList.toggle('alert', ['MEMORY LINE HIT','SHRINKING NOW',"CAN'T SHRINK YET",'SHRINK QUEUED','JUST SHRANK'].includes(lcmState.ribbon || ribbon));

      document.getElementById('details-summary-main').textContent = 'Details';
      document.getElementById('details-summary-hint').textContent = lcm.loaded
        ? `${{ribbon}} · ${{remainPct}}% room to auto-shrink · expand`
        : `${{ribbon}} · expand for numbers`;

      document.getElementById('cost-value').textContent = fmtUsd(cost.estimated_usd);
      document.getElementById('cost-sub').textContent = `${{cost.billing_mode || cost.billing_provider || 'billing'}} · ${{cost.cost_status || 'ok'}}`;
      document.getElementById('cost-chip').textContent = ribbon === 'COST WARNING' ? 'watch' : 'session';
      const burn = Number((cost.burn && cost.burn.usd_per_call_recent) || 0);
      const burnPct = Math.max(4, Math.min(100, burn * 400));
      document.getElementById('cost-burn').style.width = burnPct + '%';
      document.getElementById('cost-inst').classList.toggle('alert', ribbon === 'COST WARNING');
      document.getElementById('cost-tip').textContent = burn ? `Recent burn ${{fmtUsd(burn)}} / call` : 'Session spend estimate';

      const model = metrics.model || 'unknown';
      const short = model.includes('/') ? model.split('/').slice(-1)[0] : model;
      const orb = document.getElementById('model-orb');
      orb.textContent = short;
      orb.classList.toggle('alert', ribbon === 'MODEL CHANGED');
      document.getElementById('model-sub').textContent = `holds ~${{fmtTokens(metrics.window)}}`;
      document.getElementById('model-chip').textContent = ribbon === 'MODEL CHANGED' ? 'changed' : 'active';
      document.getElementById('model-inst').classList.toggle('alert', ribbon === 'MODEL CHANGED');
      document.getElementById('model-tip').textContent = metrics.model_alert || model;

      const freshKey = String(metrics.freshness || 'unknown');
      const ring = document.getElementById('heart-ring');
      ring.classList.remove('stale', 'offline');
      if (freshKey === 'offline' || ribbon === 'HERMES OFFLINE') ring.classList.add('offline');
      else if (freshKey === 'stale' || ribbon === 'OLD NUMBERS') ring.classList.add('stale');
      document.getElementById('heart-core').textContent = freshnessPlain[freshKey] || freshKey;
      document.getElementById('heart-sub').textContent = live.running
        ? `activity ${{secondsLabel(live.heartbeat_age_sec)}} ago`
        : 'Desktop not detected';
      document.getElementById('heart-chip').textContent = live.running ? 'running' : 'not seen';
      document.getElementById('heart-inst').classList.toggle('alert', ribbon === 'OLD NUMBERS' || ribbon === 'HERMES OFFLINE');
      document.getElementById('heart-tip').textContent = live.heartbeat_source || 'heartbeat source';

      const liveMissing = !lcm.live_snapshot_loaded;
      const liveUnavailable = 'Send one Desktop chat turn to refresh';
      setRows('lcm-kv', [
        ['In plain words', lcmState.detail || '—'],
        ['Live check', lcmState.live_status_plain || (liveMissing ? liveUnavailable : plainStatusFallback(lcm.last_compression_status))],
        ['Why not shrink', lcmState.noop_reason_plain || (liveMissing ? liveUnavailable : (lcm.last_compression_noop_reason ? String(lcm.last_compression_noop_reason) : '—'))],
        ['Recent / older', liveMissing && lcm.fresh_tail_count == null
          ? liveUnavailable
          : `${{lcm.fresh_tail_count ?? '—'}} / ${{lcm.pre_tail_message_count ?? '—'}}`],
        ['Auto-shrinks done', lcm.compressions ?? 0],
        ['Last auto-shrink', timeLabel(lcm.last_leaf_compaction_at)],
        ['Progress to line', lcm.loaded ? (fillPct + '% of LCM line') : '—'],
        ['Room left', remainPct === null ? '—' : (remainPct + '% · ' + fmtTokens(remainTokens))],
      ]);
      setRows('cost-model-kv', [
        ['Confirmed spend', fmtUsd(cost.actual_usd)],
        ['API calls', cost.api_calls ?? 0],
        ['Recent burn', cost.burn && cost.burn.usd_per_call_recent ? fmtUsd(cost.burn.usd_per_call_recent) + ' / call' : '—'],
        ['Full model', model],
        ['Alert', metrics.model_alert || 'stable'],
        ['Chat size', fmtTokens(metrics.prompt_tokens)],
      ]);
      setRows('freshness-kv', [
        ['Hermes running?', live.running ? 'yes' : 'no'],
        ['How checked', live.heartbeat_source || live.source || '—'],
        ['Activity age', secondsLabel(live.heartbeat_age_sec)],
        ['Chat file age', secondsLabel(live.state_db_age_sec)],
        ['Gateway age', secondsLabel(live.gateway_age_sec)],
        ['Recommended cmd', status.command || '—'],
      ]);

      renderActions(payload);
      const at = metrics.collected_at ? new Date(Number(metrics.collected_at) * 1000) : new Date();
      document.getElementById('footer-updated').textContent = 'Updated ' + at.toLocaleTimeString();
      return ribbon;
    }}

    async function loadOnce() {{
      try {{
        if (EMBEDDED_DEMO && typeof EMBEDDED_DEMO === 'object') {{
          if (DEMO_SCENARIO || EMBEDDED_DEMO.demo_scenario) {{
            document.getElementById('meta-mode').textContent =
              'read-only · demo:' + (DEMO_SCENARIO || EMBEDDED_DEMO.demo_scenario) + (STATIC_SHOT ? ' · static shot' : '');
          }}
          return render(EMBEDDED_DEMO);
        }}
        const res = await fetch(STATUS_URL, {{ cache: 'no-store' }});
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const payload = await res.json();
        if (DEMO_SCENARIO) {{
          document.getElementById('meta-mode').textContent = 'read-only · demo:' + DEMO_SCENARIO;
        }}
        return render(payload);
      }} catch (err) {{
        document.getElementById('banner').classList.add('critical');
        document.getElementById('banner-state').textContent = 'HERMES OFFLINE';
        document.getElementById('banner-summary').textContent = 'This cockpit page cannot reach its local status feed: ' + err;
        document.getElementById('next-text').textContent = 'wait a moment, or reopen /visor from Hermes Desktop';
        lastPayloadAt = Date.now();
        updateLivePill('HERMES OFFLINE');
        return 'HERMES OFFLINE';
      }}
    }}

    function schedulePoll(ribbon) {{
      if (pollTimer) window.clearTimeout(pollTimer);
      const ms = intervalFor(ribbon || 'ALL GOOD');
      pollTimer = window.setTimeout(async () => {{
        const next = await loadOnce();
        schedulePoll(next);
      }}, ms);
    }}

    function startStream() {{
      if (!window.EventSource) {{
        usingStream = false;
        loadOnce().then(schedulePoll);
        return;
      }}
      try {{
        stream = new EventSource(STREAM_URL);
        stream.onmessage = (ev) => {{
          try {{
            const payload = JSON.parse(ev.data);
            usingStream = true;
            render(payload);
            if (pollTimer) {{ window.clearTimeout(pollTimer); pollTimer = null; }}
          }} catch (err) {{
            // ignore malformed frames
          }}
        }};
        stream.onerror = () => {{
          if (stream) {{
            stream.close();
            stream = null;
          }}
          usingStream = false;
          loadOnce().then(schedulePoll);
        }};
      }} catch (err) {{
        usingStream = false;
        loadOnce().then(schedulePoll);
      }}
    }}

    document.addEventListener('visibilitychange', () => {{
      if (document.visibilityState === 'visible') loadOnce();
    }});
    window.addEventListener('focus', () => {{ loadOnce(); }});

    ageTimer = window.setInterval(() => {{
      const ribbon = document.getElementById('banner').dataset.state || 'ALL GOOD';
      updateLivePill(ribbon);
    }}, 1000);

    // Immediate paint, then prefer SSE. Adaptive poll is fallback only.
    // ?static=1 skips stream/poll so headless screenshots can exit cleanly.
    loadOnce().then((ribbon) => {{
      if (STATIC_SHOT) {{
        usingStream = false;
        document.getElementById('meta-mode').textContent =
          (DEMO_SCENARIO ? ('read-only · demo:' + DEMO_SCENARIO + ' · ') : 'read-only · ') + 'static shot';
        return;
      }}
      startStream();
      // If EventSource is unavailable, startStream already scheduled poll.
      // Otherwise wait for stream error before polling.
      if (!window.EventSource) {{
        schedulePoll(ribbon);
      }} else {{
        // Safety net: if no stream frame arrives within 4s, fall back to poll.
        window.setTimeout(() => {{
          if (!usingStream) schedulePoll(ribbon);
        }}, 4000);
      }}
    }});
  </script>
</body>
</html>
"""


def make_handler(*, profile: str, profile_dir: Path):
    class CockpitHandler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _payload(self) -> Dict[str, Any]:
            metrics = collect_metrics(profile, profile_dir, {})
            return build_status_payload(metrics)

        def _demo_payload(self, scenario: str) -> Dict[str, Any]:
            return build_demo_payload(scenario)

        def _stream_payload(self, payload_fn) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                while True:
                    payload = payload_fn()
                    ribbon = str((payload.get("status") or {}).get("ribbon") or "HERMES OFFLINE")
                    frame = f"data: {json.dumps(payload, default=str)}\n\n".encode("utf-8")
                    self.wfile.write(frame)
                    self.wfile.flush()
                    time.sleep(stream_interval_ms(ribbon) / 1000.0)
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                return

        def _stream_status(self) -> None:
            self._stream_payload(self._payload)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path or "/"
            qs = parse_qs(parsed.query)
            if path == "/":
                scenario = (qs.get("demo") or [None])[0]
                static_shot = "static" in qs
                html = render_cockpit_html(
                    profile=profile,
                    demo_scenario=scenario,
                    static_shot=static_shot,
                )
                self._send(HTTPStatus.OK, html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/status":
                body = json.dumps(self._payload(), indent=2, default=str).encode("utf-8")
                self._send(HTTPStatus.OK, body, "application/json; charset=utf-8")
                return
            if path == "/api/demo":
                scenario = (qs.get("scenario") or ["healthy"])[0]
                try:
                    body = json.dumps(self._demo_payload(scenario), indent=2, default=str).encode("utf-8")
                except KeyError as exc:
                    self._send(HTTPStatus.BAD_REQUEST, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
                    return
                self._send(HTTPStatus.OK, body, "application/json; charset=utf-8")
                return
            if path == "/api/stream":
                self._stream_status()
                return
            if path == "/api/demo-stream":
                scenario = (qs.get("scenario") or ["healthy"])[0]
                try:
                    build_demo_metrics(scenario)  # validate early
                except KeyError as exc:
                    self._send(HTTPStatus.BAD_REQUEST, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
                    return
                self._stream_payload(lambda: self._demo_payload(scenario))
                return
            if path == "/api/demo-scenarios":
                body = json.dumps({"scenarios": list(DEMO_SCENARIOS)}).encode("utf-8")
                self._send(HTTPStatus.OK, body, "application/json; charset=utf-8")
                return
            if path == "/operator-guide":
                guide = _operator_guide_path()
                text = guide.read_text() if guide else FALLBACK_OPERATOR_GUIDE
                html = render_operator_guide_html(text)
                self._send(HTTPStatus.OK, html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/healthz":
                expected = qs.get("profile", [profile])[0]
                ok = expected == profile
                body = json.dumps({"ok": ok, "profile": profile}).encode("utf-8")
                self._send(HTTPStatus.OK if ok else HTTPStatus.CONFLICT, body, "application/json; charset=utf-8")
                return
            if path == "/favicon.ico":
                self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
                return
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    return CockpitHandler


def serve_context_cockpit(
    *,
    profile: str,
    profile_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8421,
    open_browser_on_start: bool = False,
) -> int:
    server = ThreadingHTTPServer((host, port), make_handler(profile=profile, profile_dir=profile_dir))
    server.daemon_threads = True
    if open_browser_on_start:
        thread = threading.Thread(target=open_browser, args=(build_cockpit_url(host, port),), daemon=True)
        thread.start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
