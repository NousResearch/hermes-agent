#!/usr/bin/env python3
"""Learning Progress Dashboard for Hermes × OpenClaw.

Aggregates signals from agent_bus DB, wiki, HF cards, candidate queue, and
coaching sessions to produce a snapshot of dual-agent learning progress.

Usage:
    python scripts/learning_progress_dashboard.py
    python scripts/learning_progress_dashboard.py --write   # also write to wiki
    python scripts/learning_progress_dashboard.py --phase 1 # filter exit criteria

Spec: ~/wiki/operations/learning-progress-dashboard-spec.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

WIKI = Path.home() / "wiki"
BUS_DB = Path.home() / ".openclaw" / "workspace" / ".agent-bus" / "agent_bus.db"
HERMES_LOG = Path.home() / ".hermes" / "logs"
COACHING_SESSION = WIKI / "operations" / "coaching-2026-04-22-phase1-kickoff.md"


# ---------- Signal collectors ----------
def bus_task_summary() -> Dict[str, Any]:
    if not BUS_DB.exists():
        return {"error": f"bus DB missing: {BUS_DB}"}
    conn = sqlite3.connect(str(BUS_DB))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status"
        ).fetchall()
        status_counts = {r["status"]: r["n"] for r in rows}
        open_count = sum(
            n for s, n in status_counts.items()
            if s not in ("done", "fail", "timeout")
        )
        # finalizer-related events last 7d
        cutoff = time.time() - 7 * 86400
        events = conn.execute(
            """SELECT event_type, COUNT(*) AS n
               FROM task_events WHERE ts > ? GROUP BY event_type""",
            (cutoff,),
        ).fetchall()
        event_counts = {r["event_type"]: r["n"] for r in events}
        # keep-alive usage
        keep_alive_count = event_counts.get("keep_alive", 0)
        amend_count = event_counts.get("amend_learning", 0)
        # idempotent / rejected: those go to logger, not events. skip for now.
        return {
            "status_counts": status_counts,
            "open_count": open_count,
            "events_7d": event_counts,
            "keep_alive_7d": keep_alive_count,
            "amend_learning_7d": amend_count,
        }
    finally:
        conn.close()


def evolution_notes_summary() -> Dict[str, Any]:
    """Count *_evolution_*.md by direction (via source: frontmatter)."""
    memdir = WIKI / "memory"
    if not memdir.exists():
        return {"error": f"memory dir missing: {memdir}"}
    files = sorted(memdir.glob("*_evolution_*.md"))
    outbound = []  # Hermes → OpenClaw
    inbound = []  # OpenClaw → Hermes
    unknown = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")[:600]
        m = re.search(r"^source:\s*(\w+)", text, re.MULTILINE)
        t = re.search(r"^target:\s*(\w+)", text, re.MULTILINE)
        src = m.group(1).lower() if m else None
        tgt = t.group(1).lower() if t else None
        if src == "hermes" and tgt == "openclaw":
            outbound.append(f.name)
        elif src == "openclaw" and tgt == "hermes":
            inbound.append(f.name)
        else:
            # fallback: filename-based heuristic (non-canonical)
            unknown.append(f.name)

    # "This week" (last 7d by filename date)
    today = date.today()
    week_ago = today - timedelta(days=7)

    def in_week(name: str) -> bool:
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})_", name)
        if not m:
            return False
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return week_ago <= d <= today

    return {
        "total_outbound_hermes_to_openclaw": len(outbound),
        "total_inbound_openclaw_to_hermes": len(inbound),
        "total_unknown_direction": len(unknown),
        "this_week_outbound": sum(1 for n in outbound if in_week(n)),
        "this_week_inbound": sum(1 for n in inbound if in_week(n)),
        "outbound_files": outbound[-5:],  # most recent 5
        "inbound_files": inbound[-5:],
        "m2_symmetry_ok": len(inbound) >= len(outbound) * 0.25,
    }


def hf_card_summary() -> Dict[str, Any]:
    hfdir = WIKI / "operations" / "automode" / "handoffs"
    if not hfdir.exists():
        return {"error": f"hf dir missing: {hfdir}"}
    files = list(hfdir.glob("*.md"))
    by_status: Dict[str, List[str]] = {}
    ages_pending: List[float] = []
    now = time.time()
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")[:500]
        m = re.search(r"^status:\s*(\S+)", text, re.MULTILINE)
        status = m.group(1).lower() if m else "unknown"
        by_status.setdefault(status, []).append(f.name)
        if "pending" in status:
            ages_pending.append((now - f.stat().st_mtime) / 3600)
    oldest_pending_hr = max(ages_pending) if ages_pending else 0.0
    return {
        "total": len(files),
        "by_status": {s: len(v) for s, v in by_status.items()},
        "pending_files": by_status.get("pending-acceptance", []),
        "oldest_pending_hours": round(oldest_pending_hr, 1),
        "health_ok": not by_status.get("pending-acceptance")
        or oldest_pending_hr < 72,
    }


def queue_decay_summary() -> Dict[str, Any]:
    script = Path.home() / "dev" / "hermes-agent" / "scripts" / "queue_decay_check.py"
    if not script.exists():
        return {"error": "queue_decay_check.py not found"}
    try:
        out = subprocess.check_output(
            [sys.executable, str(script)],
            text=True, timeout=10,
        )
    except Exception as e:
        return {"error": f"run failed: {e}"}
    # Parse the summary lines
    m_fresh = re.search(r"fresh:\s*(\d+)", out)
    m_stale = re.search(r"stale:\s*(\d+)", out)
    m_force = re.search(r"force-review:\s*(\d+)", out)
    m_pilot = re.search(r"pilot-at-risk:\s*(\d+)", out)
    m_term = re.search(r"terminal:\s*(\d+)", out)
    m_total = re.search(r"Total rows:\s*(\d+)", out)
    return {
        "total": int(m_total.group(1)) if m_total else 0,
        "fresh": int(m_fresh.group(1)) if m_fresh else 0,
        "stale": int(m_stale.group(1)) if m_stale else 0,
        "force_review": int(m_force.group(1)) if m_force else 0,
        "pilot_at_risk": int(m_pilot.group(1)) if m_pilot else 0,
        "terminal": int(m_term.group(1)) if m_term else 0,
    }


def sync_bridge_summary() -> Dict[str, Any]:
    probe = Path.home() / "dev" / "hermes-agent" / "scripts" / "sync-bridge-health-probe.sh"
    if not probe.exists():
        return {"error": "probe script missing"}
    try:
        out = subprocess.check_output(
            ["bash", str(probe)], text=True, timeout=15, stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        out = e.output
    try:
        # First line of output is JSON
        first = out.strip().split("\n")[0]
        return json.loads(first)
    except Exception as e:
        return {"error": f"parse failed: {e}", "raw": out[:400]}


def coaching_session_summary() -> Dict[str, Any]:
    if not COACHING_SESSION.exists():
        return {"error": "coaching session file missing"}
    text = COACHING_SESSION.read_text(encoding="utf-8")
    # Count OpenClaw blanks vs filled
    blanks = text.count("_（待回填，T-COACH01）_")
    blanks += text.count("_（待回填）_")
    # Empty-chair predictions
    empty_chair_count = text.count("Coach 的 empty-chair 預測")
    empty_chair_count += text.count("Coach empty-chair")
    return {
        "blanks_openclaw": blanks,
        "empty_chair_filled": empty_chair_count,
        "phase": 1,
        "week": 1,
    }


def openclaw_liveness() -> Dict[str, Any]:
    try:
        out = subprocess.check_output(
            ["launchctl", "print", f"gui/{os.getuid()}/ai.openclaw.gateway"],
            text=True, stderr=subprocess.STDOUT, timeout=5,
        )
    except Exception as e:
        return {"error": str(e)}
    state = "unknown"
    if re.search(r"^\s*state\s*=\s*running", out, re.MULTILINE):
        state = "running"
    elif re.search(r"^\s*state\s*=\s*not running", out, re.MULTILINE):
        state = "not-running"
    last_exit_m = re.search(r"last exit code = (\S+)", out)
    # codex app-server presence
    try:
        ps = subprocess.check_output(["pgrep", "-lf", "codex app-server"], text=True, timeout=3)
        codex_running = bool(ps.strip())
    except subprocess.CalledProcessError:
        codex_running = False
    return {
        "gateway_state": state,
        "last_exit_code": last_exit_m.group(1) if last_exit_m else "?",
        "codex_app_server_running": codex_running,
    }


# ---------- Rendering ----------
def render_markdown(data: Dict[str, Any]) -> str:
    ts = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M %z")

    lines: List[str] = []
    lines.append(f"# Learning Progress Dashboard\n")
    lines.append(f"**Generated**: {ts}")
    lines.append(f"**Phase**: 1 (2026-04-21 → 2026-05-19)  |  **Week**: 1-2")
    lines.append("")

    # --- OpenClaw liveness ---
    live = data["openclaw"]
    lines.append("## 🟢 OpenClaw liveness")
    if "error" in live:
        lines.append(f"- ⚠️ {live['error']}")
    else:
        emoji = "🟢" if live["gateway_state"] == "running" else "🔴"
        lines.append(f"- Gateway: {emoji} `{live['gateway_state']}`")
        lines.append(f"- Last exit code: `{live['last_exit_code']}`")
        emoji = "🟢" if live["codex_app_server_running"] else "🔴"
        lines.append(f"- Codex app-server: {emoji}")
    lines.append("")

    # --- Sync bridge ---
    sync = data["sync_bridge"]
    lines.append("## 🔄 Sync bridge (se-007)")
    if "error" in sync:
        lines.append(f"- ⚠️ {sync['error']}")
    else:
        emoji = "🟢" if sync.get("overall_ok") else "🔴"
        lines.append(f"- Overall: {emoji}")
        for c in sync.get("checks", []):
            e = "🟢" if c["ok"] else "🔴"
            lines.append(f"  - {e} **{c['id']}** — {c['detail']}")
    lines.append("")

    # --- Bus tasks ---
    bus = data["bus"]
    lines.append("## 🚌 Agent bus (se-013 / se-015 / se-017)")
    if "error" in bus:
        lines.append(f"- ⚠️ {bus['error']}")
    else:
        emoji = "🟢" if bus["open_count"] == 0 else ("🟡" if bus["open_count"] < 3 else "🔴")
        lines.append(f"- Open tasks: {emoji} {bus['open_count']}")
        lines.append(f"- Status distribution: {bus['status_counts']}")
        lines.append(f"- Events last 7d: {bus['events_7d']}")
        lines.append(f"- keep-alive 7d: {bus['keep_alive_7d']}  |  amend_learning 7d: {bus['amend_learning_7d']}")
    lines.append("")

    # --- Evolution notes ---
    ev = data["evolution"]
    lines.append("## 📘 Cross-agent evolution notes (M2 symmetry)")
    if "error" in ev:
        lines.append(f"- ⚠️ {ev['error']}")
    else:
        lines.append(f"- Total **Hermes → OpenClaw**: {ev['total_outbound_hermes_to_openclaw']}")
        lines.append(f"- Total **OpenClaw → Hermes**: {ev['total_inbound_openclaw_to_hermes']}")
        lines.append(f"- Unknown direction (no frontmatter): {ev['total_unknown_direction']}")
        lines.append(f"- This week outbound: {ev['this_week_outbound']}  |  inbound: {ev['this_week_inbound']}")
        emoji = "🟢" if ev["m2_symmetry_ok"] else "🔴"
        lines.append(f"- M2 symmetry (inbound ≥ 25% outbound): {emoji}")
        if ev.get("inbound_files"):
            lines.append(f"- Recent inbound:")
            for f in ev["inbound_files"]:
                lines.append(f"  - `{f}`")
    lines.append("")

    # --- HF cards ---
    hf = data["hf"]
    lines.append("## 🃏 HF cards (se-012)")
    if "error" in hf:
        lines.append(f"- ⚠️ {hf['error']}")
    else:
        emoji = "🟢" if hf["health_ok"] else "🔴"
        lines.append(f"- Total cards: {hf['total']}")
        lines.append(f"- By status: {hf['by_status']}")
        lines.append(f"- Oldest pending: {hf['oldest_pending_hours']}h  {emoji}")
        if hf["pending_files"]:
            lines.append(f"- Pending files:")
            for f in hf["pending_files"]:
                lines.append(f"  - `{f}`")
    lines.append("")

    # --- Queue ---
    q = data["queue"]
    lines.append("## 📋 Candidate queue (M3 decay)")
    if "error" in q:
        lines.append(f"- ⚠️ {q['error']}")
    else:
        lines.append(f"- Total: {q['total']}")
        lines.append(f"- Fresh: {q['fresh']}  |  Stale: {q['stale']}  |  Force-review: {q['force_review']}")
        lines.append(f"- Pilot-at-risk: {q['pilot_at_risk']}  |  Terminal: {q['terminal']}")
        emoji = "🟢" if q["force_review"] == 0 and q["pilot_at_risk"] == 0 else "🟡"
        lines.append(f"- Health: {emoji}")
    lines.append("")

    # --- Coaching ---
    co = data["coaching"]
    lines.append("## 🎓 Coaching session (Phase 1 Kickoff)")
    if "error" in co:
        lines.append(f"- ⚠️ {co['error']}")
    else:
        lines.append(f"- OpenClaw blanks待回填: {co['blanks_openclaw']}")
        lines.append(f"- Empty-chair predictions: {co['empty_chair_filled']}")
        emoji = "🟡" if co["blanks_openclaw"] > 0 else "🟢"
        lines.append(f"- Status: {emoji}")
    lines.append("")

    # --- Daily review freshness ---
    dr = data["daily_review"]
    lines.append("## 📝 Daily review freshness")
    if "error" in dr:
        lines.append(f"- ⚠️ {dr['error']}")
    else:
        emoji = "🟢" if dr["today_done"] else ("🟡" if dr["yesterday_covered"] else "🔴")
        lines.append(f"- Hermes last reviewed: {dr['hermes_last_reviewed']}  {emoji}")
        lines.append(f"- Latest change log date: {dr['latest_change_log_date']}")
    lines.append("")

    # --- Git divergence ---
    gd = data.get("git_divergence", {})
    lines.append("## 🔀 Git divergence（hermes-agent）")
    if "error" in gd:
        lines.append(f"- ⚠️ {gd['error']}")
    else:
        ahead = gd.get("ahead", 0) or 0
        behind = gd.get("behind", 0) or 0
        dups = gd.get("duplicate_subjects", []) or []
        worktrees = gd.get("worktree_count", 0)
        last_fetch = gd.get("last_fetch_sec")
        emoji = "🟢"
        if behind > 20 or len(dups) > 0:
            emoji = "🔴"
        elif behind > 5 or ahead > 5:
            emoji = "🟡"
        lines.append(f"- main: ahead {ahead} · behind {behind}  {emoji}")
        if last_fetch is not None:
            h = last_fetch // 3600
            lines.append(f"- origin/main 上次更新: {h}h 前（> 24h 代表 fetch 久未跑）")
        lines.append(f"- worktrees: {worktrees}")
        if dups:
            lines.append(f"- **cherry-pick 重複警示**: {len(dups)} 個 subject 雙邊都有：")
            for s in dups[:5]:
                lines.append(f"  - `{s[:80]}`")
    lines.append("")

    # --- Rate-limit events ---
    rl = data["ratelimit"]
    lines.append("## ⚡ Codex rate-limit events (§9 throttle efficacy)")
    if "error" in rl:
        lines.append(f"- ⚠️ {rl['error']}")
    else:
        emoji = "🟢" if rl["last_7d"] == 0 else ("🟡" if rl["last_7d"] < 3 else "🔴")
        lines.append(f"- Last 24h: {rl['last_24h']}  |  Last 7d: {rl['last_7d']}  |  Total ever: {rl['total_ever']}  {emoji}")
        if rl["recent_files"]:
            lines.append(f"- Recent events:")
            for f in rl["recent_files"]:
                lines.append(f"  - `{f}`")
    lines.append("")

    # --- Scorecard ---
    lines.append("## 🏆 Weekly Scorecard")
    scorecard = _compute_scorecard(data)
    lines.append(f"- **Score**: {scorecard['score']}/10")
    lines.append(f"- **Trend**: {scorecard['trend']}")
    lines.append("")
    for name, (emoji, reason) in scorecard["breakdown"].items():
        lines.append(f"- {emoji} **{name}** — {reason}")
    lines.append("")

    # --- Next actions ---
    lines.append("## 🎯 Recommended next actions")
    for a in scorecard["next_actions"]:
        lines.append(f"- {a}")
    lines.append("")

    return "\n".join(lines)


def _compute_scorecard(data: Dict[str, Any]) -> Dict[str, Any]:
    breakdown: Dict[str, Tuple[str, str]] = {}
    score = 0

    # OpenClaw liveness (2)
    live = data["openclaw"]
    if live.get("gateway_state") == "running" and live.get("codex_app_server_running"):
        breakdown["OpenClaw liveness"] = ("🟢", "gateway + codex 雙活")
        score += 2
    elif live.get("gateway_state") == "running":
        breakdown["OpenClaw liveness"] = ("🟡", "gateway 活著但 codex 不在")
        score += 1
    else:
        breakdown["OpenClaw liveness"] = ("🔴", "gateway 沒跑")

    # Sync bridge (2)
    if data["sync_bridge"].get("overall_ok"):
        breakdown["Sync bridge"] = ("🟢", "4 check 全綠")
        score += 2
    else:
        breakdown["Sync bridge"] = ("🔴", "有 check 紅")

    # Bus health (2)
    bus = data["bus"]
    if bus.get("open_count", 99) == 0:
        breakdown["Bus close hygiene"] = ("🟢", "無 open task 漏關")
        score += 2
    elif bus.get("open_count", 99) < 3:
        breakdown["Bus close hygiene"] = ("🟡", f"{bus['open_count']} open task")
        score += 1
    else:
        breakdown["Bus close hygiene"] = ("🔴", f"{bus['open_count']} open task 漏關")

    # HF cards (1) — Rule 8 checklist requires 72h alarm
    hf = data["hf"]
    oldest = hf.get("oldest_pending_hours", 0) or 0
    if oldest >= 72:
        breakdown["HF discipline"] = ("🔴", f"⚠️ Rule 8 72h 例外條件觸發：最老 pending {oldest}h")
    elif oldest >= 48:
        breakdown["HF discipline"] = ("🟡", f"最老 pending {oldest}h（逼近 72h）")
        score += 1  # still counts but warn
    elif hf.get("health_ok"):
        breakdown["HF discipline"] = ("🟢", f"無超期 pending card（最老 {oldest}h）")
        score += 1
    else:
        breakdown["HF discipline"] = ("🔴", f"最老 pending {oldest}h")

    # M2 symmetry (2)
    ev = data["evolution"]
    if ev.get("this_week_inbound", 0) >= 1:
        breakdown["M2 symmetry"] = ("🟢", f"本週 inbound {ev['this_week_inbound']} 篇")
        score += 2
    elif ev.get("m2_symmetry_ok"):
        breakdown["M2 symmetry"] = ("🟡", "歷史平均 OK 但本週 inbound = 0")
        score += 1
    else:
        breakdown["M2 symmetry"] = ("🔴", f"inbound 總數 {ev.get('total_inbound_openclaw_to_hermes', 0)}，遠低於 outbound")

    # Queue decay (1)
    q = data["queue"]
    if q.get("force_review", 99) == 0 and q.get("pilot_at_risk", 99) == 0:
        breakdown["Queue decay"] = ("🟢", "無 force-review / pilot-at-risk")
        score += 1
    else:
        breakdown["Queue decay"] = ("🟡", "有待裁決候選")

    # Recommend actions based on gaps
    actions = []
    if ev.get("this_week_inbound", 0) < 1:
        actions.append("OpenClaw 寫本週第一篇 inbound `*_evolution_*.md`（主題建議：openclaw-outbound-hook，見 coaching session empty-chair Q2.3）")
    if data["coaching"].get("blanks_openclaw", 0) > 0:
        actions.append(f"OpenClaw 回填 coaching session 的 {data['coaching']['blanks_openclaw']} 個待填答")
    if oldest >= 72:
        actions.append(
            f"⚠️ Rule 8 72h 例外條件已觸發（最老 pending {oldest}h）— 觸發 Brian 拍板流程"
        )
    elif oldest >= 48:
        actions.append(f"逼近 Rule 8 72h 閾值（最老 pending {oldest}h），48h 內須處理")
    elif not hf.get("health_ok"):
        actions.append(f"處理超期 pending HF card（最老 {oldest}h）")
    if bus.get("open_count", 0) > 0:
        actions.append(f"清 bus {bus['open_count']} 個 open task，驗證 finalizer gate")
    if q.get("force_review", 0) > 0:
        actions.append(f"裁決 {q['force_review']} 個 force-review 候選")
    if not actions:
        actions.append("全部指標綠燈；本週可專注 Phase 1 Week 2 exit criteria")

    # Trend computation
    trend_info = data.get("trend", {})
    prev_score = trend_info.get("previous_score")
    prev_file = trend_info.get("previous_file")
    prev_signals = trend_info.get("previous_signals", {})

    if prev_score is None:
        trend = "🔰 initial (no previous snapshot)"
    else:
        delta = score - prev_score
        if delta > 0:
            trend = f"📈 +{delta} vs `{prev_file}` (was {prev_score}/10)"
        elif delta < 0:
            trend = f"📉 {delta} vs `{prev_file}` (was {prev_score}/10)"
        else:
            trend = f"➡️ unchanged vs `{prev_file}` ({prev_score}/10)"

        # Flipped signals
        flips = []
        for name, (emoji, _reason) in breakdown.items():
            prev_emoji = prev_signals.get(name)
            if prev_emoji and prev_emoji != emoji:
                flips.append(f"{name}: {prev_emoji}→{emoji}")
        if flips:
            trend += f"  |  flips: {', '.join(flips)}"

    return {
        "score": score,
        "max": 10,
        "breakdown": breakdown,
        "trend": trend,
        "next_actions": actions,
    }


# ---------- Extra probes (v2) ----------
def ratelimit_event_summary() -> Dict[str, Any]:
    """Count codex rate-limit events in ~/wiki/memory/ per §9 throttle rule."""
    memdir = WIKI / "memory"
    if not memdir.exists():
        return {"error": "memory dir missing"}
    files = list(memdir.glob("*_ratelimit_event.md"))
    today = date.today()
    last_7d = [f for f in files
               if _filename_date(f.name) and (today - _filename_date(f.name)).days <= 7]
    last_24h = [f for f in files
                if _filename_date(f.name) == today]
    return {
        "total_ever": len(files),
        "last_7d": len(last_7d),
        "last_24h": len(last_24h),
        "recent_files": sorted([f.name for f in last_7d])[-5:],
    }


def _filename_date(name: str):
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})_", name)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def daily_review_freshness() -> Dict[str, Any]:
    """Has Hermes done a daily review today? Look at the status change log."""
    status_path = WIKI / "operations" / "ai-evolution-status.md"
    if not status_path.exists():
        return {"error": "status file missing"}
    text = status_path.read_text(encoding="utf-8")

    today_str = date.today().strftime("%Y-%m-%d")
    yday_str = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Dates mentioned in change log headings
    heading_dates = sorted({m.group(1) for m in re.finditer(
        r"^###\s+(\d{4}-\d{2}-\d{2})", text, re.MULTILINE
    )}, reverse=True)
    last_review = heading_dates[0] if heading_dates else None

    # Hermes Last reviewed at field
    m = re.search(r"\|\s*Hermes\s*\|\s*(\d{4}-\d{2}-\d{2})[^|]*\|", text)
    hermes_last = m.group(1) if m else None

    today_done = (last_review == today_str) or (hermes_last == today_str)
    return {
        "hermes_last_reviewed": hermes_last,
        "latest_change_log_date": last_review,
        "today_done": today_done,
        "yesterday_covered": (last_review == yday_str) or (hermes_last == yday_str),
    }


def git_divergence_summary() -> Dict[str, Any]:
    """Check local main vs origin/main divergence for the hermes-agent repo.

    Signals:
    - ahead:    commits on local main not on origin/main
    - behind:   commits on origin/main not on local main
    - last_fetch_sec: seconds since last `git fetch`
    - duplicate_candidates: commits on local that share a subject line with
      commits on origin (cherry-pick signal)
    """
    import subprocess
    repo = Path.home() / "dev" / "hermes-agent"
    if not (repo / ".git").exists():
        return {"error": "repo not found"}

    def run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(
                cmd, cwd=repo, text=True, stderr=subprocess.DEVNULL, timeout=5,
            ).strip()
        except Exception:
            return ""

    # Don't auto-fetch from the dashboard — just read state
    ahead = run(["git", "rev-list", "--count", "origin/main..main"])
    behind = run(["git", "rev-list", "--count", "main..origin/main"])
    last_fetch_epoch = run(["git", "log", "-1", "--format=%ct", "origin/main"])

    # Cherry-pick duplicate detection: subjects present on both sides
    local_subjects = run(["git", "log", "origin/main..main", "--format=%s"]).splitlines()
    origin_subjects = set(
        run(["git", "log", "main..origin/main", "--format=%s"]).splitlines()
    )
    duplicates = [s for s in local_subjects if s in origin_subjects]

    # Worktree count (informational)
    worktree_list = run(["git", "worktree", "list"])
    worktree_count = len([ln for ln in worktree_list.splitlines() if ln.strip()])

    try:
        last_fetch_sec = int(time.time()) - int(last_fetch_epoch) if last_fetch_epoch else None
    except (TypeError, ValueError):
        last_fetch_sec = None

    return {
        "ahead": int(ahead) if ahead.isdigit() else None,
        "behind": int(behind) if behind.isdigit() else None,
        "last_fetch_sec": last_fetch_sec,
        "duplicate_subjects": duplicates,
        "worktree_count": worktree_count,
    }


def trend_analysis() -> Dict[str, Any]:
    """Compare current with previous snapshot — score delta + signal flips."""
    snap_dir = WIKI / "operations" / "learning-progress-snapshots"
    if not snap_dir.exists():
        return {"error": "no snapshot dir yet"}
    snapshots = sorted(snap_dir.glob("*.md"))
    # exclude the file we're about to write (by timestamp closeness)
    if not snapshots:
        return {"previous": None, "delta": None}

    prev = snapshots[-1]
    # Parse score and signal colors from previous snapshot
    text = prev.read_text(encoding="utf-8")

    m = re.search(r"\*\*Score\*\*:\s*(\d+)/10", text)
    prev_score = int(m.group(1)) if m else None

    prev_signals: Dict[str, str] = {}
    # match lines like: "- 🟢 **OpenClaw liveness** — gateway + codex 雙活"
    for m in re.finditer(
        r"^-\s+(🟢|🟡|🔴)\s+\*\*([^\*]+?)\*\*\s+—", text, re.MULTILINE
    ):
        prev_signals[m.group(2).strip()] = m.group(1)

    return {
        "previous_file": prev.name,
        "previous_score": prev_score,
        "previous_signals": prev_signals,
    }


# ---------- Main ----------
def collect_all() -> Dict[str, Any]:
    return {
        "ts": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "openclaw": openclaw_liveness(),
        "sync_bridge": sync_bridge_summary(),
        "bus": bus_task_summary(),
        "evolution": evolution_notes_summary(),
        "hf": hf_card_summary(),
        "queue": queue_decay_summary(),
        "coaching": coaching_session_summary(),
        "ratelimit": ratelimit_event_summary(),
        "daily_review": daily_review_freshness(),
        "git_divergence": git_divergence_summary(),
        "trend": trend_analysis(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write snapshot under ~/wiki/operations/learning-progress-snapshots/")
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    ap.add_argument("--live-only", action="store_true",
                    help="skip probes that take >1s (sync bridge, queue decay)")
    args = ap.parse_args()

    data = collect_all()
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        return 0

    md = render_markdown(data)
    print(md)

    if args.write:
        out_dir = WIKI / "operations" / "learning-progress-snapshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%dT%H%M")
        path = out_dir / f"{stamp}.md"
        path.write_text(md, encoding="utf-8")
        print(f"\n[saved snapshot: {path}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
