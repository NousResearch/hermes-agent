"""Safe proactive personal-assistant helpers for Hermes.

This module intentionally does not make the agent proactive by itself. It builds
and installs a cron job with a fast structured signal scan plus a small judgment
prompt so users can opt in to periodic synthesis without granting permission to
act externally.
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal

from cron.jobs import create_job, list_jobs, update_job
from hermes_constants import get_hermes_home

DEFAULT_JOB_NAME = "Proactive synthesis / safe nudges"
DEFAULT_SCHEDULE = "0 9 * * *"
DEFAULT_DELIVER = "local"
# The pre-run scanner handles session search. Keep the agent's tool surface small
# so cron runs do not wander for minutes through session_search.
DEFAULT_ENABLED_TOOLSETS = ["memory"]
DEFAULT_SCANNER_SCRIPT = "proactive_signal_scan.py"
DEFAULT_COOLDOWN_HOURS = 72
DEFAULT_SNOOZE_HOURS = 24
_ALLOWED_CONFIDENCE = {"medium", "high"}
_MAX_EXCERPT_CHARS = 280

_SCAN_BUCKETS = [
    {
        "kind": "content_opportunity",
        "query": '"meeting notes" OR speech OR "sales team" OR critique OR "X posts" OR delivery OR Notion',
        "mode": "offer_to_produce",
        "reason": "fresh notes/content may create a useful draft, critique, or synthesis opportunity",
    },
    {
        "kind": "blocker_or_waiting",
        "query": 'blocked OR blocker OR waiting OR failed OR failure OR TestFlight OR "Xcode ready" OR gateway',
        "mode": "ask_or_checked",
        "reason": "possible blocker or waiting item that may need a next action",
    },
    {
        "kind": "follow_up_or_watch",
        "query": '"want me" OR "look into" OR "follow up" OR reminder OR proactive OR "More Info"',
        "mode": "ask_to_investigate",
        "reason": "possible user-requested follow-up, watch item, or permission-shaped opportunity",
    },
    {
        "kind": "external_risk",
        "query": 'Stripe OR payment OR purchase OR order OR email OR post OR DM OR calendar OR "App Store Connect"',
        "mode": "ask_first",
        "reason": "external/money/reputation risk; draft or ask only, never act externally",
    },
]

_SUPPRESSION_QUERY = 'OwnerPath OR "on hold" OR paused OR "do not nudge" OR "don\'t nudge" OR "not useful"'


@dataclass(frozen=True)
class ProactivePromptOptions:
    lookback_days: int = 7
    max_sessions: int = 30
    min_confidence: Literal["medium", "high"] = "high"

    def normalized(self) -> "ProactivePromptOptions":
        lookback_days = max(1, min(int(self.lookback_days), 90))
        max_sessions = max(1, min(int(self.max_sessions), 200))
        min_confidence = str(self.min_confidence or "high").lower()
        if min_confidence not in _ALLOWED_CONFIDENCE:
            min_confidence = "high"
        return ProactivePromptOptions(
            lookback_days=lookback_days,
            max_sessions=max_sessions,
            min_confidence=min_confidence,  # type: ignore[arg-type]
        )


def _clip(value: Any, limit: int = _MAX_EXCERPT_CHARS) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _looks_machine_text(value: Any) -> bool:
    text = _clip(value, 500).lstrip()
    lower = text.lower()
    if not text:
        return True
    if text.startswith(("{", "[{", "[TOOL]", "[Called:")):
        return True
    machine_markers = (
        '"success":',
        '"call_id"',
        '"tool_call_id"',
        '"bytes_written"',
        '"exit_code"',
        '"files_modified"',
        "response_item_id",
        "tool_use",
    )
    return any(marker in lower for marker in machine_markers)


def _iso(ts: Any) -> str | None:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _recent(ts: Any, cutoff: float) -> bool:
    try:
        return float(ts) >= cutoff
    except Exception:
        return True


def _signal_key(signal: Dict[str, Any]) -> tuple:
    return (
        signal.get("kind"),
        signal.get("session_id"),
        _clip(signal.get("excerpt"), 100).lower(),
    )


def _ledger_path() -> Path:
    return get_hermes_home() / "proactive" / "ledger.json"


def _load_ledger() -> Dict[str, Any]:
    path = _ledger_path()
    if not path.exists():
        return {"version": 1, "nudges": [], "feedback": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("ledger root is not an object")
        data.setdefault("version", 1)
        data.setdefault("nudges", [])
        data.setdefault("feedback", [])
        if not isinstance(data["nudges"], list):
            data["nudges"] = []
        if not isinstance(data["feedback"], list):
            data["feedback"] = []
        return data
    except Exception:
        corrupt = path.with_suffix(f".corrupt-{int(time.time())}.json")
        try:
            path.replace(corrupt)
        except Exception:
            pass
        return {"version": 1, "nudges": [], "feedback": []}


def _save_ledger(ledger: Dict[str, Any]) -> None:
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _signal_fingerprint(signal: Dict[str, Any]) -> str:
    basis = "|".join(
        [
            _clip(signal.get("kind"), 80).lower(),
            _clip(signal.get("source"), 80).lower(),
            _clip(signal.get("session_id"), 120).lower(),
            _clip(signal.get("excerpt"), 220).lower(),
        ]
    )
    return hashlib.sha256(basis.encode("utf-8", "replace")).hexdigest()[:16]


def _top_signal(scan_report: Dict[str, Any]) -> Dict[str, Any]:
    signals = scan_report.get("signals") or []
    if signals and isinstance(signals[0], dict):
        return dict(signals[0])
    errors = scan_report.get("scan_errors") or []
    if errors:
        return {
            "kind": "scan_error",
            "mode": "already_checked",
            "reason": "proactive scanner reported an error",
            "excerpt": _clip(errors[0]),
        }
    return {"kind": "unknown", "mode": "ask_to_investigate", "reason": "proactive nudge", "excerpt": ""}


def apply_ledger_filters(
    report: Dict[str, Any],
    *,
    now: float | None = None,
    cooldown_hours: int = DEFAULT_COOLDOWN_HOURS,
) -> Dict[str, Any]:
    """Suppress recently nudged, snoozed, or explicitly muted signals."""

    now = time.time() if now is None else float(now)
    cooldown_secs = max(0, int(cooldown_hours)) * 3600
    ledger = _load_ledger()
    nudges = [n for n in ledger.get("nudges", []) if isinstance(n, dict)]
    filtered = json.loads(json.dumps(report))
    kept: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []

    for signal in filtered.get("signals") or []:
        if not isinstance(signal, dict):
            continue
        fp = _signal_fingerprint(signal)
        match = None
        reason = ""
        for nudge in reversed(nudges):
            if nudge.get("fingerprint") != fp:
                continue
            if nudge.get("muted") or nudge.get("last_feedback") in {"dont", "not"}:
                match, reason = nudge, "muted"
                break
            snoozed_until = float(nudge.get("snoozed_until") or 0)
            if snoozed_until > now:
                match, reason = nudge, "snoozed"
                break
            created_at = float(nudge.get("created_at") or 0)
            if cooldown_secs and created_at and now - created_at < cooldown_secs:
                match, reason = nudge, "cooldown"
                break
        if match:
            suppressed.append({"nudge_id": match.get("id"), "reason": reason, "signal": signal})
        else:
            kept.append(signal)

    filtered["signals"] = kept
    if suppressed:
        filtered["suppressed_by_ledger"] = suppressed
    filtered["wakeAgent"] = bool(kept or filtered.get("scan_errors"))
    return filtered


def record_proactive_nudge(
    *,
    job: Dict[str, Any],
    message: str,
    scan_report: Dict[str, Any],
    now: float | None = None,
) -> Dict[str, Any]:
    """Persist a delivered proactive nudge so future scans can cool it down."""

    now = time.time() if now is None else float(now)
    signal = _top_signal(scan_report)
    fingerprint = _signal_fingerprint(signal)
    nudge_id = hashlib.sha256(
        f"{fingerprint}|{now:.6f}|{_clip(message, 200)}".encode("utf-8", "replace")
    ).hexdigest()[:12]
    nudge = {
        "id": nudge_id,
        "created_at": now,
        "job_id": job.get("id"),
        "job_name": job.get("name"),
        "message": _clip(message, 800),
        "fingerprint": fingerprint,
        "signal": signal,
        "status": "sent",
    }
    ledger = _load_ledger()
    ledger.setdefault("nudges", []).append(nudge)
    _save_ledger(ledger)
    return nudge


def proactive_controls_for_nudge(nudge: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "nudge_id": nudge.get("id"),
        "buttons": ["do", "more", "later", "not", "dont"],
    }


def _find_nudge(ledger: Dict[str, Any], nudge_id: str) -> Dict[str, Any] | None:
    for nudge in ledger.get("nudges", []):
        if isinstance(nudge, dict) and str(nudge.get("id")) == str(nudge_id):
            return nudge
    return None


def _detail_card(nudge: Dict[str, Any]) -> str:
    signal = nudge.get("signal") or {}
    return (
        "**Why this surfaced**\n"
        f"- Type: `{_clip(signal.get('kind'), 80)}`\n"
        f"- Reason: {_clip(signal.get('reason'), 180)}\n"
        f"- Source: {_clip(signal.get('source') or signal.get('session_id') or 'local scan', 120)}\n"
        f"- Evidence: {_clip(signal.get('excerpt'), 360)}\n\n"
        "**Suggested next action**\n"
        f"{_clip(nudge.get('message'), 500)}\n\n"
        "I can work internally from this. I still need explicit approval before sending, posting, emailing, buying, scheduling, or changing external systems."
    )


def _agent_prompt_for_nudge(nudge: Dict[str, Any]) -> str:
    signal = nudge.get("signal") or {}
    return (
        "User tapped Do it on a proactive Hermes nudge. Execute the internal next step now.\n\n"
        f"Original nudge: {_clip(nudge.get('message'), 700)}\n"
        f"Signal type: {_clip(signal.get('kind'), 80)}\n"
        f"Reason: {_clip(signal.get('reason'), 220)}\n"
        f"Evidence: {_clip(signal.get('excerpt'), 600)}\n\n"
        "Do not send, post, email, buy, schedule with other people, delete, or modify external systems without explicit approval in this chat. "
        "Draft, inspect, summarize, or prepare internal work as appropriate, then report the concrete result."
    )


def handle_proactive_feedback(
    nudge_id: str,
    action: str,
    *,
    now: float | None = None,
) -> Dict[str, Any]:
    """Apply feedback from proactive inline buttons and return UI instructions."""

    now = time.time() if now is None else float(now)
    action = str(action or "").lower().strip()
    ledger = _load_ledger()
    nudge = _find_nudge(ledger, nudge_id)
    if not nudge:
        return {"ok": False, "ack": "This proactive nudge expired."}

    nudge["last_feedback"] = action
    nudge["last_feedback_at"] = now
    event = {"nudge_id": nudge_id, "action": action, "timestamp": now}
    ledger.setdefault("feedback", []).append(event)

    result: Dict[str, Any]
    if action == "more":
        result = {"ok": True, "ack": "More info", "followup": _detail_card(nudge)}
    elif action == "later":
        nudge["snoozed_until"] = now + DEFAULT_SNOOZE_HOURS * 3600
        result = {"ok": True, "ack": "Snoozed for later."}
    elif action == "not":
        nudge["status"] = "not_useful"
        result = {"ok": True, "ack": "Got it — I’ll tune this down."}
    elif action == "dont":
        nudge["muted"] = True
        nudge["status"] = "muted"
        result = {"ok": True, "ack": "Got it — I won't nudge this again."}
    elif action == "do":
        nudge["status"] = "accepted"
        result = {"ok": True, "ack": "Starting.", "agent_prompt": _agent_prompt_for_nudge(nudge)}
    else:
        result = {"ok": False, "ack": "Unknown proactive action."}

    _save_ledger(ledger)
    return result


def extract_scan_report_from_text(text: str) -> Dict[str, Any] | None:
    """Extract the JSON proactive scan embedded in a cron output document."""

    marker = "## Proactive signal scan"
    idx = str(text or "").find(marker)
    if idx < 0:
        return None
    json_start = str(text).find("{", idx)
    if json_start < 0:
        return None
    try:
        report, _end = json.JSONDecoder().raw_decode(str(text)[json_start:])
    except Exception:
        return None
    return report if isinstance(report, dict) else None


def prepare_delivery_controls(
    *,
    job: Dict[str, Any],
    message: str,
    output_doc: str,
    now: float | None = None,
) -> Dict[str, Any] | None:
    """Record a proactive delivery and return Telegram/adapter controls metadata."""

    if job.get("name") != DEFAULT_JOB_NAME and job.get("script") != DEFAULT_SCANNER_SCRIPT:
        return None
    scan_report = extract_scan_report_from_text(output_doc)
    if not scan_report:
        return None
    nudge = record_proactive_nudge(job=job, message=message, scan_report=scan_report, now=now)
    return proactive_controls_for_nudge(nudge)


def build_reflection_prompt(
    *,
    lookback_days: int = 7,
    max_sessions: int = 30,
    min_confidence: str = "high",
) -> str:
    """Build the self-contained prompt used by the proactive cron job.

    The cron job injects structured script output before this prompt. The model's
    job is judgment, not open-ended research: decide whether to send one tiny
    proactive PA-style message or stay silent.
    """

    opts = ProactivePromptOptions(
        lookback_days=lookback_days,
        max_sessions=max_sessions,
        min_confidence=min_confidence,  # type: ignore[arg-type]
    ).normalized()

    return f"""You are Hermes running a safe proactive personal-assistant pass for Charles.

You should receive a structured "Proactive signal scan" in the script output above. Use that scan as your primary context. Do not wander through session history; the scanner already did the broad pass. If the scan has no strong candidate, say nothing.

Role:
- Act like a friendly, competent personal assistant, not an alert bot.
- Learn from recent conversations when to scan, when to stay quiet, and when to ask before spending effort.

Outcome:
- Either send one sharp, useful nudge Charles would be glad to receive, or say nothing.
- Silence is the correct answer unless the best candidate clears the bar.

Discovery already performed:
- Recent sessions: up to {opts.max_sessions} sessions from the last {opts.lookback_days} days.
- Signal buckets: content opportunities, blockers/waiting items, follow-ups/watch items, external-risk items, cron health, and suppressed topics.
- Available access/boundaries are listed in the scan. Do not pretend Hermes has access that the scan did not verify.

Proactive modes:
- Already checked: "I looked into X because I thought it would help; here's the useful bit."
- Ask to investigate: "I thought of X; want me to look into it?" Use this when helpfulness is plausible but uncertain.
- Offer to produce: "I saw Y; I can draft/critique/synthesize Z if you want." Use this for content, meeting notes, coaching, and planning.
- Silent: no message when the idea is weak, stale, intrusive, or not worth interrupting Charles.

High-signal candidates, in order:
- A time-sensitive blocker that prevents a project from shipping and has one obvious next action.
- A user-requested follow-up/reminder/watch item that is now due or newly relevant.
- A system/job/integration failure Charles likely expects Hermes to notice.
- A fresh artifact or conversation that creates an obvious assistant opportunity, e.g. meeting notes → sales coaching critique or X post drafts.
- A near-term commitment Charles explicitly owns.
- A concise synthesis that prevents repeated work or a dropped ball.

Do not send low-value nudges:
- No generic summaries, status theater, "you might want to", or obvious reminders.
- No stale ambitions, paused/on-hold work, or old projects unless Charles recently reopened them.
- No "test this sometime" unless it is blocking something Charles is actively trying to ship.
- No nudges based only on one vague mention, weak inference, or your desire to be helpful.

Safety policy:
- Send at most one proactive message.
- Only message when you have {opts.min_confidence} confidence that it is useful, timely, and wanted.
- NEVER send anything outside Hermes/the configured delivery system. Do not email, post, DM, submit forms, call APIs that publish, schedule meetings, pay, buy, delete, or modify external systems from this cron.
- Ask before any action involving money, reputation, external recipients, calendars with other people, destructive changes, or private data sharing.
- Drafting internally is allowed when low-risk; external sending is never allowed without explicit user approval in a normal interactive session.
- Do not expose private transcript details, secrets, tokens, credentials, customer data, or internal paths.
- If a proactive message would be longer than a short text, compress it to one action and offer "More Info".

Quality gate before final:
- Would Charles plausibly reply "that is not good enough" or "why are you telling me this"? If yes, output [SILENT].
- Is the next action concrete enough to do in one reply? If not, output [SILENT].
- Is this materially better than waiting for Charles to ask? If not, output [SILENT].

Output rules:
- If there is nothing worth sending, start your final response with exactly: [SILENT]
- If there is something worth sending, send only the user-facing message. No audit log, no analysis, no wrapper.
- Keep it under 80 words unless the situation is urgent.
- Include one obvious next action when possible.
""".strip()


def collect_proactive_signals(
    *,
    lookback_days: int = 7,
    max_sessions: int = 30,
) -> Dict[str, Any]:
    """Collect fast, structured proactive signals without invoking an LLM.

    This intentionally uses local state and cron metadata instead of the
    ``session_search`` tool so scheduled proactive runs stay bounded and cheap.
    """

    opts = ProactivePromptOptions(lookback_days=lookback_days, max_sessions=max_sessions).normalized()
    now = time.time()
    cutoff = now - opts.lookback_days * 86400
    report: Dict[str, Any] = {
        "generated_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "lookback_days": opts.lookback_days,
        "max_sessions": opts.max_sessions,
        "available_access": [
            "local Hermes session history (state.db)",
            "local Hermes cron job metadata",
            "built-in memory/user preferences available to the judge",
        ],
        "boundaries": {
            "can_observe": True,
            "can_summarize": True,
            "can_draft_internally": True,
            "must_ask_before_external_action": True,
            "external_send_allowed_from_cron": False,
        },
        "recent_sessions": [],
        "signals": [],
        "suppressed_topics": [],
        "scan_errors": [],
    }

    seen: set[tuple] = set()

    def add_signal(signal: Dict[str, Any]) -> None:
        key = _signal_key(signal)
        if key in seen:
            return
        seen.add(key)
        report["signals"].append(signal)

    try:
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            sessions = db.list_sessions_rich(
                limit=opts.max_sessions,
                order_by_last_active=True,
                include_children=False,
                exclude_sources=["cron"],
            )
        except TypeError:
            # Older/mocked SessionDB implementations may not accept newer args.
            sessions = db.search_sessions(limit=opts.max_sessions)
        for session in sessions:
            ts = session.get("last_active") or session.get("started_at")
            if not _recent(ts, cutoff):
                continue
            report["recent_sessions"].append(
                {
                    "id": session.get("id"),
                    "source": session.get("source"),
                    "title": session.get("title"),
                    "last_active": _iso(ts),
                    "preview": _clip(session.get("preview"), 160),
                }
            )

        for bucket in _SCAN_BUCKETS:
            try:
                matches = db.search_messages(
                    bucket["query"],
                    role_filter=["user", "assistant"],
                    exclude_sources=["cron"],
                    limit=max(20, opts.max_sessions * 2),
                )
            except Exception as exc:
                report["scan_errors"].append(f"{bucket['kind']} search failed: {exc}")
                continue
            for match in matches:
                ts = match.get("timestamp") or match.get("session_started")
                if not _recent(ts, cutoff):
                    continue
                snippet = str(match.get("snippet") or "").strip()
                context_text = " ".join(
                    _clip((ctx or {}).get("content"), 120)
                    for ctx in (match.get("context") or [])
                    if (ctx or {}).get("content") and not _looks_machine_text((ctx or {}).get("content"))
                ).strip()
                if _looks_machine_text(snippet) and not context_text:
                    continue
                content = context_text if _looks_machine_text(snippet) else (snippet or context_text)
                if not content:
                    continue
                add_signal(
                    {
                        "kind": bucket["kind"],
                        "mode": bucket["mode"],
                        "reason": bucket["reason"],
                        "session_id": match.get("session_id"),
                        "source": match.get("source"),
                        "timestamp": _iso(ts),
                        "excerpt": _clip(content),
                    }
                )
                if len(report["signals"]) >= 12:
                    break
            if len(report["signals"]) >= 12:
                break

        try:
            suppressed = db.search_messages(
                _SUPPRESSION_QUERY,
                role_filter=["user", "assistant"],
                exclude_sources=["cron"],
                limit=20,
            )
            for match in suppressed:
                ts = match.get("timestamp") or match.get("session_started")
                if not _recent(ts, cutoff):
                    continue
                snippet = str(match.get("snippet") or "").strip()
                context_text = " ".join(
                    _clip((ctx or {}).get("content"), 120)
                    for ctx in (match.get("context") or [])
                    if (ctx or {}).get("content") and not _looks_machine_text((ctx or {}).get("content"))
                ).strip()
                if _looks_machine_text(snippet) and not context_text:
                    continue
                excerpt = context_text if _looks_machine_text(snippet) else (snippet or context_text)
                if not excerpt:
                    continue
                report["suppressed_topics"].append(
                    {
                        "session_id": match.get("session_id"),
                        "source": match.get("source"),
                        "timestamp": _iso(ts),
                        "excerpt": _clip(excerpt, 220),
                    }
                )
                if len(report["suppressed_topics"]) >= 5:
                    break
        except Exception as exc:
            report["scan_errors"].append(f"suppression search failed: {exc}")
    except Exception as exc:
        report["scan_errors"].append(f"session scan unavailable: {exc}")

    try:
        for job in list_jobs(include_disabled=True):
            last_error = job.get("last_error")
            delivery_error = job.get("last_delivery_error")
            last_status = str(job.get("last_status") or "").lower()
            state = str(job.get("state") or "").lower()
            if last_error or delivery_error or last_status in {"error", "failed"} or state in {"error", "failed"}:
                add_signal(
                    {
                        "kind": "cron_failure",
                        "mode": "already_checked",
                        "reason": "scheduled job reports an error or failed delivery",
                        "job_id": job.get("id"),
                        "name": job.get("name"),
                        "state": job.get("state"),
                        "last_status": job.get("last_status"),
                        "excerpt": _clip(last_error or delivery_error or "cron job failed"),
                    }
                )
    except Exception as exc:
        report["scan_errors"].append(f"cron scan unavailable: {exc}")

    # Keep prompt injection compact and stable.
    report["recent_sessions"] = report["recent_sessions"][:10]
    report["signals"] = report["signals"][:12]
    report["suppressed_topics"] = report["suppressed_topics"][:5]
    report["wakeAgent"] = bool(report["signals"] or report["scan_errors"])
    return report


def render_signal_scan(report: Dict[str, Any], *, include_wake_gate: bool = True) -> str:
    """Render a scan report for cron script stdout.

    The scheduler treats a final JSON line with ``{"wakeAgent": false}`` as a
    gate to skip the LLM entirely, so keep that object as the last non-empty line.
    """

    wake = bool(report.get("wakeAgent", True))
    if include_wake_gate and not wake:
        return json.dumps({"wakeAgent": False}, sort_keys=True)
    body = "## Proactive signal scan\n" + json.dumps(report, indent=2, sort_keys=True)
    if include_wake_gate:
        body += "\n" + json.dumps({"wakeAgent": wake}, sort_keys=True)
    return body


def _ensure_scanner_script() -> str:
    scripts_dir = get_hermes_home() / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    path = scripts_dir / DEFAULT_SCANNER_SCRIPT
    module_root = Path(__file__).resolve().parents[1]
    path.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(module_root)!r})\n"
        "from hermes_cli.proactive import cmd_scan_script\n"
        "raise SystemExit(cmd_scan_script())\n",
        encoding="utf-8",
    )
    return DEFAULT_SCANNER_SCRIPT


def cmd_scan_script() -> int:
    """Entrypoint used by the cron pre-run scanner script."""

    report = apply_ledger_filters(collect_proactive_signals())
    print(render_signal_scan(report, include_wake_gate=True))
    return 0


def _find_existing_job() -> Dict[str, Any] | None:
    for job in list_jobs(include_disabled=True):
        if job.get("name") == DEFAULT_JOB_NAME:
            return job
    return None


def install_proactive_job(
    *,
    schedule: str = DEFAULT_SCHEDULE,
    deliver: str = DEFAULT_DELIVER,
    lookback_days: int = 7,
    max_sessions: int = 30,
    min_confidence: str = "high",
    paused: bool = False,
) -> Dict[str, Any]:
    """Create or update the built-in proactive synthesis cron job.

    Returns a small report with ``action`` (created/updated/created_paused) and
    the resulting job dict. The job is idempotent by name so repeated installs
    tune the same schedule/prompt instead of creating duplicates.
    """

    prompt = build_reflection_prompt(
        lookback_days=lookback_days,
        max_sessions=max_sessions,
        min_confidence=min_confidence,
    )
    script = _ensure_scanner_script()
    existing = _find_existing_job()
    updates = {
        "schedule": schedule,
        "prompt": prompt,
        "name": DEFAULT_JOB_NAME,
        "deliver": deliver,
        "script": script,
        "enabled_toolsets": list(DEFAULT_ENABLED_TOOLSETS),
    }

    if existing:
        job = update_job(existing["id"], updates)
        action = "updated"
    else:
        job = create_job(
            prompt=prompt,
            schedule=schedule,
            name=DEFAULT_JOB_NAME,
            deliver=deliver,
            script=script,
            enabled_toolsets=list(DEFAULT_ENABLED_TOOLSETS),
        )
        action = "created"

    if paused and job:
        job = update_job(
            job["id"],
            {
                "enabled": False,
                "state": "paused",
                "paused_reason": "created paused for review",
            },
        )
        action = "created_paused" if action == "created" else "updated_paused"

    return {"action": action, "job": job}


def _print_report(report: Dict[str, Any], *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    action = report.get("action", "ok")
    job = report.get("job") or {}
    print(f"Proactive synthesis job {action}.")
    if job:
        print(f"  ID: {job.get('id') or job.get('job_id')}")
        print(f"  Name: {job.get('name')}")
        print(f"  Schedule: {job.get('schedule_display') or job.get('schedule')}")
        print(f"  Deliver: {job.get('deliver')}")
        print(f"  State: {job.get('state')}")
        print(f"  Script: {job.get('script')}")
        print(f"  Toolsets: {', '.join(job.get('enabled_toolsets') or []) or 'default'}")


def cmd_proactive(args) -> int:
    """CLI entrypoint for ``hermes proactive``."""

    subcmd = getattr(args, "proactive_command", None) or "prompt"
    if subcmd == "prompt":
        prompt = build_reflection_prompt(
            lookback_days=getattr(args, "lookback_days", 7),
            max_sessions=getattr(args, "max_sessions", 30),
            min_confidence=getattr(args, "min_confidence", "high"),
        )
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "lookback_days": max(1, min(int(getattr(args, "lookback_days", 7)), 90)),
                        "max_sessions": max(1, min(int(getattr(args, "max_sessions", 30)), 200)),
                        "min_confidence": str(getattr(args, "min_confidence", "high") or "high").lower(),
                        "prompt": prompt,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(prompt)
        return 0

    if subcmd == "scan":
        report = apply_ledger_filters(
            collect_proactive_signals(
                lookback_days=getattr(args, "lookback_days", 7),
                max_sessions=getattr(args, "max_sessions", 30),
            )
        )
        if getattr(args, "json", False):
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render_signal_scan(report, include_wake_gate=False))
        return 0

    if subcmd == "install":
        report = install_proactive_job(
            schedule=getattr(args, "schedule", DEFAULT_SCHEDULE),
            deliver=getattr(args, "deliver", DEFAULT_DELIVER),
            lookback_days=getattr(args, "lookback_days", 7),
            max_sessions=getattr(args, "max_sessions", 30),
            min_confidence=getattr(args, "min_confidence", "high"),
            paused=getattr(args, "paused", False),
        )
        _print_report(report, as_json=getattr(args, "json", False))
        return 0

    print("Usage: hermes proactive [prompt|scan|install]")
    return 2
