"""Read-only Daily Ops Radar projection for Cockpit.

BIF-526 intentionally keeps this module side-effect free: it reads the existing
Hermes cron job metadata and the latest saved markdown output, sanitizes it for
UI display, and never invokes git, cron, services, network, or messaging.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hermes_cli.config import get_hermes_home
from hermes_constants import get_default_hermes_root

DAILY_OPS_RADAR_JOB_ID = "a82830911bcd"
DAILY_OPS_RADAR_SCRIPT = "hermes_daily_ops_radar.py"
MAX_RAW_EXCERPT_CHARS = 2400
MAX_CATEGORY_ITEMS = 8
MAX_TOP_COMMITS = 8
MAX_UPGRADE_BRIEF_GROUPS = 5
MAX_UPGRADE_BRIEF_EXAMPLES = 3
MAX_UPGRADE_BRIEF_CHARS = 1400
RADAR_STALE_AFTER_SECONDS = 36 * 60 * 60
HIGH_BEHIND_COUNT = 50
HIGH_RELEVANT_CHANGE_COUNT = 20
RISKY_CATEGORY_KEYWORDS = (
    "provider",
    "model",
    "gateway",
    "security",
    "breaking",
    "config",
    "approval",
    "approvals",
)

_SECRET_PATTERNS = [
    re.compile(r"\b(?:sk|pk|ghp|gho|github_pat|xox[abprs])-[-_A-Za-z0-9]{12,}\b", re.I),
    re.compile(r"\b(api[_-]?key|token|secret|password|authorization|state key)\b\s*[:=]\s*\S+", re.I),
    re.compile(r"\b(chat|user|thread|channel|message|session)[_-]?id\b\s*[:=]\s*[-_:.A-Za-z0-9]{6,}", re.I),
    re.compile(r"\b\d{15,22}\b"),
]
_LOCAL_PATH_RE = re.compile(r"(?<!\w)(?:~|/(?:Users|opt|tmp|var|private|Volumes|Applications))(?:(?:/|\s+)[^\s`'\")]+)+", re.I)
_GIT_C_PATH_RE = re.compile(r"\bgit\s+-C\s+(?P<path>(?:~|/[^\s`'\")]+))\s+", re.I)
_EMOJI_PREFIX_RE = re.compile(r"^[^\w/]+\s*")
_SHA_RE = re.compile(r"\b[0-9a-f]{7,12}\b", re.I)
_CONVENTIONAL_PREFIX_RE = re.compile(r"^(?:feat|fix|docs|chore|refactor|test|tests|ci|build|perf|style)(?:\([^)]*\))?:\s*", re.I)
_SUMMARY_RE = re.compile(
    r"Hermes daily ops radar:\s*(?P<relevant>\d+)\s+relevant upstream change\(s\)\s+in\s+(?P<behind>\d+)\s+commit\(s\)\s+behind\s+(?P<base>[^.]+)",
    re.I,
)


def sanitize_daily_ops_text(value: Any) -> str:
    """Return bounded display text with secrets, local paths, and large whitespace removed."""
    text = str(value or "")
    text = _GIT_C_PATH_RE.sub("git ", text)
    text = _LOCAL_PATH_RE.sub("[local path redacted]", text)
    for pattern in _SECRET_PATTERNS:
        if pattern.pattern.startswith("\\b(api"):
            text = pattern.sub(lambda m: f"{m.group(1)}: [redacted]", text)
        elif "state key" in pattern.pattern:
            text = pattern.sub("state key: [redacted]", text)
        else:
            text = pattern.sub("[redacted]", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _latest_output_file(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists() or not output_dir.is_dir():
        return None
    candidates = [path for path in output_dir.glob("*.md") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _candidate_homes() -> list[Path]:
    """Return read-only lookup roots for legacy Daily Ops Radar cron artifacts."""
    homes: list[Path] = []
    for home in (Path(get_hermes_home()), Path(get_default_hermes_root())):
        if home not in homes:
            homes.append(home)
    return homes


def _has_daily_ops_artifacts(home: Path) -> bool:
    if (home / "cron" / "output" / DAILY_OPS_RADAR_JOB_ID).exists():
        return True
    if (home / "cron" / "jobs.json").exists():
        return bool(_load_job(home))
    return False


def _select_home_for_daily_ops() -> Path:
    """Prefer the active profile; fall back to the default root for this legacy cron job only."""
    homes = _candidate_homes()
    for home in homes:
        if _has_daily_ops_artifacts(home):
            return home
    return homes[0]


def _load_job(home: Path) -> Dict[str, Any]:
    jobs_path = home / "cron" / "jobs.json"
    try:
        payload = json.loads(jobs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    for job in payload.get("jobs", []):
        if str(job.get("id") or job.get("job_id") or "") == DAILY_OPS_RADAR_JOB_ID:
            return dict(job)
    return {}


def _split_commit_refs(text: str) -> list[Dict[str, str]]:
    items: list[Dict[str, str]] = []
    for raw in re.split(r";\s*", text):
        cleaned = sanitize_daily_ops_text(raw)
        if not cleaned:
            continue
        match = _SHA_RE.search(cleaned)
        items.append(
            {
                "sha": match.group(0) if match else "",
                "title": cleaned[match.end() :].strip(" —:-") if match else cleaned,
                "text": cleaned,
            }
        )
        if len(items) >= MAX_CATEGORY_ITEMS:
            break
    return items


def _parse_category(line: str) -> Optional[Dict[str, Any]]:
    if ":" not in line or line.startswith(("**", "#", "•")):
        return None
    label_raw, body_raw = line.split(":", 1)
    label = _EMOJI_PREFIX_RE.sub("", label_raw).strip()
    if not label or label.lower() in {"repo", "compare", "state key"}:
        return None
    body = sanitize_daily_ops_text(body_raw)
    if not body:
        return None
    more_match = re.search(r"\(\+(\d+) more\)", body)
    return {
        "label": sanitize_daily_ops_text(label),
        "summary": body,
        "more_count": _safe_int(more_match.group(1)) if more_match else 0,
        "items": _split_commit_refs(body),
    }


def _parse_top_commit(line: str) -> Optional[Dict[str, str]]:
    cleaned = sanitize_daily_ops_text(line.lstrip("• ").strip())
    if not cleaned or cleaned.startswith("…"):
        return None
    match = _SHA_RE.search(cleaned)
    if not match:
        return {"sha": "", "title": cleaned, "files": "", "text": cleaned}
    rest = cleaned[match.end() :].strip()
    title, sep, files = rest.partition(" — ")
    return {
        "sha": match.group(0),
        "title": title.strip(" :-"),
        "files": files.strip() if sep else "",
        "text": cleaned,
    }


def _plain_change_title(value: Any) -> str:
    """Return a display-safe, human-readable change title without commit noise."""
    text = sanitize_daily_ops_text(value)
    text = _SHA_RE.sub("", text).strip(" —:-")
    text = _CONVENTIONAL_PREFIX_RE.sub("", text).strip()
    text = re.sub(r"\s+—\s+.*$", "", text).strip()
    text = re.sub(r"\s*\(\+\d+ more\)\s*$", "", text).strip()
    if len(text) > 180:
        text = text[:179].rstrip() + "…"
    return text


def _category_benefit_sentence(label: str, examples: list[str], more_count: int = 0) -> str:
    lower = label.lower()
    if "gateway" in lower or "chat" in lower or "slack" in lower:
        base = "More reliable messaging and gateway behavior"
    elif "provider" in lower or "model" in lower or "routing" in lower:
        base = "Better model/provider routing and safer upgrade compatibility"
    elif "security" in lower or "auth" in lower or "secret" in lower:
        base = "Security and credential-handling hardening"
    elif "ui" in lower or "cockpit" in lower or "dashboard" in lower or "tui" in lower:
        base = "Interface and command-surface improvements"
    elif "test" in lower or "ci" in lower or "build" in lower:
        base = "Test/build reliability improvements that lower regression risk"
    elif "docs" in lower or "skill" in lower:
        base = "Documentation and skill-library improvements"
    else:
        base = f"{label} improvements"

    if examples:
        sentence = f"{base}: " + "; ".join(examples[:MAX_UPGRADE_BRIEF_EXAMPLES])
    else:
        sentence = base
    if more_count:
        sentence += f" plus {more_count} more related change{'s' if more_count != 1 else ''}"
    return sentence.rstrip(" .") + "."


def build_upgrade_brief(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize major upstream improvements in plain language, without raw commit spam."""
    categories = [category for category in summary.get("categories") or [] if isinstance(category, dict)]
    top_commits = [commit for commit in summary.get("top_commits") or [] if isinstance(commit, dict)]
    groups: list[Dict[str, Any]] = []
    seen_labels: set[str] = set()

    for category in categories[:MAX_UPGRADE_BRIEF_GROUPS]:
        label = sanitize_daily_ops_text(category.get("label") or "General") or "General"
        if label.lower() in seen_labels:
            continue
        seen_labels.add(label.lower())
        examples: list[str] = []
        for item in category.get("items") or []:
            if isinstance(item, dict):
                title = _plain_change_title(item.get("title") or item.get("text") or "")
                if title and title not in examples:
                    examples.append(title)
            if len(examples) >= MAX_UPGRADE_BRIEF_EXAMPLES:
                break
        more_count = _safe_int(category.get("more_count")) or 0
        groups.append({"label": label, "summary": _category_benefit_sentence(label, examples, more_count), "examples": examples})

    if not groups and top_commits:
        examples = []
        for commit in top_commits[:MAX_UPGRADE_BRIEF_EXAMPLES]:
            title = _plain_change_title(commit.get("title") or commit.get("text") or "")
            if title and title not in examples:
                examples.append(title)
        if examples:
            groups.append({"label": "Notable upstream changes", "summary": _category_benefit_sentence("Notable upstream changes", examples), "examples": examples})

    behind = _safe_int(summary.get("behind_count"))
    relevant = _safe_int(summary.get("relevant_change_count"))
    if groups:
        headline = "Major improvements available upstream: " + "; ".join(group["summary"].rstrip(".") for group in groups[:3]) + "."
        why = "Why this matters: the radar shows user-visible reliability, routing, or safety improvements worth reviewing before any upgrade decision."
    elif behind is not None or relevant is not None:
        headline = "No major improvement themes were extractable from the latest radar output."
        why = "Why this matters: counts alone are not enough to justify an upgrade; review source detail before acting."
    else:
        headline = "No upgrade brief is available because the radar output is missing or incomplete."
        why = "Why this matters: without a trustworthy radar artifact, Biff should not recommend upgrade action."

    brief: Dict[str, Any] = {
        "question": "What major improvements would we get?",
        "headline": sanitize_daily_ops_text(headline),
        "groups": groups,
        "why_this_matters": sanitize_daily_ops_text(why),
    }
    if len(json.dumps(brief, ensure_ascii=False)) > MAX_UPGRADE_BRIEF_CHARS:
        brief["headline"] = str(brief["headline"])[:420].rstrip() + "…"
        brief["groups"] = groups[:3]
    return brief


def _safe_raw_excerpt(lines: Iterable[str]) -> str:
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("state key:"):
            continue
        kept.append(sanitize_daily_ops_text(stripped))
    excerpt = "\n".join(kept)
    if len(excerpt) > MAX_RAW_EXCERPT_CHARS:
        return excerpt[: MAX_RAW_EXCERPT_CHARS - 1].rstrip() + "…"
    return excerpt


def parse_daily_ops_radar_markdown(text: str, source_path: Optional[Path] = None) -> Dict[str, Any]:
    lines = text.splitlines()
    summary: Dict[str, Any] = {
        "last_run": None,
        "behind_count": None,
        "relevant_change_count": None,
        "behind_ref": None,
        "repo": None,
        "categories": [],
        "top_commits": [],
        "compare_command": None,
        "upgrade_brief": None,
    }
    in_top = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("**Run Time:**"):
            summary["last_run"] = sanitize_daily_ops_text(line.split(":**", 1)[-1].strip())
            continue
        radar_match = _SUMMARY_RE.search(line)
        if radar_match:
            summary["relevant_change_count"] = int(radar_match.group("relevant"))
            summary["behind_count"] = int(radar_match.group("behind"))
            summary["behind_ref"] = sanitize_daily_ops_text(radar_match.group("base"))
            continue
        if line.startswith("Repo:"):
            summary["repo"] = sanitize_daily_ops_text(line.split(":", 1)[1].strip())
            continue
        if line == "Top commits to review:":
            in_top = True
            continue
        if line.startswith("Compare:"):
            summary["compare_command"] = sanitize_daily_ops_text(line.split(":", 1)[1].strip())
            in_top = False
            continue
        if in_top and line.startswith("•"):
            commit = _parse_top_commit(line)
            if commit and len(summary["top_commits"]) < MAX_TOP_COMMITS:
                summary["top_commits"].append(commit)
            continue
        category = _parse_category(line)
        if category:
            summary["categories"].append(category)
    if source_path is not None:
        summary["source_file"] = source_path.name
        try:
            summary["source_mtime"] = source_path.stat().st_mtime
        except OSError:
            summary["source_mtime"] = None
    summary["upgrade_brief"] = build_upgrade_brief(summary)
    return summary


def _category_labels(summary: Dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for category in summary.get("categories") or []:
        if isinstance(category, dict):
            label = sanitize_daily_ops_text(category.get("label") or "")
            if label:
                labels.append(label)
    return labels


def build_upgrade_recommendation(
    summary: Dict[str, Any],
    job: Dict[str, Any],
    source: str,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Return Biff's read-only first-slice answer to: Should we upgrade?"""
    now_ts = time.time() if now is None else now
    behind = _safe_int(summary.get("behind_count"))
    relevant = _safe_int(summary.get("relevant_change_count"))
    source_mtime = summary.get("source_mtime")
    age_seconds: Optional[int] = None
    if isinstance(source_mtime, (int, float)):
        age_seconds = max(0, int(now_ts - float(source_mtime)))

    job_status = sanitize_daily_ops_text(job.get("status") or "")
    job_state = sanitize_daily_ops_text(job.get("state") or "")
    last_basis = sanitize_daily_ops_text(
        summary.get("last_run") or job.get("last_run_at") or summary.get("source_file") or "not available"
    )
    freshness_label = "fresh"
    freshness_detail = "Latest radar artifact is within the freshness window."

    untrusted_reasons: list[str] = []
    if source != "cron_output_latest_markdown":
        untrusted_reasons.append("radar output is missing")
    if behind is None or relevant is None:
        untrusted_reasons.append("radar counts are unavailable")
    if job_status and job_status.lower() not in {"ok", "success", "succeeded", "completed"}:
        untrusted_reasons.append(f"last radar status is {job_status}")
    if job_state and job_state.lower() in {"error", "failed", "disabled"}:
        untrusted_reasons.append(f"radar job state is {job_state}")
    if age_seconds is None:
        freshness_label = "unknown"
        freshness_detail = "Radar artifact mtime is unavailable."
        untrusted_reasons.append("radar freshness is unknown")
    elif age_seconds > RADAR_STALE_AFTER_SECONDS:
        freshness_label = "stale"
        freshness_detail = "Latest radar artifact is older than 36 hours."
        untrusted_reasons.append("radar output is stale")

    if untrusted_reasons:
        return {
            "question": "Should we upgrade?",
            "label": "Wait",
            "risk_level": "unknown",
            "rationale": "Wait; radar is not trustworthy enough for an upgrade call.",
            "basis": last_basis,
            "freshness": freshness_label,
            "freshness_age_seconds": age_seconds,
            "freshness_detail": freshness_detail,
            "signals": untrusted_reasons[:4],
            "behind_count": behind,
            "relevant_change_count": relevant,
        }

    labels = _category_labels(summary)
    risky_labels = [
        label for label in labels if any(keyword in label.lower() for keyword in RISKY_CATEGORY_KEYWORDS)
    ]
    high_volume = (behind or 0) >= HIGH_BEHIND_COUNT or (relevant or 0) >= HIGH_RELEVANT_CHANGE_COUNT
    if high_volume or risky_labels:
        signals: list[str] = []
        if high_volume:
            signals.append(f"{behind} behind / {relevant} relevant")
        if risky_labels:
            signals.append("risky categories: " + ", ".join(risky_labels[:3]))
        return {
            "question": "Should we upgrade?",
            "label": "Prepare review first",
            "risk_level": "high" if high_volume else "medium",
            "rationale": "Do not upgrade blindly; prepare a read-only review first.",
            "basis": last_basis,
            "freshness": freshness_label,
            "freshness_age_seconds": age_seconds,
            "freshness_detail": freshness_detail,
            "signals": signals,
            "behind_count": behind,
            "relevant_change_count": relevant,
        }

    return {
        "question": "Should we upgrade?",
        "label": "Upgrade now",
        "risk_level": "low",
        "rationale": "Low relevant change count and no risky categories; safe to prepare upgrade review.",
        "basis": last_basis,
        "freshness": freshness_label,
        "freshness_age_seconds": age_seconds,
        "freshness_detail": freshness_detail,
        "signals": [f"{behind} behind / {relevant} relevant"],
        "behind_count": behind,
        "relevant_change_count": relevant,
    }


def get_daily_ops_radar_payload() -> Dict[str, Any]:
    """Build the authenticated Cockpit payload from existing cron artifacts only."""
    home = _select_home_for_daily_ops()
    job = _load_job(home)
    output_dir = home / "cron" / "output" / DAILY_OPS_RADAR_JOB_ID
    latest_path = _latest_output_file(output_dir)
    raw_text = ""
    if latest_path is not None:
        try:
            raw_text = latest_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            raw_text = ""
    summary = parse_daily_ops_radar_markdown(raw_text, latest_path) if raw_text else {
        "last_run": None,
        "behind_count": None,
        "relevant_change_count": None,
        "behind_ref": None,
        "repo": None,
        "categories": [],
        "top_commits": [],
        "compare_command": None,
        "upgrade_brief": None,
    }
    if not summary.get("upgrade_brief"):
        summary["upgrade_brief"] = build_upgrade_brief(summary)
    job_last_run = sanitize_daily_ops_text(job.get("last_run_at") or "") or None
    if not summary.get("last_run") and job_last_run:
        summary["last_run"] = job_last_run
    job_payload = {
        "id": DAILY_OPS_RADAR_JOB_ID,
        "name": sanitize_daily_ops_text(job.get("name") or "Hermes Agent Daily Ops Radar"),
        "script": sanitize_daily_ops_text(job.get("script") or DAILY_OPS_RADAR_SCRIPT),
        "enabled": bool(job.get("enabled", False)) if job else None,
        "state": sanitize_daily_ops_text(job.get("state") or "") or None,
        "status": sanitize_daily_ops_text(job.get("last_status") or "") or None,
        "schedule": sanitize_daily_ops_text(job.get("schedule_display") or "") or None,
        "last_run_at": job_last_run,
        "next_run_at": sanitize_daily_ops_text(job.get("next_run_at") or "") or None,
    }
    source = "cron_output_latest_markdown" if raw_text else "cron_output_missing"
    recommendation = build_upgrade_recommendation(summary, job_payload, source)
    upgrade_brief = summary.get("upgrade_brief") or build_upgrade_brief(summary)
    return {
        "schema_version": 1,
        "read_only": True,
        "actions_enabled": False,
        "external_delivery_enabled": False,
        "generated_at": time.time(),
        "source": source,
        "job": job_payload,
        "summary": summary,
        "raw_excerpt": _safe_raw_excerpt(raw_text.splitlines()),
        "upgrade": {
            "brief": upgrade_brief,
            "recommendation": recommendation,
            "prepare_review": {
                "label": "Prepare upgrade review",
                "enabled": False,
                "mutates": False,
                "method": "GET",
                "endpoint": None,
                "status": "disabled_no_safe_preflight_endpoint",
                "description": "CTA is display-only until a safe read-only preflight endpoint exists; no git pull, merge, restart, production mutation, or external send is wired here.",
            }
        },
    }
