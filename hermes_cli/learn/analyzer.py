"""Usage analyzer for Learn-created automation suggestions."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hermes_constants import get_hermes_home
from utils import atomic_replace

from . import sampler

AddSuggestion = Callable[..., Optional[Dict[str, Any]]]

_cron_suggestions_lock = threading.Lock()


def _learn_dir(home: Path) -> Path:
    return home.resolve() / "learn"


def _opportunities_path(home: Path) -> Path:
    return _learn_dir(home) / "opportunities.json"


def _secure_file(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _save_opportunities(home: Path, opportunities: List[Dict[str, Any]]) -> None:
    learn_dir = _learn_dir(home)
    learn_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(learn_dir), suffix=".tmp", prefix=".learn_opportunities_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"opportunities": opportunities}, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, _opportunities_path(home))
        _secure_file(_opportunities_path(home))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _event_duration(event: Dict[str, Any]) -> int:
    try:
        return max(0, int(event.get("duration_seconds") or 0))
    except (TypeError, ValueError):
        return 0


def _job_template(category: str, event_count: int, duration_seconds: int) -> Dict[str, Any]:
    minutes = max(1, round(duration_seconds / 60))
    if category == "communication":
        title = "Daily communication follow-up summary"
        schedule = "0 16 * * 1-5"
        prompt = (
            "Prepare a concise end-of-workday communication follow-up summary. "
            "Review connected communication sources, identify open loops and likely follow-ups, "
            "and produce a draft checklist. Do not send messages or mutate external systems."
        )
    elif category == "development":
        title = "End-of-day development summary"
        schedule = "0 17 * * 1-5"
        prompt = (
            "Prepare a concise end-of-day development summary from recent Hermes context and available repo state. "
            "List changed areas, unfinished work, failing checks to revisit, and suggested next actions."
        )
    elif category == "browser":
        title = "Weekly research packet"
        schedule = "0 16 * * 5"
        prompt = (
            "Prepare a weekly research packet from recent approved Hermes context and connected sources. "
            "Summarize recurring research themes, links to revisit, and follow-up questions."
        )
    elif category == "documents":
        title = "Weekly document and checklist review"
        schedule = "0 15 * * 5"
        prompt = (
            "Prepare a weekly document and checklist review. Summarize documents or notes the user repeatedly worked on, "
            "then draft a checklist of open items. Do not edit files automatically."
        )
    else:
        title = "Daily workflow opportunity summary"
        schedule = "0 16 * * 1-5"
        prompt = (
            "Prepare a daily workflow opportunity summary from approved Hermes context. "
            "Identify repeated work patterns and draft low-risk next-step suggestions for user review."
        )

    return {
        "title": title,
        "description": (
            f"Learn observed {event_count} metadata events in the {category} category "
            f"covering about {minutes} minute(s). This proposal is read/prepare only and requires approval."
        ),
        "job_spec": {
            "prompt": prompt,
            "schedule": schedule,
            "name": title,
            "deliver": "origin",
        },
    }


def analyze_usage(
    *,
    home: Path | None = None,
    min_events: int = 3,
    min_duration_seconds: int = 600,
) -> List[Dict[str, Any]]:
    """Aggregate repeated metadata patterns into conservative opportunities."""
    resolved_home = (home or get_hermes_home()).resolve()
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"event_count": 0, "total_duration_seconds": 0, "process_names": set(), "domains": set()}
    )
    for event in sampler._load_events(resolved_home):
        if event.get("idle") is True:
            continue
        category = str(event.get("category") or "other").strip().lower() or "other"
        group = groups[category]
        group["event_count"] += 1
        group["total_duration_seconds"] += _event_duration(event)
        if event.get("process_name"):
            group["process_names"].add(str(event["process_name"]))
        if event.get("domain"):
            group["domains"].add(str(event["domain"]))

    opportunities: List[Dict[str, Any]] = []
    for category, group in sorted(groups.items()):
        if group["event_count"] < min_events or group["total_duration_seconds"] < min_duration_seconds:
            continue
        template = _job_template(category, group["event_count"], group["total_duration_seconds"])
        opportunities.append(
            {
                "category": category,
                "dedup_key": f"learn:usage:{category}",
                "event_count": group["event_count"],
                "total_duration_seconds": group["total_duration_seconds"],
                "process_names": sorted(group["process_names"]),
                "domains": sorted(group["domains"]),
                **template,
            }
        )
    return opportunities


def _add_suggestion_for_home(home: Path, **kwargs) -> Optional[Dict[str, Any]]:
    from cron import suggestions as cron_suggestions

    with _cron_suggestions_lock:
        old_cron_dir = cron_suggestions.CRON_DIR
        old_suggestions_file = cron_suggestions.SUGGESTIONS_FILE
        cron_suggestions.CRON_DIR = home / "cron"
        cron_suggestions.SUGGESTIONS_FILE = cron_suggestions.CRON_DIR / "suggestions.json"
        try:
            return cron_suggestions.add_suggestion(**kwargs)
        finally:
            cron_suggestions.CRON_DIR = old_cron_dir
            cron_suggestions.SUGGESTIONS_FILE = old_suggestions_file


def create_usage_suggestions(
    *,
    add_fn: AddSuggestion | None = None,
    home: Path | None = None,
) -> List[Dict[str, Any]]:
    """Create pending cron suggestions from repeated Learn usage patterns."""
    resolved_home = (home or get_hermes_home()).resolve()
    opportunities = analyze_usage(home=resolved_home)
    _save_opportunities(resolved_home, opportunities)
    created: List[Dict[str, Any]] = []
    for opportunity in opportunities:
        payload = {
            "title": opportunity["title"],
            "description": opportunity["description"],
            "source": "usage",
            "job_spec": dict(opportunity["job_spec"]),
            "dedup_key": opportunity["dedup_key"],
        }
        record = add_fn(**payload) if add_fn is not None else _add_suggestion_for_home(resolved_home, **payload)
        if record is not None:
            created.append(record)
    return created
