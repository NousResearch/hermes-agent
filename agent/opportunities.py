"""Consent-first proactive opportunities.

An opportunity is a pending proposal for Hermes to become more useful without
silently changing behavior. The store is profile-scoped, capped, deduplicated,
and every accepted record turns into an ordinary user turn or an existing
command path. No model tool is added and nothing is scheduled, learned, or
created until the user accepts.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now
from utils import atomic_replace

logger = logging.getLogger(__name__)

OPPORTUNITIES_DIR = get_hermes_home().resolve() / "opportunities"
OPPORTUNITIES_FILE = OPPORTUNITIES_DIR / "opportunities.json"

VALID_SOURCES = frozenset({"starter", "usage", "manual", "integration"})
VALID_ACTION_TYPES = frozenset(
    {
        "learn_skill",
        "skill_bundle",
        "profile",
        "kanban_swarm",
        "cron_automation",
    }
)
_STATUS_PENDING = "pending"
_STATUS_ACCEPTED = "accepted"
_STATUS_DISMISSED = "dismissed"

DEFAULT_MAX_PENDING = 8
DEFAULT_SCAN_INTERVAL_HOURS = 24
DEFAULT_SCAN_RECENT_MESSAGES = 120
DEFAULT_MIN_REPEATS = 3

_opportunities_lock = threading.Lock()

_STOPWORDS = {
    "a",
    "about",
    "again",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "get",
    "give",
    "had",
    "has",
    "have",
    "help",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "just",
    "make",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "please",
    "should",
    "that",
    "the",
    "this",
    "to",
    "up",
    "us",
    "want",
    "we",
    "what",
    "with",
    "you",
    "your",
}

_ACTION_SYNONYMS = {
    "add": "build",
    "build": "build",
    "create": "build",
    "implement": "build",
    "make": "build",
    "ship": "build",
    "debug": "fix",
    "fix": "fix",
    "repair": "fix",
    "resolve": "fix",
    "audit": "review",
    "inspect": "review",
    "review": "review",
    "scan": "review",
    "analyze": "analyze",
    "analyse": "analyze",
    "compare": "analyze",
    "summarize": "summarize",
    "summarise": "summarize",
    "digest": "summarize",
    "learn": "learn",
    "remember": "learn",
    "capture": "learn",
    "schedule": "schedule",
    "monitor": "schedule",
}

_DOMAIN_KEYWORDS: dict[str, dict[str, Any]] = {
    "product-manager": {
        "action_type": "skill_bundle",
        "title": "Create a product manager vertical",
        "terms": {
            "prd",
            "backlog",
            "roadmap",
            "launch",
            "spec",
            "story",
            "stories",
            "requirement",
            "requirements",
            "product",
            "pm",
        },
        "goal": (
            "Create or update a product-manager vertical pack that captures my "
            "recurring PRD, backlog, launch, and decision workflows."
        ),
    },
    "analyst": {
        "action_type": "skill_bundle",
        "title": "Create an analyst vertical",
        "terms": {
            "analysis",
            "analyst",
            "cohort",
            "dashboard",
            "experiment",
            "forecast",
            "kpi",
            "metric",
            "metrics",
            "sql",
        },
        "goal": (
            "Create or update an analyst vertical pack that captures my "
            "recurring metric, dashboard, experiment, and evidence workflows."
        ),
    },
    "agent-ecosystem": {
        "action_type": "kanban_swarm",
        "title": "Create a specialized-agent ecosystem",
        "terms": {
            "agent",
            "agents",
            "delegate",
            "kanban",
            "orchestrator",
            "profile",
            "profiles",
            "subagent",
            "subagents",
            "swarm",
            "worker",
            "workers",
        },
        "goal": (
            "Design a reusable specialist-agent ecosystem for this recurring "
            "work, using profiles plus Kanban or swarm topology where useful."
        ),
    },
    "automation": {
        "action_type": "cron_automation",
        "title": "Turn recurring work into an automation",
        "terms": {
            "cron",
            "daily",
            "digest",
            "monitor",
            "recurring",
            "remind",
            "schedule",
            "scheduled",
            "weekly",
        },
        "goal": (
            "Turn this recurring work into a consent-first automation using "
            "existing cron blueprints or the cronjob tool."
        ),
    },
}


def _secure_file(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _ensure_dir() -> None:
    OPPORTUNITIES_DIR.mkdir(parents=True, exist_ok=True)


def _load_raw() -> Dict[str, Any]:
    if not OPPORTUNITIES_FILE.exists():
        return {"opportunities": [], "meta": {}}
    try:
        with open(OPPORTUNITIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("opportunities.json unreadable (%s); starting empty", exc)
        return {"opportunities": [], "meta": {}}
    if isinstance(data, dict):
        opportunities = data.get("opportunities")
        meta = data.get("meta")
        return {
            "opportunities": opportunities if isinstance(opportunities, list) else [],
            "meta": meta if isinstance(meta, dict) else {},
        }
    if isinstance(data, list):
        return {"opportunities": data, "meta": {}}
    logger.warning("opportunities.json malformed; starting empty")
    return {"opportunities": [], "meta": {}}


def _save_raw(opportunities: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> None:
    _ensure_dir()
    fd, tmp_path = tempfile.mkstemp(
        dir=str(OPPORTUNITIES_FILE.parent),
        suffix=".tmp",
        prefix=".opp_",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "opportunities": opportunities,
                    "meta": meta or {},
                    "updated_at": _hermes_now().isoformat(),
                },
                f,
                indent=2,
            )
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, OPPORTUNITIES_FILE)
        _secure_file(OPPORTUNITIES_FILE)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def proactive_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config if isinstance(config, dict) else _load_config()
    raw = cfg.get("proactive") if isinstance(cfg, dict) else {}
    return raw if isinstance(raw, dict) else {}


def is_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    return bool(proactive_config(config).get("enabled", False))


def notifications_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    return bool(proactive_config(config).get("notify", True))


def pending_cap(config: Optional[Dict[str, Any]] = None) -> int:
    raw = proactive_config(config).get("max_pending", DEFAULT_MAX_PENDING)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_MAX_PENDING
    return max(1, min(value, 50))


def scan_interval_hours(config: Optional[Dict[str, Any]] = None) -> float:
    raw = proactive_config(config).get("scan_interval_hours", DEFAULT_SCAN_INTERVAL_HOURS)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = DEFAULT_SCAN_INTERVAL_HOURS
    return max(0.0, value)


def scan_recent_messages(config: Optional[Dict[str, Any]] = None) -> int:
    raw = proactive_config(config).get("scan_recent_messages", DEFAULT_SCAN_RECENT_MESSAGES)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_SCAN_RECENT_MESSAGES
    return max(20, min(value, 500))


def min_repeats(config: Optional[Dict[str, Any]] = None) -> int:
    raw = proactive_config(config).get("min_repeats", DEFAULT_MIN_REPEATS)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_MIN_REPEATS
    return max(2, min(value, 20))


def load_opportunities() -> List[Dict[str, Any]]:
    return list(_load_raw().get("opportunities", []))


def load_meta() -> Dict[str, Any]:
    return dict(_load_raw().get("meta", {}))


def list_pending() -> List[Dict[str, Any]]:
    return [o for o in load_opportunities() if o.get("status") == _STATUS_PENDING]


def _coerce_action(action: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(action, dict):
        raise ValueError("action must be a dict")
    action_type = str(action.get("type") or "").strip()
    if action_type not in VALID_ACTION_TYPES:
        raise ValueError(f"unknown opportunity action type: {action_type!r}")
    return {**action, "type": action_type}


def add_opportunity(
    *,
    title: str,
    description: str,
    source: str,
    action: Dict[str, Any],
    dedup_key: str,
    evidence: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Register a pending opportunity, or return None when dedup/cap skips it."""
    if source not in VALID_SOURCES:
        raise ValueError(f"unknown opportunity source: {source!r}")
    if not title.strip() or not dedup_key.strip():
        raise ValueError("title and dedup_key are required")
    action = _coerce_action(action)
    clean_evidence = [
        str(item).strip()
        for item in (evidence or [])
        if str(item).strip()
    ][:5]

    with _opportunities_lock:
        raw = _load_raw()
        opportunities = raw.get("opportunities", [])
        meta = raw.get("meta", {})

        for existing in opportunities:
            if existing.get("dedup_key") == dedup_key:
                return None

        pending_count = sum(1 for o in opportunities if o.get("status") == _STATUS_PENDING)
        cap = pending_cap(config)
        if pending_count >= cap:
            logger.info("Opportunity backlog full (%d); dropping %r", cap, title)
            return None

        record = {
            "id": uuid.uuid4().hex[:12],
            "title": title.strip(),
            "description": description.strip(),
            "source": source,
            "action": action,
            "dedup_key": dedup_key.strip(),
            "evidence": clean_evidence,
            "status": _STATUS_PENDING,
            "created_at": _hermes_now().isoformat(),
        }
        opportunities.append(record)
        _save_raw(opportunities, meta)
        return record


def get_opportunity(ref: str) -> Optional[Dict[str, Any]]:
    opportunities = load_opportunities()
    for record in opportunities:
        if record.get("id") == ref:
            return record
    if str(ref).isdigit():
        pending = [o for o in opportunities if o.get("status") == _STATUS_PENDING]
        idx = int(str(ref)) - 1
        if 0 <= idx < len(pending):
            return pending[idx]
    lowered = str(ref).lower()
    for record in opportunities:
        if str(record.get("title", "")).lower() == lowered:
            return record
    return None


def _set_status(opportunity_id: str, status: str) -> bool:
    with _opportunities_lock:
        raw = _load_raw()
        opportunities = raw.get("opportunities", [])
        meta = raw.get("meta", {})
        changed = False
        for record in opportunities:
            if record.get("id") == opportunity_id:
                record["status"] = status
                record["resolved_at"] = _hermes_now().isoformat()
                changed = True
                break
        if changed:
            _save_raw(opportunities, meta)
        return changed


def dismiss_opportunity(ref: str) -> bool:
    record = get_opportunity(ref)
    if not record:
        return False
    return _set_status(record["id"], _STATUS_DISMISSED)


def clear_accepted() -> int:
    with _opportunities_lock:
        raw = _load_raw()
        opportunities = raw.get("opportunities", [])
        meta = raw.get("meta", {})
        kept = [o for o in opportunities if o.get("status") != _STATUS_ACCEPTED]
        removed = len(opportunities) - len(kept)
        if removed:
            _save_raw(kept, meta)
        return removed


def _evidence_block(record: Dict[str, Any]) -> str:
    evidence = [str(e).strip() for e in record.get("evidence") or [] if str(e).strip()]
    if not evidence:
        return ""
    lines = ["", "Evidence:"]
    for item in evidence[:5]:
        lines.append(f"- {item}")
    return "\n".join(lines)


def _acceptance_prompt(record: Dict[str, Any]) -> str:
    action = record.get("action") if isinstance(record.get("action"), dict) else {}
    action_type = action.get("type")
    goal = str(action.get("goal") or record.get("description") or record.get("title") or "").strip()
    evidence = _evidence_block(record)

    if action_type == "learn_skill":
        request = str(action.get("request") or goal or "the recurring workflow behind this opportunity").strip()
        if evidence:
            request = f"{request}\n\n{evidence}"
        from agent.learn_prompt import build_learn_prompt

        return build_learn_prompt(request)

    if action_type == "skill_bundle":
        bundle = str(action.get("bundle") or "").strip()
        label = f" named /{bundle}" if bundle else ""
        return (
            f"Create or update a Hermes skill bundle{label} for this recurring role or vertical:\n\n"
            f"{goal}\n"
            f"{evidence}\n\n"
            "Use existing skills where possible. If a needed skill is missing, create or update a "
            "class-level skill first. Keep the bundle as an edge capability: ordinary skill files "
            "and skill-bundles YAML, no system prompt mutation and no new core tools. Ask me before "
            "overwriting any existing bundle with unrelated content."
        ).strip()

    if action_type == "profile":
        profile = str(action.get("profile") or "").strip()
        label = f" named {profile!r}" if profile else ""
        return (
            f"Design and, if the needed details are clear, create a specialist Hermes profile{label}:\n\n"
            f"{goal}\n"
            f"{evidence}\n\n"
            "Use profile-native surfaces: SOUL.md for durable role identity, profile-local skills or "
            "skill bundles for procedures, config.yaml for behavior settings, and .env only for "
            "secrets. Ask me for any missing model, workspace, or credential decisions before making "
            "changes."
        ).strip()

    if action_type == "kanban_swarm":
        return (
            "Design a reusable multi-agent ecosystem for this recurring work:\n\n"
            f"{goal}\n"
            f"{evidence}\n\n"
            "Prefer profiles plus Kanban tasks or the existing swarm helper when persistent "
            "coordination, handoffs, verification, or specialist workers are useful. Do not create "
            "busywork agents. Ask me for missing role names, workspace, and dispatch details before "
            "starting any durable board work."
        ).strip()

    if action_type == "cron_automation":
        return (
            "Set up a recurring automation opportunity:\n\n"
            f"{goal}\n"
            f"{evidence}\n\n"
            "Use existing automation blueprints or the cronjob tool. Ask me for missing schedule, "
            "delivery, and criteria details before creating the job. If the job needs a reusable "
            "procedure, create or attach an appropriate skill instead of baking a long prompt into "
            "the schedule."
        ).strip()

    raise ValueError(f"unsupported opportunity action type: {action_type!r}")


def accept_opportunity(ref: str) -> Optional[Dict[str, Any]]:
    """Accept an opportunity and return the action the surface should take."""
    record = get_opportunity(ref)
    if not record or record.get("status") != _STATUS_PENDING:
        return None
    message = _acceptance_prompt(record)
    _set_status(record["id"], _STATUS_ACCEPTED)
    return {
        "kind": "send",
        "message": message,
        "notice": f"Accepted opportunity: {record.get('title', 'opportunity')}",
        "opportunity": record,
    }


def starter_opportunities() -> List[Dict[str, Any]]:
    return [
        {
            "title": "Learn a repeated workflow as a skill",
            "description": (
                "Review recent work for repeated workflows, corrections, and tool chains, then "
                "capture the durable procedure as a reusable skill."
            ),
            "source": "starter",
            "dedup_key": "starter:learn-recurring-workflow",
            "action": {
                "type": "learn_skill",
                "request": (
                    "Review my recent Hermes conversations for repeated workflows, corrections, "
                    "preferred formats, or tool chains. Create or update one reusable class-level "
                    "skill that would make the next similar task faster and more tailored."
                ),
            },
        },
        {
            "title": "Create a vertical role bundle",
            "description": (
                "Turn related skills into a slash-invoked role pack such as /analyst, "
                "/product-manager, /researcher, or another recurring role."
            ),
            "source": "starter",
            "dedup_key": "starter:create-vertical-bundle",
            "action": {
                "type": "skill_bundle",
                "goal": (
                    "Inspect my recurring work and skill library, then create a focused vertical "
                    "skill bundle for the strongest repeated role. Include role guidance, workflow "
                    "presets, useful connector hints, subagent brief patterns, and templates where "
                    "they add real leverage."
                ),
            },
        },
        {
            "title": "Create a specialist profile",
            "description": (
                "Package a recurring role into its own profile with identity, skills, config, "
                "workspace, and optional model choices."
            ),
            "source": "starter",
            "dedup_key": "starter:create-specialist-profile",
            "action": {
                "type": "profile",
                "goal": (
                    "Create a specialist Hermes profile for a recurring role I use often. Use a "
                    "profile-local SOUL.md, selected skills or bundles, config.yaml behavior, and "
                    "a concise profile description for routing."
                ),
            },
        },
        {
            "title": "Design a specialized-agent ecosystem",
            "description": (
                "Convert a recurring multi-step workflow into profiles plus Kanban or swarm "
                "coordination with verification and synthesis."
            ),
            "source": "starter",
            "dedup_key": "starter:create-agent-ecosystem",
            "action": {
                "type": "kanban_swarm",
                "goal": (
                    "Design a reusable specialist-agent ecosystem for a recurring workflow: "
                    "orchestrator, parallel workers, verifier, and synthesizer only where each role "
                    "adds clear value."
                ),
            },
        },
        {
            "title": "Turn recurring work into an automation",
            "description": (
                "Identify a repeated check-in, digest, monitor, or review that should become a "
                "scheduled job with explicit user consent."
            ),
            "source": "starter",
            "dedup_key": "starter:create-recurring-automation",
            "action": {
                "type": "cron_automation",
                "goal": (
                    "Find one recurring task that should become a scheduled Hermes automation. Use "
                    "an existing blueprint when possible, otherwise create a cron job only after "
                    "confirming schedule, delivery, and criteria."
                ),
            },
        },
    ]


def seed_starter_opportunities(
    *,
    add_fn=add_opportunity,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    created = []
    for item in starter_opportunities():
        rec = add_fn(
            title=item["title"],
            description=item["description"],
            source=item["source"],
            action=item["action"],
            dedup_key=item["dedup_key"],
            config=config,
        )
        if rec is not None:
            created.append(rec)
    return created


def _flatten_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        ]
        return " ".join(p for p in parts if p)
    return "" if content is None else str(content)


def _tokenize(text: str) -> List[str]:
    text = re.sub(r"https?://\S+", " ", text.lower())
    return re.findall(r"[a-z][a-z0-9_-]{2,}", text)


def _message_preview(text: str, limit: int = 120) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _intent_signature(text: str) -> Optional[str]:
    if text.lstrip().startswith("/"):
        return None
    tokens = [t for t in _tokenize(text) if t not in _STOPWORDS]
    if len(tokens) < 2:
        return None
    action = None
    domains: list[str] = []
    for tok in tokens:
        mapped = _ACTION_SYNONYMS.get(tok)
        if mapped and action is None:
            action = mapped
            continue
        if tok not in _ACTION_SYNONYMS and tok not in domains:
            domains.append(tok)
        if action and len(domains) >= 2:
            break
    if action is None:
        action = _ACTION_SYNONYMS.get(tokens[0], tokens[0])
        domains = [t for t in tokens[1:] if t not in _ACTION_SYNONYMS][:2]
    if not action or not domains:
        return None
    # Repeated workflows usually vary the second object ("review auth diff",
    # "review auth tests", "review auth blockers"). Group by action plus the
    # primary domain so the scanner catches the recurring shape without waiting
    # for near-identical phrasing.
    return ":".join([action, domains[0]])


def _query_recent_user_messages(limit: int, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    from hermes_state import SessionDB

    resolved_db_path = Path(db_path) if db_path is not None else get_hermes_home() / "state.db"
    if not resolved_db_path.exists():
        return []

    db = SessionDB(db_path=resolved_db_path, read_only=True)
    try:
        with db._lock:
            rows = db._conn.execute(
                """
                SELECT m.id, m.session_id, m.content, m.timestamp, s.source
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                WHERE m.role = 'user'
                  AND m.content IS NOT NULL
                  AND (m.active = 1 OR m.compacted = 1)
                  AND COALESCE(s.source, '') NOT IN ('cron', 'tool')
                ORDER BY m.timestamp DESC, m.id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        messages: list[dict[str, Any]] = []
        for row in rows:
            decoded = db._decode_content(row["content"])
            text = _flatten_text(decoded).strip()
            if not text:
                continue
            messages.append(
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "text": text,
                    "timestamp": row["timestamp"],
                    "source": row["source"],
                    "preview": _message_preview(text),
                }
            )
        return messages
    finally:
        db.close()


def _last_scan_ts(meta: Dict[str, Any]) -> Optional[float]:
    raw = meta.get("last_usage_scan_ts")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _mark_usage_scan(raw: Dict[str, Any]) -> None:
    meta = raw.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        raw["meta"] = meta
    stamp = _hermes_now()
    meta["last_usage_scan_at"] = stamp.isoformat()
    meta["last_usage_scan_ts"] = stamp.timestamp()


def _should_skip_for_cooldown(config: Dict[str, Any], meta: Dict[str, Any]) -> bool:
    interval = scan_interval_hours(config)
    if interval <= 0:
        return False
    last_ts = _last_scan_ts(meta)
    if last_ts is None:
        return False
    return (_hermes_now().timestamp() - last_ts) < interval * 3600


def _source_count(messages: Iterable[Dict[str, Any]]) -> int:
    return len({m.get("session_id") for m in messages if m.get("session_id")})


def scan_recent_usage(
    *,
    force: bool = False,
    config: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Scan recent user messages and register recurring-work opportunities."""
    cfg = config if isinstance(config, dict) else _load_config()
    if not force and not is_enabled(cfg):
        return {"scanned": False, "reason": "disabled", "created": []}

    with _opportunities_lock:
        raw = _load_raw()
        meta = raw.get("meta", {})
        if not force and _should_skip_for_cooldown(cfg, meta):
            return {"scanned": False, "reason": "cooldown", "created": []}
        _mark_usage_scan(raw)
        _save_raw(raw.get("opportunities", []), raw.get("meta", {}))

    messages = _query_recent_user_messages(scan_recent_messages(cfg), db_path=db_path)
    if not messages:
        return {"scanned": True, "reason": "no_messages", "created": []}

    threshold = min_repeats(cfg)
    created: list[dict[str, Any]] = []

    by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    domain_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for msg in messages:
        sig = _intent_signature(msg["text"])
        if sig:
            by_signature[sig].append(msg)
        tokens = set(_tokenize(msg["text"]))
        for domain, spec in _DOMAIN_KEYWORDS.items():
            if tokens & spec["terms"]:
                domain_hits[domain].append(msg)

    for sig, group in sorted(by_signature.items(), key=lambda item: len(item[1]), reverse=True):
        if len(group) < threshold:
            continue
        if _source_count(group) < 2 and len(group) < threshold + 1:
            continue
        label = sig.replace(":", " ")
        evidence = [m["preview"] for m in group[:3]]
        request = (
            f"Learn the recurring '{label}' workflow from my recent conversations. "
            "Use the evidence below as starting points, search session history if useful, "
            "and create or update one reusable class-level skill rather than a one-off note."
        )
        rec = add_opportunity(
            title=f"Learn recurring {label} workflow",
            description=(
                f"I found {len(group)} recent user messages with the same task shape. "
                "Accepting will run /learn-style skill authoring for that workflow."
            ),
            source="usage",
            action={"type": "learn_skill", "request": request},
            dedup_key=f"usage:learn:{sig}",
            evidence=evidence,
            config=cfg,
        )
        if rec is not None:
            created.append(rec)
        if len(created) >= 3:
            break

    # Domain opportunities complement exact repeated phrasing. They turn a
    # cluster of related requests into a role pack, profile, swarm, or cron path.
    for domain, group in sorted(domain_hits.items(), key=lambda item: len(item[1]), reverse=True):
        if len(group) < threshold:
            continue
        spec = _DOMAIN_KEYWORDS[domain]
        counts = Counter()
        for msg in group:
            counts.update(set(_tokenize(msg["text"])) & spec["terms"])
        top_terms = ", ".join(term for term, _count in counts.most_common(4))
        evidence = [m["preview"] for m in group[:3]]
        rec = add_opportunity(
            title=spec["title"],
            description=(
                f"I found {len(group)} recent messages around {top_terms or domain}. "
                "Accepting will turn that repeated domain into a stronger Hermes workflow."
            ),
            source="usage",
            action={"type": spec["action_type"], "goal": spec["goal"]},
            dedup_key=f"usage:domain:{domain}",
            evidence=evidence,
            config=cfg,
        )
        if rec is not None:
            created.append(rec)

    return {
        "scanned": True,
        "reason": "ok",
        "created": created,
        "message_count": len(messages),
    }


def maybe_scan_recent_usage(
    *,
    config: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    try:
        return scan_recent_usage(force=False, config=config, db_path=db_path)
    except Exception as exc:
        logger.debug("proactive opportunity scan failed: %s", exc)
        return {"scanned": False, "reason": f"error: {exc}", "created": []}


__all__ = [
    "VALID_ACTION_TYPES",
    "VALID_SOURCES",
    "accept_opportunity",
    "add_opportunity",
    "clear_accepted",
    "dismiss_opportunity",
    "get_opportunity",
    "is_enabled",
    "list_pending",
    "load_meta",
    "load_opportunities",
    "maybe_scan_recent_usage",
    "notifications_enabled",
    "proactive_config",
    "scan_recent_usage",
    "seed_starter_opportunities",
    "starter_opportunities",
]
