"""Adapter from Magnus GTM radar artifacts to Torben Signal actions."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .action_ledger import ActionLedger, ActionRecord
from .gtm_x_algorithm import x_algorithm_brief_line, x_algorithm_signal_lens


DEFAULT_MAGNUS_RADAR_PATH = Path("/Users/ericfreeman/magnus/state/gtm-intelligence-radar/latest.json")

PILLAR_LABELS = {
    "security_trends": "Security",
    "security_ai": "Security + AI",
    "ai_engineering_leverage": "AI engineering leverage",
}

ROUTE_LABELS = {
    "longform_article": "longform article",
    "linkedin_or_x_post": "LinkedIn/X post",
    "monitor": "monitor",
    "learn": "learning note",
}

ARTICLE_CREATION_CONTRACT = {
    "source_policy": "Papers/sources support the argument; do not make them the opening spine. Put source notes at the end unless Eric asks for a paper review.",
    "implementation_bar": "Every thought piece needs real controls: commands, configs, scripts, tools, validation checks, and escalation/rollback paths where possible.",
    "reader_value": "Give business and home/operator action items when relevant. The reader should be able to do something today.",
    "visuals": "Include workflow-image prompts with visual goal, labels, and placement. Visuals should explain the system, not decorate the article.",
    "tone": "Eric voice: conversational, sharp, practical, a little snarky/dark-humored when useful; no AI slop, beige corporate phrasing, or hot-fart thought leadership.",
    "x_algorithm_lens": x_algorithm_signal_lens(),
}


def load_magnus_gtm_radar(path: str | Path = DEFAULT_MAGNUS_RADAR_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected Magnus GTM radar JSON object: {path}")
    return payload


def build_torben_gtm_radar_adapter(
    radar: dict[str, Any],
    *,
    ledger: ActionLedger,
    state_path: str | Path,
    max_items: int = 3,
    now: datetime | None = None,
    mark_delivered: bool = True,
    stage_actions: bool = True,
) -> dict[str, Any]:
    """Stage unseen Magnus radar findings as Torben GTM actions."""

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    state_file = Path(state_path)
    state = _load_state(state_file)
    delivered = state.get("delivered_findings") if isinstance(state.get("delivered_findings"), dict) else {}
    findings = [item for item in radar.get("findings", []) if isinstance(item, dict)]
    unseen = [item for item in findings if _finding_key(item) not in delivered]
    selected = unseen[: max(0, max_items)]

    if not selected:
        payload = {
            "task": "torben_gtm_radar_adapter",
            "wakeAgent": False,
            "generated_at": now.isoformat().replace("+00:00", "Z"),
            "reason": "no new GTM radar findings",
            "radar_generated_at": radar.get("generated_at"),
            "scanned_count": radar.get("scanned_count", 0),
            "selected_count": 0,
            "suppressed_duplicate_count": max(0, len(findings) - len(unseen)),
            "llm_judge": radar.get("llm_judge") if isinstance(radar.get("llm_judge"), dict) else {},
            "quality_gate": radar.get("quality_gate") if isinstance(radar.get("quality_gate"), dict) else {},
            "cron_audit": radar.get("cron_audit") if isinstance(radar.get("cron_audit"), dict) else {},
            "public_actions_taken": 0,
            "external_mutations": 0,
        }
        payload["text"] = ""
        return payload

    actions = []
    for rank, finding in enumerate(selected, start=1):
        if stage_actions:
            actions.append(_stage_action(ledger=ledger, finding=finding, rank=rank, now=now))
        else:
            actions.append(_preview_action(finding=finding, rank=rank, now=now))
    text = render_torben_gtm_radar_text(
        radar=radar,
        findings=selected,
        actions=actions,
        now=now,
    )
    payload = {
        "task": "torben_gtm_radar_adapter",
        "wakeAgent": True,
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "radar_generated_at": radar.get("generated_at"),
        "scanned_count": radar.get("scanned_count", 0),
        "selected_count": len(selected),
        "suppressed_duplicate_count": max(0, len(findings) - len(unseen)),
        "findings": selected,
        "actions": [action.to_dict() for action in actions],
        "text": text,
        "llm_judge": radar.get("llm_judge") if isinstance(radar.get("llm_judge"), dict) else {},
        "quality_gate": radar.get("quality_gate") if isinstance(radar.get("quality_gate"), dict) else {},
        "cron_audit": radar.get("cron_audit") if isinstance(radar.get("cron_audit"), dict) else {},
        "public_actions_taken": 0,
        "external_mutations": 0,
        "delivery": {
            "surface": "signal",
            "operator": "torben",
            "source": "magnus_gtm_intelligence_radar",
            "delivery_mode": "adapter_text",
        },
    }
    if mark_delivered and stage_actions:
        _mark_delivered(state_file, state, selected, now=now)
    return payload


def render_torben_gtm_radar_text(
    *,
    radar: dict[str, Any],
    findings: Iterable[dict[str, Any]],
    actions: Iterable[ActionRecord],
    now: datetime,
) -> str:
    finding_rows = list(findings)
    action_rows = list(actions)
    audit = radar.get("cron_audit") if isinstance(radar.get("cron_audit"), dict) else {}
    judge = radar.get("llm_judge") if isinstance(radar.get("llm_judge"), dict) else {}
    judge_line = _judge_line(audit=audit, judge=judge)
    lines = [
        f"Torben / GTM Radar / {now:%Y-%m-%d %H:%M UTC}",
        "",
        (
            f"Magnus reviewed {int(radar.get('scanned_count') or 0)} source item(s) "
            f"and found {len(finding_rows)} signal(s) worth acting on."
        ),
        judge_line,
        x_algorithm_brief_line(),
        "Nothing has been posted, replied to, scheduled, or sent.",
        "",
    ]
    for idx, (finding, action) in enumerate(zip(finding_rows, action_rows), start=1):
        title = _line(finding.get("title"), 118)
        pillar = PILLAR_LABELS.get(str(finding.get("pillar") or ""), str(finding.get("pillar") or "GTM"))
        summary = _sentence_block(finding.get("summary"), limit=420, min_sentences=2, max_sentences=3)
        why = _line(finding.get("why_it_matters"), 220)
        route = ROUTE_LABELS.get(str(finding.get("content_route") or ""), str(finding.get("content_route") or "review"))
        url = str(finding.get("url") or "").strip()
        visual = _line(finding.get("image_direction"), 160)
        lines.extend(
            [
                f"{idx}. [{pillar}] {title}",
                f"Summary: {summary}",
                f"Why it matters: {why}",
                f"Content move: {route}.",
            ]
        )
        if route == "longform article":
            lines.append("Article bar: teach the core concept first; include concrete controls, configs/tools, validation checks, escalation paths, and visual prompts; cite sources at the end.")
        if visual:
            lines.append(f"Visual: {visual}")
        if url:
            lines.append(f"Source: {url}")
        lines.append(f"[{action.handle}] Reply draft {idx}, source {idx}, hold {idx}, or tell me what to change.")
        lines.append("")
    lines.append("This is staged context only. Public writing still requires explicit approval.")
    return "\n".join(lines).rstrip() + "\n"


def write_gtm_radar_adapter_artifacts(
    payload: dict[str, Any],
    *,
    json_path: str | Path,
    text_path: str | Path,
) -> None:
    json_output = Path(json_path)
    text_output = Path(text_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(json_output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_write(text_output, str(payload.get("text") or ""))


def _stage_action(
    *,
    ledger: ActionLedger,
    finding: dict[str, Any],
    rank: int,
    now: datetime,
) -> ActionRecord:
    title = _line(finding.get("title"), 110)
    url = str(finding.get("url") or "").strip()
    finding_id = str(finding.get("id") or _finding_key(finding))
    evidence_ids = [finding_id]
    if url:
        evidence_ids.append(url)
    route = str(finding.get("content_route") or "review")
    return ledger.add_action(
        scope="GTM",
        summary=f"Review GTM radar signal {rank}: {title}",
        evidence_ids=evidence_ids,
        allowed_next_actions=["draft_content", "show_source", "hold"],
        status="staged",
        risk_class="medium",
        ttl_hours=72,
        now=now,
        executor_state={
            "mutation_type": "social_content_draft",
            "mutation_status": "draft_only",
            "provider": "xai-oauth",
            "source": "magnus_gtm_intelligence_radar",
            "radar_rank": rank,
            "radar_finding_id": finding_id,
            "radar_fingerprint": _finding_key(finding),
            "source_url": url,
            "content_route": route,
            "pillar": finding.get("pillar"),
            "title": finding.get("title"),
            "summary": finding.get("summary"),
            "why_it_matters": finding.get("why_it_matters"),
            "thesis": finding.get("thesis"),
            "angle": finding.get("angle"),
            "image_direction": finding.get("image_direction"),
            "llm_judged": bool(finding.get("llm_judged")),
            "llm_score": finding.get("llm_score"),
            "llm_reason": finding.get("llm_reason"),
            "quality_gate": finding.get("quality_gate") if isinstance(finding.get("quality_gate"), dict) else {},
            "article_creation_contract": ARTICLE_CREATION_CONTRACT,
            "x_algorithm_signal_lens": x_algorithm_signal_lens(),
            "reply_actions": ["draft", "source", "hold"],
            "reply_aliases": [f"draft {rank}", f"source {rank}", f"hold {rank}"],
            "publishing_blocked_until": "explicit_signal_approval",
        },
    )


def _preview_action(*, finding: dict[str, Any], rank: int, now: datetime) -> ActionRecord:
    title = _line(finding.get("title"), 110)
    url = str(finding.get("url") or "").strip()
    finding_id = str(finding.get("id") or _finding_key(finding))
    evidence_ids = [finding_id]
    if url:
        evidence_ids.append(url)
    return ActionRecord(
        handle=f"GTM-{now:%Y%m%d}-{rank:03d}",
        scope="gtm",
        summary=f"Review GTM radar signal {rank}: {title}",
        evidence_ids=evidence_ids,
        allowed_next_actions=["draft_content", "show_source", "hold"],
        status="staged",
        risk_class="medium",
        created_at=now,
        user_visible_summary=f"Review GTM radar signal {rank}: {title}",
        executor_state={
            "mutation_type": "social_content_draft",
            "mutation_status": "preview_only",
            "provider": "xai-oauth",
            "source": "magnus_gtm_intelligence_radar",
            "radar_rank": rank,
            "radar_finding_id": finding_id,
            "radar_fingerprint": _finding_key(finding),
            "source_url": url,
            "content_route": finding.get("content_route"),
            "pillar": finding.get("pillar"),
            "llm_judged": bool(finding.get("llm_judged")),
            "llm_score": finding.get("llm_score"),
            "llm_reason": finding.get("llm_reason"),
            "quality_gate": finding.get("quality_gate") if isinstance(finding.get("quality_gate"), dict) else {},
            "article_creation_contract": ARTICLE_CREATION_CONTRACT,
            "x_algorithm_signal_lens": x_algorithm_signal_lens(),
            "reply_actions": ["draft", "source", "hold"],
            "reply_aliases": [f"draft {rank}", f"source {rank}", f"hold {rank}"],
            "publishing_blocked_until": "explicit_signal_approval",
        },
    )


def _finding_key(finding: dict[str, Any]) -> str:
    for key in ("fingerprint", "id", "url"):
        value = str(finding.get(key) or "").strip()
        if value:
            return value
    return "finding:" + _normalize(str(finding.get("title") or "untitled"))


def _judge_line(*, audit: dict[str, Any], judge: dict[str, Any]) -> str:
    llm_invoked = bool(audit.get("llm_invoked") if audit else judge.get("invoked"))
    model = str(audit.get("model") or judge.get("model") or "unknown").strip() or "unknown"
    x_search = bool(audit.get("x_search_used") if audit else judge.get("x_search_used"))
    status = str(judge.get("status") or ("accepted" if llm_invoked else "not_invoked"))
    if llm_invoked:
        return f"LLM judge: Grok ran ({model}); x_search_used={str(x_search).lower()}; status={status}."
    return "LLM judge: not invoked; treat this as deterministic fallback, not final GTM signal."


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "delivered_findings": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"schema_version": 1, "delivered_findings": {}}
    return payload if isinstance(payload, dict) else {"schema_version": 1, "delivered_findings": {}}


def _mark_delivered(path: Path, state: dict[str, Any], findings: list[dict[str, Any]], *, now: datetime) -> None:
    delivered = state.get("delivered_findings") if isinstance(state.get("delivered_findings"), dict) else {}
    now_text = now.isoformat().replace("+00:00", "Z")
    for finding in findings:
        key = _finding_key(finding)
        existing = delivered.get(key) if isinstance(delivered.get(key), dict) else {}
        delivered[key] = {
            "id": finding.get("id"),
            "title": finding.get("title"),
            "url": finding.get("url"),
            "first_delivered_at": existing.get("first_delivered_at") or now_text,
            "last_seen_at": now_text,
        }
    state = {
        "schema_version": 1,
        "updated_at": now_text,
        "delivered_findings": delivered,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(state, indent=2, sort_keys=True) + "\n")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _line(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit].rstrip()
    boundary = max(clipped.rfind(". "), clipped.rfind("; "), clipped.rfind(", "), clipped.rfind(" "))
    if boundary > limit // 2:
        clipped = clipped[:boundary].rstrip()
    return clipped.rstrip(".,;:") + "..."


def _sentence_block(value: Any, *, limit: int, min_sentences: int, max_sentences: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    selected: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*selected, sentence]).strip()
        if len(selected) < min_sentences or len(candidate) <= limit:
            selected.append(sentence)
            if len(selected) >= max_sentences:
                break
            continue
        break
    return _line(" ".join(selected), limit)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
