"""GTM response-opportunity radar for Torben.

This module uses Grok/X Search after deterministic topic selection. It stages
draft-only reply opportunities for Signal review and never posts to X.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from tools.xai_http import hermes_xai_user_agent, resolve_xai_http_credentials

from .action_ledger import ActionLedger, ActionRecord
from .gtm_x_algorithm import x_algorithm_brief_line, x_algorithm_signal_lens


DEFAULT_MODEL = "grok-4.3"
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_MAX_TOPICS = 3
DEFAULT_MAX_OPPORTUNITIES = 3


@dataclass(frozen=True)
class GTMResponseOpportunity:
    fingerprint: str
    post_url: str
    author_handle: str
    author_name: str
    post_summary: str
    why_reply: str
    reply_angle: str
    draft_reply: str
    score: int
    risk_notes: list[str]
    source_topic: str
    source_url: str


def run_gtm_engagement_radar(
    radar: dict[str, Any],
    *,
    ledger: ActionLedger,
    state_path: str | Path,
    max_topics: int = DEFAULT_MAX_TOPICS,
    max_opportunities: int = DEFAULT_MAX_OPPORTUNITIES,
    mark_delivered: bool = True,
    stage_actions: bool = True,
    now: datetime | None = None,
    discover: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Search X for reply opportunities around the latest GTM radar topics."""

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    state_file = Path(state_path)
    topics = select_engagement_topics(radar, max_topics=max_topics)
    if not topics:
        payload = _silent_payload(now=now, reason="no GTM topics available")
        payload.update(
            _audit_fields(
                discovery={},
                topic_count=0,
                opportunity_count=0,
                selected_count=0,
                wake_agent=False,
                wake_reason="no_gtm_topics_available",
                llm_invoked=False,
            )
        )
        payload.update({"topic_count": 0, "opportunity_count": 0, "selected_count": 0})
        return payload

    discover_fn = discover or discover_x_response_opportunities
    discovery = discover_fn(topics=topics, max_opportunities=max_opportunities, now=now)
    opportunities = _normalize_opportunities(
        discovery.get("opportunities") if isinstance(discovery, dict) else [],
        topics=topics,
        max_opportunities=max_opportunities,
    )
    state = _load_state(state_file)
    delivered = state.get("delivered_opportunities") if isinstance(state.get("delivered_opportunities"), dict) else {}
    fresh = [opp for opp in opportunities if opp.fingerprint not in delivered]

    if not fresh:
        payload = _silent_payload(now=now, reason="no new response opportunities")
        payload.update(
            {
                "topic_count": len(topics),
                "opportunity_count": len(opportunities),
                "selected_count": 0,
                "suppressed_duplicate_count": len(opportunities),
                "discovery": _safe_discovery_meta(discovery),
            }
        )
        payload.update(
            _audit_fields(
                discovery=discovery,
                topic_count=len(topics),
                opportunity_count=len(opportunities),
                selected_count=0,
                wake_agent=False,
                wake_reason="no_new_response_opportunities",
                llm_invoked=True,
            )
        )
        return payload

    actions = [
        _stage_response_action(ledger=ledger, opportunity=opp, rank=rank, now=now)
        if stage_actions
        else _preview_response_action(opportunity=opp, rank=rank, now=now)
        for rank, opp in enumerate(fresh[:max_opportunities], start=1)
    ]
    text = render_gtm_engagement_text(
        opportunities=fresh[:max_opportunities],
        actions=actions,
        topic_count=len(topics),
        now=now,
    )
    payload = {
        "task": "torben_gtm_engagement_radar",
        "wakeAgent": True,
        "generated_at": _iso(now),
        "topic_count": len(topics),
        "opportunity_count": len(opportunities),
        "selected_count": len(fresh[:max_opportunities]),
        "suppressed_duplicate_count": max(0, len(opportunities) - len(fresh)),
        "topics": topics,
        "opportunities": [asdict(opp) for opp in fresh[:max_opportunities]],
        "actions": [action.to_dict() for action in actions],
        "discovery": _safe_discovery_meta(discovery),
        "x_algorithm_signal_lens": x_algorithm_signal_lens(),
        "text": text,
        "public_actions_taken": 0,
        "external_mutations": 0,
    }
    payload.update(
        _audit_fields(
            discovery=discovery,
            topic_count=len(topics),
            opportunity_count=len(opportunities),
            selected_count=len(fresh[:max_opportunities]),
            wake_agent=True,
            wake_reason="llm_judged_reply_opportunities_selected",
            llm_invoked=True,
        )
    )
    if mark_delivered and stage_actions:
        _mark_delivered(state_file, state, fresh[:max_opportunities], now=now)
    return payload


def select_engagement_topics(radar: dict[str, Any], *, max_topics: int = DEFAULT_MAX_TOPICS) -> list[dict[str, str]]:
    findings = [item for item in radar.get("findings", []) if isinstance(item, dict)]
    topics: list[dict[str, str]] = []
    for finding in findings:
        route = str(finding.get("content_route") or "")
        pillar = str(finding.get("pillar") or "")
        title = _clean_line(finding.get("title"), 140)
        summary = _clean_line(finding.get("summary"), 260)
        if not title or route not in {"longform_article", "linkedin_or_x_post", "monitor"}:
            continue
        if pillar not in {"security_ai", "ai_engineering_leverage", "security_trends"}:
            continue
        topics.append(
            {
                "title": title,
                "summary": summary,
                "pillar": pillar,
                "source_url": str(finding.get("url") or "").strip(),
                "query": _query_for_finding(title=title, summary=summary, pillar=pillar),
            }
        )
        if len(topics) >= max_topics:
            break
    return topics


def discover_x_response_opportunities(
    *,
    topics: list[dict[str, str]],
    max_opportunities: int = DEFAULT_MAX_OPPORTUNITIES,
    now: datetime | None = None,
    model: str | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Use xAI Responses with x_search to find concrete reply opportunities."""

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    selected_model = str(model or os.getenv("TORBEN_GTM_ENGAGEMENT_MODEL") or DEFAULT_MODEL).strip()
    timeout = _positive_int(
        timeout_seconds or os.getenv("TORBEN_GTM_ENGAGEMENT_TIMEOUT_SECONDS"),
        DEFAULT_TIMEOUT_SECONDS,
    )
    creds = resolve_xai_http_credentials()
    api_key = str(creds.get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError("xAI OAuth/API credentials are unavailable")
    base_url = str(creds.get("base_url") or "https://api.x.ai/v1").strip().rstrip("/")
    request_payload = _responses_payload(
        topics=topics,
        max_opportunities=max_opportunities,
        model=selected_model,
    )
    response = requests.post(
        f"{base_url}/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": hermes_xai_user_agent(),
        },
        json=request_payload,
        timeout=timeout,
    )
    response.raise_for_status()
    response_payload = response.json()
    generated = _parse_json_object(_extract_response_text(response_payload))
    generated["provider"] = str(creds.get("provider") or "xai")
    generated["model"] = selected_model
    generated["x_search_enabled"] = True
    generated["x_search_used"] = True
    generated["generated_at"] = _iso(now)
    generated["citations"] = _extract_citations(response_payload)
    return generated


def render_gtm_engagement_text(
    *,
    opportunities: list[GTMResponseOpportunity],
    actions: list[ActionRecord],
    topic_count: int,
    now: datetime,
) -> str:
    lines = [
        f"Torben / GTM Response Radar / {now:%Y-%m-%d %H:%M UTC}",
        "",
        f"Grok/X reviewed {topic_count} GTM topic(s) and found {len(opportunities)} reply opportunity(s).",
        "LLM judge: Grok ran with x_search; all public writes remain approval-gated.",
        x_algorithm_brief_line(),
        "Nothing has been posted, replied to publicly, scheduled, or sent.",
        "",
    ]
    for idx, (opp, action) in enumerate(zip(opportunities, actions), start=1):
        author = f"@{opp.author_handle.lstrip('@')}" if opp.author_handle else opp.author_name or "unknown author"
        lines.extend(
            [
                f"{idx}. {author}: {_clean_line(opp.post_summary, 220)}",
                f"Why respond: {_clean_line(opp.why_reply, 220)}",
                f"Angle: {_clean_line(opp.reply_angle, 220)}",
                f"Draft reply: {_clean_line(opp.draft_reply, 280)}",
            ]
        )
        if opp.risk_notes:
            lines.append(f"Risk: {_clean_line('; '.join(opp.risk_notes[:2]), 180)}")
        if opp.post_url:
            lines.append(f"Source: {opp.post_url}")
        lines.append(f"[{action.handle}] Tell me what to change, ask for source, or hold this reply idea.")
        lines.append("")
    lines.append("This is staged context only. Public replies still require a separate explicit approval path.")
    return "\n".join(lines).rstrip() + "\n"


def write_gtm_engagement_artifacts(
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


def _responses_payload(
    *,
    topics: list[dict[str, str]],
    max_opportunities: int,
    model: str,
) -> dict[str, Any]:
    compact = {
        "topics": topics,
        "max_opportunities": max_opportunities,
        "x_algorithm_signal_lens": x_algorithm_signal_lens(),
    }
    prompt = (
        "Find concrete X reply opportunities for Eric Freeman's Magnus GTM operator.\n"
        "Use X Search for current posts only. Return only one valid JSON object.\n\n"
        "Return JSON with key opportunities, an array of objects with exactly these keys:\n"
        "post_url, author_handle, author_name, post_summary, why_reply, reply_angle, "
        "draft_reply, score, risk_notes, source_topic, source_url.\n\n"
        "Rules:\n"
        "- Pick posts where a useful reply could increase trust, profile-click intent, dwell, or follow intent.\n"
        "- Avoid outrage bait, low-context fights, vague AI hype, or posts likely to trigger not-interested, block, mute, or report signals.\n"
        "- Draft replies in Eric's voice: sharp, useful, direct, lightly dry when earned, never a personal dunk.\n"
        "- Do not ask to post. Do not claim exact private X ranking weights.\n"
        "- If there are no strong opportunities, return {\"opportunities\": []}.\n\n"
        f"Input JSON:\n{json.dumps(compact, ensure_ascii=False, sort_keys=True)}"
    )
    return {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are Magnus, Eric Freeman's GTM engagement operator. "
                    "You identify high-signal X reply opportunities and draft approval-gated replies."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "tools": [{"type": "x_search"}],
        "store": False,
    }


def _normalize_opportunities(
    rows: Any,
    *,
    topics: list[dict[str, str]],
    max_opportunities: int,
) -> list[GTMResponseOpportunity]:
    if not isinstance(rows, list):
        return []
    topic_by_title = {topic["title"]: topic for topic in topics if topic.get("title")}
    normalized: list[GTMResponseOpportunity] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        post_url = str(raw.get("post_url") or raw.get("url") or "").strip()
        post_summary = _clean_line(raw.get("post_summary") or raw.get("summary"), 420)
        why_reply = _clean_line(raw.get("why_reply"), 320)
        reply_angle = _clean_line(raw.get("reply_angle"), 320)
        draft_reply = _clean_line(raw.get("draft_reply"), 420)
        if not post_url or not post_summary or not why_reply or not draft_reply:
            continue
        score = max(0, min(100, _safe_int(raw.get("score"), 0)))
        if score < 60:
            continue
        source_topic = _clean_line(raw.get("source_topic"), 160)
        topic = topic_by_title.get(source_topic) or (topics[0] if topics else {})
        source_url = str(raw.get("source_url") or topic.get("source_url") or "").strip()
        opportunity = GTMResponseOpportunity(
            fingerprint=_opportunity_fingerprint(post_url=post_url, draft_reply=draft_reply),
            post_url=post_url,
            author_handle=str(raw.get("author_handle") or "").strip().lstrip("@"),
            author_name=_clean_line(raw.get("author_name"), 120),
            post_summary=post_summary,
            why_reply=why_reply,
            reply_angle=reply_angle,
            draft_reply=draft_reply,
            score=score,
            risk_notes=_normalize_risk_notes(raw.get("risk_notes")),
            source_topic=source_topic or str(topic.get("title") or ""),
            source_url=source_url,
        )
        normalized.append(opportunity)
        if len(normalized) >= max_opportunities:
            break
    return normalized


def _stage_response_action(
    *,
    ledger: ActionLedger,
    opportunity: GTMResponseOpportunity,
    rank: int,
    now: datetime,
) -> ActionRecord:
    return ledger.add_action(
        scope="GTM",
        summary=f"Review X reply opportunity {rank}: @{opportunity.author_handle or 'unknown'}",
        evidence_ids=[opportunity.fingerprint, opportunity.post_url],
        allowed_next_actions=["revise_reply_draft", "show_source", "hold"],
        status="staged",
        risk_class="medium",
        ttl_hours=24,
        now=now,
        executor_state={
            "mutation_type": "social_reply_draft",
            "mutation_status": "draft_only",
            "provider": "xai-oauth",
            "source": "torben_gtm_engagement_radar",
            "radar_rank": rank,
            "post_url": opportunity.post_url,
            "author_handle": opportunity.author_handle,
            "author_name": opportunity.author_name,
            "post_summary": opportunity.post_summary,
            "why_reply": opportunity.why_reply,
            "reply_angle": opportunity.reply_angle,
            "draft_reply": opportunity.draft_reply,
            "score": opportunity.score,
            "risk_notes": opportunity.risk_notes,
            "source_topic": opportunity.source_topic,
            "source_url": opportunity.source_url,
            "llm_judged": True,
            "llm_score": opportunity.score,
            "llm_reason": opportunity.why_reply,
            "x_algorithm_signal_lens": x_algorithm_signal_lens(),
            "publishing_blocked_until": "separate_explicit_public_reply_approval",
            "public_actions_taken": 0,
            "external_mutations": 0,
        },
    )


def _preview_response_action(*, opportunity: GTMResponseOpportunity, rank: int, now: datetime) -> ActionRecord:
    return ActionRecord(
        handle=f"GTM-{now:%Y%m%d}-{rank:03d}",
        scope="gtm",
        summary=f"Review X reply opportunity {rank}: @{opportunity.author_handle or 'unknown'}",
        evidence_ids=[opportunity.fingerprint, opportunity.post_url],
        allowed_next_actions=["revise_reply_draft", "show_source", "hold"],
        status="staged",
        risk_class="medium",
        created_at=now,
        user_visible_summary=f"Review X reply opportunity {rank}: @{opportunity.author_handle or 'unknown'}",
        executor_state={
            "mutation_type": "social_reply_draft",
            "mutation_status": "preview_only",
            "source": "torben_gtm_engagement_radar",
            "draft_reply": opportunity.draft_reply,
            "post_url": opportunity.post_url,
            "author_handle": opportunity.author_handle,
            "author_name": opportunity.author_name,
            "post_summary": opportunity.post_summary,
            "why_reply": opportunity.why_reply,
            "reply_angle": opportunity.reply_angle,
            "score": opportunity.score,
            "risk_notes": opportunity.risk_notes,
            "source_topic": opportunity.source_topic,
            "source_url": opportunity.source_url,
            "llm_judged": True,
            "llm_score": opportunity.score,
            "llm_reason": opportunity.why_reply,
            "x_algorithm_signal_lens": x_algorithm_signal_lens(),
            "publishing_blocked_until": "separate_explicit_public_reply_approval",
            "public_actions_taken": 0,
            "external_mutations": 0,
        },
    )


def _query_for_finding(*, title: str, summary: str, pillar: str) -> str:
    topic_terms = _topic_terms(title + " " + summary)
    pillar_hint = {
        "security_ai": "AI security agent security prompt injection MCP evals",
        "ai_engineering_leverage": "AI engineering agents evals workflow performance",
        "security_trends": "security threat hunting hardening vulnerability response",
    }.get(pillar, "AI security engineering")
    return f"{topic_terms} {pillar_hint}".strip()


def _topic_terms(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+'-]{2,}", text)
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "paper",
        "study",
        "using",
        "large",
        "model",
        "models",
    }
    selected: list[str] = []
    seen = set()
    for word in words:
        lowered = word.lower()
        if lowered in stop or lowered in seen:
            continue
        seen.add(lowered)
        selected.append(word)
        if len(selected) >= 8:
            break
    return " ".join(selected)


def _safe_discovery_meta(discovery: Any) -> dict[str, Any]:
    if not isinstance(discovery, dict):
        return {}
    return {
        "provider": discovery.get("provider"),
        "model": discovery.get("model"),
        "x_search_enabled": bool(discovery.get("x_search_enabled")),
        "x_search_used": bool(discovery.get("x_search_used", discovery.get("x_search_enabled"))),
        "generated_at": discovery.get("generated_at"),
        "citations": copy.deepcopy(discovery.get("citations") or [])[:10],
    }


def _normalize_risk_notes(value: Any) -> list[str]:
    if isinstance(value, str):
        rows = [value]
    elif isinstance(value, list):
        rows = value
    else:
        rows = []
    notes: list[str] = []
    for note in rows:
        cleaned = _clean_line(note, 160)
        if cleaned:
            notes.append(cleaned)
    return notes[:3]


def _silent_payload(*, now: datetime, reason: str) -> dict[str, Any]:
    return {
        "task": "torben_gtm_engagement_radar",
        "wakeAgent": False,
        "generated_at": _iso(now),
        "reason": reason,
        "text": "",
        "public_actions_taken": 0,
        "external_mutations": 0,
    }


def _audit_fields(
    *,
    discovery: Any,
    topic_count: int,
    opportunity_count: int,
    selected_count: int,
    wake_agent: bool,
    wake_reason: str,
    llm_invoked: bool,
) -> dict[str, Any]:
    meta = _safe_discovery_meta(discovery)
    model = str(meta.get("model") or os.getenv("TORBEN_GTM_ENGAGEMENT_MODEL") or DEFAULT_MODEL).strip()
    x_search_used = bool(meta.get("x_search_used")) if llm_invoked else False
    status = "accepted" if selected_count > 0 else ("no_new_opportunities" if llm_invoked else "not_invoked")
    return {
        "llm_judge": {
            "enabled": True,
            "required": True,
            "invoked": bool(llm_invoked),
            "provider": meta.get("provider") or ("xai-oauth" if llm_invoked else None),
            "model": model,
            "x_search_requested": bool(topic_count > 0),
            "x_search_used": x_search_used,
            "candidate_count": int(opportunity_count),
            "decision_count": int(opportunity_count),
            "accepted_count": int(selected_count),
            "rejected_count": max(0, int(opportunity_count) - int(selected_count)),
            "status": status,
            "public_actions_taken": 0,
            "external_mutations": 0,
        },
        "cron_audit": {
            "llm_invoked": bool(llm_invoked),
            "model": model,
            "x_search_used": x_search_used,
            "why_wake_agent": bool(wake_agent),
            "wake_reason": wake_reason,
            "public_actions_taken": 0,
            "external_mutations": 0,
        },
    }


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "delivered_opportunities": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"schema_version": 1, "delivered_opportunities": {}}
    return payload if isinstance(payload, dict) else {"schema_version": 1, "delivered_opportunities": {}}


def _mark_delivered(path: Path, state: dict[str, Any], opportunities: list[GTMResponseOpportunity], *, now: datetime) -> None:
    delivered = state.get("delivered_opportunities") if isinstance(state.get("delivered_opportunities"), dict) else {}
    now_text = _iso(now)
    for opportunity in opportunities:
        existing = delivered.get(opportunity.fingerprint) if isinstance(delivered.get(opportunity.fingerprint), dict) else {}
        delivered[opportunity.fingerprint] = {
            "post_url": opportunity.post_url,
            "author_handle": opportunity.author_handle,
            "first_delivered_at": existing.get("first_delivered_at") or now_text,
            "last_seen_at": now_text,
        }
    state = {
        "schema_version": 1,
        "updated_at": now_text,
        "delivered_opportunities": delivered,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(state, indent=2, sort_keys=True) + "\n")


def _extract_response_text(payload: dict[str, Any]) -> str:
    output_text = str(payload.get("output_text") or "").strip()
    if output_text:
        return output_text
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"}:
                text = str(content.get("text") or "").strip()
                if text:
                    parts.append(text)
    return "\n\n".join(parts).strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Grok response did not contain a JSON object") from None
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Grok response JSON was not an object")
    return payload


def _extract_citations(payload: dict[str, Any]) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    seen = set()
    for item in payload.get("citations", []) or []:
        if isinstance(item, str):
            url = item.strip()
            title = ""
        elif isinstance(item, dict):
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
        else:
            continue
        if url and url not in seen:
            citations.append({"url": url, "title": title})
            seen.add(url)
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            for annotation in content.get("annotations", []) or []:
                if not isinstance(annotation, dict) or annotation.get("type") != "url_citation":
                    continue
                url = str(annotation.get("url") or "").strip()
                title = str(annotation.get("title") or "").strip()
                if url and url not in seen:
                    citations.append({"url": url, "title": title})
                    seen.add(url)
    return citations


def _opportunity_fingerprint(*, post_url: str, draft_reply: str) -> str:
    key = post_url.strip() or draft_reply.strip()
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return f"gtm-response:{digest}"


def _clean_line(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit].rstrip()
    boundary = max(clipped.rfind(". "), clipped.rfind("; "), clipped.rfind(", "), clipped.rfind(" "))
    if boundary > limit // 2:
        clipped = clipped[:boundary].rstrip()
    return clipped.rstrip(".,;:") + "..."


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(str(value).strip())
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
