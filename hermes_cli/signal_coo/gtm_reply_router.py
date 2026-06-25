"""Resolve GTM radar Signal replies into draft-only content packages."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .action_ledger import HANDLE_RE, ActionLedger, ActionRecord, utc_now
from .gtm_grok_writer import enrich_package_with_grok
from .gtm_x_algorithm import x_algorithm_brief_line, x_algorithm_signal_lens


@dataclass
class GTMReplyRouteResult:
    handled: bool
    status: str
    text: str = ""
    reason: str | None = None
    package_action: ActionRecord | None = None
    referenced_actions: list[ActionRecord] = field(default_factory=list)
    artifact_path: Path | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "handled": self.handled,
            "status": self.status,
            "reason": self.reason,
            "text": self.text,
            "package_action": self.package_action.to_dict() if self.package_action else None,
            "referenced_actions": [action.to_dict() for action in self.referenced_actions],
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
            "payload": self.payload,
            "public_actions_taken": 0,
            "external_mutations": 0,
        }


def route_gtm_radar_reply(
    *,
    ledger: ActionLedger,
    reply_text: str,
    output_dir: str | Path,
    now: datetime | None = None,
    approved_by: str = "signal-reply",
) -> GTMReplyRouteResult:
    """Stage a GTM content package from a reply to a Torben radar action.

    This intentionally performs only local ledger/artifact mutations. It never
    posts, schedules, replies on X/LinkedIn, or sends content outside Hermes.
    """

    now = (now or utc_now()).astimezone(timezone.utc)
    reply_text = str(reply_text or "").strip()
    if not reply_text:
        return GTMReplyRouteResult(handled=False, status="ignored", reason="empty reply")

    targets, reason = _resolve_targets(ledger, reply_text, now=now)
    if not targets:
        return GTMReplyRouteResult(handled=False, status="not_found", reason=reason)

    non_radar = [target for target in targets if not _is_gtm_radar_action(target)]
    if non_radar:
        return GTMReplyRouteResult(
            handled=False,
            status="not_gtm_radar",
            reason="reply did not target an open Magnus GTM radar action",
        )

    intent = _infer_intent(reply_text)
    if intent == "hold":
        return _hold_targets(
            ledger=ledger,
            targets=targets,
            reply_text=reply_text,
            now=now,
            approved_by=approved_by,
        )
    if intent == "source":
        return _show_sources(targets=targets, reply_text=reply_text)

    return _stage_content_package(
        ledger=ledger,
        targets=targets,
        reply_text=reply_text,
        output_dir=Path(output_dir),
        now=now,
        approved_by=approved_by,
    )


def _resolve_targets(
    ledger: ActionLedger,
    reply_text: str,
    *,
    now: datetime,
) -> tuple[list[ActionRecord], str | None]:
    handles = []
    seen = set()
    for match in HANDLE_RE.finditer(reply_text.upper()):
        handle = match.group("handle")
        if handle not in seen:
            handles.append(handle)
            seen.add(handle)

    if handles:
        targets: list[ActionRecord] = []
        missing = []
        closed = []
        for handle in handles:
            record = ledger.get(handle)
            if record is None:
                missing.append(handle)
                continue
            if not record.is_open(now):
                closed.append(handle)
                continue
            targets.append(record)
        if missing:
            return [], f"unknown handle(s): {', '.join(missing)}"
        if closed:
            return [], f"closed or expired handle(s): {', '.join(closed)}"
        return targets, None

    resolution = ledger.resolve_reply(reply_text, now=now)
    if resolution.record is None or resolution.status not in {
        "resolved",
        "resolved_alias",
        "resolved_recent",
    }:
        return [], resolution.reason or resolution.status
    return [resolution.record], resolution.reason


def _stage_content_package(
    *,
    ledger: ActionLedger,
    targets: list[ActionRecord],
    reply_text: str,
    output_dir: Path,
    now: datetime,
    approved_by: str,
) -> GTMReplyRouteResult:
    package_kind = _infer_package_kind(reply_text, targets)
    title = _package_title(targets)
    evidence_ids = _package_evidence_ids(targets)
    package_action = ledger.add_action(
        scope="GTM",
        summary=f"Draft GTM content package: {title}",
        evidence_ids=evidence_ids,
        allowed_next_actions=[
            "approve_article_draft",
            "approve_linkedin_draft",
            "approve_x_thread_draft",
            "revise_package",
            "hold",
        ],
        status="staged",
        risk_class="medium",
        ttl_hours=168,
        now=now,
        executor_state={
            "mutation_type": "social_content_package",
            "mutation_status": "approval_ready_draft_package",
            "provider": "xai-oauth",
            "source": "torben_gtm_reply_router",
            "content_package_kind": package_kind,
            "content_package_status": "approval_required",
            "referenced_handles": [target.handle for target in targets],
            "reply_text": reply_text,
            "publishing_blocked_until": "explicit_signal_approval",
            "draft_generation_mode": "deterministic_from_magnus_radar",
            "public_actions_taken": 0,
            "external_mutations": 0,
        },
    )

    package_payload = _build_package_payload(
        package_action=package_action,
        targets=targets,
        reply_text=reply_text,
        package_kind=package_kind,
        now=now,
    )
    package_payload = enrich_package_with_grok(package_payload, now=now)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{package_action.handle.lower()}-package.json"
    _atomic_write(artifact_path, json.dumps(package_payload, indent=2, sort_keys=True) + "\n")

    records = ledger.load()
    target_handles = {target.handle for target in targets}
    for record in records:
        if record.handle == package_action.handle:
            record.executor_state.update(
                {
                    "artifact_path": str(artifact_path),
                    "package_title": title,
                    "package_payload": package_payload,
                }
            )
            continue
        if record.handle not in target_handles:
            continue
        record.status = "executed"
        record.executor_state.update(
            {
                "mutation_status": "content_package_staged",
                "content_package_handle": package_action.handle,
                "content_package_artifact_path": str(artifact_path),
                "resolved_at": _iso(now),
            }
        )
        record.resolution_history.append(
            {
                "at": _iso(now),
                "status": "content_package_staged",
                "reason": f"GTM radar reply routed by {approved_by}.",
                "content_package_handle": package_action.handle,
            }
        )
    ledger.save(records)

    refreshed_package = ledger.get(package_action.handle) or package_action
    text = _render_package_ack(
        package_action=refreshed_package,
        targets=targets,
        artifact_path=artifact_path,
        package_payload=package_payload,
    )
    return GTMReplyRouteResult(
        handled=True,
        status="content_package_staged",
        text=text,
        package_action=refreshed_package,
        referenced_actions=targets,
        artifact_path=artifact_path,
        payload=package_payload,
    )


def _hold_targets(
    *,
    ledger: ActionLedger,
    targets: list[ActionRecord],
    reply_text: str,
    now: datetime,
    approved_by: str,
) -> GTMReplyRouteResult:
    target_handles = {target.handle for target in targets}
    records = ledger.load()
    for record in records:
        if record.handle not in target_handles:
            continue
        record.status = "discarded"
        record.executor_state.update(
            {
                "mutation_status": "held_by_signal_reply",
                "resolved_at": _iso(now),
                "reply_text": reply_text,
            }
        )
        record.resolution_history.append(
            {
                "at": _iso(now),
                "status": "held",
                "reason": f"GTM radar action held by {approved_by}.",
            }
        )
    ledger.save(records)
    lines = ["Torben / GTM Radar", "", "Held these GTM radar action(s):"]
    for target in targets:
        lines.append(f"- [{target.handle}] {_title_for(target)}")
    lines.append("")
    lines.append("Nothing was posted, scheduled, or sent.")
    return GTMReplyRouteResult(
        handled=True,
        status="held",
        text="\n".join(lines).rstrip() + "\n",
        referenced_actions=targets,
    )


def _show_sources(*, targets: list[ActionRecord], reply_text: str) -> GTMReplyRouteResult:
    lines = ["Torben / GTM Radar Sources", ""]
    for target in targets:
        state = target.executor_state or {}
        lines.append(f"[{target.handle}] {_title_for(target)}")
        summary = _clean_line(state.get("summary"), 280)
        thesis = _clean_line(state.get("thesis"), 240)
        angle = _clean_line(state.get("angle"), 240)
        url = str(state.get("source_url") or "").strip()
        if summary:
            lines.append(f"Summary: {summary}")
        if thesis:
            lines.append(f"Thesis: {thesis}")
        if angle:
            lines.append(f"Angle: {angle}")
        if url:
            lines.append(f"Source: {url}")
        lines.append("")
    lines.append("Nothing was posted, scheduled, or sent.")
    return GTMReplyRouteResult(
        handled=True,
        status="source_shown",
        text="\n".join(lines).rstrip() + "\n",
        reason=reply_text,
        referenced_actions=targets,
    )


def _build_package_payload(
    *,
    package_action: ActionRecord,
    targets: list[ActionRecord],
    reply_text: str,
    package_kind: str,
    now: datetime,
) -> dict[str, Any]:
    sources = [_source_entry(target) for target in targets]
    working_title = _package_title(targets)
    thesis = _package_thesis(sources)
    content_moves = _content_moves(package_kind)
    optimization_lens = x_algorithm_signal_lens()
    outline = _outline(sources)
    visual_plan = _visual_plan(sources=sources, working_title=working_title)
    drafts = _draft_assets(
        working_title=working_title,
        thesis=thesis,
        sources=sources,
        package_kind=package_kind,
        visual_plan=visual_plan,
        optimization_lens=optimization_lens,
    )
    return {
        "schema_version": 1,
        "generated_at": _iso(now),
        "package_handle": package_action.handle,
        "package_kind": package_kind,
        "operator": "torben",
        "source": "torben_gtm_reply_router",
        "reply_text": reply_text,
        "public_actions_taken": 0,
        "external_mutations": 0,
        "publishing_blocked_until": "explicit_signal_approval",
        "source_handles": [target.handle for target in targets],
        "sources": sources,
        "optimization_lens": optimization_lens,
        "content_package_status": "approval_required",
        "brief": {
            "working_title": working_title,
            "thesis": thesis,
            "audience": "security, AI, and founder/operator audience on LinkedIn and X",
            "voice": "sharp, direct, founder-led, useful, high-conviction",
            "content_moves": content_moves,
            "x_algorithm_pressure_test": x_algorithm_brief_line(),
            "outline": outline,
            "visual_direction": visual_plan["summary"],
            "next_step": "Review, revise, or explicitly approve one draft asset. Public posting remains blocked.",
        },
        "drafts": drafts,
        "visual_plan": visual_plan,
        "approval_actions": _approval_actions(package_action.handle, content_moves),
    }


def _render_package_ack(
    *,
    package_action: ActionRecord,
    targets: list[ActionRecord],
    artifact_path: Path,
    package_payload: dict[str, Any],
) -> str:
    brief = package_payload.get("brief") if isinstance(package_payload.get("brief"), dict) else {}
    drafts = package_payload.get("drafts") if isinstance(package_payload.get("drafts"), dict) else {}
    linkedin = drafts.get("linkedin_post") if isinstance(drafts.get("linkedin_post"), dict) else {}
    thread = drafts.get("x_thread") if isinstance(drafts.get("x_thread"), dict) else {}
    article = drafts.get("article") if isinstance(drafts.get("article"), dict) else {}
    visual_plan = package_payload.get("visual_plan") if isinstance(package_payload.get("visual_plan"), dict) else {}
    grok_authoring = (
        package_payload.get("grok_authoring")
        if isinstance(package_payload.get("grok_authoring"), dict)
        else {}
    )
    thread_posts = thread.get("posts") if isinstance(thread.get("posts"), list) else []
    authoring_status = str(grok_authoring.get("status") or "deterministic").strip()
    if authoring_status == "success":
        authoring_line = (
            f"Authoring: Grok ({grok_authoring.get('model') or 'model unknown'})"
            f"{' with X Search context' if grok_authoring.get('x_search_enabled') else ''}."
        )
    elif authoring_status == "failed":
        authoring_line = (
            "Authoring: deterministic fallback because Grok failed "
            f"({grok_authoring.get('error_type') or 'unknown'})."
        )
    elif authoring_status == "disabled":
        authoring_line = "Authoring: deterministic fallback because Grok drafting is disabled."
    else:
        authoring_line = "Authoring: deterministic draft from Magnus radar evidence."
    lines = [
        "Torben / GTM Content Package",
        "",
        (
            f"Got it. I staged an approval-ready Magnus package as [{package_action.handle}], "
            "not as a generic chat reply."
        ),
        "",
        "Inputs:",
    ]
    for target in targets:
        url = str((target.executor_state or {}).get("source_url") or "").strip()
        suffix = f" - {url}" if url else ""
        lines.append(f"- [{target.handle}] {_title_for(target)}{suffix}")
    lines.extend(
        [
            "",
            f"Working title: {_clean_line(brief.get('working_title'), 180)}",
            f"Thesis: {_clean_line(brief.get('thesis'), 260)}",
            "",
            "Drafted now:",
            f"- {authoring_line}",
            f"- Article: {_clean_line(article.get('dek') or article.get('title'), 220)}",
            f"- LinkedIn opener: {_clean_line(linkedin.get('opener'), 240)}",
            f"- X opener: {_clean_line(thread_posts[0] if thread_posts else '', 240)}",
            f"- Visual: {_clean_line(visual_plan.get('summary'), 220)}",
            f"- {x_algorithm_brief_line()}",
            "",
            "Reply with: approve article, approve linkedin, approve x thread, revise, or hold.",
            f"Artifact: {artifact_path}",
            "",
            "Nothing has been posted, scheduled, replied to publicly, or sent.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _is_gtm_radar_action(record: ActionRecord) -> bool:
    state = record.executor_state or {}
    return (
        record.scope == "gtm"
        and state.get("mutation_type") == "social_content_draft"
        and state.get("source") == "magnus_gtm_intelligence_radar"
    )


def _infer_intent(reply_text: str) -> str:
    text = f" {reply_text.lower()} "
    if re.search(r"\b(hold|skip|ignore|discard|not now)\b", text):
        return "hold"
    if re.search(r"\b(source|sources|link|links|show source|show me source)\b", text):
        return "source"
    return "draft"


def _infer_package_kind(reply_text: str, targets: list[ActionRecord]) -> str:
    text = f" {reply_text.lower()} "
    if re.search(r"\b(article|longform|essay)\b", text):
        return "longform_article"
    if re.search(r"\b(thread|x thread|twitter thread)\b", text):
        return "x_thread"
    if len(targets) > 1:
        return "multi_source_article_thread"
    routes = {str((target.executor_state or {}).get("content_route") or "") for target in targets}
    if "longform_article" in routes:
        return "longform_article"
    if "linkedin_or_x_post" in routes:
        return "linkedin_or_x_post"
    return "gtm_content_package"


def _package_title(targets: list[ActionRecord]) -> str:
    titles = [_title_for(target) for target in targets if _title_for(target)]
    if not titles:
        return "GTM radar package"
    if len(titles) == 1:
        return titles[0]
    head = " + ".join(titles[:2])
    if len(titles) > 2:
        head += f" + {len(titles) - 2} more"
    return _clean_line(head, 160)


def _package_evidence_ids(targets: list[ActionRecord]) -> list[str]:
    values: list[str] = []
    seen = set()
    for target in targets:
        for value in [target.handle, *target.evidence_ids]:
            text = str(value or "").strip()
            if text and text not in seen:
                values.append(text)
                seen.add(text)
    return values


def _source_entry(target: ActionRecord) -> dict[str, Any]:
    state = target.executor_state or {}
    return {
        "handle": target.handle,
        "title": state.get("title") or target.summary,
        "pillar": state.get("pillar"),
        "content_route": state.get("content_route"),
        "summary": state.get("summary"),
        "why_it_matters": state.get("why_it_matters"),
        "thesis": state.get("thesis"),
        "angle": state.get("angle"),
        "source_url": state.get("source_url"),
        "image_direction": state.get("image_direction"),
        "evidence_ids": target.evidence_ids,
    }


def _package_thesis(sources: list[dict[str, Any]]) -> str:
    explicit = [_clean_line(source.get("thesis"), 220) for source in sources if source.get("thesis")]
    if explicit:
        return " ".join(explicit[:2])
    titles = [_clean_line(source.get("title"), 120) for source in sources if source.get("title")]
    if len(titles) >= 2:
        return f"Connect {titles[0]} and {titles[1]} into one operator lesson about AI, security, and control."
    if titles:
        return f"Turn {titles[0]} into a practical operator lesson."
    return "Turn the selected GTM signal into a practical operator lesson."


def _content_moves(package_kind: str) -> list[str]:
    if package_kind in {"longform_article", "multi_source_article_thread"}:
        return ["article", "linkedin_post", "x_thread", "visual_direction"]
    if package_kind == "x_thread":
        return ["x_thread", "linkedin_post", "visual_direction"]
    return ["linkedin_post", "x_post", "visual_direction"]


def _outline(sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    outline = [
        {
            "section": "hook",
            "point": "Open with the operator problem, not the paper title or tool announcement.",
        }
    ]
    for index, source in enumerate(sources, start=1):
        title = _clean_line(source.get("title"), 140) or f"source {index}"
        summary = _clean_line(source.get("summary"), 260)
        point = summary or "Explain the signal and why it changes execution, security, or AI engineering."
        outline.append({"section": f"source_{index}", "point": f"{title}: {point}"})
    outline.append(
        {
            "section": "operator_takeaway",
            "point": "Make the practical move explicit: what builders or security leaders should change this week.",
        }
    )
    return outline


def _visual_direction(sources: list[dict[str, Any]]) -> str:
    visuals = [_clean_line(source.get("image_direction"), 140) for source in sources if source.get("image_direction")]
    if visuals:
        return " / ".join(visuals[:2])
    return "clean control-plane or workflow diagram that ties the sources into one execution loop"


def _visual_plan(*, sources: list[dict[str, Any]], working_title: str) -> dict[str, Any]:
    summary = _visual_direction(sources)
    source_titles = [_clean_line(source.get("title"), 80) for source in sources if source.get("title")]
    components = [
        "source signal",
        "agent or AI system behavior",
        "control point",
        "operator decision",
    ]
    if len(source_titles) >= 2:
        components.insert(1, "shared pattern across sources")
    return {
        "status": "approval_required",
        "summary": summary,
        "image_prompt": (
            f"Create a clean editorial diagram for '{working_title}'. Show {', '.join(components)}. "
            "Use a crisp security/operator aesthetic, readable labels, and no fake logos."
        ),
        "alt_text": f"Diagram showing the operator control loop behind {working_title}.",
        "components": components,
    }


def _draft_assets(
    *,
    working_title: str,
    thesis: str,
    sources: list[dict[str, Any]],
    package_kind: str,
    visual_plan: dict[str, Any],
    optimization_lens: dict[str, Any],
) -> dict[str, Any]:
    hook = _draft_hook(sources)
    operator_move = _operator_move(sources)
    source_links = _source_links(sources)
    algorithm_note = _algorithm_note(optimization_lens)
    article = {
        "status": "approval_required",
        "title": working_title,
        "dek": thesis,
        "hook": hook,
        "sections": _article_sections(
            sources=sources,
            thesis=thesis,
            operator_move=operator_move,
            algorithm_note=algorithm_note,
        ),
        "close": (
            "The teams that win here will not be the ones with the most theatrical demos. "
            "They will be the ones that can see the loop, score the loop, and shut it down when it goes sideways."
        ),
        "source_links": source_links,
    }
    linkedin_body, linkedin_opener = _linkedin_post(
        title=working_title,
        thesis=thesis,
        sources=sources,
        hook=hook,
        operator_move=operator_move,
        source_links=source_links,
        algorithm_note=algorithm_note,
    )
    x_posts = _x_thread_posts(
        thesis=thesis,
        sources=sources,
        hook=hook,
        operator_move=operator_move,
        source_links=source_links,
        algorithm_note=algorithm_note,
    )
    return {
        "article": article,
        "linkedin_post": {
            "status": "approval_required",
            "opener": linkedin_opener,
            "body": linkedin_body,
            "source_links": source_links,
        },
        "x_thread": {
            "status": "approval_required",
            "posts": x_posts,
            "source_links": source_links,
        },
        "x_single_post": {
            "status": "approval_required",
            "body": _x_single_post(
                thesis=thesis,
                sources=sources,
                operator_move=operator_move,
                algorithm_note=algorithm_note,
            ),
            "source_links": source_links,
        },
        "visual_plan": visual_plan,
        "package_kind": package_kind,
    }


def _draft_hook(sources: list[dict[str, Any]]) -> str:
    pillars = {str(source.get("pillar") or "") for source in sources}
    if "security_ai" in pillars:
        return (
            "The useful AI security conversation is moving past prompt hygiene. "
            "The real fight is whether you can control the system once it starts acting."
        )
    if "ai_engineering_leverage" in pillars:
        return (
            "The next AI productivity jump will not come from vibes. "
            "It will come from instrumented loops that can be measured, corrected, and reused."
        )
    if "security_trends" in pillars:
        return (
            "Security keeps relearning the same lesson: the fragile part is usually the workflow, "
            "not the headline exploit."
        )
    return (
        "The signal is not the paper or the headline by itself. "
        "The signal is what it changes about how operators should build, secure, or decide."
    )


def _operator_move(sources: list[dict[str, Any]]) -> str:
    pillars = {str(source.get("pillar") or "") for source in sources}
    if "security_ai" in pillars:
        return (
            "put provenance, observability, rollback, and policy checks around retrieval, memory, "
            "tool use, and agent-to-agent handoffs"
        )
    if "ai_engineering_leverage" in pillars:
        return "score every AI loop by output quality, latency, cost, recovery behavior, and human-review burden"
    if "security_trends" in pillars:
        return "turn the trend into a concrete control, detection, hardening task, or tabletop scenario this week"
    return "turn the signal into one concrete execution change instead of another saved link"


def _article_sections(
    *,
    sources: list[dict[str, Any]],
    thesis: str,
    operator_move: str,
    algorithm_note: str,
) -> list[dict[str, str]]:
    sections = [
        {
            "heading": "The pattern",
            "draft": (
                f"{thesis} The important part is not that another paper exists. "
                "The important part is that the same operator problem keeps showing up: the system is moving faster "
                "than the controls around it."
            ),
        }
    ]
    for index, source in enumerate(sources, start=1):
        title = _clean_line(source.get("title"), 120) or f"Source {index}"
        summary = _clean_line(source.get("summary"), 360)
        why = _clean_line(source.get("why_it_matters"), 260)
        draft = summary or "This source adds another concrete signal to the operating pattern."
        if why:
            draft = f"{draft} {why}"
        sections.append({"heading": title, "draft": draft})
    sections.append(
        {
            "heading": "The operator move",
            "draft": (
                f"The move is simple and non-glamorous: {operator_move}. "
                "That is how this becomes useful instead of another AI take wearing a lab coat."
            ),
        }
    )
    sections.append(
        {
            "heading": "The distribution test",
            "draft": algorithm_note,
        }
    )
    return sections


def _linkedin_post(
    *,
    title: str,
    thesis: str,
    sources: list[dict[str, Any]],
    hook: str,
    operator_move: str,
    source_links: list[dict[str, str]],
    algorithm_note: str,
) -> tuple[str, str]:
    opener = hook
    source_lines = []
    for source in sources[:3]:
        title_line = _clean_line(source.get("title"), 100)
        summary = _clean_line(source.get("summary"), 220)
        if title_line and summary:
            source_lines.append(f"- {title_line}: {summary}")
        elif title_line:
            source_lines.append(f"- {title_line}")
    link_lines = [f"- {item['title']}: {item['url']}" for item in source_links[:3]]
    body_parts = [
        opener,
        "",
        thesis,
        "",
        "The useful read:",
        *source_lines,
        "",
        f"The operator move: {operator_move}.",
        "",
        (
            "This is the part most AI content skips. You do not get leverage because the model sounds confident. "
            "You get leverage when the loop can be observed, scored, constrained, and improved."
        ),
        "",
        algorithm_note,
    ]
    if link_lines:
        body_parts.extend(["", "Sources:", *link_lines])
    body_parts.extend(["", f"Working title: {title}"])
    return "\n".join(body_parts).strip(), opener


def _x_thread_posts(
    *,
    thesis: str,
    sources: list[dict[str, Any]],
    hook: str,
    operator_move: str,
    source_links: list[dict[str, str]],
    algorithm_note: str,
) -> list[str]:
    posts = [
        _clean_line(f"{hook} {thesis}", 270),
    ]
    for source in sources[:3]:
        title = _clean_line(source.get("title"), 90)
        summary = _clean_line(source.get("summary"), 170)
        if title and summary:
            posts.append(_clean_line(f"{title}: {summary}", 270))
        elif title:
            posts.append(_clean_line(title, 270))
    posts.extend(
        [
            _clean_line(f"The pattern: agent value is becoming an operating-system problem, not a demo problem.", 270),
            _clean_line(f"The move: {operator_move}.", 270),
            _clean_line(algorithm_note, 270),
        ]
    )
    if source_links:
        link = source_links[0]
        posts.append(_clean_line(f"Source to start with: {link['title']} {link['url']}", 270))
    return posts


def _x_single_post(
    *,
    thesis: str,
    sources: list[dict[str, Any]],
    operator_move: str,
    algorithm_note: str,
) -> str:
    source_title = _clean_line(sources[0].get("title"), 90) if sources else "the latest signal"
    return _clean_line(
        (
            f"{source_title} points at the useful AI/security takeaway: {thesis} "
            f"The move is to {operator_move}. {algorithm_note}"
        ),
        280,
    )


def _algorithm_note(optimization_lens: dict[str, Any]) -> str:
    negatives = optimization_lens.get("negative_signals") if isinstance(optimization_lens, dict) else []
    negative_text = ", ".join(str(value) for value in negatives[:4]) if isinstance(negatives, list) else ""
    return (
        "Distribution test: use the public X algorithm lens to optimize for "
        "reply, repost/quote, profile_click, dwell, and follow_author "
        f"while avoiding {negative_text or 'not_interested, block_author, mute_author, and report'}."
    )


def _source_links(sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    links = []
    seen = set()
    for source in sources:
        url = str(source.get("source_url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        links.append(
            {
                "title": _clean_line(source.get("title"), 120) or url,
                "url": url,
            }
        )
    return links


def _approval_actions(handle: str, content_moves: list[str]) -> list[dict[str, str]]:
    actions = [
        {
            "reply": f"revise {handle}: <change>",
            "effect": "update the draft package without public posting",
        },
        {
            "reply": f"hold {handle}",
            "effect": "discard the package from the active review queue",
        },
    ]
    if "article" in content_moves:
        actions.insert(
            0,
            {
                "reply": f"approve article {handle}",
                "effect": "queue the article draft for publishing workflow approval",
            },
        )
    if "linkedin_post" in content_moves:
        actions.insert(
            1,
            {
                "reply": f"approve linkedin {handle}",
                "effect": "queue the LinkedIn draft for publishing workflow approval",
            },
        )
    if "x_thread" in content_moves:
        actions.insert(
            2,
            {
                "reply": f"approve x thread {handle}",
                "effect": "queue the X thread draft for publishing workflow approval",
            },
        )
    if "x_post" in content_moves:
        actions.insert(
            2,
            {
                "reply": f"approve x post {handle}",
                "effect": "queue the X post draft for publishing workflow approval",
            },
        )
    return actions


def _title_for(record: ActionRecord) -> str:
    state = record.executor_state or {}
    return _clean_line(state.get("title") or record.user_visible_summary or record.summary, 140)


def _clean_line(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit].rstrip()
    boundary = max(clipped.rfind(". "), clipped.rfind("; "), clipped.rfind(", "), clipped.rfind(" "))
    if boundary > limit // 2:
        clipped = clipped[:boundary].rstrip()
    return clipped.rstrip(".,;:") + "..."


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
