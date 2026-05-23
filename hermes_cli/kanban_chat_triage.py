"""Kanban-side chat triage routing helpers.

This module is intentionally gateway-agnostic: platform adapters/classifiers can
hand it normalized message metadata plus a deterministic classification, and it
will resolve/create the right board, create exactly one triage task for the
source message, and return the identifiers the gateway needs to acknowledge the
queueing action.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from hermes_cli import kanban_db as kb

_ROUTING_VERSION = 1
_MAX_TITLE = 80
_GENERIC_PROJECT_NAMES = {
    "bug",
    "general",
    "misc",
    "stuff",
    "task",
    "tasks",
    "thing",
    "things",
    "todo",
    "work",
}
_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|password|secret|oauth[_-]?code)\s*[:=]\s*\S+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
)


class ChatTriageError(RuntimeError):
    """Actionable failure while resolving/creating a chat triage task."""


@dataclass(frozen=True)
class BoardDecision:
    board_slug: str
    source: str
    created_board: bool = False
    fallback_used: bool = False
    reason: str = ""
    unresolved_hint: Optional[str] = None


def route_chat_triage_task(
    *,
    message_text: str,
    classification: Mapping[str, Any],
    source: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    context: Optional[Mapping[str, Any]] = None,
    board_override: Optional[str] = None,
    created_by: str = "chat-triage-router",
) -> dict[str, Any]:
    """Create or reuse a board and park the gateway message in triage.

    Returns a stable dict containing ``board_id`` and ``task_id`` for gateway
    acknowledgements. Duplicate deliveries of the same platform/chat/thread/
    message id reuse the original task by passing an idempotency key through to
    ``kanban_db.create_task``.
    """

    if not str(message_text or "").strip():
        raise ChatTriageError("cannot create triage task: message_text is required")
    cfg = dict(config or {})
    src = {str(k): v for k, v in dict(source or {}).items() if v is not None}
    cls = dict(classification or {})

    decision = _resolve_or_create_board(
        message_text=message_text,
        classification=cls,
        source=src,
        config=cfg,
        board_override=board_override,
    )
    assignee = _select_assignee(cls, cfg)
    title = _derive_title(message_text, cls)
    priority = _derive_priority(message_text, cls)
    body = _build_task_body(
        message_text=message_text,
        classification=cls,
        source=src,
        context=dict(context or {}),
        board_decision=decision,
    )
    idem = _idempotency_key(src, message_text)

    try:
        with kb.connect(board=decision.board_slug) as conn:
            task_id = kb.create_task(
                conn,
                title=title,
                body=body,
                assignee=assignee,
                created_by=created_by,
                priority=priority,
                triage=True,
                idempotency_key=idem,
            )
            routing_metadata = _build_routing_metadata(
                source=src,
                message_text=message_text,
                classification=cls,
                board_decision=decision,
                task_id=task_id,
                assignee=assignee,
                priority=priority,
            )
            _record_chat_routing_event(conn, task_id, routing_metadata)
    except Exception as exc:  # pragma: no cover - exact sqlite details are platform-dependent
        if isinstance(exc, ChatTriageError):
            raise
        raise ChatTriageError(
            f"failed to create chat triage task on board {decision.board_slug!r}: {exc}"
        ) from exc

    ack = _format_ack(cfg, decision, task_id)
    return {
        "board_id": decision.board_slug,
        "board_slug": decision.board_slug,
        "task_id": task_id,
        "created_board": decision.created_board,
        "fallback_used": decision.fallback_used,
        "ack": ack,
        "chat_routing": routing_metadata,
    }


def _resolve_or_create_board(
    *,
    message_text: str,
    classification: Mapping[str, Any],
    source: Mapping[str, Any],
    config: Mapping[str, Any],
    board_override: Optional[str],
) -> BoardDecision:
    fallback = _safe_new_board_slug(config.get("fallback_board") or kb.DEFAULT_BOARD) or kb.DEFAULT_BOARD
    candidate, decision_source, reason, explicit = _select_board_candidate(
        message_text=message_text,
        classification=classification,
        config=config,
        board_override=board_override,
    )
    candidate = candidate or fallback

    try:
        candidate_slug = _safe_new_board_slug(candidate)
    except ValueError as exc:
        raise ChatTriageError(f"invalid selected board {candidate!r}: {exc}") from exc
    if not candidate_slug:
        if candidate and str(candidate).strip() and str(candidate).strip().casefold() != fallback:
            if not kb.board_exists(fallback):
                if not bool(config.get("create_missing_boards", True)):
                    raise ChatTriageError(
                        f"fallback board {fallback!r} does not exist and create_missing_boards is false; "
                        "create the fallback board or enable create_missing_boards"
                    )
                _create_board(fallback, classification={"category": "general"}, reason="missing fallback")
                fallback_created = True
            else:
                fallback_created = False
            return BoardDecision(
                fallback,
                "fallback",
                created_board=fallback_created,
                fallback_used=True,
                reason=f"selected board hint {candidate!r} was generic or invalid",
                unresolved_hint=str(candidate),
            )
        candidate_slug = fallback

    if kb.board_exists(candidate_slug):
        return BoardDecision(candidate_slug, decision_source, reason=reason)

    if _may_create_board(
        slug=candidate_slug,
        explicit=explicit,
        classification=classification,
        config=config,
        source=decision_source,
    ):
        _create_board(candidate_slug, classification=classification, reason=reason)
        return BoardDecision(
            candidate_slug,
            decision_source,
            created_board=True,
            reason=f"created missing board {candidate_slug!r} ({reason})",
        )

    # Missing/uncertain selected board falls back, but do not silently create
    # the fallback if config explicitly disabled board creation.
    if not kb.board_exists(fallback):
        if not bool(config.get("create_missing_boards", True)):
            raise ChatTriageError(
                f"fallback board {fallback!r} does not exist and create_missing_boards is false; "
                "create the fallback board or enable create_missing_boards"
            )
        _create_board(fallback, classification={"category": "general"}, reason="missing fallback")
        fallback_created = True
    else:
        fallback_created = False

    return BoardDecision(
        fallback,
        "fallback",
        created_board=fallback_created,
        fallback_used=True,
        reason=f"selected board {candidate_slug!r} missing or not allowed by policy",
        unresolved_hint=candidate_slug,
    )


def _select_board_candidate(
    *,
    message_text: str,
    classification: Mapping[str, Any],
    config: Mapping[str, Any],
    board_override: Optional[str],
) -> tuple[Optional[str], str, str, bool]:
    if board_override:
        return str(board_override), "override", "explicit board override", True

    text = message_text or ""
    explicit = _explicit_board_or_project(text)
    if explicit:
        alias = _match_configured_board_alias(explicit.replace("-", " "), config)
        if alias:
            slug, matched = alias
            return slug, "alias_match", f"explicit hint matched configured alias {matched!r}", True
        return explicit, "explicit_project", f"explicit board/project hint {explicit!r}", True

    project_hint = str(classification.get("project_hint") or "").strip()
    explicit_hint = _explicit_board_or_project(project_hint)
    if explicit_hint:
        return explicit_hint, "explicit_project", f"explicit classifier project hint {explicit_hint!r}", True

    alias = _match_configured_board_alias(project_hint or text, config)
    if alias:
        slug, matched = alias
        return slug, "alias_match", f"matched configured alias {matched!r}", False

    if project_hint:
        return project_hint, "project_hint", f"classifier project_hint {project_hint!r}", False

    category = str(classification.get("category") or "general").strip() or "general"
    category_cfg = _mapping(config.get("categories")).get(category, {})
    if isinstance(category_cfg, Mapping) and category_cfg.get("board"):
        return str(category_cfg["board"]), "category_default", f"category {category!r} default board", False

    return str(config.get("fallback_board") or kb.DEFAULT_BOARD), "fallback", "global fallback board", False


def _explicit_board_or_project(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = (
        r"\bboard\s*[:=]\s*([A-Za-z0-9][A-Za-z0-9_-]*)",
        r"\bproject\s*[:=]\s*([^\n:;,.—–-]+)",
        r"\bfor\s+([A-Za-z0-9][A-Za-z0-9 _-]{1,63}?)(?:\s*[:—-]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            value = match.group(1).strip()
            slug = _safe_new_board_slug(value)
            if slug:
                return slug
    return None


def _match_configured_board_alias(text: str, config: Mapping[str, Any]) -> Optional[tuple[str, str]]:
    haystack = f" {str(text or '').casefold()} "
    boards = _mapping(config.get("boards"))
    matches: list[tuple[str, str]] = []
    for raw_slug, raw_meta in boards.items():
        try:
            slug = kb._normalize_board_slug(str(raw_slug))
        except ValueError:
            continue
        if not slug:
            continue
        aliases = [slug]
        if isinstance(raw_meta, Mapping):
            aliases.extend(str(a) for a in raw_meta.get("aliases") or [] if a)
        for alias in aliases:
            needle = f" {alias.casefold()} "
            if needle in haystack:
                matches.append((slug, alias))
                break
    if not matches:
        return None
    # Deterministic: prefer longest alias (most specific), then slug order.
    matches.sort(key=lambda item: (-len(item[1]), item[0]))
    return matches[0]


def _may_create_board(
    *,
    slug: str,
    explicit: bool,
    classification: Mapping[str, Any],
    config: Mapping[str, Any],
    source: str,
) -> bool:
    if not bool(config.get("create_missing_boards", True)):
        return False
    if _is_generic_project_slug(slug):
        return False
    policy = str(config.get("board_create_policy") or "explicit_project_only")
    if policy == "never":
        return False
    if policy == "always":
        return True
    confidence = _float(classification.get("confidence"), default=0.0)
    if policy == "confident_project":
        return confidence >= 0.85
    if policy == "explicit_project_only":
        return explicit or source == "override"
    return False


def _create_board(slug: str, *, classification: Mapping[str, Any], reason: str) -> None:
    category = str(classification.get("category") or "general")
    name = _display_name(slug)
    description = (
        "Auto-created by chat triage routing. Default Kanban statuses include "
        f"triage; initial category={category}; reason={reason}."
    )
    try:
        kb.create_board(slug, name=name, description=description)
    except Exception as exc:
        raise ChatTriageError(f"failed to create board {slug!r}: {exc}") from exc


def _select_assignee(classification: Mapping[str, Any], config: Mapping[str, Any]) -> Optional[str]:
    if classification.get("assignee"):
        return str(classification["assignee"])
    category = str(classification.get("category") or "general").strip() or "general"
    category_cfg = _mapping(config.get("categories")).get(category, {})
    if isinstance(category_cfg, Mapping) and category_cfg.get("assignee"):
        return str(category_cfg["assignee"])
    if config.get("triage_assignee"):
        return str(config["triage_assignee"])
    return None


def _derive_title(message_text: str, classification: Mapping[str, Any]) -> str:
    raw = str(classification.get("title") or classification.get("extracted_title") or "").strip()
    if not raw:
        raw = re.split(r"[.!?\n]", message_text.strip(), maxsplit=1)[0].strip()
    raw = re.sub(r"\s+", " ", raw)
    if len(raw) > _MAX_TITLE:
        raw = raw[: _MAX_TITLE - 1].rstrip() + "…"
    return raw or "Triage chat request"


def _derive_priority(message_text: str, classification: Mapping[str, Any]) -> int:
    urgency = str(classification.get("urgency") or "").casefold()
    text = message_text.casefold()
    if urgency in {"urgent", "high"} or any(w in text for w in ("urgent", "asap", "emergency")):
        return 10
    if urgency == "low" or any(w in text for w in ("someday", "backlog", "low priority")):
        return -5
    return 0


def _build_task_body(
    *,
    message_text: str,
    classification: Mapping[str, Any],
    source: Mapping[str, Any],
    context: Mapping[str, Any],
    board_decision: BoardDecision,
) -> str:
    safe_text = _redact(message_text.strip())
    lines = [
        "Triage this request, assign the right specialist, and refine acceptance criteria before implementation.",
        "",
        "## Original message",
        safe_text,
        "",
        "## Source",
        _json_block(source),
        "",
        "## Classification",
        _json_block(classification),
        "",
        "## Board decision",
        _json_block(board_decision.__dict__),
    ]
    acceptance = classification.get("acceptance_criteria") or classification.get("criteria")
    if acceptance:
        lines.extend(["", "## Extracted acceptance criteria", _format_listish(acceptance)])
    links = classification.get("links") or classification.get("link_urls")
    if links:
        lines.extend(["", "## Links", _format_listish(links)])
    attachments = classification.get("attachments")
    if attachments:
        lines.extend(["", "## Attachments", _format_listish(attachments)])
    if context:
        lines.extend(["", "## Context", _json_block(context)])
    return "\n".join(lines).rstrip() + "\n"


def _build_routing_metadata(
    *,
    source: Mapping[str, Any],
    message_text: str,
    classification: Mapping[str, Any],
    board_decision: BoardDecision,
    task_id: str,
    assignee: Optional[str],
    priority: int,
) -> dict[str, Any]:
    return {
        "routing_version": _ROUTING_VERSION,
        "source": dict(source),
        "message": {
            "text_sha256": hashlib.sha256(message_text.encode("utf-8")).hexdigest(),
            "text_excerpt": _redact(message_text.strip())[:280],
            "attachment_count": _attachment_count(classification),
            "link_urls": list(classification.get("link_urls") or classification.get("links") or []),
        },
        "classification": dict(classification),
        "board_decision": dict(board_decision.__dict__),
        "task_decision": {
            "task_id": task_id,
            "assignee": assignee,
            "priority": priority,
            "status": "triage",
        },
        "ack": {"sent": False, "format": None, "reply_message_id": None},
        "bypass": {"bypassed": False, "rule": None},
    }


def _record_chat_routing_event(conn, task_id: str, routing_metadata: Mapping[str, Any]) -> None:
    # Duplicate gateway deliveries reuse create_task(idempotency_key=...) and
    # should not spam duplicate routing events either.
    if any(event.kind == "chat_routing" for event in kb.list_events(conn, task_id)):
        return
    with kb.write_txn(conn):
        kb._append_event(conn, task_id, "chat_routing", {"chat_routing": dict(routing_metadata)})


def _format_ack(config: Mapping[str, Any], decision: BoardDecision, task_id: str) -> str:
    template = str(config.get("ack_template") or "Queued for triage: board={board_slug} task={task_id}")
    try:
        ack = template.format(board_slug=decision.board_slug, board_id=decision.board_slug, task_id=task_id)
    except Exception:
        ack = f"Queued for triage: board={decision.board_slug} task={task_id}"
    if decision.created_board and "created" not in ack.lower():
        ack += " (created new board)"
    elif decision.fallback_used and decision.unresolved_hint:
        ack += f" (could not confidently match {decision.unresolved_hint})"
    return ack


def _idempotency_key(source: Mapping[str, Any], message_text: str) -> str:
    identity_parts = [
        source.get("platform"),
        source.get("chat_id") or source.get("channel_id"),
        source.get("thread_id"),
        source.get("message_id"),
    ]
    if any(identity_parts):
        raw = "|".join(str(p or "") for p in identity_parts)
    else:
        raw = "text|" + hashlib.sha256(message_text.encode("utf-8")).hexdigest()
    return "chat-triage:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_new_board_slug(value: Any) -> Optional[str]:
    s = str(value or "").strip().casefold()
    if not s:
        return None
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")[:64].strip("-")
    if not s or _is_generic_project_slug(s):
        return None
    return kb._normalize_board_slug(s)


def _is_generic_project_slug(slug: str) -> bool:
    return slug.casefold().strip("-_") in _GENERIC_PROJECT_NAMES


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _display_name(slug: str) -> str:
    return " ".join(part.capitalize() for part in slug.replace("_", "-").split("-") if part)


def _redact(text: str) -> str:
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(lambda m: f"{m.group(1) if m.groups() else 'secret'}=[REDACTED]", redacted)
    return redacted


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n```"


def _format_listish(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return "\n".join(f"- {item}" for item in value)
    return str(value)


def _attachment_count(classification: Mapping[str, Any]) -> int:
    attachments = classification.get("attachments")
    if isinstance(attachments, (list, tuple, set)):
        return len(attachments)
    try:
        return int(classification.get("attachment_count") or 0)
    except Exception:
        return 0


def _float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
