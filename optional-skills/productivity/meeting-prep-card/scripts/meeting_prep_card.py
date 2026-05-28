#!/usr/bin/env python3
"""Fixture-only, privacy-safe meeting context card renderer.

This helper is intentionally stdlib-only. It reads a synthetic JSON fixture,
selects relevant evidence for an external meeting, and renders public-safe
Markdown or JSON. It does not read live sources or perform mutations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from email.utils import parseaddr
from pathlib import Path
from typing import Any

GENERIC_DOMAINS = {"gmail.com", "outlook.com", "hotmail.com", "yahoo.com", "icloud.com", "me.com", "proton.me"}
DEFAULT_INTERNAL_DOMAINS = {"example.internal", "local.test"}
SOURCE_PRIORITY = {"crm": 40, "notes": 35, "email": 28, "chat": 24, "manual": 20, "calendar": 15}
OPEN_LOOP_KINDS = {"open_loop", "waiting_on", "decision", "risk", "doc"}
STOPWORDS = {"intro", "call", "sync", "meeting", "catchup", "followup", "follow", "up", "the", "and", "with", "advisory"}
ACTION_TERMS = ["i'll", "i will", "will send", "send", "confirm", "waiting", "follow up", "intro", "deck", "term sheet", "owe", "review"]

PHONE_RE = re.compile(r"(?<![A-Za-z])(?:\+\d{1,3}[\d\s\-()*.]{6,}\d|\b0\d{8,14}\b)")
JID_RE = re.compile(r"\b[\w.+-]+@(s\.whatsapp\.net|g\.us|lid)\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@(?!s\.whatsapp\.net\b|g\.us\b|lid\b)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE)
TOKEN_RE = re.compile(r"(?i)(access_token|api[_-]?key|signature|x-amz-signature|token)=([^\s&]+)")
OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
LONG_SECRET_RE = re.compile(r"\b(?=[A-Za-z0-9_\-]{40,}\b)(?=[A-Za-z0-9_\-]*\d)(?=[A-Za-z0-9_\-]*[A-Z])[A-Za-z0-9_\-]+\b")
SLACK_MENTION_RE = re.compile(r"<!?(?:here|channel|subteam\^[^>]+)>")
RISK_PATTERNS = [PHONE_RE, JID_RE, EMAIL_RE, URL_RE, BEARER_RE, TOKEN_RE, OPENAI_KEY_RE, LONG_SECRET_RE, SLACK_MENTION_RE]
SAFE_ENUM_RE = re.compile(r"[^a-z0-9_-]+")


def safe_enum(value: str | None, default: str = "unknown", limit: int = 40) -> str:
    cleaned = SAFE_ENUM_RE.sub("_", str(value or "").lower()).strip("_")[:limit]
    return cleaned or default


def sanitize_text(text: str | None, limit: int = 260) -> str:
    if not text:
        return ""
    value = str(text).replace("\n", " ").replace("\r", " ")
    value = re.sub(r"\s+", " ", value).strip()
    value = BEARER_RE.sub("[redacted-secret]", value)
    value = OPENAI_KEY_RE.sub("[redacted-secret]", value)
    value = TOKEN_RE.sub(lambda m: f"{m.group(1)}=[redacted-secret]", value)
    value = JID_RE.sub("[redacted-id]", value)
    value = EMAIL_RE.sub("[redacted-email]", value)
    value = URL_RE.sub("[redacted-link]", value)
    value = PHONE_RE.sub("[redacted-number]", value)
    value = SLACK_MENTION_RE.sub("[redacted-mention]", value)
    placeholders: dict[str, str] = {}
    for idx, marker in enumerate(["[redacted-secret]", "[redacted-id]", "[redacted-link]", "[redacted-number]", "[redacted-email]", "[redacted-mention]"]):
        key = f"__PLACEHOLDER_{idx}__"
        placeholders[key] = marker
        value = value.replace(marker, key)
    value = LONG_SECRET_RE.sub("[redacted-secret]", value)
    for key, marker in placeholders.items():
        value = value.replace(key, marker)
    if len(value) > limit:
        value = value[: max(0, limit - 1)].rstrip() + "…"
    return value


def contains_risky_text(text: str | None) -> bool:
    return bool(text and any(pattern.search(str(text)) for pattern in RISK_PATTERNS))


def parse_datetime(value: str, path: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception as exc:
        raise ValueError(f"invalid datetime at {path}: {sanitize_text(value, limit=80)}") from exc


def require(obj: dict[str, Any], key: str, path: str) -> Any:
    if key not in obj or obj[key] in (None, ""):
        raise ValueError(f"missing required field: {path}.{key}")
    return obj[key]


@dataclass(frozen=True)
class Attendee:
    name: str = ""
    email: str = ""

    @property
    def domain(self) -> str:
        if "@" not in self.email:
            return ""
        return self.email.rsplit("@", 1)[1].lower().strip()

    @property
    def safe_display(self) -> str:
        if self.name:
            return sanitize_text(self.name, limit=80)
        if self.email:
            return sanitize_text(self.email.split("@", 1)[0], limit=80)
        return "Unknown attendee"


@dataclass(frozen=True)
class MeetingEvent:
    id: str
    title: str
    start: datetime
    end: datetime | None = None
    attendees: tuple[Attendee, ...] = ()
    status: str = "confirmed"
    location: str = ""
    has_link: bool = False

    @property
    def domains(self) -> set[str]:
        return {a.domain for a in self.attendees if a.domain}

    @property
    def safe_title(self) -> str:
        return sanitize_text(self.title, limit=120)


@dataclass(frozen=True)
class SourcePointer:
    source: str
    source_id: str
    label: str
    timestamp: datetime
    confidence: str = "possible"

    @property
    def safe_label(self) -> str:
        return sanitize_text(f"{self.source}: {self.label}", limit=100)


@dataclass(frozen=True)
class EvidenceItem:
    pointer: SourcePointer
    kind: str
    text: str
    people: tuple[str, ...] = ()
    company: str | None = None
    score: float = 0.0
    match_reasons: tuple[str, ...] = ()
    freshness: str = "unknown"

    def with_score(self, score: float, reasons: list[str], freshness: str) -> "EvidenceItem":
        return replace(self, score=score, match_reasons=tuple(reasons), freshness=freshness)


@dataclass(frozen=True)
class PrepCard:
    event: MeetingEvent
    generated_at: datetime
    headline: str
    why_it_matters: str
    internal_domains: tuple[str, ...] = tuple(sorted(DEFAULT_INTERNAL_DOMAINS))
    last_touch: EvidenceItem | None = None
    open_loops: list[EvidenceItem] = field(default_factory=list)
    context_bullets: list[EvidenceItem] = field(default_factory=list)
    suggested_cta: str = "Confirm agenda and desired outcome."
    warnings: tuple[str, ...] = ()
    sources_used: dict[str, int] = field(default_factory=dict)


def parse_attendee(raw: dict[str, Any]) -> Attendee:
    return Attendee(name=sanitize_text(str(raw.get("name", "")), limit=100), email=str(raw.get("email", "")).lower().strip())


def parse_event(raw: dict[str, Any], idx: int) -> MeetingEvent:
    path = f"event[{idx}]"
    return MeetingEvent(
        id=str(require(raw, "id", path)),
        title=sanitize_text(str(require(raw, "title", path)), limit=160),
        start=parse_datetime(str(require(raw, "start", path)), f"{path}.start"),
        end=parse_datetime(str(raw["end"]), f"{path}.end") if raw.get("end") else None,
        attendees=tuple(parse_attendee(a) for a in raw.get("attendees", [])),
        status=str(raw.get("status", "confirmed")).lower(),
        location=sanitize_text(str(raw.get("location", "")), limit=80),
        has_link=bool(raw.get("url")),
    )


def parse_evidence(raw: dict[str, Any], idx: int) -> EvidenceItem:
    path = f"evidence[{idx}]"
    source = safe_enum(str(require(raw, "source", path)), default="manual")
    pointer = SourcePointer(
        source=source,
        source_id=str(raw.get("source_id", f"{source}_{idx}")),
        label=sanitize_text(str(raw.get("label", source.title())), limit=100),
        timestamp=parse_datetime(str(require(raw, "timestamp", path)), f"{path}.timestamp"),
        confidence=safe_enum(str(raw.get("confidence", "possible")), default="possible"),
    )
    return EvidenceItem(
        pointer=pointer,
        kind=safe_enum(str(raw.get("kind", "context")), default="context"),
        text=sanitize_text(str(raw.get("text", "")), limit=320),
        people=tuple(str(p).strip()[:160] for p in raw.get("people", []) if str(p).strip()),
        company=sanitize_text(str(raw["company"]), limit=120) if raw.get("company") else None,
    )


def parse_fixture(data: dict[str, Any]) -> tuple[list[MeetingEvent], list[EvidenceItem], datetime]:
    now = parse_datetime(str(require(data, "now", "fixture")), "fixture.now")
    events = [parse_event(raw, i) for i, raw in enumerate(data.get("events", []))]
    evidence = [parse_evidence(raw, i) for i, raw in enumerate(data.get("evidence", []))]
    return events, evidence, now


def is_external_event(event: MeetingEvent, internal_domains: set[str] | None = None) -> bool:
    internal = {d.lower() for d in (internal_domains or DEFAULT_INTERNAL_DOMAINS)}
    if event.status in {"declined", "cancelled", "canceled", "tentative"}:
        return False
    if any(term in event.title.lower() for term in ["focus", "hold", "blocked", "webinar"]):
        return False
    return bool(event.domains) and any(domain not in internal for domain in event.domains)


def domain(value: str) -> str:
    email = parseaddr(value)[1] or value
    return email.rsplit("@", 1)[1].lower().strip() if "@" in email else ""


def tokens(text: str | None) -> set[str]:
    return {tok for tok in re.findall(r"[A-Za-z0-9]{3,}", text.lower() if text else "") if tok not in STOPWORDS}


def freshness_for(ts: datetime, now: datetime) -> str:
    delta_days = abs((now - ts).days)
    if delta_days <= 7:
        return "fresh"
    if delta_days <= 45:
        return "stale"
    return "archival"


def score_evidence(event: MeetingEvent, item: EvidenceItem, now: datetime) -> tuple[float, list[str], str]:
    score = float(SOURCE_PRIORITY.get(item.pointer.source, 10))
    reasons: list[str] = []
    freshness = freshness_for(item.pointer.timestamp, now)
    score += {"fresh": 25, "stale": 10, "archival": -10}.get(freshness, 0)

    event_emails = {a.email.lower() for a in event.attendees if a.email}
    item_emails = {((parseaddr(p)[1] or p).lower().strip()) for p in item.people if "@" in p}
    if event_emails & item_emails:
        score += 65
        reasons.append("attendee_email")

    event_domains = {d for d in event.domains if d not in GENERIC_DOMAINS}
    item_domains = {domain(p) for p in item.people if domain(p) and domain(p) not in GENERIC_DOMAINS}
    if event_domains & item_domains:
        score += 35
        reasons.append("domain")

    event_tokens = tokens(event.title) | {tok for a in event.attendees for tok in tokens(a.name)}
    overlap = event_tokens & (tokens(item.company) | tokens(item.text))
    if overlap:
        score += min(25, len(overlap) * 8)
        reasons.append("token")
    if item.company and tokens(item.company) & event_tokens:
        score += 25
        reasons.append("company")

    identity_reasons = set(reasons)
    if item.kind in OPEN_LOOP_KINDS or any(term in item.text.lower() for term in ACTION_TERMS):
        score += 22
        if "open_loop" not in reasons:
            reasons.append("open_loop")
    if not identity_reasons:
        return 0.0, [], freshness
    return score, reasons, freshness


def match_evidence(event: MeetingEvent, evidence: list[EvidenceItem], now: datetime, limit: int = 5) -> list[EvidenceItem]:
    scored = []
    for item in evidence:
        score, reasons, freshness = score_evidence(event, item, now)
        if score >= 55 and reasons:
            scored.append(item.with_score(score, reasons, freshness))
    scored.sort(key=lambda item: (-item.score, -item.pointer.timestamp.timestamp(), item.pointer.source_id))
    return scored[:limit]


def first_of_kind(items: list[EvidenceItem], kinds: set[str]) -> EvidenceItem | None:
    return next((item for item in items if item.kind in kinds), None)


def cta_from(items: list[EvidenceItem]) -> str:
    combined = " ".join(item.text.lower() for item in items[:3])
    if "intro" in combined:
        return "Confirm whether the promised introduction should be sent and request the latest deck."
    if "deck" in combined:
        return "Confirm deck status and the specific decision needed from this meeting."
    if "waiting" in combined or "confirm" in combined:
        return "Confirm the pending ask, owner, and deadline before the call ends."
    return "Confirm agenda and desired outcome; do not assume missing context."


def build_prep_card(event: MeetingEvent, matched: list[EvidenceItem], now: datetime, internal_domains: set[str] | None = None) -> PrepCard:
    last_touch = first_of_kind(matched, {"last_touch"})
    open_loops = [item for item in matched if item.kind in OPEN_LOOP_KINDS][:3]
    context = [item for item in matched if item is not last_touch and item not in open_loops][:2]
    warnings: list[str] = []
    if not matched:
        warnings.append("No matching context found; use this as a cold-call prep card.")
    elif all(item.freshness != "fresh" for item in matched):
        warnings.append("Matched context is stale; verify before relying on it.")
    why = "Open loop or recent touch found." if open_loops else ("Recent booking/context found; main value is confirming agenda and filling gaps." if matched else "No reliable source context found yet.")
    return PrepCard(
        event=event,
        generated_at=now,
        headline=f"Prep: {event.safe_title}",
        why_it_matters=why,
        internal_domains=tuple(sorted(d.lower() for d in (internal_domains or DEFAULT_INTERNAL_DOMAINS))),
        last_touch=last_touch,
        open_loops=open_loops,
        context_bullets=context,
        suggested_cta=cta_from(matched),
        warnings=tuple(warnings),
        sources_used=dict(Counter(item.pointer.source for item in matched)),
    )


def build_cards_from_fixture(fixture: dict[str, Any], event_id: str | None = None, include_internal: bool = False, internal_domains: set[str] | None = None) -> list[PrepCard]:
    events, evidence, now = parse_fixture(fixture)
    selected = [event for event in events if event_id is None or event.id == event_id]
    if event_id and not selected:
        raise ValueError(f"event_id not found: {sanitize_text(event_id, limit=80)}")
    cards = []
    for event in selected:
        if not include_internal and not is_external_event(event, internal_domains):
            continue
        cards.append(build_prep_card(event, match_evidence(event, evidence, now), now=now, internal_domains=internal_domains))
    return cards


def source_tag(item: EvidenceItem) -> str:
    reasons = "/".join(item.match_reasons[:2]) or "match"
    return f"[{item.pointer.source}, {item.pointer.timestamp.date().isoformat()}, confidence: {item.pointer.confidence}, match: {reasons}, {item.freshness}]"


def line_for(item: EvidenceItem, prefix: str) -> str:
    return f"- {prefix}: {sanitize_text(item.text, limit=170)} {source_tag(item)}"


def attendees_for(card: PrepCard) -> str:
    externals = external_attendees(card)
    return ", ".join(externals[:4]) if externals else "external attendees"


def external_attendees(card: PrepCard) -> list[str]:
    internal = set(card.internal_domains)
    return [a.safe_display for a in card.event.attendees if a.domain and a.domain not in internal]


def render_markdown(card: PrepCard, max_chars: int = 1500) -> str:
    lines = [
        f"*{sanitize_text(card.headline, limit=120)}*",
        f"When: {card.event.start.strftime('%Y-%m-%d %H:%M %z')} · With: {attendees_for(card)}",
    ]
    if card.event.has_link:
        lines.append("Link: [meeting link available — hidden]")
    if card.warnings:
        lines.append("⚠️ " + sanitize_text(card.warnings[0], limit=160))
    lines.append(f"Why: {sanitize_text(card.why_it_matters, limit=140)}")
    lines.append(line_for(card.last_touch, "Last touch") if card.last_touch else "- Last touch: not found in fixture sources [confidence: possible]")
    if card.open_loops:
        for item in card.open_loops[:2]:
            lines.append(line_for(item, "Gap/risk" if item.kind == "risk" else "Open loop"))
    else:
        lines.append("- Open loop: none found; do not invent one [confidence: possible]")
    for item in card.context_bullets[:2]:
        lines.append(line_for(item, "Context"))
    lines.append(f"Suggested move: {sanitize_text(card.suggested_cta, limit=160)}")
    source_summary = ", ".join(f"{source} {count}" for source, count in sorted(card.sources_used.items())) or "none"
    lines.append(f"Sources: {source_summary}. No messages sent; no calendar/CRM changes.")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    if contains_risky_text(text):
        raise ValueError("rendered markdown contains risky unsanitized text")
    return text


def safe_pointer(item: EvidenceItem) -> dict[str, Any]:
    safe_hash = hashlib.sha256(f"{item.pointer.source}:{item.pointer.source_id}".encode()).hexdigest()[:12]
    return {"source": item.pointer.source, "safe_ref": safe_hash, "label": item.pointer.safe_label, "date": item.pointer.timestamp.date().isoformat(), "confidence": item.pointer.confidence, "match_reasons": list(item.match_reasons), "freshness": item.freshness}


def public_item(item: EvidenceItem) -> dict[str, Any]:
    return {"kind": item.kind, "text": sanitize_text(item.text, limit=220), "pointer": safe_pointer(item)}


def card_to_public_json(card: PrepCard) -> dict[str, Any]:
    safe_event_ref = hashlib.sha256(card.event.id.encode()).hexdigest()[:12]
    data = {
        "event": {"safe_ref": safe_event_ref, "title": sanitize_text(card.event.title, limit=120), "start": card.event.start.isoformat(), "attendees": external_attendees(card), "has_link": card.event.has_link},
        "generated_at": card.generated_at.isoformat(),
        "headline": sanitize_text(card.headline, limit=120),
        "why_it_matters": sanitize_text(card.why_it_matters, limit=160),
        "last_touch": public_item(card.last_touch) if card.last_touch else None,
        "open_loops": [public_item(item) for item in card.open_loops],
        "context_bullets": [public_item(item) for item in card.context_bullets],
        "suggested_cta": sanitize_text(card.suggested_cta, limit=180),
        "warnings": [sanitize_text(w, limit=180) for w in card.warnings],
        "sources_used": dict(card.sources_used),
        "mutation_status": "read-only: no messages sent; no external changes",
    }
    if contains_risky_text(json.dumps(data, ensure_ascii=False)):
        raise ValueError("public json contains risky unsanitized text")
    return data


def load_fixture(path: Path, now_override: str | None = None) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"failed to read fixture {sanitize_text(str(path), limit=140)}: {sanitize_text(str(exc), limit=160)}") from exc
    if now_override:
        parse_datetime(now_override, "--now")
        data["now"] = now_override
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render fixture-only meeting context cards")
    parser.add_argument("--fixture", required=True, help="Path to synthetic fixture JSON")
    parser.add_argument("--event-id", help="Specific event ID to render")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--include-internal", action="store_true", help="Include internal meetings")
    parser.add_argument("--max-chars", type=int, default=1500, help="Markdown character budget")
    parser.add_argument("--now", help="Override fixture now ISO datetime")
    parser.add_argument("--output", help="Optional local output path")
    parser.add_argument("--strict", action="store_true", help="Fail if output contains risky unsanitized patterns")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cards = build_cards_from_fixture(load_fixture(Path(args.fixture), args.now), event_id=args.event_id, include_internal=args.include_internal)
        if args.format == "json":
            output = json.dumps([card_to_public_json(card) for card in cards], ensure_ascii=False, indent=2)
        else:
            output = "\n\n---\n\n".join(render_markdown(card, max_chars=args.max_chars) for card in cards)
        if args.strict and contains_risky_text(output):
            raise ValueError("strict privacy scan failed")
        if args.output:
            Path(args.output).write_text(output)
        else:
            print(output)
        return 0
    except Exception as exc:
        print(sanitize_text(str(exc), limit=220), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
