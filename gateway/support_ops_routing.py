"""Support Ops teammate routing guards for Discord sends.

This module is intentionally small and deterministic: it does not call Discord,
read secrets, create canonical events, or decide business meaning.  It only
performs pre-send safety on already-authored Support Ops output:

* block unsafe/wrong teammate mentions;
* normalize well-known display handles;
* enforce exact known Discord mentions when a route is already explicit.

It must not become a keyword-authority router.  Hermes/LLM reasoning remains
responsible for deciding operational meaning and Canonical Brain events.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

UNKNOWN_USER_RE = re.compile(r"(?<![\w-])@unknown-user\b", re.IGNORECASE)
BACKEND_TEXT_MENTION_RE = re.compile(
    r"(?<![\w<@-])@(алекс|ивчо|иво|alex|ivcho|ivo|ivo\s+popov)\b",
    re.IGNORECASE,
)
KOZHUHAROV_TEXT_MENTION_RE = re.compile(
    r"(?<![\w<@-])@(кожухаров|емо\s+к|emo\s+k|kozhuharov)\b",
    re.IGNORECASE,
)
FATIH_TEXT_MENTION_RE = re.compile(r"(?<![\w<@-])@(фатих|fatih)\b", re.IGNORECASE)
ANY_EXACT_MENTION_RE = re.compile(r"<@!?\d+>")
RAW_QUOTE_RE = re.compile(r"[\"“”'„‚`].{0,80}(?:пламена|plamena).{0,80}[\"“”'„‚`]", re.IGNORECASE | re.DOTALL)

EMIL_OWNER_MENTION = "<@1279454038731264061>"
KOZHUHAROV_MENTION = "<@1282729392883372174>"
ALEX_MENTION = "<@1282940511962791959>"
IVCHO_MENTION = "<@1283039346295050271>"
FATIH_MENTION = "<@779368140512821268>"
PLAMENA_MENTION = "<@1282940574533423125>"
BACKEND_MENTION = f"{ALEX_MENTION} {IVCHO_MENTION}"
KNOWN_TEAMMATE_MENTIONS = {KOZHUHAROV_MENTION, ALEX_MENTION, IVCHO_MENTION, FATIH_MENTION, PLAMENA_MENTION}


@dataclass(frozen=True)
class TeammateRoute:
    lane: str
    mention: str
    reason: str


@dataclass(frozen=True)
class MentionLintResult:
    ok: bool
    content: str
    route: Optional[TeammateRoute] = None
    blocked_reason: Optional[str] = None


def _lower(text: str) -> str:
    return (text or "").casefold()


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = _lower(text)
    return any(needle.casefold() in lowered for needle in needles)


def _exact_mentions(text: str) -> set[str]:
    return set(ANY_EXACT_MENTION_RE.findall(text or ""))


def _contains_dm_request(text: str) -> bool:
    return _has_any(text, ("dm ", "дм ", "direct message", "директно съобщение", "на лично", "лично съобщение"))


def _replace_display_handles(text: str) -> tuple[str, Optional[str]]:
    """Normalize canonical display handles only when not an ambiguous raw quote."""
    if RAW_QUOTE_RE.search(text):
        return text, "blocked_plamena_raw_quote_ambiguity"
    text = re.sub(r"\bПламена\b", "Пламенка", text)
    text = re.sub(r"\bPlamena\b", "Пламенка", text, flags=re.IGNORECASE)
    return text, None


def resolve_teammate_route(text: str) -> Optional[TeammateRoute]:
    """Resolve approved Support Ops teammate lane from explicit name + topic.

    This resolution is mention-safety only.  It requires an already explicit
    colleague/lane alias plus narrow domain context; it does not create tasks,
    route-backs, or Canonical Brain state.
    """

    if not text:
        return None

    kozhuharov_alias = KOZHUHAROV_MENTION in (text or "") or _has_any(
        text,
        (
            "кожухаров",
            "емо к",
            "емо кожухаров",
            "emo k",
            "emo kozhuharov",
            "kozhuharov",
        ),
    )
    pbx_context = _has_any(
        text,
        (
            "pbx",
            "sip",
            "централ",
            "ip",
            "firewall",
            "network",
            "мреж",
            "37.63.76.203",
        ),
    )
    if kozhuharov_alias and pbx_context:
        return TeammateRoute(
            lane="devops_kozhuharov",
            mention=KOZHUHAROV_MENTION,
            reason="Кожухаров/PBX-SIP-network context resolved to approved DevOps mention-safety lane",
        )

    backend_alias = _has_any(
        text,
        (
            "алекс",
            "ивчо",
            "иво",
            "alex",
            "ivcho",
            "ivo",
        ),
    )
    backend_context = _has_any(
        text,
        (
            "voucher",
            "ваучер",
            "reservation",
            "резервац",
            "booking",
            "backend",
            "api",
            "automation",
            "автоматич",
            "vs941215",
        ),
    )
    if backend_alias and backend_context:
        return TeammateRoute(
            lane="backend_alex_ivcho",
            mention=BACKEND_MENTION,
            reason="Алекс/Ивчо voucher-reservation/backend context resolved to approved backend mention-safety lane",
        )

    fatih_alias = _has_any(text, ("фатих", "fatih"))
    frontend_context = _has_any(text, ("frontend", "front-end", "fe", "фронтенд", "ui", "визуал", "чат бутон", "fab"))
    if fatih_alias and frontend_context:
        return TeammateRoute(
            lane="frontend_fatih",
            mention=FATIH_MENTION,
            reason="Фатих/frontend context resolved to approved frontend mention-safety lane",
        )

    return None


def _has_wrong_exact_mention(text: str, allowed: set[str]) -> bool:
    mentions = _exact_mentions(text)
    return any(m in KNOWN_TEAMMATE_MENTIONS and m not in allowed for m in mentions)


def lint_and_resolve_discord_content(content: str) -> MentionLintResult:
    """Resolve known internal mention safety and fail closed on unsafe sends."""

    text = str(content or "")
    text, handle_block = _replace_display_handles(text)
    if handle_block:
        return MentionLintResult(ok=False, content=text, blocked_reason=handle_block)

    route = resolve_teammate_route(text)

    # DM requests to colleagues are a higher-risk delivery intent.  This lint
    # layer cannot verify owner DM approval or obtain a delivery receipt, so it
    # fail-closes instead of pretending to send a DM.
    if route and _contains_dm_request(text):
        return MentionLintResult(ok=False, content=text, route=route, blocked_reason="blocked_dm_requires_exact_owner_approval")

    if route and route.lane == "devops_kozhuharov":
        if _has_wrong_exact_mention(text, {KOZHUHAROV_MENTION}) or BACKEND_TEXT_MENTION_RE.search(text) or FATIH_TEXT_MENTION_RE.search(text) or UNKNOWN_USER_RE.search(text) or KOZHUHAROV_TEXT_MENTION_RE.search(text):
            return MentionLintResult(ok=False, content=text, route=route, blocked_reason="blocked_kozhuharov_route_requires_exact_mention")
        if KOZHUHAROV_MENTION not in text:
            return MentionLintResult(ok=False, content=text, route=route, blocked_reason="blocked_kozhuharov_route_missing_exact_mention")

    if route and route.lane == "backend_alex_ivcho":
        if _has_wrong_exact_mention(text, {ALEX_MENTION, IVCHO_MENTION}) or UNKNOWN_USER_RE.search(text):
            return MentionLintResult(ok=False, content=text, route=route, blocked_reason="blocked_backend_route_wrong_or_unknown_mention")
        if BACKEND_TEXT_MENTION_RE.search(text):
            text = BACKEND_TEXT_MENTION_RE.sub(route.mention, text)
            text = re.sub(
                rf"{re.escape(route.mention)}(?:\s*(?:/|,|и|and)?\s*{re.escape(route.mention)})+",
                route.mention,
                text,
                flags=re.IGNORECASE,
            )
        elif ALEX_MENTION not in text and IVCHO_MENTION not in text:
            return MentionLintResult(ok=False, content=text, route=route, blocked_reason="blocked_plain_name_route_requires_explicit_text_at_mention")
        # Backend resolver context should use Ивчо as display handle.
        text = re.sub(r"\bИво\b", "Ивчо", text)

    if route and route.lane == "frontend_fatih":
        if _has_wrong_exact_mention(text, {FATIH_MENTION}) or UNKNOWN_USER_RE.search(text):
            return MentionLintResult(ok=False, content=text, route=route, blocked_reason="blocked_frontend_route_wrong_or_unknown_mention")
        if FATIH_TEXT_MENTION_RE.search(text):
            text = FATIH_TEXT_MENTION_RE.sub(route.mention, text)
        elif FATIH_MENTION not in text:
            return MentionLintResult(ok=False, content=text, route=route, blocked_reason="blocked_plain_name_route_requires_explicit_text_at_mention")

    if UNKNOWN_USER_RE.search(text):
        return MentionLintResult(ok=False, content=text, route=route, blocked_reason="blocked_unresolved_unknown_user_placeholder")

    if BACKEND_TEXT_MENTION_RE.search(text) or KOZHUHAROV_TEXT_MENTION_RE.search(text) or FATIH_TEXT_MENTION_RE.search(text):
        return MentionLintResult(ok=False, content=text, route=route, blocked_reason="blocked_unresolved_text_teammate_mention")

    return MentionLintResult(ok=True, content=text, route=route)


def classify_support_ops_case_signal(text: str) -> Optional[str]:
    """Return a compact internal case signal for public-safe audit payloads."""

    lowered = _lower(text)
    closure_markers = (
        "централата вече работи",
        "случаят е готов",
        "case is ready",
        "case closed",
        "resolved",
    )
    if any(marker in lowered for marker in closure_markers):
        return "case_closure"
    return None
