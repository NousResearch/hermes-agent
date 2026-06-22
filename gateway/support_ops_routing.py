"""Support Ops teammate routing guards for Discord sends.

This module is intentionally small and deterministic: it does not call Discord,
read secrets, or mutate config.  It only resolves well-known internal Support
Ops routing phrases to exact Discord mentions and blocks unresolved placeholder
handles before they can be sent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

UNKNOWN_USER_RE = re.compile(r"(?<![\w-])@unknown-user\b", re.IGNORECASE)

KOZHUHAROV_MENTION = "<@1504852485083496561>"
BACKEND_MENTION = "<@1504852408227069993>"


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


def resolve_teammate_route(text: str) -> Optional[TeammateRoute]:
    """Resolve approved Support Ops teammate lane from names + topic context.

    Exact routing requires BOTH a teammate/lane alias and domain keywords so a
    casual mention of a colleague does not get auto-pinged.
    """

    if not text:
        return None

    kozhuharov_alias = _has_any(
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
            reason="Кожухаров/PBX-SIP-network context resolved to approved DevOps lane",
        )

    backend_alias = _has_any(
        text,
        (
            "алекс",
            "ивчо",
            "alex",
            "ivcho",
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
            reason="Алекс/Ивчо voucher-reservation/backend context resolved to approved backend lane",
        )

    return None


def lint_and_resolve_discord_content(content: str) -> MentionLintResult:
    """Resolve known internal routes and fail-closed on unresolved placeholders.

    ``@unknown-user`` is never sendable.  If the surrounding message contains
    enough Support Ops context for an approved route, replace the placeholder
    with the exact Discord mention.  Otherwise block the send with a public-safe
    reason.
    """

    text = str(content or "")
    route = resolve_teammate_route(text)
    if route and UNKNOWN_USER_RE.search(text):
        text = UNKNOWN_USER_RE.sub(route.mention, text)

    if UNKNOWN_USER_RE.search(text):
        return MentionLintResult(
            ok=False,
            content=text,
            route=route,
            blocked_reason="blocked_unresolved_unknown_user_placeholder",
        )

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
