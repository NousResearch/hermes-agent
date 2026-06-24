"""Support Ops mention lint guards for Discord sends.

This module is intentionally small and deterministic: it does not call Discord,
read secrets, create canonical events, or decide business meaning.  It only
performs pre-send linting on already-authored Support Ops output:

* block unresolved teammate text mentions and placeholders;
* normalize well-known display handles;

It must not infer a route/lane from business words such as PBX/SIP, voucher,
backend, or frontend.  Hermes/LLM reasoning and the explicit channel directory
remain responsible for deciding where a message should go.
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
RAW_QUOTE_RE = re.compile(r"[\"“”'„‚`].{0,80}(?:пламена|plamena).{0,80}[\"“”'„‚`]", re.IGNORECASE | re.DOTALL)

EMIL_OWNER_MENTION = "<@1279454038731264061>"
KOZHUHAROV_MENTION = "<@1282729392883372174>"
ALEX_MENTION = "<@1282940511962791959>"
IVCHO_MENTION = "<@1283039346295050271>"
FATIH_MENTION = "<@779368140512821268>"
PLAMENA_MENTION = "<@1282940574533423125>"
BACKEND_MENTION = f"{ALEX_MENTION} {IVCHO_MENTION}"


@dataclass(frozen=True)
class MentionLintResult:
    ok: bool
    content: str
    blocked_reason: Optional[str] = None


def _replace_display_handles(text: str) -> tuple[str, Optional[str]]:
    """Normalize canonical display handles only when not an ambiguous raw quote."""
    if RAW_QUOTE_RE.search(text):
        return text, "blocked_plamena_raw_quote_ambiguity"
    text = re.sub(r"\bПламена\b", "Пламенка", text)
    text = re.sub(r"\bPlamena\b", "Пламенка", text, flags=re.IGNORECASE)
    return text, None


def lint_and_resolve_discord_content(content: str) -> MentionLintResult:
    """Fail closed on unresolved teammate mentions without inferring routes."""

    text = str(content or "")
    text, handle_block = _replace_display_handles(text)
    if handle_block:
        return MentionLintResult(ok=False, content=text, blocked_reason=handle_block)

    if UNKNOWN_USER_RE.search(text):
        return MentionLintResult(ok=False, content=text, blocked_reason="blocked_unresolved_unknown_user_placeholder")

    if BACKEND_TEXT_MENTION_RE.search(text) or KOZHUHAROV_TEXT_MENTION_RE.search(text) or FATIH_TEXT_MENTION_RE.search(text):
        return MentionLintResult(ok=False, content=text, blocked_reason="blocked_unresolved_text_teammate_mention")

    return MentionLintResult(ok=True, content=text)
