"""/draft slash command — Atlas-aware draft (Plan 030-A/B).

This is the **first two phases** of Plan 030 ("Atlas-aware draft skill" —
R2 CP3). 030-A shipped the slash-command skeleton + recipient resolution.
030-B layers in the Atlas context fetch: before any LLM call, /draft
fires three parallel ``atlas_ask`` questions against Atlas to gather
prior-commitment, contradiction, and last-interaction context for the
recipient. The actual draft *composition* is still deferred to 030-C.

Usage::

    /draft sarah@example.com follow-up on the term sheet

The first whitespace-delimited token is the recipient. Two shapes are
supported in v1:

* **Email** — matches ``RFC5322`` lite (``local@domain.tld``). Returned
  in the reply as the resolved recipient and used as the lookup key
  for the future Atlas / Gmail context fetch (030-B).
* **Slack handle** — ``@somebody``. v1 surfaces it as-is in the reply;
  Slack ``users.lookupByEmail`` resolution lands in 030-C when the
  Slack client is wired through. For now we strip the leading ``@``
  and pass the bare handle through so 030-C can drop in the lookup
  without changing the public command surface.

The remaining tokens form the intent string. We preserve quoting
loosely by joining on a single space — Slack's slash-command transport
already strips outer quotes, but if Blake double-quotes the intent we
strip a single matched pair so the surface ``/draft a@b.com "context"``
behaves like ``/draft a@b.com context``.

Auth: the Hermes gateway's ``SLACK_ALLOWED_USERS`` gate is enforced at
the platform layer (gateway/platforms/slack.py) — by the time this
handler runs we already trust the caller. We do **not** re-check the
allowlist here.

The handler is sync (``fn(raw_args: str) -> str``) per the
``PluginContext.register_command`` contract that the Plan 020-E
``/resume`` and ``/skip`` handlers established.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recipient parsing
# ---------------------------------------------------------------------------

# Email regex is intentionally permissive — Slack already normalizes
# auto-linked emails, and overly strict patterns reject valid plus-tags
# and subdomains. Mirrors the shape used by hermes_storage email mapping.
_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)

# A Slack handle is ``@`` followed by 1+ word chars (letters, digits,
# underscore, dot, dash). We do **not** accept bare-word handles in v1 —
# the leading ``@`` is the disambiguator.
_HANDLE_RE = re.compile(r"^@([a-zA-Z0-9._\-]+)$")


@dataclass(frozen=True)
class Recipient:
    """Parsed recipient.

    ``kind`` is one of ``"email"``, ``"handle"``, ``"unresolved"``. The
    ``value`` is the raw token (with leading ``@`` stripped for handles
    so callers don't need to special-case it). ``display`` is what we
    surface in the Slack reply — for emails we synthesize a friendly
    first-name display, for handles we keep ``@handle``, for unresolved
    we echo the raw token so Blake sees exactly what he typed.
    """

    kind: str
    value: str
    display: str


def _friendly_name_from_email(email: str) -> str:
    """Derive a display name from the email local-part.

    ``sarah.connor+pe@example.com`` → ``Sarah``. Strips plus-tags and
    dot-separated segments after the first. Falls back to the raw
    local-part title-cased if no separator is present.
    """
    local = email.split("@", 1)[0]
    # Drop plus-tag (``user+tag`` → ``user``)
    local = local.split("+", 1)[0]
    # First dot-segment is conventionally the first name
    first = local.split(".", 1)[0]
    if not first:
        return email
    return first[:1].upper() + first[1:].lower()


def _parse_recipient(token: str) -> Recipient:
    """Classify the first arg as email / handle / unresolved."""
    if _EMAIL_RE.match(token):
        return Recipient(
            kind="email",
            value=token,
            display=_friendly_name_from_email(token),
        )
    handle_match = _HANDLE_RE.match(token)
    if handle_match:
        bare = handle_match.group(1)
        return Recipient(kind="handle", value=bare, display=f"@{bare}")
    return Recipient(kind="unresolved", value=token, display=token)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DraftArgs:
    """Parsed ``/draft`` invocation."""

    recipient: Recipient
    intent: str


def _strip_matched_quotes(s: str) -> str:
    """Strip a single matched pair of leading/trailing ASCII quotes."""
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def _parse_args(raw_args: str) -> Optional[DraftArgs]:
    """Split into (recipient_token, intent_string).

    Returns ``None`` when there's nothing to parse — the handler turns
    that into a usage reply.
    """
    if not raw_args or not raw_args.strip():
        return None
    parts = raw_args.strip().split(None, 1)
    recipient_token = parts[0]
    intent = parts[1].strip() if len(parts) > 1 else ""
    intent = _strip_matched_quotes(intent)
    recipient = _parse_recipient(recipient_token)
    return DraftArgs(recipient=recipient, intent=intent)


# ---------------------------------------------------------------------------
# Stub reply composition
# ---------------------------------------------------------------------------


def _usage() -> str:
    return (
        "Usage: /draft <recipient> <context>\n"
        'Example: /draft sarah@example.com "follow-up on the term sheet"\n'
        "Recipient may be an email (``user@host``) or Slack handle (``@somebody``)."
    )


def _compose_stub_reply(args: DraftArgs) -> str:
    """Render the 030-A stub message (header + intent line only).

    Phase 030-B layers Atlas context underneath this header via
    :func:`_compose_full_reply`. The original 030-A acceptance smoke
    tests assert the TODO markers are present in the *combined* reply,
    so we keep this helper focused on the header/intent prefix and let
    030-B's composer append the context blocks + TODO 030-C marker.
    """
    if args.recipient.kind == "email":
        header = (
            f"Drafting message to {args.recipient.display} "
            f"({args.recipient.value})"
        )
    elif args.recipient.kind == "handle":
        header = f"Drafting message to {args.recipient.display}"
    else:
        header = f"Drafting message to {args.recipient.display} (unresolved)"

    intent_line = (
        f"Intent: {args.intent}" if args.intent else "Intent: (none provided)"
    )
    return f"{header}\n{intent_line}"


# ---------------------------------------------------------------------------
# 030-B — Atlas context fetch (3 parallel asks)
# ---------------------------------------------------------------------------

# The three context questions Hermes asks Atlas before composing a draft.
# Each is a (label, template) pair. ``{recipient}`` is substituted with
# the recipient's display name (for emails: friendly first name; for
# handles: ``@handle``; for unresolved: raw token). The labels become
# the section headers Blake sees in Slack.
_CONTEXT_QUESTIONS: List[Tuple[str, str]] = [
    (
        "Prior commitments",
        "What outstanding commitments do I have with {recipient}?",
    ),
    (
        "Open contradictions",
        "Are there any open contradictions involving {recipient} I should be aware of?",
    ),
    (
        "Last interaction",
        "What was the last meaningful interaction I had with {recipient}, and what was its outcome?",
    ),
]

# Empty/no-fact responses each section falls back to when Atlas returns
# nothing meaningful. Keyed by section label so the reply still surfaces
# three sections even on a cold corpus (AC1).
_EMPTY_FALLBACKS: dict[str, str] = {
    "Prior commitments": "No commitments found.",
    "Open contradictions": "No contradictions found.",
    "Last interaction": "No prior interactions found.",
}


def _default_ask_factory():
    """Build a thread-safe ``ask(question) -> dict`` callable.

    Lazy-instantiates a single :class:`AtlasMemoryProvider` and returns
    its ``_ask`` bound method. Imported lazily so the slash module can
    still be imported in test contexts that don't have Atlas configured
    (AC: ``test_no_llm_imports_at_module_load`` still passes).
    """
    # Local import — keeps module load free of provider dependencies.
    from plugins.memory.atlas import AtlasMemoryProvider

    provider = AtlasMemoryProvider()
    provider.initialize(session_id="draft-slash")
    return provider._ask


def _extract_answer(payload: dict | str | None) -> str:
    """Pull a human-readable answer string out of an /v1/ask response.

    The Atlas ``AskResponse`` shape is ``{"answer": "...",
    "citations": [...]}``; we preserve any ``[cite:<chunk_id>]`` markers
    verbatim. If the payload is a string (already-rendered), return it.
    If it's empty/None, return the empty string so the caller can pick a
    section-specific fallback.
    """
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.strip()
    if not isinstance(payload, dict):
        return ""
    answer = payload.get("answer") or payload.get("result") or ""
    if isinstance(answer, str):
        return answer.strip()
    # Some Atlas envelopes wrap the answer dict — coerce to JSON so the
    # user at least sees the raw data rather than a bare ``{}``.
    try:
        return json.dumps(answer)
    except Exception:
        return str(answer)


def _is_empty_answer(answer: str) -> bool:
    """Heuristic for "Atlas has nothing on this".

    Atlas's /v1/ask synthesizer sometimes returns phrases like "I don't
    have information" or "no records" when the corpus is sparse. We
    don't try to be exhaustive — only the obvious zero-information
    phrases trigger the fallback. Real answers (even short ones) pass
    through verbatim so citations are preserved.
    """
    if not answer:
        return True
    lowered = answer.lower().strip()
    if len(lowered) < 3:
        return True
    empty_markers = (
        "no information",
        "i don't have",
        "i do not have",
        "no records",
        "no data",
        "not aware of",
        "nothing found",
    )
    return any(m in lowered for m in empty_markers)


def fetch_atlas_context(
    recipient_display: str,
    *,
    ask_fn: Optional[Callable[..., dict]] = None,
    max_workers: int = 3,
) -> List[Tuple[str, str]]:
    """Fire the three Atlas questions in parallel and collect answers.

    Returns a list of ``(label, answer_text)`` tuples in the same order
    as :data:`_CONTEXT_QUESTIONS`. Each entry is guaranteed non-empty —
    if Atlas returned nothing useful, the section-specific fallback from
    :data:`_EMPTY_FALLBACKS` is substituted. If a single ask raises, its
    section falls back to a short error sentinel; the other two are
    unaffected (best-effort parallel fan-out).

    ``ask_fn`` is the injection seam tests use to replace the real
    Atlas client. Production callers pass ``None`` and we lazily build
    the provider via :func:`_default_ask_factory`.
    """
    if ask_fn is None:
        try:
            ask_fn = _default_ask_factory()
        except Exception as e:
            logger.warning("draft.atlas_unavailable err=%s", e)
            # Atlas not configured — fall back to all-empty sections so
            # the surface still works (AC1: always 3 sections).
            return [(label, _EMPTY_FALLBACKS[label]) for label, _ in _CONTEXT_QUESTIONS]

    questions = [
        (label, template.format(recipient=recipient_display))
        for label, template in _CONTEXT_QUESTIONS
    ]

    def _ask_one(label_q: Tuple[str, str]) -> Tuple[str, str]:
        label, question = label_q
        try:
            payload = ask_fn(question=question)
            answer = _extract_answer(payload)
            if _is_empty_answer(answer):
                return (label, _EMPTY_FALLBACKS[label])
            return (label, answer)
        except Exception as e:
            logger.info("draft.atlas_ask_failed label=%s err=%s", label, e)
            return (label, f"(Atlas lookup failed: {e})")

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="draft-atlas") as pool:
        results = list(pool.map(_ask_one, questions))
    return results


def _compose_context_block(sections: List[Tuple[str, str]]) -> str:
    """Render fetched Atlas sections as a Slack-friendly block.

    Markdown is intentionally minimal — Slack's slash-command response
    surface renders ``*bold*`` and plain newlines but not full markdown.
    Each section is one bold label followed by the answer body.
    """
    lines = ["Context found:"]
    for label, answer in sections:
        lines.append("")
        lines.append(f"*{label}:*")
        lines.append(answer)
    return "\n".join(lines)


def _compose_full_reply(
    args: DraftArgs,
    *,
    ask_fn: Optional[Callable[..., dict]] = None,
) -> str:
    """030-B full reply: header + intent + Atlas context + 030-C TODO."""
    head = _compose_stub_reply(args)
    sections = fetch_atlas_context(args.recipient.display, ask_fn=ask_fn)
    context_block = _compose_context_block(sections)
    return (
        f"{head}\n\n"
        f"{context_block}\n\n"
        "Draft TODO 030-C — context loaded, ready for composition."
    )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def handle_draft(
    raw_args: str,
    *,
    ask_fn: Optional[Callable[..., dict]] = None,
) -> str:
    """``/draft <recipient> <context>`` — Plan 030-A/B handler.

    030-A parses the recipient + intent. 030-B fans out three parallel
    ``atlas_ask`` calls (prior commitments, contradictions, last
    interaction) and renders the result as a Slack context block. The
    LLM draft composition itself is still deferred to 030-C and surfaces
    as the trailing ``Draft TODO 030-C`` line.

    ``ask_fn`` is the test seam used by ``test_draft.py`` to inject a
    mock Atlas client. Production callers omit it; the handler lazily
    builds an :class:`AtlasMemoryProvider` and reuses its ``_ask`` path.
    """
    args = _parse_args(raw_args)
    if args is None:
        return _usage()
    logger.info(
        "draft.invoked recipient_kind=%s recipient_value=%s intent_len=%d",
        args.recipient.kind,
        args.recipient.value,
        len(args.intent),
    )
    return _compose_full_reply(args, ask_fn=ask_fn)
