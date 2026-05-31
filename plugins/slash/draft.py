"""/draft slash command — Atlas-aware draft (Plan 030-A/B/C).

Three phases of Plan 030 ("Atlas-aware draft skill" — R2 CP3) live
here. 030-A shipped the slash-command skeleton + recipient resolution.
030-B layers in the Atlas context fetch: three parallel ``atlas_ask``
questions before any LLM call. 030-C (this phase) feeds the recipient
+ intent + Atlas context into Amazon Bedrock's Nova-Pro and renders the
resulting draft in Slack with three action buttons:

    Send     — write an ``atlas:AgentDraft`` triple + post send-confirm
    Edit     — open a Slack modal for tweaks (handler stub; modal in 030-E)
    Discard  — drop the draft, log nothing

The slash handler itself is sync and returns text — the Slack platform
adapter (``gateway/platforms/slack.py``) detects the ``[DRAFT_ACTIONS:<id>]``
marker emitted by :func:`compose_action_marker` and replaces it with a
Block Kit action row before posting. This keeps the slash protocol's
"return string" contract intact while enabling interactive UX.

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
import os
import re
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 030-E — Usage metric instrumentation
# ---------------------------------------------------------------------------
#
# Plan 030 (R2 CP3) projects /draft as a 5–15x/week activity. To validate
# that claim — and to monitor draft funnel health (invoked → sent vs
# edited vs discarded, context-hit ratio, latency tail) — we emit
# structured events at INFO. The events are scraped by the CloudWatch
# Logs Insights query documented in
# ``agentic-hub/docs/runbooks/030-usage-tracking.md`` and rolled up
# weekly.
#
# Event grammar (matches the existing ``draft.<verb>`` prefix used
# elsewhere in this module):
#
#   event=draft.invoked          recipient_kind=<email|handle|unresolved> intent_len=<n>
#   event=draft.context_found    found=<1|0> sections_non_empty=<0..3>
#   metric=draft.latency_ms      value=<int> recipient_kind=<...>
#   event=draft.sent             recipient_kind=<...> contradiction=<1|0>
#   event=draft.edited           recipient_kind=<...>
#   event=draft.discarded        recipient_kind=<...>
#
# We use the existing stdlib ``logger.info`` path — Hermes does not use
# structlog. CloudWatch Logs Insights parses ``key=value`` tokens for
# free, so the line format is the structured surface.


def _emit_metric(event_name: str, **fields: Any) -> None:
    """Emit one structured metric line at INFO.

    The first positional segment is ``event=<name>`` (or
    ``metric=<name>`` for distribution-shaped events like latency, by
    convention) followed by space-separated ``key=value`` pairs. Values
    are coerced to strings; ``None`` becomes the literal string
    ``"none"`` so the parser sees a valid token.

    This helper is a no-IO, exception-safe shim — if logging itself
    raises we swallow it so a metric path never breaks a user flow.
    """
    try:
        prefix = "metric" if event_name.endswith("_ms") or event_name.startswith("metric.") else "event"
        parts = [f"{prefix}={event_name}"]
        for k, v in fields.items():
            if v is None:
                v = "none"
            # Bool first (bool is a subclass of int).
            if isinstance(v, bool):
                v = 1 if v else 0
            parts.append(f"{k}={v}")
        logger.info(" ".join(parts))
    except Exception:  # pragma: no cover — defensive
        pass

# Default Bedrock model id for Nova-Pro. Override with NOVA_PRO_MODEL_ID
# to point at a regional inference profile (e.g. ``us.amazon.nova-pro-v1:0``).
_DEFAULT_NOVA_PRO_MODEL = "amazon.nova-pro-v1:0"

# Length cap for the Atlas context payload we feed into Nova-Pro. Atlas
# answers can run long; we cap each section so prompt size stays bounded.
_CONTEXT_SECTION_CHAR_CAP = 1200


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
    compose_fn: Optional[Callable[..., str]] = None,
) -> str:
    """030-B/C full reply: header + intent + Atlas context + Nova-Pro draft.

    Trailing ``[DRAFT_ACTIONS:<draft_id>]`` marker is detected by the
    Slack adapter and swapped for a Block Kit action row (Send / Edit /
    Discard). The draft itself is stored in :data:`_DRAFT_STORE` keyed
    by ``draft_id`` so button handlers can recover it without
    round-tripping the body through the Slack action ``value`` field
    (which has a 2000-char cap).
    """
    head = _compose_stub_reply(args)
    sections = fetch_atlas_context(args.recipient.display, ask_fn=ask_fn)

    # 030-E: emit context_found ratio so we can validate the R2 CP3
    # claim that the corpus is dense enough to be useful. A section is
    # considered non-empty when its answer is not one of the
    # _EMPTY_FALLBACKS sentinels and not an "(Atlas lookup failed: ...)"
    # error string.
    non_empty = sum(
        1
        for label, ans in sections
        if ans
        and ans != _EMPTY_FALLBACKS.get(label)
        and not ans.startswith("(Atlas lookup failed")
    )
    _emit_metric(
        "draft.context_found",
        found=1 if non_empty > 0 else 0,
        sections_non_empty=non_empty,
        recipient_kind=args.recipient.kind,
    )

    context_block = _compose_context_block(sections)

    draft_body, draft_error = _safe_compose(
        args, sections, compose_fn=compose_fn,
    )
    draft_id = _store_draft(args, draft_body)

    if draft_error:
        draft_block = (
            "*Draft (fallback):*\n"
            f"{draft_body}\n\n"
            f"_Nova-Pro unavailable ({draft_error}); using fallback template._"
        )
    else:
        draft_block = f"*Draft:*\n{draft_body}"

    return (
        f"{head}\n\n"
        f"{context_block}\n\n"
        f"{draft_block}\n\n"
        f"{compose_action_marker(draft_id)}"
    )


# ---------------------------------------------------------------------------
# 030-C — Nova-Pro draft composition
# ---------------------------------------------------------------------------

_DRAFT_SYSTEM_PROMPT = (
    "You are Blake's outbound writing assistant. You compose short, direct, "
    "professional emails on Blake's behalf. House style: warm but concise; "
    "no filler; no apologies; one ask per message; no em-dashes (use commas "
    "or periods instead); never invent facts. Honor every prior commitment "
    "surfaced in the context block — if the context says Blake already "
    "agreed to X, the draft must not contradict X. If the context section "
    "lists an open contradiction, acknowledge it tactfully or steer around "
    "it. Preserve any ``[cite:...]`` markers verbatim if you reference a "
    "specific cited fact. Output ONLY the email body (no subject line, no "
    "signature, no preamble like 'Here is the draft:')."
)


def _truncate(text: str, cap: int = _CONTEXT_SECTION_CHAR_CAP) -> str:
    """Char-cap a section so the prompt size stays bounded."""
    if not text:
        return ""
    if len(text) <= cap:
        return text
    return text[: cap - 3].rstrip() + "..."


def build_nova_prompt(
    args: DraftArgs,
    sections: List[Tuple[str, str]],
) -> Tuple[str, str]:
    """Render (system, user) prompts for Nova-Pro.

    Pure function — no IO. Exposed for unit-testing the prompt shape so
    the prompt-engineering layer can evolve independently of the Bedrock
    wire format.
    """
    context_lines: List[str] = []
    for label, answer in sections:
        context_lines.append(f"## {label}")
        context_lines.append(_truncate(answer))
        context_lines.append("")
    context_blob = "\n".join(context_lines).strip()

    intent_line = args.intent if args.intent else "(no specific intent provided)"
    recipient_label = args.recipient.display
    if args.recipient.kind == "email":
        recipient_label = f"{args.recipient.display} ({args.recipient.value})"

    user_msg = (
        f"Recipient: {recipient_label}\n"
        f"Blake's intent for this draft: {intent_line}\n\n"
        f"Context Blake has accumulated about this recipient (from Atlas memory):\n"
        f"{context_blob if context_blob else '(no prior context found)'}\n\n"
        f"Write the email body now."
    )
    return _DRAFT_SYSTEM_PROMPT, user_msg


def _default_compose_factory() -> Callable[..., str]:
    """Build a callable that invokes Nova-Pro via Bedrock ``invoke_model``.

    Returns a thunk ``compose(system, user) -> str``. Raises at call time
    if boto3 / Bedrock are unreachable — the caller's ``_safe_compose``
    catches that and falls back to a deterministic template.
    """
    region = os.environ.get("BEDROCK_REGION", "us-east-1")
    model_id = os.environ.get("NOVA_PRO_MODEL_ID", _DEFAULT_NOVA_PRO_MODEL)

    def _compose(system: str, user: str) -> str:
        import boto3  # local import — keeps test envs without boto3 happy

        client = boto3.client("bedrock-runtime", region_name=region)
        # Nova family uses the Bedrock Converse-style "messages" payload.
        payload = {
            "schemaVersion": "messages-v1",
            "system": [{"text": system}],
            "messages": [
                {"role": "user", "content": [{"text": user}]},
            ],
            "inferenceConfig": {
                "maxTokens": 800,
                "temperature": 0.4,
                "topP": 0.9,
            },
        }
        resp = client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json",
        )
        raw = resp["body"].read()
        data = json.loads(raw)
        # Nova response shape: {"output": {"message": {"content": [{"text": "..."}]}}, ...}
        try:
            return data["output"]["message"]["content"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Nova-Pro returned malformed payload: {exc}")

    return _compose


def _fallback_draft(args: DraftArgs) -> str:
    """Deterministic template used when Nova-Pro is unavailable.

    Keeps /draft useful in dev/test environments without Bedrock creds.
    Intentionally bland — the user will rewrite, but they get a header,
    a one-line ask, and a sign-off they can build on.
    """
    greeting_name = args.recipient.display if args.recipient.kind == "email" else args.recipient.display
    intent = args.intent or "following up on our recent thread"
    return (
        f"Hi {greeting_name},\n\n"
        f"Quick note {intent}. Let me know what makes sense on your end "
        f"and I'll work around it.\n\n"
        f"Thanks,\nBlake"
    )


def _safe_compose(
    args: DraftArgs,
    sections: List[Tuple[str, str]],
    *,
    compose_fn: Optional[Callable[..., str]] = None,
) -> Tuple[str, Optional[str]]:
    """Invoke Nova-Pro with a graceful fallback.

    Returns ``(draft_body, error_message_or_None)``. If ``compose_fn`` is
    None we lazily build the Bedrock-backed default. Any exception
    (boto3 missing, AWS creds missing, throttling, malformed payload) is
    swallowed and the fallback template is returned; the caller surfaces
    the error string in the Slack reply so Blake knows the draft is a
    fallback rather than Nova-Pro output.
    """
    system, user = build_nova_prompt(args, sections)
    try:
        if compose_fn is None:
            compose_fn = _default_compose_factory()
        body = compose_fn(system, user)
        if not body or not body.strip():
            raise RuntimeError("empty draft body")
        return body.strip(), None
    except Exception as exc:
        logger.info("draft.nova_compose_failed err=%s", exc)
        return _fallback_draft(args), str(exc) or type(exc).__name__


# ---------------------------------------------------------------------------
# 030-C — Draft store + action marker (Slack button wiring seam)
# ---------------------------------------------------------------------------


@dataclass
class StoredDraft:
    """A composed draft awaiting Blake's Send / Edit / Discard decision."""

    draft_id: str
    recipient_kind: str
    recipient_value: str
    recipient_display: str
    intent: str
    body: str
    created_at: float = field(default_factory=time.time)


_DRAFT_STORE: Dict[str, StoredDraft] = {}
_DRAFT_STORE_LOCK = threading.Lock()
# Drafts older than this are eligible for sweep by 030-E. We don't sweep
# here (that's 030-E's cleanup job); we just bound memory with a soft
# cap so a runaway /draft loop can't OOM the gateway.
_DRAFT_STORE_SOFT_CAP = 256


def _store_draft(args: DraftArgs, body: str) -> str:
    """Persist a freshly composed draft and return its short id."""
    draft_id = secrets.token_urlsafe(8)
    stored = StoredDraft(
        draft_id=draft_id,
        recipient_kind=args.recipient.kind,
        recipient_value=args.recipient.value,
        recipient_display=args.recipient.display,
        intent=args.intent,
        body=body,
    )
    with _DRAFT_STORE_LOCK:
        if len(_DRAFT_STORE) >= _DRAFT_STORE_SOFT_CAP:
            # Evict the oldest entry to keep the dict bounded.
            oldest = min(_DRAFT_STORE.values(), key=lambda d: d.created_at)
            _DRAFT_STORE.pop(oldest.draft_id, None)
        _DRAFT_STORE[draft_id] = stored
    return draft_id


def get_stored_draft(draft_id: str) -> Optional[StoredDraft]:
    """Public lookup used by the Slack action handlers."""
    with _DRAFT_STORE_LOCK:
        return _DRAFT_STORE.get(draft_id)


def pop_stored_draft(draft_id: str) -> Optional[StoredDraft]:
    """Remove and return a stored draft (used on Send / Discard)."""
    with _DRAFT_STORE_LOCK:
        return _DRAFT_STORE.pop(draft_id, None)


# Marker the Slack adapter scans for. Format chosen to be visually
# inert if a non-Slack surface accidentally renders the raw reply.
_ACTION_MARKER_RE = re.compile(r"\[DRAFT_ACTIONS:([A-Za-z0-9_\-]{4,32})\]")


def compose_action_marker(draft_id: str) -> str:
    return f"[DRAFT_ACTIONS:{draft_id}]"


def extract_action_draft_id(reply: str) -> Optional[str]:
    """Pull the ``draft_id`` out of a reply, or None if no marker present.

    Used by the Slack adapter to decide whether to attach an action row.
    """
    m = _ACTION_MARKER_RE.search(reply or "")
    return m.group(1) if m else None


def strip_action_marker(reply: str) -> str:
    """Return ``reply`` with the action marker removed (for non-Slack surfaces)."""
    return _ACTION_MARKER_RE.sub("", reply or "").rstrip()


# ---------------------------------------------------------------------------
# 030-D — atlas:AgentDraft write-back on Send (hardened)
# ---------------------------------------------------------------------------
#
# Per master plan 030 Decision 5 + Phase 030-D spec, when Blake clicks Send
# we persist one ``atlas:AgentDraft`` triple to ``urn:atlas:graph:events``
# with full provenance:
#
#   urn:atlas:agent-draft:<ulid>
#     atlas:authorAgent          "hermes"
#     atlas:recipient            <urn:atlas:person:...>
#     atlas:draftBody            "<literal text Blake sent>"
#     atlas:contextSourceCount   3   (the three atlas_ask outputs)
#     atlas:sentAt               <iso>
#     atlas:validFrom            <iso>
#     atlas:provenanceSource     "slack_draft_button"
#     atlas:triggerSlackMessage  <slack message id>
#
# The write goes through the existing AtlasMemoryProvider._write_fact shim
# (POST /v1/memory/hermes/write) — same path as 026-B's AgentDecision —
# rather than a bespoke HTTP route. The Atlas-side typed-entity expander
# parses the ``[AgentDraft] {...json}`` envelope into RDF (Plan 015 /
# 025-C typed-entity convention).
#
# Edit and Discard do NOT write — only Send commits (Decision 5).

# Default graph for AgentDraft writes. The master plan calls for
# ``urn:atlas:graph:events`` (matching AgentDecision from 026-B). The
# _write_fact target argument is the *memory* level routing key; the
# provider maps target="memory" → events graph on the Atlas side.
_AGENT_DRAFT_WRITE_TARGET = "memory"

# Provenance source tag — used by Atlas-side typed-entity expansion to
# distinguish slash-button sends from cron / batch / replay writes.
_AGENT_DRAFT_PROVENANCE = "slack_draft_button"

# Number of context sources fed into Nova-Pro composition. Mirrors the
# three parallel atlas_ask calls in fetch_atlas_context.
_AGENT_DRAFT_CONTEXT_SOURCE_COUNT = 3

# Contradiction probe template — fires once before the writeback to
# surface anything Blake may have committed to in the last 30 days that
# this draft contradicts. Per master plan 030-D the probe does NOT block
# the write (Blake's intent is final per Decision 7); a non-empty result
# is logged as a warning event so the audit trail captures it.
_CONTRADICTION_PROBE_TEMPLATE = (
    "Does a draft to {recipient} saying \"{summary}\" contradict anything "
    "I have committed to with this recipient in the last 30 days?"
)


def _now_iso_utc() -> str:
    """UTC ISO-8601 timestamp with second resolution.

    Local helper (don't reuse daily_writeback._now_iso to keep modules
    independently testable). Second resolution mirrors AgentDecision.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _mint_agent_draft_urn() -> str:
    """Mint a ULID-shaped URN for an AgentDraft.

    The spec calls for ``urn:atlas:agent-draft:<ulid>``. Hermes doesn't
    pull in the ``ulid`` package, so we approximate with a sortable
    token: <unix-ms-hex>-<random>. Atlas-side parsers don't validate
    ULID format; they only require a stable globally-unique IRI.
    """
    ts_ms = int(time.time() * 1000)
    ts_hex = format(ts_ms, "x")
    rand = secrets.token_hex(6)
    return f"urn:atlas:agent-draft:{ts_hex}-{rand}"


def _recipient_iri(draft: "StoredDraft") -> str:
    """Build the ``atlas:recipient`` IRI for a draft.

    Email recipients map to a deterministic person IRI; handle recipients
    map to a slack-handle IRI. Unresolved recipients fall through to a
    text IRI so the triple still binds something Atlas can dedup on.
    """
    if draft.recipient_kind == "email":
        return f"urn:atlas:person:email:{draft.recipient_value}"
    if draft.recipient_kind == "handle":
        return f"urn:atlas:person:slack-handle:{draft.recipient_value}"
    return f"urn:atlas:person:unresolved:{draft.recipient_value}"


def _summarize_for_probe(body: str, *, cap: int = 160) -> str:
    """Cap the draft body to a one-liner the contradiction probe can use.

    The probe goes back through atlas_ask which has its own LLM budget;
    feeding the full 800-token body would inflate the question
    needlessly. The first line + char cap is enough for the probe to
    spot Commitment-level mismatches.
    """
    if not body:
        return ""
    first_line = body.split("\n", 1)[0].strip()
    if len(first_line) > cap:
        first_line = first_line[: cap - 3].rstrip() + "..."
    return first_line


def run_contradiction_probe(
    draft: "StoredDraft",
    *,
    ask_fn: Optional[Callable[..., dict]] = None,
) -> Optional[str]:
    """Probe Atlas for 30-day commitment contradictions touching this draft.

    Returns the contradiction-answer string when Atlas flags one; ``None``
    when the corpus is clean (or unreachable). Per Decision 7 the result
    does NOT block the write — the caller logs a warning and writes
    anyway. The return value is included in the AgentDraft's
    ``contradiction_warning`` field so the typed-entity expander can
    surface it later.
    """
    if ask_fn is None:
        try:
            ask_fn = _default_ask_factory()
        except Exception as exc:
            logger.info("draft.contradiction_probe_unavailable err=%s", exc)
            return None

    summary = _summarize_for_probe(draft.body)
    if not summary:
        return None

    question = _CONTRADICTION_PROBE_TEMPLATE.format(
        recipient=draft.recipient_display, summary=summary,
    )
    try:
        payload = ask_fn(question=question)
    except Exception as exc:
        logger.info("draft.contradiction_probe_failed err=%s", exc)
        return None
    answer = _extract_answer(payload)
    if _is_empty_answer(answer):
        return None
    return answer


def build_agent_draft_content(
    draft: "StoredDraft",
    *,
    trigger_slack_message: Optional[str] = None,
    contradiction_warning: Optional[str] = None,
    sent_at: Optional[str] = None,
    urn: Optional[str] = None,
) -> Tuple[str, dict]:
    """Render the ``[AgentDraft] {...json}`` payload for Atlas write.

    Returns ``(content_str, body_dict)`` — the dict is exposed so tests
    can assert on every provenance field without having to re-parse the
    JSON envelope.

    Mirrors the AgentDecision shape from 026-B: a single-line
    ``[<TypeTag>] {json}`` content blob that the Atlas typed-entity
    expander parses back into RDF triples.
    """
    now = sent_at or _now_iso_utc()
    body = {
        "type": "atlas:AgentDraft",
        "urn": urn or _mint_agent_draft_urn(),
        "authorAgent": "hermes",
        "recipient": _recipient_iri(draft),
        "recipientDisplay": draft.recipient_display,
        "recipientKind": draft.recipient_kind,
        "recipientValue": draft.recipient_value,
        "intent": draft.intent,
        "draftBody": draft.body,
        "contextSourceCount": _AGENT_DRAFT_CONTEXT_SOURCE_COUNT,
        "sentAt": now,
        "validFrom": now,
        "provenanceSource": _AGENT_DRAFT_PROVENANCE,
        "triggerSlackMessage": trigger_slack_message or "",
        "draftId": draft.draft_id,
    }
    if contradiction_warning:
        body["contradictionWarning"] = contradiction_warning
    return f"[AgentDraft] {json.dumps(body, separators=(',', ':'))}", body


# Type signature mirroring daily_writeback.AtlasWriter so the Slack
# adapter can pass the same provider-bound writer.
AtlasDraftWriter = Callable[[str, str], None]
"""``atlas_writer(target, content)`` — best-effort writer.

Wraps ``AtlasMemoryProvider._write_fact``; tests inject a recording
fake. Per the spec the AgentDraft goes to ``target="memory"`` (events
graph), matching the AgentDecision pattern from 026-B.
"""


def _default_atlas_draft_writer(target: str, content: str) -> None:
    """Real writer — defers Atlas import so test envs without atlas pass.

    Mirrors :func:`plugins.slash.daily_writeback._default_atlas_writer`
    one-for-one so 030-D inherits the same auth / breaker / config path
    as 026-B. If the provider isn't installed or isn't configured, we
    silently no-op — the Slack-side "Sent" UX still resolves and we log
    the degradation.
    """
    try:
        from plugins.memory.atlas import AtlasMemoryProvider  # type: ignore
    except Exception as exc:
        logger.info("draft.writeback_provider_unavailable err=%s", exc)
        return

    provider = AtlasMemoryProvider()
    if not provider.is_available():
        logger.info("draft.writeback_provider_not_configured")
        return
    provider._write_fact(target=target, action="add", content=content)


def write_agent_draft(
    draft: "StoredDraft",
    *,
    trigger_slack_message: Optional[str] = None,
    contradiction_warning: Optional[str] = None,
    atlas_writer: Optional[AtlasDraftWriter] = None,
) -> Tuple[bool, dict]:
    """Persist one ``atlas:AgentDraft`` triple. Returns (ok, body_dict).

    Never raises — the Slack "Sent" UX must resolve even if Atlas is
    down. The returned body dict is what was written (or would have
    been); callers use it for the audit log + the in-thread reply.
    """
    content, body = build_agent_draft_content(
        draft,
        trigger_slack_message=trigger_slack_message,
        contradiction_warning=contradiction_warning,
    )
    writer = atlas_writer or _default_atlas_draft_writer
    try:
        writer(_AGENT_DRAFT_WRITE_TARGET, content)
        logger.info(
            "draft.atlas_writeback_ok urn=%s recipient_kind=%s",
            body["urn"], draft.recipient_kind,
        )
        return True, body
    except Exception as exc:
        logger.warning(
            "draft.atlas_writeback_failed urn=%s err=%s", body["urn"], exc,
        )
        return False, body


def _writeback_is_legacy(fn: Callable) -> bool:
    """Detect a 030-C-shaped writeback (``(draft=, decision=) -> dict``).

    030-D widened the writeback contract to
    ``(draft=, trigger_slack_message=, contradiction_warning=) -> (ok, dict)``.
    Old call sites + tests still pass the 030-C shape; we sniff the
    signature so both keep working without a breaking API change.
    """
    try:
        import inspect
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    # A **kwargs sink is ambiguous — assume modern shape so the new path
    # is exercised by the new tests. The 030-C tests that use **kw spies
    # all either raise (bad/discard) or no-op, so misclassifying them as
    # modern is harmless.
    if "decision" in params and "trigger_slack_message" not in params:
        return True
    return False


def record_draft_decision(
    draft_id: str,
    decision: str,
    *,
    trigger_slack_message: Optional[str] = None,
    probe_fn: Optional[Callable[..., Optional[str]]] = None,
    writeback_fn: Optional[Callable[..., Any]] = None,
) -> Tuple[bool, str]:
    """Record a Send / Edit / Discard decision and (for Send) write Atlas.

    Returns ``(ok, message)``. Used by the Slack button-action handlers
    (gateway/platforms/slack.py). Per master plan 030 Decision 5 only
    ``decision="send"`` writes an ``atlas:AgentDraft`` triple — Edit and
    Discard are local-only.

    The contradiction probe (030-D) fires once before the write. A
    non-empty hit is logged + carried into the writeback's
    ``contradictionWarning`` field but does NOT block the send (per
    Decision 7: Blake's intent is final).
    """
    if decision == "send":
        draft = pop_stored_draft(draft_id)
        if draft is None:
            return False, "Draft not found (already actioned or expired)."
        # 030-E: emit ``event=draft.sent`` BEFORE the writeback so the
        # send-button click counts even if Atlas write fails. The
        # ``contradiction`` flag is filled in after the probe below via a
        # follow-up structured log line.
        _emit_metric(
            "draft.sent",
            recipient_kind=draft.recipient_kind,
        )

        # Per 030-D + Decision 7: probe first, warn but don't block.
        try:
            warning = run_contradiction_probe(draft, ask_fn=None) \
                if probe_fn is None else probe_fn(draft=draft)
        except Exception as exc:
            logger.info("draft.contradiction_probe_threw err=%s", exc)
            warning = None
        if warning:
            logger.warning(
                "draft.contradiction_detected draft_id=%s warning=%s",
                draft_id, warning[:240],
            )
        # 030-E: contradiction-on-send flag — feeds the runbook query
        # that surfaces "% of sends that fired a ⚠️".
        _emit_metric(
            "draft.contradiction_on_send",
            contradiction=1 if warning else 0,
            recipient_kind=draft.recipient_kind,
        )

        # Legacy 030-C spy signature → adapt to the new writeback shape
        # so old tests keep passing without a breaking surface change.
        if writeback_fn is not None and _writeback_is_legacy(writeback_fn):
            try:
                writeback_fn(draft=draft, decision="send")
                tail = " (contradiction logged)" if warning else ""
                return True, f"Draft sent to {draft.recipient_display}; logged to Atlas.{tail}"
            except Exception as exc:
                logger.warning("draft.legacy_writeback_failed err=%s", exc)
                return True, f"Draft marked sent (Atlas writeback deferred: {exc})."

        actual_writeback = writeback_fn or (
            lambda **kw: write_agent_draft(**kw)
        )
        try:
            result = actual_writeback(
                draft=draft,
                trigger_slack_message=trigger_slack_message,
                contradiction_warning=warning,
            )
            # Modern shape returns (ok, body_dict); legacy **kw spies that
            # don't unpack right return whatever they return. Coerce.
            if isinstance(result, tuple) and len(result) == 2:
                ok, _body = result
            else:
                ok = True
        except Exception as exc:
            logger.warning("draft.atlas_writeback_threw draft_id=%s err=%s", draft_id, exc)
            return True, f"Draft marked sent (Atlas writeback deferred: {exc})."

        if not ok:
            return True, (
                f"Draft sent to {draft.recipient_display}; "
                "Atlas writeback deferred."
            )
        tail = " (contradiction logged)" if warning else ""
        return True, f"Draft sent to {draft.recipient_display}; logged to Atlas.{tail}"

    if decision == "discard":
        # Per Decision 5 / AC: Discard does NOT write.
        popped = pop_stored_draft(draft_id)
        # 030-E: emit the discard counter only when we actually found
        # the draft — a duplicate Discard click on an already-actioned
        # draft is not a new decision.
        if popped is not None:
            _emit_metric(
                "draft.discarded",
                recipient_kind=popped.recipient_kind,
            )
        return True, "Draft discarded."

    if decision == "edit":
        # Per Decision 5 / AC: Edit does NOT write. Keep draft alive
        # for the modal flow (030-E).
        draft = get_stored_draft(draft_id)
        if draft is None:
            return False, "Draft not found (already actioned or expired)."
        # 030-E: emit edit counter — only on successful lookup so we
        # don't double-count stale-button clicks.
        _emit_metric(
            "draft.edited",
            recipient_kind=draft.recipient_kind,
        )
        return True, f"Edit draft: {draft.body}"

    return False, f"Unknown decision: {decision}"


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def handle_draft(
    raw_args: str,
    *,
    ask_fn: Optional[Callable[..., dict]] = None,
    compose_fn: Optional[Callable[..., str]] = None,
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
        # No metric — usage replies don't count as invocations against
        # the 5–15x/week target; they're help-text fetches.
        return _usage()

    # 030-E: emit one ``event=draft.invoked`` per real invocation so the
    # CloudWatch Logs Insights rollup can count weekly /draft fires per
    # recipient_kind. Latency is wall-clock around the compose path —
    # excludes Slack round-trip on either side (the slash transport
    # measures that separately).
    started = time.monotonic()
    _emit_metric(
        "draft.invoked",
        recipient_kind=args.recipient.kind,
        intent_len=len(args.intent),
    )
    logger.info(
        "draft.invoked recipient_kind=%s recipient_value=%s intent_len=%d",
        args.recipient.kind,
        args.recipient.value,
        len(args.intent),
    )
    try:
        reply = _compose_full_reply(args, ask_fn=ask_fn, compose_fn=compose_fn)
    except Exception:
        # 030-E AC: latency metric is NOT recorded on error — error
        # latency would skew the distribution. We still re-raise so the
        # surface behavior is unchanged.
        _emit_metric(
            "draft.error",
            recipient_kind=args.recipient.kind,
        )
        raise
    elapsed_ms = int((time.monotonic() - started) * 1000)
    _emit_metric(
        "draft.latency_ms",
        value=elapsed_ms,
        recipient_kind=args.recipient.kind,
    )
    return reply
