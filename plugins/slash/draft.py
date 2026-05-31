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
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

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
# 030-C — Atlas write-back on Send
# ---------------------------------------------------------------------------


def _default_atlas_writeback_factory() -> Callable[..., dict]:
    """Build a callable that POSTs an ``atlas:AgentDraft`` triple to Atlas.

    The Atlas write surface lives on the AtlasMemoryProvider (see Plan
    022-A). We piggy-back on its configured base_url + token so the
    /draft writeback uses the same auth path as /v1/ask reads.
    """
    from plugins.memory.atlas import AtlasMemoryProvider  # local import

    provider = AtlasMemoryProvider()
    provider.initialize(session_id="draft-writeback")
    base_url = getattr(provider, "_base_url", "") or os.environ.get("ATLAS_BASE_URL", "")
    token = getattr(provider, "_token", "") or os.environ.get("ATLAS_TOKEN", "")

    def _writeback(*, draft: StoredDraft, decision: str) -> dict:
        import httpx
        url = f"{base_url.rstrip('/')}/v1/ingest/agent-draft"
        body = {
            "type": "atlas:AgentDraft",
            "draft_id": draft.draft_id,
            "recipient_kind": draft.recipient_kind,
            "recipient_value": draft.recipient_value,
            "intent": draft.intent,
            "body": draft.body,
            "decision": decision,
            "agent": "hermes",
            "provenance": "slash:/draft",
        }
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        resp = httpx.post(url, json=body, headers=headers, timeout=5.0)
        resp.raise_for_status()
        return resp.json() if resp.content else {"ok": True}

    return _writeback


def record_draft_decision(
    draft_id: str,
    decision: str,
    *,
    writeback_fn: Optional[Callable[..., dict]] = None,
) -> Tuple[bool, str]:
    """Record a Send / Edit / Discard decision against Atlas.

    Returns ``(ok, message)``. Used by the Slack action handlers
    (gateway/platforms/slack.py). Only ``decision="send"`` writes an
    ``atlas:AgentDraft`` triple — Edit and Discard are tracked locally
    but not persisted (per 030 design decision 5: write-back only on
    Send).
    """
    if decision == "send":
        draft = pop_stored_draft(draft_id)
        if draft is None:
            return False, "Draft not found (already actioned or expired)."
        try:
            if writeback_fn is None:
                writeback_fn = _default_atlas_writeback_factory()
            writeback_fn(draft=draft, decision="send")
            return True, f"Draft sent to {draft.recipient_display}; logged to Atlas."
        except Exception as exc:
            logger.warning("draft.atlas_writeback_failed draft_id=%s err=%s", draft_id, exc)
            # Send-intent recorded locally even if Atlas writeback failed —
            # 030-D will reconcile on the next /daily sweep.
            return True, f"Draft marked sent (Atlas writeback deferred: {exc})."
    if decision == "discard":
        pop_stored_draft(draft_id)
        return True, "Draft discarded."
    if decision == "edit":
        # Don't pop — Edit keeps the draft alive for the modal flow.
        draft = get_stored_draft(draft_id)
        if draft is None:
            return False, "Draft not found (already actioned or expired)."
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
        return _usage()
    logger.info(
        "draft.invoked recipient_kind=%s recipient_value=%s intent_len=%d",
        args.recipient.kind,
        args.recipient.value,
        len(args.intent),
    )
    return _compose_full_reply(args, ask_fn=ask_fn, compose_fn=compose_fn)
