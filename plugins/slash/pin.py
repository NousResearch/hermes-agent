"""📌 reaction handler — manual Slack-to-Atlas pin ingest (Plan 026-C).

When Blake reacts with 📌 (Slack emoji name ``pushpin``) on any message,
this handler:

1. Fetches the surrounding thread context (full thread if it's a reply, or
   the single message if not — see ``_collect_thread_messages``).
2. Chunks the thread (one chunk per Slack message).
3. POSTs the chunks to Atlas ``/v1/ingest`` with provenance metadata that
   marks the source as ``slack_manual_pin`` so downstream queries can
   distinguish deliberate-seed corpus from the passive firehose (Plan
   022-B).
4. Replies in-thread with the Atlas ``job_id`` so Blake gets a real-time
   confirmation that the ingest fired. If anything fails along the way,
   the reply surfaces the error rather than silently dropping (per Plan
   026-C AC: "no silent drops").

This is the **R2 CP2 corpus-seed precursor** — until Plan 022-B's passive
firehose lands, Blake's 📌 reaction is the deliberate-seeding lever that
populates the Atlas graph so the `/daily` brief (Plan 026-A) has
something to cite.

Auth: ``SLACK_ALLOWED_USERS`` is enforced **in this handler** because
Slack ``reaction_added`` events are not slash commands — the gateway's
slash-command allowlist doesn't catch them. Non-Blake reactions are
silently dropped.

The reply confirmation includes a stable URN of the form
``urn:atlas:ingest:<job_id>`` so Blake can grep / click through later.
The chunk-level IRIs are minted asynchronously by the ingest pipeline and
are not available at reply time.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional

logger = logging.getLogger(__name__)

# The Slack emoji name for 📌 is ``pushpin`` (Slack canonicalises emoji to
# their colon-name form when delivering reaction events). We accept the
# raw Unicode and the ``pushpin``/``round_pushpin`` Slack names so the
# handler is robust to whichever the workspace emits.
PIN_EMOJI_NAMES = frozenset({"pushpin", "round_pushpin", "📌"})

# Provenance source tag — matches the spec in
# plans/026-slack-daily-brief/026-slack-daily-brief.md §"Phase 026-C".
PROVENANCE_SOURCE = "slack_manual_pin"

# Max thread messages to ingest. Per the master plan: "last 20 messages
# of context, or full thread if <50". We materialise both ends as
# constants so the test surface is explicit.
_MAX_FULL_THREAD = 50
_TAIL_CONTEXT_WINDOW = 20

# /v1/ingest timeout. Atlas writes are routed through the connector
# pipeline and can take a couple of seconds even on the happy path; the
# CP2 path is best-effort so we cap at 10s and bail with an error reply
# rather than blocking the Slack event loop.
_INGEST_TIMEOUT_SECS = 10.0


# ---------------------------------------------------------------------------
# Allowlist gate
# ---------------------------------------------------------------------------


def _allowed_users() -> set[str]:
    """Parse ``SLACK_ALLOWED_USERS`` env var into a set of Slack user IDs.

    A literal ``*`` means "allow any user". An empty value means "no
    allowlist configured → fail closed" so the pin handler never ingests
    arbitrary content from an unauthenticated workspace.
    """
    raw = (os.getenv("SLACK_ALLOWED_USERS") or "").strip()
    if not raw:
        return set()
    return {uid.strip() for uid in raw.split(",") if uid.strip()}


def is_user_allowed(user_id: str) -> bool:
    """Return True iff ``user_id`` may pin to Atlas.

    Fail-closed: an empty / unset allowlist denies. Wildcard ``*`` allows.
    """
    if not user_id:
        return False
    allowed = _allowed_users()
    if not allowed:
        return False
    if "*" in allowed:
        return True
    return user_id in allowed


def is_pin_reaction(emoji_name: str) -> bool:
    """Return True iff the emoji name matches one of our pin aliases."""
    if not emoji_name:
        return False
    return emoji_name.strip(": ") in PIN_EMOJI_NAMES


# ---------------------------------------------------------------------------
# Thread collection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlackMessage:
    """Minimal Slack message shape we ingest.

    We deliberately keep this small so callers (test + real) only need to
    provide ``ts`` + ``user`` + ``text``. Anything else the Slack API
    returns is ignored — Atlas re-extracts entities from raw_text anyway.
    """

    ts: str
    user: str
    text: str


@dataclass(frozen=True)
class PinContext:
    """Resolved context for one 📌 reaction event."""

    channel: str
    thread_ts: str            # parent message ts (== pinned ts if not in thread)
    pinned_ts: str            # the message the user reacted to
    pinned_by_user: str
    messages: tuple[SlackMessage, ...]  # ordered oldest → newest


async def _fetch_thread_via_slack(
    slack_client: Any,
    *,
    channel: str,
    pinned_ts: str,
    thread_ts: Optional[str],
) -> List[SlackMessage]:
    """Fetch thread messages via the Slack Web API.

    Strategy:
      * If we have ``thread_ts`` (the reaction was on a threaded reply),
        use ``conversations.replies`` to pull the whole thread.
      * Otherwise, use ``conversations.history`` with ``latest=pinned_ts``
        and ``inclusive=True``, ``limit=1`` to just pull the pinned
        message itself. Single-message pins are the common case for
        "📌 this one fact".

    Returns messages oldest → newest. On any Slack API error the caller
    falls back to a stub single-message context so the ingest still
    fires (Slack outages shouldn't swallow a pin).
    """
    msgs: list[SlackMessage] = []
    if thread_ts and thread_ts != pinned_ts:
        resp = await slack_client.conversations_replies(
            channel=channel,
            ts=thread_ts,
            limit=_MAX_FULL_THREAD,
        )
        raw = (resp.get("messages") or []) if isinstance(resp, dict) else []
    else:
        # Try thread_ts == pinned_ts first (the pinned message might be
        # a thread root). conversations.replies returns the root alone
        # when there are no replies, which is exactly what we want.
        try:
            resp = await slack_client.conversations_replies(
                channel=channel,
                ts=pinned_ts,
                limit=_MAX_FULL_THREAD,
            )
            raw = (resp.get("messages") or []) if isinstance(resp, dict) else []
        except Exception:
            raw = []
        if not raw:
            # Fall back to a history slice that contains the pinned ts.
            resp = await slack_client.conversations_history(
                channel=channel,
                latest=pinned_ts,
                inclusive=True,
                limit=1,
            )
            raw = (resp.get("messages") or []) if isinstance(resp, dict) else []

    for m in raw:
        ts = str(m.get("ts") or "")
        user = str(m.get("user") or m.get("bot_id") or "")
        text = str(m.get("text") or "")
        if not ts or not text:
            continue
        msgs.append(SlackMessage(ts=ts, user=user, text=text))

    # Trim to the documented window: full thread if <_MAX_FULL_THREAD;
    # otherwise tail _TAIL_CONTEXT_WINDOW so we keep the most recent
    # context around the pin.
    if len(msgs) >= _MAX_FULL_THREAD:
        msgs = msgs[-_TAIL_CONTEXT_WINDOW:]

    # Ensure oldest → newest. Slack returns oldest-first for both
    # endpoints but we sort defensively in case a workspace bot reorders.
    msgs.sort(key=lambda m: m.ts)
    return msgs


# ---------------------------------------------------------------------------
# Ingest payload
# ---------------------------------------------------------------------------


def _format_thread_as_raw_text(ctx: PinContext) -> str:
    """Flatten a thread into a single ``raw_text`` string for /v1/ingest.

    Atlas's ingest pipeline chunks on its own pass — we provide one
    deterministic newline-separated wire format with ``[U<user>]`` user
    tags so downstream extraction can attribute statements to speakers.
    """
    lines: list[str] = []
    lines.append(
        f"Slack thread pinned by <@{ctx.pinned_by_user}> "
        f"in channel {ctx.channel} (thread_ts={ctx.thread_ts}, "
        f"pinned_ts={ctx.pinned_ts})"
    )
    for m in ctx.messages:
        speaker = f"<@{m.user}>" if m.user else "<unknown>"
        lines.append(f"[{m.ts}] {speaker}: {m.text}")
    return "\n".join(lines)


def build_ingest_payload(ctx: PinContext) -> dict:
    """Build the JSON body for ``POST /v1/ingest``.

    Schema mirrors ``atlas.ingest.events.IngestRequest`` (pydantic v2,
    ``extra='forbid'``) so we must stay within its allowed fields:

      * ``source`` → ``IngestSource`` (``kind='manual'``,
        ``connector='slack_manual_pin'``, ``resource_id=pinned_ts``,
        ``run_id=thread_ts``)
      * ``provenance`` → ``IngestProvenance`` (``actor`` = pinning user,
        ``ref`` = channel id)
      * ``raw_text`` → the flattened thread (one chunk per message in
        wire format; Atlas's chunker preserves the per-message boundaries)
    """
    raw_text = _format_thread_as_raw_text(ctx)
    return {
        "source": {
            "kind": "manual",
            "connector": PROVENANCE_SOURCE,
            "resource_id": ctx.pinned_ts,
            "run_id": ctx.thread_ts,
        },
        "raw_text": raw_text,
        "provenance": {
            "actor": ctx.pinned_by_user or "blake",
            "ref": ctx.channel,
        },
    }


async def _post_ingest(
    base_url: str,
    bearer: str,
    payload: dict,
    *,
    httpx_module: Any = None,
) -> dict:
    """POST the payload to ``/v1/ingest``.

    Returns the parsed JSON body on 2xx. Raises on HTTP error or
    transport failure — the caller turns exceptions into a Slack reply.

    ``httpx_module`` is injectable so the unit tests can substitute a
    fake without monkeypatching the import system.
    """
    if httpx_module is None:  # pragma: no cover - trivial branch
        import httpx as _httpx
        httpx_module = _httpx
    headers = {"Content-Type": "application/json"}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    url = f"{base_url.rstrip('/')}/v1/ingest"
    async with httpx_module.AsyncClient(timeout=_INGEST_TIMEOUT_SECS) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Reply formatting
# ---------------------------------------------------------------------------


def format_success_reply(job_id: str) -> str:
    """Render the success confirmation message.

    Includes a stable ``urn:atlas:ingest:<job_id>`` reference Blake can
    paste back into ``/v1/ask`` once the chunks land. We deliberately do
    not promise a chunk URN here because chunk minting is async.
    """
    return (
        f"✓ Pinned to Atlas: urn:atlas:ingest:{job_id}\n"
        f"(chunks will be queryable via /v1/ask in <60s)"
    )


def format_error_reply(reason: str) -> str:
    """Render the failure message.

    Per the AC: "no silent drops". We surface the reason inline so Blake
    sees the failure mode the moment it happens.
    """
    # Trim ridiculously long error strings — Slack wraps but a multi-line
    # 500 trace is just noise. Keep ~240 chars worth of signal.
    short = reason if len(reason) <= 240 else reason[:237] + "..."
    return f"⚠ pin failed: {short}"


# ---------------------------------------------------------------------------
# Top-level handler
# ---------------------------------------------------------------------------


@dataclass
class PinHandlerConfig:
    """Inputs the gateway wires through to the handler."""

    atlas_base_url: str
    atlas_bearer: str
    # Allow tests to inject a fake httpx module without monkeypatching.
    httpx_module: Any = None


async def handle_pin_reaction(
    *,
    event: dict,
    slack_client: Any,
    reply_in_thread: Callable[[str, str, str], Awaitable[None]],
    config: PinHandlerConfig,
) -> Optional[str]:
    """Process a ``reaction_added`` event that may be a 📌 pin.

    Parameters
    ----------
    event : dict
        The raw Slack reaction_added event payload.
    slack_client : Any
        Async Slack Web API client. Used to fetch thread context. The
        handler only calls ``conversations_replies`` and
        ``conversations_history`` so the test mock surface is small.
    reply_in_thread : callable
        ``async def reply(channel, thread_ts, text) -> None`` — posts the
        confirmation back into the same thread. Injected so the handler
        is decoupled from the gateway's specific post-message stack.
    config : PinHandlerConfig
        Atlas endpoint config.

    Returns
    -------
    Optional[str]
        The job_id on success, ``None`` if the event was filtered
        (non-pin emoji, unauthorized user) or ingest failed. The reply
        side-effect carries the user-visible result; this return value
        exists for tests to assert on.

    Failures are caught and surfaced as a Slack reply — they never
    raise out of this function so the gateway event loop is never
    broken by a single bad pin.
    """
    emoji_name = (event.get("reaction") or "").strip()
    if not is_pin_reaction(emoji_name):
        return None  # Not a pin event — silent drop (other reactions handle elsewhere).

    reactor = (event.get("user") or "").strip()
    if not is_user_allowed(reactor):
        logger.info(
            "[pin] dropping 📌 from unauthorized user %r (not in SLACK_ALLOWED_USERS)",
            reactor,
        )
        return None

    item = event.get("item") or {}
    channel = (item.get("channel") or "").strip()
    pinned_ts = (item.get("ts") or "").strip()
    if not channel or not pinned_ts:
        logger.warning("[pin] malformed reaction event (missing channel/ts): %s", event)
        return None

    # Fetch thread context. On Slack-side failure we still ingest with
    # whatever we have (a single synthetic message constructed from the
    # event) so the user sees a confirmation rather than a Slack outage
    # masquerading as a pin bug.
    try:
        messages = await _fetch_thread_via_slack(
            slack_client,
            channel=channel,
            pinned_ts=pinned_ts,
            thread_ts=(item.get("thread_ts") or pinned_ts),
        )
    except Exception as exc:
        logger.warning("[pin] slack thread fetch failed: %s", exc)
        messages = []

    if not messages:
        # Synthesise a placeholder so the ingest payload is still well-formed.
        messages = [
            SlackMessage(
                ts=pinned_ts,
                user=reactor,
                text=f"(pinned message {pinned_ts} content unavailable)",
            )
        ]

    ctx = PinContext(
        channel=channel,
        thread_ts=(item.get("thread_ts") or pinned_ts),
        pinned_ts=pinned_ts,
        pinned_by_user=reactor,
        messages=tuple(messages),
    )

    payload = build_ingest_payload(ctx)

    try:
        body = await _post_ingest(
            config.atlas_base_url,
            config.atlas_bearer,
            payload,
            httpx_module=config.httpx_module,
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.warning("[pin] /v1/ingest failed: %s", reason)
        await _safe_reply(reply_in_thread, channel, ctx.thread_ts, format_error_reply(reason))
        return None

    job_id = str(body.get("job_id") or "")
    if not job_id:
        await _safe_reply(
            reply_in_thread, channel, ctx.thread_ts,
            format_error_reply("Atlas accepted ingest but returned no job_id"),
        )
        return None

    await _safe_reply(reply_in_thread, channel, ctx.thread_ts, format_success_reply(job_id))
    logger.info(
        "[pin] ingest accepted job_id=%s channel=%s thread_ts=%s messages=%d",
        job_id, channel, ctx.thread_ts, len(ctx.messages),
    )
    return job_id


async def _safe_reply(
    reply_in_thread: Callable[[str, str, str], Awaitable[None]],
    channel: str,
    thread_ts: str,
    text: str,
) -> None:
    """Best-effort reply; logs (never raises) on transport failure."""
    try:
        await reply_in_thread(channel, thread_ts, text)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[pin] reply failed (channel=%s ts=%s): %s", channel, thread_ts, exc)


# ---------------------------------------------------------------------------
# Manual smoke-test instructions
# ---------------------------------------------------------------------------

MANUAL_SMOKE_TEST = """\
Manual smoke test — Plan 026-C 📌 pin handler
=============================================

Prerequisites
-------------
* Hermes gateway running with Slack Socket Mode connected.
* ``SLACK_ALLOWED_USERS`` includes Blake's Slack user ID (``U…``).
* ``ATLAS_BASE_URL`` and ``ATLAS_BEARER_TOKEN`` set; Atlas reachable.
* Slack app manifest grants ``reactions:read`` + ``conversations:history``
  + ``conversations:replies`` scopes (already in the app manifest as of
  Plan 004-A reaction capture).

Steps
-----
1. Post a test message in any channel where Hermes is present (e.g.
   ``#bossman2``): ``"This is a test pin for 026-C smoke."``
2. React 📌 to that message.
3. Within ~5s, Hermes should reply in-thread:
     ``✓ Pinned to Atlas: urn:atlas:ingest:<job_id>``
4. Wait ~60s. Query ``/v1/ask?question="test pin for 026-C smoke"`` —
   the answer should cite the pinned chunk.
5. Failure path: temporarily set ``ATLAS_BASE_URL`` to a bad host;
   re-pin; confirm Hermes replies with ``⚠ pin failed: …`` (not silent).
"""
