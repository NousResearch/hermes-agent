"""/daily slash command — Slack-native morning brief (Plan 026-A).

Fans out in parallel to five sources and synthesizes a 3-bullet brief
in Slack block-kit shape. Sources (all via ``asyncio.gather`` with a
10s wall-clock budget):

1. **Calendar** — today-onward events, derived from the Atlas
   ``GET /v1/today`` cockpit (``upcoming_meetings``). There is no live
   Google path in the deployed runtime (Plan 008's MCPGateway is a stub,
   no Google creds in Fargate), but Phase 1 ingests Calendar into the
   Atlas graph, so the brief reads them back via one SPARQL-backed
   round-trip — no LLM cost.

2. **Inbox** — most-recent email threads, derived from the SAME
   ``GET /v1/today`` payload (``overdue_emails``). Calendar + inbox share
   one fetch; the two ``_*_from_today`` derivers split it into two
   ``SourceResult``s.

3. **Atlas open commitments** — ``atlas_ask`` with
   ``intent_hint="commitment_audit"``. Goes through the existing
   in-process Atlas memory provider (plugins/memory/atlas) so we don't
   need a separate HTTP client; the provider already handles bearer +
   breaker.

4. **Atlas pending contradictions** — ``atlas_ask`` with
   ``intent_hint="contradiction_audit"``. Same provider path.

5. **Atlas contacts overdue** — ``atlas_ask`` with
   ``intent_hint="contacts_overdue"``. Same provider path.

6. **Orchestrator status** — HTTP ``GET
   $ORCHESTRATOR_BASE_URL/orchestrator/drain/status``. Plan 020-F
   already ships this surface. If the env var is unset or the request
   fails, the section is dropped from the brief with a degraded marker.

Auth model mirrors ``plugins/slash/draft.py`` / ``orchestrator.py``: the
Hermes gateway's ``SLACK_ALLOWED_USERS`` gate is enforced at the
platform layer, so by the time ``handle_daily`` runs we trust the caller.

The handler signature is ``handle_daily(raw_args: str) -> str`` per the
existing ``PluginContext.register_command`` contract. The string return
is a JSON-encoded Slack ``blocks`` payload (the gateway recognizes JSON
replies starting with ``{"blocks":...}`` and passes them through to the
Slack ``chat.postMessage`` ``blocks`` parameter). For non-Slack callers
(e.g. unit tests), the same JSON deserializes back into a plain dict.

Synthesis is intentionally **deterministic** in v1: the 3-bullet
ranker is a small, testable function that pulls "the most urgent /
highest-leverage one item per category" without an LLM call. The
Nova-Pro path is wired through ``daily_synthesizer.synthesize`` and
gated on ``HERMES_DAILY_USE_NOVA=true`` (default off) so the unit
tests and the 10s latency budget are not subject to model drift /
Bedrock cold-start. When 026-E's operator UAT exposes whether the
deterministic ranker is good enough on real data, we either keep it
or flip the env flag.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# 10s wall-clock budget for the entire fan-out (per master plan §"Design
# decisions" #2). Individual source timeouts are smaller so a single
# slow source can't eat the whole budget.
FANOUT_BUDGET_SECS = 10.0
PER_SOURCE_TIMEOUT_SECS = 8.0

# Atlas ask sub-queries. Tracked here (not in the synthesizer) so the
# test surface and the orchestrator fan-out share the same source-of-truth.
ATLAS_QUERIES: tuple[tuple[str, str, str], ...] = (
    # (key, question, intent_hint)
    (
        "commitments",
        "What commitments do I have open today?",
        "commitment_audit",
    ),
    (
        "contradictions",
        "What contradictions are open and unresolved?",
        "contradiction_audit",
    ),
    (
        "contacts_overdue",
        "Which contacts are overdue for follow-up?",
        "contacts_overdue",
    ),
)


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceResult:
    """One fan-out result.

    ``status`` is ``"ok"``, ``"empty"``, ``"timeout"``, or ``"error"``.
    ``items`` is a list of short human-readable strings (the per-source
    fetcher is responsible for trimming to ~5 items max — the synthesizer
    only ever surfaces 1 item per source in the top-line bullets, but the
    expandable section can show the rest).
    ``citations`` is a list of source IRIs / URLs the bullet can link to.
    """

    key: str
    status: str
    items: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class BriefBundle:
    """All five (or six) source results plus elapsed wall-clock."""

    calendar: SourceResult
    inbox: SourceResult
    commitments: SourceResult
    contradictions: SourceResult
    contacts_overdue: SourceResult
    orchestrator: SourceResult
    elapsed_secs: float

    @property
    def atlas_all_empty(self) -> bool:
        """True iff all three atlas_ask streams returned empty/error.

        Drives the ``⚠ Atlas corpus warming up`` footer per AC.
        """
        return all(
            r.status in ("empty", "error", "timeout")
            for r in (self.commitments, self.contradictions, self.contacts_overdue)
        )


# ---------------------------------------------------------------------------
# Source fetchers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Calendar + Inbox via the Atlas /v1/today cockpit
# ---------------------------------------------------------------------------
#
# There is NO live-Google path in the deployed Hermes runtime (Plan 008's
# MCPGateway is a stub, no Google credentials in Fargate). But Phase 1 now
# ingests Gmail + Calendar bodies into the Atlas graph, and Atlas exposes a
# purpose-built cockpit endpoint — ``GET /v1/today`` (army-of-one
# ask_routes:get_today) — that returns ``upcoming_meetings`` (today-onward
# CalendarEvents) and ``overdue_emails`` (most-recent EmailMessages) straight
# from the graph via SPARQL (no LLM cost). So calendar + inbox are derived
# from ONE round-trip to that endpoint, not from a separate live data source.
#
# Each list item is a TimelineEntry: {iri, kind, canonical_name, event_time,
# summary}. The two pure ``_*_from_today`` functions turn one payload into the
# two SourceResults; ``_default_today_fetch`` is the wired source.


def _short_when(event_time: Optional[str]) -> str:
    """Render an ISO-8601 ``event_time`` as a compact ``HH:MM`` for the bullet.

    Returns ``""`` if the value is missing or unparseable, so an untimed /
    all-day event degrades to name-only rather than showing a bogus time.
    """
    if not event_time:
        return ""
    raw = str(event_time).strip()
    try:
        from datetime import datetime

        # Normalise a trailing Z to an explicit offset for fromisoformat.
        normalised = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalised)
        return dt.strftime("%H:%M")
    except (ValueError, TypeError):
        return ""


def _entries_to_items(
    entries: list, *, with_time: bool
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Map a list of TimelineEntry dicts to (items, citations), capped at 5."""
    items: list[str] = []
    citations: list[str] = []
    for e in entries[:5]:
        if not isinstance(e, dict):
            continue
        name = str(e.get("canonical_name") or "").strip()
        if not name:
            continue
        if with_time:
            when = _short_when(e.get("event_time"))
            items.append(f"{when} — {name}" if when else name)
        else:
            items.append(name)
        iri = str(e.get("iri") or "").strip()
        if iri:
            citations.append(iri)
    return tuple(items), tuple(citations)


def _calendar_from_today(payload: dict) -> SourceResult:
    """Derive the calendar SourceResult from a ``/v1/today`` payload."""
    if not isinstance(payload, dict):
        return SourceResult(key="calendar", status="error", error="non-dict today response")
    if payload.get("error"):
        return SourceResult(key="calendar", status="error", error=str(payload["error"]))
    meetings = payload.get("upcoming_meetings") or []
    items, citations = _entries_to_items(meetings, with_time=True)
    status = "ok" if items else "empty"
    return SourceResult(key="calendar", status=status, items=items, citations=citations)


def _inbox_from_today(payload: dict) -> SourceResult:
    """Derive the inbox SourceResult from a ``/v1/today`` payload."""
    if not isinstance(payload, dict):
        return SourceResult(key="inbox", status="error", error="non-dict today response")
    if payload.get("error"):
        return SourceResult(key="inbox", status="error", error=str(payload["error"]))
    emails = payload.get("overdue_emails") or []
    items, citations = _entries_to_items(emails, with_time=False)
    status = "ok" if items else "empty"
    return SourceResult(key="inbox", status=status, items=items, citations=citations)


def _derive_calendar_inbox(
    today_task: "asyncio.Task[dict]",
) -> tuple[SourceResult, SourceResult]:
    """Map the bounded ``/v1/today`` task into (calendar, inbox) results.

    A cancelled task (fan-out budget exceeded) or a ``_status`` sentinel
    (per-source timeout / transport error from ``_bounded_today``) degrades
    BOTH sections identically — they share the one round-trip. Otherwise the
    pure ``_*_from_today`` derivers run (which themselves handle the
    breaker-open ``{"error": ...}`` payload and the normal case).
    """
    if today_task.cancelled():
        ts = SourceResult(key="calendar", status="timeout", error="fan-out budget exceeded")
        return ts, SourceResult(key="inbox", status="timeout", error="fan-out budget exceeded")
    try:
        payload = today_task.result()
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        return (
            SourceResult(key="calendar", status="error", error=err),
            SourceResult(key="inbox", status="error", error=err),
        )
    sentinel = payload.get("_status") if isinstance(payload, dict) else None
    if sentinel in ("timeout", "error"):
        err = str(payload.get("error") or sentinel)
        return (
            SourceResult(key="calendar", status=sentinel, error=err),
            SourceResult(key="inbox", status=sentinel, error=err),
        )
    return _calendar_from_today(payload), _inbox_from_today(payload)


async def _default_today_fetch() -> dict:
    """Fetch ``GET /v1/today`` through the in-process Atlas memory provider.

    Mirrors ``_default_atlas_ask``: the provider's ``_today`` is a synchronous
    httpx call, so we offload to a thread to keep the fan-out parallel. If the
    provider isn't loaded / Atlas is disabled, we return an empty payload so
    calendar + inbox degrade to "empty" rather than erroring. A breaker-open /
    transport failure surfaces as ``{"error": ...}`` so the sections honestly
    show the degradation.
    """
    try:
        from plugins.memory.atlas import AtlasMemoryProvider  # type: ignore
    except Exception as exc:
        logger.info("[daily] atlas memory provider unavailable: %s", exc)
        return {}

    provider = AtlasMemoryProvider()
    if not provider.is_available():
        return {}

    def _call() -> dict:
        try:
            return provider._today()
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash the brief
            return {"error": f"{type(exc).__name__}: {exc}"}

    return await asyncio.to_thread(_call)


def _summarize_atlas_response(payload: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Pull (items, citations) out of an ``atlas_ask`` JSON payload.

    Atlas's AskResponse shape (per army-of-one atlas/api/ask_routes.py):

      * ``answer``: str — the synthesized natural-language answer
      * ``citations``: list[{chunk_id, source, snippet, ...}] — typed cites

    For the brief we want one short bullet line per Atlas response. The
    ``answer`` is usually multi-sentence; we take the first sentence (or
    first 240 chars) as the bullet text. Citation IRIs become click-through
    links in the block-kit output.
    """
    answer = (payload.get("answer") or "").strip()
    items: tuple[str, ...] = ()
    if answer:
        # First sentence-ish.
        first_sentence = answer.split("\n", 1)[0].strip()
        if len(first_sentence) > 240:
            first_sentence = first_sentence[:237] + "..."
        items = (first_sentence,) if first_sentence else ()

    cites_raw = payload.get("citations") or []
    citations: list[str] = []
    for c in cites_raw[:5]:  # cap at 5 per source for block-kit sanity
        if not isinstance(c, dict):
            continue
        # Prefer an explicit IRI / URN; fall back to chunk_id wrapped in
        # the canonical urn:atlas:chunk:... shape.
        cid = c.get("iri") or c.get("urn") or c.get("chunk_id") or ""
        cid = str(cid).strip()
        if not cid:
            continue
        if not cid.startswith("urn:") and not cid.startswith("http"):
            cid = f"urn:atlas:chunk:{cid}"
        citations.append(cid)
    return items, tuple(citations)


async def _fetch_atlas(
    key: str,
    question: str,
    intent_hint: str,
    *,
    atlas_ask: Callable[[str, str], Awaitable[dict]],
) -> SourceResult:
    """One ``atlas_ask`` invocation, wrapped in a SourceResult.

    ``atlas_ask`` is injected so tests can substitute a fake without
    monkeypatching the memory-provider import system. The real wire-up
    (in ``_default_atlas_ask``) goes through the in-process Atlas
    memory provider.
    """
    try:
        payload = await atlas_ask(question, intent_hint)
    except asyncio.TimeoutError:
        return SourceResult(key=key, status="timeout", error="atlas_ask timed out")
    except Exception as exc:  # pragma: no cover - exercised via fakes
        return SourceResult(key=key, status="error", error=f"{type(exc).__name__}: {exc}")

    if not isinstance(payload, dict):
        return SourceResult(key=key, status="error", error="non-dict atlas response")

    # Atlas can return {"error": ...} on breaker-open / 5xx.
    if payload.get("error"):
        return SourceResult(key=key, status="error", error=str(payload["error"]))

    items, citations = _summarize_atlas_response(payload)
    status = "ok" if items else "empty"
    return SourceResult(key=key, status=status, items=items, citations=citations)


async def _fetch_orchestrator_status(
    base_url: str,
    *,
    httpx_module: Any = None,
) -> SourceResult:
    """``GET $ORCHESTRATOR_BASE_URL/orchestrator/drain/status``.

    Returns one bullet summarising in-flight ``drainTierGraph``
    workflows + escalation count. If the env var is unset, returns
    ``empty`` (not ``error``) so the brief doesn't lie about an outage
    when the surface simply isn't configured in this environment.
    """
    if not base_url:
        return SourceResult(
            key="orchestrator",
            status="empty",
            error="ORCHESTRATOR_BASE_URL not set",
        )

    if httpx_module is None:  # pragma: no cover - trivial
        import httpx as _httpx
        httpx_module = _httpx

    url = f"{base_url.rstrip('/')}/orchestrator/drain/status"
    try:
        async with httpx_module.AsyncClient(timeout=PER_SOURCE_TIMEOUT_SECS) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            body = resp.json()
    except asyncio.TimeoutError:
        return SourceResult(key="orchestrator", status="timeout", error="GET drain/status timed out")
    except Exception as exc:
        return SourceResult(
            key="orchestrator",
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )

    in_flight = body.get("in_flight") or body.get("active") or []
    escalations = int(body.get("escalations") or 0)
    if not in_flight and escalations == 0:
        return SourceResult(key="orchestrator", status="empty")

    items: list[str] = []
    if in_flight:
        items.append(f"{len(in_flight)} drainTierGraph workflows in flight")
    if escalations:
        items.append(f"{escalations} phase(s) escalated for review")
    return SourceResult(
        key="orchestrator",
        status="ok",
        items=tuple(items),
        citations=(f"{url}",),
    )


# ---------------------------------------------------------------------------
# Atlas ask wiring (default — uses the in-process memory provider)
# ---------------------------------------------------------------------------


async def _default_atlas_ask(question: str, intent_hint: str) -> dict:
    """Default ``atlas_ask`` shim that goes through the in-process Atlas
    memory provider.

    The provider's ``_ask`` is synchronous (it uses requests-style httpx
    blocking calls), so we offload to a thread to keep the fan-out fully
    parallel. If the provider isn't loaded (Atlas memory disabled), we
    return an empty-shaped response rather than raising — the brief will
    show ``⚠ corpus warming up`` and degrade to Calendar+Gmail only.
    """
    try:
        from plugins.memory.atlas import AtlasMemoryProvider  # type: ignore
    except Exception as exc:
        logger.info("[daily] atlas memory provider unavailable: %s", exc)
        return {"answer": "", "citations": []}

    provider = AtlasMemoryProvider()
    if not provider.is_available():
        return {"answer": "", "citations": []}

    def _call() -> dict:
        return provider._ask(
            question=question,
            intent_hint=intent_hint,
        )

    return await asyncio.to_thread(_call)


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


async def _gather_brief(
    *,
    atlas_ask: Callable[[str, str], Awaitable[dict]] | None = None,
    today_fetcher: Callable[[], Awaitable[dict]] | None = None,
    orchestrator_base_url: str | None = None,
    httpx_module: Any = None,
) -> BriefBundle:
    """Run all fan-outs in parallel with a shared 10s budget.

    Calendar + inbox share a single ``/v1/today`` round-trip (the
    ``today`` task): one fetch, two derived ``SourceResult``s. The three
    Atlas ``ask`` streams and the orchestrator status run as their own
    tasks. Each task is wrapped in ``asyncio.wait_for(PER_SOURCE_TIMEOUT_SECS)``
    so a single slow source can't starve the others, and the outer
    ``asyncio.wait_for`` honours the overall ``FANOUT_BUDGET_SECS``. Any
    task still pending at budget expiry is cancelled and marked
    ``timeout`` — for ``today`` that degrades BOTH calendar and inbox.
    """
    atlas_ask = atlas_ask or _default_atlas_ask
    today_fetcher = today_fetcher or _default_today_fetch
    if orchestrator_base_url is None:
        orchestrator_base_url = os.getenv("ORCHESTRATOR_BASE_URL", "")

    start = time.monotonic()

    async def _bounded(coro: Awaitable[SourceResult], key: str) -> SourceResult:
        try:
            return await asyncio.wait_for(coro, timeout=PER_SOURCE_TIMEOUT_SECS)
        except asyncio.TimeoutError:
            return SourceResult(key=key, status="timeout", error="per-source timeout")
        except Exception as exc:
            return SourceResult(key=key, status="error", error=f"{type(exc).__name__}: {exc}")

    async def _bounded_today() -> dict:
        """The /v1/today fetch as a dict-returning bounded task.

        Returns a sentinel ``{"_status": ...}`` on timeout/error so the
        calendar/inbox derivation can map it to per-section status.
        """
        try:
            return await asyncio.wait_for(today_fetcher(), timeout=PER_SOURCE_TIMEOUT_SECS)
        except asyncio.TimeoutError:
            return {"_status": "timeout", "error": "per-source timeout"}
        except Exception as exc:
            return {"_status": "error", "error": f"{type(exc).__name__}: {exc}"}

    today_task = asyncio.create_task(_bounded_today())
    tasks: dict[str, asyncio.Task[SourceResult]] = {
        "orchestrator": asyncio.create_task(
            _bounded(
                _fetch_orchestrator_status(orchestrator_base_url, httpx_module=httpx_module),
                "orchestrator",
            )
        ),
    }
    for key, question, hint in ATLAS_QUERIES:
        tasks[key] = asyncio.create_task(
            _bounded(_fetch_atlas(key, question, hint, atlas_ask=atlas_ask), key)
        )

    all_tasks = [today_task, *tasks.values()]
    try:
        await asyncio.wait_for(
            asyncio.gather(*all_tasks, return_exceptions=True),
            timeout=FANOUT_BUDGET_SECS,
        )
    except asyncio.TimeoutError:
        # Budget exceeded — cancel anything still pending. Each task
        # cancelled this way will surface as ``timeout`` below.
        for t in all_tasks:
            if not t.done():
                t.cancel()
        # Drain cancellations so the event loop isn't holding refs.
        await asyncio.gather(*all_tasks, return_exceptions=True)

    # Derive calendar + inbox from the single /v1/today result.
    calendar, inbox = _derive_calendar_inbox(today_task)

    results: dict[str, SourceResult] = {}
    for key, task in tasks.items():
        if task.cancelled():
            results[key] = SourceResult(key=key, status="timeout", error="fan-out budget exceeded")
            continue
        try:
            results[key] = task.result()
        except Exception as exc:
            results[key] = SourceResult(key=key, status="error", error=f"{type(exc).__name__}: {exc}")

    elapsed = time.monotonic() - start
    return BriefBundle(
        calendar=calendar,
        inbox=inbox,
        commitments=results.get("commitments", SourceResult(key="commitments", status="error", error="missing")),
        contradictions=results.get("contradictions", SourceResult(key="contradictions", status="error", error="missing")),
        contacts_overdue=results.get("contacts_overdue", SourceResult(key="contacts_overdue", status="error", error="missing")),
        orchestrator=results.get("orchestrator", SourceResult(key="orchestrator", status="error", error="missing")),
        elapsed_secs=elapsed,
    )


# ---------------------------------------------------------------------------
# Synthesizer (deterministic 3-bullet ranker)
# ---------------------------------------------------------------------------


# Source priority order for the deterministic top-line ranker. The first
# 3 sources that have items become the 3 bullets. This deliberately
# keeps the more time-sensitive / leverage-y sources first.
_RANK_ORDER: tuple[str, ...] = (
    "commitments",       # Atlas-derived; most leverage-y
    "calendar",          # time-bound
    "inbox",             # action items in flight
    "contradictions",    # graph hygiene
    "contacts_overdue",  # relationship maintenance
    "orchestrator",      # background drain status
)


def _bullet_label(key: str) -> str:
    """Human-readable section label for the block-kit output."""
    return {
        "calendar": "📅 Calendar",
        "inbox": "📥 Inbox",
        "commitments": "✅ Commitments",
        "contradictions": "⚠ Contradictions",
        "contacts_overdue": "👥 Contacts overdue",
        "orchestrator": "🛠 Orchestrator",
    }.get(key, key)


def synthesize_bullets(bundle: BriefBundle) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """Pick up to 3 top-line bullets from the bundle.

    Returns a tuple of ``(source_key, text, citations)``. The text always
    starts with the source label so the Slack reader knows what they're
    looking at without expanding the section.

    Deterministic v1 algorithm:
      1. Walk sources in ``_RANK_ORDER``.
      2. For each source with at least one item AND status=="ok", take
         the first item as the bullet text.
      3. Stop after 3 bullets.
      4. If fewer than 3 sources have items, the bullet list is short —
         we don't pad with "nothing to report" filler.

    This is the path that runs in unit tests and as the latency-budget
    safe default. The optional Nova-Pro path (gated on
    ``HERMES_DAILY_USE_NOVA=true``) is in ``daily_synthesizer.synthesize``
    and not exercised in this phase's tests.
    """
    sources_by_key = {
        "calendar": bundle.calendar,
        "inbox": bundle.inbox,
        "commitments": bundle.commitments,
        "contradictions": bundle.contradictions,
        "contacts_overdue": bundle.contacts_overdue,
        "orchestrator": bundle.orchestrator,
    }
    bullets: list[tuple[str, str, tuple[str, ...]]] = []
    for key in _RANK_ORDER:
        if len(bullets) >= 3:
            break
        sr = sources_by_key[key]
        if sr.status != "ok" or not sr.items:
            continue
        label = _bullet_label(key)
        text = f"{label} — {sr.items[0]}"
        bullets.append((key, text, sr.citations))
    return tuple(bullets)


# ---------------------------------------------------------------------------
# Block-kit builder
# ---------------------------------------------------------------------------


def _citation_links_md(citations: tuple[str, ...]) -> str:
    """Render citations as Slack-mrkdwn link list.

    ``urn:atlas:chunk:foo`` is rendered verbatim (Slack doesn't make
    URNs clickable but Blake can copy-paste). HTTP(S) URLs become real
    ``<url|short>`` mrkdwn links.
    """
    if not citations:
        return ""
    parts: list[str] = []
    for c in citations[:3]:  # cap visible cites at 3 in top-line bullet
        if c.startswith("http"):
            parts.append(f"<{c}|src>")
        else:
            parts.append(f"`{c}`")
    return " " + " ".join(parts)


def build_blocks(bundle: BriefBundle) -> dict:
    """Build the Slack block-kit JSON payload.

    Shape:

      * header — "Daily brief — YYYY-MM-DD"
      * section: 3 top-line bullets (always visible)
      * divider
      * one section per source with status + items (expandable in iOS
        via Slack's auto-collapse on long messages; on desktop they all
        render inline)
      * footer context — elapsed time + ⚠ corpus-warming-up if applies

    Returns ``{"blocks": [...], "text": <fallback>}``. ``text`` is the
    plain-text fallback Slack uses for desktop notifications and
    accessibility readers.
    """
    bullets = synthesize_bullets(bundle)

    blocks: list[dict] = []

    # Header
    blocks.append({
        "type": "header",
        "text": {"type": "plain_text", "text": "Daily brief"},
    })

    # Top-line bullets
    if bullets:
        lines: list[str] = []
        for _key, text, citations in bullets:
            lines.append(f"• {text}{_citation_links_md(citations)}")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(lines)},
        })
    else:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "_No actionable items found across any source._",
            },
        })

    blocks.append({"type": "divider"})

    # Per-source expandable sections
    source_order = (
        ("calendar", bundle.calendar),
        ("inbox", bundle.inbox),
        ("commitments", bundle.commitments),
        ("contradictions", bundle.contradictions),
        ("contacts_overdue", bundle.contacts_overdue),
        ("orchestrator", bundle.orchestrator),
    )
    for key, sr in source_order:
        label = _bullet_label(key)
        if sr.status == "ok" and sr.items:
            body = "\n".join(f"  · {item}" for item in sr.items[:5])
            if sr.citations:
                body += "\n  " + " ".join(
                    f"`{c}`" if not c.startswith("http") else f"<{c}|src>"
                    for c in sr.citations[:5]
                )
        elif sr.status == "timeout":
            body = "  · _timeout — source exceeded fan-out budget_"
        elif sr.status == "error":
            body = f"  · _error: {sr.error or 'unknown'}_"
        else:  # empty
            body = (
                f"  · _empty{f' ({sr.error})' if sr.error else ''}_"
            )
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{label}*\n{body}"},
        })

    # Footer
    footer_parts: list[str] = []
    footer_parts.append(f"fan-out: {bundle.elapsed_secs:.2f}s")
    if bundle.atlas_all_empty:
        footer_parts.append("⚠ Atlas corpus warming up — degraded brief")
    blocks.append({
        "type": "context",
        "elements": [
            {"type": "mrkdwn", "text": " · ".join(footer_parts)},
        ],
    })

    # Plain-text fallback for notifications.
    fallback_lines = ["Daily brief"]
    for _k, text, _c in bullets:
        fallback_lines.append(f"• {text}")
    if not bullets:
        fallback_lines.append("(no items)")
    fallback = "\n".join(fallback_lines)

    return {"blocks": blocks, "text": fallback}


# ---------------------------------------------------------------------------
# Top-level handler
# ---------------------------------------------------------------------------


@dataclass
class DailyHandlerConfig:
    """Inputs the gateway / cron can inject for testing.

    All fields default to ``None`` → use module-level defaults (real
    Atlas provider, env-var orchestrator URL, real httpx). Tests
    construct one with fakes to exercise the timeout / partial-failure
    paths without network.
    """

    atlas_ask: Optional[Callable[[str, str], Awaitable[dict]]] = None
    today_fetcher: Optional[Callable[[], Awaitable[dict]]] = None
    orchestrator_base_url: Optional[str] = None
    httpx_module: Any = None


async def build_daily_brief(config: Optional[DailyHandlerConfig] = None) -> dict:
    """Async entry-point — runs the fan-out and returns the block-kit dict.

    Exposed so 026-D's cron job can ``await build_daily_brief()``
    directly without going through the sync slash-command shim.
    """
    cfg = config or DailyHandlerConfig()
    bundle = await _gather_brief(
        atlas_ask=cfg.atlas_ask,
        today_fetcher=cfg.today_fetcher,
        orchestrator_base_url=cfg.orchestrator_base_url,
        httpx_module=cfg.httpx_module,
    )
    return build_blocks(bundle)


def handle_daily(raw_args: str) -> str:
    """``/daily`` slash command entry point — sync per plugin contract.

    Returns a JSON-encoded block-kit payload. The Slack gateway
    recognises JSON replies starting with ``{"blocks"`` and forwards the
    parsed structure to ``chat.postMessage`` ``blocks`` + ``text``
    params. For plain-text Slack clients (or unit tests asserting on the
    string), ``json.loads`` round-trips back to the same dict.

    ``raw_args`` is accepted but ignored in v1 — the brief has no
    parameters. Future surface may accept ``/daily yesterday`` /
    ``/daily compact``; not in scope for 026-A.
    """
    logger.info("daily.invoked raw_args_len=%d", len(raw_args or ""))

    # Detect "am I inside a running event loop?" without creating an
    # orphan coroutine. ``asyncio.get_running_loop`` raises RuntimeError
    # iff no loop is running on this thread.
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False

    try:
        if in_loop:
            payload = _run_in_thread(build_daily_brief())
        else:
            payload = asyncio.run(build_daily_brief())
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("[daily] fan-out crashed")
        payload = _error_payload(f"{type(exc).__name__}: {exc}")

    return json.dumps(payload)


def _run_in_thread(coro: Awaitable[dict]) -> dict:
    """Run an awaitable on a fresh event loop in a worker thread.

    Used when ``handle_daily`` is invoked from inside an already-running
    asyncio event loop (e.g. the Slack platform adapter). Keeps the
    slash-command signature sync while still letting the fan-out be
    concurrent.
    """
    import concurrent.futures

    def _runner() -> dict:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_runner).result(timeout=FANOUT_BUDGET_SECS + 2.0)


def _error_payload(reason: str) -> dict:
    """Render a Slack reply when the whole fan-out fails catastrophically.

    Per ACs we want a visible failure, not a silent drop.
    """
    return {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"⚠ /daily failed: `{reason}`",
                },
            }
        ],
        "text": f"/daily failed: {reason}",
    }
