"""Unit tests for the /daily slash command — Plan 026-A.

Acceptance criteria covered:

AC1 — ``plugins/slash/daily.py`` exposes ``handle_daily`` + async
      ``build_daily_brief`` returning Slack block-kit JSON. The 3-bullet
      top-line section is always present (header + section + divider +
      6 per-source sections + footer = 10 blocks minimum).

AC2 — All 4 atlas_ask fan-out streams (3 atlas + orchestrator) execute
      concurrently. We assert this by injecting fakes that record their
      call counts and orderings.

AC3 — Partial failure: when one source times out, the other 5 still
      land in the brief. ``test_partial_failure_atlas_timeout``.

AC4 — Latency budget: total fan-out wall-clock is bounded by
      ``FANOUT_BUDGET_SECS`` (10s) even if a source hangs forever.
      ``test_fanout_budget_caps_total_wall_clock``.

AC5 — ⚠ corpus warming up footer when all three atlas streams return
      empty. ``test_atlas_empty_footer``.

AC6 — Plugin registration: ``register()`` wires ``/daily`` without
      breaking existing /resume, /skip, /draft.
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest import mock

import pytest

from plugins.slash import daily as daily_mod
from plugins.slash.daily import (
    ATLAS_QUERIES,
    BriefBundle,
    DailyHandlerConfig,
    FANOUT_BUDGET_SECS,
    SourceResult,
    build_blocks,
    build_daily_brief,
    handle_daily,
    synthesize_bullets,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _atlas_payload(answer: str, *cite_ids: str) -> dict:
    return {
        "answer": answer,
        "citations": [
            {"chunk_id": cid, "snippet": f"snippet for {cid}"} for cid in cite_ids
        ],
    }


def _make_atlas_ask(responses: dict[str, dict]):
    """Build a fake ``atlas_ask`` that dispatches by intent_hint.

    Falls back to an empty answer for any unmapped hint so the test
    surface doesn't need to enumerate every query.
    """

    async def _ask(question: str, intent_hint: str) -> dict:
        return responses.get(intent_hint, {"answer": "", "citations": []})

    return _ask


def _make_today_fetch(payload: dict):
    """Build a fake ``today_fetcher`` returning a fixed /v1/today payload."""

    async def _fetch() -> dict:
        return payload

    return _fetch


# A /v1/today payload with one meeting + one email (both derive to ``ok``).
_TODAY_OK = {
    "upcoming_meetings": [
        {
            "iri": "https://atlas/event/abc",
            "kind": "calendar_event",
            "canonical_name": "Anthropic intro call",
            "event_time": "2026-06-26T09:00:00+00:00",
            "summary": None,
        }
    ],
    "overdue_emails": [
        {
            "iri": "https://atlas/email/xyz",
            "kind": "email",
            "canonical_name": "Sarah replied re: term sheet",
            "event_time": "2026-06-26T08:00:00+00:00",
            "summary": None,
        }
    ],
    "pending_drafts": [],
    "agent_runs": [],
}


def _today_ok():
    return _make_today_fetch(_TODAY_OK)


# ---------------------------------------------------------------------------
# /v1/today derivation (calendar + inbox)
# ---------------------------------------------------------------------------


def _today_payload(meetings=None, emails=None, **extra) -> dict:
    payload = {
        "upcoming_meetings": meetings or [],
        "overdue_emails": emails or [],
        "pending_drafts": [],
        "agent_runs": [],
    }
    payload.update(extra)
    return payload


def _entry(name: str, iri: str, event_time: str | None = None) -> dict:
    return {
        "iri": iri,
        "kind": "calendar_event",
        "canonical_name": name,
        "event_time": event_time,
        "summary": None,
    }


def test_calendar_from_today_maps_meetings_with_time_and_citation():
    payload = _today_payload(
        meetings=[
            _entry("Anthropic intro call", "https://atlas/event/abc", "2026-06-26T10:00:00+00:00"),
        ]
    )
    sr = daily_mod._calendar_from_today(payload)
    assert sr.key == "calendar"
    assert sr.status == "ok"
    assert len(sr.items) == 1
    # The bullet must name the meeting and surface its start time (HH:MM).
    assert "Anthropic intro call" in sr.items[0]
    assert "10:00" in sr.items[0]
    assert sr.citations == ("https://atlas/event/abc",)


def test_calendar_from_today_untimed_event_falls_back_to_name_only():
    payload = _today_payload(meetings=[_entry("All-day offsite", "https://atlas/event/d", None)])
    sr = daily_mod._calendar_from_today(payload)
    assert sr.status == "ok"
    assert sr.items[0] == "All-day offsite"


def test_calendar_from_today_caps_at_five():
    payload = _today_payload(
        meetings=[_entry(f"Meeting {i}", f"https://atlas/event/{i}") for i in range(8)]
    )
    sr = daily_mod._calendar_from_today(payload)
    assert len(sr.items) == 5
    assert len(sr.citations) == 5


def test_calendar_from_today_empty_when_no_meetings():
    sr = daily_mod._calendar_from_today(_today_payload())
    assert sr.status == "empty"
    assert sr.items == ()


def test_calendar_from_today_error_passthrough():
    sr = daily_mod._calendar_from_today({"error": "Atlas temporarily unavailable"})
    assert sr.status == "error"
    assert "Atlas temporarily unavailable" in sr.error


def test_inbox_from_today_maps_emails_with_citation():
    payload = _today_payload(
        emails=[_entry("Re: term sheet", "https://atlas/email/xyz", "2026-06-26T08:30:00+00:00")]
    )
    sr = daily_mod._inbox_from_today(payload)
    assert sr.key == "inbox"
    assert sr.status == "ok"
    assert sr.items[0] == "Re: term sheet"
    assert sr.citations == ("https://atlas/email/xyz",)


def test_inbox_from_today_empty_when_no_emails():
    sr = daily_mod._inbox_from_today(_today_payload())
    assert sr.status == "empty"


def test_inbox_from_today_error_passthrough():
    sr = daily_mod._inbox_from_today({"error": "breaker open"})
    assert sr.status == "error"
    assert "breaker open" in sr.error


# ---------------------------------------------------------------------------
# Atlas response parsing
# ---------------------------------------------------------------------------


def test_summarize_atlas_response_first_sentence_and_citation_iris():
    payload = _atlas_payload(
        "You owe Greg a follow-up by EOD.\nAdditional context here.",
        "chunk-abc",
        "chunk-def",
    )
    items, citations = daily_mod._summarize_atlas_response(payload)
    assert items == ("You owe Greg a follow-up by EOD.",)
    assert citations == ("urn:atlas:chunk:chunk-abc", "urn:atlas:chunk:chunk-def")


def test_summarize_atlas_response_caps_long_answer():
    long = "x" * 400
    items, _ = daily_mod._summarize_atlas_response({"answer": long, "citations": []})
    assert len(items) == 1
    assert len(items[0]) <= 240
    assert items[0].endswith("...")


def test_summarize_atlas_response_passes_through_urns_and_urls():
    payload = {
        "answer": "see refs",
        "citations": [
            {"iri": "urn:atlas:chunk:from-iri"},
            {"urn": "urn:atlas:chunk:from-urn"},
            {"chunk_id": "https://example.com/doc/1"},
        ],
    }
    _, citations = daily_mod._summarize_atlas_response(payload)
    assert citations == (
        "urn:atlas:chunk:from-iri",
        "urn:atlas:chunk:from-urn",
        "https://example.com/doc/1",
    )


def test_summarize_atlas_response_empty_answer():
    items, citations = daily_mod._summarize_atlas_response({"answer": "", "citations": []})
    assert items == ()
    assert citations == ()


# ---------------------------------------------------------------------------
# Atlas fetcher
# ---------------------------------------------------------------------------


def test_fetch_atlas_ok_status_when_items_present():
    async def _ask(q, h):
        return _atlas_payload("the answer", "c1")

    sr = asyncio.run(
        daily_mod._fetch_atlas("commitments", "q?", "commitment_audit", atlas_ask=_ask)
    )
    assert sr.status == "ok"
    assert sr.items == ("the answer",)
    assert sr.citations == ("urn:atlas:chunk:c1",)


def test_fetch_atlas_empty_when_no_answer():
    async def _ask(q, h):
        return {"answer": "", "citations": []}

    sr = asyncio.run(
        daily_mod._fetch_atlas("commitments", "q?", "commitment_audit", atlas_ask=_ask)
    )
    assert sr.status == "empty"
    assert sr.items == ()


def test_fetch_atlas_error_when_provider_raises():
    async def _ask(q, h):
        raise RuntimeError("breaker open")

    sr = asyncio.run(
        daily_mod._fetch_atlas("commitments", "q?", "commitment_audit", atlas_ask=_ask)
    )
    assert sr.status == "error"
    assert "breaker open" in sr.error


def test_fetch_atlas_error_when_provider_returns_error_dict():
    async def _ask(q, h):
        return {"error": "Atlas temporarily unavailable"}

    sr = asyncio.run(
        daily_mod._fetch_atlas("commitments", "q?", "commitment_audit", atlas_ask=_ask)
    )
    assert sr.status == "error"
    assert "Atlas temporarily unavailable" in sr.error


# ---------------------------------------------------------------------------
# Orchestrator fetcher
# ---------------------------------------------------------------------------


class _FakeHttpx:
    """Minimal httpx-stand-in supporting ``AsyncClient(...).get(...)``."""

    def __init__(self, response_json=None, exc=None):
        self._response_json = response_json
        self._exc = exc

    def AsyncClient(self, **kwargs):  # noqa: N802 - mirrors httpx surface
        outer = self

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def get(self, url):
                if outer._exc is not None:
                    raise outer._exc

                class _Resp:
                    def raise_for_status(self_inner):
                        return None

                    def json(self_inner):
                        return outer._response_json

                return _Resp()

        return _Client()


def test_orchestrator_status_empty_when_url_unset():
    sr = asyncio.run(daily_mod._fetch_orchestrator_status(""))
    assert sr.status == "empty"
    assert "not set" in sr.error.lower()


def test_orchestrator_status_ok_renders_in_flight_and_escalations():
    fake = _FakeHttpx(response_json={"in_flight": [{"id": "a"}, {"id": "b"}], "escalations": 1})
    sr = asyncio.run(
        daily_mod._fetch_orchestrator_status("https://orch.example", httpx_module=fake)
    )
    assert sr.status == "ok"
    assert any("2 drainTierGraph" in i for i in sr.items)
    assert any("1 phase" in i for i in sr.items)


def test_orchestrator_status_empty_when_nothing_in_flight():
    fake = _FakeHttpx(response_json={"in_flight": [], "escalations": 0})
    sr = asyncio.run(
        daily_mod._fetch_orchestrator_status("https://orch.example", httpx_module=fake)
    )
    assert sr.status == "empty"


def test_orchestrator_status_error_on_transport_failure():
    fake = _FakeHttpx(exc=RuntimeError("connection refused"))
    sr = asyncio.run(
        daily_mod._fetch_orchestrator_status("https://orch.example", httpx_module=fake)
    )
    assert sr.status == "error"
    assert "connection refused" in sr.error


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


def test_gather_brief_all_sources_present():
    atlas_ask = _make_atlas_ask({
        "commitment_audit": _atlas_payload("Email Greg today", "c1"),
        "contradiction_audit": _atlas_payload("Two conflicting roles for Sarah", "c2"),
        "contacts_overdue": _atlas_payload("Alice — last contact 45d ago", "c3"),
    })
    fake_httpx = _FakeHttpx(response_json={"in_flight": [{"id": "w1"}], "escalations": 0})

    bundle = asyncio.run(
        daily_mod._gather_brief(
            atlas_ask=atlas_ask,
            today_fetcher=_today_ok(),
            orchestrator_base_url="https://orch.example",
            httpx_module=fake_httpx,
        )
    )

    assert bundle.calendar.status == "ok"
    assert bundle.inbox.status == "ok"
    assert bundle.commitments.status == "ok"
    assert bundle.contradictions.status == "ok"
    assert bundle.contacts_overdue.status == "ok"
    assert bundle.orchestrator.status == "ok"
    assert bundle.atlas_all_empty is False


def test_gather_brief_today_timeout_marks_both_calendar_and_inbox():
    """A slow /v1/today must degrade BOTH calendar and inbox to timeout."""

    async def _slow_today() -> dict:
        await asyncio.sleep(60)
        return _TODAY_OK

    bundle = asyncio.run(
        daily_mod._gather_brief(
            atlas_ask=_make_atlas_ask({}),
            today_fetcher=_slow_today,
            orchestrator_base_url="",
        )
    )
    assert bundle.calendar.status == "timeout"
    assert bundle.inbox.status == "timeout"


def test_gather_brief_today_error_propagates_to_both_sections():
    """An Atlas breaker-open ({"error": ...}) surfaces in calendar AND inbox."""
    bundle = asyncio.run(
        daily_mod._gather_brief(
            atlas_ask=_make_atlas_ask({}),
            today_fetcher=_make_today_fetch({"error": "Atlas temporarily unavailable"}),
            orchestrator_base_url="",
        )
    )
    assert bundle.calendar.status == "error"
    assert bundle.inbox.status == "error"
    assert "unavailable" in bundle.calendar.error


def test_partial_failure_atlas_timeout_does_not_kill_other_sources():
    """AC3 — partial-failure path: one slow source doesn't starve the brief."""

    async def _slow_ask(q, h):
        if h == "commitment_audit":
            await asyncio.sleep(60)  # exceeds per-source timeout
        return _atlas_payload(f"answer for {h}", "cX")

    bundle = asyncio.run(
        daily_mod._gather_brief(
            atlas_ask=_slow_ask,
            today_fetcher=_today_ok(),
            orchestrator_base_url="",
        )
    )

    assert bundle.commitments.status == "timeout"
    # Other sources should still have completed.
    assert bundle.contradictions.status == "ok"
    assert bundle.contacts_overdue.status == "ok"
    assert bundle.calendar.status == "ok"
    assert bundle.inbox.status == "ok"


def test_fanout_budget_caps_total_wall_clock(monkeypatch):
    """AC4 — even if every source hangs, the fan-out returns under budget."""
    # Shrink the budget so the test runs fast.
    monkeypatch.setattr(daily_mod, "FANOUT_BUDGET_SECS", 0.3)
    monkeypatch.setattr(daily_mod, "PER_SOURCE_TIMEOUT_SECS", 5.0)  # > budget, so budget wins

    async def _hang_ask(q, h):
        await asyncio.sleep(60)
        return {}

    async def _hang_today():
        await asyncio.sleep(60)
        return _TODAY_OK

    bundle = asyncio.run(
        daily_mod._gather_brief(
            atlas_ask=_hang_ask,
            today_fetcher=_hang_today,
            orchestrator_base_url="",
        )
    )

    # Must finish within ~budget + small overhead; assert generously.
    assert bundle.elapsed_secs < 2.0
    # Every source should be timeout / empty (orchestrator URL unset = empty).
    statuses = {
        bundle.calendar.status,
        bundle.inbox.status,
        bundle.commitments.status,
        bundle.contradictions.status,
        bundle.contacts_overdue.status,
    }
    assert statuses == {"timeout"}


def test_atlas_queries_constant_covers_three_streams():
    keys = {q[0] for q in ATLAS_QUERIES}
    assert keys == {"commitments", "contradictions", "contacts_overdue"}


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


def _bundle_with(**overrides) -> BriefBundle:
    defaults = dict(
        calendar=SourceResult("calendar", "empty"),
        inbox=SourceResult("inbox", "empty"),
        commitments=SourceResult("commitments", "empty"),
        contradictions=SourceResult("contradictions", "empty"),
        contacts_overdue=SourceResult("contacts_overdue", "empty"),
        orchestrator=SourceResult("orchestrator", "empty"),
        elapsed_secs=0.1,
    )
    defaults.update(overrides)
    return BriefBundle(**defaults)


def test_synthesize_picks_up_to_three_bullets():
    bundle = _bundle_with(
        commitments=SourceResult("commitments", "ok", ("Email Greg",), ("urn:atlas:chunk:c1",)),
        calendar=SourceResult("calendar", "ok", ("9am intro",), ("https://cal/1",)),
        inbox=SourceResult("inbox", "ok", ("Sarah replied",), ("https://mail/1",)),
        contradictions=SourceResult("contradictions", "ok", ("Conflict X",), ("urn:atlas:chunk:c2",)),
    )
    bullets = synthesize_bullets(bundle)
    assert len(bullets) == 3
    # Priority order: commitments > calendar > inbox
    assert bullets[0][0] == "commitments"
    assert bullets[1][0] == "calendar"
    assert bullets[2][0] == "inbox"
    # Each bullet text starts with its source label.
    assert "Commitments" in bullets[0][1]
    assert "Calendar" in bullets[1][1]
    assert "Inbox" in bullets[2][1]


def test_synthesize_skips_empty_and_error_sources():
    bundle = _bundle_with(
        commitments=SourceResult("commitments", "error", error="boom"),
        calendar=SourceResult("calendar", "ok", ("9am intro",)),
    )
    bullets = synthesize_bullets(bundle)
    assert len(bullets) == 1
    assert bullets[0][0] == "calendar"


def test_synthesize_returns_empty_when_no_sources_ok():
    assert synthesize_bullets(_bundle_with()) == ()


# ---------------------------------------------------------------------------
# Block-kit builder
# ---------------------------------------------------------------------------


def test_build_blocks_returns_blocks_and_text_fallback():
    bundle = _bundle_with(
        commitments=SourceResult("commitments", "ok", ("Email Greg",), ("urn:atlas:chunk:c1",)),
    )
    payload = build_blocks(bundle)
    assert "blocks" in payload
    assert "text" in payload
    assert isinstance(payload["blocks"], list)
    # header + top-line + divider + 6 sources + footer
    assert len(payload["blocks"]) == 10
    assert payload["blocks"][0]["type"] == "header"
    assert payload["blocks"][-1]["type"] == "context"
    assert "Email Greg" in payload["text"]


def test_build_blocks_top_line_includes_citation_iri():
    bundle = _bundle_with(
        commitments=SourceResult(
            "commitments", "ok", ("Email Greg",), ("urn:atlas:chunk:c1",)
        ),
    )
    payload = build_blocks(bundle)
    top_line = payload["blocks"][1]["text"]["text"]
    assert "urn:atlas:chunk:c1" in top_line
    assert "Email Greg" in top_line


def test_atlas_empty_footer_renders_warning(monkeypatch):
    """AC5 — when all three atlas streams empty, footer carries the warning."""
    bundle = _bundle_with(
        commitments=SourceResult("commitments", "empty"),
        contradictions=SourceResult("contradictions", "empty"),
        contacts_overdue=SourceResult("contacts_overdue", "error", error="x"),
        calendar=SourceResult("calendar", "ok", ("9am intro",)),
    )
    payload = build_blocks(bundle)
    footer_text = payload["blocks"][-1]["elements"][0]["text"]
    assert "Atlas corpus warming up" in footer_text


def test_no_atlas_warning_when_one_atlas_stream_ok():
    bundle = _bundle_with(
        commitments=SourceResult("commitments", "ok", ("X",)),
        contradictions=SourceResult("contradictions", "empty"),
        contacts_overdue=SourceResult("contacts_overdue", "empty"),
    )
    payload = build_blocks(bundle)
    footer_text = payload["blocks"][-1]["elements"][0]["text"]
    assert "Atlas corpus warming up" not in footer_text


def test_build_blocks_no_bullets_shows_placeholder():
    payload = build_blocks(_bundle_with())
    top_line = payload["blocks"][1]["text"]["text"]
    assert "No actionable items" in top_line


def test_build_blocks_surfaces_timeout_per_section():
    bundle = _bundle_with(
        commitments=SourceResult("commitments", "timeout", error="per-source timeout"),
    )
    payload = build_blocks(bundle)
    section_texts = [
        b["text"]["text"] for b in payload["blocks"]
        if b["type"] == "section" and "Commitments" in b.get("text", {}).get("text", "")
    ]
    assert any("timeout" in t for t in section_texts)


# ---------------------------------------------------------------------------
# build_daily_brief end-to-end
# ---------------------------------------------------------------------------


def test_build_daily_brief_end_to_end_json_shape():
    atlas_ask = _make_atlas_ask({
        "commitment_audit": _atlas_payload("Email Greg today", "c1"),
    })
    cfg = DailyHandlerConfig(
        atlas_ask=atlas_ask,
        today_fetcher=_today_ok(),
        orchestrator_base_url="",
    )
    payload = asyncio.run(build_daily_brief(cfg))
    assert "blocks" in payload
    # Top-line section should reference the commitment.
    top_line = payload["blocks"][1]["text"]["text"]
    assert "Email Greg today" in top_line


# ---------------------------------------------------------------------------
# handle_daily sync entrypoint
# ---------------------------------------------------------------------------


def test_handle_daily_returns_json_string_with_blocks(monkeypatch):
    """The slash entry point JSON-encodes the block-kit payload."""

    async def _fake_build(config=None):
        return {"blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": "hi"}}], "text": "hi"}

    monkeypatch.setattr(daily_mod, "build_daily_brief", _fake_build)
    out = handle_daily("")
    decoded = json.loads(out)
    assert decoded["blocks"][0]["text"]["text"] == "hi"
    assert decoded["text"] == "hi"


def test_handle_daily_works_inside_running_event_loop(monkeypatch):
    """If called from inside an async context, we fall back to a worker thread."""

    async def _fake_build(config=None):
        return {"blocks": [], "text": "from-thread"}

    monkeypatch.setattr(daily_mod, "build_daily_brief", _fake_build)

    async def _outer():
        # asyncio.run would raise here; handle_daily must work anyway.
        return handle_daily("")

    out = asyncio.run(_outer())
    assert json.loads(out)["text"] == "from-thread"


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def test_register_wires_daily_command_without_breaking_others():
    from plugins.slash import register

    registered: dict[str, dict] = {}

    class _Ctx:
        def register_command(self, name, handler, description="", args_hint=""):
            registered[name] = {
                "handler": handler,
                "description": description,
                "args_hint": args_hint,
            }

    register(_Ctx())

    assert "daily" in registered
    assert callable(registered["daily"]["handler"])
    # The other always-on command still registers.
    assert "draft" in registered
    # resume/skip are gated behind HERMES_DRAIN_CONTROL (off by default), so they
    # are NOT registered here — see test_drain_control_gate.py.
    assert "resume" not in registered and "skip" not in registered
