"""
Tests for the /api/events/recent read-only endpoint.

The endpoint synthesises structural session lifecycle events
from the persisted sessions table — ``session.start`` and
``session.end`` — and returns them newest-first. It carries
only structural metadata (id, source, model, timestamp, plus
end-time aggregates). Message bodies, system prompts and tool
output are deliberately absent.

Coverage:

* Unit: ``_read_recent_events`` plus the two ISO/epoch helpers
  across happy and degenerate paths.
* Integration: HTTP responses, auth gate, query-param plumbing,
  and a regression guard asserting no system-prompt content
  appears in the response.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _epoch_to_iso,
    _parse_iso_to_epoch,
    _read_recent_events,
    cors_middleware,
)
from hermes_state import SessionDB


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra: Dict[str, Any] = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_get("/api/events/recent", adapter._handle_recent_events)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    return app


def _seed_sessions(db_path, *, with_system_prompt: bool = False) -> None:
    """Create three sessions with a mix of ended and live state."""
    db = SessionDB(db_path)
    try:
        db.create_session("s-old", "cli", model="claude-opus-4.6")
        if with_system_prompt:
            db.update_system_prompt(
                "s-old",
                "SECRET_PROMPT_CANARY_dragon_42 — full identity block here.",
            )
        db.end_session("s-old", "user_exit")

        db.create_session("s-mid", "api_server", model="gpt-5.5")
        db.end_session("s-mid", "completed")

        # Live session — only session.start should be emitted.
        db.create_session("s-live", "cron", model="claude-opus-4.6")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Module-level — ISO/epoch helpers
# ---------------------------------------------------------------------------


class TestIsoEpochHelpers:
    def test_parse_iso_accepts_z_suffix(self):
        epoch = _parse_iso_to_epoch("2026-01-01T00:00:00Z")
        assert epoch is not None
        assert isinstance(epoch, float)

    def test_parse_iso_accepts_explicit_offset(self):
        epoch = _parse_iso_to_epoch("2026-01-01T00:00:00+00:00")
        assert epoch is not None

    def test_parse_iso_returns_none_for_none(self):
        assert _parse_iso_to_epoch(None) is None

    def test_parse_iso_returns_none_for_malformed(self):
        assert _parse_iso_to_epoch("not-a-timestamp") is None

    def test_epoch_to_iso_round_trips_via_parse(self):
        epoch = _parse_iso_to_epoch("2026-05-23T12:34:56Z")
        iso = _epoch_to_iso(epoch)
        assert iso is not None
        # Should be ISO with Z suffix.
        assert iso.endswith("Z")
        # Should re-parse to (approximately) the same epoch.
        assert _parse_iso_to_epoch(iso) == pytest.approx(epoch, abs=0.001)

    def test_epoch_to_iso_returns_none_for_none(self):
        assert _epoch_to_iso(None) is None


# ---------------------------------------------------------------------------
# Module-level — _read_recent_events
# ---------------------------------------------------------------------------


class TestReadRecentEvents:
    def test_empty_db_returns_empty_list(self, tmp_path):
        db_path = tmp_path / "state.db"
        SessionDB(db_path).close()  # create empty schema
        out = _read_recent_events(db_path)
        assert out == []

    def test_each_session_yields_start_event(self, tmp_path):
        db_path = tmp_path / "state.db"
        _seed_sessions(db_path)
        out = _read_recent_events(db_path, limit=100)
        starts = [e for e in out if e["type"] == "session.start"]
        ids = {e["session_id"] for e in starts}
        assert {"s-old", "s-mid", "s-live"} <= ids

    def test_ended_sessions_yield_end_event_with_metadata(self, tmp_path):
        db_path = tmp_path / "state.db"
        _seed_sessions(db_path)
        out = _read_recent_events(db_path, limit=100)
        ends = {e["session_id"]: e for e in out if e["type"] == "session.end"}
        assert "s-old" in ends
        assert "s-mid" in ends
        # Live session must NOT have an end event.
        assert "s-live" not in ends
        meta = ends["s-old"]["metadata"]
        assert meta["end_reason"] == "user_exit"
        assert "message_count" in meta
        assert "tool_call_count" in meta
        assert "duration_seconds" in meta

    def test_events_are_newest_first(self, tmp_path):
        db_path = tmp_path / "state.db"
        _seed_sessions(db_path)
        out = _read_recent_events(db_path, limit=100)
        timestamps = [e["timestamp"] for e in out if e["timestamp"]]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_limit_caps_result(self, tmp_path):
        db_path = tmp_path / "state.db"
        _seed_sessions(db_path)
        out = _read_recent_events(db_path, limit=2)
        assert len(out) == 2

    def test_since_filters_older_events(self, tmp_path):
        db_path = tmp_path / "state.db"
        _seed_sessions(db_path)
        # Future timestamp filters everything out.
        out = _read_recent_events(
            db_path, limit=100, since_iso="9999-01-01T00:00:00Z"
        )
        assert out == []

    def test_missing_columns_yield_empty_not_error(self, tmp_path):
        """Legacy DB without one of the expected columns must
        degrade gracefully — empty list rather than 500."""
        db_path = tmp_path / "legacy.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "CREATE TABLE sessions (id TEXT, started_at REAL)"
            )
        out = _read_recent_events(db_path)
        assert out == []

    def test_payload_never_includes_system_prompt(self, tmp_path):
        """Regression guard: even when sessions carry a populated
        ``system_prompt``, the synthesised event stream must not
        expose any of it."""
        db_path = tmp_path / "state.db"
        _seed_sessions(db_path, with_system_prompt=True)
        out = _read_recent_events(db_path, limit=100)
        flat = repr(out)
        assert "SECRET_PROMPT_CANARY_dragon_42" not in flat
        # No key called ``system_prompt`` at any depth either.
        for event in out:
            assert "system_prompt" not in event
            assert "system_prompt" not in event.get("metadata", {})


# ---------------------------------------------------------------------------
# HTTP integration
# ---------------------------------------------------------------------------


class TestRecentEventsEndpoint:
    @pytest.mark.asyncio
    async def test_returns_empty_when_db_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/events/recent")
            assert resp.status == 200
            body = await resp.json()
        assert body == {"events": [], "limit": 50}

    @pytest.mark.asyncio
    async def test_returns_events_when_db_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_sessions(tmp_path / "state.db")
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/events/recent")
            assert resp.status == 200
            body = await resp.json()
        assert body["limit"] == 50
        assert len(body["events"]) >= 3  # 3 starts + 2 ends = 5
        # Each event has the documented top-level shape.
        for ev in body["events"]:
            assert ev["type"] in ("session.start", "session.end")
            assert "timestamp" in ev
            assert "session_id" in ev
            assert "source" in ev
            assert "model" in ev

    @pytest.mark.asyncio
    async def test_requires_auth_when_key_configured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        app = _create_app(_make_adapter(api_key="sk-secret"))
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/events/recent")
            assert resp.status == 401
            resp = await cli.get(
                "/api/events/recent",
                headers={"Authorization": "Bearer sk-secret"},
            )
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_limit_param_propagates(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_sessions(tmp_path / "state.db")
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/events/recent", params={"limit": "2"})
            body = await resp.json()
        assert body["limit"] == 2
        assert len(body["events"]) <= 2

    @pytest.mark.asyncio
    async def test_since_param_filters(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_sessions(tmp_path / "state.db")
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get(
                "/api/events/recent",
                params={"since": "9999-01-01T00:00:00Z"},
            )
            body = await resp.json()
        assert body["events"] == []

    @pytest.mark.asyncio
    async def test_response_never_contains_system_prompt(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_sessions(tmp_path / "state.db", with_system_prompt=True)
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/events/recent")
            raw = await resp.text()
        assert "SECRET_PROMPT_CANARY_dragon_42" not in raw, (
            "system_prompt content leaked into events response"
        )


class TestCapabilitiesAdvertisesRecentEvents:
    @pytest.mark.asyncio
    async def test_capabilities_lists_recent_events_endpoint(self):
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            body = await resp.json()
        assert body["features"]["remote_recent_events"] is True
        assert body["endpoints"]["recent_events"]["path"] == "/api/events/recent"
        assert body["endpoints"]["recent_events"]["method"] == "GET"
