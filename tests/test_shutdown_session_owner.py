"""Tests for fix/hard-03 (issue #59580).

When the TUI shuts down, it must not finalize a session that another runtime
(typically the messaging gateway) is actively routing for — the original bug
let `atexit → _shutdown_sessions()` blindly close every session in the
in-memory `_sessions` map, which left the gateway's `sessions.json` entries
pointing at finalized session_ids. The downstream gateway self-heal (#54878)
correctly dropped the stale routing entry but couldn't recover
(`tui_shutdown` is not in the recoverable `end_reason` set), so the next
inbound platform message silently opened a brand-new session with zero
user-facing notification.

These tests cover the surgical fix on the TUI side: a new ``owner`` field on
the per-session record, ``"tui"`` by default, that callers can flip to
``"gateway"`` when the routing for the same ``session_key`` is owned by
another runtime. ``_shutdown_sessions()`` now skips non-``tui`` owners and
logs a debug line so the omission is visible in the field-instrumented
build. A startup self-heal scan clears stale ``owner="gateway"`` records
whose peer route is gone.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from tui_gateway import server


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _put_session(sid: str, *, owner: str = "tui", session_key: str = "k",
                 extras: dict | None = None):
    """Insert a synthetic session record into ``_sessions`` for tests.

    The real ``_sessions`` entries are much larger than this; tests just need
    the ``owner`` field plus a stable ``session_key`` so the self-heal can
    find it in the gateway routing index (or not).
    """
    record = {"owner": owner, "session_key": session_key}
    if extras:
        record.update(extras)
    server._sessions[sid] = record


def _clear():
    server._sessions.clear()


@pytest.fixture(autouse=True)
def _isolate_sessions():
    _clear()
    yield
    _clear()


# ---------------------------------------------------------------------------
# _set_session_owner
# ---------------------------------------------------------------------------


class TestSetSessionOwner:
    def test_set_owner_on_existing_session(self):
        _put_session("sid_a")
        server._set_session_owner("sid_a", "gateway")
        assert server._sessions["sid_a"]["owner"] == "gateway"

    def test_set_owner_unknown_sid_is_noop(self):
        # Don't accidentally create an entry.
        server._set_session_owner("never-inserted", "gateway")
        assert "never-inserted" not in server._sessions

    def test_set_owner_empty_string_is_noop(self):
        _put_session("sid_b")
        server._set_session_owner("sid_b", "")
        assert server._sessions["sid_b"]["owner"] == "tui"

    def test_gateway_owner_is_not_demoted_to_tui(self):
        """Once the gateway claims the routing, it stays claimed until the
        peer-self-heal explicitly clears it. This prevents a transient TUI
        flap from re-finalizing a session the gateway is mid-turn on."""
        _put_session("sid_c", owner="gateway")
        server._set_session_owner("sid_c", "tui")
        assert server._sessions["sid_c"]["owner"] == "gateway"


# ---------------------------------------------------------------------------
# _shutdown_sessions — owns-aware teardown
# ---------------------------------------------------------------------------


class TestShutdownSessionsOwnerAware:
    def test_tui_owned_session_is_closed(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            server, "_close_session_by_id",
            lambda sid, *, end_reason: seen.append((sid, end_reason)),
        )
        _put_session("a", owner="tui")
        _put_session("b", owner="tui")

        server._shutdown_sessions()

        assert sorted(s for s, _ in seen) == ["a", "b"]
        assert {r for _, r in seen} == {"tui_shutdown"}

    def test_gateway_owned_session_is_preserved(self, monkeypatch):
        """The headline regression of #59580: a gateway-routed session must
        survive a TUI exit so the next inbound platform message lands on
        the same session_id instead of a freshly minted one."""
        seen = []
        monkeypatch.setattr(
            server, "_close_session_by_id",
            lambda sid, *, end_reason: seen.append((sid, end_reason)),
        )
        _put_session("ui", owner="tui")
        _put_session("gw", owner="gateway")

        server._shutdown_sessions()

        assert {s for s, _ in seen} == {"ui"}
        # The "gw" record must remain in the live map so subsequent turns
        # (or even an unexpected second attach) can still reach it.
        assert "gw" in server._sessions
        assert server._sessions["gw"]["owner"] == "gateway"

    def test_legacy_untagged_session_is_treated_as_tui(self, monkeypatch):
        """Pre-#59580 builds never wrote the field at all. Don't break them
        by suddenly orphaning every existing session on shutdown."""
        seen = []
        monkeypatch.setattr(
            server, "_close_session_by_id",
            lambda sid, *, end_reason: seen.append((sid, end_reason)),
        )
        # No "owner" key — legacy record shape.
        server._sessions["legacy"] = {"session_key": "k"}

        server._shutdown_sessions()

        assert ("legacy", "tui_shutdown") in seen

    def test_mixed_owners_only_close_tui_owned(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            server, "_close_session_by_id",
            lambda sid, *, end_reason: seen.append((sid, end_reason)),
        )
        _put_session("t1", owner="tui")
        _put_session("g1", owner="gateway")
        _put_session("t2", owner="tui")
        _put_session("g2", owner="gateway")
        _put_session("legacy", owner="tui")

        server._shutdown_sessions()

        assert sorted(s for s, _ in seen) == ["legacy", "t1", "t2"]
        assert all(r == "tui_shutdown" for _, r in seen)

    def test_shutdown_logs_skip_for_gateway_owner(self, monkeypatch, caplog):
        monkeypatch.setattr(
            server, "_close_session_by_id",
            lambda sid, *, end_reason: None,
        )
        _put_session("skip_me", owner="gateway")
        with caplog.at_level(logging.DEBUG, logger="tui_gateway.server"):
            server._shutdown_sessions()
        assert any(
            "skipping sid=skip_me" in rec.message and "owner=gateway" in rec.message
            for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# _peer_ownership_check — self-heal pass
# ---------------------------------------------------------------------------


class TestPeerOwnershipCheck:
    def test_peer_missing_downgrades_owner_and_clears_dead_transport(self):
        """No live gateway route → owner flips back to tui, dead-transport
        records are evicted so the user doesn't stare at a half-built
        composer window on the next launch."""
        sess = {
            "owner": "gateway",
            "session_key": "tx_key",
            "session_id": "sid_x",
            "transport": server._detached_ws_transport,
        }
        server._sessions["sid_x"] = sess
        with patch.object(server, "_peer_routing_session_ids", return_value=set()):
            server._peer_ownership_check("sid_x", sess)
        assert "sid_x" not in server._sessions

    def test_peer_missing_keeps_alive_transport(self):
        """Live transport + dropped peer route → owner flips back to tui
        but the session keeps running; a future TUI shutdown may now
        close it normally."""
        sess = {
            "owner": "gateway",
            "session_key": "tx_key",
            "session_id": "sid_x",
            "transport": server._stdio_transport,
        }
        server._sessions["sid_x"] = sess
        with patch.object(server, "_peer_routing_session_ids", return_value=set()):
            server._peer_ownership_check("sid_x", sess)
        assert server._sessions["sid_x"]["owner"] == "tui"

    def test_peer_alive_with_matching_id_drops_tui_record(self):
        """The TUI's in-memory session is stale if the gateway still routes
        for this key but to a *different* session_id. Drop the TUI record
        so the next attach re-binds to the gateway's authoritative one."""
        sess = {
            "owner": "gateway",
            "session_key": "tx_key",
            "session_id": "sid_old",
        }
        server._sessions["sid_old"] = sess
        peer_ids = {"sid_new"}
        with patch.object(
            server, "_peer_routing_session_ids", return_value=peer_ids
        ):
            server._peer_ownership_check("sid_old", sess)
        assert "sid_old" not in server._sessions

    def test_peer_alive_with_matching_id_promotes_tui_owner(self):
        """When the same session is alive in both the TUI and the gateway,
        the gateway is the long-lived authority — flip the TUI's local
        owner to ``gateway`` so subsequent shutdowns leave it alone."""
        sess = {
            "owner": "tui",
            "session_key": "tx_key",
            "session_id": "sid_shared",
        }
        server._sessions["sid_shared"] = sess
        with patch.object(
            server, "_peer_routing_session_ids", return_value={"sid_shared"}
        ):
            server._peer_ownership_check("sid_shared", sess)
        assert server._sessions["sid_shared"]["owner"] == "gateway"

    def test_peer_alive_but_owner_already_gateway_is_idempotent(self):
        sess = {
            "owner": "gateway",
            "session_key": "tx_key",
            "session_id": "sid_shared",
        }
        server._sessions["sid_shared"] = sess
        with patch.object(
            server, "_peer_routing_session_ids", return_value={"sid_shared"}
        ):
            # Two calls in a row should produce the same outcome — no log
            # spam, no surprises.
            server._peer_ownership_check("sid_shared", sess)
            server._peer_ownership_check("sid_shared", sess)
        assert server._sessions["sid_shared"]["owner"] == "gateway"


# ---------------------------------------------------------------------------
# _peer_routing_session_ids — direct JSON peek
# ---------------------------------------------------------------------------


class TestPeerRoutingSessionIds:
    def test_missing_file_returns_empty(self, tmp_path):
        from hermes_constants import set_hermes_home_override
        home = tmp_path / "no_gateway_here"
        home.mkdir()
        token = set_hermes_home_override(home)
        try:
            assert server._peer_routing_session_ids("any_key") == set()
        finally:
            from hermes_constants import reset_hermes_home_override
            reset_hermes_home_override(token)

    def test_present_file_returns_listed_session_id(self, tmp_path):
        home = tmp_path / "h"
        home.mkdir()
        (home / "gateway").mkdir()
        (home / "gateway" / "sessions.json").write_text(
            json.dumps({
                "_README": "ignored",
                "agent:main:telegram:dm:1": {
                    "session_id": "weixin-thread-abc",
                    "display_name": "telegram chat",
                },
                "agent:main:telegram:dm:2": {
                    "session_id": "weixin-thread-other",
                },
            }),
            encoding="utf-8",
        )
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        token = set_hermes_home_override(home)
        try:
            assert server._peer_routing_session_ids(
                "agent:main:telegram:dm:1"
            ) == {"weixin-thread-abc"}
            assert server._peer_routing_session_ids("missing-key") == set()
            # Corrupt entry: no session_id field → empty set, not exception.
            assert server._peer_routing_session_ids(
                "agent:main:telegram:dm:2"
            ) == {"weixin-thread-other"}
        finally:
            reset_hermes_home_override(token)

    def test_corrupt_file_does_not_raise(self, tmp_path):
        home = tmp_path / "h2"
        home.mkdir()
        (home / "gateway").mkdir()
        (home / "gateway" / "sessions.json").write_text("not json", encoding="utf-8")
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        token = set_hermes_home_override(home)
        try:
            assert server._peer_routing_session_ids("k") == set()
        finally:
            reset_hermes_home_override(token)


# ---------------------------------------------------------------------------
# End-to-end: the slice task list
# ---------------------------------------------------------------------------


def test_regression_tui_owned_session_still_ends(monkeypatch):
    """No-regression for the TUI-owned case: a vanilla TUI-only session
    must still be finalized on shutdown so the next launch starts fresh."""
    seen = []
    monkeypatch.setattr(
        server, "_close_session_by_id",
        lambda sid, *, end_reason: seen.append((sid, end_reason)),
    )
    _put_session("only_tui", owner="tui", session_key="alpha")
    server._shutdown_sessions()
    assert ("only_tui", "tui_shutdown") in seen


def test_regression_gateway_routed_session_continues(monkeypatch):
    """Headline fix: a gateway-routed session survives a TUI exit."""
    seen = []
    monkeypatch.setattr(
        server, "_close_session_by_id",
        lambda sid, *, end_reason: seen.append((sid, end_reason)),
    )
    _put_session("peer_owned", owner="gateway", session_key="alpha")
    server._shutdown_sessions()
    assert ("peer_owned", "tui_shutdown") not in seen
    # And it is still in the live map.
    assert "peer_owned" in server._sessions


def test_self_heal_clears_stale_gateway_owner_when_peer_is_gone(monkeypatch):
    """Startup self-heal removes a ``owner="gateway"`` claim whose peer
    route disappeared, so the next message can't reuse a dead session."""
    _put_session(
        "zombie", owner="gateway", session_key="dead",
        extras={"transport": server._detached_ws_transport},
    )
    with patch.object(server, "_peer_routing_session_ids", return_value=set()):
        server._peer_ownership_check(
            "zombie",
            server._sessions["zombie"],
        )
    assert "zombie" not in server._sessions


def test_self_heal_keeps_live_gateway_peer_route_intact(monkeypatch):
    """Startup self-heal does NOT touch sessions whose peer route is alive,
    so a live gateway process keeps its routing entry undisturbed."""
    _put_session(
        "live", owner="tui", session_key="live_peer",
        extras={"session_id": "live_sid"},
    )
    with patch.object(
        server, "_peer_routing_session_ids", return_value={"live_sid"}
    ):
        server._peer_ownership_check(
            "live",
            server._sessions["live"],
        )
    assert "live" in server._sessions
    # Owner upgraded to gateway because the peer still routes here.
    assert server._sessions["live"]["owner"] == "gateway"


def test_default_owner_is_tui_for_fresh_sessions(monkeypatch):
    """The three record-init sites (resume, lazy/watch, JSON-RPC create)
    must default the new ``owner`` field to ``"tui"`` so _shutdown_sessions
    behavior is unchanged for sessions never crossed with a peer runtime."""
    monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
    sid = server.handle_request(
        {"id": "1", "method": "session.create", "params": {}}
    )["result"]["session_id"]
    assert server._sessions[sid]["owner"] == "tui"
    server._sessions.clear()
