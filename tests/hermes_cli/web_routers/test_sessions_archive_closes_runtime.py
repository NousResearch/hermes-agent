"""Archiving a session must end its runtime; hiding it must not.

PATCH ``/api/sessions/{id}`` used to flip the ``archived`` column only. The core treats
archive as end-of-life (``_finalize_session``, #105588: the durable row stays resumable
"until the user explicitly closes or archives it"), so leaving the runtime, its active-session
lease and the agent resident is the seam that orphans a delivery lease and a
max_concurrent_sessions slot. The route now calls ``_close_session_by_id(sid,
end_reason="archived")`` on the explicit archive.

``hidden`` is deliberately NOT a close trigger: the canonical Bot Chat is born hidden
(``tui_gateway/methods_session.py``), so acting on it would tear down a live, intended session
and kill its live delivery.

These tests exercise the REAL route (FastAPI ``manage_router`` via ``TestClient``) against a REAL
``SessionDB`` and a REAL ``tui_gateway.server`` session registered with a REAL active-session lease
in a real registry file -- no stubs for the path under test. They fail on the pre-patch route.
"""

import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli.active_sessions import try_acquire_active_session
import hermes_cli.web_routers.sessions as sessions_mod
import tui_gateway.server as gateway
from hermes_state import SessionDB

_SID = "20261001_000000_archtest"


class _StubAgent:
    """Only what ``_finalize_session``/``_teardown_session`` touch on a live session."""

    def __init__(self, sid: str):
        self.session_id = sid
        self.model = "stub-model"
        self.platform = "tui"

    def close(self):
        pass


def _live_session(home: Path, sid: str) -> dict:
    return {
        "agent": _StubAgent(sid),
        "session_key": sid,
        "history": [],
        "history_lock": threading.Lock(),
        "running": False,
        "source": "desktop",
        "profile_home": str(home),
        "created_at": 0.0,
        "last_active": 0.0,
        "active_session_lease": None,
        "slash_worker": None,
    }


def _exercise(home: Path, *, archived=None, hidden=None) -> dict:
    """Drive the real PATCH route against a live session and return the observable state."""
    (home / "runtime").mkdir(parents=True, exist_ok=True)
    db = SessionDB(home / "state.db")
    db.create_session(_SID, "desktop")
    db.set_session_title(_SID, "Bot Chat arch test")
    db.close()

    original_open = sessions_mod._open_session_db_for_profile
    sessions_mod._open_session_db_for_profile = lambda profile, read_only: SessionDB(
        home / "state.db", read_only=read_only
    )

    lease, refusal = try_acquire_active_session(
        session_id=_SID,
        surface="desktop",
        config={},
        registry_home=home,
        metadata={"live_session_id": _SID},
        track_liveness=False,
    )
    assert lease is not None, f"could not acquire active-session lease: {refusal}"

    session = _live_session(home, _SID)
    session["active_session_lease"] = lease
    gateway._sessions[_SID] = session

    try:
        body = {}
        if archived is not None:
            body["archived"] = archived
        if hidden is not None:
            body["hidden"] = hidden

        app = FastAPI()
        app.include_router(sessions_mod.manage_router)
        response = TestClient(app).patch(f"/api/sessions/{_SID}", json=body)
        assert response.status_code == 200, response.text

        db = SessionDB(home / "state.db", read_only=True)
        row = db.get_session(_SID)
        db.close()

        registry = home / "runtime" / "active_sessions.json"
        return {
            "runtime_alive": _SID in gateway._sessions,
            "lease_released": lease.released,
            "ended_at": row.get("ended_at"),
            "end_reason": row.get("end_reason"),
            "registry_empty": (not registry.exists())
            or ('"entries": []' in registry.read_text()),
        }
    finally:
        gateway._sessions.pop(_SID, None)
        sessions_mod._open_session_db_for_profile = original_open


def test_archive_closes_runtime_and_releases_lease(tmp_path):
    """``archived=True`` ends the runtime via the ``_finalize_session`` funnel: lease released,
    the durable row ended, and the live session gone from the in-process gateway registry."""
    state = _exercise(tmp_path, archived=True)

    assert state["runtime_alive"] is False
    assert state["lease_released"] is True
    assert state["registry_empty"] is True
    assert state["ended_at"] is not None
    assert state["end_reason"] == "archived"


def test_hidden_does_not_close_runtime(tmp_path):
    """``hidden=True`` (no archive) is the canonical Bot Chat's normal state: the runtime, the
    lease and the open row are all left untouched."""
    state = _exercise(tmp_path, hidden=True)

    assert state["runtime_alive"] is True
    assert state["lease_released"] is False
    assert state["registry_empty"] is False
    assert state["ended_at"] is None
