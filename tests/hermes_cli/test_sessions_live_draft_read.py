"""#draft-read: an unpersisted Desktop draft must read as an EMPTY page, never a 404.

The Desktop reads /timeline and /messages by STORED id (``20260930_104013_a03380``), not
by the 8-hex runtime id — an earlier cut of this guard keyed on the runtime shape and
therefore never fired in production. These tests pin the stored-id mapping and the store
check that keeps a PERSISTED session on the normal read path.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hermes_cli.web_routers import sessions as sessions_router  # noqa: E402
from tui_gateway import server as gateway_server  # noqa: E402

STORED = "20260930_104013_a03380"
RUNTIME = "f14220b9"


class _Db:
    """Minimal store stub: ``rows`` = ids that exist; ``boom`` raises on every read."""

    def __init__(self, rows=(), boom=False):
        self._rows, self._boom = set(rows), boom

    def _read_one(self, sql, params):
        if self._boom:
            raise RuntimeError("store unavailable")
        return {"1": 1} if params[0] in self._rows else None

    def resolve_session_id(self, session_id):
        if self._boom:
            raise RuntimeError("store unavailable")
        return session_id if session_id in self._rows else None


@pytest.fixture
def draft(monkeypatch):
    monkeypatch.setattr(gateway_server, "_sessions", {}, raising=False)
    monkeypatch.setattr(sessions_router, "_cron_profile_home",
                        lambda profile: (profile or "default", Path("/home/x") / (profile or "default")))
    gateway_server._sessions[RUNTIME] = {"session_key": STORED, "profile_home": None}
    return STORED


def test_draft_stored_id_is_readable_empty(draft):
    assert sessions_router._unpersisted_draft_readable(_Db(), draft, None) is True


def test_matching_is_by_stored_id_not_runtime_shape(draft):
    """The production miss: the Desktop asks by stored id, never by the 8-hex runtime id."""
    assert sessions_router._live_draft_stored_id(draft, None) == draft
    assert sessions_router._live_draft_stored_id(RUNTIME, None) is None


def test_persisted_session_falls_through_to_real_read(draft):
    """Once the first prompt writes the row, the same stored id must NOT be served empty."""
    assert sessions_router._unpersisted_draft_readable(_Db([draft]), draft, None) is False


def test_unknown_stored_id_stays_404(draft):
    assert sessions_router._unpersisted_draft_readable(_Db(), "20260930_000000_dead00", None) is False
    assert sessions_router._unpersisted_draft_readable(_Db(), "deadbeef", None) is False


def test_bg_backlog_ids_never_match(draft):
    assert sessions_router._unpersisted_draft_readable(_Db(), "bg_6be84f", None) is False


def test_unreadable_store_is_never_a_draft(draft):
    """A corrupt/unavailable store must fall through to the honest 503/404, not paint empty."""
    assert sessions_router._unpersisted_draft_readable(_Db(boom=True), draft, None) is False


def test_finalized_or_closing_draft_is_not_served(draft):
    gateway_server._sessions[RUNTIME]["_finalized"] = True
    assert sessions_router._unpersisted_draft_readable(_Db(), draft, None) is False

    gateway_server._sessions[RUNTIME].pop("_finalized")
    gateway_server._sessions[RUNTIME]["_closing"] = True
    assert sessions_router._unpersisted_draft_readable(_Db(), draft, None) is False


def test_session_without_key_is_not_a_draft():
    gateway_server._sessions["ffffffff"] = {"profile_home": None}
    assert sessions_router._live_draft_stored_id(STORED, None) is None


def test_other_profiles_draft_only_matches_its_own_profile(draft):
    gateway_server._sessions[RUNTIME]["profile_home"] = "/home/x/other"
    assert sessions_router._unpersisted_draft_readable(_Db(), draft, "other") is True
    assert sessions_router._unpersisted_draft_readable(_Db(), draft, "default") is False
