"""Machine-only sessions (one-shot ``-z``) must be BORN hidden.

Session visibility used to be opt-out: a scripted run minted a normal visible row and each new
automated entry point had to be patched out of the session list downstream (#246 was exactly such a
patch, for a2a). ``declare_hidden_session()`` flips that for the one-shot channel: the entry point
declares itself machine-only and ``AIAgent._ensure_db_session`` stamps the row hidden at creation.
Hidden rows stay searchable and resumable, so nothing is lost.

The contrast under test: an interactive channel (no declaration) must keep minting visible rows, and
resuming an EXISTING row through a machine channel must not retro-hide a human's session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gateway import session_context as sc  # noqa: E402


class _FakeSessionDB:
    """Records the listing-relevant calls ``_ensure_db_session`` makes."""

    def __init__(self, existing: bool = False):
        self.existing = existing
        self.created: list[dict] = []
        self.hidden: list[tuple[str, bool]] = []

    def get_session(self, session_id):
        return {"id": session_id} if self.existing else None

    def create_session(self, **kwargs):
        self.created.append(kwargs)
        return kwargs.get("session_id")

    def set_session_hidden(self, session_id, hidden):
        self.hidden.append((session_id, hidden))
        return True


class _FakeAgent:
    """The slice of the AIAgent surface ``_ensure_db_session`` touches."""

    def __init__(self, db, session_id: str = "20260910_142145_916ee8"):
        self._session_db = db
        self._session_db_created = False
        self._persist_disabled = False
        self.session_id = session_id
        self.model = "test-model"
        self.platform = "cli"
        self._cached_system_prompt = "sp"
        self._parent_session_id = None

    def _session_row_model_config(self):
        return {"yolo_mode": True}


@pytest.fixture(autouse=True)
def _no_leaked_declaration(monkeypatch):
    monkeypatch.delenv(sc.HIDDEN_SESSION_ENV, raising=False)


def _mint_row(agent):
    from run_agent import AIAgent

    AIAgent._ensure_db_session(agent)


def test_declaration_roundtrip():
    assert sc.hidden_session_declared() is False
    sc.declare_hidden_session()
    assert sc.hidden_session_declared() is True


def test_oneshot_entry_point_declares_hidden_channel():
    """Pin the wiring: the `-z` entry point is what makes a scripted run invisible."""
    oneshot_src = (REPO_ROOT / "hermes_cli" / "oneshot.py").read_text(encoding="utf-8")
    assert "declare_hidden_session()" in oneshot_src


def test_declared_channel_mints_hidden_row():
    sc.declare_hidden_session()
    db = _FakeSessionDB()
    agent = _FakeAgent(db)

    _mint_row(agent)

    assert db.created, "the session row must still be created (audit + resume depend on it)"
    assert db.hidden == [(agent.session_id, True)]


def test_resumed_existing_session_is_not_retro_hidden():
    """A human's visible session continued through `hermes -z -r <id>` stays visible."""
    sc.declare_hidden_session()
    db = _FakeSessionDB(existing=True)
    agent = _FakeAgent(db)

    _mint_row(agent)

    assert db.created, "resume still upserts its row"
    assert db.hidden == []


def test_interactive_channel_stays_visible():
    db = _FakeSessionDB()
    agent = _FakeAgent(db)

    _mint_row(agent)

    assert db.created
    assert db.hidden == []
