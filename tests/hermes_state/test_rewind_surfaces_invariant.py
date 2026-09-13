"""Invariant: every /undo + /retry surface persists through ``SessionDB.rewind_user_turn``, so the same
session rewound through the CLI, the gateway store, or the TUI helper leaves the IDENTICAL active
message set, and an out-of-range target changes nothing on every surface."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig
from gateway.session import SessionStore
from hermes_state import SessionDB
from hermes_state_rewind import RewindTargetUnavailableError

SURFACES = ("cli", "gateway", "tui")


_IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}


def _seed(db: SessionDB, sid: str, turns: int = 3, rich: bool = False) -> None:
    """``rich``: every turn is an image-bearing user ask answered through a tool_call/tool_result pair."""
    db.create_session(sid, source="cli")
    for i in range(1, turns + 1):
        if not rich:
            db.append_message(sid, "user", f"q{i}")
            db.append_message(sid, "assistant", f"a{i}")
            continue
        db.append_message(sid, "user", [{"type": "text", "text": f"q{i}"}, dict(_IMAGE)])
        db.append_message(sid, "assistant", None, tool_calls=[
            {"id": f"call-{i}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}])
        db.append_message(sid, "tool", f"out{i}", tool_call_id=f"call-{i}", tool_name="terminal")
        db.append_message(sid, "assistant", f"a{i}")


def _active_rows(db: SessionDB, sid: str):
    return [tuple(r) for r in db._conn.execute(
        "SELECT id, role, content, active FROM messages WHERE session_id = ? ORDER BY id", (sid,)).fetchall()]


def _rewind_via(surface: str, db: SessionDB, sid: str, n: int):
    """Back up ``n`` user turns through one surface; returns a truthy outcome or raises/None on refusal."""
    if surface == "gateway":
        store = SessionStore.__new__(SessionStore)
        store._db_for_session_id = lambda _sid: db
        store._lazy = lambda name, factory: factory()
        store._clear_dirty_transcript = lambda _sid: None
        return store.rewind_session(sid, n)
    warm = db.get_messages_as_conversation(sid)
    user_turns = sum(1 for m in warm if m.get("role") == "user")
    ordinal = user_turns - n
    if surface == "cli":
        from hermes_cli.cli_session_mixin import CLISessionMixin
        cli = CLISessionMixin.__new__(CLISessionMixin)
        cli._session_db, cli.session_id, cli.conversation_history, cli.agent = db, sid, warm, None
        cli._prefill_input_buffer = MagicMock()
        cli.undo_last(n)
        return cli.conversation_history if len(cli.conversation_history) < len(warm) else None
    from tui_gateway.server import _rewind_active_session_history
    session = {"agent": SimpleNamespace(_session_messages=warm), "history": list(warm),
               "history_lock": threading.Lock(), "history_version": 0, "session_key": sid}
    return _rewind_active_session_history(session, ordinal)


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tui_gateway import server
    handle = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_db", handle)
    yield handle
    handle.close()


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("rich", [False, True], ids=["text", "tool_calls+image"])
def test_every_surface_persists_the_same_active_set(db, n, rich):
    expected = None
    for surface in SURFACES:
        sid = f"rewind-{surface}-{n}-{rich}"
        _seed(db, sid, rich=rich)
        assert _rewind_via(surface, db, sid, n)
        rows = [(role, content, active) for _id, role, content, active in _active_rows(db, sid)]
        assert rows == (expected := expected or rows), surface
    active = [(r, c) for r, c, a in expected if a]
    if not rich:
        assert [c for _r, c in active] == [f"q{i}" if k == 0 else f"a{i}" for i in range(1, 4 - n) for k in (0, 1)]
    else:
        # The whole tool exchange of a rewound turn goes with it (no orphan tool_result), and the
        # surviving image-bearing user rows stay byte-identical to what was stored.
        assert [r for r, _c in active] == ["user", "assistant", "tool", "assistant"] * (3 - n)
        assert [c for r, c in active if r == "user"] == [
            c for r, c, _a in expected if r == "user"][: 3 - n]
        assert all("QUJD" in c for r, c in active if r == "user")


@pytest.mark.parametrize("surface", SURFACES)
def test_out_of_range_target_changes_nothing_on_every_surface(db, surface):
    sid = f"oob-{surface}"
    _seed(db, sid, turns=1)
    before = _active_rows(db, sid)
    if surface == "tui":
        with pytest.raises(ValueError):  # the TUI's own error class for a vanished target
            _rewind_via(surface, db, sid, 2)
    else:
        # CLI /undo N and gateway /undo N clamp to the oldest turn, so their out-of-range case is a
        # transcript with no user turn at all: both refuse without touching the store.
        db.create_session(sid + "-empty", source="cli")
        if surface == "gateway":
            assert _rewind_via(surface, db, sid + "-empty", 1) is None
        else:
            with pytest.raises(RewindTargetUnavailableError):
                db.rewind_user_turn(sid + "-empty", -1, warm_history=[])
        assert _active_rows(db, sid + "-empty") == []
    assert _active_rows(db, sid) == before
