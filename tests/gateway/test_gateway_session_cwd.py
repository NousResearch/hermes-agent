"""Gateway-created session rows carry a usable ``cwd`` (#93625).

The desktop sidebar groups sessions by the ``sessions.cwd`` column. The messaging
gateway's INSERT path used to never pass a cwd, so every platform row landed with
``cwd = NULL`` and fell out of every project lane. ``gateway/run.py`` resolves
``terminal.cwd`` placeholders into ``TERMINAL_CWD`` at startup (``$HOME`` fallback,
unset when unresolvable), so seeding the row from it gives each gateway session the
same workspace its tools execute in — mirroring ``tui_gateway``'s
``_default_session_cwd``.
"""
from __future__ import annotations

import os

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


@pytest.fixture()
def store(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def _slack_source() -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        user_id="U123",
        chat_id="C123",
        chat_type="channel",
        thread_id="T1",
    )


class TestGatewaySessionCwdSeeding:
    def test_create_persists_resolved_terminal_cwd(self, store, monkeypatch):
        """A configured ``terminal.cwd`` (bridged to TERMINAL_CWD at startup) becomes the
        session row's workspace, so the sidebar groups the session under that project."""
        monkeypatch.setenv("TERMINAL_CWD", "/work/research")
        entry = store.get_or_create_session(_slack_source())
        row = store._db.get_session(entry.session_id)
        assert row["cwd"] == "/work/research"

    def test_create_without_terminal_cwd_falls_back_to_home(self, store):
        """No configured cwd: the row is stamped with $HOME (the placeholder resolution
        gateway/run.py applies), never left NULL to fall out of every lane."""
        entry = store.get_or_create_session(_slack_source())
        row = store._db.get_session(entry.session_id)
        assert row["cwd"] == os.path.expanduser("~")

    def test_existing_cwd_is_never_overwritten(self, store, monkeypatch):
        """The seed only fills creation; a workspace the user moved the session to (or a
        later explicit update) keeps winning over the launch default on refresh."""
        monkeypatch.setenv("TERMINAL_CWD", "/work/other")
        entry = store.get_or_create_session(_slack_source())
        store._db.update_session_cwd(entry.session_id, "/work/explicit")
        # A reset routes through _session_create_kwargs again (new session_id) — but the
        # per-turn peer refresh on the SAME row must not drag it back to the launch cwd.
        store.get_or_create_session(_slack_source())
        assert store._db.get_session(entry.session_id)["cwd"] == "/work/explicit"
