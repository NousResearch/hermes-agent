"""T16 (#134889): classic-CLI ``/status`` names a gateway session by its canonical ``local-…`` id.

Main printed the ``YYYYMMDD_HHMMSS_<hex>`` id it minted itself. Under the one gateway the
session row's only id is the canonical ``local-`` + sha256 receipt identity (``local_identity``);
no legacy alias exists to print. ``/status`` shows that full id because it is the value every
other surface uses for the same session: the exit epilogue's ``hermes --resume <id>``, the TUI
and gateway ``/status``, ``hermes sessions list`` and ``state.db``. A shortened or reformatted
id would be one more spelling that a user copies and the resolver may not accept.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from hermes_cli.cli_session_mixin import CLISessionMixin
from hermes_state import SessionDB
from hermes_state_keys import local_identity


def _status_text(session_db, session_id: str) -> str:
    rendered: list[str] = []
    cli = SimpleNamespace(
        _session_db=session_db, session_id=session_id, session_start=datetime(2031, 1, 1), agent=None,
        provider="custom", model="loopback-model", _agent_running=False, reasoning_config=None,
        show_reasoning=None, session_key="", _get_status_bar_snapshot=dict,
        _console_print=lambda text, **_kw: rendered.append(text),
    )
    CLISessionMixin._show_session_status(cli)
    return "\n".join(rendered)


def test_status_prints_the_full_canonical_id_that_resume_accepts(tmp_path, monkeypatch):
    sid = local_identity("default", "local-user", "req-t16")
    assert sid.startswith("local-") and len(sid) == len("local-") + 64
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session(sid, "cli", model="loopback-model")
        db.set_session_title(sid, "T16 canonical id")
        text = _status_text(db, sid)

    line = next(row for row in text.splitlines() if row.startswith("Session ID:"))
    assert line == f"Session ID: {sid}", text
    assert "T16 canonical id" in text

    # The printed id is what `hermes --resume <id>` takes, unshortened and unchanged.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli import main as hermes_main

    monkeypatch.setattr(hermes_main, "get_hermes_home", lambda: tmp_path, raising=False)
    assert hermes_main._resolve_session_by_name_or_id(line.split(": ", 1)[1]) == sid
