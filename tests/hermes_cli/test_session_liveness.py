"""Session liveness comes from the active-session lease registry, not an activity window.

A CLI that crashes leaves ``ended_at`` NULL and a fresh ``last_active``; its lease reads
dead the moment the pid is gone. ``hermes status --json`` and the session lists
(/api/sessions, /api/profiles/sessions) must both report it dead immediately.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hermes_constants import get_hermes_home


def _lease_from_exited_process(session_id: str) -> None:
    """Acquire a CLI lease in a child process that then exits without releasing it."""
    code = textwrap.dedent(f"""
        import os
        from hermes_cli.active_sessions import try_acquire_active_session
        lease, err = try_acquire_active_session(session_id={session_id!r}, surface="cli", config={{}},
                                                metadata={{"live_session_id": {session_id!r}}})
        assert err is None, err
        os._exit(0)  # a crash: no atexit release
    """)
    subprocess.run([sys.executable, "-c", code], check=True, stdin=subprocess.DEVNULL,
                   cwd=str(Path(__file__).resolve().parents[2]))


def _own_lease(session_id: str):
    from hermes_cli.active_sessions import try_acquire_active_session

    lease, err = try_acquire_active_session(session_id=session_id, surface="cli", config={},
                                            metadata={"live_session_id": session_id})
    assert err is None
    return lease


def _store_rows(*rows: tuple[str, str]) -> None:
    from hermes_state import SessionDB

    db = SessionDB(db_path=get_hermes_home() / "state.db")
    try:
        for sid, source in rows:
            db.create_session(sid, source, model="model-" + sid)
            db.append_message(sid, "user", "hello")
    finally:
        db.close()


def test_status_json_lists_live_owner_and_drops_crashed_cli(capsys):
    from hermes_cli.status import show_status
    from types import SimpleNamespace

    _store_rows(("crashed-cli", "cli"), ("open-cli", "cli"))
    _lease_from_exited_process("crashed-cli")
    lease = _own_lease("open-cli")
    try:
        assert show_status(SimpleNamespace(json=True)) == 0
    finally:
        lease.release()
    data = json.loads(capsys.readouterr().out)
    live = data["sessions"]["live"]
    assert [s["session_id"] for s in live] == ["open-cli"]
    import os
    assert live[0]["pid"] == os.getpid()
    assert live[0]["surface"] == "cli" and live[0]["model"] == "model-open-cli"
    assert data["schema_version"] == 1 and data["errors"] == {}


def test_status_json_never_prints_secret_values(monkeypatch, capsys):
    from hermes_cli.status import show_status
    from types import SimpleNamespace

    sentinel = "sk-or-SENTINEL-do-not-leak-0042"
    monkeypatch.setenv("OPENROUTER_API_KEY", sentinel)
    show_status(SimpleNamespace(json=True))
    out = capsys.readouterr().out
    assert sentinel not in out
    assert "OpenRouter" in json.loads(out)["auth"]["api_keys"]


def test_crashed_cli_reads_inactive_while_a_recent_gateway_row_stays_active():
    import time
    from hermes_cli.session_liveness import leased_session_ids, session_is_active

    now = time.time()
    _lease_from_exited_process("crashed-cli")
    lease = _own_lease("open-cli")
    try:
        leased = leased_session_ids(get_hermes_home())
    finally:
        lease.release()
    row = {"ended_at": None, "last_active": now - 5}
    assert session_is_active({**row, "id": "crashed-cli", "source": "cli"}, now, leased) is False
    assert session_is_active({**row, "id": "open-cli", "source": "cli"}, now, leased) is True
    # Gateway turns lease per turn: between turns the activity window still decides.
    assert session_is_active({**row, "id": "tg", "source": "telegram"}, now, leased) is True
    # An unreadable registry falls back to the window rather than calling everything dead.
    assert session_is_active({**row, "id": "crashed-cli", "source": "cli"}, now, None) is True


def test_unreadable_registry_is_reported_not_treated_as_empty():
    from hermes_cli.session_liveness import leased_session_ids

    runtime = get_hermes_home() / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "active_sessions.json").write_text("{not json", encoding="utf-8")
    assert leased_session_ids(get_hermes_home()) is None


def test_cli_lease_follows_the_session_id_the_gateway_hands_back():
    """The CLI claims its lease for a minted id, then adopts the gateway's ``local-…`` id."""
    from cli import HermesCLI
    from hermes_cli.active_sessions import read_live_session_leases

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "20261026_000000_minted"
    cli._active_session_lease = _own_lease("20261026_000000_minted")
    try:
        cli.session_id = "local-adopted"
        assert [e["session_id"] for e in read_live_session_leases()] == ["local-adopted"]
    finally:
        cli._active_session_lease.release()


@pytest.fixture
def client(monkeypatch):
    starlette = pytest.importorskip("starlette.testclient")
    import hermes_state
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    c = starlette.TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_dashboard_session_lists_use_the_lease_verdict(client):
    _store_rows(("crashed-cli", "cli"), ("open-cli", "cli"))
    _lease_from_exited_process("crashed-cli")
    lease = _own_lease("open-cli")
    try:
        rows = {s["id"]: s["is_active"] for s in client.get("/api/sessions").json()["sessions"]}
        sidebar = client.get("/api/profiles/sessions").json()
    finally:
        lease.release()
    assert rows == {"crashed-cli": False, "open-cli": True}
    side_rows = {s["id"]: s["is_active"] for s in sidebar["sessions"]}
    assert side_rows == {"crashed-cli": False, "open-cli": True}
