"""A browse read opens ``state.db`` once.

The Desktop's session reads each take a disposable read-only handle through ``_open_session_db_at_path``.
The declared-schema probe that decides whether the store needs its one browse catch-up runs on that
handle's own connection, so a read pays one SQLite open, as the read-only SessionDB browse did.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.web_server_sessions import _open_session_db_at_path
from persistence.conversation.messages import TranscriptRead
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session


def _seeded(home: Path) -> Path:
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    with open_persistence(profile_scope(home=home)) as store:
        seed_session(store, "kept", "cli")
        seed_messages(store, "kept", *({"role": "user", "content": f"m{i}"} for i in range(5)))
    return home / "state.db"


def _count_opens(monkeypatch, db_path: Path) -> list:
    import hermes_cli.sqlite_safe_read as ssr

    original, target, opens = ssr.connect_tracked, db_path.resolve(), []

    def counting(path, *args, **kwargs):
        tracked = kwargs.get("tracking_path")
        if tracked is not None and Path(tracked).resolve() == target:
            opens.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(ssr, "connect_tracked", counting)
    return opens


def test_browse_read_opens_the_store_once(tmp_path, monkeypatch):
    path = _seeded(tmp_path)
    opens = _count_opens(monkeypatch, path)

    store = _open_session_db_at_path(path, read_only=True)
    try:
        assert store.sessions.get("kept") is not None
        assert len(store.messages.read_transcript("kept", TranscriptRead(limit=50, latest=True))) == 5
    finally:
        store.close()
    assert len(opens) == 1, opens


@pytest.mark.parametrize("route", ["/api/sessions/kept", "/api/sessions/kept/messages?limit=50&order=latest"])
def test_desktop_session_reads_open_the_store_once(route, monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    from hermes_constants import get_hermes_home

    path = _seeded(get_hermes_home())
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    assert client.get(route).status_code == 200  # first request: app startup work out of the count

    opens = _count_opens(monkeypatch, path)
    response = client.get(route)
    assert response.status_code == 200, response.text
    assert len(opens) == 1, opens
