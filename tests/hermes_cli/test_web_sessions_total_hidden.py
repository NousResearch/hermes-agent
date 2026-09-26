"""A listing's ``total`` counts exactly the rows its pages can serve.

Hidden sessions (the generic ``hidden`` flag, e.g. Bot Mode chats) never appear in the
session lists, so counting them made the Sessions page promise pages that come back empty.
"""

from pathlib import Path

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_state
    from hermes_state import SessionDB

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    db = SessionDB(home / "state.db")
    for i in range(7):
        sid = f"s{i}"
        db.create_session(sid, "cli")
        db.append_message(sid, "user", f"hello {i}")
    for sid in ("s5", "s6"):
        db.set_session_hidden(sid, True)
    db.close()

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _page_through(client, path):
    served, offset, total = [], 0, None
    while True:
        body = client.get(f"{path}?limit=2&offset={offset}").json()
        total = body["total"]
        served += [row["id"] for row in body["sessions"]]
        offset += 2
        if offset >= total:
            return total, served


@pytest.mark.parametrize("path", ["/api/sessions", "/api/profiles/sessions"])
def test_listing_total_matches_the_rows_its_pages_serve(client, path):
    total, served = _page_through(client, path)

    assert len(served) == len(set(served))
    assert total == len(served)
