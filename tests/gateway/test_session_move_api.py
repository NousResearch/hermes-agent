"""HRM-T0a step 6: API endpoints for session-move primitive.

These tests exercise ``POST /api/sessions/{id}/move/last`` and
``POST /api/sessions/{id}/move/range`` against an in-process aiohttp
TestServer, with a real ``SessionDB`` backing the underlying
``move_turns`` primitive. The handlers must remain thin wrappers:
validation + 4xx/5xx mapping, no reimplementation of transactional
state.

Covered:
  * dry-run plans (no mutation, idempotency key optional)
  * commit happy path (rows moved + tombstoned, response shape)
  * idempotency replay (body byte-equal except ``replay: true``)
  * invalid source / destination / count / range / dst-equals-src
  * missing idempotency_key on commit (400)
  * default-deny auth posture (401 without bearer token)
  * capabilities advertises both endpoints
"""

from __future__ import annotations

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    a = APIServerAdapter(PlatformConfig(enabled=True))
    a._session_db = session_db
    return a


@pytest.fixture
def auth_adapter(session_db):
    a = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
    a._session_db = session_db
    return a


def _create_app(a: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/capabilities", a._handle_capabilities)
    app.router.add_post(
        "/api/sessions/{session_id}/move/last", a._handle_session_move_last
    )
    app.router.add_post(
        "/api/sessions/{session_id}/move/range", a._handle_session_move_range
    )
    return app


def _seed(db: SessionDB, *, src_n: int = 4, dst_n: int = 1) -> None:
    db.create_session(session_id="src", source="api_server", user_id="u")
    db.create_session(session_id="dst", source="api_server", user_id="u")
    for i in range(src_n):
        db.append_message("src", "user", f"src-{i}")
    for i in range(dst_n):
        db.append_message("dst", "user", f"dst-{i}")


# ── capabilities advertisement ────────────────────────────────────────


@pytest.mark.asyncio
async def test_capabilities_advertises_move_endpoints(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities")
        assert resp.status == 200
        data = await resp.json()

    assert data["features"]["session_move"] is True
    assert data["endpoints"]["session_move_last"] == {
        "method": "POST",
        "path": "/api/sessions/{session_id}/move/last",
    }
    assert data["endpoints"]["session_move_range"] == {
        "method": "POST",
        "path": "/api/sessions/{session_id}/move/range",
    }


# ── move/last happy path + replay ─────────────────────────────────────


@pytest.mark.asyncio
async def test_move_last_dry_run_plans_without_mutation(adapter, session_db):
    _seed(session_db, src_n=4, dst_n=0)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/src/move/last",
            json={"dst_session_id": "dst", "count": 2, "dry_run": True},
        )
        assert resp.status == 200, await resp.text()
        body = await resp.json()

    assert body["object"] == "hermes.session.move"
    assert body["dry_run"] is True
    assert body["src_session_id"] == "src"
    assert body["dst_session_id"] == "dst"
    assert body["range_spec"] == "last:2"
    assert len(body["src_message_ids"]) == 2
    assert body["dst_message_ids"] == []
    # No mutation occurred.
    src_active = session_db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'src' AND active = 1"
    ).fetchone()[0]
    assert src_active == 4
    dst_total = session_db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'dst'"
    ).fetchone()[0]
    assert dst_total == 0


@pytest.mark.asyncio
async def test_move_last_commit_mutates_and_logs(adapter, session_db):
    _seed(session_db, src_n=4, dst_n=1)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/src/move/last",
            json={
                "dst_session_id": "dst",
                "count": 2,
                "idempotency_key": "move-1",
            },
        )
        assert resp.status == 200, await resp.text()
        body = await resp.json()

    assert body["object"] == "hermes.session.move"
    assert body["dry_run"] is False
    assert body["replay"] is False
    assert body["idempotency_key"] == "move-1"
    assert len(body["src_message_ids"]) == 2
    assert len(body["dst_message_ids"]) == 2

    src_active = session_db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'src' AND active = 1"
    ).fetchone()[0]
    assert src_active == 2
    dst_total = session_db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'dst'"
    ).fetchone()[0]
    assert dst_total == 3  # 1 original + 2 moved
    log_count = session_db._conn.execute(
        "SELECT COUNT(*) FROM move_log WHERE idempotency_key = 'move-1'"
    ).fetchone()[0]
    assert log_count == 1


@pytest.mark.asyncio
async def test_move_last_replay_returns_cached_body(adapter, session_db):
    _seed(session_db, src_n=4, dst_n=0)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        first = await cli.post(
            "/api/sessions/src/move/last",
            json={
                "dst_session_id": "dst",
                "count": 2,
                "idempotency_key": "rk",
            },
        )
        first_body = await first.json()
        second = await cli.post(
            "/api/sessions/src/move/last",
            json={
                "dst_session_id": "dst",
                "count": 2,
                "idempotency_key": "rk",
            },
        )
        second_body = await second.json()

    assert first.status == 200 and second.status == 200
    assert first_body["replay"] is False
    assert second_body["replay"] is True
    # All other fields byte-equal.
    f = dict(first_body)
    s = dict(second_body)
    f.pop("replay")
    s.pop("replay")
    assert f == s
    # Second call did NOT add a second batch.
    dst_total = session_db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'dst'"
    ).fetchone()[0]
    assert dst_total == 2


@pytest.mark.asyncio
async def test_move_last_accepts_idempotency_key_header(adapter, session_db):
    """Generic OpenAI-style retry middleware should compose with this endpoint."""
    _seed(session_db, src_n=3, dst_n=0)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/src/move/last",
            json={"dst_session_id": "dst", "count": 1},
            headers={"Idempotency-Key": "hdr-key"},
        )
        body = await resp.json()
    assert resp.status == 200
    assert body["idempotency_key"] == "hdr-key"
    assert body["replay"] is False


# ── move/last validation / error contracts ────────────────────────────


@pytest.mark.asyncio
async def test_move_last_missing_idempotency_key_on_commit_is_400(adapter, session_db):
    _seed(session_db, src_n=2, dst_n=0)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/src/move/last",
            json={"dst_session_id": "dst", "count": 1},
        )
        body = await resp.json()
    assert resp.status == 400
    assert body["error"]["code"] == "missing_idempotency_key"


@pytest.mark.asyncio
async def test_move_last_unknown_src_is_404(adapter, session_db):
    # Only dst exists.
    session_db.create_session(session_id="dst", source="api_server")
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/missing/move/last",
            json={"dst_session_id": "dst", "count": 1, "idempotency_key": "k"},
        )
        body = await resp.json()
    assert resp.status == 404
    assert body["error"]["code"] == "session_not_found"


@pytest.mark.asyncio
async def test_move_last_unknown_dst_is_404(adapter, session_db):
    session_db.create_session(session_id="src", source="api_server")
    session_db.append_message("src", "user", "hi")
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/src/move/last",
            json={"dst_session_id": "ghost", "count": 1, "idempotency_key": "k"},
        )
        body = await resp.json()
    assert resp.status == 404
    assert body["error"]["code"] == "dst_session_not_found"


@pytest.mark.asyncio
async def test_move_last_rejects_same_src_and_dst(adapter, session_db):
    _seed(session_db, src_n=1, dst_n=0)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/src/move/last",
            json={"dst_session_id": "src", "count": 1, "idempotency_key": "k"},
        )
        body = await resp.json()
    assert resp.status == 400
    assert body["error"]["code"] == "same_src_and_dst"


@pytest.mark.asyncio
async def test_move_last_rejects_missing_count(adapter, session_db):
    _seed(session_db, src_n=1, dst_n=0)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/src/move/last",
            json={"dst_session_id": "dst", "idempotency_key": "k"},
        )
        body = await resp.json()
    assert resp.status == 400
    assert body["error"]["code"] == "missing_count"


@pytest.mark.asyncio
async def test_move_last_rejects_non_positive_count(adapter, session_db):
    _seed(session_db, src_n=1, dst_n=0)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        for bad in (0, -1, "abc"):
            resp = await cli.post(
                "/api/sessions/src/move/last",
                json={"dst_session_id": "dst", "count": bad, "idempotency_key": "k"},
            )
            body = await resp.json()
            assert resp.status == 400, (bad, body)
            assert body["error"]["code"] == "invalid_count"


@pytest.mark.asyncio
async def test_move_last_rejects_missing_dst(adapter, session_db):
    _seed(session_db, src_n=1, dst_n=0)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/src/move/last",
            json={"count": 1, "idempotency_key": "k"},
        )
        body = await resp.json()
    assert resp.status == 400
    assert body["error"]["code"] == "missing_dst_session_id"


# ── move/last — replay against already-moved rows ─────────────────────


@pytest.mark.asyncio
async def test_move_last_already_moved_rows_skipped_on_second_distinct_call(adapter, session_db):
    """After commit-1 tombstones the rows, a fresh commit with a NEW key
    must NOT re-move them — the resolver filters ``moved_to_session_id``."""
    _seed(session_db, src_n=3, dst_n=0)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        # Move the last 2 messages out.
        first = await cli.post(
            "/api/sessions/src/move/last",
            json={"dst_session_id": "dst", "count": 2, "idempotency_key": "k1"},
        )
        assert first.status == 200
        # Ask to move "last 5" with a DIFFERENT key — only the 1 remaining
        # active row should travel.
        second = await cli.post(
            "/api/sessions/src/move/last",
            json={"dst_session_id": "dst", "count": 5, "idempotency_key": "k2"},
        )
        second_body = await second.json()
    assert second.status == 200
    assert len(second_body["src_message_ids"]) == 1
    src_active = session_db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'src' AND active = 1"
    ).fetchone()[0]
    assert src_active == 0


# ── move/range happy path + validation ────────────────────────────────


@pytest.mark.asyncio
async def test_move_range_commit_happy_path(adapter, session_db):
    _seed(session_db, src_n=5, dst_n=0)
    # Discover the actual id range for src — append_message returns the
    # row id, but the seed helper doesn't capture it; query directly.
    src_ids = [
        r[0]
        for r in session_db._conn.execute(
            "SELECT id FROM messages WHERE session_id = 'src' ORDER BY id ASC"
        ).fetchall()
    ]
    assert len(src_ids) == 5
    from_id, to_id = src_ids[1], src_ids[3]  # move 3 of 5

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/src/move/range",
            json={
                "dst_session_id": "dst",
                "from_id": from_id,
                "to_id": to_id,
                "idempotency_key": "r1",
            },
        )
        body = await resp.json()

    assert resp.status == 200, body
    assert body["object"] == "hermes.session.move"
    assert body["range_spec"] == f"range:{from_id}..{to_id}"
    assert body["src_message_ids"] == src_ids[1:4]
    assert len(body["dst_message_ids"]) == 3


@pytest.mark.asyncio
async def test_move_range_dry_run_does_not_mutate(adapter, session_db):
    _seed(session_db, src_n=3, dst_n=0)
    src_ids = [
        r[0]
        for r in session_db._conn.execute(
            "SELECT id FROM messages WHERE session_id = 'src' ORDER BY id ASC"
        ).fetchall()
    ]
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/src/move/range",
            json={
                "dst_session_id": "dst",
                "from_id": src_ids[0],
                "to_id": src_ids[-1],
                "dry_run": True,
            },
        )
        body = await resp.json()
    assert resp.status == 200
    assert body["dry_run"] is True
    assert body["dst_message_ids"] == []
    src_active = session_db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'src' AND active = 1"
    ).fetchone()[0]
    assert src_active == 3


@pytest.mark.asyncio
async def test_move_range_rejects_inverted_bounds(adapter, session_db):
    _seed(session_db, src_n=2, dst_n=0)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/src/move/range",
            json={
                "dst_session_id": "dst",
                "from_id": 5,
                "to_id": 2,
                "idempotency_key": "k",
            },
        )
        body = await resp.json()
    assert resp.status == 400
    assert body["error"]["code"] == "invalid_range"


@pytest.mark.asyncio
async def test_move_range_rejects_missing_bounds(adapter, session_db):
    _seed(session_db, src_n=2, dst_n=0)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/src/move/range",
            json={"dst_session_id": "dst", "idempotency_key": "k"},
        )
        body = await resp.json()
    assert resp.status == 400
    assert body["error"]["code"] == "missing_range"


@pytest.mark.asyncio
async def test_move_range_replay_returns_cached_body(adapter, session_db):
    _seed(session_db, src_n=3, dst_n=0)
    src_ids = [
        r[0]
        for r in session_db._conn.execute(
            "SELECT id FROM messages WHERE session_id = 'src' ORDER BY id ASC"
        ).fetchall()
    ]
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        first = await cli.post(
            "/api/sessions/src/move/range",
            json={
                "dst_session_id": "dst",
                "from_id": src_ids[0],
                "to_id": src_ids[1],
                "idempotency_key": "rr",
            },
        )
        second = await cli.post(
            "/api/sessions/src/move/range",
            json={
                "dst_session_id": "dst",
                "from_id": src_ids[0],
                "to_id": src_ids[1],
                "idempotency_key": "rr",
            },
        )
        first_body = await first.json()
        second_body = await second.json()
    assert first.status == 200 and second.status == 200
    assert first_body["replay"] is False
    assert second_body["replay"] is True
    f = dict(first_body)
    s = dict(second_body)
    f.pop("replay")
    s.pop("replay")
    assert f == s


@pytest.mark.asyncio
async def test_move_range_empty_window_commits_logged(adapter, session_db):
    """Range that resolves to zero rows is still a successful commit and
    the move_log row exists so a replay returns byte-equal."""
    _seed(session_db, src_n=2, dst_n=0)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/src/move/range",
            json={
                "dst_session_id": "dst",
                "from_id": 999_000,
                "to_id": 999_999,
                "idempotency_key": "empty",
            },
        )
        body = await resp.json()
    assert resp.status == 200
    assert body["src_message_ids"] == []
    assert body["dst_message_ids"] == []
    # Replay still byte-equal.
    async with TestClient(TestServer(app)) as cli:
        resp2 = await cli.post(
            "/api/sessions/src/move/range",
            json={
                "dst_session_id": "dst",
                "from_id": 999_000,
                "to_id": 999_999,
                "idempotency_key": "empty",
            },
        )
        body2 = await resp2.json()
    assert resp2.status == 200
    assert body2["replay"] is True


# ── auth + posture ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_move_endpoints_require_auth_when_key_configured(auth_adapter, session_db):
    _seed(session_db, src_n=1, dst_n=0)
    app = _create_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        # No bearer token → 401, no mutation.
        resp_last = await cli.post(
            "/api/sessions/src/move/last",
            json={"dst_session_id": "dst", "count": 1, "idempotency_key": "k"},
        )
        assert resp_last.status == 401
        body_last = await resp_last.json()
        assert body_last["error"]["code"] == "invalid_api_key"

        resp_range = await cli.post(
            "/api/sessions/src/move/range",
            json={
                "dst_session_id": "dst",
                "from_id": 1,
                "to_id": 1,
                "idempotency_key": "k",
            },
        )
        assert resp_range.status == 401

        # With the correct token → 200.
        ok = await cli.post(
            "/api/sessions/src/move/last",
            json={"dst_session_id": "dst", "count": 1, "idempotency_key": "k"},
            headers={"Authorization": "Bearer sk-test"},
        )
        assert ok.status == 200

    # Auth-rejected calls left state alone.
    moved = session_db._conn.execute(
        "SELECT COUNT(*) FROM move_log"
    ).fetchone()[0]
    assert moved == 1  # only the authorized call wrote a row


@pytest.mark.asyncio
async def test_move_invalid_idempotency_key_is_400(adapter, session_db):
    _seed(session_db, src_n=1, dst_n=0)
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/src/move/last",
            json={
                "dst_session_id": "dst",
                "count": 1,
                "idempotency_key": "bad\nkey",
            },
        )
        body = await resp.json()
    assert resp.status == 400
    assert body["error"]["code"] == "invalid_idempotency_key"
