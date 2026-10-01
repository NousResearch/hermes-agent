"""Tests for POST /api/kanban/create — the HTTP Kanban task-creation endpoint for callers with
no local filesystem access to HERMES_KANBAN_DB (e.g. a Windmill agent_mail flow running in a
momo container, mirroring tools/hermes_kanban_enqueue_endpoint/entry_server.py's validation
contract from qwickapps/aos so its idempotency guarantee doesn't regress).

Covers:
- Auth enforcement (401 with no/invalid Bearer token, same API_SERVER_KEY as every other route)
- Input validation (missing/oversized title/body/assignee/idempotency_key, idempotency_key
  must start with "mail:", invalid JSON/non-object body, invalid board type)
- Real end-to-end task creation against a temp Kanban DB (no mocked kanban_db layer)
- Idempotency: a repeated idempotency_key returns the SAME task_id, never a duplicate row
- Never logs title/body content — only sanitized identifiers
"""

import logging

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware

SECRET_BODY_MARKER = "«redacted:sk-…»"


def _make_adapter(api_key: str = "sk-secret") -> APIServerAdapter:
    config = PlatformConfig(enabled=True, extra={"key": api_key} if api_key else {})
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_post("/api/kanban/create", adapter._handle_kanban_create)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture(autouse=True)
def _isolated_kanban_db(tmp_path, monkeypatch):
    """Every test gets its own temp Kanban root — never the real board. HERMES_KANBAN_HOME
    (not HERMES_KANBAN_DB) so per-board resolution/validation (_normalize_board_slug) still
    engages instead of being short-circuited by a single pinned DB path. Also seeds a fake
    on-disk "tanvi" profile (HERMES_HOME=<root>/profiles/<anything>, so
    kanban_db.list_profiles_on_disk() resolves <root>) so assignee validation has a real
    profile to accept, matching prime's requirement that an unknown assignee 400s."""
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "kanban-root"))
    hermes_root = tmp_path / "hermes-home"
    (hermes_root / "profiles" / "tanvi").mkdir(parents=True)
    (hermes_root / "profiles" / "tanvi" / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(hermes_root / "profiles" / "tanvi"))


def _payload(**overrides):
    base = {
        "title": "Deploy momo relay",
        "body": "Windmill agent_mail fan-out needs a network-callable create path.",
        "assignee": "tanvi",
        "idempotency_key": "mail:msgid-<abc@example.com>",
    }
    base.update(overrides)
    return base


class TestAuth:
    @pytest.mark.asyncio
    async def test_rejects_unauthenticated(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/kanban/create", json=_payload())
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_rejects_wrong_key(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer wrong-key"},
                json=_payload())
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_valid_key_admitted(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=_payload())
            assert resp.status == 200


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_title_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            payload = _payload()
            del payload["title"]
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=payload)
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_blank_title_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=_payload(title="   "))
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_title_too_long_400(self, adapter):
        from gateway.platforms.api_server import MAX_KANBAN_CREATE_TITLE_LENGTH
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=_payload(title="x" * (MAX_KANBAN_CREATE_TITLE_LENGTH + 1)))
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_missing_body_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            payload = _payload()
            del payload["body"]
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=payload)
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_body_too_long_400(self, adapter):
        from gateway.platforms.api_server import MAX_KANBAN_CREATE_BODY_LENGTH
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=_payload(body="x" * (MAX_KANBAN_CREATE_BODY_LENGTH + 1)))
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_missing_assignee_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            payload = _payload()
            del payload["assignee"]
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=payload)
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_assignee_too_long_400(self, adapter):
        from gateway.platforms.api_server import MAX_KANBAN_CREATE_ASSIGNEE_LENGTH
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=_payload(assignee="x" * (MAX_KANBAN_CREATE_ASSIGNEE_LENGTH + 1)))
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_unknown_assignee_400(self, adapter):
        """prime's requirement: an assignee that isn't a real on-disk profile must 4xx, never
        silently create a ready-forever card no dispatcher will ever pick up."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=_payload(assignee="no-such-profile"))
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_assignee_case_insensitive_match(self, adapter):
        """normalize_profile_name lowercases before the on-disk check, same as every other
        Kanban assignment surface."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=_payload(assignee="TANVI"))
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_missing_idempotency_key_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            payload = _payload()
            del payload["idempotency_key"]
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=payload)
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_idempotency_key_too_long_400(self, adapter):
        from gateway.platforms.api_server import MAX_KANBAN_CREATE_IDEMPOTENCY_KEY_LENGTH
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=_payload(
                    idempotency_key="mail:" + "k" * MAX_KANBAN_CREATE_IDEMPOTENCY_KEY_LENGTH))
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_idempotency_key_wrong_prefix_400(self, adapter):
        """Regression contract for the mail pipeline: entry_server.py's `mail:` prefix rule
        must survive the move from the local CLI to this HTTP endpoint."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=_payload(idempotency_key="not-mail:abc"))
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_invalid_json_body_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret", "Content-Type": "application/json"},
                data=b"not json")
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_non_object_body_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=["not", "an", "object"])
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_non_string_board_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=_payload(board=123))
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_invalid_board_slug_400(self, adapter):
        """A malformed board slug (e.g. path-traversal-shaped) must 400, not 500 or escape the
        Kanban root — hermes_cli.kanban_db._normalize_board_slug already enforces this; the
        endpoint must surface it as a client error."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=_payload(board="../../etc"))
            assert resp.status == 400


class TestRealTaskCreation:
    """End-to-end against a real temp Kanban SQLite DB — no mocked kanban_db layer, per the
    house rule that file/DB-touching changes get a real path exercised, not just mocks."""

    @pytest.mark.asyncio
    async def test_creates_real_task(self, adapter):
        from hermes_cli import kanban_db, kanban_db_connect

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=_payload())
            assert resp.status == 200
            data = await resp.json()
            assert data["idempotency_key"] == "mail:msgid-<abc@example.com>"
            task_id = data["task_id"]
            assert task_id

        with kanban_db_connect.connect_closing() as conn:
            task = kanban_db.get_task(conn, task_id)
        assert task is not None
        assert task.title == "Deploy momo relay"
        assert task.assignee == "tanvi"

    @pytest.mark.asyncio
    async def test_repeated_idempotency_key_returns_same_task_never_a_duplicate(self, adapter):
        """The mail pipeline's core contract: resending the same idempotency_key must return
        the existing non-archived task, never create a second one."""
        from hermes_cli import kanban_db, kanban_db_connect

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=_payload())
            assert first.status == 200
            first_id = (await first.json())["task_id"]

            second = await cli.post(
                "/api/kanban/create",
                headers={"Authorization": "Bearer sk-secret"},
                json=_payload(title="A different title — must be ignored"))
            assert second.status == 200
            second_id = (await second.json())["task_id"]

        assert first_id == second_id
        with kanban_db_connect.connect_closing() as conn:
            rows = conn.execute(
                "SELECT COUNT(*) AS n FROM tasks WHERE idempotency_key = ?",
                ("mail:msgid-<abc@example.com>",),
            ).fetchone()
        assert rows["n"] == 1


class TestSanitizedLogging:
    @pytest.mark.asyncio
    async def test_body_never_logged(self, adapter, caplog):
        app = _create_app(adapter)
        with caplog.at_level(logging.DEBUG):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/api/kanban/create",
                    headers={"Authorization": "Bearer sk-secret"},
                    json=_payload(body=SECRET_BODY_MARKER))
                assert resp.status == 200
        for record in caplog.records:
            assert SECRET_BODY_MARKER not in record.getMessage()

    @pytest.mark.asyncio
    async def test_body_never_logged_on_error(self, adapter, caplog, monkeypatch):
        from hermes_cli import kanban_db

        def boom(*args, **kwargs):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(kanban_db, "create_task", boom)
        app = _create_app(adapter)
        with caplog.at_level(logging.DEBUG):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    "/api/kanban/create",
                    headers={"Authorization": "Bearer sk-secret"},
                    json=_payload(body=SECRET_BODY_MARKER))
                assert resp.status == 502
        for record in caplog.records:
            assert SECRET_BODY_MARKER not in record.getMessage()
