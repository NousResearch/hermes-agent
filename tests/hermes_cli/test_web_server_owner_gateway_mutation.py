"""A standalone serve's fenced sidebar mutation rides a ready gateway owner (#140118).

The two-process layout (``hermes serve`` + ``hermes gateway run``) leaves no process with
the session authority reachable by the desktop, so the offline-maintenance writer always
collides with the live owner's lock. The fix: discovery finds the ready owner and sends the
exact ``session.mutate`` the CLI sends; anything but a ready owner keeps the offline path.
"""

from contextlib import asynccontextmanager
from uuid import uuid4

import pytest


class _OwnerRPC:
    """The control-socket client as the mutation path sees it: one ``session.mutate``."""

    def __init__(self, reply=None, error=None):
        self.calls = []
        self._reply = reply if reply is not None else {
            'session_id': 's1', 'operation': 'sidebar', 'revision': 4, 'execution_generation': 3}
        self._error = error

    async def rpc(self, method, **params):
        from hermes_cli.gateway_client import GatewayRPCError

        self.calls.append((method, params))
        if self._error is not None:
            raise GatewayRPCError(self._error)
        return dict(self._reply)


def _owner_discovery(monkeypatch, owner, *, state="ready"):
    from hermes_cli import gateway_client, gateway_runtime

    monkeypatch.setattr(gateway_runtime, "discover_gateway_endpoint",
                        lambda home, **kwargs: type("Found", (), {"state": state, "endpoint": object()})())

    @asynccontextmanager
    async def _connect(endpoint=None):
        yield owner

    monkeypatch.setattr(gateway_client, "connect_gateway", _connect)
    return owner


class TestStandaloneMutationDelegatesToReadyOwner:
    @pytest.fixture(autouse=True)
    def _setup_client(self, monkeypatch, _isolate_hermes_home):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        from hermes_state import SessionDB

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
        db = SessionDB()
        try:
            db.create_session(session_id="s1", source="cli")
            db.append_message(session_id="s1", role="user", content="hi")
        finally:
            db.close()
        client = TestClient(app)
        client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        self.client = client

    def _patch(self, payload):
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            row = db.get_session("s1")
        finally:
            db.close()
        return self.client.patch("/api/sessions/s1", json={
            **payload, "request_id": uuid4().hex, "expected_revision": row["runtime_revision"]})

    def test_ready_owner_receives_the_fenced_sidebar_mutation(self, monkeypatch):
        owner = _owner_discovery(monkeypatch, _OwnerRPC())
        resp = self._patch({"archived": True})
        assert resp.status_code == 200, resp.text
        assert resp.json()["ok"] is True
        method, params = owner.calls[0]
        assert method == "session.mutate"
        assert params["operation"] == "sidebar"
        assert params["payload"] == {"archived": True}
        assert params["session_id"] == "s1"
        assert params["expected_revision"] is not None
        assert params["request_id"]

    def test_without_a_ready_owner_the_offline_writer_still_applies(self, monkeypatch):
        owner = _owner_discovery(monkeypatch, _OwnerRPC(), state="stopped")
        resp = self._patch({"unread": True})
        assert resp.status_code == 200, resp.text
        assert resp.json()["unread"] is True
        assert owner.calls == []

    def test_owner_refusal_keeps_its_http_status(self, monkeypatch):
        _owner_discovery(monkeypatch, _OwnerRPC(error="revision_conflict"))
        resp = self._patch({"archived": True})
        assert resp.status_code == 409
        assert resp.json()["detail"] == "revision_conflict"

    def test_owner_permission_denied_is_403(self, monkeypatch):
        _owner_discovery(monkeypatch, _OwnerRPC(error="permission_denied"))
        resp = self._patch({"archived": True})
        assert resp.status_code == 403
        assert resp.json()["detail"] == "permission_denied"

    def test_undialable_ready_owner_never_falls_back_to_the_refused_lock(self, monkeypatch):
        from hermes_cli import gateway_client, gateway_runtime

        monkeypatch.setattr(gateway_runtime, "discover_gateway_endpoint",
                            lambda home, **kwargs: type("Found", (), {"state": "ready", "endpoint": object()})())

        @asynccontextmanager
        async def _down(endpoint=None):
            raise gateway_client.GatewayClientError("Gateway connection failed; no local fallback")
            yield

        monkeypatch.setattr(gateway_client, "connect_gateway", _down)
        resp = self._patch({"archived": True})
        assert resp.status_code == 503
        assert resp.json()["detail"] == "owner_unavailable"
