"""Web UI sessions must register configured shell hooks and outbound webhooks.

Regression for a Web UI gap where CLI, TUI and messaging sessions invoke the
configured lifecycle hooks but ordinary `hermes dashboard` Web UI sessions did
not — so the persistent Relay observability (and any config-defined shell
hook) silently disappeared for Web UI chats.

The fix registers declarative hooks from cli-config.yaml during the FastAPI
lifespan, mirroring the gateway/CLI call sites. These tests assert that the
web server calls register_from_config (and outbound-webhook registration) at
startup with the loaded config.
"""

import pytest

from hermes_cli import web_server

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient  # noqa: E402


@pytest.fixture
def client():
    previous = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield test_client
    finally:
        if previous is None:
            try:
                delattr(web_server.app.state, "auth_required")
            except AttributeError:
                pass
        else:
            web_server.app.state.auth_required = previous


class TestWebSessionRegistersConfigHooks:
    def test_lifespan_registers_shell_hooks_from_config(self, monkeypatch):
        # The web server's lifespan imports register_from_config from the
        # source module, so patch at the source to intercept the call.
        captured = {}

        def fake_register(cfg, *, accept_hooks):
            captured["cfg"] = cfg
            captured["accept_hooks"] = accept_hooks
            return []

        monkeypatch.setattr(
            "agent.shell_hooks.register_from_config", fake_register
        )
        monkeypatch.setattr(
            "agent.outbound_webhooks.register_from_config",
            lambda cfg: captured.setdefault("outbound", cfg),
        )

        # Entering the TestClient runs the lifespan, which must call the hook
        # registration with the loaded config and consent resolved internally
        # (accept_hooks=False — the web server has no TTY, so consent comes
        # from --accept-hooks / HERMES_ACCEPT_HOOKS / hooks_auto_accept).
        with TestClient(web_server.app) as test_client:
            test_client.headers[web_server._SESSION_HEADER_NAME] = (
                web_server._SESSION_TOKEN
            )
            resp = test_client.get("/api/health")
            assert resp.status_code == 200

        assert captured["accept_hooks"] is False
        assert isinstance(captured["cfg"], dict)
        assert captured["outbound"] is captured["cfg"]
