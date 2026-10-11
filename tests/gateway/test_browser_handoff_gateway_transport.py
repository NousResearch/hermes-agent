"""Gateway transport acceptance for the browser-handoff Desktop identity bridge.

This is deliberately not a unit test of ``_set_session_context``.  It opens the
real dashboard WebSocket with a one-use server ticket, creates a Desktop session,
and lets an explicit fixture model tool-call dispatch through the profile plugin
registry.  No request/model parameter supplies the provider or subject.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from hermes_cli import plugins
from hermes_cli import web_server
from hermes_cli.dashboard_auth.ws_tickets import _reset_for_tests, mint_ticket
from tui_gateway import server


def _fleet_source_root() -> Path:
    """Return the explicit test-only fleet checkout used by this cross-repo probe."""
    configured = os.environ.get("HERMES_BROWSER_HANDOFF_TEST_FLEET_ROOT")
    if not configured:
        pytest.skip("set HERMES_BROWSER_HANDOFF_TEST_FLEET_ROOT for browser-handoff transport acceptance")
    assert configured is not None
    root = Path(configured)
    if not (root / "image/plugins/browser-handoff").is_dir() or not (root / "services/browser-broker").is_dir():
        pytest.skip("HERMES_BROWSER_HANDOFF_TEST_FLEET_ROOT lacks browser-handoff fixture sources")
    return root


def _plugin_source() -> Path:
    return _fleet_source_root() / "image/plugins/browser-handoff"


def _broker_source() -> Path:
    return _fleet_source_root() / "services/browser-broker"


@pytest.fixture(autouse=True)
def _clean_gateway_state(monkeypatch):
    """Keep the process-global gateway/plugin registries isolated per transport case."""
    original_required = getattr(web_server.app.state, "auth_required", False)
    web_server.app.state.auth_required = True
    _reset_for_tests()
    plugins._reset_plugin_managers_for_tests()
    with server._sessions_lock:
        baseline = set(server._sessions)
    try:
        yield
    finally:
        with server._sessions_lock:
            for sid in set(server._sessions) - baseline:
                server._sessions.pop(sid, None)
        plugins._reset_plugin_managers_for_tests()
        _reset_for_tests()
        web_server.app.state.auth_required = original_required


def _fixture_profile(tmp_path: Path, *, subjects: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    """Make a real profile config, plugin directory and renderer-shaped JSON projection."""
    plugin_source = _plugin_source()
    home = tmp_path / "fixture-profile"
    plugin_dir = home / "plugins" / "browser-handoff"
    plugin_dir.parent.mkdir(parents=True)
    shutil.copytree(plugin_source, plugin_dir)
    # This is the generated deployment ABI consumed by the plugin, rather than
    # a config allowlist.  The fixture is intentionally local and non-secret.
    allowlist = home / "desktop-authorized-subjects.json"
    allowlist.write_text(json.dumps(subjects, sort_keys=True), encoding="utf-8")
    cap = home / "broker.cap"
    cap.write_text("fixture-capability\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled: [browser-handoff]\n"
        "  entries:\n"
        "    browser-handoff:\n"
        "      settings:\n"
        "        handoff_protocol_version: 1\n"
        f"        desktop_authorized_subjects_file: {allowlist}\n"
        "        broker_timeout: 2\n",
        encoding="utf-8",
    )
    return home, allowlist, cap


def _start_real_broker(monkeypatch, tmp_path: Path, *, profile: str, principals: list[dict]):
    """Start the fleet broker HTTP handler; only TLS/viewer auth is out of scope."""
    sys.path[:0] = [str(_broker_source())]
    import origin

    principals_file = tmp_path / "principals.json"
    principals_file.write_text(json.dumps(principals), encoding="utf-8")
    monkeypatch.setenv("HANDOFF_PRINCIPALS_FILE", str(principals_file))
    loaded = origin.load_principals(profile)
    broker = origin.HandoffBroker(loaded, clock=time.time, configured_agent=profile)
    httpd = origin.Server(("127.0.0.1", 0), origin.AdminHandler, {}, None, broker,
                          {"fixture-capability": profile}, loaded)
    httpd.configured_agent = profile
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


class _FixtureModel:
    """The sole model seam: it requests exactly the named registered fixture tool."""

    model = "fixture-no-network"
    provider = "fixture"
    api_mode = "fixture"
    base_url = api_key = platform = _cached_system_prompt = ""
    reasoning_config = service_tier = context_compressor = None
    _config_context_length = 200_000
    session_input_tokens = session_output_tokens = session_prompt_tokens = 0
    session_completion_tokens = session_reasoning_tokens = session_total_tokens = 0
    session_api_calls = 0

    def __init__(self, session_id: str, tool_name: str, scope: str, observed: list[dict], barrier=None):
        self.session_id, self.tool_name, self.scope, self.observed = session_id, tool_name, scope, observed
        self.barrier = barrier
        self.result = None
        self.history = []

    def clear_interrupt(self):
        pass

    def interrupt(self):
        pass

    close = interrupt

    def run_conversation(self, _message, *, conversation_history=None, task_id="", **_kwargs):
        import model_tools
        # This is the production model tool-call choke point.  It obtains the
        # invocation context from the live gateway turn; it is not passed here.
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        result = json.loads(model_tools.handle_function_call(
            self.tool_name, {}, task_id=task_id, session_id=self.session_id,
            tool_call_id="fixture-call"))
        self.result = result
        self.observed.append(result)
        self.history = list(conversation_history or [])
        return {"final_response": json.dumps(result), "messages": self.history, "error": None,
                "interrupted": False, "last_reasoning": None}


def _run_ticketed_tool(monkeypatch, tmp_path, *, allowlist, subject: str, tool: str) -> dict:
    profile = "default"
    home, _allowlist_file, cap = _fixture_profile(tmp_path, subjects=allowlist)
    principals = [{
        "principal_id": "fixture-principal", "access_email": "fixture@example.test",
        "routes": [["telegram", "fixture-user", None]], "agents": [profile],
        "desktop_routes": [["fixture-oidc", subject]],
    }]
    httpd, thread = _start_real_broker(monkeypatch, tmp_path, profile=profile, principals=principals)
    monkeypatch.setenv("BROWSER_HANDOFF_URL", f"http://127.0.0.1:{httpd.server_port}")
    monkeypatch.setenv("BROWSER_HANDOFF_CAP_FILE", str(cap))
    monkeypatch.setenv("HERMES_HOME", str(home))
    # This is the launch profile, not a fabricated named profile.  The server
    # still creates the session through its normal startup path.
    monkeypatch.setattr(server, "_hermes_home", str(home))
    # Real plugin discovery reads temp config and registers a profile-scoped tool.
    plugins.discover_plugins(force=True)
    scope = plugins.get_plugin_manager().scope_key
    observed: list[dict] = []
    monkeypatch.setattr(server, "_make_agent", lambda sid, key, **_kw: _FixtureModel(sid, tool, scope, observed))
    try:
        # The test ticket authority is the server's test-only in-memory signer,
        # not a dashboard token, provider credential, or production identity.
        ticket = mint_ticket(user_id=subject, provider="fixture-oidc")
        with TestClient(web_server.app) as client:
            with client.websocket_connect(f"/api/ws?ticket={ticket}") as ws:
                ready = json.loads(ws.receive_text())
                assert ready["params"]["type"] == "gateway.ready"
                ws.send_text(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session.create",
                                         "params": {"source": "desktop"}}))
                created = next(
                    frame for _ in range(12)
                    if (frame := json.loads(ws.receive_text())).get("id") == 1
                )
                sid = created["result"]["session_id"]
                ws.send_text(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "prompt.submit",
                                         "params": {"session_id": sid, "text": "FIXTURE_CALL"}}))
                # prompt.submit is async; consume frames until the fixture model tool-call ran.
                deadline = time.monotonic() + 10
                while not observed and time.monotonic() < deadline:
                    ws.receive_text()
        assert observed, "fixture model did not dispatch the registered plugin tool"
        return observed[-1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_ticket_subject_maps_through_real_ws_profile_plugin_and_broker(monkeypatch, tmp_path):
    result = _run_ticketed_tool(
        monkeypatch, tmp_path,
        allowlist=[{"profile": "default", "provider": "fixture-oidc", "subject": "mapped-subject"}],
        subject="mapped-subject", tool="browser_handoff_start",
    )
    # A successful start returns the broker handoff payload (rather than an
    # ``ok`` envelope); authorization failures below remain explicit envelopes.
    assert result["session_id"]
    assert result["url"].startswith("https://")


def test_empty_generated_allowlist_denies_without_model_identity(monkeypatch, tmp_path):
    result = _run_ticketed_tool(monkeypatch, tmp_path, allowlist=[], subject="mapped-subject",
                                tool="browser_handoff_status")
    assert result == {"ok": False, "error": "unauthorized_desktop_subject"}


def test_unmatched_ticket_subject_denies_without_model_identity(monkeypatch, tmp_path):
    result = _run_ticketed_tool(
        monkeypatch, tmp_path,
        allowlist=[{"profile": "default", "provider": "fixture-oidc", "subject": "mapped-subject"}],
        subject="unmapped-subject", tool="browser_handoff_end",
    )
    assert result == {"ok": False, "error": "unauthorized_desktop_subject"}


def test_ticket_subject_cannot_attach_or_submit_another_subjects_live_session(monkeypatch, tmp_path):
    """A ticket subject cannot take a live session before it mutates or dispatches."""
    profile = "default"
    home, _allowlist_file, cap = _fixture_profile(tmp_path, subjects=[])
    principals = [{
        "principal_id": "fixture-principal", "access_email": "fixture@example.test",
        "routes": [["telegram", "fixture-user", None]], "agents": [profile],
        "desktop_routes": [["fixture-oidc", "alpha"], ["fixture-oidc", "beta"]],
    }]
    httpd, thread = _start_real_broker(monkeypatch, tmp_path, profile=profile, principals=principals)
    monkeypatch.setenv("BROWSER_HANDOFF_URL", f"http://127.0.0.1:{httpd.server_port}")
    monkeypatch.setenv("BROWSER_HANDOFF_CAP_FILE", str(cap))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", str(home))
    dispatched = []
    monkeypatch.setattr(server, "_make_agent", lambda sid, key, **_kw: _FixtureModel(
        sid, "browser_handoff_status", "fixture", dispatched))

    def receive_reply(ws, request_id):
        return next(frame for _ in range(12)
                    if (frame := json.loads(ws.receive_text())).get("id") == request_id)

    try:
        with TestClient(web_server.app) as client:
            alpha_ticket = mint_ticket(user_id="alpha", provider="fixture-oidc")
            with client.websocket_connect(f"/api/ws?ticket={alpha_ticket}") as alpha:
                assert json.loads(alpha.receive_text())["params"]["type"] == "gateway.ready"
                alpha.send_text(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session.create",
                                             "params": {"source": "desktop"}}))
                created = receive_reply(alpha, 1)["result"]
                sid, stored_id = created["session_id"], created["stored_session_id"]
                session = server._sessions[sid]
                alpha_transport = session["transport"]

                beta_ticket = mint_ticket(user_id="beta", provider="fixture-oidc")
                with client.websocket_connect(f"/api/ws?ticket={beta_ticket}") as beta:
                    assert json.loads(beta.receive_text())["params"]["type"] == "gateway.ready"
                    for request_id, method, params in (
                        (2, "session.activate", {"session_id": sid, "omit_messages": True}),
                        (3, "session.resume", {"session_id": stored_id, "omit_messages": True}),
                        (4, "prompt.submit", {"session_id": sid, "text": "must not run"}),
                    ):
                        beta.send_text(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method,
                                                    "params": params}))
                        assert receive_reply(beta, request_id)["error"]["code"] == 4013
                assert session["transport"] is alpha_transport
                assert session["history"] == []
                assert not dispatched
                # A reconnect with the exact server-verified owner remains a
                # shared live-session viewer and can use the resume fast path.
                alpha_reconnect_ticket = mint_ticket(user_id="alpha", provider="fixture-oidc")
                with client.websocket_connect(f"/api/ws?ticket={alpha_reconnect_ticket}") as alpha_reconnect:
                    assert json.loads(alpha_reconnect.receive_text())["params"]["type"] == "gateway.ready"
                    for request_id, method, params in (
                        (5, "session.activate", {"session_id": sid, "omit_messages": True}),
                        (6, "session.resume", {"session_id": stored_id, "omit_messages": True}),
                    ):
                        alpha_reconnect.send_text(json.dumps({
                            "jsonrpc": "2.0", "id": request_id, "method": method, "params": params}))
                        assert "result" in receive_reply(alpha_reconnect, request_id)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_concurrent_ticket_subjects_do_not_cross_attach_or_leak_context(monkeypatch, tmp_path):
    """Concurrent ticketed turns retain their own server-stamped identity."""
    profile = "default"
    home, _allowlist_file, cap = _fixture_profile(
        tmp_path,
        subjects=[{"profile": profile, "provider": "fixture-oidc", "subject": "alpha"}],
    )
    principals = [{
        "principal_id": "fixture-principal", "access_email": "fixture@example.test",
        "routes": [["telegram", "fixture-user", None]], "agents": [profile],
        # The broker itself recognizes both subjects. Only alpha is in the
        # generated profile projection, so a cross-attached alpha transport
        # would incorrectly authorize beta.
        "desktop_routes": [["fixture-oidc", "alpha"], ["fixture-oidc", "beta"]],
    }]
    httpd, thread = _start_real_broker(monkeypatch, tmp_path, profile=profile, principals=principals)
    monkeypatch.setenv("BROWSER_HANDOFF_URL", f"http://127.0.0.1:{httpd.server_port}")
    monkeypatch.setenv("BROWSER_HANDOFF_CAP_FILE", str(cap))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", str(home))
    plugins.discover_plugins(force=True)
    barrier, observed, models = threading.Barrier(2), [], {}
    scope = plugins.get_plugin_manager().scope_key

    def make_agent(sid, key, **_kw):
        model = _FixtureModel(sid, "browser_handoff_status", scope, observed, barrier)
        models[sid] = model
        return model

    monkeypatch.setattr(server, "_make_agent", make_agent)

    def submit(subject):
        ticket = mint_ticket(user_id=subject, provider="fixture-oidc")
        with TestClient(web_server.app) as client:
            with client.websocket_connect(f"/api/ws?ticket={ticket}") as ws:
                assert json.loads(ws.receive_text())["params"]["type"] == "gateway.ready"
                ws.send_text(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session.create",
                                         "params": {"source": "desktop"}}))
                created = next(frame for _ in range(12)
                               if (frame := json.loads(ws.receive_text())).get("id") == 1)
                sid = created["result"]["session_id"]
                ws.send_text(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "prompt.submit",
                                         "params": {"session_id": sid, "text": "FIXTURE_CALL"}}))
                deadline = time.monotonic() + 10
                while models.get(sid) is None or models[sid].result is None:
                    assert time.monotonic() < deadline, "concurrent fixture model did not dispatch"
                    ws.receive_text()
                return models[sid].result

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            alpha, beta = list(executor.map(submit, ("alpha", "beta")))
        assert alpha != {"ok": False, "error": "unauthorized_desktop_subject"}
        assert beta == {"ok": False, "error": "unauthorized_desktop_subject"}
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)
