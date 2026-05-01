"""Tests for the lightweight Hermes ACP client."""

from __future__ import annotations

import io
import json
import threading
from types import SimpleNamespace

from acp_adapter.client import (
    HermesACPClient,
    _select_auth_method_id,
    build_hermes_acp_env,
    build_hermes_acp_command,
)


def test_select_auth_method_id_uses_preferred_match():
    init_result = {"authMethods": [{"id": "openrouter"}, {"id": "anthropic"}]}

    assert _select_auth_method_id(init_result, preferred_method="anthropic") == "anthropic"


def test_select_auth_method_id_falls_back_to_first_method():
    init_result = {"authMethods": [{"id": "openrouter"}, {"id": "anthropic"}]}

    assert _select_auth_method_id(init_result) == "openrouter"


def test_select_auth_method_id_returns_none_without_methods():
    assert _select_auth_method_id({}) is None
    assert _select_auth_method_id({"authMethods": []}) is None


def test_build_command_includes_default_profile_for_python_launcher():
    cmd = build_hermes_acp_command("default", hermes_bin="/usr/bin/python3")

    assert cmd == ["/usr/bin/python3", "-m", "hermes_cli.main", "-p", "default", "acp"]


def test_build_command_includes_profile_for_named_profile():
    cmd = build_hermes_acp_command("researcher", hermes_bin="/usr/local/bin/hermes")

    assert cmd == ["/usr/local/bin/hermes", "-p", "researcher", "acp"]


def test_permission_request_auto_denies_by_default():
    client = HermesACPClient("default")
    out = io.StringIO()
    client.proc = SimpleNamespace(stdin=out, poll=lambda: None)
    client._lock = threading.Lock()

    client._respond_to_server_request(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "session/requestPermission",
            "params": {},
        }
    )

    payload = json.loads(out.getvalue())
    assert payload["id"] == 7
    assert payload["result"]["outcome"]["outcome"] == "cancelled"


def test_client_marks_acp_process_to_use_profile_toolsets(monkeypatch):
    captured = {}

    class FakePopen:
        stdin = io.StringIO()
        stdout = io.StringIO()
        stderr = io.StringIO()

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs["env"]

        def poll(self):
            return None

    monkeypatch.setattr("acp_adapter.client.subprocess.Popen", FakePopen)
    monkeypatch.setattr("acp_adapter.client.threading.Thread", lambda *a, **kw: SimpleNamespace(start=lambda: None))

    client = HermesACPClient("default", hermes_bin="/usr/bin/python3")
    client._start()

    assert captured["env"]["HERMES_ACP_USE_PROFILE_TOOLSETS"] == "1"


def test_python_launcher_env_preserves_checkout_on_pythonpath(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)

    env = build_hermes_acp_env(hermes_bin="/usr/bin/python3")

    assert env["HERMES_ACP_USE_PROFILE_TOOLSETS"] == "1"
    assert "hermes-agent" in env["PYTHONPATH"]


def test_start_replaces_exited_process(monkeypatch):
    starts = []

    class DeadProcess:
        def poll(self):
            return 1

    class FakePopen:
        stdin = io.StringIO()
        stdout = io.StringIO()
        stderr = io.StringIO()

        def __init__(self, cmd, **kwargs):
            starts.append(cmd)

        def poll(self):
            return None

    monkeypatch.setattr("acp_adapter.client.subprocess.Popen", FakePopen)
    monkeypatch.setattr("acp_adapter.client.threading.Thread", lambda *a, **kw: SimpleNamespace(start=lambda: None))

    client = HermesACPClient("default", hermes_bin="/usr/bin/python3")
    client.proc = DeadProcess()
    client._start()

    assert len(starts) == 1


def test_load_session_fails_when_server_returns_null():
    class NullLoadClient(HermesACPClient):
        def _send_rpc_raw(self, method, params=None, timeout=120.0):
            return None

    client = NullLoadClient("default")

    try:
        client.load_session("missing-session")
    except RuntimeError as exc:
        assert "missing-session" in str(exc)
    else:
        raise AssertionError("load_session should fail on null ACP result")
