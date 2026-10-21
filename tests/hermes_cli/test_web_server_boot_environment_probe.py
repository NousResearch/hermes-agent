"""The desktop/dashboard backend starts the host toolchain probe at boot: Desktop creates a chat's
session when its first message is sent, so that turn would otherwise wait on the probe."""

import sys

from fastapi.testclient import TestClient

from tools import env_probe


def test_backend_lifespan_warms_the_probe_once(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(env_probe, "warm_environment_probe_at_boot", lambda: calls.append(1))
    from hermes_cli import web_server

    with TestClient(web_server.app, raise_server_exceptions=False):
        assert calls == [1]


def test_backend_boots_when_the_probe_cannot_import(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setitem(sys.modules, "tools.env_probe", None)  # the lazy import raises ImportError
    from hermes_cli import web_server

    with TestClient(web_server.app) as client:
        assert client.app is web_server.app
