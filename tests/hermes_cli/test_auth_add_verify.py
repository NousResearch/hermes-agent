"""``hermes auth add`` checks an API key once against the provider before storing it."""

from __future__ import annotations

import http.server
import json
import threading
from types import SimpleNamespace

import pytest
import hermes_yaml as yaml


class _Provider(http.server.BaseHTTPRequestHandler):
    status = 200
    body: dict | str = {"data": [{"id": "m"}]}

    def do_GET(self):
        payload = self.body.encode() if isinstance(self.body, str) else json.dumps(self.body).encode()
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def provider_home(tmp_path, monkeypatch):
    """A HERMES_HOME whose ``acme`` provider points at a local stand-in server."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _Provider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({"providers": {"acme": {
        "name": "Acme", "base_url": f"http://127.0.0.1:{server.server_port}/v1"}}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home
    server.shutdown()


def _add(**extra):
    from hermes_cli.auth_commands import auth_add_command
    auth_add_command(SimpleNamespace(provider="acme", auth_type="api-key", api_key="sk-acme", label="k", **extra))


def _pool(home) -> list:
    path = home / "auth.json"
    return json.loads(path.read_text(encoding="utf-8-sig"))["credential_pool"].get("acme", []) if path.exists() else []


def test_rejected_key_is_refused_and_not_stored(provider_home, monkeypatch, capsys):
    monkeypatch.setattr(_Provider, "status", 401)
    monkeypatch.setattr(_Provider, "body", {"error": {"message": "Invalid API key provided"}})
    with pytest.raises(SystemExit) as exc:
        _add()
    assert "Invalid API key provided" in str(exc.value.code)
    assert _pool(provider_home) == []

    _add(no_verify=True)  # the offline/CI escape hatch stores it unchecked
    assert [e["access_token"] for e in _pool(provider_home)] == ["sk-acme"]


@pytest.mark.parametrize("status,message,body", [
    (402, "Insufficient credits", {"error": {"message": "Insufficient credits"}}),
    (403, "Key limit exceeded (total limit)", {"error": {"message": "Key limit exceeded (total limit)"}}),
    (403, "Insufficient credits", "Insufficient credits"),  # plain-text body, no JSON
])
def test_out_of_credit_key_is_stored_benched(provider_home, monkeypatch, capsys, status, message, body):
    monkeypatch.setattr(_Provider, "status", status)
    monkeypatch.setattr(_Provider, "body", body)
    _add()
    (entry,) = _pool(provider_home)
    assert (entry["last_status"], entry["last_error_code"], entry["failure_reason"]) == ("exhausted", status, "billing")
    assert message in capsys.readouterr().err
