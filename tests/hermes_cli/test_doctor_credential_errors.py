"""Doctor diagnostics keep malformed credentials out of rows and issue summaries (#144458)."""

from __future__ import annotations

import argparse
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from hermes_cli import doctor_connectivity as dc
from hermes_cli import doctor_live as live


@pytest.fixture
def local_http(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(401)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.05), daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}/models"
    try:
        with httpx.Client(trust_env=False, timeout=2) as client:
            # Dedicated providers have fixed production URLs; keep the real HTTP/header
            # validation path but route every request to this test's loopback server.
            monkeypatch.setattr(httpx, "get", lambda _url, **kwargs: client.get(url, **kwargs))
            yield url
    finally:
        server.shutdown()
        worker.join(timeout=3)
        server.server_close()


def _run_probe(route, key, url, monkeypatch):
    monkeypatch.setenv("AUDIT_API_KEY", key)
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    monkeypatch.setenv("GITHUB_TOKEN", key)
    monkeypatch.setenv("ELEVENLABS_API_KEY", key)
    monkeypatch.setattr("hermes_cli.auth.get_anthropic_key", lambda: key)
    probes = {
        "generic": lambda: dc._probe_apikey_provider("Audit", ("AUDIT_API_KEY",), url, None, True),
        "openrouter": dc._probe_openrouter,
        "anthropic": dc._probe_anthropic,
        "github": dc._probe_github_token,
        "live-bearer": lambda: live._keyed_probe("Audit", url, "AUDIT_API_KEY", "Bearer", 2),
        "live-key": lambda: live._keyed_probe("Audit", url, "AUDIT_API_KEY", "Key", 2),
        "live-xi": lambda: live._probe_audio("tts", {"tts": {"provider": "elevenlabs"}}, 2),
    }
    issues = []
    if route.startswith("live-"):
        result = live._run_one("Audit", probes[route], issues)
        return result.detail, issues
    result = probes[route]()
    return str(result.lines), result.issues


@pytest.mark.parametrize("route", ["generic", "openrouter", "anthropic", "github", "live-bearer", "live-key", "live-xi"])
@pytest.mark.parametrize("token", ["sk-proj-A1B2C3D4E5F6G7H8I9J0aASYNTHETICONLY", "OpaqueSyntheticCredentialA1B2C3D4E5"])
def test_malformed_credentials_are_not_echoed(route, token, local_http, monkeypatch, capsys):
    detail, issues = _run_probe(route, token + "\ninvalid", local_http, monkeypatch)
    output = capsys.readouterr().out
    assert token not in detail + str(issues) + output
    assert "header" in detail.lower()
    assert "check" in detail.lower()


@pytest.mark.parametrize("error_type", [RuntimeError, TimeoutError])
@pytest.mark.parametrize("surface", ["provider", "live", "live-crash"])
def test_other_probe_errors_remain_useful_without_credentials(error_type, surface, monkeypatch, capsys):
    token = "OpaqueSyntheticCredentialA1B2C3D4E5"

    def fail(*args, **kwargs):
        raise error_type(f"connection failed: Authorization: Bearer {token}")

    issues = []
    if surface == "provider":
        monkeypatch.setenv("AUDIT_API_KEY", token)
        monkeypatch.setattr(httpx, "get", fail)
        result = dc._probe_apikey_provider("Audit", ("AUDIT_API_KEY",), "http://unused.invalid", None, True)
        detail = str(result.lines)
    elif surface == "live":
        detail = live._run_one("Audit", fail, issues).detail
    else:
        monkeypatch.setattr(live, "run_live_checks", fail)
        live.maybe_run_live_checks(argparse.Namespace(live=True), issues)
        detail = ""
    output = capsys.readouterr().out
    assert token not in detail + str(issues) + output
    assert "connection failed" in detail + output
