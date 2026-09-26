"""``hermes doctor`` validates a configured GITHUB_TOKEN/GH_TOKEN against api.github.com (#115257).

An expired PAT left in ``.env`` used to be reported as "GitHub token configured" while every
git-auth clone failed with a Git-level message that never named the token. The doctor now sends
the token to the REST API and, on a refusal, names the variable and the ``.env`` file carrying it.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hermes_cli import doctor_connectivity as dc

# GitHub's documented 401 body for a bad/expired token (REST API "Authentication" docs).
_BAD_CREDENTIALS = {"message": "Bad credentials", "documentation_url": "https://docs.github.com/rest"}
_NOT_ACCESSIBLE = {"message": "Resource not accessible by integration", "documentation_url": "https://docs.github.com/rest"}


@pytest.fixture
def github_stand_in(monkeypatch):
    seen: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        status = 401

        def do_GET(self):  # noqa: N802 - http.server API
            auth = self.headers.get("Authorization") or ""
            seen.append({"path": self.path, "authorization": auth})
            status = self.status
            if status == 200 and self.path == "/user" and auth.startswith("Bearer ghs_"):
                # GitHub's documented answer for an App installation token on /user.
                status, body = 403, json.dumps(_NOT_ACCESSIBLE).encode()
            else:
                body = json.dumps(_BAD_CREDENTIALS if status == 401 else {"login": "octocat"}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002 - http.server API; quiet
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    # Keep the production path so the stand-in can tell /user from /rate_limit.
    from urllib.parse import urlsplit
    path = urlsplit(dc.GITHUB_API_PROBE_URL).path
    monkeypatch.setattr(dc, "GITHUB_API_PROBE_URL", f"http://127.0.0.1:{srv.server_port}{path}")
    yield Handler, seen
    srv.shutdown()


def _github_probe_row(monkeypatch, tmp_path, dotenv: str | None, *, name: str = "home"):
    """Drive the production entry point: the doctor's probe table, run by ``run_probes``."""
    home = tmp_path / name
    home.mkdir()
    if dotenv is not None:
        (home / ".env").write_text(dotenv, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(dc, "_APIKEY_PROVIDERS_CACHE", [])
    probes = [(label, fn) for label, fn in dc.build_probes() if label == "GitHub token"]
    assert probes, "GitHub token probe missing from the doctor's connectivity table"
    (result,) = dc.run_probes(probes)
    return result


def test_expired_dotenv_token_is_rejected_and_the_env_file_is_named(monkeypatch, tmp_path, github_stand_in):
    handler, seen = github_stand_in
    handler.status = 401
    result = _github_probe_row(monkeypatch, tmp_path, "GITHUB_TOKEN=ghp_expired000000000000000000000000000000\n")
    ((glyph, _label, detail),) = result.lines
    assert "✗" in glyph
    assert "GITHUB_TOKEN" in detail and "/.env" in detail and "rejected" in detail
    assert result.issues and "GITHUB_TOKEN" in result.issues[0] and "/.env" in result.issues[0]
    # The real token went to the API (not a synthetic pass) and never leaks into the row text.
    assert seen and seen[0]["authorization"] == "Bearer ghp_expired000000000000000000000000000000"
    assert "ghp_expired" not in detail and "ghp_expired" not in result.issues[0]


def test_valid_token_is_ok_and_no_token_is_skipped(monkeypatch, tmp_path, github_stand_in):
    handler, seen = github_stand_in
    handler.status = 200
    result = _github_probe_row(monkeypatch, tmp_path, "GH_TOKEN=gho_valid00000000000000000000000000000000\n")
    ((glyph, _label, detail),) = result.lines
    assert "✓" in glyph and "GH_TOKEN" in detail and not result.issues

    seen.clear()
    result = _github_probe_row(monkeypatch, tmp_path, None, name="home-without-token")
    assert result.lines == [] and result.issues == [] and seen == []  # nothing configured: no request, no row


def test_actions_installation_token_is_accepted(monkeypatch, tmp_path, github_stand_in):
    """A ``ghs_`` App installation token (the GITHUB_TOKEN every Actions job exports) is valid yet
    GitHub answers 403 on /user; the doctor must probe an endpoint every token type can reach."""
    handler, seen = github_stand_in
    handler.status = 200
    result = _github_probe_row(monkeypatch, tmp_path, "GITHUB_TOKEN=ghs_install000000000000000000000000000000\n")
    ((glyph, _label, detail),) = result.lines
    assert "✓" in glyph and "GITHUB_TOKEN" in detail and not result.issues, (detail, seen)
