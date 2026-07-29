"""Tests for the mcp-oauth-remote-gateway optional skill.

Covers the diagnose-oauth-mcp.py decision tree (TOKEN_OK / REFRESH_FIXED /
SESSION_REVOKED / REFRESH_DEAD), the HERMES_HOME resolution fallback, the
atomic --write persistence path, and SKILL.md frontmatter invariants.
No live network calls — urllib is mocked throughout.
"""
from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "mcp"
    / "mcp-oauth-remote-gateway"
)
SCRIPT_PATH = SKILL_DIR / "scripts" / "diagnose-oauth-mcp.py"
SKILL_MD = SKILL_DIR / "SKILL.md"


def load_module():
    spec = importlib.util.spec_from_file_location("diagnose_oauth_mcp", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, status=200, body=b"{}", headers=None):
        self.status = status
        self._body = body
        self.headers = headers or {}

    def read(self):
        return self._body


def _write_token_files(tokens_dir: Path, server="stripe", resource="https://mcp.example.com",
                       refresh_token="rt-1"):
    tokens_dir.mkdir(parents=True, exist_ok=True)
    tok = {
        "access_token": "at-stored",
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": refresh_token,
        "scope": "read",
        "resource": resource,
        "expires_at": 0,
    }
    if refresh_token is None:
        del tok["refresh_token"]
    (tokens_dir / f"{server}.json").write_text(json.dumps(tok))
    (tokens_dir / f"{server}.client.json").write_text(
        json.dumps({"client_id": "cid-1", "token_endpoint_auth_method": "none"})
    )
    return tok


def _run_main(mod, tokens_dir, argv, responses):
    """Run mod.main() with urlopen mocked; returns captured stdout.

    ``responses`` is a list consumed in call order; each item is either a
    FakeResponse or an Exception to raise.
    """
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    with patch.object(mod.os, "environ", dict(mod.os.environ, HERMES_HOME=str(tokens_dir.parent))), \
         patch.object(mod.urllib.request, "urlopen", side_effect=fake_urlopen), \
         patch.object(sys, "argv", ["diagnose-oauth-mcp.py", *argv]):
        # Force the env-var fallback path (ignore any importable hermes_constants).
        with patch.object(mod, "_hermes_home", lambda: str(tokens_dir.parent)):
            buf = io.StringIO()
            from contextlib import redirect_stdout
            with redirect_stdout(buf):
                mod.main()
    return buf.getvalue(), calls


def _init_ok_body():
    return json.dumps({"jsonrpc": "2.0", "id": 1,
                       "result": {"serverInfo": {"name": "x"}, "capabilities": {}}}).encode()


def _init_revoked_error(code=401):
    body = json.dumps({"error": {"code": -32002, "message": "Session expired. Please re-authenticate."}}).encode()
    return urllib.error.HTTPError("https://mcp.example.com", code, "Unauthorized",
                                  {"WWW-Authenticate": 'Bearer error="invalid_token"'},
                                  io.BytesIO(body))


def test_token_ok_branch(tmp_path):
    mod = load_module()
    tokens_dir = tmp_path / "mcp-tokens"
    _write_token_files(tokens_dir)
    out, calls = _run_main(mod, tokens_dir, ["stripe"], [FakeResponse(200, _init_ok_body())])
    assert "BRANCH=TOKEN_OK" in out
    assert len(calls) == 1  # never touched the token endpoint


def test_refresh_dead_no_refresh_token(tmp_path):
    mod = load_module()
    tokens_dir = tmp_path / "mcp-tokens"
    _write_token_files(tokens_dir, refresh_token=None)
    out, _ = _run_main(mod, tokens_dir, ["stripe"], [_init_revoked_error()])
    assert "BRANCH=REFRESH_DEAD" in out


def test_refresh_dead_invalid_grant(tmp_path):
    mod = load_module()
    tokens_dir = tmp_path / "mcp-tokens"
    _write_token_files(tokens_dir)
    grant_err = urllib.error.HTTPError(
        "https://as.example.com/token", 400, "Bad Request", {},
        io.BytesIO(json.dumps({"error": "invalid_grant"}).encode()))
    out, _ = _run_main(
        mod, tokens_dir,
        ["stripe", "--token-endpoint", "https://as.example.com/token"],
        [_init_revoked_error(), grant_err],
    )
    assert "BRANCH=REFRESH_DEAD" in out
    assert "invalid_grant" in out


def test_refresh_fixed_branch_without_write_does_not_persist(tmp_path):
    mod = load_module()
    tokens_dir = tmp_path / "mcp-tokens"
    _write_token_files(tokens_dir)
    refreshed = json.dumps({"access_token": "at-new", "token_type": "Bearer",
                            "expires_in": 7200, "scope": "read"}).encode()
    out, _ = _run_main(
        mod, tokens_dir,
        ["stripe", "--token-endpoint", "https://as.example.com/token"],
        [_init_revoked_error(), FakeResponse(200, refreshed), FakeResponse(200, _init_ok_body())],
    )
    assert "BRANCH=REFRESH_FIXED" in out
    # No --write → stored file untouched
    on_disk = json.loads((tokens_dir / "stripe.json").read_text())
    assert on_disk["access_token"] == "at-stored"
    # Secret values are never printed
    assert "at-new" not in out
    assert "at-stored" not in out
    assert "rt-1" not in out


def test_refresh_fixed_write_persists_atomically(tmp_path):
    mod = load_module()
    tokens_dir = tmp_path / "mcp-tokens"
    _write_token_files(tokens_dir)
    refreshed = json.dumps({"access_token": "at-new", "token_type": "Bearer",
                            "expires_in": 7200, "scope": "read write",
                            "refresh_token": "rt-rotated"}).encode()
    out, _ = _run_main(
        mod, tokens_dir,
        ["stripe", "--token-endpoint", "https://as.example.com/token", "--write"],
        [_init_revoked_error(), FakeResponse(200, refreshed), FakeResponse(200, _init_ok_body())],
    )
    assert "BRANCH=REFRESH_FIXED" in out
    on_disk = json.loads((tokens_dir / "stripe.json").read_text())
    assert on_disk["access_token"] == "at-new"
    assert on_disk["refresh_token"] == "rt-rotated"
    assert on_disk["scope"] == "read write"
    assert on_disk["expires_at"] > 0
    assert not (tokens_dir / "stripe.json.tmp").exists()  # atomic replace, no leftover
    mode = (tokens_dir / "stripe.json").stat().st_mode & 0o777
    assert mode == 0o600


def test_session_revoked_branch(tmp_path):
    mod = load_module()
    tokens_dir = tmp_path / "mcp-tokens"
    _write_token_files(tokens_dir)
    refreshed = json.dumps({"access_token": "at-new", "token_type": "Bearer",
                            "expires_in": 7200}).encode()
    out, _ = _run_main(
        mod, tokens_dir,
        ["stripe", "--token-endpoint", "https://as.example.com/token"],
        [_init_revoked_error(), FakeResponse(200, refreshed), _init_revoked_error()],
    )
    assert "BRANCH=SESSION_REVOKED" in out
    # New token failed too — file must not have been mutated
    on_disk = json.loads((tokens_dir / "stripe.json").read_text())
    assert on_disk["access_token"] == "at-stored"


def test_hermes_home_env_fallback(tmp_path, monkeypatch):
    mod = load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "custom-home"))
    # Block the hermes_constants import so the env fallback is exercised
    with patch.dict(sys.modules, {"hermes_constants": None}):
        home = mod._hermes_home()
    assert home == str(tmp_path / "custom-home")


def test_requests_send_httpx_user_agent(tmp_path):
    """Cloudflare 403s bare urllib UAs — every request must carry the httpx UA."""
    mod = load_module()
    tokens_dir = tmp_path / "mcp-tokens"
    _write_token_files(tokens_dir)
    _, calls = _run_main(mod, tokens_dir, ["stripe"], [FakeResponse(200, _init_ok_body())])
    for req in calls:
        assert req.get_header("User-agent") == mod.UA


def test_skill_md_frontmatter_invariants():
    yaml = pytest.importorskip("yaml")
    content = SKILL_MD.read_text()
    assert content.startswith("---\n")
    fm = yaml.safe_load(re.search(r"^---\n(.*?)\n---", content, re.DOTALL).group(1))
    assert len(fm["description"]) <= 60
    assert fm["description"].endswith(".")
    assert "platforms" in fm and len(fm["platforms"]) >= 1
    assert fm["author"].split(",")[0].strip() != "Hermes Agent"  # human credited first
