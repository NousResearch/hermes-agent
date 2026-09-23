"""Tests for the update check's GitHub credential ladder (``banner._github_api_get``).

The passive update check (CLI banner, startup self-check, dashboard ``/api/hermes/update/check``)
used to be fully anonymous: 60 requests/hour per network address, shared by every install behind
one NAT/VPN/proxy exit (#108804, #112615 for the desktop twin). Reusing the Skills Hub credential
ladder (GITHUB_TOKEN/GH_TOKEN -> gh CLI -> GitHub App -> anonymous) spends the user's 5,000/hour
quota instead. These tests pin that a resolved credential rides on every request, that a rejected
credential (HTTP 401) retries once anonymously with a source-only log line (never the token), and
that the resolution is cached per process. Everything is monkeypatched: no network, no ``.env``,
no real tokens, no ``gh``.
"""

import json
from unittest.mock import MagicMock

import pytest
import urllib.error
import urllib.request

import hermes_cli.banner as banner

SHA_A = "a" * 40


@pytest.fixture
def reset_auth_cache():
    """Reset the module-level credential/401 caches so each test starts clean."""
    banner._update_github_auth = None
    banner._update_github_401_warned = False
    yield
    banner._update_github_auth = None
    banner._update_github_401_warned = False


class _FakeAuth:
    """Fake ``GitHubAuth`` answering the two methods the banner ladder touches."""

    def __init__(self, method="pat", authorization="token FAKE"):
        self._method = method
        self._authorization = authorization

    def auth_method(self):
        return self._method

    def get_headers(self):
        return {"Accept": "application/vnd.github.v3+json",
                **({"Authorization": self._authorization} if self._authorization else {})}


@pytest.fixture
def fake_github_auth(monkeypatch):
    """Route the banner's import to a fake ``GitHubAuth`` constructor."""
    def _install(factory):
        monkeypatch.setattr("tools.skills_hub_github.GitHubAuth", factory)
        banner._update_github_auth = None
    return _install


def test_headers_carry_credential_and_keep_our_media_type(reset_auth_cache, monkeypatch):
    """The resolved credential rides along; our Accept media type and UA are untouched."""
    monkeypatch.setattr("tools.skills_hub_github.GitHubAuth", lambda: _FakeAuth())
    banner._update_github_auth = None

    headers = banner._github_api_headers("application/vnd.github.sha")
    assert headers["Authorization"] == "token FAKE"
    assert headers["Accept"] == "application/vnd.github.sha"
    assert headers["User-Agent"] == "hermes-cli-update-check"


def test_headers_stay_anonymous_when_github_auth_is_unavailable(reset_auth_cache, monkeypatch):
    """GitHubAuth constructor raising must leave the headers anonymous and not raise."""

    def _boom():
        raise RuntimeError("no module")

    monkeypatch.setattr("tools.skills_hub_github.GitHubAuth", _boom)
    banner._update_github_auth = None

    headers = banner._github_api_headers("application/vnd.github.sha")
    assert "Authorization" not in headers
    assert headers["Accept"] == "application/vnd.github.sha"


def test_branch_tip_request_sends_credential(reset_auth_cache, monkeypatch):
    """``_github_branch_tip`` sends the credential on its request and returns the tip SHA."""
    monkeypatch.setattr("tools.skills_hub_github.GitHubAuth", lambda: _FakeAuth())
    banner._update_github_auth = None
    captured = {}
    fake_resp = MagicMock()
    fake_resp.__enter__.return_value.read.return_value = f"  {SHA_A}\n".encode()

    def fake_urlopen(req, timeout=10):
        captured["req"] = req
        return fake_resp

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert banner._github_branch_tip("nousresearch/hermes-agent", "main") == SHA_A
    req = captured["req"]
    assert req.headers.get("Authorization") == "token FAKE"
    assert req.headers.get("Accept") == "application/vnd.github.sha"
    assert req.headers.get("User-Agent", req.headers.get("User-agent")) == "hermes-cli-update-check"
    assert req.full_url == "https://api.github.com/repos/nousresearch/hermes-agent/commits/main"


def test_compare_request_sends_credential(reset_auth_cache, monkeypatch):
    """``_github_compare`` sends the credential and parses the returned JSON."""
    monkeypatch.setattr("tools.skills_hub_github.GitHubAuth", lambda: _FakeAuth())
    banner._update_github_auth = None
    captured = {}
    payload = {"status": "ahead", "ahead_by": 3}
    fake_resp = MagicMock()
    fake_resp.__enter__.return_value.read.return_value = json.dumps(payload).encode()

    def fake_urlopen(req, timeout=10):
        captured["req"] = req
        return fake_resp

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert banner._github_compare(SHA_A, "b" * 40) == payload
    req = captured["req"]
    assert req.headers.get("Authorization") == "token FAKE"
    assert req.headers.get("Accept") == "application/vnd.github+json"
    assert req.headers.get("User-Agent", req.headers.get("User-agent")) == "hermes-cli-update-check"
    assert req.full_url == (f"https://api.github.com/repos/nousresearch/hermes-agent/"
                            f"compare/{SHA_A}...{'b' * 40}")


def test_401_retries_anonymously_once_and_logs_source_not_token(reset_auth_cache, monkeypatch, caplog):
    """A rejected credential (401) retries once anonymously; the warning names the source, never the token."""
    monkeypatch.setattr("tools.skills_hub_github.GitHubAuth", lambda: _FakeAuth(method="pat"))
    banner._update_github_auth = None
    calls = []

    class _Resp:
        def __init__(self, body):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self._body

    def fake_urlopen(req, timeout=10):
        calls.append(req)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)
        return _Resp(SHA_A.encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with caplog.at_level("WARNING", logger="hermes_cli.banner"):
        assert banner._github_api_get("https://api.github.com/x", "application/vnd.github.sha") == SHA_A.encode()

    assert len(calls) == 2
    assert "Authorization" in calls[0].headers
    assert "Authorization" not in calls[1].headers
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "401" in warnings[0].getMessage()
    assert "pat" in warnings[0].getMessage()
    assert "FAKE" not in caplog.text


def test_anonymous_source_produces_a_single_request(reset_auth_cache, monkeypatch):
    """Anonymous: no Authorization header, exactly one request, no retry."""
    monkeypatch.setattr("tools.skills_hub_github.GitHubAuth",
                        lambda: _FakeAuth(method="anonymous", authorization=None))
    banner._update_github_auth = None
    calls = []

    def fake_urlopen(req, timeout=10):
        calls.append(req)
        return MagicMock(**{"__enter__.return_value.read.return_value": SHA_A.encode()})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert banner._github_api_get("https://api.github.com/x", "application/vnd.github.sha") == SHA_A.encode()
    assert len(calls) == 1
    assert "Authorization" not in calls[0].headers


def test_credential_resolution_is_cached_while_the_scope_is_unchanged(reset_auth_cache, monkeypatch):
    """``GitHubAuth`` is constructed once per credential scope: an unchanged scope reuses it."""
    constructions = []

    class _Counting:
        def __init__(self):
            constructions.append(1)

        def auth_method(self):
            return "pat"

        def get_headers(self):
            return {"Authorization": "token FAKE"}

    monkeypatch.setattr("tools.skills_hub_github.GitHubAuth", _Counting)
    monkeypatch.setattr(banner, "_github_credential_key", lambda: ("scope-a", "ghp_a", None))
    banner._update_github_auth = None

    assert banner._github_auth() is not None
    assert banner._github_auth() is not None
    assert len(constructions) == 1


def test_changed_credential_scope_rebuilds_the_resolver(reset_auth_cache, monkeypatch):
    """A profile switch (or a rotated PAT) must not keep serving the earlier credential."""
    constructions = []
    state = {"token": "ghp_scope_a"}

    class _Recording:
        def __init__(self):
            constructions.append(self)

        def auth_method(self):
            return "pat"

        def get_headers(self):
            return {"Authorization": f"token {state['token']}"}

    monkeypatch.setattr("tools.skills_hub_github.GitHubAuth", _Recording)
    monkeypatch.setattr(banner, "_github_credential_key", lambda: ("scope", state["token"], None))
    banner._update_github_auth = None

    accept = "application/vnd.github.sha"
    assert "ghp_scope_a" in banner._github_api_headers(accept)["Authorization"]
    assert len(constructions) == 1

    state["token"] = "ghp_scope_b"
    assert "ghp_scope_b" in banner._github_api_headers(accept)["Authorization"]
    assert len(constructions) == 2, "the changed scope must build a fresh resolver"