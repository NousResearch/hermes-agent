"""End-to-end credential resolution for the update check (real imports, temp ``HERMES_HOME``).

``tests/hermes_cli/test_update_check_github_auth.py`` covers the header logic with everything
monkeypatched. This file exercises the *real* resolution chain instead, because the change it
guards sits on a security boundary (which credential ends up on an outgoing request) and on
network I/O — both call for real imports against a temp ``HERMES_HOME`` rather than mocks.

Chain under test: ``$HERMES_HOME/.env`` → process env → ``agent.secret_scope.get_secret`` →
``tools.skills_hub_github.GitHubAuth`` → the ``Authorization`` header of the request the update
check actually builds. No test here performs a network call: every ``urlopen`` is replaced by a
recorder.
"""

from __future__ import annotations

import email.message
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import hermes_cli.banner as banner
from agent.secret_scope import reset_secret_scope, set_secret_scope
from hermes_cli.env_loader import load_hermes_dotenv

CREDENTIAL = "ghp_e2e_placeholder_token"
ACCEPT_SHA = "application/vnd.github.sha"


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """A throwaway ``HERMES_HOME`` with the ambient credential variables cleared.

    ``monkeypatch.delenv`` also restores whatever the process had at teardown, so a ``.env``
    loaded into ``os.environ`` by one test cannot leak into the next.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    banner._update_github_auth = None
    banner._update_github_401_warned = False
    banner._compare_payload_cache.clear()  # process-wide memo; never let one test seed another
    yield home
    banner._update_github_auth = None
    banner._update_github_401_warned = False
    banner._compare_payload_cache.clear()


def _write_env(home: Path, **values: str) -> None:
    body = "".join(f"{key}={value}\n" for key, value in values.items())
    (home / ".env").write_text(body, encoding="utf-8")


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _recording_urlopen(responses=None):
    """Replace ``urllib.request.urlopen``; return the list it appends each request to."""
    calls: list[dict] = []
    queue = list(responses or [])

    def fake_urlopen(request, timeout=10):
        calls.append({"url": request.full_url, "headers": dict(request.headers), "timeout": timeout})
        if queue:
            outcome = queue.pop(0)
        else:
            outcome = _FakeResponse(b"deadbeef" * 5 + b"\n")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return calls, fake_urlopen


def _authorization(request_record: dict) -> str:
    # urllib canonicalizes header names ("Authorization" stays, "User-Agent" becomes "User-agent").
    for name, value in request_record["headers"].items():
        if name.lower() == "authorization":
            return value
    return ""


# --- the real .env -> header chain ---------------------------------------------------------------


def test_env_file_credential_reaches_the_request_header(temp_home):
    """A token in ``$HERMES_HOME/.env`` must end up in the header the update check builds."""
    _write_env(temp_home, GITHUB_TOKEN=CREDENTIAL)

    loaded = load_hermes_dotenv(hermes_home=temp_home, load_external_secrets=False)
    assert temp_home / ".env" in loaded, "the temp home's .env was not loaded"

    headers = banner._github_api_headers(ACCEPT_SHA)
    assert CREDENTIAL in _authorization({"headers": headers})
    assert headers["Accept"] == ACCEPT_SHA, "our media type must survive credential resolution"
    assert banner._github_auth_source() == "pat"


def test_branch_tip_request_carries_the_env_file_credential(temp_home, monkeypatch):
    """Full path: loaded .env -> the request that ``_github_branch_tip`` really sends."""
    _write_env(temp_home, GITHUB_TOKEN=CREDENTIAL)
    load_hermes_dotenv(hermes_home=temp_home, load_external_secrets=False)

    calls, fake_urlopen = _recording_urlopen()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    sha = banner._github_branch_tip("NousResearch/hermes-agent", "main")

    assert sha is not None, "the fetch should parse the fake branch tip, not fall over"
    assert len(calls) == 1
    assert CREDENTIAL in _authorization(calls[0])
    assert calls[0]["url"].startswith("https://api.github.com/")


# --- profile scope (A -> B -> A) ------------------------------------------------------------------


def test_profile_secret_scope_is_honoured_and_restored(temp_home):
    """The check reads credentials through the profile scope, not the ambient env.

    Two homes in sequence (A -> B -> A): each scope's token must appear on the header while it
    is installed, and A must be back once B is popped — the property multiplexed gateways rely on.
    """
    _write_env(temp_home, GITHUB_TOKEN="ghp_ambient_env")
    load_hermes_dotenv(hermes_home=temp_home, load_external_secrets=False)

    token_a = set_secret_scope({"GITHUB_TOKEN": "ghp_profile_a"})
    try:
        banner._update_github_auth = None  # fresh resolution under scope A
        assert "ghp_profile_a" in _authorization({"headers": banner._github_api_headers(ACCEPT_SHA)})

        token_b = set_secret_scope({"GITHUB_TOKEN": "ghp_profile_b"})
        try:
            banner._update_github_auth = None  # fresh resolution under scope B
            assert "ghp_profile_b" in _authorization({"headers": banner._github_api_headers(ACCEPT_SHA)})
        finally:
            reset_secret_scope(token_b)

        banner._update_github_auth = None  # back under scope A
        assert "ghp_profile_a" in _authorization({"headers": banner._github_api_headers(ACCEPT_SHA)})
    finally:
        reset_secret_scope(token_a)


# --- the other rungs of the ladder ----------------------------------------------------------------


def test_gh_cli_login_is_used_when_no_pat(temp_home, monkeypatch):
    """Without ``GITHUB_TOKEN``/``GH_TOKEN`` the ``gh auth token`` rung still authenticates."""
    # no .env at all in the temp home

    def fake_run(argv, *args, **kwargs):
        if list(argv)[:2] == ["gh", "auth"]:
            return subprocess.CompletedProcess(argv, 0, stdout="gho_from_cli_login\n", stderr="")
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(subprocess, "run", fake_run)

    headers = banner._github_api_headers(ACCEPT_SHA)
    assert "gho_from_cli_login" in _authorization({"headers": headers})
    assert banner._github_auth_source() == "gh-cli"


def test_anonymous_when_nothing_resolves(temp_home, monkeypatch):
    """No credential anywhere: no ``Authorization`` header, and the request still goes out."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("gh")))

    calls, fake_urlopen = _recording_urlopen()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert banner._github_auth_source() == "anonymous"
    assert banner._github_branch_tip("NousResearch/hermes-agent", "main") is not None
    assert len(calls) == 1
    assert _authorization(calls[0]) == ""


# --- a rejected credential ------------------------------------------------------------------------


def test_401_with_env_token_falls_back_anonymously(temp_home, monkeypatch, caplog):
    """A rejected credential is retried anonymously, reported once by source, and never logged."""
    _write_env(temp_home, GITHUB_TOKEN=CREDENTIAL)
    load_hermes_dotenv(hermes_home=temp_home, load_external_secrets=False)

    rejection = urllib.error.HTTPError(
        "https://api.github.com/", 401, "Unauthorized", email.message.Message(), None
    )
    calls, fake_urlopen = _recording_urlopen(
        responses=[rejection, _FakeResponse(b"deadbeef" * 5 + b"\n")]
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with caplog.at_level("WARNING"):
        assert banner._github_branch_tip("NousResearch/hermes-agent", "main") is not None

    assert len(calls) == 2, "the rejected credential must trigger exactly one anonymous retry"
    assert CREDENTIAL in _authorization(calls[0])
    assert _authorization(calls[1]) == "", "the retry must be anonymous"

    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1, "one line per process, not one per call"
    assert "401" in warnings[0].getMessage()
    assert "pat" in warnings[0].getMessage(), "the log must name the credential source"
    assert CREDENTIAL not in caplog.text, "the token itself must never be logged"


def test_compare_uses_the_same_credential_path(temp_home, monkeypatch):
    """``_github_compare`` is the sibling call path — it must not stay anonymous either."""
    _write_env(temp_home, GH_TOKEN=CREDENTIAL)
    load_hermes_dotenv(hermes_home=temp_home, load_external_secrets=False)

    payload = {"status": "behind", "behind_by": 3, "total_commits": 3, "commits": []}
    calls, fake_urlopen = _recording_urlopen(responses=[_FakeResponse(json.dumps(payload).encode())])
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert banner._github_compare("1" * 40, "2" * 40) is not None
    assert len(calls) == 1
    assert CREDENTIAL in _authorization(calls[0])
