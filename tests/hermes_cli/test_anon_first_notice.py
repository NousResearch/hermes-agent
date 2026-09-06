"""CLI one-time notice: an explicit-provider install learns once that the Nous free tier exists."""

import base64
import json
import time

import pytest

from hermes_cli import anon_auth
from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


def _jwt(**claims) -> str:
    def seg(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()
    payload = {"sub": "nas_user:1", "client_id": "nas-anonymous", "account_tier": "anonymous",
               "scope": "inference:invoke tool:invoke", "exp": int(time.time()) + 900, **claims}
    return f"{seg({'alg': 'RS256'})}.{seg(payload)}.sig"


def _seed_guest() -> None:
    with _auth_store_lock():
        store = _load_auth_store()
        store.setdefault("providers", {})["nous"] = {
            "auth_method": anon_auth.ANON_AUTH_METHOD, "account_tier": "anonymous", "anon_token": "anon_0001",
            "client_id": "nas-anonymous", "access_token": _jwt(), "expires_at": "2999-01-01T00:00:00+00:00",
            "inference_base_url": "https://welcome-api.nousresearch.com/v1"}
        _save_auth_store(store)


class _RecordingConsole:
    def __init__(self):
        self.lines: list[str] = []

    def print(self, *args, **kwargs):
        self.lines.append(" ".join(str(a) for a in args))


class _NoticeCLI(CLIAgentSetupMixin):
    """Only what the notice path touches: the console seam."""

    def __init__(self):
        self.console = _RecordingConsole()
        self._app = None

    def _console_print(self, *args, **kwargs):
        self.console.print(*args, **kwargs)


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-store"))
    monkeypatch.delenv("HERMES_FORCE_GUEST", raising=False)


def test_notice_prints_once_after_identity_appears_and_persists_the_flag():
    cli = _NoticeCLI()

    # Background mint has not landed yet: nothing to say.
    cli._maybe_print_free_tier_available_notice()
    assert cli.console.lines == []

    _seed_guest()
    cli._maybe_print_free_tier_available_notice()
    assert len(cli.console.lines) == 1
    text = cli.console.lines[0]
    assert anon_auth.FREE_TIER_AVAILABLE_NOTICE in text
    assert "guest" not in text.lower() and "anonymous" not in text.lower()

    # Same process, and a fresh process reading the store: never again.
    cli._maybe_print_free_tier_available_notice()
    _NoticeCLI()._maybe_print_free_tier_available_notice()
    assert len(cli.console.lines) == 1
    assert _load_auth_store()["providers"]["nous"][anon_auth.GUEST_NOTICE_FLAG] is True
    assert not anon_auth.guest_notice_pending()


def test_signed_in_account_is_never_told_about_the_free_tier():
    with _auth_store_lock():
        store = _load_auth_store()
        store.setdefault("providers", {})["nous"] = {
            "auth_method": "oauth", "access_token": _jwt(client_id="hermes-cli", account_tier="pro"),
            "refresh_token": "rt", "expires_at": "2999-01-01T00:00:00+00:00"}
        _save_auth_store(store)
    cli = _NoticeCLI()

    cli._maybe_print_free_tier_available_notice()

    assert cli.console.lines == []
    assert anon_auth.mark_guest_notice_shown() is False
    assert anon_auth.GUEST_NOTICE_FLAG not in _load_auth_store()["providers"]["nous"]
