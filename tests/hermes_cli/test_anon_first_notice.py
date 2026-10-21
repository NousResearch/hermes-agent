"""The one-time free-tier notice flag: pending once for a guest that carries the trial, then never again."""

import base64
import json
import time

import pytest

from hermes_cli import anon_auth
from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
from tests.hermes_cli.anon_portal import as_local_desktop_backend, claimed_trial, install_portal


def _jwt(**claims) -> str:
    def seg(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()
    payload = {"sub": "nas_user:1", "client_id": "nas-anonymous", "account_tier": "anonymous",
               "scope": "inference:invoke", "exp": int(time.time()) + 900, **claims}
    return f"{seg({'alg': 'RS256'})}.{seg(payload)}.sig"


@pytest.fixture
def portal(monkeypatch, tmp_path):
    return install_portal(monkeypatch, tmp_path)


def test_a_claimed_trial_has_the_notice_pending_once_and_the_flag_persists(portal, monkeypatch):
    claimed_trial(monkeypatch)
    assert anon_auth.guest_notice_pending()

    assert anon_auth.mark_guest_notice_shown() is True

    assert _load_auth_store()["providers"]["nous"][anon_auth.GUEST_NOTICE_FLAG] is True
    assert not anon_auth.guest_notice_pending()


def test_a_connector_only_guest_has_no_notice_pending(portal, monkeypatch):
    as_local_desktop_backend(monkeypatch)
    anon_auth.ensure_portal_identity(explicit=True)

    assert anon_auth.has_guest() and not anon_auth.guest_notice_pending()


def test_signed_in_account_is_never_told_about_the_free_tier(portal):
    with _auth_store_lock():
        store = _load_auth_store()
        store.setdefault("providers", {})["nous"] = {
            "auth_method": "oauth", "access_token": _jwt(client_id="hermes-cli", account_tier="pro"),
            "refresh_token": "rt", "expires_at": "2999-01-01T00:00:00+00:00"}
        _save_auth_store(store)

    assert not anon_auth.guest_notice_pending()
    assert anon_auth.mark_guest_notice_shown() is False
    assert anon_auth.GUEST_NOTICE_FLAG not in _load_auth_store()["providers"]["nous"]
