"""SIWC token generations stay intact across stale pool persistence."""

from copy import deepcopy

import pytest

from agent.credential_pool import load_pool
from hermes_cli import auth, auth_chatgpt


@pytest.fixture
def saved_chatgpt_account():
    row = {
        "id": "registration-one",
        "label": "Personal",
        "source": "manual:chatgpt",
        "auth_type": "oauth",
        "priority": 0,
        "access_token": "access-before",
        "refresh_token": "refresh-before",
        "expires_at_ms": 4102444800000,
        "chatgpt": {
            "client_id": "issued-client-one",
            "subject": "subject-one",
            "scopes": ["openid", auth_chatgpt.DIRECT_SCOPE],
            "id_token": "identity-before",
        },
    }
    auth._save_auth_store({
        "version": 1,
        "providers": {auth_chatgpt.PROVIDER: {"active_credential_id": row["id"]}},
        "credential_pool": {auth_chatgpt.PROVIDER: [row]},
    })
    return row


@pytest.mark.parametrize("access_token,metadata", [
    ("access-after", {"scopes": ["openid"], "id_token": "identity-after"}),
    ("access-before", {"pending_refresh": {
        "response": {"access_token": "access-after", "refresh_token": "refresh-after",
                     "id_token": "identity-to-validate", "expires_in": 3600,
                     "scope": "openid " + auth_chatgpt.DIRECT_SCOPE},
        "received_at_ms": 1900000000000,
    }}),
], ids=["scope-downgrade", "pending-identity-validation"])
def test_stale_flush_adopts_peer_chatgpt_generation(saved_chatgpt_account, access_token, metadata):
    stale = load_pool(auth_chatgpt.PROVIDER)
    peer = deepcopy(saved_chatgpt_account)
    peer.update(access_token=access_token, refresh_token="refresh-after")
    peer["chatgpt"].update(metadata)
    if "pending_refresh" in metadata:
        peer["expires_at_ms"] = 0
    auth.write_credential_pool(auth_chatgpt.PROVIDER, [peer])

    stale._persist()

    stored = auth.read_credential_pool(auth_chatgpt.PROVIDER)[0]
    live = stale.entries()[0]
    assert (stored["access_token"], stored["refresh_token"], stored["chatgpt"]) == (
        access_token, "refresh-after", peer["chatgpt"])
    assert (live.access_token, live.refresh_token, live.extra["chatgpt"]) == (
        access_token, "refresh-after", peer["chatgpt"])


@pytest.mark.parametrize("recovery,flushes", [
    ("none", 1), ("none", 2), ("forced-refresh", 2), ("transient-recovery", 2),
])
def test_stale_flush_cannot_restore_logged_out_chatgpt_tokens(
    saved_chatgpt_account, monkeypatch, recovery, flushes,
):
    stale = load_pool(auth_chatgpt.PROVIDER)
    old_entry = stale.entries()[0]
    calls = []

    def refresh(entry):
        calls.append(entry.refresh_token)
        return {"access_token": "access-after", "refresh_token": "refresh-after",
                "expires_at_ms": 4102444800000, "chatgpt": dict(entry.extra["chatgpt"])}

    monkeypatch.setattr(auth_chatgpt, "refresh_credential", refresh)
    monkeypatch.setattr(auth_chatgpt, "_revoke", lambda _registration: False)
    auth_chatgpt._logout(saved_chatgpt_account)

    result = None
    if recovery == "forced-refresh":
        result = stale.try_refresh_matching(credential_id=old_entry.id)
    elif recovery == "transient-recovery":
        result = stale._recover_failed_refresh(old_entry, RuntimeError("offline"))
    for _ in range(flushes):
        stale._persist()

    assert result is None
    assert calls == []
    stored = auth.read_credential_pool(auth_chatgpt.PROVIDER)[0]
    assert not stored.get("access_token") and not stored.get("refresh_token")
    assert "id_token" not in stored["chatgpt"]
    assert stored["chatgpt"]["client_id"] == saved_chatgpt_account["chatgpt"]["client_id"]
    assert not auth.get_provider_auth_state(auth_chatgpt.PROVIDER).get("active_credential_id")
