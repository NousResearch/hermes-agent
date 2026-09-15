"""Codex singleton adoption must stay within one ChatGPT principal."""

from __future__ import annotations

import base64
import json

from agent.credential_pool import CredentialPool, PooledCredential


def _jwt_with_claims(claims: dict) -> str:
    def _part(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{_part({'alg': 'none', 'typ': 'JWT'})}.{_part(claims)}.sig"


def test_manual_codex_entry_adopts_singleton_only_for_same_principal(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    account_a_access = _jwt_with_claims({
        "https://api.openai.com/auth": {"chatgpt_account_id": "account-a"},
        "sub": "user-a",
    })
    account_b_other_user_access = _jwt_with_claims({
        "https://api.openai.com/auth": {"chatgpt_account_id": "account-b"},
        "sub": "user-other",
    })
    account_b_old_access = _jwt_with_claims({
        "https://api.openai.com/auth": {"chatgpt_account_id": "account-b"},
        "sub": "user-b",
        "generation": "old",
    })
    account_b_new_access = _jwt_with_claims({
        "https://api.openai.com/auth": {"chatgpt_account_id": "account-b"},
        "sub": "user-b",
        "generation": "new",
    })
    pool_entries = [
        {
            "id": "account-a",
            "label": "account-a",
            "auth_type": "oauth",
            "priority": 0,
            "source": "manual:device_code",
            "access_token": account_a_access,
            "refresh_token": "account-a-refresh",
        },
        {
            "id": "account-b-other-user",
            "label": "account-b-other-user",
            "auth_type": "oauth",
            "priority": 1,
            "source": "manual:device_code",
            "access_token": account_b_other_user_access,
            "refresh_token": "account-b-other-user-refresh",
        },
        {
            "id": "account-b",
            "label": "account-b",
            "auth_type": "oauth",
            "priority": 2,
            "source": "manual:device_code",
            "access_token": account_b_old_access,
            "refresh_token": "account-b-old-refresh",
        },
    ]
    auth_path = hermes_home / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "openai-codex": {
                        "tokens": {
                            "access_token": account_b_new_access,
                            "refresh_token": "account-b-new-refresh",
                        },
                    },
                },
                "credential_pool": {"openai-codex": pool_entries},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    credentials = [
        PooledCredential.from_dict("openai-codex", entry)
        for entry in pool_entries
    ]
    pool = CredentialPool("openai-codex", credentials)

    account_a = pool._sync_entry_from_auth_store(credentials[0])
    account_b_other_user = pool._sync_entry_from_auth_store(credentials[1])
    account_b = pool._sync_entry_from_auth_store(credentials[2])

    assert (account_a.access_token, account_a.refresh_token) == (
        account_a_access,
        "account-a-refresh",
    )
    assert (
        account_b_other_user.access_token,
        account_b_other_user.refresh_token,
    ) == (account_b_other_user_access, "account-b-other-user-refresh")
    assert (account_b.access_token, account_b.refresh_token) == (
        account_b_new_access,
        "account-b-new-refresh",
    )
    stored = json.loads(auth_path.read_text(encoding="utf-8"))
    stored_by_id = {
        entry["id"]: entry
        for entry in stored["credential_pool"]["openai-codex"]
    }
    assert (
        stored_by_id["account-b-other-user"]["access_token"],
        stored_by_id["account-b-other-user"]["refresh_token"],
    ) == (account_b_other_user_access, "account-b-other-user-refresh")
    assert (
        stored_by_id["account-b"]["access_token"],
        stored_by_id["account-b"]["refresh_token"],
    ) == (account_b_new_access, "account-b-new-refresh")
