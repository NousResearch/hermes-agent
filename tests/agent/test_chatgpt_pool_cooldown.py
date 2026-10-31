"""Only the selected, authorized ChatGPT account contributes to pool recovery deadlines."""

import time

import pytest

from agent.credential_pool import EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS, load_pool
from hermes_cli import auth, auth_chatgpt


def _account(account_id, *, now, age=10):
    return {
        "id": account_id, "label": account_id, "source": "manual:chatgpt", "auth_type": "oauth",
        "access_token": f"access-{account_id}", "refresh_token": f"refresh-{account_id}",
        "expires_at_ms": 4102444800000, "last_status": "exhausted",
        "last_status_at": now - age, "last_error_code": 429,
        "chatgpt": {"client_id": f"client-{account_id}", "subject": f"subject-{account_id}",
                    "scopes": ["openid", auth_chatgpt.DIRECT_SCOPE]},
    }


def _save_accounts(rows, active_id):
    auth._save_auth_store({
        "version": 1, "providers": {auth_chatgpt.PROVIDER: {"active_credential_id": active_id}},
        "credential_pool": {auth_chatgpt.PROVIDER: rows},
    })
    return load_pool(auth_chatgpt.PROVIDER)


@pytest.mark.parametrize("other_state", ["inactive", "signed-out"])
@pytest.mark.parametrize("age", [10, 90], ids=["cooling", "recovered"])
@pytest.mark.parametrize("model_delay", [None, 30, 120], ids=["account-only", "short-model", "long-model"])
def test_retained_accounts_do_not_extend_selected_account_cooldown(other_state, age, model_delay):
    now = time.time()
    selected = _account("selected", now=now, age=age)
    if model_delay is not None:
        selected["model_cooldowns"] = {"limited": now + model_delay}
    other = _account("other", now=now)
    other["last_error_reset_at"] = now + 5
    if other_state == "signed-out":
        other.update(access_token="", refresh_token=None, expires_at_ms=None)
    pool = _save_accounts([selected, other], "selected")

    if age < EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS or model_delay is not None:
        assert pool.select(model="limited") is None
        expected = max(selected["last_status_at"] + EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS,
                       now + (model_delay or 0))
        assert pool.next_available_at(model="limited") == pytest.approx(expected, rel=0, abs=0.01)
    else:
        entry = pool.select(model="limited")
        assert entry is not None and entry.id == "selected"
        assert pool.next_available_at(model="limited") is None


@pytest.mark.parametrize("state", ["signed-out", "scope-lost"])
def test_ineligible_accounts_do_not_report_a_recoverable_cooldown(state):
    row = _account("selected", now=time.time())
    if state == "scope-lost":
        row["chatgpt"]["scopes"] = ["openid"]
    pool = _save_accounts([row], None if state == "signed-out" else "selected")

    assert pool.select() is None
    assert pool.next_available_at() is None
