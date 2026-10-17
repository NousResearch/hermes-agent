"""SIWC runtime resolution keeps its bearer on OpenAI and preserves cooldowns."""

import pytest

from agent.credential_pool import load_pool
from hermes_cli import auth, auth_chatgpt
from hermes_cli.auth_constants import AuthError
from hermes_cli.config import save_config
from hermes_cli.runtime_provider import resolve_runtime_provider


@pytest.fixture
def account():
    row = {
        "id": "runtime-account", "label": "Account", "source": "manual:chatgpt",
        "auth_type": "oauth", "priority": 0, "access_token": "local-test-bearer",
        "refresh_token": "local-test-renewal", "expires_at_ms": 4102444800000,
        "chatgpt": {"client_id": "local-client", "subject": "local-subject",
                    "scopes": [auth_chatgpt.DIRECT_SCOPE]},
    }
    auth._save_auth_store({"version": 1,
        "providers": {auth_chatgpt.PROVIDER: {"active_credential_id": row["id"]}},
        "credential_pool": {auth_chatgpt.PROVIDER: [row]}})
    return row


@pytest.mark.parametrize("explicit", [False, True], ids=["saved-model-url", "explicit-url"])
def test_runtime_pins_chatgpt_endpoint_before_any_client_probe(account, explicit):
    save_config({"model": {"provider": auth_chatgpt.PROVIDER, "default": "local-model",
                           "base_url": "https://relay.example/v1"}})
    kwargs = {"explicit_base_url": "https://another-relay.example/v1"} if explicit else {}
    resolved = resolve_runtime_provider(requested=auth_chatgpt.PROVIDER, target_model="local-model", **kwargs)
    assert resolved["provider"] == auth_chatgpt.PROVIDER
    assert resolved["api_key"] == account["access_token"]
    assert resolved["base_url"] == "https://api.openai.com/v1"
    assert resolved["api_mode"] == "codex_responses"


def test_cooling_chatgpt_account_does_not_require_a_new_sign_in(account):
    pool = load_pool(auth_chatgpt.PROVIDER)
    pool.select()
    pool.mark_exhausted_and_rotate(credential_id=account["id"], status_code=429)
    with pytest.raises(AuthError) as caught:
        resolve_runtime_provider(requested=auth_chatgpt.PROVIDER, target_model="local-model")
    error = caught.value
    assert not error.relogin_required
    assert error.retryable is True
    assert 0 < error.retry_after <= 60
    assert error.code == "rate_limit_exceeded"
    assert auth.read_credential_pool(auth_chatgpt.PROVIDER)[0]["refresh_token"] == account["refresh_token"]
