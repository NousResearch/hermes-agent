"""The Anthropic model flow must reuse a login that lives only in the credential pool.

``hermes auth add anthropic --type oauth`` stores the login as a ``manual:hermes_pkce`` pool row
and writes no ``.anthropic_oauth.json``. The runtime resolves it from the pool, so
``hermes model`` / ``hermes fallback add`` must offer it instead of demanding a new login.
Real path: a real ``auth.json`` pool row in the isolated ``HERMES_HOME``; only the terminal
prompts and the host's Claude Code credentials file are stubbed.
"""
import time
from unittest.mock import MagicMock, patch

import hermes_cli.model_setup_flows as flows
from hermes_cli.auth import write_credential_pool

POOL_TOKEN = "sk-ant-oat01-pooltoken"


class _PickerReached(Exception):
    """Sentinel: the flow got past authentication and into model selection."""


def _add_pool_only_login():
    write_credential_pool("anthropic", [{
        "id": "pool01", "label": "anthropic-oauth-1", "auth_type": "oauth", "priority": 0,
        "source": "manual:hermes_pkce", "access_token": POOL_TOKEN,
        "refresh_token": "sk-ant-ort01-refresh", "base_url": "https://api.anthropic.com",
        "expires_at_ms": int((time.time() + 3600) * 1000),
    }])


def _run_flow(choice):
    _add_pool_only_login()
    authenticate = MagicMock(return_value=True)
    prompt = MagicMock(return_value=choice)
    with (
        patch("agent.anthropic_credentials.read_claude_code_credentials", return_value=None),
        patch.object(flows, "_prompt_auth_credentials_choice", prompt),
        patch.object(flows, "_anthropic_authenticate", authenticate),
        patch.object(flows, "_pick_model_or_prompt", side_effect=_PickerReached),
    ):
        try:
            flows._model_flow_anthropic({}, "")
        except _PickerReached:
            pass
    return authenticate, prompt


def test_pool_only_login_is_reused_without_a_new_login(capsys):
    authenticate, prompt = _run_flow("use")
    authenticate.assert_not_called()
    prompt.assert_called_once()
    assert POOL_TOKEN not in capsys.readouterr().out  # a pool secret is never echoed


def test_pool_login_still_offers_an_explicit_reauth():
    authenticate, prompt = _run_flow("reauth")
    prompt.assert_called_once()
    authenticate.assert_called_once()
