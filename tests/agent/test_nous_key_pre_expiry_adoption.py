"""Nous agent keys live ~1 h. Every agent in a process must not learn about expiry from its own 401.

In a 1,393-subagent run the hourly expiry produced 620 authentication_error 401s (177 in one hour) and
the credential pool benched the sole credential for every worker at once; the CLI process never
started the proactive keepalive, and nothing adopted a fresh key before a request was sent.
"""
import base64
import json
import time
from unittest.mock import patch

from agent.client_lifecycle import ClientLifecycleMixin


def _jwt(exp: float, sub: str = "acct-A") -> str:
    def b64(o):
        return base64.urlsafe_b64encode(json.dumps(o).encode()).rstrip(b"=").decode()
    return f"{b64({'alg': 'none'})}.{b64({'exp': exp, 'sub': sub})}.sig"


class _Agent(ClientLifecycleMixin):
    def __init__(self, key):
        self.provider, self.api_mode, self.api_key, self.base_url = "nous", "chat_completions", key, "https://inference-api.nousresearch.com/v1"
        self._client_kwargs, self.adopted = {}, []

    def _adopt_openai_credentials(self, api_key, base_url, *, reason):
        self.adopted.append((api_key, reason))
        self.api_key = api_key
        return True


def test_key_far_from_expiry_is_left_alone_without_touching_the_store():
    agent = _Agent(_jwt(time.time() + 3000))
    with patch("hermes_cli.auth.resolve_nous_runtime_credentials", side_effect=AssertionError("must not hit the store")):
        assert agent._adopt_nous_key_before_expiry() is False
    assert agent.adopted == []


def test_key_inside_the_skew_adopts_the_stores_fresh_key_without_forcing_a_refresh():
    agent = _Agent(_jwt(time.time() + 60))
    calls = []

    fresh = _jwt(time.time() + 3600)  # same account A

    def resolve(**kw):
        calls.append(kw)
        return {"api_key": fresh, "base_url": agent.base_url}

    with patch("hermes_cli.auth.resolve_nous_runtime_credentials", side_effect=resolve):
        assert agent._adopt_nous_key_before_expiry() is True
    assert calls[0]["force_refresh"] is False  # the keepalive/peer refresh is adopted, never re-minted
    assert agent.adopted == [(fresh, "nous_credential_refresh")]


def test_a_fresh_key_for_a_different_account_is_never_adopted():
    """Independent-review witness: an explicitly supplied account-A key near expiry was replaced by the
    logged-in singleton's account-B key and the next real request went out as B. Identity is preserved."""
    agent = _Agent(_jwt(time.time() + 60, sub="acct-A"))
    other = _jwt(time.time() + 3600, sub="acct-B")
    with patch("hermes_cli.auth.resolve_nous_runtime_credentials", return_value={"api_key": other, "base_url": agent.base_url}):
        assert agent._adopt_nous_key_before_expiry() is False
    assert agent.adopted == [] and agent.api_key != other


def test_a_key_without_an_account_claim_is_left_alone_proactively():
    agent = _Agent(_jwt(time.time() + 60, sub=""))
    with patch("hermes_cli.auth.resolve_nous_runtime_credentials", side_effect=AssertionError("must not hit the store")):
        assert agent._adopt_nous_key_before_expiry() is False


def test_same_key_back_from_the_store_is_not_readopted():
    """No client rebuild when the store still holds the key in hand (refresh pending elsewhere)."""
    key = _jwt(time.time() + 60)
    agent = _Agent(key)
    with patch("hermes_cli.auth.resolve_nous_runtime_credentials", return_value={"api_key": key, "base_url": agent.base_url}):
        assert agent._adopt_nous_key_before_expiry() is False
    assert agent.adopted == []


def test_keepalive_thread_starts_when_an_agent_routes_to_nous(monkeypatch, tmp_path):
    """Real construction path: the CLI process builds agents through AIAgent, never through the gateway boot."""
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hh"))
    started = []
    monkeypatch.setattr("hermes_cli.nous_auth_keepalive.start_nous_auth_keepalive", lambda: started.append(1))
    AIAgent(api_key="k", base_url="https://inference-api.nousresearch.com/v1", provider="nous",
            model="anthropic/claude-fable-5.1", quiet_mode=True, skip_context_files=True, skip_memory=True)
    assert started == [1]
    AIAgent(api_key="k", base_url="https://openrouter.ai/api/v1", provider="openrouter",
            model="anthropic/claude-fable-5.1", quiet_mode=True, skip_context_files=True, skip_memory=True)
    assert started == [1]
