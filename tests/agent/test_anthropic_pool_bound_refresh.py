"""Pre-request Anthropic refresh must keep the pool-bound account on the wire (#118379).

``_try_refresh_anthropic_client_credentials`` runs before every native Anthropic
request. On a pool-bound agent it used to call the global ``resolve_anthropic_token``
(env vars, Claude Code credentials, first pool row) and rebuild the client with
whatever that returned, reverting pool rotation while 429/401 attribution still
pointed at the bound entry.
"""

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from agent import anthropic_adapter, anthropic_credentials
from agent.credential_pool import CredentialPool, PooledCredential, load_pool
from run_agent import AIAgent


BOUND = "sk-ant-oat01-bound-account-test-only"
OTHER = "sk-ant-oat01-other-account-test-only"
CLI = "sk-ant-oat01-claude-code-login-test-only"
RENEWED = "sk-ant-oat01-bound-account-renewed-test-only"


def _entry(entry_id: str, token: str, priority: int, **overrides) -> PooledCredential:
    entry = PooledCredential(
        provider="anthropic", id=entry_id, label=entry_id, auth_type="oauth", priority=priority,
        source="manual:hermes_pkce", access_token=token, refresh_token=f"fake-refresh-{entry_id}",
        expires_at_ms=int((time.time() + 3600) * 1000), base_url="https://api.anthropic.com",
    )
    return replace(entry, **overrides)


@pytest.fixture
def pool(tmp_path, monkeypatch):
    """Two pooled accounts plus a Claude Code login the global resolver would pick first."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cli_dir = tmp_path / ".claude"
    cli_dir.mkdir()
    (cli_dir / ".credentials.json").write_text(json.dumps({
        "claudeAiOauth": {"accessToken": CLI, "expiresAt": int((time.time() + 3600) * 1000)},
    }))
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text("model:\n  provider: anthropic\n")
    (home / "auth.json").write_text(json.dumps({
        "credential_pool": {"anthropic": [_entry("first", OTHER, 0).to_dict(), _entry("second", BOUND, 1).to_dict()]},
        "suppressed_sources": {"anthropic": ["claude_code"]},
    }))
    return load_pool("anthropic")


@pytest.fixture
def no_refresh_post(monkeypatch):
    refresh = Mock(side_effect=AssertionError("a healthy token must not be renewed"))
    monkeypatch.setattr(anthropic_credentials, "refresh_anthropic_oauth_pure", refresh)
    return refresh


@pytest.fixture
def wire(monkeypatch):
    captured = []

    def send(client, request, **kwargs):
        captured.append(request)
        return httpx.Response(200, request=request, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-test",
            "content": [{"type": "text", "text": "OK"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    monkeypatch.setattr(httpx.Client, "send", send)
    return captured


def _agent(pool, entry_id="second"):
    entry = next(e for e in pool.entries() if e.id == entry_id)
    agent = AIAgent.__new__(AIAgent)
    agent.provider, agent.model, agent.api_mode = "anthropic", "claude-test", "anthropic_messages"
    agent.base_url = agent._anthropic_base_url = "https://api.anthropic.com"
    agent.api_key = agent._anthropic_api_key = entry.runtime_api_key
    agent._credential_pool, agent._credential_pool_entry_id = pool, entry.id
    agent._oauth_1m_beta_disabled = False
    agent._anthropic_client = anthropic_adapter.build_anthropic_client(entry.runtime_api_key, agent.base_url)
    return agent


def _send(agent):
    client = agent._create_request_anthropic_client(reason="pool_bound_refresh_test")
    try:
        client.messages.create(model=agent.model, max_tokens=8, messages=[{"role": "user", "content": "OK"}])
    finally:
        agent._close_request_anthropic_client(client, reason="request_error_cleanup")
        agent._anthropic_client.close()


def test_rotated_account_stays_on_the_wire(pool, wire, no_refresh_post):
    """After rotation to the second entry, the next request must not revert to another account."""
    pool.select()  # The pool cursor is on "first"; the agent is bound to "second" (post-rotation).
    agent = _agent(pool, "second")
    _send(agent)
    assert wire[0].headers["Authorization"] == f"Bearer {BOUND}"
    assert agent.api_key == agent._anthropic_api_key == BOUND
    assert agent._credential_pool_entry_id == "second"
    assert pool.current().id == "first"  # A pre-request refresh never moves the cursor.
    no_refresh_post.assert_not_called()


def test_expiring_bound_entry_renews_in_place_and_keeps_attribution(pool, wire, monkeypatch):
    entry = next(e for e in pool.entries() if e.id == "second")
    pool._replace_entry(entry, replace(entry, expires_at_ms=int(time.time() * 1000) - 1))
    refresh = Mock(return_value={
        "access_token": RENEWED, "refresh_token": "fake-refresh-second-next",
        "expires_at_ms": int((time.time() + 3600) * 1000),
    })
    monkeypatch.setattr(anthropic_credentials, "refresh_anthropic_oauth_pure", refresh)
    agent = _agent(pool, "second")
    _send(agent)
    assert wire[0].headers["Authorization"] == f"Bearer {RENEWED}"
    assert agent.api_key == agent._anthropic_api_key == RENEWED
    assert agent._credential_pool_entry_id == "second"
    assert pool.entry_id_for_api_key(agent.api_key) == "second"
    refresh.assert_called_once_with("fake-refresh-second", use_json=True)
    assert agent._try_refresh_anthropic_client_credentials() is False  # Fresh now: no second POST.
    assert refresh.call_count == 1


def test_bound_entry_renewed_elsewhere_is_adopted(pool, no_refresh_post):
    agent = _agent(pool, "second")
    entry = next(e for e in pool.entries() if e.id == "second")
    pool._replace_entry(entry, replace(entry, access_token=RENEWED))
    assert agent._try_refresh_anthropic_client_credentials() is True
    assert agent.api_key == agent._anthropic_api_key == RENEWED
    assert agent._credential_pool_entry_id == "second"
    agent._anthropic_client.close()


def test_key_binding_beats_a_stale_entry_id(pool, no_refresh_post):
    agent = _agent(pool, "second")
    agent._credential_pool_entry_id = "first"
    assert agent._try_refresh_anthropic_client_credentials() is True
    assert agent.api_key == agent._anthropic_api_key == BOUND
    assert agent._credential_pool_entry_id == "second"
    agent._anthropic_client.close()


def test_missing_bound_entry_keeps_its_key_instead_of_borrowing(pool, no_refresh_post):
    agent = _agent(pool, "second")
    agent._credential_pool = CredentialPool("anthropic", [])
    assert agent._try_refresh_anthropic_client_credentials() is False
    assert agent.api_key == agent._anthropic_api_key == BOUND
    agent._anthropic_client.close()


def test_unknown_binding_without_a_key_does_not_fall_back_to_the_cursor(pool, no_refresh_post):
    agent = _agent(pool, "second")
    agent.api_key, agent._credential_pool_entry_id = "", "removed-entry"
    assert agent._try_refresh_anthropic_client_credentials() is False
    assert agent._anthropic_api_key == BOUND
    agent._anthropic_client.close()


@pytest.mark.parametrize("auth_type,token", [("api_key", "sk-ant-api03-test-only"), ("oauth", BOUND)])
def test_unrefreshable_bound_entry_is_not_benched(pool, auth_type, token):
    entry = replace(pool.entries()[1], auth_type=auth_type, access_token=token, refresh_token=None, expires_at_ms=1)
    single = CredentialPool("anthropic", [entry])
    agent = _agent(single, entry.id)
    assert agent._try_refresh_anthropic_client_credentials() is False
    assert single.entries()[0].last_status is None
    assert agent.api_key == agent._anthropic_api_key == token
    agent._anthropic_client.close()


def test_borrowed_claude_code_entry_follows_its_own_file(pool, no_refresh_post, tmp_path):
    """A bound ``claude_code`` row adopts the CLI's out-of-band rotation of that same login."""
    borrowed = _entry("cli", OTHER, 0, source="claude_code", refresh_token="fake-refresh-cli")
    single = CredentialPool("anthropic", [borrowed])
    agent = _agent(single, "cli")
    assert agent._try_refresh_anthropic_client_credentials() is True
    assert agent.api_key == agent._anthropic_api_key == CLI
    assert agent._credential_pool_entry_id == "cli"
    agent._anthropic_client.close()


def test_unbound_agent_keeps_the_global_resolver(pool, monkeypatch):
    agent = _agent(pool, "second")
    agent._credential_pool = None
    monkeypatch.setattr(anthropic_credentials, "resolve_anthropic_token", lambda **_: RENEWED)
    assert agent._try_refresh_anthropic_client_credentials() is True
    assert agent._anthropic_api_key == RENEWED
    agent._anthropic_client.close()


@pytest.mark.parametrize("provider,base", [
    ("minimax", "https://api.minimax.io/anthropic"),
    ("anthropic", "https://example.azure.com/anthropic"),
])
def test_non_native_routes_are_unchanged(pool, provider, base):
    agent = _agent(pool, "second")
    agent.provider, agent._anthropic_base_url = provider, base
    assert agent._try_refresh_anthropic_client_credentials() is False
    assert agent.api_key == agent._anthropic_api_key == BOUND
    agent._anthropic_client.close()
