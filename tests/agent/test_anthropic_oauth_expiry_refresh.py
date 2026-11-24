"""A session that outlives its Anthropic OAuth access token must refresh it after the 401, not die (#149065).

OAuth access tokens live 8 h. An agent with no bound credential pool (local and desktop sessions) resolves
its token through ``resolve_anthropic_token``, which reads the pool without refreshing. Before the fix the
401 recovery got the same expired token back, so the turn ended with "Non-retryable client error: 401"
although the refresh token was fine. Both wire paths route errors through ``recover_after_classification``.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import pytest

from run_agent import AIAgent

_STALE_ACCESS = "sk-ant-oat01-session-stale"
_STALE_REFRESH = "sk-ant-ort01-session-stale"
_FRESH_ACCESS = "sk-ant-oat01-session-fresh"
_FRESH_REFRESH = "sk-ant-ort01-session-fresh"
_MODEL = "claude-opus-5-5"


def _now_ms() -> int:
    return int(time.time() * 1000)


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HOME and HERMES_HOME; the shared Claude Code credentials file is never reachable."""
    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    from agent import anthropic_credentials as AA
    monkeypatch.setattr(AA, "read_claude_code_credentials", lambda *a, **k: None)
    monkeypatch.setattr(AA, "claude_code_credentials_path", lambda: tmp_path / "no-claude" / ".credentials.json")
    (home / "config.yaml").write_text(
        f"model:\n  default: {_MODEL}\n  provider: anthropic\nauth:\n  adopt_external_logins: false\n"
    )
    return home


def _write_pool(home, *, expires_at_ms: int) -> None:
    (home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {"anthropic": [{
            "id": "sess01", "label": "session", "auth_type": "oauth", "priority": 0,
            "source": "manual:hermes_pkce", "access_token": _STALE_ACCESS, "refresh_token": _STALE_REFRESH,
            "last_status": "ok", "base_url": "https://api.anthropic.com", "expires_at_ms": expires_at_ms,
            "request_count": 0,
        }]},
    }))


def _stored_access_token(home) -> str:
    return json.loads((home / "auth.json").read_text())["credential_pool"]["anthropic"][0]["access_token"]


@pytest.fixture
def refresh_posts(monkeypatch):
    """Stand-in for the refresh POST; records each refresh token spent."""
    from agent import anthropic_credentials as AA
    posts = []

    def _refresh(refresh_token, **_kwargs):
        posts.append(refresh_token)
        return {"access_token": _FRESH_ACCESS, "refresh_token": _FRESH_REFRESH, "expires_at_ms": _now_ms() + 8 * 3600_000}

    monkeypatch.setattr(AA, "refresh_anthropic_oauth_pure", _refresh)
    return posts


@pytest.fixture
def unpooled_agent(hermes_home):
    """An agent holding the pool's token but with no credential pool bound (the local-session shape)."""
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
    ):
        agent = AIAgent(
            model=_MODEL, provider="anthropic", base_url="", api_key=_STALE_ACCESS,
            api_mode="anthropic_messages", credential_pool=None,
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    assert agent._credential_pool is None
    assert agent._is_anthropic_oauth
    return agent


def _expired_401() -> anthropic.AuthenticationError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {
        "type": "authentication_error", "message": "OAuth access token has expired. Re-authenticate to continue.",
    }}
    return anthropic.AuthenticationError(
        f"Error code: 401 - {body}", response=httpx.Response(401, request=request, json=body), body=body,
    )


def _recover(agent, error, retry_state):
    from agent.error_classifier import classify_api_error
    from agent.turn_recovery import recover_after_classification
    classified = classify_api_error(error, provider="anthropic", model=_MODEL)
    with patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()):
        retry_now, _ = recover_after_classification(
            agent, error, classified, retry_state, status_code=error.status_code,
            error_context={}, messages=[], api_messages=[],
        )
    return retry_now


def test_expired_oauth_401_without_bound_pool_refreshes_once_and_retries(hermes_home, refresh_posts, unpooled_agent):
    from agent.turn_retry_state import TurnRetryState
    _write_pool(hermes_home, expires_at_ms=_now_ms() - 60_000)
    retry_state = TurnRetryState()

    assert _recover(unpooled_agent, _expired_401(), retry_state) is True

    assert refresh_posts == [_STALE_REFRESH]
    assert unpooled_agent._anthropic_api_key == _FRESH_ACCESS
    assert unpooled_agent.api_key == _FRESH_ACCESS
    assert _stored_access_token(hermes_home) == _FRESH_ACCESS
    # A second 401 in the same attempt surfaces as before: one refresh per attempt, no loop.
    assert _recover(unpooled_agent, _expired_401(), retry_state) is False
    assert refresh_posts == [_STALE_REFRESH]


def test_request_client_build_does_not_force_refresh_a_pool_row(hermes_home, refresh_posts, unpooled_agent):
    """Only a 401 forces the pool refresh; the per-request credential check leaves pool rows alone."""
    _write_pool(hermes_home, expires_at_ms=_now_ms() - 60_000)

    with patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()):
        unpooled_agent._create_request_anthropic_client(reason="anthropic_stream_request")

    assert refresh_posts == []
    assert _stored_access_token(hermes_home) == _STALE_ACCESS
