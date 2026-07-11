"""Regression for #61276: api_server agent entry under multiplex isolation.

When gateway.multiplex_profiles is on, get_secret fails closed without a
profile secret scope. Messaging platforms install scope via
GatewayRunner._profile_runtime_scope; api_server did not, so SSE chat
completions crashed on OPENROUTER_BASE_URL (and any other credential)
resolution.

These tests code-simulate the bug class — no live gateway or network.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent import secret_scope as ss
from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _api_server_profile_secret_scope,
)


@pytest.fixture(autouse=True)
def _reset_multiplex():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)
    # Ensure no leaked scope from a failed test.
    if ss.current_secret_scope() is not None:
        # best-effort: cannot reset without token; flag off is enough for
        # subsequent tests that check is_multiplex_active.
        pass


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True))


class TestApiServerProfileSecretScope:
    def test_noop_when_multiplex_off(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://from-environ.example/v1")
        with _api_server_profile_secret_scope():
            # Legacy path: unscoped get_secret reads os.environ.
            assert ss.get_secret("OPENROUTER_BASE_URL") == "https://from-environ.example/v1"
        assert ss.current_secret_scope() is None

    def test_installs_and_tears_down_under_multiplex(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text(
            "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: tmp_path,
        )
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://leak.example/v1")
        ss.set_multiplex_active(True)

        with _api_server_profile_secret_scope():
            assert ss.current_secret_scope() is not None
            # Profile .env wins; process env must not leak through.
            assert ss.get_secret("OPENROUTER_BASE_URL") == "https://openrouter.ai/api/v1"

        assert ss.current_secret_scope() is None
        with pytest.raises(ss.UnscopedSecretError):
            ss.get_secret("OPENROUTER_BASE_URL")


@pytest.mark.asyncio
async def test_run_agent_scopes_secret_reads_under_multiplex(
    adapter, tmp_path, monkeypatch,
):
    """Code-sim repro of #61276: _create_agent path reads OPENROUTER_BASE_URL.

    Without the scope install inside the executor thread, this raises
    UnscopedSecretError during SSE agent setup.
    """
    (tmp_path / ".env").write_text(
        "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    observed: dict = {}

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, session_id: str):
            self.session_id = session_id

        def run_conversation(self, user_message, conversation_history, task_id):
            # Mid-run credential reads (tool/MCP/provider) must also see scope.
            observed["during_run"] = ss.get_secret("OPENROUTER_BASE_URL")
            return {"final_response": "ok"}

    def fake_create_agent(**kwargs):
        # Mirrors _resolve_runtime_agent_kwargs -> runtime_provider._getenv
        # which is where the reporter's UnscopedSecretError fired.
        observed["during_create"] = ss.get_secret("OPENROUTER_BASE_URL")
        observed["scope_during_create"] = ss.current_secret_scope() is not None
        return FakeAgent(kwargs.get("session_id") or "s")

    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)
    ss.set_multiplex_active(True)

    result, usage = await adapter._run_agent(
        user_message="hi",
        conversation_history=[],
        session_id="chatcmpl-repro-61276",
    )

    assert result["final_response"] == "ok"
    assert usage["total_tokens"] == 0
    assert observed["scope_during_create"] is True
    assert observed["during_create"] == "https://openrouter.ai/api/v1"
    assert observed["during_run"] == "https://openrouter.ai/api/v1"
    # Scope must not leak after the turn.
    assert ss.current_secret_scope() is None


@pytest.mark.asyncio
async def test_v1_runs_scopes_agent_creation_and_executor_run(
    adapter, tmp_path, monkeypatch,
):
    """The independent /v1/runs path scopes both of its execution regions."""
    (tmp_path / ".env").write_text(
        "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    observed: dict = {}

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def run_conversation(self, user_message, conversation_history, task_id):
            observed["during_run"] = ss.get_secret("OPENROUTER_BASE_URL")
            observed["scope_during_run"] = ss.current_secret_scope() is not None
            return {"final_response": "ok"}

    def fake_create_agent(**kwargs):
        observed["during_create"] = ss.get_secret("OPENROUTER_BASE_URL")
        observed["scope_during_create"] = ss.current_secret_scope() is not None
        return FakeAgent()

    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)
    ss.set_multiplex_active(True)

    class FakeRequest:
        headers = {}

        async def json(self):
            return {"input": "hi"}

    response = await adapter._handle_runs(FakeRequest())
    assert response.status == 202
    run_id = json.loads(response.text)["run_id"]

    for _ in range(20):
        status = adapter._run_statuses[run_id]
        if status["status"] in {"completed", "failed"}:
            break
        await asyncio.sleep(0.05)

    assert status["status"] == "completed"
    assert observed == {
        "during_create": "https://openrouter.ai/api/v1",
        "scope_during_create": True,
        "during_run": "https://openrouter.ai/api/v1",
        "scope_during_run": True,
    }
    assert ss.current_secret_scope() is None
