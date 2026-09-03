"""Regression tests for #57899 — provider auth errors classify as failed.

When ``_resolve_session_agent_runtime`` raises (e.g. a bad or missing
provider API key), the gateway must surface the turn as a *failed* result —
``failed=True`` — rather than an apparently-normal empty result.  Only then
does the downstream adapter retry / failed-turn handling engage: the platform
delivery path treats ``failed=True`` as an explicit failure (so it is never
counted as intentional silence and never silently dropped), and the result
contract matches what ``run_conversation`` produces for a real provider error.

This test forces the resolver to raise and asserts both halves of the
contract that the original PR added: the early-return carries ``failed=True``
(``api_calls`` stays 0, no messages/tools are fabricated) and the gateway's
failed-turn classifier recognises it as a failure rather than silence.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.response_filters import is_intentional_silence_agent_result
from gateway.session import SessionSource

SESSION_KEY = "agent:main:telegram:dm:12345"


class _Adapter:
    SUPPORTS_MESSAGE_EDITING = True
    _pending_messages = {}

    def get_pending_message(self, _session_key):
        return None

    async def send_typing(self, *_args, **_kwargs):
        return None

    async def stop_typing(self, *_args, **_kwargs):
        return None


def _runner(monkeypatch):
    """Minimal GatewayRunner wired so ``_run_agent_inner`` reaches the
    resolver block, with ``_resolve_session_agent_runtime`` set to raise."""
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: _Adapter()}
    runner.config = SimpleNamespace(
        streaming=None,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner.hooks = SimpleNamespace(loaded_hooks=False, emit=AsyncMock())
    runner.session_store = MagicMock()
    runner._session_db = MagicMock()
    runner._session_db.get_telegram_topic_binding_by_session.return_value = None
    runner._agent_cache = {}
    runner._agent_cache_lock = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._pending_skills_reload_notes = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._draining = False
    runner._get_proxy_url = lambda: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._begin_session_run_generation = lambda _key: 1
    runner._release_running_agent_state = MagicMock()
    runner._evict_cached_agent = MagicMock()

    # Force the resolver to raise — this is the auth-error path under test.
    def _raise_auth(*_args, **_kwargs):
        raise RuntimeError("Invalid API key provided: test-invalid-key")

    runner._resolve_session_agent_runtime = _raise_auth

    # Install the module-level hooks the gateway relies on.
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")

    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(
        tools_config, "_get_platform_tools", lambda *_args, **_kwargs: {"core"}
    )
    return runner


def _run_turn(runner, source):
    return asyncio.run(
        asyncio.wait_for(
            runner._run_agent(
                message="hello",
                context_prompt="",
                history=[],
                source=source,
                session_id="sess-auth-fail",
                session_key=SESSION_KEY,
            ),
            timeout=5,
        )
    )


def test_provider_auth_error_marks_result_failed(monkeypatch):
    """The resolver raising produces a result whose ``failed`` flag is set.

    This is the contract the PR fixes: without it the gateway would see a
    normal-looking empty result (``final_response`` set, ``failed`` unset) and
    could misclassify the turn as intentional silence / a silent drop instead
    of a genuine provider failure that should be retried or surfaced.
    """
    runner = _runner(monkeypatch)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="user-1"
    )

    result = _run_turn(runner, source)

    assert result is not None
    assert result.get("failed") is True
    assert result.get("api_calls", 0) == 0
    assert result.get("messages", []) == []
    assert result.get("tools", []) == []
    assert "Provider authentication failed" in result.get("final_response", "")
    assert "Invalid API key provided" in result.get("final_response", "")


def test_provider_auth_error_is_not_intentional_silence(monkeypatch):
    """A failed auth result must never be suppressed as intentional silence.

    The delivery guard only suppresses successful turns; marking the result
    ``failed=True`` is what guarantees the auth error is actually sent back to
    the user (and reaches the adapter's failed-turn handling) rather than
    being dropped as an empty/silent reply.
    """
    runner = _runner(monkeypatch)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="user-1"
    )

    result = _run_turn(runner, source)
    assert result.get("failed") is True

    assert is_intentional_silence_agent_result(result, result.get("final_response")) is False
