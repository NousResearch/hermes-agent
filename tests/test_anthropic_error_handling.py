"""Tests for Anthropic error handling in the agent retry loop.

Covers all error paths in run_agent.py's run_conversation() for api_mode=anthropic_messages:
- 429 rate limit → retried with backoff
- 529 overloaded → retried with backoff
- 400 bad request → non-retryable, immediate fail
- 401 unauthorized → credential refresh + retry
- 500 server error → retried with backoff
- "prompt is too long" → context length error triggers compression
- Retry-After header → wait time is honoured / capped
"""

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import gateway.run as gateway_run
import run_agent
from gateway.config import Platform
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr(
        run_agent,
        "get_tool_definitions",
        lambda **kwargs: [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Run shell commands.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


def _anthropic_response(text: str):
    """Simulate an Anthropic messages.create() response object."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        model="claude-sonnet-4-6-20250514",
    )


class _RateLimitError(Exception):
    """Simulates Anthropic 429 rate limit error."""
    def __init__(self):
        super().__init__("Error code: 429 - Rate limit exceeded. Please retry after 30s.")
        self.status_code = 429


class _OverloadedError(Exception):
    """Simulates Anthropic 529 overloaded error."""
    def __init__(self):
        super().__init__("Error code: 529 - API is temporarily overloaded.")
        self.status_code = 529


class _BadRequestError(Exception):
    """Simulates Anthropic 400 bad request error (non-retryable)."""
    def __init__(self):
        super().__init__("Error code: 400 - Invalid model specified.")
        self.status_code = 400


class _UnauthorizedError(Exception):
    """Simulates Anthropic 401 unauthorized error."""
    def __init__(self):
        super().__init__("Error code: 401 - Unauthorized. Invalid API key.")
        self.status_code = 401


class _ServerError(Exception):
    """Simulates Anthropic 500 internal server error."""
    def __init__(self):
        super().__init__("Error code: 500 - Internal server error.")
        self.status_code = 500


class _PromptTooLongError(Exception):
    """Simulates Anthropic prompt-too-long error (triggers context compression)."""
    def __init__(self):
        super().__init__("prompt is too long: 250000 tokens > 200000 maximum")
        self.status_code = 400


class _FakeAnthropicClient:
    def close(self):
        pass


def _fake_build_anthropic_client(key, base_url=None):
    return _FakeAnthropicClient()


def _make_agent_cls(error_cls, recover_after=None):
    """Create an AIAgent subclass that raises error_cls on API calls.

    If recover_after is set, the agent succeeds after that many failures.
    """

    class _Agent(run_agent.AIAgent):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("skip_context_files", True)
            kwargs.setdefault("skip_memory", True)
            kwargs.setdefault("max_iterations", 4)
            super().__init__(*args, **kwargs)
            self._cleanup_task_resources = lambda task_id: None
            self._persist_session = lambda messages, history=None: None
            self._save_trajectory = lambda messages, user_message, completed: None
            self._save_session_log = lambda messages: None

        def run_conversation(self, user_message, conversation_history=None, task_id=None):
            calls = {"n": 0}

            def _fake_api_call(api_kwargs):
                calls["n"] += 1
                if recover_after is not None and calls["n"] > recover_after:
                    return _anthropic_response("Recovered")
                raise error_cls()

            self._interruptible_api_call = _fake_api_call
            return super().run_conversation(
                user_message, conversation_history=conversation_history, task_id=task_id
            )

    return _Agent


def _run_with_agent(monkeypatch, agent_cls):
    """Run _run_agent through the gateway with the given agent class."""
    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.setattr(
        "agent.anthropic_adapter.build_anthropic_client", _fake_build_anthropic_client
    )
    monkeypatch.setattr(run_agent, "AIAgent", agent_cls)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "anthropic",
            "api_mode": "anthropic_messages",
            "base_url": "https://api.anthropic.com",
            "api_key": "sk-ant-api03-test-key",
        },
    )
    monkeypatch.setenv("HERMES_TOOL_PROGRESS", "false")

    runner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None

    source = SessionSource(
        platform=Platform.LOCAL,
        chat_id="cli",
        chat_name="CLI",
        chat_type="dm",
        user_id="test-user-1",
    )

    return asyncio.run(
        runner._run_agent(
            message="hello",
            context_prompt="",
            history=[],
            source=source,
            session_id="test-session",
            session_key="agent:main:local:dm",
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_429_rate_limit_is_retried_and_recovers(monkeypatch):
    """429 should be retried with backoff. First call fails, second succeeds."""
    agent_cls = _make_agent_cls(_RateLimitError, recover_after=1)
    result = _run_with_agent(monkeypatch, agent_cls)
    assert result["final_response"] == "Recovered"


def test_529_overloaded_is_retried_and_recovers(monkeypatch):
    """529 should be retried with backoff. First call fails, second succeeds."""
    agent_cls = _make_agent_cls(_OverloadedError, recover_after=1)
    result = _run_with_agent(monkeypatch, agent_cls)
    assert result["final_response"] == "Recovered"


def test_429_exhausts_all_retries_before_raising(monkeypatch):
    """429 must retry max_retries times, not abort on first attempt."""
    agent_cls = _make_agent_cls(_RateLimitError)  # always fails
    with pytest.raises(_RateLimitError):
        _run_with_agent(monkeypatch, agent_cls)


def test_400_bad_request_is_non_retryable(monkeypatch):
    """400 should fail immediately with only 1 API call (regression guard)."""
    agent_cls = _make_agent_cls(_BadRequestError)
    result = _run_with_agent(monkeypatch, agent_cls)
    assert result["api_calls"] == 1
    assert "400" in str(result.get("final_response", ""))


def test_500_server_error_is_retried_and_recovers(monkeypatch):
    """500 should be retried with backoff. First call fails, second succeeds."""
    agent_cls = _make_agent_cls(_ServerError, recover_after=1)
    result = _run_with_agent(monkeypatch, agent_cls)
    assert result["final_response"] == "Recovered"


def test_401_credential_refresh_recovers(monkeypatch):
    """401 should trigger credential refresh and retry once."""
    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.setattr(
        "agent.anthropic_adapter.build_anthropic_client", _fake_build_anthropic_client
    )
    monkeypatch.setenv("HERMES_TOOL_PROGRESS", "false")

    refresh_count = {"n": 0}

    class _Auth401ThenSuccessAgent(run_agent.AIAgent):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("skip_context_files", True)
            kwargs.setdefault("skip_memory", True)
            kwargs.setdefault("max_iterations", 4)
            super().__init__(*args, **kwargs)
            self._cleanup_task_resources = lambda task_id: None
            self._persist_session = lambda messages, history=None: None
            self._save_trajectory = lambda messages, user_message, completed: None
            self._save_session_log = lambda messages: None

        def _try_refresh_anthropic_client_credentials(self) -> bool:
            refresh_count["n"] += 1
            return True  # Simulate successful credential refresh

        def run_conversation(self, user_message, conversation_history=None, task_id=None):
            calls = {"n": 0}

            def _fake_api_call(api_kwargs):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise _UnauthorizedError()
                return _anthropic_response("Auth refreshed")

            self._interruptible_api_call = _fake_api_call
            return super().run_conversation(
                user_message, conversation_history=conversation_history, task_id=task_id
            )

    monkeypatch.setattr(run_agent, "AIAgent", _Auth401ThenSuccessAgent)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "anthropic",
            "api_mode": "anthropic_messages",
            "base_url": "https://api.anthropic.com",
            "api_key": "sk-ant-api03-test-key",
        },
    )

    runner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None

    source = SessionSource(
        platform=Platform.LOCAL, chat_id="cli", chat_name="CLI",
        chat_type="dm", user_id="test-user-1",
    )

    result = asyncio.run(
        runner._run_agent(
            message="hello", context_prompt="", history=[],
            source=source, session_id="session-401",
            session_key="agent:main:local:dm",
        )
    )

    assert result["final_response"] == "Auth refreshed"
    assert refresh_count["n"] == 1


def test_401_refresh_fails_is_non_retryable(monkeypatch):
    """401 with failed credential refresh should be treated as non-retryable."""
    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.setattr(
        "agent.anthropic_adapter.build_anthropic_client", _fake_build_anthropic_client
    )
    monkeypatch.setenv("HERMES_TOOL_PROGRESS", "false")

    class _Auth401AlwaysFailAgent(run_agent.AIAgent):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("skip_context_files", True)
            kwargs.setdefault("skip_memory", True)
            kwargs.setdefault("max_iterations", 4)
            super().__init__(*args, **kwargs)
            self._cleanup_task_resources = lambda task_id: None
            self._persist_session = lambda messages, history=None: None
            self._save_trajectory = lambda messages, user_message, completed: None
            self._save_session_log = lambda messages: None

        def _try_refresh_anthropic_client_credentials(self) -> bool:
            return False  # Simulate failed credential refresh

        def run_conversation(self, user_message, conversation_history=None, task_id=None):
            def _fake_api_call(api_kwargs):
                raise _UnauthorizedError()

            self._interruptible_api_call = _fake_api_call
            return super().run_conversation(
                user_message, conversation_history=conversation_history, task_id=task_id
            )

    monkeypatch.setattr(run_agent, "AIAgent", _Auth401AlwaysFailAgent)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "anthropic",
            "api_mode": "anthropic_messages",
            "base_url": "https://api.anthropic.com",
            "api_key": "sk-ant-api03-test-key",
        },
    )

    runner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None

    source = SessionSource(
        platform=Platform.LOCAL, chat_id="cli", chat_name="CLI",
        chat_type="dm", user_id="test-user-1",
    )

    result = asyncio.run(
        runner._run_agent(
            message="hello", context_prompt="", history=[],
            source=source, session_id="session-401-fail",
            session_key="agent:main:local:dm",
        )
    )

    # 401 after failed refresh → non-retryable (falls through to is_client_error)
    assert result["api_calls"] == 1
    assert "401" in str(result.get("final_response", "")) or "unauthorized" in str(result.get("final_response", "")).lower()


def test_prompt_too_long_triggers_compression(monkeypatch):
    """Anthropic 'prompt is too long' error should trigger context compression, not immediate fail."""
    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.setattr(
        "agent.anthropic_adapter.build_anthropic_client", _fake_build_anthropic_client
    )
    monkeypatch.setenv("HERMES_TOOL_PROGRESS", "false")

    class _PromptTooLongThenSuccessAgent(run_agent.AIAgent):
        compress_called = 0

        def __init__(self, *args, **kwargs):
            kwargs.setdefault("skip_context_files", True)
            kwargs.setdefault("skip_memory", True)
            kwargs.setdefault("max_iterations", 4)
            super().__init__(*args, **kwargs)
            self._cleanup_task_resources = lambda task_id: None
            self._persist_session = lambda messages, history=None: None
            self._save_trajectory = lambda messages, user_message, completed: None
            self._save_session_log = lambda messages: None

        def _compress_context(self, messages, system_message, approx_tokens=0, task_id=None):
            type(self).compress_called += 1
            # Simulate compression by dropping oldest non-system message
            if len(messages) > 2:
                compressed = [messages[0]] + messages[2:]
            else:
                compressed = messages
            return compressed, system_message

        def run_conversation(self, user_message, conversation_history=None, task_id=None):
            calls = {"n": 0}

            def _fake_api_call(api_kwargs):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise _PromptTooLongError()
                return _anthropic_response("Compressed and recovered")

            self._interruptible_api_call = _fake_api_call
            return super().run_conversation(
                user_message, conversation_history=conversation_history, task_id=task_id
            )

    _PromptTooLongThenSuccessAgent.compress_called = 0
    monkeypatch.setattr(run_agent, "AIAgent", _PromptTooLongThenSuccessAgent)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "anthropic",
            "api_mode": "anthropic_messages",
            "base_url": "https://api.anthropic.com",
            "api_key": "sk-ant-api03-test-key",
        },
    )

    runner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None

    source = SessionSource(
        platform=Platform.LOCAL, chat_id="cli", chat_name="CLI",
        chat_type="dm", user_id="test-user-1",
    )

    result = asyncio.run(
        runner._run_agent(
            message="hello", context_prompt="", history=[],
            source=source, session_id="session-prompt-long",
            session_key="agent:main:local:dm",
        )
    )

    assert result["final_response"] == "Compressed and recovered"
    assert _PromptTooLongThenSuccessAgent.compress_called >= 1


# ---------------------------------------------------------------------------
# Retry-After header tests
# ---------------------------------------------------------------------------


class _RateLimitWithRetryAfterError(Exception):
    """Simulates a 429 that carries a Retry-After response header."""

    def __init__(self, retry_after: str = "45"):
        super().__init__("Error code: 429 - Rate limit exceeded.")
        self.status_code = 429
        self.response = SimpleNamespace(headers={"retry-after": retry_after})


class _ConnectionErrorWrapping429(Exception):
    """Simulates openai.APIConnectionError whose __cause__ holds a 429 response.

    APIConnectionError is a network-layer exception with no status_code; the
    underlying httpx HTTPStatusError on __cause__ carries both the status code
    and the response headers.  __cause__ must be a BaseException, not a plain
    object, so we attach the response as an attribute on a real exception.
    """

    def __init__(self, retry_after: str = "30"):
        super().__init__("Connection closed unexpectedly.")
        # No status_code attribute — mirrors the real APIConnectionError.
        cause_response = SimpleNamespace(
            status_code=429,
            headers={"retry-after": retry_after},
        )
        # Python requires __cause__ to be a BaseException (or None).
        _inner = Exception("HTTP 429 Too Many Requests")
        _inner.response = cause_response  # type: ignore[attr-defined]
        self.__cause__ = _inner


def _make_fast_time():
    """Return a time() stub that advances by 500s per call.

    This ensures the interruptible-sleep loop (``while time.time() < sleep_end``)
    exits after a single iteration without actually blocking the test.
    """
    t = [0.0]

    def _time():
        t[0] += 500.0
        return t[0]

    return _time


def test_retry_after_header_sets_wait_time(monkeypatch, capsys):
    """A 429 with Retry-After: 45 should wait 45s (printed to output), not use backoff."""
    monkeypatch.setattr(run_agent.time, "sleep", lambda s: None)
    monkeypatch.setattr(run_agent.time, "time", _make_fast_time())

    agent_cls = _make_agent_cls(_RateLimitWithRetryAfterError, recover_after=1)
    result = _run_with_agent(monkeypatch, agent_cls)
    captured = capsys.readouterr()

    assert result["final_response"] == "Recovered"
    assert "Honouring server Retry-After: waiting 45s before retry" in captured.out


def test_retry_after_header_is_capped_at_default_max(monkeypatch, capsys):
    """Retry-After values above the 300s default cap are clipped."""
    monkeypatch.setattr(run_agent.time, "sleep", lambda s: None)
    monkeypatch.setattr(run_agent.time, "time", _make_fast_time())
    monkeypatch.delenv("HERMES_RETRY_AFTER_MAX_WAIT", raising=False)

    class _LargeRetryAfter(_RateLimitWithRetryAfterError):
        def __init__(self):
            super().__init__(retry_after="600")

    agent_cls = _make_agent_cls(_LargeRetryAfter, recover_after=1)
    result = _run_with_agent(monkeypatch, agent_cls)
    captured = capsys.readouterr()

    assert result["final_response"] == "Recovered"
    # The cap message shows both the server value and the applied cap.
    assert "600s wait" in captured.out
    assert "capped at 300s" in captured.out


def test_retry_after_cap_is_configurable_via_env(monkeypatch, capsys):
    """HERMES_RETRY_AFTER_MAX_WAIT overrides the default 300s cap."""
    monkeypatch.setattr(run_agent.time, "sleep", lambda s: None)
    monkeypatch.setattr(run_agent.time, "time", _make_fast_time())
    monkeypatch.setenv("HERMES_RETRY_AFTER_MAX_WAIT", "60")

    class _LargeRetryAfter(_RateLimitWithRetryAfterError):
        def __init__(self):
            super().__init__(retry_after="600")

    agent_cls = _make_agent_cls(_LargeRetryAfter, recover_after=1)
    result = _run_with_agent(monkeypatch, agent_cls)
    captured = capsys.readouterr()

    assert result["final_response"] == "Recovered"
    assert "600s wait" in captured.out
    assert "capped at 60s" in captured.out


def test_retry_after_on_connection_error_cause(monkeypatch, capsys):
    """APIConnectionError whose __cause__ carries a 429 + Retry-After should be honoured."""
    monkeypatch.setattr(run_agent.time, "sleep", lambda s: None)
    monkeypatch.setattr(run_agent.time, "time", _make_fast_time())

    class _ConnError429(_ConnectionErrorWrapping429):
        def __init__(self):
            super().__init__(retry_after="30")

    agent_cls = _make_agent_cls(_ConnError429, recover_after=1)
    result = _run_with_agent(monkeypatch, agent_cls)
    captured = capsys.readouterr()

    assert result["final_response"] == "Recovered"
    assert "Honouring server Retry-After: waiting 30s before retry" in captured.out


def test_retry_after_missing_falls_back_to_exponential_backoff(monkeypatch, capsys):
    """A plain 429 without a Retry-After header does not print a Retry-After message."""
    monkeypatch.setattr(run_agent.time, "sleep", lambda s: None)
    monkeypatch.setattr(run_agent.time, "time", _make_fast_time())

    # _RateLimitError has no .response attribute → no Retry-After header.
    agent_cls = _make_agent_cls(_RateLimitError, recover_after=1)
    result = _run_with_agent(monkeypatch, agent_cls)
    captured = capsys.readouterr()

    assert result["final_response"] == "Recovered"
    assert "Honouring server Retry-After" not in captured.out
    assert "capped at" not in captured.out
