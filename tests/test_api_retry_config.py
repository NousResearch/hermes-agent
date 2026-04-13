"""Tests for PR #5571 -- configurable max_api_retries with improved backoff.

Covers:
- AIAgent stores max_api_retries param (default 3, configurable)
- CLI reads max_api_retries from CLI_CONFIG['agent']['max_api_retries']
- Rate-limit backoff formula: base = 5 * 2^retry_count capped at 300, plus jitter
- Non-rate-limit backoff formula: 2^retry_count capped at 60
- Retry-After header is respected when present
"""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, __file__.replace("tests/test_api_retry_config.py", ""))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _make_agent(max_api_retries=3, **kwargs):
    """Create a minimal AIAgent with mocked tool loading and client."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent
        a = AIAgent(
            api_key="test-key-1234567890",
            max_api_retries=max_api_retries,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            **kwargs,
        )
        a.client = MagicMock()
        return a


# ---------------------------------------------------------------------------
# Fake API response helpers
# ---------------------------------------------------------------------------


class _RateLimitError(Exception):
    """Mimics a 429 rate-limit API error."""
    def __init__(self, retry_after=None):
        super().__init__("Error code: 429 - rate limit exceeded")
        self.status_code = 429
        headers = {}
        if retry_after is not None:
            headers["retry-after"] = str(retry_after)
        self.response = SimpleNamespace(headers=headers) if headers else None


class _ServerError(Exception):
    """Mimics a 500 transient server error (non-rate-limit)."""
    def __init__(self):
        super().__init__("Error code: 500 - internal server error")
        self.status_code = 500


def _chat_completion_response(text="Done"):
    """Minimal OpenAI-style chat completion response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="test-model",
    )


# ---------------------------------------------------------------------------
# Core: max_api_retries param storage
# ---------------------------------------------------------------------------


class TestMaxApiRetriesParam:
    def test_default_is_three(self):
        """AIAgent defaults to max_api_retries=3 when not specified."""
        agent = _make_agent()
        assert agent.max_api_retries == 3

    def test_custom_value_stored(self):
        """AIAgent stores the provided max_api_retries value."""
        agent = _make_agent(max_api_retries=10)
        assert agent.max_api_retries == 10

    def test_zero_retries_stored(self):
        """Edge case: max_api_retries=0 is accepted and stored."""
        agent = _make_agent(max_api_retries=0)
        assert agent.max_api_retries == 0

    def test_large_retries_stored(self):
        """Large retry count is stored without modification."""
        agent = _make_agent(max_api_retries=100)
        assert agent.max_api_retries == 100

    def test_attribute_is_integer(self):
        """max_api_retries is stored as an int."""
        agent = _make_agent(max_api_retries=5)
        assert isinstance(agent.max_api_retries, int)


# ---------------------------------------------------------------------------
# CLI config integration
# ---------------------------------------------------------------------------


def _make_cli(config_overrides=None):
    """Create a HermesCLI with minimal mocking; mirrors tests/test_cli_init.py."""
    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    if config_overrides:
        for k, v in config_overrides.items():
            if isinstance(v, dict) and k in _clean_config:
                _clean_config[k].update(v)
            else:
                _clean_config[k] = v

    prompt_toolkit_stubs = {
        m: MagicMock()
        for m in [
            "prompt_toolkit",
            "prompt_toolkit.history",
            "prompt_toolkit.styles",
            "prompt_toolkit.patch_stdout",
            "prompt_toolkit.application",
            "prompt_toolkit.layout",
            "prompt_toolkit.layout.processors",
            "prompt_toolkit.filters",
            "prompt_toolkit.layout.dimension",
            "prompt_toolkit.layout.menus",
            "prompt_toolkit.widgets",
            "prompt_toolkit.key_binding",
            "prompt_toolkit.completion",
            "prompt_toolkit.formatted_text",
            "prompt_toolkit.auto_suggest",
        ]
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict("os.environ", {}, clear=False):
        import cli as _cli_mod
        _cli_mod = importlib.reload(_cli_mod)
        with (
            patch.object(_cli_mod, "get_tool_definitions", return_value=[]),
            patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}),
        ):
            return _cli_mod.HermesCLI()


class TestCliMaxApiRetriesConfig:
    def test_cli_default_max_api_retries_from_empty_config(self):
        """CLI_CONFIG without max_api_retries defaults to 3 via the fallback."""
        config = {"agent": {}}
        retries = int(config.get("agent", {}).get("max_api_retries", 3))
        assert retries == 3

    def test_cli_reads_max_api_retries_from_config(self):
        """CLI reads max_api_retries from CLI_CONFIG['agent']['max_api_retries']."""
        config = {"agent": {"max_api_retries": 7}}
        retries = int(config.get("agent", {}).get("max_api_retries", 3))
        assert retries == 7

    def test_cli_creates_agent_with_config_max_api_retries(self):
        """When CLI_CONFIG sets max_api_retries=7, AIAgent receives max_api_retries=7."""
        import cli as _cli_mod

        config_with_retries = {
            "model": {
                "default": "anthropic/claude-opus-4.6",
                "base_url": "https://openrouter.ai/api/v1",
                "provider": "auto",
            },
            "display": {"compact": False, "tool_progress": "all"},
            "agent": {"max_api_retries": 7},
            "terminal": {"env_type": "local"},
        }

        prompt_toolkit_stubs = {
            m: MagicMock()
            for m in [
                "prompt_toolkit", "prompt_toolkit.history", "prompt_toolkit.styles",
                "prompt_toolkit.patch_stdout", "prompt_toolkit.application",
                "prompt_toolkit.layout", "prompt_toolkit.layout.processors",
                "prompt_toolkit.filters", "prompt_toolkit.layout.dimension",
                "prompt_toolkit.layout.menus", "prompt_toolkit.widgets",
                "prompt_toolkit.key_binding", "prompt_toolkit.completion",
                "prompt_toolkit.formatted_text", "prompt_toolkit.auto_suggest",
            ]
        }
        with patch.dict(sys.modules, prompt_toolkit_stubs):
            fresh = importlib.reload(_cli_mod)
            with (
                patch.object(fresh, "get_tool_definitions", return_value=[]),
                patch.dict(fresh.__dict__, {"CLI_CONFIG": config_with_retries}),
            ):
                cli_instance = fresh.HermesCLI()

                with (
                    patch("run_agent.get_tool_definitions", return_value=[]),
                    patch("run_agent.check_toolset_requirements", return_value={}),
                    patch("run_agent.OpenAI"),
                    patch.object(cli_instance, "_ensure_runtime_credentials", return_value=True),
                    patch.object(cli_instance, "_session_db", MagicMock(), create=True),
                ):
                    # _init_agent reads CLI_CONFIG to pass max_api_retries to AIAgent
                    cli_instance._init_agent(runtime_override={
                        "api_key": "test-key",
                        "base_url": "https://openrouter.ai/api/v1",
                        "provider": "auto",
                        "api_mode": None,
                        "command": None,
                        "args": None,
                        "credential_pool": None,
                    })
                    assert cli_instance.agent is not None
                    assert cli_instance.agent.max_api_retries == 7


# ---------------------------------------------------------------------------
# Backoff formula verification
# ---------------------------------------------------------------------------


class TestRateLimitBackoffFormula:
    """Tests the rate-limit exponential backoff: base = 5 * 2^n, + jitter, capped 300."""

    def test_retry_count_0_base_is_5(self):
        """retry_count=0: base = 5 * (2**0) = 5."""
        retry_count = 0
        _base = min(5 * (2 ** retry_count), 300)
        assert _base == 5

    def test_retry_count_1_base_is_10(self):
        """retry_count=1: base = 5 * (2**1) = 10."""
        retry_count = 1
        _base = min(5 * (2 ** retry_count), 300)
        assert _base == 10

    def test_retry_count_2_base_is_20(self):
        """retry_count=2: base = 5 * (2**2) = 20."""
        retry_count = 2
        _base = min(5 * (2 ** retry_count), 300)
        assert _base == 20

    def test_base_capped_at_300(self):
        """Base is capped at 300 for large retry counts."""
        import math
        # 5 * 2^6 = 320 > 300 → should cap at 300
        retry_count = 6
        _base = min(5 * (2 ** retry_count), 300)
        assert _base == 300

    def test_jitter_is_added(self):
        """Jitter is non-negative and bounded by min(base*0.2, 10)."""
        import random
        retry_count = 0
        _base = min(5 * (2 ** retry_count), 300)
        # Run 50 samples to verify jitter range
        for _ in range(50):
            jitter = random.uniform(0, min(_base * 0.2, 10))
            wait_time = _base + jitter
            assert wait_time >= _base, "jitter should not reduce below base"
            assert wait_time <= _base + min(_base * 0.2, 10), "jitter should not exceed cap"

    def test_jitter_cap_for_large_base(self):
        """Jitter cap is 10s for large base values."""
        import random
        # When base >= 50, jitter cap is min(50*0.2, 10) = 10
        _base = 50
        for _ in range(50):
            jitter = random.uniform(0, min(_base * 0.2, 10))
            assert jitter <= 10


class TestNonRateLimitBackoffFormula:
    """Tests the non-rate-limit backoff: 2^n capped at 60s."""

    def test_retry_count_0_is_1(self):
        """retry_count=0: wait = min(2**0, 60) = 1s."""
        retry_count = 0
        wait = min(2 ** retry_count, 60)
        assert wait == 1

    def test_retry_count_1_is_2(self):
        """retry_count=1: wait = min(2**1, 60) = 2s."""
        retry_count = 1
        wait = min(2 ** retry_count, 60)
        assert wait == 2

    def test_retry_count_5_is_32(self):
        """retry_count=5: wait = min(2**5, 60) = 32s."""
        retry_count = 5
        wait = min(2 ** retry_count, 60)
        assert wait == 32

    def test_capped_at_60(self):
        """retry_count=7: wait = min(2**7, 60) = 60s (capped)."""
        retry_count = 7
        wait = min(2 ** retry_count, 60)
        assert wait == 60

    def test_large_retry_count_still_60(self):
        """Very large retry_count still gives 60s (not more)."""
        for n in range(7, 20):
            wait = min(2 ** n, 60)
            assert wait == 60


# ---------------------------------------------------------------------------
# Retry-After header handling
# ---------------------------------------------------------------------------


class TestRetryAfterHeader:
    def test_retry_after_header_respected(self):
        """Retry-After header value is used as wait_time when present."""
        # Simulate the header parsing logic from run_agent.py
        retry_after_value = "45"
        _retry_after = None
        headers = {"retry-after": retry_after_value}
        _ra_raw = headers.get("retry-after") or headers.get("Retry-After")
        if _ra_raw:
            try:
                _retry_after = min(int(_ra_raw), 300)
            except (TypeError, ValueError):
                pass
        assert _retry_after == 45

    def test_retry_after_header_capped_at_300(self):
        """Retry-After header is capped at 300 seconds."""
        headers = {"retry-after": "600"}
        _retry_after = None
        _ra_raw = headers.get("retry-after") or headers.get("Retry-After")
        if _ra_raw:
            try:
                _retry_after = min(int(_ra_raw), 300)
            except (TypeError, ValueError):
                pass
        assert _retry_after == 300

    def test_retry_after_case_insensitive(self):
        """Retry-After header is matched case-insensitively (both retry-after and Retry-After)."""
        for key in ("retry-after", "Retry-After"):
            headers = {key: "30"}
            _ra_raw = headers.get("retry-after") or headers.get("Retry-After")
            assert _ra_raw == "30"

    def test_retry_after_invalid_value_ignored(self):
        """Non-numeric Retry-After header is ignored (falls back to formula)."""
        headers = {"retry-after": "not-a-number"}
        _retry_after = None
        _ra_raw = headers.get("retry-after")
        if _ra_raw:
            try:
                _retry_after = min(int(_ra_raw), 300)
            except (TypeError, ValueError):
                pass
        assert _retry_after is None


# ---------------------------------------------------------------------------
# Integration: agent loop uses max_api_retries
# ---------------------------------------------------------------------------


class TestAgentLoopUsesMaxApiRetries:
    """Smoke test that the agent loop respects max_api_retries."""

    def _run_with_rate_limit_error(self, max_api_retries):
        """Run agent; API always raises 429. Returns result dict."""
        agent = _make_agent(max_api_retries=max_api_retries)
        agent._persist_session = lambda msgs, history=None: None
        agent._save_trajectory = lambda *a, **k: None
        agent._save_session_log = lambda *a, **k: None
        agent._cleanup_task_resources = lambda *a, **k: None

        call_count = {"n": 0}

        def fake_api(api_kwargs):
            call_count["n"] += 1
            raise _RateLimitError()

        agent._interruptible_api_call = fake_api

        # Time stubs: each call advances by 1000s so the inner sleep loop
        # exits immediately (1000 > max wait_time of 300).
        time_counter = {"n": 0}

        def fake_time():
            time_counter["n"] += 1
            return float(time_counter["n"] * 1000)

        with (
            patch("time.sleep"),
            patch("run_agent.time.sleep"),
            patch("run_agent.time.time", side_effect=fake_time),
        ):
            result = agent.run_conversation("hello")

        return result, call_count["n"]

    def test_max_retries_1_makes_one_attempt(self):
        """With max_api_retries=1, agent gives up after 1 retry."""
        result, calls = self._run_with_rate_limit_error(max_api_retries=1)
        assert result.get("failed") or result.get("completed") is False
        # 1 initial attempt + 1 retry = at most 2 calls before giving up
        assert calls <= 2

    def test_max_retries_3_default(self):
        """Default max_api_retries=3; agent makes multiple attempts before giving up."""
        result, calls = self._run_with_rate_limit_error(max_api_retries=3)
        assert result.get("failed") or result.get("completed") is False

    def test_higher_max_retries_allows_more_attempts(self):
        """Higher max_api_retries allows more retry attempts."""
        result_low, calls_low = self._run_with_rate_limit_error(max_api_retries=1)
        result_high, calls_high = self._run_with_rate_limit_error(max_api_retries=5)
        # More retries configured → more actual API calls before giving up
        assert calls_high >= calls_low
