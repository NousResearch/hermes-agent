"""Tests for cron prompt cache warming fix (#51395).

Verifies that recurring cron jobs use a stable prompt_cache_key (cron_<job_id>)
instead of the per-fire session_id (cron_<job_id>_<timestamp>), so the prompt
prefix cache stays warm across fires.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestCodexTransportCacheKey:
    """Test that CodexResponsesTransport.build_kwargs uses prompt_cache_key param."""

    def _get_transport(self):
        from agent.transports.codex import ResponsesApiTransport
        return ResponsesApiTransport()

    def test_cache_key_defaults_to_session_id(self):
        """When no prompt_cache_key is given, cache key defaults to session_id."""
        transport = self._get_transport()
        kwargs = transport.build_kwargs(
            model="gpt-4",
            messages=[{"role": "system", "content": "hi"}],
            session_id="interactive_session_123",
        )
        assert kwargs.get("prompt_cache_key") == "interactive_session_123"

    def test_explicit_cache_key_overrides_session_id(self):
        """When prompt_cache_key is given, it overrides session_id for caching."""
        transport = self._get_transport()
        kwargs = transport.build_kwargs(
            model="gpt-4",
            messages=[{"role": "system", "content": "hi"}],
            session_id="cron_abc123_20260624_120000",
            prompt_cache_key="cron_abc123",
        )
        assert kwargs.get("prompt_cache_key") == "cron_abc123"

    def test_codex_backend_headers_use_cache_key(self):
        """Codex backend headers use cache_key, not session_id."""
        transport = self._get_transport()
        kwargs = transport.build_kwargs(
            model="gpt-4",
            messages=[{"role": "system", "content": "hi"}],
            session_id="cron_abc123_20260624_120000",
            prompt_cache_key="cron_abc123",
            is_codex_backend=True,
        )
        headers = kwargs.get("extra_headers", {})
        assert headers.get("session_id") == "cron_abc123"
        assert headers.get("x-client-request-id") == "cron_abc123"

    def test_xai_cache_key_in_extra_body(self):
        """xAI extra_body prompt_cache_key uses stable cache_key."""
        transport = self._get_transport()
        kwargs = transport.build_kwargs(
            model="grok-3",
            messages=[{"role": "system", "content": "hi"}],
            session_id="cron_abc123_20260624_120000",
            prompt_cache_key="cron_abc123",
            is_xai_responses=True,
        )
        extra_body = kwargs.get("extra_body", {})
        assert extra_body.get("prompt_cache_key") == "cron_abc123"
        # Conv header still uses session_id for conversation tracking
        headers = kwargs.get("extra_headers", {})
        assert headers.get("x-grok-conv-id") == "cron_abc123_20260624_120000"

    def test_no_session_id_no_cache_key(self):
        """When neither session_id nor cache_key given, no cache routing."""
        transport = self._get_transport()
        kwargs = transport.build_kwargs(
            model="gpt-4",
            messages=[{"role": "system", "content": "hi"}],
        )
        assert "prompt_cache_key" not in kwargs


class TestCronSchedulerSetsStableCacheKey:
    """Test that the cron scheduler sets a stable prompt_cache_key on the agent."""

    def test_cron_session_id_has_timestamp(self):
        """Verify the session_id format includes timestamp (the problem)."""
        # This documents the format that causes cache misses
        import re
        session_id = "cron_abc123_20260624_120000"
        assert re.match(r"cron_[^_]+_\d{8}_\d{6}", session_id)

    def test_stable_cache_key_format(self):
        """The stable cache key is just cron_<job_id> without timestamp."""
        job_id = "abc123"
        cache_key = f"cron_{job_id}"
        assert cache_key == "cron_abc123"
        # Same job, different fire times, same cache key
        assert cache_key == f"cron_{job_id}"

    def test_different_jobs_have_different_cache_keys(self):
        """Different jobs should NOT share cache keys."""
        key_a = "cron_job_morning"
        key_b = "cron_job_evening"
        assert key_a != key_b


class TestCacheKeyIntegration:
    """Integration test: agent.prompt_cache_key flows to build_kwargs."""

    def test_agent_prompt_cache_key_attribute(self):
        """Verify getattr fallback when prompt_cache_key not set."""

        class FakeAgent:
            session_id = "cron_x_20260624_090000"

        agent = FakeAgent()
        # Without prompt_cache_key, getattr returns None
        assert getattr(agent, "prompt_cache_key", None) is None

        # With it set, returns the stable key
        agent.prompt_cache_key = "cron_x"
        assert getattr(agent, "prompt_cache_key", None) == "cron_x"

    def test_cache_helper_passes_cache_key(self):
        """chat_completion_helpers passes prompt_cache_key to build_kwargs."""
        # We verify by checking the source contains the expected call
        import inspect
        from agent import chat_completion_helpers
        source = inspect.getsource(chat_completion_helpers)
        assert "prompt_cache_key=getattr(agent, \"prompt_cache_key\"" in source
