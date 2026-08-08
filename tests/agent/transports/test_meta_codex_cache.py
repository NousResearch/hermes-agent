"""Tests for Meta api.meta.ai prompt_cache_retention and transport plumbing."""

import json
from types import SimpleNamespace

import pytest

from agent.transports import get_transport
from agent.transports.codex import _default_prompt_cache_retention_for_request


@pytest.fixture
def transport():
    import agent.transports.codex  # noqa: F401
    return get_transport("codex_responses")


class TestMetaRetention:
    def test_meta_retention_24h(self, transport):
        kw = transport.build_kwargs(
            model="muse-spark-1.2",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            base_url="https://api.meta.ai/v1",
            session_id="test-session",
        )
        assert kw.get("prompt_cache_retention") == "24h"

    def test_meta_retention_also_for_generic_model_name(self, transport):
        for model in ["muse-spark", "meta/muse-spark-1.2-2026-04-01", "gpt-5.4", ""]:
            kw = transport.build_kwargs(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                base_url="https://api.meta.ai/v1",
                session_id="sid",
            )
            assert kw.get("prompt_cache_retention") == "24h", f"model={model!r}"

    def test_meta_retention_helper_direct(self):
        assert _default_prompt_cache_retention_for_request("muse-spark-1.2", "https://api.meta.ai/v1") == "24h"
        assert _default_prompt_cache_retention_for_request("muse-spark-1.2", "https://API.META.AI/v1") == "24h"
        assert _default_prompt_cache_retention_for_request("muse-spark-1.2", "https://api.meta.ai:443/v1") == "24h"

    def test_meta_retention_not_sent_when_overridden(self, transport):
        kw = transport.build_kwargs(
            model="muse-spark-1.2",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            base_url="https://api.meta.ai/v1",
            session_id="sid",
            request_overrides={"prompt_cache_retention": "in_memory"},
        )
        assert kw.get("prompt_cache_retention") == "in_memory"

    def test_non_meta_no_retention(self, transport):
        kw = transport.build_kwargs(
            model="muse-spark-1.2",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            base_url="https://generic.example.com/v1",
            session_id="sid",
        )
        assert "prompt_cache_retention" not in kw

    def test_non_meta_no_retention_helper(self):
        assert _default_prompt_cache_retention_for_request("muse-spark-1.2", "https://generic.example.com/v1") is None

    def test_meta_prompt_cache_key_is_content_addressed(self, transport):
        messages = [{"role": "user", "content": "Hi"}]
        kw = transport.build_kwargs(
            model="muse-spark-1.2",
            messages=messages,
            tools=[],
            base_url="https://api.meta.ai/v1",
            session_id="cron_job_xxx_20260624_143000",
        )
        pck = kw.get("prompt_cache_key", "")
        assert pck.startswith("pck_")
        # stable across different cron fire timestamps (same scope)
        kw2 = transport.build_kwargs(
            model="muse-spark-1.2",
            messages=messages,
            tools=[],
            base_url="https://api.meta.ai/v1",
            session_id="cron_job_xxx_20260624_143500",
        )
        assert kw["prompt_cache_key"] == kw2["prompt_cache_key"]

    def test_meta_reasoning_effort_passthrough(self, transport):
        kw = transport.build_kwargs(
            model="muse-spark-1.2",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            base_url="https://api.meta.ai/v1",
            session_id="sid",
            reasoning_config={"effort": "high", "enabled": True},
        )
        assert kw.get("reasoning") == {"effort": "high", "summary": "auto"}

    def test_meta_request_hits_responses_plan(self, transport):
        # Transport api_mode must be codex_responses so conversation_loop hits /v1/responses
        assert transport.api_mode == "codex_responses"
        kw = transport.build_kwargs(
            model="muse-spark-1.2",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[{"type": "function", "function": {"name": "terminal", "description": "x", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}],
            base_url="https://api.meta.ai/v1",
            session_id="sid",
        )
        # Ensure instructions/input/tools and retention present - i.e., responses shape
        assert "instructions" in kw or "input" in kw
        assert kw.get("prompt_cache_retention") == "24h"
        assert "prompt_cache_key" in kw
