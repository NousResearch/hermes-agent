"""Tests for per-job fallback chains (`job["fallback"]`).

The scheduler historically read the fallback chain ONLY from config.yaml
(`fallback_providers` / `fallback_model`), then pin-filtered it. That left
codex-pinned jobs with no usable fallback (the global chain's only same-provider
entry is the same model they already run). `_resolve_job_fallback_chain` lets a
job declare its OWN same-provider fallback (e.g. codex/gpt-5.5 → codex/gpt-5.4)
so it survives a backend hiccup without leaving the pinned provider.

Per-job chains still pass through `_filter_fallback_chain_for_pinned_job`, so a
job can never declare a cross-provider escape hatch (codex→opus stays impossible).
"""
import logging
import os

import pytest


class TestResolveJobFallbackChain:
    def teardown_method(self):
        os.environ.pop("HERMES_CRON_ALLOW_CROSS_PROVIDER_FALLBACK", None)

    @property
    def _fn(self):
        from cron.scheduler import _resolve_job_fallback_chain
        return _resolve_job_fallback_chain

    GLOBAL_CHAIN = [
        {"provider": "claude-api-proxy-f1", "model": "claude-opus-4-8"},
        {"provider": "openai-codex", "model": "gpt-5.5"},
    ]

    def test_job_fallback_overrides_global_chain(self):
        """A job with its own fallback uses it, not the global chain."""
        job = {
            "id": "md", "provider": "openai-codex", "model": "gpt-5.5",
            "fallback": [{"provider": "openai-codex", "model": "gpt-5.4"}],
        }
        out = self._fn(job, list(self.GLOBAL_CHAIN), job["id"])
        assert out == [{"provider": "openai-codex", "model": "gpt-5.4"}]

    def test_job_fallback_still_pin_filtered(self):
        """Even a per-job fallback can't cross providers — codex→opus is stripped."""
        job = {
            "id": "md", "provider": "openai-codex", "model": "gpt-5.5",
            "fallback": [
                {"provider": "claude-api-proxy-f1", "model": "claude-opus-4-8"},
                {"provider": "openai-codex", "model": "gpt-5.4"},
            ],
        }
        out = self._fn(job, list(self.GLOBAL_CHAIN), job["id"])
        assert out == [{"provider": "openai-codex", "model": "gpt-5.4"}]
        assert all("opus" not in (e.get("model") or "").lower() for e in out)

    def test_no_job_fallback_falls_back_to_global(self):
        """Jobs without their own fallback get the (pin-filtered) global chain."""
        job = {"id": "x", "provider": "openai-codex", "model": "gpt-5.5"}
        out = self._fn(job, list(self.GLOBAL_CHAIN), job["id"])
        # global chain pin-filtered → only the codex entry survives
        assert out == [{"provider": "openai-codex", "model": "gpt-5.5"}]

    def test_unpinned_job_with_job_fallback_keeps_it_unfiltered(self):
        """No provider pin → per-job chain is used as-is (nothing to filter against)."""
        job = {
            "id": "u", "provider": None, "model": None,
            "fallback": [{"provider": "openai-codex", "model": "gpt-5.4"}],
        }
        out = self._fn(job, list(self.GLOBAL_CHAIN), job["id"])
        assert out == [{"provider": "openai-codex", "model": "gpt-5.4"}]

    def test_single_dict_job_fallback_normalized_to_list(self):
        job = {
            "id": "s", "provider": "openai-codex", "model": "gpt-5.5",
            "fallback": {"provider": "openai-codex", "model": "gpt-5.4"},
        }
        out = self._fn(job, None, job["id"])
        assert out == [{"provider": "openai-codex", "model": "gpt-5.4"}]

    def test_empty_job_fallback_uses_global(self):
        """An empty per-job fallback is treated as 'unset' → global chain."""
        job = {"id": "e", "provider": "openai-codex", "model": "gpt-5.5", "fallback": []}
        out = self._fn(job, list(self.GLOBAL_CHAIN), job["id"])
        assert out == [{"provider": "openai-codex", "model": "gpt-5.5"}]

    def test_override_env_disables_pin_filter_on_job_chain(self):
        os.environ["HERMES_CRON_ALLOW_CROSS_PROVIDER_FALLBACK"] = "1"
        job = {
            "id": "o", "provider": "openai-codex", "model": "gpt-5.5",
            "fallback": [{"provider": "claude-api-proxy-f1", "model": "claude-opus-4-8"}],
        }
        out = self._fn(job, list(self.GLOBAL_CHAIN), job["id"])
        # revert switch → per-job chain passes through unfiltered
        assert out == [{"provider": "claude-api-proxy-f1", "model": "claude-opus-4-8"}]
