"""Cron-level regression for the empty-api-key → fallback promotion (#33936).

Before this fix, ``resolve_runtime_provider("deepseek")`` returned
``{"api_key": "", ...}`` when ``DEEPSEEK_API_KEY`` was missing rather
than raising :class:`AuthError`. The cron scheduler's ``except AuthError``
branch therefore never fired and the configured ``fallback_providers``
chain was skipped — the job was built with empty credentials and then
crashed at API-call time with a generic 401, leaving operators to
debug a credential leak instead of a missing-key configuration.

The cron path now wraps the primary resolution with
:func:`hermes_cli.runtime_provider.ensure_runtime_credentials_or_raise`,
so the registry api-key providers (DeepSeek, Z.AI/GLM, Kimi, …) flip
empty-key into AuthError. This test pins that wiring end-to-end.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestCronFallbackOnMissingPrimaryApiKey:
    """Empty primary api_key should trigger the configured fallback chain."""

    def test_deepseek_empty_key_falls_through_to_openrouter(self, tmp_path):
        """Mirror the reporter's setup in #33936: ``provider: deepseek``
        as primary, ``openrouter`` configured as fallback. With
        ``DEEPSEEK_API_KEY`` unset the cron scheduler must try the
        openrouter fallback instead of failing the job outright.
        """
        from cron.scheduler import run_job

        # config.yaml: deepseek primary, openrouter fallback.
        (tmp_path / "config.yaml").write_text(
            "model:\n"
            "  provider: deepseek\n"
            "  default: deepseek-chat\n"
            "fallback_providers:\n"
            "  - provider: openrouter\n"
            "    model: meta-llama/llama-4-maverick\n"
            "    api_key: or-fallback-real-key-12345\n"
        )

        fake_db = MagicMock()

        # First call (primary) returns empty key for deepseek (silent
        # failure mode that #33936 reported). Second call (fallback)
        # returns a fully-resolved openrouter runtime.
        call_log: list[dict] = []

        def _mock_resolve(**kwargs):
            call_log.append(dict(kwargs))
            if (kwargs.get("requested") or "").lower() == "deepseek":
                return {
                    "api_key": "",
                    "base_url": "https://api.deepseek.com/v1",
                    "provider": "deepseek",
                    "api_mode": "chat_completions",
                }
            return {
                "api_key": "or-fallback-real-key-12345",
                "base_url": "https://openrouter.ai/api/v1",
                "provider": "openrouter",
                "api_mode": "chat_completions",
            }

        job = {
            "id": "test-fallback",
            "name": "fallback",
            "prompt": "hello",
            "provider": "deepseek",
        }

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("dotenv.load_dotenv"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 side_effect=_mock_resolve,
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            success, _output, final_response, error = run_job(job)

        assert success is True, f"job should have succeeded via fallback, got error={error!r}"
        assert final_response == "ok"

        # The resolver was called at least twice: once for the primary
        # (deepseek, returned empty key, promoted to AuthError) and once
        # for the openrouter fallback.
        primary_calls = [c for c in call_log if (c.get("requested") or "").lower() == "deepseek"]
        fallback_calls = [c for c in call_log if (c.get("requested") or "").lower() == "openrouter"]
        assert primary_calls, "primary deepseek call was never attempted"
        assert fallback_calls, (
            "openrouter fallback was never attempted — empty primary "
            "api_key did not promote to AuthError so the fallback "
            "chain in cron/scheduler.py stayed dormant (#33936 regression)."
        )

        # AIAgent received the fallback runtime's credentials, not the
        # empty primary ones.
        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["provider"] == "openrouter"
        assert kwargs["api_key"] == "or-fallback-real-key-12345"

    def test_realistic_primary_key_skips_fallback(self, tmp_path):
        """Sanity: when the primary api-key provider has a usable key
        the fallback chain is not consulted."""
        from cron.scheduler import run_job

        (tmp_path / "config.yaml").write_text(
            "model:\n"
            "  provider: deepseek\n"
            "  default: deepseek-chat\n"
            "fallback_providers:\n"
            "  - provider: openrouter\n"
            "    model: meta-llama/llama-4-maverick\n"
            "    api_key: or-fallback-real-key-12345\n"
        )

        fake_db = MagicMock()

        call_log: list[dict] = []

        def _mock_resolve(**kwargs):
            call_log.append(dict(kwargs))
            return {
                "api_key": "sk-deepseek-real-key-test-12345",
                "base_url": "https://api.deepseek.com/v1",
                "provider": "deepseek",
                "api_mode": "chat_completions",
            }

        job = {
            "id": "happy",
            "name": "happy",
            "prompt": "hello",
            "provider": "deepseek",
        }

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("dotenv.load_dotenv"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 side_effect=_mock_resolve,
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            success, _output, _final_response, error = run_job(job)

        assert success is True, f"job should have succeeded, got error={error!r}"
        # Only one resolver call — primary succeeded, no fallback needed.
        assert len(call_log) == 1
        assert (call_log[0].get("requested") or "").lower() == "deepseek"

        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["provider"] == "deepseek"
        assert kwargs["api_key"] == "sk-deepseek-real-key-test-12345"
