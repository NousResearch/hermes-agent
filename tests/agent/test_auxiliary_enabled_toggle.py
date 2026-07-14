"""Tests for auxiliary task enabled toggle (regression test for #27877)."""
import pytest
from unittest.mock import patch, MagicMock


class TestAuxiliaryEnabledToggle:
    """call_llm / async_call_llm skip execution when enabled=false."""

    def test_call_llm_returns_none_when_disabled(self):
        """call_llm() returns None immediately when auxiliary.<task>.enabled is false."""
        from agent.auxiliary_client import call_llm

        disabled_config = {"enabled": False}
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=disabled_config), \
             patch("agent.auxiliary_client._resolve_task_provider_model") as mock_resolve:
            result = call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}])
        assert result is None
        mock_resolve.assert_not_called()

    def test_call_llm_resolves_when_enabled_true(self):
        """call_llm() reaches _resolve_task_provider_model when enabled is True."""
        from agent.auxiliary_client import call_llm

        enabled_config = {"enabled": True}
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=enabled_config), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("openai", "gpt-4o-mini", None, "sk-test", None)) as mock_resolve, \
             patch("agent.auxiliary_client._get_task_extra_body", return_value={}), \
             patch("agent.auxiliary_client.logger"):
            try:
                call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}])
            except Exception:
                pass  # We only care that resolve was called, not the full execution
        mock_resolve.assert_called_once()

    def test_call_llm_resolves_when_enabled_missing(self):
        """call_llm() reaches _resolve when enabled key absent (default True)."""
        from agent.auxiliary_client import call_llm

        empty_config = {}
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=empty_config), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("openai", "gpt-4o-mini", None, "sk-test", None)) as mock_resolve, \
             patch("agent.auxiliary_client._get_task_extra_body", return_value={}), \
             patch("agent.auxiliary_client.logger"):
            try:
                call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}])
            except Exception:
                pass
        mock_resolve.assert_called_once()

    def test_call_llm_no_task_always_resolves(self):
        """call_llm() without a task always proceeds to resolve."""
        from agent.auxiliary_client import call_llm

        with patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("openai", "gpt-4o-mini", None, "sk-test", None)) as mock_resolve, \
             patch("agent.auxiliary_client._get_task_extra_body", return_value={}), \
             patch("agent.auxiliary_client.logger"):
            try:
                call_llm(task=None, messages=[{"role": "user", "content": "hi"}])
            except Exception:
                pass
        mock_resolve.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_call_llm_returns_none_when_disabled(self):
        """async_call_llm() returns None immediately when enabled is false."""
        from agent.auxiliary_client import async_call_llm

        disabled_config = {"enabled": False}
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=disabled_config), \
             patch("agent.auxiliary_client._resolve_task_provider_model") as mock_resolve:
            result = await async_call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
        assert result is None
        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_call_llm_resolves_when_enabled(self):
        """async_call_llm() reaches _resolve when enabled is True."""
        from agent.auxiliary_client import async_call_llm

        enabled_config = {"enabled": True}
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=enabled_config), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("openai", "gpt-4o-mini", None, "sk-test", None)) as mock_resolve, \
             patch("agent.auxiliary_client._get_task_extra_body", return_value={}), \
             patch("agent.auxiliary_client.logger"):
            try:
                await async_call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
            except Exception:
                pass
        mock_resolve.assert_called_once()
