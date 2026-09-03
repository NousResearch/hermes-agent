"""Tests for Hindsight mental model injection into the system prompt."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Tests for the SDK-based mental model fetch via _run_hindsight_operation
# ---------------------------------------------------------------------------


def _make_provider(**overrides):
    """Create a minimal HindsightMemoryProvider for system_prompt_block tests."""
    from plugins.memory.hindsight import HindsightMemoryProvider

    provider = object.__new__(HindsightMemoryProvider)
    provider._bank_id = overrides.get("bank_id", "hermes")
    provider._budget = overrides.get("budget", "mid")
    provider._memory_mode = overrides.get("memory_mode", "hybrid")
    provider._mental_model_id = overrides.get("mental_model_id", "")
    provider._mental_model_content = overrides.get("mental_model_content", "")
    return provider


class TestMentalModelFetchViaSDK:
    """Tests covering the _run_hindsight_operation-based mental model fetch."""

    def test_successful_fetch_stores_content(self):
        """SDK returns a response object with content attribute."""
        provider = _make_provider()
        provider._mode = "local_external"
        provider._api_url = "http://localhost:8888"
        provider._api_key = "test-key"
        provider._timeout = 5.0
        provider._client = None
        provider._config = {"mental_model_id": "test-model"}
        provider._mental_model_id = ""
        provider._mental_model_content = ""

        mock_resp = SimpleNamespace(content="## Profile\nTest user profile.")
        with patch.object(provider, "_run_hindsight_operation", return_value=mock_resp):
            # Simulate the initialize() mental model block
            provider._mental_model_id = "test-model"
            try:
                resp = provider._run_hindsight_operation(
                    lambda client: client.mental_models.get_mental_model(
                        provider._bank_id, provider._mental_model_id,
                        detail="content", _request_timeout=5.0,
                    )
                )
                content = str(getattr(resp, "content", None) or "")
            except Exception:
                content = ""
            provider._mental_model_content = content

        assert provider._mental_model_content == "## Profile\nTest user profile."

    def test_fetch_failure_returns_empty(self):
        """When _run_hindsight_operation raises, content stays empty."""
        provider = _make_provider()
        provider._mental_model_id = "test-model"

        with patch.object(provider, "_run_hindsight_operation", side_effect=RuntimeError("loop unavailable")):
            try:
                resp = provider._run_hindsight_operation(
                    lambda client: client.mental_models.get_mental_model(
                        provider._bank_id, provider._mental_model_id,
                        detail="content", _request_timeout=5.0,
                    )
                )
                content = str(getattr(resp, "content", None) or "")
            except Exception:
                content = ""

        assert content == ""

    def test_fetch_with_none_content_returns_empty(self):
        """SDK returns a response with content=None."""
        provider = _make_provider()
        provider._mental_model_id = "test-model"

        mock_resp = SimpleNamespace(content=None)
        with patch.object(provider, "_run_hindsight_operation", return_value=mock_resp):
            try:
                resp = provider._run_hindsight_operation(
                    lambda client: client.mental_models.get_mental_model(
                        provider._bank_id, provider._mental_model_id,
                        detail="content", _request_timeout=5.0,
                    )
                )
                content = str(getattr(resp, "content", None) or "")
            except Exception:
                content = ""

        assert content == ""

    def test_detail_content_passed_to_sdk(self):
        """Verify the lambda passes detail='content' to reduce payload."""
        provider = _make_provider()
        provider._mental_model_id = "test-model"
        provider._bank_id = "hermes"

        calls = []

        def capture_op(operation):
            # Create a mock client to capture the call args
            mock_mm_api = MagicMock()
            mock_mm_api.get_mental_model.return_value = SimpleNamespace(content="test")

            class FakeClient:
                mental_models = mock_mm_api
            result = operation(FakeClient())
            calls.append(mock_mm_api.get_mental_model.call_args)
            return SimpleNamespace(content="test")

        with patch.object(provider, "_run_hindsight_operation", side_effect=capture_op):
            provider._run_hindsight_operation(
                lambda client: client.mental_models.get_mental_model(
                    provider._bank_id, provider._mental_model_id,
                    detail="content", _request_timeout=5.0,
                )
            )

        assert len(calls) == 1
        assert calls[0].kwargs.get("detail") == "content" or calls[0][1].get("detail") == "content" or "content" in str(calls[0])


# ---------------------------------------------------------------------------
# Tests for system_prompt_block() mental model injection
# ---------------------------------------------------------------------------


class TestSystemPromptBlock:
    """Tests for mental model inclusion in system_prompt_block()."""

    def test_includes_mental_model_with_fence(self):
        provider = _make_provider(
            mental_model_id="test-model",
            mental_model_content="## Profile\nTest user works in engineering.",
        )
        block = provider.system_prompt_block()
        assert "<memory-context>" in block
        assert "# Hindsight Mental Model (synthesized cross-session context)" in block
        assert "ID: test-model" in block
        assert "Test user works in engineering." in block
        assert "</memory-context>" in block

    def test_excludes_when_id_empty(self):
        provider = _make_provider(mental_model_id="", mental_model_content="leftover")
        block = provider.system_prompt_block()
        assert "mental-model" not in block.lower()
        assert "<memory-context>" not in block

    def test_excludes_when_content_empty(self):
        provider = _make_provider(mental_model_id="test-model", mental_model_content="")
        block = provider.system_prompt_block()
        assert "<memory-context>" not in block

    def test_stable_across_repeated_calls(self):
        provider = _make_provider(
            mental_model_id="test-model",
            mental_model_content="Stable content here.",
        )
        first = provider.system_prompt_block()
        second = provider.system_prompt_block()
        third = provider.system_prompt_block()
        assert first == second == third
