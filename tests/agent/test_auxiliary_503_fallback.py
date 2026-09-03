"""Test that exhausted 503 / 5xx transient transport errors trigger provider fallback (#96073)."""

from unittest.mock import MagicMock, patch
import pytest

from agent.auxiliary_client import call_llm


def test_exhausted_503_triggers_fallback():
    """When a provider returns 503 and retries are exhausted, it should fall back."""
    class HTTP503Error(Exception):
        status_code = 503

    fail_client = MagicMock()
    fail_client.chat.completions.create.side_effect = HTTP503Error("503 Service Unavailable")

    success_resp = MagicMock()
    success_resp.choices = [MagicMock(message=MagicMock(content="fallback summary ok"))]

    fallback_client = MagicMock()
    fallback_client.chat.completions.create.return_value = success_resp

    with patch("agent.auxiliary_client.resolve_provider_client") as mock_resolve, \
         patch("agent.auxiliary_client._transient_retry_count", return_value=1), \
         patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(fallback_client, "fallback-model", "fallback-provider")):
        mock_resolve.return_value = (fail_client, "gemini-2.5-pro")

        resp = call_llm(
            task="compression",
            provider="gemini",
            model="gemini-2.5-pro",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert resp.choices[0].message.content == "fallback summary ok"
