"""Bedrock mid-call stale escalation vs the half-open give-up probe (#89587).

The Bedrock stale watchdog re-checks the cross-turn breaker while it is already
ending the call as stale, so it calls ``_check_stale_giveup(allow_probe=False)``.
Harness mirrors ``TestBedrockStreamLivenessWatchdog`` in tests/agent/test_streaming.py.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from tests.agent.test_streaming import _BlockingEventStream


def _make_bedrock_agent():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "bedrock_converse"
    agent._interrupt_requested = False
    return agent


def test_midcall_escalation_raises_even_when_probe_window_elapsed(monkeypatch):
    """The mid-call escalation must keep raising unconditionally
    (allow_probe=False): the call is already ending as stale, so admitting it
    as the half-open probe could never dispatch anything — it would only end
    the call with a plain TimeoutError and consume the window's single probe
    claim, pushing real recovery out by a full cooldown."""
    pytest.importorskip("botocore", reason="botocore required for Bedrock tests")

    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.5")
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "1")

    agent = _make_bedrock_agent()
    agent._consecutive_stale_streams = 0  # entry guard passes cleanly
    probe_after = time.monotonic() - 10.0  # window elapsed
    agent._stale_probe_after = probe_after
    release = threading.Event()

    client = MagicMock()
    client.converse_stream.return_value = _BlockingEventStream(release)

    try:
        with patch(
            "agent.bedrock_adapter._get_bedrock_runtime_client",
            return_value=client,
        ):
            # The in-call watchdog bumps the streak to the giveup=1
            # threshold; the mid-call check must raise the breaker error,
            # never treat the elapsed window as a probe opportunity.
            with pytest.raises(RuntimeError, match="consecutive stale attempts"):
                agent._interruptible_streaming_api_call(
                    {"modelId": agent.model, "messages": []}
                )
    finally:
        release.set()

    # The probe claim is still unspent for the next call's entry guard.
    assert agent._stale_probe_after == probe_after
