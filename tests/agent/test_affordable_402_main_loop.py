"""Main-loop credit-limited 402: retry once with the affordable output cap (#49769).

OpenRouter answers a request whose ``max_tokens`` exceeds the remaining balance with
``402 ... You requested up to 65536 tokens, but can only afford 56272``. The account has
credit; the cap was too large. A real ``AIAgent`` turn against a recording fake provider must
retry the same provider once with the cap lowered to the affordable budget (the auxiliary
client's ``_affordable_max_tokens_from_error`` parse) and answer — and a second 402 must
still end the turn as billing, without a retry loop.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Text, write_hermes_home

_CREDIT_LIMITED_402 = (
    "This request requires more credits, or fewer max_tokens. You requested up to 65536 tokens, "
    "but can only afford 56272. To increase, visit https://openrouter.ai/settings/credits and add more credits"
)


def _turn(script):
    from run_agent import AIAgent

    home = Path(os.environ["HERMES_HOME"])
    with FakeLLMServer(script) as srv:
        write_hermes_home(home, srv.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW)
        agent = AIAgent(provider="custom", base_url=srv.base_url, api_key="sk-fake", model="fake-model",
                        session_id="affordable-402", quiet_mode=True, platform="cli", skip_memory=True,
                        max_tokens=65536)
        try:
            result = agent.run_conversation("hi", conversation_history=[], task_id="t")
        finally:
            agent.close()
        caps = [r.get("max_tokens") or r.get("max_completion_tokens") for r in srv.main_requests()]
    return result, caps


@pytest.mark.parametrize("repeat_402", [False, True], ids=["retry-answers", "second-402-is-billing"])
def test_credit_limited_402_retries_once_with_the_affordable_cap(repeat_402):
    script = [Error(402, _CREDIT_LIMITED_402)] * (2 if repeat_402 else 1) + [Text("answered")]
    result, caps = _turn(script)

    assert caps == [65536, 56272 - 64]
    if repeat_402:
        assert result.get("failed") and "Billing" in (result.get("final_response") or ""), result
    else:
        assert result.get("completed") and result.get("final_response") == "answered", result
