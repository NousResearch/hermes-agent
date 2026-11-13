from types import SimpleNamespace

import pytest

from agent.api_request_hooks import ApiRequestHooksMixin
from agent.usage_pricing import reasoning_tokens_reported


@pytest.mark.parametrize("usage, reported", [
    ({"prompt_tokens": 10, "completion_tokens": 5}, False),  # e.g. a chat route without details
    ({"completion_tokens_details": None}, False),
    ({"completion_tokens_details": {"reasoning_tokens": None}}, False),
    ({"completion_tokens_details": {"reasoning_tokens": 0}}, True),  # a measured zero
    ({"output_tokens_details": {"reasoning_tokens": 2400}}, True),
    ({"output_tokens_details": {"thinking_tokens": 7}}, True),
    (SimpleNamespace(completion_tokens_details=SimpleNamespace(reasoning_tokens=0)), True),
    (SimpleNamespace(input_tokens=3, output_tokens=4), False),
])
def test_reasoning_tokens_reported_separates_zero_from_absent(usage, reported):
    assert reasoning_tokens_reported(usage) is reported


@pytest.mark.parametrize("usage, expected", [
    ({"prompt_tokens": 10, "completion_tokens": 5}, (0, False)),
    ({"prompt_tokens": 10, "completion_tokens": 5, "completion_tokens_details": {"reasoning_tokens": 0}}, (0, True)),
    ({"prompt_tokens": 10, "completion_tokens": 5, "completion_tokens_details": {"reasoning_tokens": 3}}, (3, True)),
])
def test_post_api_request_usage_says_whether_reasoning_was_reported(usage, expected):
    agent = SimpleNamespace(provider="openrouter", api_mode="chat_completions")
    summary = ApiRequestHooksMixin._usage_summary_for_api_request_hook(agent, SimpleNamespace(usage=usage))
    assert (summary["reasoning_tokens"], summary["reasoning_tokens_reported"]) == expected
