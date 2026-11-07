"""SIWC's mandatory streaming survives optional-streaming provider routing."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.turn_api_call import _should_stream
from providers import get_provider_profile


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("consumers", [False, True])
def test_chatgpt_requires_streaming_despite_optional_session_preferences(disabled, consumers):
    profile = get_provider_profile("openai-chatgpt")
    assert profile is not None
    agent = SimpleNamespace(
        provider=profile.name,
        base_url=profile.base_url,
        client=Mock(),
        _disable_streaming=disabled,
        _has_stream_consumers=lambda: consumers,
    )

    assert _should_stream(agent) is True
