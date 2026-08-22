"""Named local Ollama endpoints must still get CustomProfile.

``providers.ollama-launch`` is a user-defined key, not a registered
profile name. Without the local-server fallback, Qwen 3.8 skips the
max_tokens floor, reasoning_effort, and user-query injection.
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent.chat_completion_helpers import _resolve_chat_provider_profile


def test_named_local_ollama_uses_custom_profile():
    import model_tools  # noqa: F401
    import providers

    agent = SimpleNamespace(
        provider="ollama-launch",
        base_url="http://127.0.0.1:11434/v1",
        api_key="",
    )
    with patch(
        "agent.model_metadata.detect_local_server_type",
        return_value="ollama",
    ):
        profile = _resolve_chat_provider_profile(agent)
    assert profile is providers.get_provider_profile("custom")


def test_registered_provider_is_unchanged():
    import model_tools  # noqa: F401
    import providers

    agent = SimpleNamespace(
        provider="ollama-cloud",
        base_url="https://ollama.com/v1",
        api_key="sk-test",
    )
    profile = _resolve_chat_provider_profile(agent)
    assert profile is providers.get_provider_profile("ollama-cloud")


def test_unknown_remote_provider_stays_unresolved():
    agent = SimpleNamespace(
        provider="mystery-lab",
        base_url="https://api.example.com/v1",
        api_key="sk-test",
    )
    assert _resolve_chat_provider_profile(agent) is None
