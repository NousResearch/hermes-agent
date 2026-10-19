"""Copilot provider detection used by the turn-retry recovery gates."""

from __future__ import annotations


def test_copilot_provider_check_accepts_alias_spellings():
    """`/model` and profile configs can leave `github-copilot` / `github` as
    the provider spelling; the recovery gates must not silently skip them."""
    from agent.conversation_loop import _is_copilot_provider
    from run_agent import AIAgent

    class _Agent:
        # Reuse the real single-owner check unbound; only provider/_base_url
        # state is faked.
        _is_copilot_provider = AIAgent._is_copilot_provider
        _is_copilot_url = AIAgent._is_copilot_url

        def __init__(self, provider, base_url=""):
            self.provider = provider
            self._base_url_lower = base_url.lower()

    assert _is_copilot_provider(_Agent("copilot"))
    assert _is_copilot_provider(_Agent("github-copilot"))
    assert _is_copilot_provider(_Agent("GitHub-Copilot"))
    assert _is_copilot_provider(_Agent("github"))
    # URL fallback: unnormalized provider but a Copilot base URL.
    assert _is_copilot_provider(_Agent("custom", "https://api.githubcopilot.com"))
    assert not _is_copilot_provider(_Agent("openrouter", "https://openrouter.ai/api/v1"))

    class _NoMethod:
        provider = "github-copilot"

    # Fallback path when the agent object lacks the method entirely.
    assert _is_copilot_provider(_NoMethod())
