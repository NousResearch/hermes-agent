"""Regression tests for _spawn_background_review credential propagation.

Covers issue #15543: when a custom provider is configured via config.yaml
(not a built-in PROVIDER_REGISTRY entry), the background review agent must
inherit the parent agent's already-resolved api_key, base_url, and api_mode
rather than re-resolving them from scratch (which fails for custom providers).
"""

import threading
from types import SimpleNamespace
from unittest.mock import patch

from run_agent import AIAgent


# Real method captured before any patching so we can call it on a stub.
_spawn_background_review = AIAgent._spawn_background_review


def _make_parent(
    *,
    provider="openrouter",
    api_key="sk-or-test-key-1234",
    base_url="https://openrouter.ai/api/v1",
    api_mode="chat_completions",
):
    """Return a minimal SimpleNamespace that satisfies _spawn_background_review."""
    return SimpleNamespace(
        model="deepseek/deepseek-chat",
        platform="cli",
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        api_mode=api_mode,
        session_id="sess-test-001",
        _memory_store=None,
        _memory_enabled=False,
        _user_profile_enabled=False,
        _memory_nudge_interval=0,
        _skill_nudge_interval=0,
        _session_messages=[],
        _safe_print=lambda *a: None,
        background_review_callback=None,
        _MEMORY_REVIEW_PROMPT="review-memory",
        _SKILL_REVIEW_PROMPT="review-skills",
        _COMBINED_REVIEW_PROMPT="review-both",
        _summarize_background_review_actions=staticmethod(lambda msgs, snap: []),
        _emit_auxiliary_failure=lambda *a: None,
    )


def _run_and_capture(parent, *, review_memory=True, review_skills=False):
    """Invoke _spawn_background_review and return the AIAgent constructor kwargs."""
    captured = {}
    done = threading.Event()

    def fake_agent_factory(**kwargs):
        captured.update(kwargs)
        done.set()
        stub = SimpleNamespace(
            _session_messages=[],
            run_conversation=lambda **k: None,
            close=lambda: None,
        )
        return stub

    with patch("run_agent.AIAgent", side_effect=fake_agent_factory):
        _spawn_background_review(
            parent,
            [{"role": "user", "content": "hello"}],
            review_memory=review_memory,
            review_skills=review_skills,
        )
        done.wait(timeout=5)

    return captured


class TestBackgroundReviewCredentialPropagation:
    """AIAgent._spawn_background_review forwards resolved credentials to the review agent."""

    def test_api_key_forwarded(self):
        """api_key from the parent agent reaches the review agent constructor."""
        parent = _make_parent(api_key="sk-custom-key-abcdefgh")
        kwargs = _run_and_capture(parent)
        assert kwargs.get("api_key") == "sk-custom-key-abcdefgh"

    def test_base_url_forwarded(self):
        """base_url from the parent agent reaches the review agent constructor."""
        parent = _make_parent(base_url="https://myhost.internal/v1")
        kwargs = _run_and_capture(parent)
        assert kwargs.get("base_url") == "https://myhost.internal/v1"

    def test_api_mode_forwarded(self):
        """api_mode from the parent agent reaches the review agent constructor."""
        parent = _make_parent(api_mode="chat_completions")
        kwargs = _run_and_capture(parent)
        assert kwargs.get("api_mode") == "chat_completions"

    def test_all_three_credentials_forwarded_for_custom_provider(self):
        """Full regression case: custom provider with non-standard credentials."""
        parent = _make_parent(
            provider="custom-self-hosted",
            api_key="Bearer my-local-token",
            base_url="http://localhost:11434/v1",
            api_mode="chat_completions",
        )
        kwargs = _run_and_capture(parent)
        assert kwargs["api_key"] == "Bearer my-local-token"
        assert kwargs["base_url"] == "http://localhost:11434/v1"
        assert kwargs["api_mode"] == "chat_completions"
        assert kwargs["provider"] == "custom-self-hosted"

    def test_empty_credentials_forwarded_unchanged(self):
        """Empty api_key and base_url are forwarded as-is, not substituted."""
        parent = _make_parent(api_key="", base_url="")
        kwargs = _run_and_capture(parent)
        assert kwargs.get("api_key") == ""
        assert kwargs.get("base_url") == ""

    def test_anthropic_messages_mode_forwarded(self):
        """anthropic_messages api_mode is preserved for Anthropic-compat providers."""
        parent = _make_parent(
            provider="minimax",
            base_url="https://api.minimax.chat/v1/anthropic",
            api_mode="anthropic_messages",
        )
        kwargs = _run_and_capture(parent)
        assert kwargs.get("api_mode") == "anthropic_messages"

    def test_other_fields_still_forwarded(self):
        """Non-credential fields (model, platform, provider) remain correct."""
        parent = _make_parent(provider="openai")
        kwargs = _run_and_capture(parent)
        assert kwargs.get("model") == "deepseek/deepseek-chat"
        assert kwargs.get("platform") == "cli"
        assert kwargs.get("provider") == "openai"
        assert kwargs.get("parent_session_id") == "sess-test-001"
