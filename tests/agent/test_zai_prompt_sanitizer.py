"""Tests for agent/zai_prompt_sanitizer.py — outbound request-boundary sanitization.

Z.ai/Zhipu Coding Plan returns a bogus HTTP 429 / code 1305 when the system
prompt contains the exact phrase "Hermes Agent" (#47685, #53002). The
sanitizer rewrites that phrase on the outbound API-message copy only — never
the cached or persisted prompt — so resumed sessions are covered and the
byte-stability invariant for prefix-cache warmth is preserved.
"""
from types import SimpleNamespace

from agent.zai_prompt_sanitizer import (
    is_zai_request,
    sanitize_zai_api_messages,
    sanitize_zai_system_content,
)


def _agent(**overrides):
    base = dict(provider="zai", model="glm-5.2", base_url="https://api.z.ai/api/coding/paas/v4")
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# is_zai_request — detection across provider / model / base_url
# ---------------------------------------------------------------------------


class TestIsZaiRequest:
    def test_detects_by_provider(self):
        assert is_zai_request(_agent(provider="zai"))
        assert is_zai_request(_agent(provider="zhipu"))

    def test_detects_by_model_slug(self):
        assert is_zai_request(_agent(provider="custom", model="glm-5.2"))

    def test_detects_by_base_url(self):
        assert is_zai_request(_agent(provider="custom", model="x", base_url="https://api.z.ai/api/coding/paas/v4"))
        assert is_zai_request(_agent(provider="custom", model="x", base_url="https://open.bigmodel.cn/api/paas/v4"))

    def test_detects_anthropic_bridge_endpoint(self):
        assert is_zai_request(_agent(base_url="https://api.z.ai/api/anthropic"))

    def test_not_zai_for_other_providers(self):
        assert not is_zai_request(_agent(provider="openai-codex", model="gpt-5.5", base_url="https://chatgpt.com/backend-api/codex"))
        assert not is_zai_request(_agent(provider="openrouter", model="anthropic/claude-opus-4-1", base_url="https://openrouter.ai/api/v1"))
        assert not is_zai_request(_agent(provider="openrouter", model="anthropic/claude-opus-4-1", base_url="https://openrouter.ai/api/v1"))


# ---------------------------------------------------------------------------
# sanitize_zai_api_messages — outbound copy, cache untouched
# ---------------------------------------------------------------------------


class TestSanitizeApiMessages:
    def test_rewrites_phrase_in_system_message(self):
        agent = _agent()
        original = [
            {"role": "system", "content": "You are Hermes Agent, an assistant."},
            {"role": "user", "content": "hi"},
        ]
        out = sanitize_zai_api_messages(agent, original)
        assert "Hermes Agent" not in out[0]["content"]
        assert "Hermes framework" in out[0]["content"]

    def test_does_not_mutate_input(self):
        """The caller's list/dicts must not be mutated — cache & history stay intact."""
        agent = _agent()
        original = [
            {"role": "system", "content": "You are Hermes Agent, an assistant."},
            {"role": "user", "content": "hi"},
        ]
        original_system_before = original[0]["content"]
        sanitize_zai_api_messages(agent, original)
        assert original[0]["content"] == original_system_before  # input unchanged

    def test_user_and_assistant_messages_untouched(self):
        agent = _agent()
        original = [
            {"role": "system", "content": "Hermes Agent"},
            {"role": "user", "content": "tell me about Hermes Agent"},
            {"role": "assistant", "content": "Hermes Agent is here"},
        ]
        out = sanitize_zai_api_messages(agent, original)
        # Only system sanitized; user/assistant content is verbatim.
        assert "Hermes Agent" in out[1]["content"]
        assert "Hermes Agent" in out[2]["content"]
        assert "Hermes Agent" not in out[0]["content"]

    def test_resumed_branded_prompt_is_sanitized(self):
        """Resumed sessions restore a stored prompt verbatim (conversation_loop
        :304-308 returns early, bypassing assembly). The request-boundary
        sanitizer must still catch it on the outbound copy."""
        agent = _agent()
        # Simulate a persisted/resumed branded prompt.
        resumed = [
            {"role": "system", "content": "You are Hermes Agent, created by Nous Research.\nConversation started: 2026-06-26"},
            {"role": "user", "content": "continue"},
        ]
        out = sanitize_zai_api_messages(agent, resumed)
        assert "Hermes Agent" not in out[0]["content"]
        assert "Hermes framework" in out[0]["content"]
        # Conversation metadata preserved.
        assert "Conversation started" in out[0]["content"]

    def test_handles_content_block_list_form(self):
        """OpenAI content-block list form ([{"type":"text","text":...}])."""
        agent = _agent()
        original = [
            {"role": "system", "content": [{"type": "text", "text": "You are Hermes Agent."}]},
            {"role": "user", "content": "hi"},
        ]
        out = sanitize_zai_api_messages(agent, original)
        assert "Hermes Agent" not in out[0]["content"][0]["text"]
        assert "Hermes framework" in out[0]["content"][0]["text"]

    def test_no_system_message_passes_through(self):
        agent = _agent()
        original = [{"role": "user", "content": "hi"}]
        out = sanitize_zai_api_messages(agent, original)
        assert out == original


# ---------------------------------------------------------------------------
# Non-zai providers — zero overhead, byte-identical
# ---------------------------------------------------------------------------


class TestNonZaiPassthrough:
    def test_returns_same_reference_for_non_zai(self):
        """Non-zai requests get the original list back (same object) — no copy."""
        agent = _agent(provider="openrouter", model="anthropic/claude-opus-4-1", base_url="https://openrouter.ai/api/v1")
        original = [{"role": "system", "content": "You are Hermes Agent."}]
        out = sanitize_zai_api_messages(agent, original)
        assert out is original  # same reference — zero overhead

    def test_phrase_preserved_for_non_zai(self):
        agent = _agent(provider="openrouter", model="anthropic/claude-opus-4-1", base_url="https://openrouter.ai/api/v1")
        original = [{"role": "system", "content": "You are Hermes Agent, an assistant."}]
        out = sanitize_zai_api_messages(agent, original)
        assert "Hermes Agent" in out[0]["content"]
        assert "Hermes framework" not in out[0]["content"]


# ---------------------------------------------------------------------------
# sanitize_zai_system_content — unit-level
# ---------------------------------------------------------------------------


class TestSanitizeSystemContent:
    def test_string_rewrite(self):
        assert sanitize_zai_system_content("Hermes Agent here") == "Hermes framework here"

    def test_string_no_phrase_unchanged(self):
        assert sanitize_zai_system_content("just a prompt") == "just a prompt"

    def test_non_string_unchanged(self):
        assert sanitize_zai_system_content(None) is None
        assert sanitize_zai_system_content(42) == 42

    def test_replaces_all_occurrences(self):
        assert sanitize_zai_system_content("Hermes Agent and Hermes Agent") == "Hermes framework and Hermes framework"
