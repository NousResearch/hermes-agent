"""Tests for system-prompt model-identity sync across provider failover.

The system prompt is session-stable and embeds ``Model:``/``Provider:``
identity lines.  When ``try_activate_fallback`` swaps the runtime, the
prompt must be rewritten in place (and synced into the in-flight
``api_messages``) or the agent reports the primary model's name while a
fallback model is answering — e.g. a local gemma fallback claiming to be
gpt-5.4-mini after a Codex usage-limit 429.
"""

from types import SimpleNamespace

from agent.chat_completion_helpers import (
    rewrite_prompt_model_identity,
    sync_lmstudio_active_model,
)
from agent.conversation_loop import _sync_failover_system_message


_PROMPT = (
    "You are a helpful assistant.\n"
    "\n"
    "Memory note at line start:\n"
    "Model: decoy-from-memory\n"
    "\n"
    "Conversation started: Wednesday, June 10, 2026\n"
    "Model: gpt-5.4-mini\n"
    "Provider: openai-codex"
)


def _agent(prompt=_PROMPT, ephemeral=None):
    return SimpleNamespace(
        _cached_system_prompt=prompt,
        ephemeral_system_prompt=ephemeral,
    )


class TestRewritePromptModelIdentity:
    def test_swaps_identity_lines_to_fallback_runtime(self):
        agent = _agent()
        rewrite_prompt_model_identity(agent, "gemma4:e2b-mlx", "custom")
        assert "Model: gemma4:e2b-mlx" in agent._cached_system_prompt
        assert "Provider: custom" in agent._cached_system_prompt
        assert "Model: gpt-5.4-mini" not in agent._cached_system_prompt
        assert "Provider: openai-codex" not in agent._cached_system_prompt

    def test_only_last_occurrence_is_rewritten(self):
        agent = _agent()
        rewrite_prompt_model_identity(agent, "gemma4:e2b-mlx", "custom")
        # Earlier matching lines may be user content (memory snapshots,
        # context files) and must survive untouched.
        assert "Model: decoy-from-memory" in agent._cached_system_prompt

    def test_round_trip_restores_byte_identical_prompt(self):
        # restore_primary_runtime rewrites the lines back; the result must
        # match the stored prompt byte-for-byte so the primary's prefix
        # cache still hits after restoration.
        agent = _agent()
        rewrite_prompt_model_identity(agent, "gemma4:e2b-mlx", "custom")
        rewrite_prompt_model_identity(agent, "gpt-5.4-mini", "openai-codex")
        assert agent._cached_system_prompt == _PROMPT

    def test_noop_when_prompt_missing_or_empty(self):
        for prompt in (None, ""):
            agent = _agent(prompt=prompt)
            rewrite_prompt_model_identity(agent, "m", "p")
            assert agent._cached_system_prompt == prompt

    def test_empty_values_leave_lines_unchanged(self):
        agent = _agent()
        rewrite_prompt_model_identity(agent, "", "")
        assert agent._cached_system_prompt == _PROMPT


_LMSTUDIO_PROMPT = (
    "You are a helpful assistant.\n"
    "\n"
    "Conversation started: Wednesday, June 10, 2026\n"
    "Model: gemma-3-4b\n"
    "Provider: lmstudio"
)


def _lmstudio_agent(model="gemma-3-4b", prompt=_LMSTUDIO_PROMPT):
    return SimpleNamespace(
        provider="lmstudio",
        model=model,
        _cached_system_prompt=prompt,
        ephemeral_system_prompt=None,
    )


class TestSyncLmstudioActiveModel:
    def test_adopts_swapped_model_and_rewrites_identity(self):
        # User swapped Gemma -> Qwen in the LM Studio app mid-session (#54454).
        agent = _lmstudio_agent()
        sync_lmstudio_active_model(agent, SimpleNamespace(model="qwen3-6b"))
        assert agent.model == "qwen3-6b"
        assert "Model: qwen3-6b" in agent._cached_system_prompt
        assert "Model: gemma-3-4b" not in agent._cached_system_prompt

    def test_noop_when_model_unchanged(self):
        agent = _lmstudio_agent()
        sync_lmstudio_active_model(agent, SimpleNamespace(model="gemma-3-4b"))
        assert agent.model == "gemma-3-4b"
        assert agent._cached_system_prompt == _LMSTUDIO_PROMPT

    def test_noop_on_slug_vs_basename_match(self):
        # LM Studio native API returns "publisher/slug"; the configured value is
        # the bare slug — these are the same model, not a drift.
        agent = _lmstudio_agent(model="gemma-3-4b")
        sync_lmstudio_active_model(agent, SimpleNamespace(model="google/gemma-3-4b"))
        assert agent.model == "gemma-3-4b"
        assert agent._cached_system_prompt == _LMSTUDIO_PROMPT

    def test_noop_for_non_lmstudio_provider(self):
        agent = _lmstudio_agent()
        agent.provider = "openai"
        sync_lmstudio_active_model(agent, SimpleNamespace(model="qwen3-6b"))
        assert agent.model == "gemma-3-4b"
        assert agent._cached_system_prompt == _LMSTUDIO_PROMPT

    def test_noop_when_response_has_no_model(self):
        agent = _lmstudio_agent()
        sync_lmstudio_active_model(agent, SimpleNamespace(model=None))
        assert agent.model == "gemma-3-4b"
        sync_lmstudio_active_model(agent, SimpleNamespace(model="   "))
        assert agent.model == "gemma-3-4b"


class TestSyncFailoverSystemMessage:
    def test_patches_in_flight_system_message(self):
        agent = _agent()
        rewrite_prompt_model_identity(agent, "gemma4:e2b-mlx", "custom")
        api_messages = [
            {"role": "system", "content": _PROMPT},
            {"role": "user", "content": "what model are you?"},
        ]
        result = _sync_failover_system_message(agent, api_messages, _PROMPT)
        assert "Model: gemma4:e2b-mlx" in api_messages[0]["content"]
        assert result == agent._cached_system_prompt

    def test_appends_ephemeral_system_prompt(self):
        agent = _agent(ephemeral="Stay terse.")
        api_messages = [{"role": "system", "content": _PROMPT}]
        _sync_failover_system_message(agent, api_messages, _PROMPT)
        assert api_messages[0]["content"].endswith("Stay terse.")

    def test_noop_without_cached_prompt(self):
        agent = _agent(prompt=None)
        api_messages = [{"role": "system", "content": "original"}]
        result = _sync_failover_system_message(agent, api_messages, "active")
        assert api_messages[0]["content"] == "original"
        assert result == "active"

    def test_noop_when_first_message_is_not_system(self):
        agent = _agent()
        api_messages = [{"role": "user", "content": "hi"}]
        result = _sync_failover_system_message(agent, api_messages, "active")
        assert api_messages == [{"role": "user", "content": "hi"}]
        # Still returns the cached prompt for subsequent call-block rebuilds.
        assert result == agent._cached_system_prompt
