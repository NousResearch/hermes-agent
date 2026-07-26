"""Tests for system-prompt model-identity sync across provider failover.

The system prompt is session-stable and embeds ``Model:``/``Provider:``
identity lines.  When ``try_activate_fallback`` swaps the runtime, the
prompt must be rewritten in place (and synced into the in-flight
``api_messages``) or the agent reports the primary model's name while a
fallback model is answering — e.g. a local gemma fallback claiming to be
gpt-5.4-mini after a Codex usage-limit 429.
"""

from types import SimpleNamespace

from agent.chat_completion_helpers import rewrite_prompt_model_identity
from agent.conversation_loop import _sync_failover_system_message
from agent.prompt_caching import apply_anthropic_cache_control


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


class TestSyncFailoverPreservesCacheDecoration:
    """The sync must not flatten a cache-decorated system message.

    ``apply_anthropic_cache_control`` runs once per call block, before the retry
    loop, splitting the system prompt into ``[static prefix, volatile tail]``
    blocks that carry the cache_control breakpoints. A failover fires *inside*
    that retry loop, so overwriting the list with a bare string drops both
    breakpoints and the retried request re-bills the whole system prompt.
    """

    _STATIC = "You are a helpful assistant.\n\nStable brief.\n"

    def _decorated(self, prompt):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "what model are you?"},
        ]
        return apply_anthropic_cache_control(
            messages,
            cache_ttl=None,
            native_anthropic=True,
            static_system_prefix=self._STATIC,
        )

    def test_keeps_breakpoints_and_static_prefix(self):
        prompt = self._STATIC + "Model: gpt-5.4-mini\nProvider: openai-codex"
        agent = _agent(prompt=prompt)
        api_messages = self._decorated(prompt)
        assert isinstance(api_messages[0]["content"], list)

        rewrite_prompt_model_identity(agent, "gemma4:e2b-mlx", "custom")
        _sync_failover_system_message(agent, api_messages, prompt)

        content = api_messages[0]["content"]
        assert isinstance(content, list), "cache decoration was flattened to a string"
        assert len(content) == 2
        assert all(part.get("cache_control") for part in content), (
            "the failover retry would ship zero system cache breakpoints"
        )
        # The static prefix must stay byte-identical or its cache entry misses.
        assert content[0]["text"] == self._STATIC
        # The identity refresh still lands, in the volatile tail.
        assert "Model: gemma4:e2b-mlx" in content[1]["text"]
        assert "Provider: custom" in content[1]["text"]

    def test_keeps_single_block_shape(self):
        prompt = "Model: gpt-5.4-mini\nProvider: openai-codex"
        agent = _agent(prompt=prompt)
        # No static prefix match -> the single-block fallback layout.
        api_messages = apply_anthropic_cache_control(
            [{"role": "system", "content": prompt}],
            cache_ttl=None,
            native_anthropic=True,
            static_system_prefix=None,
        )
        assert isinstance(api_messages[0]["content"], list)
        assert len(api_messages[0]["content"]) == 1

        rewrite_prompt_model_identity(agent, "gemma4:e2b-mlx", "custom")
        _sync_failover_system_message(agent, api_messages, prompt)

        content = api_messages[0]["content"]
        assert isinstance(content, list) and len(content) == 1
        assert content[0].get("cache_control")
        assert "Model: gemma4:e2b-mlx" in content[0]["text"]

    def test_ephemeral_prompt_lands_in_the_volatile_tail(self):
        prompt = self._STATIC + "Model: gpt-5.4-mini\nProvider: openai-codex"
        agent = _agent(prompt=prompt, ephemeral="Stay terse.")
        api_messages = self._decorated(prompt)

        _sync_failover_system_message(agent, api_messages, prompt)

        content = api_messages[0]["content"]
        assert content[0]["text"] == self._STATIC
        assert content[1]["text"].endswith("Stay terse.")

    def test_unknown_block_shape_falls_back_to_string(self):
        agent = _agent()
        api_messages = [
            {"role": "system", "content": [{"type": "image", "source": {}}]},
        ]
        _sync_failover_system_message(agent, api_messages, _PROMPT)
        assert api_messages[0]["content"] == agent._cached_system_prompt
