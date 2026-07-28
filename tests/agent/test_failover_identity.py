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
from agent.conversation_loop import (
    _redecorate_prompt_cache_for_provider,
    _sync_failover_system_message,
)
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


def _count_cache_markers(messages):
    count = 0
    for msg in messages:
        if "cache_control" in msg:
            count += 1
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "cache_control" in part:
                    count += 1
    return count


def _cache_agent(
    *,
    use_caching,
    native=False,
    prompt=_PROMPT,
    static=None,
    cache_ttl="5m",
    provider="openai",
):
    return SimpleNamespace(
        _cached_system_prompt=prompt,
        _cached_system_prompt_static=static,
        ephemeral_system_prompt=None,
        _use_prompt_caching=use_caching,
        _use_native_cache_layout=native,
        _cache_ttl=cache_ttl,
        provider=provider,
        client=None,
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


class TestRedecoratePromptCacheOnPolicyChange:
    """#72626: failover must re-render breakpoints for the *new* cache policy.

    ``TestSyncFailoverPreservesCacheDecoration`` only covers same-policy
    identity sync (native→native). These cases cover the policy-change class.
    """

    _STATIC = "You are a helpful assistant.\n\nStable brief.\n"

    def test_cache_off_to_cache_on_adds_breakpoints(self):
        prompt = self._STATIC + "Model: gpt-5.4-mini\nProvider: openai"
        # Primary never decorated (cache-off).
        undecorated = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "again"},
        ]
        assert _count_cache_markers(undecorated) == 0

        agent = _cache_agent(
            use_caching=True,
            native=True,
            prompt=prompt,
            static=self._STATIC,
            provider="anthropic",
        )
        decorated, _ = _redecorate_prompt_cache_for_provider(agent, undecorated)
        assert _count_cache_markers(decorated) >= 2
        assert isinstance(decorated[0]["content"], list)
        assert decorated[0]["content"][0]["text"] == self._STATIC

    def test_cache_on_to_cache_off_strips_breakpoints(self):
        prompt = self._STATIC + "Model: claude\nProvider: anthropic"
        decorated = apply_anthropic_cache_control(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "hello"},
            ],
            native_anthropic=True,
            static_system_prefix=self._STATIC,
        )
        assert _count_cache_markers(decorated) >= 2

        agent = _cache_agent(
            use_caching=False,
            native=False,
            prompt=prompt,
            static=self._STATIC,
            provider="custom",
        )
        stripped, _ = _redecorate_prompt_cache_for_provider(agent, decorated)
        assert _count_cache_markers(stripped) == 0
        assert stripped[0]["content"] == prompt

    def test_native_to_envelope_relocates_breakpoints_to_carriers(self):
        prompt = "Model: claude\nProvider: anthropic"
        # Native layout can mark empty assistant turns; envelope cannot.
        native = apply_anthropic_cache_control(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "do it"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
                {"role": "tool", "content": "ok", "tool_call_id": "1"},
                {"role": "user", "content": "thanks"},
            ],
            native_anthropic=True,
        )
        agent = _cache_agent(
            use_caching=True,
            native=False,
            prompt=prompt,
            provider="openrouter",
        )
        envelope, _ = _redecorate_prompt_cache_for_provider(agent, native)
        # Empty assistant must not carry a wasted top-level marker on envelope.
        empty_assistant = next(
            m for m in envelope if m.get("role") == "assistant" and not m.get("content")
        )
        assert "cache_control" not in empty_assistant
        assert _count_cache_markers(envelope) >= 1

    def test_preserves_mutated_tool_result_bytes(self):
        # Redecoration must use the in-flight mutated list, not a pristine
        # pre-decoration snapshot — image-shrink / ASCII recoveries live here.
        prompt = "sys"
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "run"},
            {"role": "tool", "content": "SHRUNK_IMAGE_PAYLOAD", "tool_call_id": "t1"},
        ]
        agent = _cache_agent(use_caching=True, native=True, prompt=prompt)
        out, _ = _redecorate_prompt_cache_for_provider(agent, messages)
        tool = next(m for m in out if m.get("role") == "tool")
        text = (
            tool["content"]
            if isinstance(tool["content"], str)
            else tool["content"][0]["text"]
        )
        assert text == "SHRUNK_IMAGE_PAYLOAD"

    def test_moa_guidance_stays_outside_last_breakpoint(self):
        prompt = "sys"
        guidance = (
            "[Mixture of Agents context — use this as private guidance for the "
            "normal Hermes agent loop.]\nAggregator: agg\n\nadvice"
        )
        base = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "task\n\n" + guidance},
        ]
        decorated = apply_anthropic_cache_control(base, native_anthropic=True)

        class _Completions:
            def rebase_prepared_request(self, prepared, messages):
                from agent.moa_loop import _attach_reference_guidance

                out = [dict(m) for m in messages]
                _attach_reference_guidance(out, prepared["guidance"])
                return {**prepared, "messages": out}

        agent = _cache_agent(use_caching=True, native=True, prompt=prompt, provider="moa")
        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
        prepared = {"guidance": guidance, "messages": decorated}
        out, new_prepared = _redecorate_prompt_cache_for_provider(
            agent, decorated, moa_prepared=prepared
        )
        assert new_prepared is not None
        # Guidance must be present, but the cache marker on the last user
        # turn's content parts must terminate *before* the guidance text.
        last = out[-1]
        assert last.get("role") == "user"
        content = last["content"]
        if isinstance(content, list):
            marked = [p for p in content if isinstance(p, dict) and "cache_control" in p]
            guidance_parts = [
                p for p in content
                if isinstance(p, dict) and guidance in (p.get("text") or "")
            ]
            assert marked, "base transcript should still carry breakpoints"
            assert guidance_parts, "guidance must be re-attached"
            # Marked part should not contain the guidance body.
            assert all(guidance not in (p.get("text") or "") for p in marked)
        else:
            assert guidance in content


    def test_moa_no_guidance_prepared_messages_still_refreshed(self):
        # guidance=None is a real prepared shape (all references failed /
        # silent degraded policy). The MoA facade sends prepared["messages"],
        # so the rebase must refresh the prepared object even without
        # guidance — otherwise the aggregator ships the STALE decoration
        # and #72626 persists for the no-guidance MoA sub-path.
        prompt = "sys"
        decorated = apply_anthropic_cache_control(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "task"},
            ],
            native_anthropic=True,
        )

        class _Completions:
            def rebase_prepared_request(self, prepared, messages):
                # Mirrors MoAChatCompletions.rebase_prepared_request with
                # falsy guidance: copy messages, skip the attach.
                return {**prepared, "messages": [dict(m) for m in messages]}

        # Policy change while staying on moa: cache-on -> cache-off.
        agent = _cache_agent(use_caching=False, prompt=prompt, provider="moa")
        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
        prepared = {"guidance": None, "messages": decorated}
        out, new_prepared = _redecorate_prompt_cache_for_provider(
            agent, decorated, moa_prepared=prepared
        )
        assert new_prepared is not None
        # The prepared object the facade will send must carry the
        # redecorated (stripped) messages, not the stale decorated list.
        assert _count_cache_markers(new_prepared["messages"]) == 0
        assert new_prepared["messages"] == out


class TestPeelReferenceGuidanceRoundTrip:
    """peel must invert every attach shape — the two live adjacent in
    moa_loop.py precisely so this contract can't drift silently."""

    _GUIDANCE = "[Mixture of Agents reference context]\nAdvice body."

    def _round_trip(self, base):
        import copy

        from agent.moa_loop import _attach_reference_guidance, peel_reference_guidance

        attached = copy.deepcopy(base)
        _attach_reference_guidance(attached, self._GUIDANCE)
        assert attached != base, "attach must change the transcript"
        return peel_reference_guidance(attached, self._GUIDANCE)

    def test_string_merge_shape(self):
        base = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        assert self._round_trip(base) == base

    def test_list_part_shape(self):
        base = [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "task", "cache_control": {"type": "ephemeral"}}],
            },
        ]
        assert self._round_trip(base) == base

    def test_appended_user_message_shape(self):
        # No trailing user turn — attach appends a separate user message.
        base = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "done"},
        ]
        assert self._round_trip(base) == base

    def test_guidance_only_part_drops_message_not_empty_residue(self):
        # If the guidance part is the only content left after peeling, the
        # whole message goes — an empty-content user turn must never remain.
        from agent.moa_loop import peel_reference_guidance

        messages = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "ok"},
            {
                "role": "user",
                "content": [{"type": "text", "text": self._GUIDANCE}],
            },
        ]
        peeled = peel_reference_guidance(messages, self._GUIDANCE)
        assert peeled == messages[:-1]
