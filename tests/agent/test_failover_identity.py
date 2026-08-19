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
    model="gpt-4o",
    tools=None,
    direct_tool_cache=False,
):
    return SimpleNamespace(
        _cached_system_prompt=prompt,
        _cached_system_prompt_static=static,
        ephemeral_system_prompt=None,
        _use_prompt_caching=use_caching,
        _use_native_cache_layout=native,
        _cache_ttl=cache_ttl,
        provider=provider,
        model=model,
        tools=tools or [],
        _direct_native_anthropic_tool_cache_capability=lambda: direct_tool_cache,
        client=None,
    )


class TestRewritePromptModelIdentity:

    def test_only_last_occurrence_is_rewritten(self):
        agent = _agent()
        rewrite_prompt_model_identity(agent, "gemma4:e2b-mlx", "custom")
        # Earlier matching lines may be user content (memory snapshots,
        # context files) and must survive untouched.
        assert "Model: decoy-from-memory" in agent._cached_system_prompt


    def test_noop_when_prompt_missing_or_empty(self):
        for prompt in (None, ""):
            agent = _agent(prompt=prompt)
            rewrite_prompt_model_identity(agent, "m", "p")
            assert agent._cached_system_prompt == prompt



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


    def test_noop_without_cached_prompt(self):
        agent = _agent(prompt=None)
        api_messages = [{"role": "system", "content": "original"}]
        result = _sync_failover_system_message(agent, api_messages, "active")
        assert api_messages[0]["content"] == "original"
        assert result == "active"



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

    def test_replans_tools_for_the_active_destination(self):
        from agent.prompt_caching import build_prompt_cache_plan

        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]
        messages = [
            {"role": "system", "content": self._STATIC + "volatile"},
            {"role": "user", "content": "lookup"},
        ]
        source = build_prompt_cache_plan(
            messages,
            tools,
            native_anthropic=True,
            static_system_prefix=self._STATIC,
            direct_native_tool_cache=True,
        )
        agent = _cache_agent(
            use_caching=True,
            native=False,
            prompt=messages[0]["content"],
            static=self._STATIC,
            provider="openrouter",
            tools=tools,
            direct_tool_cache=False,
        )

        fallback_messages, _, fallback_tools = _redecorate_prompt_cache_for_provider(
            agent,
            source.messages,
            tools_for_api=source.tools,
        )

        assert "cache_control" in source.tools[-1]
        assert "cache_control" not in fallback_tools[-1]
        assert "cache_control" not in tools[-1]
        assert all(
            part.get("cache_control")
            for part in fallback_messages[0]["content"]
            if isinstance(part, dict)
        )




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


    def test_list_part_shape(self):
        base = [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "task", "cache_control": {"type": "ephemeral"}}],
            },
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


# =============================================================================
# Failover vision support sync
# =============================================================================


class TestFailoverVisionSupport:
    """``_sync_failover_vision_support`` replaces image parts with auxiliary
    vision descriptions when the failover model is text-only, so a retry
    doesn't 400 on embedded image_url parts and the model still learns what
    the image contained."""

    def _agent(self, *, provider="deepseek", model="deepseek-v4-flash"):
        return SimpleNamespace(
            provider=provider,
            model=model,
            requested_provider=provider,
            _vision_supported=True,
        )

    def _image_messages(self):
        return [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
                ],
            },
        ]

    def _patch_lookup(self, monkeypatch, result):
        def _fake_lookup(provider, model, cfg, requested_provider=""):
            return result

        monkeypatch.setattr(
            "agent.image_routing._lookup_supports_vision", _fake_lookup
        )

    def _patch_aux_vision(self, monkeypatch, description="a screenshot of a dashboard"):
        def _fake_aux_vision(image_ref, timeout=90.0):
            return description

        monkeypatch.setattr(
            "agent.conversation_loop._run_aux_vision_sync", _fake_aux_vision
        )

    def test_text_only_fallback_replaces_with_description(self, monkeypatch):
        agent = self._agent(provider="deepseek", model="deepseek-v4-flash")
        messages = self._image_messages()
        self._patch_lookup(monkeypatch, False)
        self._patch_aux_vision(monkeypatch, "deepseek dashboard with balance $0.98")

        from agent.conversation_loop import _sync_failover_vision_support

        replaced = _sync_failover_vision_support(agent, messages)
        assert replaced is True
        assert agent._vision_supported is False
        user_content = messages[1]["content"]
        # Image part replaced with a text description, original text kept.
        assert all(p.get("type") != "image_url" for p in user_content)
        texts = [p.get("text", "") for p in user_content if p.get("type") == "text"]
        assert any("look at this" in t for t in texts)
        assert any("dashboard with balance" in t for t in texts)

    def test_aux_vision_failure_falls_back_to_placeholder(self, monkeypatch):
        agent = self._agent(provider="deepseek", model="deepseek-v4-flash")
        messages = self._image_messages()
        self._patch_lookup(monkeypatch, False)
        self._patch_aux_vision(monkeypatch, "")  # aux vision failed

        from agent.conversation_loop import _sync_failover_vision_support

        replaced = _sync_failover_vision_support(agent, messages)
        assert replaced is True
        assert agent._vision_supported is False
        user_content = messages[1]["content"]
        texts = [p.get("text", "") for p in user_content if p.get("type") == "text"]
        assert any("image content removed" in t for t in texts)

    def test_vision_fallback_keeps_images(self, monkeypatch):
        agent = self._agent(provider="litellm", model="kimi-router")
        messages = self._image_messages()
        self._patch_lookup(monkeypatch, True)
        self._patch_aux_vision(monkeypatch)

        from agent.conversation_loop import _sync_failover_vision_support

        replaced = _sync_failover_vision_support(agent, messages)
        assert replaced is False
        assert agent._vision_supported is True
        user_content = messages[1]["content"]
        assert any(p.get("type") == "image_url" for p in user_content)

    def test_already_vision_unsupported_skips(self, monkeypatch):
        agent = self._agent()
        agent._vision_supported = False
        messages = self._image_messages()
        self._patch_lookup(monkeypatch, True)
        self._patch_aux_vision(monkeypatch)

        from agent.conversation_loop import _sync_failover_vision_support

        replaced = _sync_failover_vision_support(agent, messages)
        assert replaced is False
        # Images untouched.
        user_content = messages[1]["content"]
        assert any(p.get("type") == "image_url" for p in user_content)

    def test_tool_message_placeholder_preserves_linkage(self, monkeypatch):
        """A tool message whose content was entirely images keeps its
        message (and tool_call_id linkage); only the part is replaced."""
        agent = self._agent()
        self._patch_lookup(monkeypatch, False)
        self._patch_aux_vision(monkeypatch, "")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}}],
            },
        ]
        from agent.conversation_loop import _sync_failover_vision_support

        replaced = _sync_failover_vision_support(agent, messages)
        assert replaced is True
        assert len(messages) == 3  # tool message NOT deleted
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "call_1"
        content = messages[2]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert "image content removed" in content[0]["text"]


class TestFailoverVisionDisablesFastPath:
    """When _sync_failover_vision_support fires, it must also set the
    module-level ``_failover_vision_disabled`` flag in vision_tools so that
    ``_should_use_native_vision_fast_path`` returns False for the rest of the
    turn.  Otherwise the fast path (which reads stale runtime-main globals
    still pointing at the vision-capable primary) would deliver image bytes
    to the text-only fallback model on every vision_analyze tool call."""

    def test_text_only_failover_sets_vision_tools_flag(self, monkeypatch):
        import tools.vision_tools as vt

        vt._failover_vision_disabled = False  # reset
        agent = SimpleNamespace(
            provider="deepseek",
            model="deepseek-v4-flash",
            requested_provider="deepseek",
            _vision_supported=True,
        )
        messages = [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
                ],
            },
        ]

        def _fake_lookup(provider, model, cfg, requested_provider=""):
            return False

        def _fake_aux(image_ref, timeout=90.0):
            return "a red car"

        monkeypatch.setattr("agent.image_routing._lookup_supports_vision", _fake_lookup)
        monkeypatch.setattr("agent.conversation_loop._run_aux_vision_sync", _fake_aux)

        from agent.conversation_loop import _sync_failover_vision_support

        _sync_failover_vision_support(agent, messages)
        assert vt._failover_vision_disabled is True

    def test_fast_path_returns_false_after_failover(self, monkeypatch):
        import tools.vision_tools as vt

        vt._failover_vision_disabled = True
        assert vt._should_use_native_vision_fast_path() is False
        vt._failover_vision_disabled = False  # cleanup
