"""Review/side-question forks get their OWN cache scope on slot-keyed caches.

A background-review fork shares the parent's ``session_id`` so content-
addressed caches (Anthropic, DeepSeek, Gemini) and OpenAI-style routing keys
serve it warm. xAI is different: ``x-grok-conv-id`` / ``prompt_cache_key``
select ONE server-side conversation slot, so the fork's divergent request
stream evicts the parent's — measured on xai-oauth, the parent's next call
read 1,152 of ~356k prompt tokens after a fork. The resolver derives
``<scope>::<tag>`` for tagged forks on xAI routes only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.prompt_cache_scope import (
    is_fork_cache_scope,
    is_slot_keyed_cache_route,
    resolve_prompt_cache_scope,
)

SYS = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "hi"},
]


def _agent(provider, model="grok-4.3", tag=None, base_url=""):
    a = SimpleNamespace(
        session_id="parent-sess",
        _session_db=None,
        provider=provider,
        model=model,
        base_url=base_url,
    )
    if tag is not None:
        a._prompt_cache_fork_tag = tag
    return a


class TestSlotKeyedRoute:
    @pytest.mark.parametrize(
        "provider,model,base_url",
        [
            ("xai-oauth", "grok-4.3", ""),
            ("xai", "grok-4", ""),
            ("custom", "grok-4", "https://api.x.ai/v1"),
            ("openrouter", "x-ai/grok-4", ""),
        ],
    )
    def test_xai_routes_are_slot_keyed(self, provider, model, base_url):
        assert is_slot_keyed_cache_route(provider, model, base_url)

    @pytest.mark.parametrize(
        "provider,model",
        [
            ("anthropic", "claude-opus-4-8"),
            ("claude-bpr", "claude-opus-4-8"),
            ("openai-codex", "gpt-5.5"),
            ("openrouter", "anthropic/claude-sonnet-4.5"),
            ("gemini", "gemini-3-pro"),
        ],
    )
    def test_content_addressed_and_routing_providers_are_not(self, provider, model):
        assert not is_slot_keyed_cache_route(provider, model, "")


class TestResolverDerivesForkScope:
    def test_parent_on_xai_keeps_its_scope(self):
        assert resolve_prompt_cache_scope(_agent("xai-oauth")) == "parent-sess"

    def test_review_fork_on_xai_gets_derived_scope(self):
        scope = resolve_prompt_cache_scope(_agent("xai-oauth", tag="review"))
        assert scope == "parent-sess::review"
        assert is_fork_cache_scope(scope)

    def test_side_question_fork_gets_its_own_tag(self):
        assert (
            resolve_prompt_cache_scope(_agent("xai-oauth", tag="side_question"))
            == "parent-sess::side_question"
        )

    @pytest.mark.parametrize(
        "provider,model",
        [("anthropic", "claude-opus-4-8"), ("openai-codex", "gpt-5.5"),
         ("openrouter", "anthropic/claude-sonnet-4.5")],
    )
    def test_fork_on_shared_key_provider_keeps_parent_scope(self, provider, model):
        # #109964-style sharing stays where it is a measured win.
        assert (
            resolve_prompt_cache_scope(_agent(provider, model, tag="review"))
            == "parent-sess"
        )

    def test_memo_is_not_polluted_and_fallback_is_reevaluated(self):
        fork = _agent("xai-oauth", tag="review")
        assert resolve_prompt_cache_scope(fork) == "parent-sess::review"
        assert resolve_prompt_cache_scope(fork) == "parent-sess::review"
        fork.provider, fork.model = "anthropic", "claude-opus-4-8"  # fallback hop
        assert resolve_prompt_cache_scope(fork) == "parent-sess"

    def test_inherited_parent_scope_is_derived_on_xai_only(self):
        """#109964 inheritance stays for content-addressed providers, derives on xAI."""
        fork = _agent("xai-oauth", tag="review")
        fork._inherited_cache_scope = "gwk_parentscope"
        assert resolve_prompt_cache_scope(fork) == "gwk_parentscope::review"
        fork.provider, fork.model = "anthropic", "claude-opus-4-8"
        assert resolve_prompt_cache_scope(fork) == "gwk_parentscope"

    def test_gateway_style_single_colon_ids_are_not_fork_scopes(self):
        assert not is_fork_cache_scope("agent:main:telegram:dm:42")


class TestTransportWire:
    """What actually goes on the wire for parent vs fork."""

    def _xai(self, agent):
        from agent.transports.codex import ResponsesApiTransport

        return ResponsesApiTransport().build_kwargs(
            model="grok-4.3", messages=SYS, tools=[],
            session_id=agent.session_id,
            cache_scope_id=resolve_prompt_cache_scope(agent),
            is_xai_responses=True,
        )

    def test_xai_fork_sends_distinct_conv_id_and_cache_key(self):
        parent = self._xai(_agent("xai-oauth"))
        fork = self._xai(_agent("xai-oauth", tag="review"))
        assert parent["extra_headers"]["x-grok-conv-id"] == "parent-sess"
        assert fork["extra_headers"]["x-grok-conv-id"] == "parent-sess::review"
        assert (
            parent["extra_body"]["prompt_cache_key"]
            != fork["extra_body"]["prompt_cache_key"]
        )

    def test_negative_control_untagged_fork_collides(self):
        """Pre-fix shape: same scope -> same slot -> the eviction."""
        parent = self._xai(_agent("xai-oauth"))
        fork = self._xai(_agent("xai-oauth"))
        assert parent["extra_headers"]["x-grok-conv-id"] == fork["extra_headers"]["x-grok-conv-id"]
        assert parent["extra_body"]["prompt_cache_key"] == fork["extra_body"]["prompt_cache_key"]

    def test_codex_fork_keeps_shared_prompt_cache_key(self):
        from agent.transports.codex import ResponsesApiTransport

        def build(agent):
            return ResponsesApiTransport().build_kwargs(
                model="gpt-5.5", messages=SYS, tools=[],
                session_id=agent.session_id,
                cache_scope_id=resolve_prompt_cache_scope(agent),
                is_codex_backend=True,
            )

        parent = build(_agent("openai-codex", "gpt-5.5"))
        fork = build(_agent("openai-codex", "gpt-5.5", tag="review"))
        assert parent["prompt_cache_key"] == fork["prompt_cache_key"]

    def test_openrouter_grok_fork_overrides_ambient_conversation(self):
        from agent.portal_tags import reset_conversation_context, set_conversation_context
        from providers import get_provider_profile

        p = get_provider_profile("openrouter")
        token = set_conversation_context("parent-sess")  # fork copies parent's Context
        try:
            _, parent = p.build_api_kwargs_extras(
                model="x-ai/grok-4", session_id="parent-sess",
                cache_scope_id="parent-sess",
            )
            _, fork = p.build_api_kwargs_extras(
                model="x-ai/grok-4", session_id="parent-sess",
                cache_scope_id="parent-sess::review",
            )
        finally:
            reset_conversation_context(token)
        assert parent["extra_headers"]["x-grok-conv-id"] == "parent-sess"
        assert fork["extra_headers"]["x-grok-conv-id"] == "parent-sess::review"

    def test_chat_completions_threads_cache_scope_to_profile(self):
        from agent.transports.chat_completions import ChatCompletionsTransport
        from providers import get_provider_profile

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="x-ai/grok-4", messages=SYS, tools=None,
            provider_profile=get_provider_profile("openrouter"),
            session_id="parent-sess", cache_scope_id="parent-sess::review",
        )
        assert kwargs["extra_headers"]["x-grok-conv-id"] == "parent-sess::review"
