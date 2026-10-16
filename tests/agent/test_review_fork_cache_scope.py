"""A background-review fork gets its OWN cache scope on xAI once it has compacted.

A same-model fork shares the parent's scope (#109964), so its first request is a warm
prefix read. xAI's ``x-grok-conv-id`` / ``prompt_cache_key`` pin ONE server slot, so
once the fork's own in-place compaction diverges its stream it would evict the
parent's slot; the resolver then derives ``<scope>::<tag>`` on xAI routes only.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.prompt_cache_scope import resolve_prompt_cache_scope

SYS = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "hi"},
]


def _agent(provider, model="grok-4.3", tag=None, compactions=0):
    a = SimpleNamespace(
        session_id="parent-sess", _session_db=None, provider=provider, model=model,
        base_url="", context_compressor=SimpleNamespace(compression_count=compactions),
    )
    if tag is not None:
        a._prompt_cache_fork_tag = tag
    return a


def test_inherited_parent_scope_is_derived_on_xai_only_after_fork_compaction():
    """Request #1 shares the parent's key; post-compaction derives on xAI, never elsewhere."""
    fork = _agent("xai-oauth", tag="review")
    fork._inherited_cache_scope = "gwk_parentscope"
    assert resolve_prompt_cache_scope(fork) == "gwk_parentscope"
    fork.context_compressor.compression_count = 1  # the fork's own in-place compaction
    assert resolve_prompt_cache_scope(fork) == "gwk_parentscope::review"
    fork.provider, fork.model = "anthropic", "claude-opus-4-8"
    assert resolve_prompt_cache_scope(fork) == "gwk_parentscope"


def test_xai_fork_sends_distinct_conv_id_and_cache_key():
    from agent.transports.codex import ResponsesApiTransport

    def build(agent):
        return ResponsesApiTransport().build_kwargs(
            model="grok-4.3", messages=SYS, tools=[], session_id=agent.session_id,
            cache_scope_id=resolve_prompt_cache_scope(agent), is_xai_responses=True,
        )

    parent = build(_agent("xai-oauth"))
    fork = build(_agent("xai-oauth", tag="review", compactions=1))
    assert parent["extra_headers"]["x-grok-conv-id"] == "parent-sess"
    assert fork["extra_headers"]["x-grok-conv-id"] == "parent-sess::review"
    assert parent["extra_body"]["prompt_cache_key"] != fork["extra_body"]["prompt_cache_key"]
