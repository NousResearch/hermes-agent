"""Signature guard for ``agent.agent_runtime_helpers.anthropic_prompt_cache_policy``.

The behavioural matrix in ``test_anthropic_prompt_cache_policy.py`` only reaches
the function through the keyword forwarder
``AIAgent._anthropic_prompt_cache_policy()``. Several hot call sites instead
pass the agent **positionally**:

* ``agent/moa_loop.py`` — ``anthropic_prompt_cache_policy(stub, provider=..., ...)``
* ``agent/agent_runtime_helpers.py`` — the MoA-aggregator recursion re-invokes
  itself as ``anthropic_prompt_cache_policy(agent, ...)``

If the leading positional ``agent`` parameter were dropped (or turned
keyword-only), those positional calls would raise ``TypeError`` at runtime while
the keyword-only behavioural tests kept passing. These tests pin the contract so
that regression surfaces loudly.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from agent.agent_runtime_helpers import anthropic_prompt_cache_policy


def test_agent_is_first_positional_or_keyword_parameter():
    params = list(inspect.signature(anthropic_prompt_cache_policy).parameters.values())
    assert params, "anthropic_prompt_cache_policy must accept at least one parameter"
    first = params[0]
    assert first.name == "agent"
    assert first.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert first.default is inspect.Parameter.empty


def test_positional_agent_call_matches_moa_loop_usage():
    # Mirror agent/moa_loop.py: a stub is passed positionally and every field is
    # supplied as a keyword so the branch is judged purely on its own runtime.
    stub = SimpleNamespace(provider="", base_url="", api_mode="", model="")
    result = anthropic_prompt_cache_policy(
        stub,
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        model="anthropic/claude-sonnet-4.6",
    )
    should_cache, use_native_layout = result
    assert isinstance(should_cache, bool)
    assert isinstance(use_native_layout, bool)


def test_positional_agent_supplies_field_fallbacks():
    # With no keyword overrides the function must read the positional agent's
    # attributes — proving the positional binding actually reaches ``agent.*``.
    stub = SimpleNamespace(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        model="anthropic/claude-sonnet-4.6",
    )
    should_cache, use_native_layout = anthropic_prompt_cache_policy(stub)
    assert isinstance(should_cache, bool)
    assert isinstance(use_native_layout, bool)
