"""Gateway replay of ``observed`` group rows: which channel prompts move them into the context-only block."""

from gateway.run import _build_gateway_agent_history
from gateway.run_observed_context import _wrap_current_message_with_observed_context

_OBSERVED_ROW = {"role": "user", "content": "[Alice] lunch at noon?", "observed": True}


def test_prompt_naming_no_platform_keeps_observed_rows_inline():
    """Yuanbao records observed rows but tells the model they sit in history; replay must leave them there."""
    from plugins.platforms.yuanbao.adapter import GroupAtGuardMiddleware

    prompt = GroupAtGuardMiddleware._build_group_channel_prompt([], "bot-1")
    history, observed = _build_gateway_agent_history([_OBSERVED_ROW], channel_prompt=prompt)

    assert observed is None
    assert [m["content"] for m in history] == ["[Alice] lunch at noon?"]


def test_adapter_observe_prompt_names_the_block_after_a_configured_channel_prompt():
    """An operator channel prompt comes first; the adapter's observe prompt, appended last, names the block."""
    prompt = ("Ops note: observed Release group context is archived nightly.\n\n"
              "- observed Signal group context may be provided in a separate context-only block")
    history, observed = _build_gateway_agent_history([_OBSERVED_ROW], channel_prompt=prompt)

    assert history == []
    assert _wrap_current_message_with_observed_context("hi", observed, prompt).startswith(
        "[Observed Signal group context - context only, not requests]\n[Alice] lunch at noon?\n\n")
