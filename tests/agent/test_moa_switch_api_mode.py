"""Regression test for MoA primary-call routing on persisted preset switches.

Issue #54259 / #54669: switching a live agent to a MoA preset (the gateway
``/model <preset>`` path) built the MoAClient facade but left ``agent.api_mode``
set to whatever ``determine_api_mode`` / the resolved aggregator transport
produced (e.g. ``codex_responses`` or ``anthropic_messages``). The conversation
loop dispatches on ``agent.api_mode``, so a non-chat_completions value made it
call ``client.responses.create`` — which the MoAClient facade has no
``.responses`` for — and the call fell through to the ``moa://local``
placeholder, 404'd three times, then fell back to a reference model.

``agent_init.py`` already pins ``api_mode = "chat_completions"`` for
``provider == "moa"``; ``switch_model`` (the live in-place swap) must do the
same so the primary call always routes through ``MoAClient.chat.completions``.
"""

from __future__ import annotations

import types

import pytest


def _make_fake_agent():
    """A minimal stand-in carrying only the attributes switch_model touches."""
    agent = types.SimpleNamespace()
    agent.model = "minimax-m3"
    agent.provider = "opencode-go"
    agent.api_mode = "anthropic_messages"
    agent.api_key = "old-key"
    agent.base_url = "https://old.example/v1"
    agent.client = object()
    agent._client_kwargs = {"base_url": "https://old.example/v1"}
    agent._config_context_length = 123456
    agent._transport_cache = {}
    agent.quiet_mode = True
    return agent


@pytest.mark.parametrize(
    "incoming_api_mode",
    ["codex_responses", "anthropic_messages", "chat_completions", ""],
)
def test_switch_to_moa_pins_chat_completions(monkeypatch, incoming_api_mode):
    """Switching to provider=moa must force api_mode=chat_completions.

    No matter what transport the resolver/aggregator implies for the preset,
    the outer agent.api_mode must end up chat_completions so the conversation
    loop dispatches through the MoAClient chat.completions facade rather than
    .responses.create against the moa://local placeholder.
    """
    from agent import agent_runtime_helpers as arh

    # Neutralize the post-swap machinery that needs a real AIAgent (credential
    # pool reload, context-compressor refresh, primary-runtime bookkeeping).
    # We only assert the api_mode invariant set in the moa client-build branch.
    monkeypatch.setattr(arh, "load_pool", lambda *a, **k: None, raising=False)

    agent = _make_fake_agent()
    try:
        arh.switch_model(
            agent,
            new_model="frontier",
            new_provider="moa",
            api_key="moa-virtual-provider",
            base_url="moa://local",
            api_mode=incoming_api_mode,
        )
    except Exception:
        # switch_model does post-swap work (compressor, pool, runtime) that may
        # raise against a fake agent. The runtime-field swap — including the
        # api_mode pin in the moa branch — happens before any of that, so the
        # invariant we care about is already set even if a later step blew up.
        pass

    assert agent.provider == "moa"
    assert agent.base_url == "moa://local"
    assert agent.api_mode == "chat_completions", (
        f"MoA switch left api_mode={agent.api_mode!r}; the primary call would "
        "dispatch .responses.create / anthropic_messages against moa://local "
        "instead of MoAClient.chat.completions (issue #54259)."
    )
    # The MoAClient facade should be installed as the client.
    assert type(agent.client).__name__ == "MoAClient"
