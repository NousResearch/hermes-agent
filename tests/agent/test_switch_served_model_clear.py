"""switch_model() must not carry ``last_served_model`` across a route switch.

``agent.last_served_model`` has exactly one writer repo-wide: the httpx response hook
installed by ``install_served_model_capture()``, and that hook is installed in exactly
one place — ``create_openai_client()``. So ONLY OpenAI-wire destinations ever refresh it.

``switch_model()`` used to leave the field untouched, and ``_build_switched_client()``
returns early on the moa / bedrock / ``anthropic_messages`` branches — none of which
installs a hook. A mid-chat ``/model`` switch reuses the live agent (``_init_agent``
returns early when ``self.agent is not None``), so the PREVIOUS route's served model
survives indefinitely, and ``result_model_fields()`` then reads it as this turn's fact:

    agent.model = "claude-sonnet-5"  (native Anthropic)
    agent.last_served_model = "poolside/laguna-s-2.1:free"  (left by an earlier combo route)
    -> "/usage" renders 'claude-sonnet-5 ->poolside/laguna-s-2.1:free (anthropic)'

That is the exact misattribution the PR sets out to fix, now inverted. The truthy value
also skips the ``_fallback_activated`` fallback branch in ``result_model_fields()``,
masking a real fallback.

Two further constraints the fix must satisfy:

* The clear must be rolled back when a rebuild fails — the old value still describes the
  still-live old client, so ``_SWITCH_SNAPSHOT_FIELDS`` has to carry the field.
* It must be unconditional (not only on the non-OpenAI branches): an OpenAI-wire
  destination that has not yet served a response must also start from None, otherwise the
  OLD route's model is reported until the first new response lands.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.agent_runtime_helpers import switch_model
from agent.served_model import result_model_fields


def _make_agent(current_provider, current_model, current_pool=None):
    """Bare agent object with the minimum attributes switch_model touches.

    Mirrors tests/agent/test_switch_model_pool_reload.py, which is a proven-good fixture
    for driving the real switch_model() end to end.
    """
    agent = MagicMock(name=f"Agent[{current_provider}]")
    agent.provider = current_provider
    agent.model = current_model
    agent.base_url = f"https://{current_provider}.example/v1"
    agent.api_key = f"{current_provider}-key"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock(name="Client")
    agent._client_kwargs = {"api_key": "k", "base_url": f"https://{current_provider}.example/v1"}
    agent._anthropic_client = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._is_anthropic_oauth = False
    agent._config_context_length = None
    agent._transport_cache = {}
    agent._cached_system_prompt = "cached-system-prompt"
    agent.context_compressor = None
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._credential_pool = current_pool
    agent._credential_pool_entry_id = None
    # The stale value under test: left behind by an earlier combo (OpenAI-wire) route.
    agent.last_served_model = "poolside/laguna-s-2.1:free"
    agent._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
    agent._ensure_lmstudio_runtime_loaded = MagicMock()
    return agent


def _switch(agent, **overrides):
    """Drive switch_model() with load_pool stubbed, as the pool-reload suite does."""
    kwargs = dict(
        new_model="claude-sonnet-5",
        new_provider="anthropic",
        api_key="anthropic-key",
        base_url="https://api.anthropic.com/v1",
        api_mode="chat_completions",
    )
    kwargs.update(overrides)
    with patch("agent.credential_pool.load_pool", return_value=MagicMock(name="Pool")):
        switch_model(agent, **kwargs)


class TestServedModelClearedOnSwitch:
    """A route switch must start from "not yet served", never from the previous route."""

    def test_switch_to_non_openai_wire_destination_clears_it(self):
        """The reviewer's repro: native Anthropic keeps the combo route's stale member.

        ``anthropic_messages`` is one of the branches that return early from
        ``_build_switched_client()`` without installing the capture hook, so nothing
        would ever refresh the value again.
        """
        agent = _make_agent("opencode-go", "some-combo-alias")
        assert agent.last_served_model is not None  # precondition: stale value present

        _switch(agent, api_mode="anthropic_messages")

        assert agent.model == "claude-sonnet-5"
        assert agent.last_served_model is None, (
            "switch_model must clear last_served_model: only create_openai_client() "
            "installs the capture hook, so a non-OpenAI destination would otherwise "
            "report the PREVIOUS route's model as this turn's"
        )

    def test_result_model_fields_no_longer_reports_the_previous_route(self):
        """End-to-end: /usage must not render a member that never served this turn."""
        agent = _make_agent("opencode-go", "some-combo-alias")
        agent._fallback_activated = False

        _switch(agent, api_mode="anthropic_messages")

        fields = result_model_fields(agent)
        assert fields["requested_model"] == "claude-sonnet-5"
        assert fields["served_model"] in (None, ""), (
            f"result_model_fields reported a stale served model: {fields['served_model']!r}"
        )
        rendered = f"{fields['requested_model']} ->{fields['served_model']} ({agent.provider})"
        assert "poolside" not in rendered

    def test_switch_to_openai_wire_destination_also_clears_it(self):
        """Unconditional clear: an OpenAI-wire destination that has not served yet must
        start from None too, or the OLD route's model shows until the first new response."""
        agent = _make_agent("opencode-go", "some-combo-alias")

        _switch(
            agent,
            new_model="llama-3.3-70b",
            new_provider="groq",
            api_key="groq-key",
            base_url="https://api.groq.com/openai/v1",
        )

        assert agent.last_served_model is None

    def test_failed_switch_restores_the_previous_value(self):
        """Rollback must keep the OLD value — it still describes the still-live old client.

        ``_SWITCH_SNAPSHOT_FIELDS`` now carries ``last_served_model``; without it the
        snapshot would silently drop the attribute and a failed rebuild would leave the
        field cleared while ``agent.model`` had already rolled back to the old model.
        """
        agent = _make_agent("opencode-go", "some-combo-alias")

        # Fail inside _swap_switch_runtime AFTER the clear has run (it is the last step).
        with patch(
            "agent.agent_runtime_helpers._build_switched_client",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                _switch(agent, api_mode="anthropic_messages")

        assert agent.model == "some-combo-alias"  # identity rolled back ...
        assert agent.last_served_model == "poolside/laguna-s-2.1:free", (
            "a failed switch must restore last_served_model along with the identity, "
            "otherwise the cleared field describes a client that never got rebuilt"
        )

    def test_field_is_in_the_switch_snapshot(self):
        """Guard: the field must stay in _SWITCH_SNAPSHOT_FIELDS or the test above is the
        only thing keeping the rollback honest."""
        from agent.agent_runtime_helpers import _SWITCH_SNAPSHOT_FIELDS

        assert "last_served_model" in _SWITCH_SNAPSHOT_FIELDS
