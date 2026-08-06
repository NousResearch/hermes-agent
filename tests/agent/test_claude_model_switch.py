"""Contract for switching models on a live Claude Agent SDK session.

The SDK owns the conversation context, the message UUIDs, and the upstream
prompt cache. Hermes cannot reconstruct any of them, so an in-session model
switch has to go through the SDK's control plane rather than through a
teardown-and-reconnect — a rebuild silently costs the user a cold cache and a
lost context, and neither failure announces itself.

Also pinned: an unknown model is rejected here, with the allowed set named,
instead of being handed to the CLI's ``--model`` where it fails inside a spawned
subprocess with nothing the user can act on.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.agent_runtime_helpers import switch_claude_agent_sdk_model, switch_model
from hermes_cli.models import (
    claude_subscription_models,
    is_valid_claude_subscription_model,
    validate_requested_model,
)


def _make_session(*, accepted=True, closed=False):
    session = MagicMock()
    session.closed = closed
    session.set_model.return_value = accepted
    return session


def _make_claude_agent(session=None):
    """A duck-typed agent carrying the fields ``switch_model`` touches."""
    agent = MagicMock()
    agent.model = "claude-sonnet-5"
    agent.provider = "claude-code"
    agent.requested_provider = "claude-code"
    agent.api_mode = "claude_agent_sdk"
    agent.base_url = "claude-sdk://subscription"
    agent.api_key = ""
    agent.client = None
    agent._client_kwargs = {}
    agent._session_db = None
    agent._claude_session = session
    agent._anthropic_prompt_cache_policy.return_value = (True, False)
    agent._effective_lmstudio_context_length.return_value = 200_000

    def _release():
        live = agent._claude_session
        agent._claude_session = None
        if live is not None:
            live.close()

    agent._release_claude_agent_sdk_session.side_effect = _release
    return agent


# ---------------------------------------------------------------------------
# The control-plane switch
# ---------------------------------------------------------------------------


class TestInPlaceSwitch:
    def test_uses_set_model_and_keeps_the_session(self):
        session = _make_session(accepted=True)
        agent = _make_claude_agent(session)

        outcome = switch_claude_agent_sdk_model(agent, "claude-opus-5")

        assert outcome == "set_model"
        session.set_model.assert_called_once_with("claude-opus-5")
        # The whole point: no teardown, so Claude keeps context + prompt cache.
        session.close.assert_not_called()
        assert agent._claude_session is session

    def test_a_refused_switch_retires_rather_than_lying(self):
        """A session still pinned to the old model would answer as it silently."""
        session = _make_session(accepted=False)
        agent = _make_claude_agent(session)

        outcome = switch_claude_agent_sdk_model(agent, "claude-opus-5")

        assert outcome == "retired"
        session.close.assert_called_once()
        assert agent._claude_session is None

    def test_a_raising_control_plane_also_retires(self):
        session = _make_session()
        session.set_model.side_effect = RuntimeError("transport wedged")
        agent = _make_claude_agent(session)

        assert switch_claude_agent_sdk_model(agent, "claude-opus-5") == "retired"
        assert agent._claude_session is None

    def test_no_live_session_is_a_no_op(self):
        agent = _make_claude_agent(None)
        assert switch_claude_agent_sdk_model(agent, "claude-opus-5") == "none"
        agent._release_claude_agent_sdk_session.assert_not_called()

    def test_closed_session_is_a_no_op(self):
        session = _make_session(closed=True)
        agent = _make_claude_agent(session)
        assert switch_claude_agent_sdk_model(agent, "claude-opus-5") == "none"
        session.set_model.assert_not_called()


class TestSwitchModelIntegration:
    def test_switch_model_retargets_instead_of_rebuilding(self):
        session = _make_session(accepted=True)
        agent = _make_claude_agent(session)

        switch_model(
            agent,
            "claude-opus-5",
            "claude-code",
            api_key="",
            base_url="claude-sdk://subscription",
            api_mode="claude_agent_sdk",
        )

        assert agent.model == "claude-opus-5"
        session.set_model.assert_called_once_with("claude-opus-5")
        session.close.assert_not_called()
        # No HTTP client is built: `claude-sdk://subscription` is not an endpoint.
        agent._create_openai_client.assert_not_called()
        assert agent.client is None
        assert agent.api_key == ""

    def test_leaving_the_runtime_releases_the_session(self):
        session = _make_session()
        agent = _make_claude_agent(session)

        switch_model(
            agent,
            "gpt-5.5",
            "openai-api",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            api_mode="chat_completions",
        )

        agent._release_claude_agent_sdk_session.assert_called_once()
        session.close.assert_called_once()
        # And the model switch itself still happened.
        assert agent.model == "gpt-5.5"
        assert agent.provider == "openai-api"

    def test_a_non_claude_agent_is_untouched_by_the_release_hook(self):
        agent = _make_claude_agent(None)
        agent.api_mode = "chat_completions"
        agent.provider = "openai-api"
        agent.base_url = "https://api.openai.com/v1"

        switch_model(
            agent,
            "gpt-5.5",
            "openai-api",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            api_mode="chat_completions",
        )

        agent._release_claude_agent_sdk_session.assert_not_called()


# ---------------------------------------------------------------------------
# The curated model set
# ---------------------------------------------------------------------------


@pytest.fixture
def subscription_gate_open(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.claude_code.subscription_enabled", lambda config=None: True
    )


class TestCuratedModelSet:
    def test_derived_from_the_catalog_this_module_already_keeps(self):
        """Not a second hand-maintained list: every entry traces to a source."""
        from hermes_cli.models import (
            CLAUDE_SUBSCRIPTION_MODEL_ALIASES,
            OPENROUTER_MODELS,
            _PROVIDER_MODELS,
        )

        models = claude_subscription_models()
        assert models, "the curated set must not be empty"

        known = {m.lower() for m in _PROVIDER_MODELS["anthropic"]}
        known |= {
            mid.split("/", 1)[1].replace(".", "-").lower()
            for mid, _ in OPENROUTER_MODELS
            if mid.startswith("anthropic/")
        }
        known |= {a.lower() for a in CLAUDE_SUBSCRIPTION_MODEL_ALIASES}
        for model in models:
            assert model.lower() in known

    def test_registered_for_the_picker(self):
        from hermes_cli.models import _PROVIDER_MODELS

        assert _PROVIDER_MODELS["claude-code"] == claude_subscription_models()

    def test_cli_short_aliases_are_accepted(self):
        for alias in ("sonnet", "opus", "haiku", "default"):
            assert is_valid_claude_subscription_model(alias)

    def test_matching_is_case_insensitive(self):
        first = claude_subscription_models()[0]
        assert is_valid_claude_subscription_model(first.upper())

    def test_unknown_model_is_rejected_with_the_allowed_set(
        self, subscription_gate_open
    ):
        result = validate_requested_model("claude-imaginary-9", "claude-code")
        assert result["accepted"] is False
        assert result["persist"] is False
        for model in claude_subscription_models():
            assert model in result["message"]

    def test_known_model_is_accepted_without_a_network_probe(
        self, subscription_gate_open, monkeypatch
    ):
        def _no_probe(*args, **kwargs):  # pragma: no cover - guard
            raise AssertionError("claude-sdk:// is not a reachable endpoint")

        monkeypatch.setattr(
            "hermes_cli.models._urlopen_model_catalog_request", _no_probe
        )
        result = validate_requested_model(
            claude_subscription_models()[0], "claude-code"
        )
        assert result["accepted"] is True
        assert result["recognized"] is True

    def test_gate_closed_keeps_the_legacy_anthropic_behaviour(self, monkeypatch):
        """While the gate is shut, `claude-code` still means `anthropic`."""
        monkeypatch.setattr(
            "hermes_cli.claude_code.subscription_enabled", lambda config=None: False
        )
        from hermes_cli.models import is_claude_subscription_slug

        assert is_claude_subscription_slug("claude-code") is False
