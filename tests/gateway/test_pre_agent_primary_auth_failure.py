"""The primary provider's failed login, found while resolving a gateway turn's route, belongs to that turn.

Drives the production path (``_resolve_session_agent_runtime`` bound to the runner, then
``TurnRunner.run_sync``), as ``test_pre_agent_fallback_notice`` does: another chat resolving its own
route on another thread must not erase or replace this turn's failure, and a later step that swaps
the turn onto a different provider (``/model --once``, an API caller's provider) drops it.
"""
import threading
from types import SimpleNamespace
from unittest.mock import patch

from gateway.session import Platform, SessionSource
from gateway.turn_context import TurnContext
from hermes_cli.auth import AuthError
from tests.gateway.test_pre_agent_fallback_notice import _RecordingAgent, _runner_with_real_runtime_resolution

_FALLBACK = {"provider": "anthropic", "model": "claude-sonnet-5", "api_key": "k", "base_url": "u"}


def _primary_auth_fails(**kw):
    if kw.get("requested") is None:  # the primary, resolved from config.yaml
        raise AuthError("No access token found for Nous Portal", provider="nous")
    return dict(_FALLBACK)


def _ctx(agent_cls=None):
    return TurnContext(
        source=SessionSource(platform=Platform.LOCAL, chat_id="c", user_id="u"),
        message="hi", history=[], session_id="sid", session_key="test-session-key", user_config={},
        AIAgent=agent_cls or _RecordingAgent, resolve_display_setting=lambda *_a: False, _run_still_current=lambda: True,
        _hooks_ref=SimpleNamespace(loaded_hooks=False),
    )


def _run(runner, ctx, *extra):
    from gateway.run_turn_runner import TurnRunner

    patches = [
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_primary_auth_fails),
        patch("hermes_cli.runtime_provider._get_model_config", return_value={"provider": "nous", "default": "hermes-5"}),
        patch("gateway.run._load_gateway_config",
              return_value={"fallback_providers": [{"provider": "anthropic", "model": "claude-sonnet-5"}]}),
        patch("gateway.run._resolve_gateway_model", return_value="hermes-5"),
        patch("gateway.run._get_channel_override", return_value=None),
        patch("gateway.run._credential_pool_for_provider", return_value=None),
        *extra,
    ]
    for p in patches:
        p.start()
    try:
        TurnRunner(runner, ctx).run_sync()
    finally:
        for p in reversed(patches):
            p.stop()
    return ctx.agent_holder[0]


def test_another_chats_resolution_on_another_thread_keeps_this_turns_primary_failure():
    runner = _runner_with_real_runtime_resolution()
    override = {"model": "gpt-5", "provider": "openrouter", "api_key": "k2", "base_url": "u2"}
    # The other chat has a /model override: its resolution takes the fast path, which clears any stash.
    runner._peek_session_state.side_effect = lambda *a, **k: (
        SimpleNamespace(conversation=SimpleNamespace(model_override=override))
        if threading.current_thread().name == "other-chat" else None)
    from gateway import run_turn_prepare

    real_channel = run_turn_prepare._resolve_channel_runtime

    def other_chat_resolves_meanwhile(*args, **kwargs):
        other = threading.Thread(target=runner._resolve_session_agent_runtime, name="other-chat",
                                 kwargs={"session_key": "other-key"})
        other.start()
        other.join(10)
        return real_channel(*args, **kwargs)

    agent = _run(runner, _ctx(), patch.object(run_turn_prepare, "_resolve_channel_runtime",
                                              side_effect=other_chat_resolves_meanwhile))
    failure = agent._primary_auth_failure
    assert failure is not None and failure.provider == "nous"


def test_a_route_replaced_after_resolution_drops_the_primary_failure():
    """``/model --once`` moves this turn to another provider: the backup the failure names is not the
    route that runs, so a failure of that route must not tell the user to sign in to Nous."""
    runner = _runner_with_real_runtime_resolution()

    def once_to_openrouter(_turn, _policy, model, runtime_kwargs):
        return "gpt-5", {**runtime_kwargs, "provider": "openrouter", "base_url": "https://openrouter.test/v1"}

    agent = _run(runner, _ctx(_AgentWithProvider),
                 patch("gateway.session_cli_turn.one_turn_route", side_effect=once_to_openrouter))
    assert agent.provider == "openrouter"
    assert agent._primary_auth_failure is None


def test_the_fallback_route_that_runs_keeps_the_primary_failure():
    agent = _run(_runner_with_real_runtime_resolution(), _ctx(_AgentWithProvider))
    assert agent.provider == "anthropic"
    assert agent._primary_auth_failure is not None and agent._primary_auth_failure.backup == "anthropic"


class _AgentWithProvider(_RecordingAgent):
    """The real AIAgent keeps the provider it was built with on ``.provider``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.provider = kwargs.get("provider")


class _AgentWithRoute(_RecordingAgent):
    """The real AIAgent keeps the provider and endpoint it was built with."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.provider = kwargs.get("provider")
        self.base_url = kwargs.get("base_url")


def test_a_route_swapped_to_another_endpoint_of_the_same_provider_drops_the_primary_failure():
    """``/model --once`` keeps the backup's provider slug (``anthropic``) but moves the turn to another
    endpoint: that endpoint is not the backup the failure names, so its failure is its own."""
    runner = _runner_with_real_runtime_resolution()

    def once_to_another_endpoint(_turn, _policy, model, runtime_kwargs):
        return model, {**runtime_kwargs, "base_url": "https://other-gateway.test/v1", "api_key": "k2"}

    agent = _run(runner, _ctx(_AgentWithRoute),
                 patch("gateway.session_cli_turn.one_turn_route", side_effect=once_to_another_endpoint))
    assert (agent.provider, agent.base_url) == ("anthropic", "https://other-gateway.test/v1")
    assert agent._primary_auth_failure is None


def test_the_backup_endpoint_that_runs_keeps_the_primary_failure():
    agent = _run(_runner_with_real_runtime_resolution(), _ctx(_AgentWithRoute))
    assert (agent.provider, agent.base_url) == ("anthropic", "u")
    assert agent._primary_auth_failure is not None
