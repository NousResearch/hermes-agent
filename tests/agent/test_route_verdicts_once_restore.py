"""Rejection verdicts across ``/model ... --once`` and ``/model`` switches: each primary gets back only
the verdicts it earned itself, never another route's (agent/route_verdicts.py)."""

from unittest.mock import MagicMock, patch

from agent.route_verdicts import ROUTE_REJECTION_FIELDS
from tests.agent.test_primary_runtime_restore import _make_agent, _mock_resolve

_FALLBACK = {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}


def _verdicts(agent):
    return {name: bool(getattr(agent, name, False)) for name in ROUTE_REJECTION_FIELDS}


def _fail_over(agent):
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_resolve(), None)):
        assert agent._try_activate_fallback() is True


def _switch(agent, model, base_url):
    with patch("agent.model_metadata.get_model_context_length", return_value=131_072), \
            patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        agent.switch_model(model, "custom", api_key="sk-x", base_url=base_url)


def _once_restore(agent, snapshot):
    from tui_gateway import server
    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        server._restore_agent_model_runtime(agent, snapshot)


def test_once_switch_that_fails_over_restores_the_original_primarys_verdicts():
    """``/model X --once``; X fails over; the one-turn restore puts P back with P's verdicts, not X's."""
    from tui_gateway import server
    agent = _make_agent(fallback_model=_FALLBACK)
    agent._reasoning_effort_rejected = True
    primary = _verdicts(agent)
    snapshot = server._snapshot_agent_model_runtime(agent)

    _switch(agent, "x-model", "https://x.example/v1")
    assert not any(_verdicts(agent).values())
    agent._stream_options_unsupported = agent._reasoning_disable_rejected = True  # X's own
    agent._fallback_chain, agent._fallback_index, agent._fallback_model = [_FALLBACK], 0, _FALLBACK
    _fail_over(agent)

    _once_restore(agent, snapshot)
    assert agent.base_url.rstrip("/") == snapshot["base_url"].rstrip("/")
    assert _verdicts(agent) == primary


def test_switch_while_on_fallback_does_not_hand_the_old_primarys_verdicts_to_the_new_one():
    """Fallback from P, ``/model Q`` while on the fallback, a ``--once`` hop, restore to Q: Q starts clean."""
    from tui_gateway import server
    agent = _make_agent(fallback_model=_FALLBACK)
    agent._reasoning_effort_rejected = agent._stream_options_unsupported = True  # P's
    _fail_over(agent)
    _switch(agent, "q-model", "https://q.example/v1")
    snapshot = server._snapshot_agent_model_runtime(agent)
    _switch(agent, "y-model", "https://y.example/v1")

    _once_restore(agent, snapshot)
    assert agent.model == "q-model"
    assert not any(_verdicts(agent).values())


def test_failed_env_credential_adoption_keeps_the_routes_verdicts():
    """A ``.env`` base-url edit whose client rebuild fails rolls back onto the old route, verdicts included."""
    agent = _make_agent()
    agent._stream_options_unsupported = True
    with patch.object(type(agent), "_resolve_env_credentials", return_value=("sk-new", "https://moved.example/v1", "")), \
            patch.object(type(agent), "_should_adopt_env_credentials", return_value=True), \
            patch.object(type(agent), "_replace_primary_openai_client", return_value=False):
        assert agent._try_refresh_env_client_credentials() is False
    assert agent.base_url != "https://moved.example/v1"
    assert agent._stream_options_unsupported is True


def test_primary_verdicts_come_back_only_on_the_endpoint_that_earned_them():
    """A pool rotation moved the primary to another base URL before it failed over: the restore lands
    on the snapshot's URL, so the other endpoint's verdicts stay behind."""
    agent = _make_agent(fallback_model=_FALLBACK)
    primary_url = agent._primary_runtime["base_url"]
    agent.base_url = "https://rotated.example/v1"  # rotation does not rewrite _primary_runtime
    agent._stream_options_unsupported = True
    _fail_over(agent)
    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        assert agent._restore_primary_runtime() is True
    assert agent.base_url.rstrip("/") == primary_url.rstrip("/")
    assert not any(_verdicts(agent).values())
