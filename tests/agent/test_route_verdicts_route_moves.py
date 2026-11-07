"""Every path that moves ``base_url`` to another endpoint starts that endpoint without the old one's
rejection verdicts; a refresh that keeps the endpoint keeps them (agent/route_verdicts.py)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.route_verdicts import ROUTE_REJECTION_FIELDS
from tests.agent.test_primary_runtime_restore import _make_agent, _make_transport_error

_PAID = "https://inference-api.nousresearch.com/v1"
_WELCOME = "https://welcome-api.nousresearch.com/v1"


def _verdicts(agent):
    return {name: bool(getattr(agent, name, False)) for name in ROUTE_REJECTION_FIELDS}


def _tripped(agent):
    for name in ROUTE_REJECTION_FIELDS:
        setattr(agent, name, True)
    return agent


@pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages"])
@pytest.mark.parametrize(("new_url", "kept"), [(_WELCOME, False), (_PAID + "/", True)])
def test_nous_refresh_onto_another_host_drops_the_old_hosts_verdicts(api_mode, new_url, kept):
    """The welcome-tier heal moves the session between the welcome and paid hosts: a verdict one
    host earned must not follow it. A refreshed key on the same host keeps them."""
    agent = _tripped(_make_agent(provider="nous", base_url=_PAID))
    agent.api_mode = api_mode
    agent._rebuild_anthropic_client = MagicMock()
    with patch("hermes_cli.auth.resolve_nous_runtime_credentials",
               return_value={"api_key": "fresh-agent-key", "base_url": new_url}), \
            patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        assert agent._try_refresh_nous_client_credentials(force=True) is True
    assert agent.base_url == new_url.rstrip("/")
    assert _verdicts(agent) == dict.fromkeys(ROUTE_REJECTION_FIELDS, kept)


def test_provider_refresh_hook_returning_another_base_url_drops_the_old_verdicts():
    agent = _tripped(_make_agent())
    profile = SimpleNamespace(name="p", refresh_runtime_credentials=lambda **_kw: ("tok", "https://moved.example/v1"))
    with patch("agent.conversation_loop._credential_refresh_profile", return_value=profile), \
            patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        assert agent._try_refresh_provider_runtime_credentials(reason="unauthorized") is True
    assert agent.base_url == "https://moved.example/v1"
    assert not any(_verdicts(agent).values())


def test_transport_recovery_back_to_the_snapshot_url_drops_the_rotated_urls_verdicts():
    """A pool rotation moved the primary to U2 and U2 learned a verdict; transport recovery puts
    the agent back on the snapshot URL, which never earned it."""
    agent = _make_agent()
    snapshot_url = agent._primary_runtime["base_url"]
    agent.base_url = "https://rotated.example/v1"
    agent._stream_options_unsupported = True
    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()), patch("time.sleep"):
        assert agent._try_recover_primary_transport(_make_transport_error(), retry_count=3, max_retries=3) is True
    assert agent.base_url == snapshot_url
    assert agent._stream_options_unsupported is False


def test_transport_recovery_on_the_same_url_keeps_the_verdicts():
    agent = _tripped(_make_agent())
    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()), patch("time.sleep"):
        assert agent._try_recover_primary_transport(_make_transport_error(), retry_count=3, max_retries=3) is True
    assert all(_verdicts(agent).values())


def _fail_over(agent):
    from tests.agent.test_primary_runtime_restore import _mock_resolve
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_resolve(), None)):
        assert agent._try_activate_fallback() is True


_FALLBACK = {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}


def test_a_route_that_refused_streaming_does_not_turn_streaming_off_for_the_next_route():
    """``_disable_streaming`` learned from one provider's "streaming not supported" stays with that
    route: the fallback streams again, and the primary gets its own verdict back on restore."""
    agent = _make_agent(fallback_model=_FALLBACK)
    assert agent._disable_streaming is False
    agent._disable_streaming = True  # the primary refused a stream
    _fail_over(agent)
    assert agent._disable_streaming is False
    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        assert agent._restore_primary_runtime() is True
    assert agent._disable_streaming is True


def test_a_route_change_restores_the_configured_streaming_seed_not_streaming_on():
    """``model.streaming: false`` is configuration, not a verdict: a route change keeps it."""
    with patch("hermes_cli.config.load_config_readonly", return_value={"model": {"streaming": False}}):
        agent = _make_agent(fallback_model=_FALLBACK)
    assert agent._disable_streaming is True
    _fail_over(agent)
    assert agent._disable_streaming is True
    with patch("agent.model_metadata.get_model_context_length", return_value=131_072), \
            patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        agent.switch_model("q-model", "custom", api_key="sk-x", base_url="https://q.example/v1")
    assert agent._disable_streaming is True


def _rotation_agent():
    """A wire-level agent whose 429 recovery rotates the pool through the real ``_swap_credential``."""
    from types import MethodType

    from agent.client_lifecycle import ClientLifecycleMixin
    from tests.agent.test_reasoning_disable_recovery import _WireAgent

    agent = _WireAgent({"enabled": True, "effort": "max"})
    agent.api_key = "key-a"
    agent._client_kwargs = {"api_key": "key-a", "base_url": agent.base_url}
    agent._disable_streaming = agent._disable_streaming_seed = False
    agent._reapply_route_client_config = MethodType(ClientLifecycleMixin._reapply_route_client_config, agent)
    agent._apply_client_headers_for_base_url = MagicMock()
    agent._replace_primary_openai_client = MagicMock(return_value=True)
    agent._swap_credential = MethodType(ClientLifecycleMixin._swap_credential, agent)
    return agent


def _recover_from(agent, retry, status_code, message):
    from agent.error_classifier import classify_api_error
    from agent.turn_recovery import recover_after_classification
    from tests.agent.test_reasoning_disable_recovery import _FakeApiError

    err = _FakeApiError(status_code, message)
    classified = classify_api_error(err, provider=agent.provider, model=agent.model)
    with patch("hermes_cli.models_reasoning_caps.refresh_reasoning_caps_async", lambda provider: None), \
            patch("hermes_cli.config.load_config_readonly", return_value={}):
        return recover_after_classification(
            agent, err, classified, retry, status_code=status_code, error_context={}, messages=[], api_messages=[],
        )


@pytest.mark.parametrize(("rotated_url", "moved"), [("https://second.example/v1", True), (None, False)])
def test_a_rotation_to_another_endpoint_gives_its_effort_rejection_a_recovery(rotated_url, moved):
    """A rejects ``effort: max``, the repaired retry hits a 429 and the pool rotates. On another
    endpoint the verdict resets and the configured effort goes out again: when that endpoint rejects
    it too, the same call repairs it again instead of taking the fallback chain. A rotation that keeps
    the endpoint keeps the verdict and the spent guard, so recovery stays one shot per route."""
    from agent.turn_retry_state import TurnRetryState
    from tests.agent.test_reasoning_disable_recovery import _STRUCTURED_LEVEL_400, _wire_reasoning_config

    agent = _rotation_agent()
    retry = TurnRetryState()
    assert _wire_reasoning_config(agent) == {"enabled": True, "effort": "max"}  # the request A rejects
    assert _recover_from(agent, retry, 400, _STRUCTURED_LEVEL_400)[0] is True
    assert _wire_reasoning_config(agent) is None

    entry = SimpleNamespace(runtime_api_key="key-b", runtime_base_url=rotated_url or agent.base_url, id="b")

    def rotate(**kwargs):
        assert kwargs["status_code"] == 429
        assert agent._swap_credential(entry) is True
        return True, True

    with patch.object(agent, "_recover_with_credential_pool", side_effect=rotate):
        assert _recover_from(agent, retry, 429, "rate limit exceeded") == (True, True)

    if moved:
        assert _wire_reasoning_config(agent) == {"enabled": True, "effort": "max"}  # the request B rejects
        assert _recover_from(agent, retry, 400, _STRUCTURED_LEVEL_400)[0] is True
        assert _wire_reasoning_config(agent) is None
    else:
        assert _wire_reasoning_config(agent) is None
        assert retry.reasoning_mandatory_retry_attempted is True
        assert _recover_from(agent, retry, 400, _STRUCTURED_LEVEL_400)[0] is False
    assert agent.activated == []
