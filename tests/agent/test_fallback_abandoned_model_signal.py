"""``agent._fallback_abandoned_model`` — the signal a successful fallback stamps with the model
it walked away from.

forge #142709 (gateway model-pin self-heal): a session pin pointing at a dead model makes every
turn fail with ``model_not_found`` while the fallback rescues delivery — but the pin itself is
never cleared, so the gateway rebuilds the same broken agent every turn, forever. The gateway
post-turn hook is the only layer that can clear the pin, and it may do that only when the agent
reports WHICH model was abandoned and WHY: only a ``model_not_found`` abandonment proves the pin
is the disease; a rate-limit fallback left a perfectly healthy model behind and must not touch it.

Contract (state.md ``## 契约规约``):
- write point: ``try_activate_fallback`` success branch only, overwritten on EVERY activation
  (``None`` when the activation reason is not ``FailoverReason.model_not_found``);
- value: the exact model string the agent served before the switch (``old_model``);
- consumers read defensively (``getattr(agent, "_fallback_abandoned_model", None)``).

Every test drives the real ``classify_api_error`` → ``try_activate_fallback`` path (the same
objects production hands each other at ``agent/turn_api_error.py``), never a source-shape check.
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_api_error import settle_unrecovered_error
from agent.turn_retry_state import TurnRetryState
from run_agent import AIAgent


class _Err(Exception):
    """API-error shape the classifier reads: status_code + message + response headers."""

    def __init__(self, status_code, message, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.response = SimpleNamespace(headers=headers or {})
        self.body = None


_FALLBACKS = [
    {"provider": "zai", "model": "glm-5.2"},
    {"provider": "deepseek", "model": "deepseek-v4-flash"},
]
_CLIENTS = [
    ("https://api.z.ai/v1", "glm-5.2"),
    ("https://api.deepseek.com/v1", "deepseek-v4-flash"),
]


def _make_agent(*, primary="ghost-model-x", fallbacks=None):
    """Real AIAgent whose current model is the session pin's model (default: a poisoned one)."""
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallbacks if fallbacks is not None else _FALLBACKS[:1],
        )
    agent.model = primary
    agent.provider = "custom"
    return agent


def _bound_clients():
    return [_mock_client(base_url=url) for url, _ in _CLIENTS]


def _mock_client(base_url):
    mock = SimpleNamespace(base_url=base_url, api_key="fb-key")
    return mock


def _resolve_patch(count=1):
    """Patch client resolution (patch where production reads: route_binding imports lazily)."""
    clients = _bound_clients()
    return patch(
        "agent.auxiliary_client.resolve_provider_client",
        side_effect=[(clients[i], _CLIENTS[i][1]) for i in range(count)],
    )


def test_model_not_found_activation_records_abandoned_model():
    """A fallback triggered by model_not_found stamps the model it abandoned — the self-heal signal."""
    agent = _make_agent(primary="ghost-model-x")
    err = _Err(404, "The model 'ghost-model-x' does not exist")
    classified = classify_api_error(err, provider=agent.provider)
    assert classified.reason == FailoverReason.model_not_found  # the verdict this signal keys on
    with _resolve_patch():
        assert agent._try_activate_fallback(reason=classified.reason) is True
    assert agent._fallback_abandoned_model == "ghost-model-x"


def test_unrelated_reason_activation_records_none():
    """A non-model_not_found fallback abandoned a healthy model — the signal must read None."""
    agent = _make_agent(primary="gpt-5.6-sol")
    err = _Err(429, "Rate limit exceeded", headers={"Retry-After": "5"})
    classified = classify_api_error(err, provider=agent.provider)
    assert classified.reason == FailoverReason.rate_limit
    with _resolve_patch():
        assert agent._try_activate_fallback(reason=classified.reason) is True
    assert agent._fallback_abandoned_model is None


def test_every_activation_overwrites_the_signal():
    """Per-activation state, never sticky: the latest activation wins, in both orders."""
    agent = _make_agent(primary="ghost-model-x", fallbacks=_FALLBACKS)
    with _resolve_patch(count=2):
        assert agent._try_activate_fallback(reason=FailoverReason.model_not_found) is True
        assert agent._fallback_abandoned_model == "ghost-model-x"
        assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is True
        assert agent._fallback_abandoned_model is None

    # Reverse order: a healthy-model fallback first, then the poisoned one — the poisoned
    # model is what the latest walk-away abandoned, so THAT is the signal.
    agent = _make_agent(primary="gpt-5.6-sol", fallbacks=_FALLBACKS)
    with _resolve_patch(count=2):
        assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is True
        assert agent._fallback_abandoned_model is None
        assert agent._try_activate_fallback(reason=FailoverReason.model_not_found) is True
        assert agent._fallback_abandoned_model == "glm-5.2"


def test_production_error_path_stamps_the_signal():
    """The real settle path (classify_api_error → settle_unrecovered_error → try_activate_fallback,
    the exact chain at agent/turn_api_error.py) delivers the classified reason to the signal."""
    agent = _make_agent(primary="ghost-model-x")
    err = _Err(404, "The model 'ghost-model-x' does not exist")
    classified = classify_api_error(err, provider=agent.provider)
    with _resolve_patch(), patch("agent.conversation_loop._arm_fallback_restart", lambda a, m, s, r: s):
        verdict = settle_unrecovered_error(
            agent, api_error=err, classified=classified, _retry=TurnRetryState(), status_code=404,
            error_msg=str(err).lower(), is_context_length_error=False, is_rate_limited=False,
            _is_zai_coding_overload=False, _provider=agent.provider, _base="http://x", _model=agent.model,
            messages=[], api_messages=[], api_kwargs=None, active_system_prompt=None,
            conversation_history=None, approx_tokens=0, retry_count=3, max_retries=3,
            compression_attempts=0, api_call_count=1,
        )
    assert verdict.action == "break"  # fallback took over the turn
    assert agent._fallback_abandoned_model == "ghost-model-x"
