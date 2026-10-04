"""Fast-mode params must be re-scoped to the FALLBACK route, not carried over.

Static ``/fast`` (``service_tier == "priority"``) pins the primary route's fast
param into ``agent.request_overrides`` at build time: ``{"speed": "fast"}`` on
native Anthropic, ``{"service_tier": "priority"}`` on OpenAI/xAI. Those are
route-specific. When the agent falls back to a provider that does not support
fast mode, a carried-over ``speed`` reaches ``Completions.create()`` as an
unknown kwarg and the turn dies with TypeError before any request is sent —
observed live as anthropic/claude-opus-5 -> nous/stepfun, which killed inbound
``message_agent`` delivery for the whole profile.

``_rescope_fallback_extra_body`` already does this for custom-provider
extra_body keys; fast params need the same treatment.
"""

from types import SimpleNamespace

from agent.chat_completion_helpers import _rescope_fallback_fast_mode


def _agent(**kw):
    base = dict(
        service_tier="priority",
        model="stepfun/step-3.7-flash:free",
        provider="nous",
        base_url="https://inference-api.nousresearch.com/v1/",
        api_mode="chat_completions",
        request_overrides={"speed": "fast"},
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_unsupported_fallback_route_drops_stale_fast_param():
    """anthropic(speed=fast) -> nous: the param the new route rejects is gone."""
    agent = _agent(request_overrides={"speed": "fast", "extra_body": {"keep": 1}})

    _rescope_fallback_fast_mode(agent)

    assert "speed" not in agent.request_overrides
    assert "service_tier" not in agent.request_overrides
    # unrelated overrides survive untouched
    assert agent.request_overrides["extra_body"] == {"keep": 1}


def test_fast_param_is_reshaped_for_the_new_route():
    """anthropic(speed) -> openai(service_tier): each route gets its OWN param."""
    agent = _agent(
        model="gpt-5.4", provider="openai", base_url="https://api.openai.com/v1",
        request_overrides={"speed": "fast"},
    )

    _rescope_fallback_fast_mode(agent)

    assert agent.request_overrides == {"service_tier": "priority"}


def test_non_fast_sessions_are_not_given_fast_params():
    """A session that never asked for /fast stays normal across a fallback."""
    agent = _agent(
        service_tier=None, model="gpt-5.4", provider="openai",
        base_url="https://api.openai.com/v1", request_overrides={"extra_body": {"keep": 1}},
    )

    _rescope_fallback_fast_mode(agent)

    assert agent.request_overrides == {"extra_body": {"keep": 1}}


def test_bounded_windows_are_left_to_the_per_request_layer():
    """auto/cold are layered per request by agent.fast_mode; don't pin them here."""
    agent = _agent(
        service_tier="auto", model="claude-opus-5", provider="anthropic",
        base_url="https://api.anthropic.com", request_overrides={},
    )

    _rescope_fallback_fast_mode(agent)

    assert agent.request_overrides == {}
