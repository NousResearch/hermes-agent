"""Generic ProviderProfile wire seams, exercised through a fake plugin profile.

``api_mode_for_model`` (per-model wire), ``client_headers`` (endpoint client headers, re-added on
snapshot rebuilds), ``request_headers`` (first-call-of-a-user-turn attribution) and
``max_tokens_param``: a ``custom`` route pointed at the profile's host gets them too.
"""

from unittest.mock import MagicMock, patch

import pytest

HOST = "https://api.wirefake.test/v1"


@pytest.fixture
def wire_profile():
    import providers
    from providers.base import ProviderProfile
    from utils import base_url_host_matches

    seen_keys = []

    class WireFakeProfile(ProviderProfile):
        def api_mode_for_model(self, model, *, api_key=None):
            seen_keys.append(api_key)
            return "chat_completions" if model.startswith("gpt-5-lite") else None

        def client_headers(self, *, base_url, api_key="", has_images=False):
            if not base_url_host_matches(base_url, "wirefake.test"):
                return None
            return {"X-Wire": "1", **({"X-Wire-Images": "1"} if has_images else {})}

        def request_headers(self, *, base_url, model="", user_initiated=False):
            return {"X-Attr": "user"} if user_initiated and base_url_host_matches(base_url, "wirefake.test") else {}

        def max_tokens_param(self, *, base_url, model=""):
            return "max_completion_tokens" if base_url_host_matches(base_url, "wirefake.test") else None

    providers.list_providers()  # discovery first, so the fake survives it
    profile = WireFakeProfile(name="wirefake", base_url=HOST, hostnames=("wirefake.test",))
    providers.register_provider(profile)
    yield profile, seen_keys
    import hermes_cli.auth
    providers._REGISTRY.pop("wirefake", None)
    providers._SOURCES.pop("wirefake", None)
    providers._PROVIDER_LIST_CACHE = None
    hermes_cli.auth.PROVIDER_REGISTRY.pop("wirefake", None)


def _agent(provider, base_url, model="gpt-4.1"):
    from run_agent import AIAgent

    with patch("agent.process_bootstrap.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        return AIAgent(api_key="k", base_url=base_url, provider=provider, model=model, quiet_mode=True,
                       skip_context_files=True, skip_memory=True)


def test_api_mode_for_model_decides_responses_routing_per_model(wire_profile):
    from hermes_cli.model_switch import model_derived_api_mode
    from run_agent import AIAgent

    # The profile keeps this gpt-5 id on chat completions; the generic GPT-5 rule would upgrade it.
    assert AIAgent._provider_model_requires_responses_api("gpt-5-lite", provider="wirefake") is False
    # None defers to the generic rule.
    assert AIAgent._provider_model_requires_responses_api("gpt-5.4", provider="wirefake") is True
    # /model re-derives the wire from the profile, handing it the credential for catalog-aware ids.
    assert model_derived_api_mode("wirefake", "gpt-5-lite", "tok") == "chat_completions"
    _, seen_keys = wire_profile
    assert seen_keys[-1] == "tok"
    assert None in seen_keys  # the routing path asked the pure (no-credential) form


def test_custom_route_to_profile_host_gets_client_headers_and_max_tokens_param(wire_profile):
    from agent.agent_runtime_helpers import _ensure_route_client_headers

    agent = _agent("custom", HOST, model="llama-3.1-8b")
    assert agent._client_kwargs["default_headers"]["X-Wire"] == "1"
    assert agent._max_tokens_param(7) == {"max_completion_tokens": 7}

    # Snapshot rebuilds re-add missing endpoint headers but never override one already set.
    kwargs = {"base_url": HOST, "default_headers": {"x-wire": "user-set"}}
    _ensure_route_client_headers(kwargs, "custom")
    assert kwargs["default_headers"] == {"x-wire": "user-set"}
    kwargs = {"base_url": HOST}
    _ensure_route_client_headers(kwargs, "custom")
    assert kwargs["default_headers"] == {"X-Wire": "1"}


def test_other_endpoints_keep_generic_behaviour(wire_profile):
    from agent.agent_runtime_helpers import _ensure_route_client_headers

    agent = _agent("custom", "https://llm.example.test/v1", model="llama-3.1-8b")
    assert "X-Wire" not in (agent._client_kwargs.get("default_headers") or {})
    assert agent._max_tokens_param(7) == {"max_tokens": 7}
    kwargs = {"base_url": "https://llm.example.test/v1"}
    _ensure_route_client_headers(kwargs, "custom")
    assert "default_headers" not in kwargs


def test_image_request_client_carries_image_client_headers(wire_profile):
    agent = _agent("wirefake", HOST)
    built = []
    agent.client = object()
    image_request = {"model": "m", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "x"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]}]}
    with patch.object(agent, "_is_openai_client_closed", return_value=False), patch.object(
            agent, "_create_openai_client", side_effect=lambda kw, *, reason, shared: built.append(kw) or MagicMock()):
        agent._create_request_openai_client(reason="t", api_kwargs=image_request)
    assert built[-1]["default_headers"]["X-Wire-Images"] == "1"
    assert built[-1]["default_headers"]["X-Wire"] == "1"


def _payload(agent, api_call_count=1):
    from agent.turn_api_request import build_request_payload

    agent._empty_content_retries = 0
    build = build_request_payload(
        agent, api_messages=[{"role": "user", "content": "hi"}], _moa_prepared_request=None,
        tools_for_api=agent.tools, system_message="", api_call_count=api_call_count, api_request_id="r",
        effective_task_id="t", turn_id="u")
    return build.api_kwargs


def test_user_turn_attribution_is_consumed_by_the_first_call_only(wire_profile):
    agent = _agent("wirefake", HOST)
    agent._is_user_initiated_turn = True
    assert _payload(agent).get("extra_headers") == {"X-Attr": "user"}
    assert agent._is_user_initiated_turn is False
    assert "X-Attr" not in (_payload(agent, api_call_count=2).get("extra_headers") or {})  # tool-loop follow-up


def test_only_the_first_attempt_of_a_user_turn_carries_user_attribution(wire_profile):
    # A 429/5xx retry (or a fallback activation) rebuilds the payload for the same API call; like Hermes
    # always has, it goes out with the route's default attribution (billing parity), not the user's.
    agent = _agent("wirefake", HOST)
    agent._is_user_initiated_turn = True
    assert _payload(agent, api_call_count=1).get("extra_headers") == {"X-Attr": "user"}
    assert "X-Attr" not in (_payload(agent, api_call_count=1).get("extra_headers") or {})  # retried attempt
    assert "X-Attr" not in (_payload(agent, api_call_count=2).get("extra_headers") or {})


def test_route_without_attribution_still_consumes_the_turn_flag(wire_profile):
    agent = _agent("custom", "https://llm.example.test/v1")
    agent._is_user_initiated_turn = True
    assert "X-Attr" not in (_payload(agent).get("extra_headers") or {})
    assert agent._is_user_initiated_turn is False


def test_failing_request_headers_hook_never_bills_a_follow_up_as_the_user(wire_profile):
    profile, _ = wire_profile
    agent = _agent("wirefake", HOST)
    agent._is_user_initiated_turn = True
    with patch.object(type(profile), "request_headers", side_effect=RuntimeError("plugin bug")):
        assert "X-Attr" not in (_payload(agent, api_call_count=1).get("extra_headers") or {})
    assert agent._is_user_initiated_turn is False
    assert "X-Attr" not in (_payload(agent, api_call_count=2).get("extra_headers") or {})  # tool-loop follow-up
