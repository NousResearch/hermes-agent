from types import SimpleNamespace

from agent.chat_completion_helpers import build_api_kwargs
from agent.vertex_adapter import (
    build_vertex_base_url,
    is_vertex_openapi_base_url,
    normalize_vertex_model_for_request,
)


class _FakeChatTransport:
    def build_kwargs(self, **kwargs):
        return kwargs


def _make_chat_agent(*, provider="vertex", model="gemini-3.1-flash-lite", base_url=None):
    if base_url is None:
        base_url = build_vertex_base_url("test-project", "global")

    agent = SimpleNamespace()
    agent.api_mode = "chat_completions"
    agent.provider = provider
    agent.model = model
    agent.base_url = base_url
    agent._base_url_lower = base_url.lower()
    agent._base_url_hostname = "aiplatform.googleapis.com"
    agent.tools = []
    agent.max_tokens = None
    agent.reasoning_config = None
    agent.request_overrides = {}
    agent.session_id = "test-session"
    agent.providers_allowed = []
    agent.providers_ignored = []
    agent.providers_order = []
    agent.provider_sort = None
    agent.provider_require_parameters = False
    agent.provider_data_collection = None
    agent.openrouter_min_coding_score = None
    agent._ephemeral_max_output_tokens = None
    agent._ollama_num_ctx = None
    agent._get_transport = lambda: _FakeChatTransport()
    agent._is_qwen_portal = lambda: False
    agent._is_openrouter_url = lambda: False
    agent._resolved_api_call_timeout = lambda: 30
    agent._max_tokens_param = lambda n: {"max_tokens": n}
    agent._supports_reasoning_extra_body = lambda: False
    agent._github_models_reasoning_extra_body = lambda: None
    agent._prepare_messages_for_non_vision_model = lambda messages: messages
    agent._qwen_prepare_chat_messages = lambda messages: messages
    agent._qwen_prepare_chat_messages_inplace = lambda messages: messages
    agent._lmstudio_reasoning_options_cached = lambda: None
    return agent


def test_vertex_openapi_base_url_detection_global_and_regional():
    assert is_vertex_openapi_base_url(
        "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/endpoints/openapi"
    )
    assert is_vertex_openapi_base_url(
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/p/locations/us-central1/endpoints/openapi/"
    )


def test_vertex_openapi_base_url_detection_rejects_third_party_url():
    assert not is_vertex_openapi_base_url("https://litellm.example.com/v1")
    assert not is_vertex_openapi_base_url("https://example-aiplatform.googleapis.com.evil.test/v1/endpoints/openapi")


def test_normalize_vertex_model_prefixes_bare_gemini_for_official_vertex():
    base_url = build_vertex_base_url("test-project", "global")
    assert (
        normalize_vertex_model_for_request("gemini-3.1-flash-lite", base_url)
        == "google/gemini-3.1-flash-lite"
    )


def test_normalize_vertex_model_does_not_double_prefix_or_touch_non_gemini():
    base_url = build_vertex_base_url("test-project", "global")
    assert (
        normalize_vertex_model_for_request("google/gemini-3.1-flash-lite", base_url)
        == "google/gemini-3.1-flash-lite"
    )
    assert (
        normalize_vertex_model_for_request("anthropic/claude-sonnet-4", base_url)
        == "anthropic/claude-sonnet-4"
    )


def test_normalize_vertex_model_does_not_touch_third_party_openai_compatible_url():
    assert (
        normalize_vertex_model_for_request("gemini-3.1-flash-lite", "https://litellm.example.com/v1")
        == "gemini-3.1-flash-lite"
    )


def test_build_api_kwargs_prefixes_native_vertex_request_model(monkeypatch):
    monkeypatch.setattr("providers.get_provider_profile", lambda provider: None)
    agent = _make_chat_agent()

    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

    assert kwargs["model"] == "google/gemini-3.1-flash-lite"
    assert agent.model == "gemini-3.1-flash-lite"


def test_build_api_kwargs_does_not_prefix_vertex_relay_url(monkeypatch):
    from hermes_cli.model_normalize import normalize_model_for_provider

    monkeypatch.setattr("providers.get_provider_profile", lambda provider: None)
    configured_model = normalize_model_for_provider("gemini-3.1-flash-lite", "vertex")
    agent = _make_chat_agent(
        provider="vertex",
        model=configured_model,
        base_url="https://litellm.example.com/v1",
    )

    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

    assert configured_model == "gemini-3.1-flash-lite"
    assert kwargs["model"] == "gemini-3.1-flash-lite"


def test_provider_alias_is_repaired_before_official_vertex_request(monkeypatch):
    from hermes_cli.model_normalize import normalize_model_for_provider

    monkeypatch.setattr("providers.get_provider_profile", lambda provider: None)
    configured_model = normalize_model_for_provider(
        "google-vertex/gemini-3.1-flash-lite", "vertex"
    )
    agent = _make_chat_agent(provider="vertex", model=configured_model)

    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

    assert configured_model == "gemini-3.1-flash-lite"
    assert kwargs["model"] == "google/gemini-3.1-flash-lite"
