from types import SimpleNamespace

from agent.chat_completion_helpers import build_api_kwargs
from agent.transports import get_transport
import agent.transports.codex  # noqa: F401 - register transport


def _make_codex_agent():
    agent = SimpleNamespace()
    agent.api_mode = "codex_responses"
    agent.model = "gpt-5.5"
    agent.tools = []
    agent.reasoning_config = {}
    agent.session_id = "rotated-session"
    agent.provider_cache_session_id = "original-cache-scope"
    agent.max_tokens = None
    agent.request_overrides = {}
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.provider = "openai-codex"
    agent._base_url_hostname = "chatgpt.com"
    agent._base_url_lower = "https://chatgpt.com/backend-api/codex"
    agent._get_transport = lambda: get_transport("codex_responses")
    agent._prepare_messages_for_non_vision_model = lambda messages: messages
    agent._resolved_api_call_timeout = lambda: 120
    agent._github_models_reasoning_extra_body = lambda: None
    return agent


def test_codex_build_api_kwargs_uses_stable_provider_cache_scope_after_session_rotation():
    """Compression rotates Hermes session_id for DB lineage, but Codex cache
    routing should keep the original provider-side scope for the continuing
    in-memory conversation so prompt-cache buckets survive compaction.
    """
    agent = _make_codex_agent()

    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

    assert kwargs["prompt_cache_key"].startswith("pck_")
    assert kwargs["extra_headers"]["session_id"] == "original-cache-scope"
    assert kwargs["extra_headers"]["x-client-request-id"] == "original-cache-scope"


def test_codex_build_api_kwargs_falls_back_to_session_id_for_older_agents():
    agent = _make_codex_agent()
    delattr(agent, "provider_cache_session_id")

    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

    assert kwargs["prompt_cache_key"].startswith("pck_")
    assert kwargs["extra_headers"]["session_id"] == "rotated-session"
    assert kwargs["extra_headers"]["x-client-request-id"] == "rotated-session"


def test_codex_cache_scope_does_not_replace_logical_provider_session_id():
    agent = _make_codex_agent()

    class CapturingTransport:
        def build_kwargs(self, **kwargs):
            return kwargs

    agent.api_mode = "chat_completions"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent._base_url_lower = agent.base_url
    agent._base_url_hostname = "openrouter.ai"
    agent._get_transport = lambda: CapturingTransport()
    agent._is_qwen_portal = lambda: False
    agent._is_openrouter_url = lambda: True
    agent._max_tokens_param = lambda *_args, **_kwargs: "max_tokens"
    agent._ollama_num_ctx = None
    agent._supports_reasoning_extra_body = lambda: False
    agent.providers_allowed = []
    agent.providers_ignored = []
    agent.providers_order = []
    agent.provider_sort = None
    agent.provider_require_parameters = False
    agent.provider_data_collection = None
    agent.openrouter_min_coding_score = None

    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

    assert kwargs["session_id"] == "rotated-session"
