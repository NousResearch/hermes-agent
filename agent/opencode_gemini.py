"""Native Gemini wire selection for the official OpenCode Go relay."""

from urllib.parse import urlsplit, urlunsplit


def go_gemini_base(provider, model, base_url):
    """Return the native Go base, leaving custom endpoint/transport pins alone."""
    from agent.opencode_affinity import opencode_transport
    from hermes_cli.models import normalize_opencode_model_id

    if provider != "opencode-go":
        return None
    url = urlsplit(str(base_url or ""))
    if url.scheme != "https" or url.hostname != "opencode.ai" or url.port not in (None, 443):
        return None
    if url.path.rstrip("/") not in {"/zen/go", "/zen/go/v1", "/inference/go", "/inference/go/v1"}:
        return None
    if not normalize_opencode_model_id(provider, model).lower().startswith("gemini-"):
        return None
    if opencode_transport(provider, model, base_url)[0] is None:
        return None
    return urlunsplit((url.scheme, url.netloc, "/inference/go/v1", url.query, ""))


def main_go_gemini_base(agent, base_url):
    provider = agent.provider
    if provider == "custom":
        provider = getattr(agent, "requested_provider", provider)
    return go_gemini_base(provider, getattr(agent, "model", ""), base_url)


def compressor_provider(agent):
    """Keep built-in Go provenance after the agent leaves its internal custom alias.

    Plugin engines retain the runtime provider and their existing update contract.
    During built-in construction there is no compressor yet.
    """
    from agent.context_compressor import ContextCompressor

    cc = getattr(agent, "context_compressor", None)
    if cc is None or isinstance(cc, ContextCompressor):
        if main_go_gemini_base(agent, agent.base_url):
            return "opencode-go"
    return agent.provider


def compressor_budget(agent):
    """Recompute the built-in output reservation without widening plugin ABI."""
    from agent.context_compressor import ContextCompressor
    from agent.agent_init import _compressor_max_tokens
    cc = agent.context_compressor
    if not isinstance(cc, ContextCompressor):
        return {}
    if main_go_gemini_base(agent, agent.base_url) or go_gemini_base(cc.provider, cc.model, cc.base_url):
        return {"max_tokens": _compressor_max_tokens(agent) or 0}
    return {}


def make_go_gemini_client(base_url, **kwargs):
    """Reuse native conversion without Google's credential-remediation advice."""
    from agent.gemini_native_adapter import GeminiNativeClient, gemini_http_error

    class GoGeminiClient(GeminiNativeClient):
        def _create_chat_completion(self, **kwargs):
            from agent.transports.chat_completions import _build_gemini_thinking_config
            extra = dict(kwargs.get("extra_body") or {})
            reasoning = extra.get("reasoning")
            if reasoning is None and kwargs.get("reasoning_effort") is not None:
                reasoning = {"effort": kwargs["reasoning_effort"]}
            if "thinking_config" not in extra and "thinkingConfig" not in extra:
                thinking = _build_gemini_thinking_config(kwargs.get("model", ""), reasoning)
                if thinking:
                    extra["thinking_config"] = thinking
            kwargs["extra_body"] = extra
            return super()._create_chat_completion(**kwargs)

        @property
        def _client(self):
            return self._http

        @property
        def default_headers(self):
            return dict(self._default_headers)

        def _http_error(self, response, body_text=None):
            return gemini_http_error(response, body_text=body_text, key_guidance=False)

    return GoGeminiClient(base_url=base_url, **kwargs)
