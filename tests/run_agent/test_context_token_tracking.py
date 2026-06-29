
import tempfile
_old_cleanup = tempfile.TemporaryDirectory.cleanup
def safe_cleanup(self):
    try:
        _old_cleanup(self)
    except Exception:
        pass
tempfile.TemporaryDirectory.cleanup = safe_cleanup
"""Tests for context token tracking in run_agent.py's usage extraction.

The context counter (status bar) must show the TOTAL prompt tokens including
Anthropic's cached portions. This is an integration test for the token
extraction in run_conversation(), not the ContextCompressor itself (which
is tested in tests/agent/test_context_compressor.py).
"""

import sys
import types
from types import SimpleNamespace

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent


def _patch_bootstrap(monkeypatch):
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kwargs: [{
        "type": "function",
        "function": {"name": "t", "description": "t", "parameters": {"type": "object", "properties": {}}},
    }])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


class _FakeAnthropicClient:
    def close(self):
        pass


class _FakeOpenAIClient:
    """Fake OpenAI client returned by mocked resolve_provider_client."""
    api_key = "fake-codex-key"
    base_url = "https://api.openai.com/v1"
    _default_headers = None


def _make_agent(monkeypatch, api_mode, provider, response_fn):
    print("_make_agent starts", flush=True)
    _patch_bootstrap(monkeypatch)
    print("_patch_bootstrap done", flush=True)
    monkeypatch.setattr("agent.context_compressor.get_model_context_length", lambda *a, **kw: 128000)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **kw: 128000)
    if api_mode == "anthropic_messages":
        monkeypatch.setattr("agent.providers.anthropic_adapter.build_anthropic_client", lambda k, b=None, **kwargs: _FakeAnthropicClient())
    if provider == "openai-codex":
        monkeypatch.setattr(
            "agent.auxiliary_client.resolve_provider_client",
            lambda *a, **kw: (_FakeOpenAIClient(), "test-model"),
        )
    print("monkeypatching done", flush=True)

    class _A(run_agent.AIAgent):
        def __init__(self, *a, **kw):
            print("_A.__init__ starts", flush=True)
            kw.update(skip_context_files=True, skip_memory=True, max_iterations=2)
            super().__init__(*a, **kw)
            print("_A.__init__ super done", flush=True)
            self._cleanup_task_resources = self._persist_session = lambda *a, **k: None
            self._save_trajectory = lambda *a, **k: None

        def run_conversation(self, msg, conversation_history=None, task_id=None):
            def mock_api_call(kw):
                print("mock_api_call invoked", flush=True)
                raw = response_fn()
                return raw
            self._interruptible_api_call = mock_api_call
            self._disable_streaming = True
            try:
                print(f"DEBUG run_conversation start", flush=True)
                return super().run_conversation(msg, conversation_history=conversation_history, task_id=task_id)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise

    print("about to return _A", flush=True)
    return _A(model="test-model", api_key="test-key", base_url="http://localhost:1234/v1", provider=provider, api_mode=api_mode)


def _anthropic_resp(input_tok, output_tok, cache_read=0, cache_creation=0):
    usage_fields = {"input_tokens": input_tok, "output_tokens": output_tok}
    if cache_read:
        usage_fields["cache_read_input_tokens"] = cache_read
    if cache_creation:
        usage_fields["cache_creation_input_tokens"] = cache_creation
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        stop_reason="end_turn",
        usage=SimpleNamespace(**usage_fields),
        model="claude-sonnet-4-6",
    )


# -- Anthropic: cached tokens must be included --

def test_anthropic_cache_read_and_creation_added(monkeypatch):
    agent = _make_agent(monkeypatch, "anthropic_messages", "anthropic",
                        lambda: _anthropic_resp(3, 10, cache_read=15000, cache_creation=2000))
    agent.run_conversation("hi")
    assert agent.context_compressor.last_prompt_tokens == 17003  # 3+15000+2000
    assert agent.session_prompt_tokens == 17003


def test_anthropic_no_cache_fields(monkeypatch):
    print("test_anthropic_no_cache_fields starts", flush=True)
    agent = _make_agent(monkeypatch, "anthropic_messages", "anthropic",
                        lambda: _anthropic_resp(500, 20))
    print("agent created", flush=True)
    agent.run_conversation("hi")
    print("run_conversation finished", flush=True)
    assert agent.context_compressor.last_prompt_tokens == 500


def test_anthropic_cache_read_only(monkeypatch):
    agent = _make_agent(monkeypatch, "anthropic_messages", "anthropic",
                        lambda: _anthropic_resp(5, 15, cache_read=17666, cache_creation=15))
    agent.run_conversation("hi")
    assert agent.context_compressor.last_prompt_tokens == 17686  # 5+17666+15


# -- OpenAI: prompt_tokens already total --

def test_openai_prompt_tokens_unchanged(monkeypatch):
    resp = lambda: SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=SimpleNamespace(
            role="assistant", content="ok", tool_calls=None, reasoning_content=None,
        ), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=5000, completion_tokens=100, total_tokens=5100),
        model="gpt-4o",
    )
    agent = _make_agent(monkeypatch, "chat_completions", "openrouter", resp)
    agent.run_conversation("hi")
    assert agent.context_compressor.last_prompt_tokens == 5000


# -- Codex: no cache fields, getattr returns 0 --

def test_codex_no_cache_fields(monkeypatch):
    resp = lambda: SimpleNamespace(
        output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="ok")])],
        usage=SimpleNamespace(input_tokens=3000, output_tokens=50, total_tokens=3050),
        status="completed", model="gpt-5-codex",
    )
    agent = _make_agent(monkeypatch, "codex_responses", "openai-codex", resp)
    agent.run_conversation("hi")
    assert agent.context_compressor.last_prompt_tokens == 3000
