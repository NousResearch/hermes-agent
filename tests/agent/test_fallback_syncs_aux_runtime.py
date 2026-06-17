"""Regression: activating a fallback model must re-sync the auxiliary-routing
runtime globals (``set_runtime_main``) to the fallback provider+model.

Root cause (proven from live logs 2026-06-16): ``turn_context`` calls
``set_runtime_main(primary_provider, primary_model)`` once at turn start. When
the turn then fails over (e.g. claude-pool 503 -> openai-codex/gpt-5.5), the
agent's ``model``/``provider`` are swapped but the process-global
``_RUNTIME_MAIN_MODEL`` keeps the *primary* model name. Auxiliary tasks
(context compression, etc.) whose explicit model gets dropped in
``_resolve_auto`` then fall back to ``_read_main_model()`` -> the stale primary
name, while the provider resolves to codex. Result: the Claude model name
``claude-opus-4-8`` is sent to the ChatGPT Codex endpoint ->
``400 ... not supported when using Codex with a ChatGPT account``.

The fix: ``try_activate_fallback`` syncs the runtime globals to the fallback
tuple, and ``restore_primary_runtime`` syncs them back to the primary.
"""
import types

import pytest

import agent.auxiliary_client as ac
from agent.chat_completion_helpers import try_activate_fallback
from agent.error_classifier import FailoverReason


@pytest.fixture(autouse=True)
def _isolate_runtime_globals():
    """Never leak the process-global runtime tuple into other tests.

    These globals are module-level in auxiliary_client; a test that sets them
    and then hits an assertion before its own cleanup would otherwise pollute
    test_auxiliary_main_first (which asserts on _read_main_*). Clear on both
    sides unconditionally.
    """
    ac.clear_runtime_main()
    try:
        yield
    finally:
        ac.clear_runtime_main()


def _fake_agent(*, model, provider, base_url, api_mode):
    """Minimal AIAgent stand-in carrying just what try_activate_fallback reads/writes."""
    a = types.SimpleNamespace()
    a.model = model
    a.provider = provider
    a.base_url = base_url
    a.api_mode = api_mode
    a.api_key = "primary-key"
    a.reasoning_config = None
    a.reasoning_effort = "high"
    a._config_context_length = 200000
    a._fallback_activated = False
    a._transport_cache = {}
    a._credential_pool = None
    a.context_compressor = None
    a._primary_runtime = None
    a._fallback_index = 0
    a._rate_limited_until = 0.0
    # fallback chain: a single codex/gpt-5.5 entry
    a._fallback_chain = [{"provider": "openai-codex", "model": "gpt-5.5",
                          "reasoning_effort": "xhigh"}]
    a.fallback_model = list(a._fallback_chain)

    # Stubs for the machinery try_activate_fallback calls into.
    a._snapshot_primary_runtime = lambda: None
    a._restore_primary_runtime = lambda: None
    a._try_activate_fallback = lambda *x, **k: False
    a._anthropic_prompt_cache_policy = lambda **k: (False, False)
    a._ensure_lmstudio_runtime_loaded = lambda: None
    a._is_azure_openai_url = lambda u: False
    a._is_direct_openai_url = lambda u: False
    a._provider_model_requires_responses_api = lambda *x, **k: True
    a._buffer_status = lambda *x, **k: None
    a._replace_primary_openai_client = lambda **k: None
    return a


def test_fallback_resyncs_runtime_globals_to_fallback_model(monkeypatch):
    # Build a fallback OpenAI-style client whose base_url is the codex endpoint.
    fb_client = types.SimpleNamespace(
        api_key="codex-token",
        base_url="https://chatgpt.com/backend-api/codex/",
        _custom_headers=None,
        default_headers=None,
    )

    # try_activate_fallback does `from agent.auxiliary_client import
    # resolve_provider_client` locally, so patch it on that module.
    monkeypatch.setattr(
        ac, "resolve_provider_client",
        lambda provider, model, **kw: (fb_client, model),
        raising=False,
    )

    # Turn start: globals reflect the PRIMARY (the live bug's precondition).
    ac.set_runtime_main(
        "claude-pool", "claude-opus-4-8",
        base_url="http://127.0.0.1:18810/anthropic",
        api_key="primary-key",
        api_mode="anthropic_messages",
    )
    assert ac._RUNTIME_MAIN_MODEL == "claude-opus-4-8"

    agent = _fake_agent(
        model="claude-opus-4-8", provider="claude-pool",
        base_url="http://127.0.0.1:18810/anthropic", api_mode="anthropic_messages",
    )
    activated = try_activate_fallback(agent, reason=FailoverReason.overloaded)
    assert activated is True, "fallback should activate"

    # The live agent swapped to codex/gpt-5.5 ...
    assert agent.model == "gpt-5.5"
    assert agent.provider == "openai-codex"

    # ... and the aux-routing globals MUST follow, so a compression call whose
    # model is dropped in _resolve_auto reads gpt-5.5, NOT the stale Claude name.
    assert ac._RUNTIME_MAIN_PROVIDER == "openai-codex", ac._RUNTIME_MAIN_PROVIDER
    assert ac._RUNTIME_MAIN_MODEL == "gpt-5.5", (
        f"runtime global still stale: {ac._RUNTIME_MAIN_MODEL!r} "
        "(this is the codex-400 bug)"
    )
    # The fallback model name is never a Claude name on the codex endpoint.
    assert "claude" not in ac._RUNTIME_MAIN_MODEL.lower()


def test_restore_primary_resyncs_runtime_globals_back(monkeypatch):
    """The mirror: restoring the primary runtime re-syncs the aux globals back
    to the primary model, so a post-recovery compression call routes to the
    primary again (not the stale fallback)."""
    from agent.agent_runtime_helpers import restore_primary_runtime

    # Globals currently reflect the FALLBACK (we were failed over to codex).
    ac.set_runtime_main(
        "openai-codex", "gpt-5.5",
        base_url="https://chatgpt.com/backend-api/codex/",
        api_key="",
        api_mode="codex_responses",
    )

    # An agent mid-fallback whose primary snapshot is claude-pool/claude-opus-4-8.
    a = _fake_agent(
        model="gpt-5.5", provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex/", api_mode="codex_responses",
    )
    a._fallback_activated = True
    a._rate_limited_until = 0.0
    a.api_key = "primary-key"
    a._use_prompt_caching = False
    a._client_kwargs = {}

    class _CC:
        def update_model(self, **kw):
            pass
    a.context_compressor = _CC()
    a._create_openai_client = lambda *x, **k: object()
    a._primary_runtime = {
        "model": "claude-opus-4-8", "provider": "claude-pool",
        "base_url": "http://127.0.0.1:18810/anthropic", "api_mode": "anthropic_messages",
        "api_key": "primary-key", "client_kwargs": {}, "use_prompt_caching": False,
        "use_native_cache_layout": False,
        "anthropic_api_key": "primary-key",
        "anthropic_base_url": "http://127.0.0.1:18810/anthropic",
        "is_anthropic_oauth": False,
        "compressor_model": "claude-opus-4-8", "compressor_context_length": 200000,
        "compressor_base_url": "http://127.0.0.1:18810/anthropic",
        "compressor_api_key": "primary-key", "compressor_provider": "claude-pool",
        "compressor_api_mode": "anthropic_messages",
    }

    # The anthropic_messages restore branch builds a native client — stub it.
    monkeypatch.setattr(
        "agent.anthropic_adapter.build_anthropic_client",
        lambda *a, **k: object(), raising=False,
    )

    restored = restore_primary_runtime(a)
    assert restored is True
    assert a.model == "claude-opus-4-8" and a.provider == "claude-pool"
    # Globals must be back on the primary.
    assert ac._RUNTIME_MAIN_PROVIDER == "claude-pool", ac._RUNTIME_MAIN_PROVIDER
    assert ac._RUNTIME_MAIN_MODEL == "claude-opus-4-8", ac._RUNTIME_MAIN_MODEL
