"""Phase A — model fallback is ANNOUNCED in chat (always-emitted, deduped).

When the primary model fails and a fallback activates SUCCESSFULLY, the user
must see a single status line naming the destination model + provider (and the
context-window change when it changes). The pre-existing ``_buffer_status``
message is suppressed on successful recovery (it only flushes on terminal
failure), which is why the 2026-06-19 opus->gpt-5.5 fallback was invisible.

Invariant I5: the announce is emitted exactly once per (old->new) transition
(not per retry, not per turn) and reaches the gateway status_callback
(Discord/Telegram), not just CLI _vprint.
"""

import types

import pytest

import agent.auxiliary_client as ac
from agent.chat_completion_helpers import try_activate_fallback
from agent.error_classifier import FailoverReason


@pytest.fixture(autouse=True)
def _isolate_runtime_globals():
    ac.clear_runtime_main()
    try:
        yield
    finally:
        ac.clear_runtime_main()


class _Compressor:
    """Minimal compressor stand-in carrying a context_length the announce reads."""
    def __init__(self, context_length):
        self.context_length = context_length
        self.threshold_percent = 0.5

    def update_model(self, *, model, context_length, **kw):
        self.context_length = context_length


def _fake_agent(*, model, provider, base_url, api_mode, context_length=1_000_000):
    a = types.SimpleNamespace()
    a.model = model
    a.provider = provider
    a.base_url = base_url
    a.api_mode = api_mode
    a.api_key = "primary-key"
    a.reasoning_config = None
    a.reasoning_effort = "high"
    a._config_context_length = context_length
    a._fallback_activated = False
    a._transport_cache = {}
    a._credential_pool = None
    a.context_compressor = _Compressor(context_length)
    a._primary_runtime = None
    a._fallback_index = 0
    a._rate_limited_until = 0.0
    a._fallback_chain = [{"provider": "openai-codex", "model": "gpt-5.5",
                          "reasoning_effort": "xhigh"}]
    a.fallback_model = list(a._fallback_chain)
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
    # Capture announce emissions (the always-emitted lifecycle path).
    a._announced = []
    a.status_callback = lambda kind, msg: a._announced.append((kind, msg))

    def _emit_status(message):
        a._announced.append(("lifecycle", message))
    a._emit_status = _emit_status
    a._vprint = lambda *x, **k: None
    a.log_prefix = ""
    a._last_fallback_announced = None
    return a


def _patch_resolver(monkeypatch):
    fb_client = types.SimpleNamespace(
        api_key="codex-token",
        base_url="https://chatgpt.com/backend-api/codex/",
        _custom_headers=None,
        default_headers=None,
    )
    monkeypatch.setattr(
        ac, "resolve_provider_client",
        lambda provider, model, **kw: (fb_client, model),
        raising=False,
    )
    # The fallback resolves the new model's context window; pin it to 272K.
    import agent.chat_completion_helpers as cch
    monkeypatch.setattr(
        cch, "get_model_context_length",
        lambda *a, **k: 272_000, raising=False,
    )


def test_fallback_announces_once_with_model_provider_and_window(monkeypatch):
    _patch_resolver(monkeypatch)
    ac.set_runtime_main("claude-pool", "claude-opus-4-8",
                        base_url="http://127.0.0.1:18810/anthropic",
                        api_key="primary-key", api_mode="anthropic_messages")
    agent = _fake_agent(model="claude-opus-4-8", provider="claude-pool",
                        base_url="http://127.0.0.1:18810/anthropic",
                        api_mode="anthropic_messages", context_length=1_000_000)
    activated = try_activate_fallback(agent, reason=FailoverReason.overloaded)
    assert activated is True

    announces = [m for (k, m) in agent._announced if "fallback" in m.lower()]
    assert len(announces) == 1, f"expected exactly one announce, got {announces!r}"
    msg = announces[0]
    assert "gpt-5.5" in msg
    assert "openai-codex" in msg
    # Window delta is the load-bearing diagnostic fact.
    assert "1M" in msg or "1,000,000" in msg or "272" in msg


def test_fallback_announce_deduped_on_reentrant_same_destination(monkeypatch):
    """A re-entrant fallback to the SAME destination in one turn emits once."""
    _patch_resolver(monkeypatch)
    ac.set_runtime_main("claude-pool", "claude-opus-4-8",
                        base_url="http://127.0.0.1:18810/anthropic",
                        api_key="primary-key", api_mode="anthropic_messages")
    agent = _fake_agent(model="claude-opus-4-8", provider="claude-pool",
                        base_url="http://127.0.0.1:18810/anthropic",
                        api_mode="anthropic_messages")
    try_activate_fallback(agent, reason=FailoverReason.overloaded)
    # Simulate the loop re-entering the same fallback site (same destination):
    # the agent is already on gpt-5.5; calling again must not re-announce.
    agent.model = "claude-opus-4-8"  # pretend primary retried then failed again
    agent.provider = "claude-pool"
    try_activate_fallback(agent, reason=FailoverReason.overloaded)
    announces = [m for (k, m) in agent._announced if "fallback" in m.lower()]
    assert len(announces) == 1, f"re-entrant fallback should announce once, got {announces!r}"


def test_no_announce_when_destination_equals_source(monkeypatch):
    """No model transition -> no announce (a same-model no-op must be silent)."""
    _patch_resolver(monkeypatch)
    ac.set_runtime_main("openai-codex", "gpt-5.5",
                        base_url="https://chatgpt.com/backend-api/codex/",
                        api_key="primary-key", api_mode="")
    agent = _fake_agent(model="gpt-5.5", provider="openai-codex",
                        base_url="https://chatgpt.com/backend-api/codex/",
                        api_mode="", context_length=272_000)
    # Fallback chain points at the SAME model/provider -> try_activate_fallback
    # short-circuits (no transition). No announce should fire.
    agent._fallback_chain = [{"provider": "openai-codex", "model": "gpt-5.5"}]
    agent.fallback_model = list(agent._fallback_chain)
    try_activate_fallback(agent, reason=FailoverReason.overloaded)
    announces = [m for (k, m) in agent._announced if "fallback" in m.lower()]
    assert announces == [], f"same-model no-op must be silent, got {announces!r}"
