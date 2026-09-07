"""Regression tests for #35314 — empty model on the post-interrupt recovery turn.

After a ``stream_interrupt_abort`` during an active gateway session, the recovery
turn was sometimes built with ``model=""`` (a transient config-cache miss returned
an empty ``user_config``). Every API call then failed HTTP 400 "No models
provided", "trying fallback..." was logged but never executed (the user had no
fallback configured), and the session went silent until the user re-sent.

These tests pin two fixes:
  1. ``_resolve_session_agent_runtime`` caches the last successfully-resolved
     model per session and recovers it when a fresh resolution comes back empty.
  2. ``_has_pending_fallback`` gates the "trying fallback..." status so it is only
     announced when a fallback chain actually exists.
"""

import threading

import gateway.run as gateway_run


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_model_overrides = {}
    runner._last_resolved_model = {}
    runner._service_tier = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    return runner


def _patch_resolution(monkeypatch, *, model_from_config: str, provider: str = "openrouter"):
    """Stub gateway model + runtime resolution to a known state."""
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda cfg=None: model_from_config)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": provider,
            "api_key": "x",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        },
    )


def test_empty_model_recovers_session_last_good(monkeypatch):
    runner = _make_runner()
    sk = "agent:main:discord:dm:123"

    # Turn 1: config has the model — cache it.
    _patch_resolution(monkeypatch, model_from_config="deepseek/deepseek-v4-flash")
    runner._resolve_session_agent_runtime(session_key=sk, user_config={"model": {"default": "x"}})

    # Turn 2: simulate the transient empty config read (the #35314 race).
    _patch_resolution(monkeypatch, model_from_config="", provider="")
    model, _ = runner._resolve_session_agent_runtime(session_key=sk, user_config={})

    assert model == "deepseek/deepseek-v4-flash", "recovery turn must reuse last-known-good, not build model=''"


def test_bare_runner_without_cache_attr_does_not_crash(monkeypatch):
    """object.__new__ runners (test helpers / pitfall #17) lack _last_resolved_model.

    The getattr guard must tolerate the missing attribute.
    """
    _patch_resolution(monkeypatch, model_from_config="deepseek/deepseek-v4-flash")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_model_overrides = {}
    runner._service_tier = None
    # Deliberately omit _last_resolved_model.

    model, _ = runner._resolve_session_agent_runtime(session_key="x", user_config={"model": {}})

    assert model == "deepseek/deepseek-v4-flash"


# ── _has_pending_fallback gate ──────────────────────────────────────────────


def _bare_agent():
    import run_agent

    return object.__new__(run_agent.AIAgent)


def test_has_pending_fallback_empty_chain():
    agent = _bare_agent()
    agent._fallback_chain = []
    agent._fallback_index = 0
    assert agent._has_pending_fallback() is False


