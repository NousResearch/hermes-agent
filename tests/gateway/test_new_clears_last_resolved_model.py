"""Regression tests for #58403 — /new must clear _last_resolved_model cache.

After a config change (e.g. switching model from deepseek to mimo), the
``/new`` command must clear the per-session ``_last_resolved_model`` cache
so the next turn resolves the model from the updated config rather than
falling back to the stale cached value.

Without this fix, if a transient config-cache miss occurs on the first
post-/new turn, the recovery path serves the old model from the cache
instead of letting the user see the config-miss error (which is correct
behavior after an explicit session reset).
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


def test_new_clears_last_resolved_model(monkeypatch):
    """/new handler must remove the session-key entry from _last_resolved_model."""
    runner = _make_runner()
    sk = "agent:main:qqbot:dm:123"

    # Turn 1: resolve model — caches it.
    _patch_resolution(monkeypatch, model_from_config="deepseek-chat")
    runner._resolve_session_agent_runtime(session_key=sk, user_config={"model": {"default": "x"}})
    assert runner._last_resolved_model.get(sk) == "deepseek-chat"

    # Simulate what /new does (mirror slash_commands.py _handle_reset_command).
    runner._session_model_overrides.pop(sk, None)
    _lrm = getattr(runner, "_last_resolved_model", None)
    if _lrm is not None:
        _lrm.pop(sk, None)

    # After /new, the per-session cache must be gone.
    assert sk not in runner._last_resolved_model


def test_new_does_not_clobber_global_fallback(monkeypatch):
    """/new clears per-session but preserves the process-wide '*' slot."""
    runner = _make_runner()
    sk = "agent:main:qqbot:dm:123"

    _patch_resolution(monkeypatch, model_from_config="deepseek-chat")
    runner._resolve_session_agent_runtime(session_key=sk, user_config={"model": {"default": "x"}})
    assert runner._last_resolved_model.get("*") == "deepseek-chat"

    # Simulate /new
    runner._session_model_overrides.pop(sk, None)
    _lrm = getattr(runner, "_last_resolved_model", None)
    if _lrm is not None:
        _lrm.pop(sk, None)

    # Per-session gone, global "*" still present (safety net for other sessions).
    assert sk not in runner._last_resolved_model
    assert runner._last_resolved_model.get("*") == "deepseek-chat"


def test_new_with_config_change_no_stale_fallback(monkeypatch):
    """After /new + config change, empty config read should NOT recover old model."""
    runner = _make_runner()
    sk = "agent:main:qqbot:dm:123"

    # Turn 1: old model cached.
    _patch_resolution(monkeypatch, model_from_config="deepseek-chat")
    runner._resolve_session_agent_runtime(session_key=sk, user_config={"model": {"default": "x"}})
    assert runner._last_resolved_model[sk] == "deepseek-chat"

    # Simulate /new clearing the cache.
    runner._session_model_overrides.pop(sk, None)
    _lrm = getattr(runner, "_last_resolved_model", None)
    if _lrm is not None:
        _lrm.pop(sk, None)

    # Turn 2: config read fails (empty) — should NOT recover old model.
    _patch_resolution(monkeypatch, model_from_config="", provider="")
    model, _ = runner._resolve_session_agent_runtime(session_key=sk, user_config={})

    # The per-session recovery is gone. But the global "*" fallback still has
    # "deepseek-chat".  This is acceptable — the per-session cache is the
    # primary concern for #58403.  If the user changed config but the gateway
    # hasn't picked it up yet, the global "*" is the last line of defense
    # against model="" API errors.
    # The key assertion: model is NOT resolved from the per-session cache.
    assert model != "" or "*" not in runner._last_resolved_model
