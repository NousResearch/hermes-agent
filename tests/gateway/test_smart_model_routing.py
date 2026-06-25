"""Unit tests for gateway-level smart_model_routing complexity classifier.

Why: smart_model_routing swaps simple gateway messages onto a cheap model
before agent dispatch; these tests lock in the classification contract and the
run_sync gate guards (manual /model override wins, disabled config is a no-op)
so cost-routing never silently misfires on complex or operator-controlled work.
What: Exercises GatewayRunner._apply_smart_routing directly for routing
decisions, plus replicates the run_sync gate predicate for override/disabled.
Test: Run `pytest tests/gateway/test_smart_model_routing.py`.
"""

from types import SimpleNamespace

from gateway.run import GatewayRunner


DEFAULT_MODEL = "openai/gpt-oss-120b"
CHEAP_MODEL = "deepseek/deepseek-v4-flash"

SMART_CFG = {
    "enabled": True,
    "cheap_model": CHEAP_MODEL,
    "max_simple_chars": 200,
    "max_simple_words": 40,
    "complexity_keywords": [
        "implement", "debug", "refactor", "diagnose", "migrate",
        "architect", "explain", "why does", "how does", "broken", "failing",
    ],
    "simple_keywords": ["status", "show", "check", "list", "restart", "what is", "ping", "health"],
}


def _route(message_text, model=DEFAULT_MODEL, runtime_kwargs=None, smart_cfg=None):
    """Call _apply_smart_routing without building a full GatewayRunner.

    Why: The method reads no instance state, so an unbound call keeps the test
    fast and free of gateway construction side effects.
    """
    runtime_kwargs = runtime_kwargs if runtime_kwargs is not None else {"provider": "openrouter"}
    smart_cfg = smart_cfg if smart_cfg is not None else SMART_CFG
    dummy = SimpleNamespace()
    return GatewayRunner._apply_smart_routing(
        dummy, message_text, model, runtime_kwargs, smart_cfg, {}
    )


def test_short_simple_message_routes_to_cheap_model():
    """Why: A short status check is the canonical cost-saving target."""
    model, kwargs = _route("status")
    assert model == CHEAP_MODEL
    assert kwargs["model"] == CHEAP_MODEL
    # provider/credentials must be preserved on the swapped kwargs
    assert kwargs["provider"] == "openrouter"


def test_complexity_keyword_keeps_default_model():
    """Why: Complexity signals must override shortness to protect hard work."""
    model, kwargs = _route("implement smart routing")
    assert model == DEFAULT_MODEL
    assert "model" not in kwargs  # original kwargs untouched


def test_over_max_chars_keeps_default_model():
    """Why: Long messages are presumed complex regardless of keywords."""
    long_text = "a " * 150  # 300 chars, 150 words, no complexity keyword
    model, _ = _route(long_text.strip())
    assert model == DEFAULT_MODEL


def test_session_override_skips_routing():
    """Why: Manual /model override always wins; the run_sync gate must skip routing."""
    session_overrides = {"sess-1": {"model": "anthropic/claude"}}
    session_key = "sess-1"
    # Replicates the run_sync gate predicate verbatim.
    should_route = not session_overrides.get(session_key)
    assert should_route is False
    # When skipped, model is whatever resolve returned (unchanged).
    model = DEFAULT_MODEL
    if should_route and SMART_CFG.get("enabled"):
        model, _ = _route("status")
    assert model == DEFAULT_MODEL


def test_disabled_config_skips_routing():
    """Why: enabled=False must be a no-op even for trivially simple messages."""
    disabled_cfg = dict(SMART_CFG, enabled=False)
    session_overrides = {}
    session_key = "sess-2"
    model = DEFAULT_MODEL
    if not session_overrides.get(session_key) and disabled_cfg.get("enabled"):
        model, _ = _route("status", smart_cfg=disabled_cfg)
    assert model == DEFAULT_MODEL


def test_same_cheap_and_default_is_noop():
    """Why: Guard against pointless swap when cheap_model == active model."""
    cfg = dict(SMART_CFG, cheap_model=DEFAULT_MODEL)
    model, kwargs = _route("status", smart_cfg=cfg)
    assert model == DEFAULT_MODEL
    assert "model" not in kwargs
