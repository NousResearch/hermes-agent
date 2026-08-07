"""Regression tests for #60955: gateway must not freeze fallback_providers.

Cron reloads ``fallback_providers`` from disk on every job. The gateway used to
freeze ``self._fallback_model`` at process start, so a chain configured (or
edited) after ``hermes gateway`` was already running never reached messaging
sessions — even though cron in the same process fell back correctly.

These tests pin the reload + cached-agent apply helpers without driving the
full Feishu session path.
"""

from __future__ import annotations

import time
from types import SimpleNamespace


def test_refresh_fallback_model_rereads_config(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
    )

    runner = SimpleNamespace(
        _fallback_model=None,
    )
    runner._load_fallback_model = GatewayRunner._load_fallback_model
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)
    chain = bound()

    assert chain == [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    assert runner._fallback_model == chain

    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: openrouter\n"
        "    model: anthropic/claude-sonnet-4.6\n"
    )
    updated = bound()
    assert updated == [
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"}
    ]
    assert runner._fallback_model == updated


def test_apply_fallback_chain_skips_while_cooldown_holds_fallback():
    """Do not clobber a live fallback activation during its cooldown window."""
    from gateway.run import GatewayRunner

    live = [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    agent = SimpleNamespace(
        _fallback_chain=live,
        _fallback_model=live[0],
        _fallback_index=1,
        _fallback_activated=True,
        _rate_limited_until=time.monotonic() + 30,
    )
    GatewayRunner._apply_fallback_chain_to_agent(
        agent,
        [{"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"}],
    )

    assert agent._fallback_chain == live
    assert agent._fallback_index == 1
    assert agent._fallback_activated is True


def test_background_agent_path_uses_a_refreshed_chain(monkeypatch):
    """The background-task agent must be built with a freshly read chain.

    Drives the real ``_run_background_task_inner`` and captures what reaches
    ``AIAgent``. Replaces an assertion that used to count call sites in
    gateway/run.py's source text — that could not tell a correctly wired call
    from a coincidentally matching string, and broke the moment the call site
    moved to gateway/watchers.py without any behaviour change.
    """
    import asyncio

    import run_agent
    from gateway.watchers import GatewayWatchersMixin

    # The startup snapshot the gateway used to freeze, and the chain a later
    # config edit produces. The agent must be built with the latter.
    stale = [{"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"}]
    refreshed = [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    captured = {}

    class _FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_conversation(self, **_kw):
            return {"final_response": ""}

    monkeypatch.setattr(run_agent, "AIAgent", _FakeAgent)
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr("gateway.run._platform_config_key", lambda p: "discord")
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 10)
    monkeypatch.setattr("gateway.run._checkpoint_agent_kwargs", lambda cfg: {})
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools", lambda cfg, key: set()
    )

    class _Adapter:
        async def send(self, *a, **k):
            return None

        def extract_media(self, response):
            return [], response

        def extract_images(self, response):
            return [], response

    source = SimpleNamespace(
        platform="discord", chat_id="c1", chat_name="chat", chat_type="dm",
        user_id="u1", user_id_alt=None, user_name="u", thread_id=None,
    )

    runner = SimpleNamespace(
        _adapter_for_source=lambda src: _Adapter(),
        _thread_metadata_for_source=lambda src, mid: None,
        _resolve_session_agent_runtime=lambda **kw: (
            "deepseek-v4-flash", {"api_key": "sk-x", "base_url": "https://x/v1"},
        ),
        _provider_routing={},
        _resolve_session_reasoning_config=lambda **kw: None,
        _resolve_session_service_tier=lambda **kw: None,
        _resolve_turn_agent_config=lambda *a, **kw: {
            "model": "deepseek-v4-flash",
            "runtime": {"api_key": "sk-x", "base_url": "https://x/v1"},
        },
        _enrich_message_with_vision=lambda *a, **kw: a[0] if a else "",
        _reasoning_config=None,
        _service_tier=None,
        _session_db=None,
        _fallback_model=stale,  # the frozen startup snapshot
        _refresh_fallback_model=lambda: refreshed,
        _cleanup_agent_resources=lambda agent: None,
        _run_in_executor_with_context=lambda fn: _immediate(fn),
    )

    async def _immediate(fn):
        return fn()

    runner._run_in_executor_with_context = _immediate
    bound = GatewayWatchersMixin._run_background_task_inner.__get__(runner)

    asyncio.run(bound(prompt="do a thing", source=source, task_id="t1"))

    assert captured, "AIAgent was never constructed — the test proves nothing"
    assert captured.get("fallback_model") == refreshed, (
        "the background agent was built with the frozen startup snapshot "
        f"instead of a freshly reloaded chain (#60955): "
        f"{captured.get('fallback_model')}"
    )


def test_cached_agent_reuse_applies_the_refreshed_chain():
    """A cached agent must adopt a chain configured after it was created.

    ``_apply_fallback_chain_to_agent`` is the reuse path's apply step; this
    pins that a refreshed chain actually lands on an agent whose own chain is
    stale and not mid-activation.
    """
    from gateway.run import GatewayRunner

    agent = SimpleNamespace(
        _fallback_chain=[{"provider": "openrouter", "model": "old/model"}],
        _fallback_model={"provider": "openrouter", "model": "old/model"},
        _fallback_index=0,
        _fallback_activated=False,
        _rate_limited_until=0,
    )
    fresh = [{"provider": "deepseek", "model": "deepseek-v4-flash"}]

    GatewayRunner._apply_fallback_chain_to_agent(agent, fresh)

    assert agent._fallback_chain == fresh
    assert agent._fallback_model == fresh[0]
