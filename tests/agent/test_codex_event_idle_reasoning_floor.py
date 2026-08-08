"""Codex event-idle watchdog must respect reasoning-model floors.

Regression for Desktop + xAI OAuth (grok-4.5) and other codex_responses
reasoning paths: the post-first-byte SSE idle kill used a tiered default
(capped at 180s for large tool payloads) and never consulted
``get_reasoning_stale_timeout_floor``. Healthy multi-minute thinking after
the opening SSE frame was killed as errno 32 / Broken pipe even when the
UI context meter showed small conversation fill (tool schemas inflate the
request estimate used only for tier selection).

Wall-clock stale already applied the reasoning floor; event-idle did not.
"""

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


@pytest.fixture(autouse=True)
def _isolate_provider_stale_config(tmp_path, monkeypatch):
    """Pure floor math must not see the developer's ~/.hermes config."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: {})


@pytest.mark.parametrize(
    "base,model,provider,expected",
    [
        (180.0, "gpt-4o", "openai", 180.0),
        (12.0, "gpt-5.5", "openai-codex", 12.0),
        (180.0, "grok-4.5", "xai-oauth", 600.0),
        (180.0, "x-ai/grok-4.5", "xai-oauth", 600.0),
        (60.0, "openai/o3", "openai-codex", 600.0),
        (120.0, "deepseek/deepseek-r1", "deepseek", 600.0),
        (0.0, "grok-4.5", "xai-oauth", 0.0),
        (-1.0, "grok-4.5", "xai-oauth", -1.0),
        (180.0, "grok-4-fast-non-reasoning", "xai-oauth", 180.0),
        (900.0, "grok-4.5", "xai-oauth", 900.0),
    ],
)
def test_resolve_codex_event_idle_timeout_floors(base, model, provider, expected):
    from agent.chat_completion_helpers import resolve_codex_event_idle_timeout

    got = resolve_codex_event_idle_timeout(
        base_timeout=base, model=model, provider=provider
    )
    assert got == expected


def test_resolve_respects_provider_stale_config(monkeypatch):
    """providers.<id>.stale_timeout_seconds raises event-idle without env."""
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda: {
            "providers": {
                "xai-oauth": {
                    "stale_timeout_seconds": 900,
                    "models": {"grok-4.5": {"stale_timeout_seconds": 900}},
                }
            }
        },
    )

    from agent.chat_completion_helpers import resolve_codex_event_idle_timeout

    got = resolve_codex_event_idle_timeout(
        base_timeout=180.0, model="grok-4.5", provider="xai-oauth"
    )
    assert got == 900.0


def _make_xai_agent(tmp_path, monkeypatch, model="grok-4.5"):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model=model,
        provider="xai-oauth",
        api_key="sk-dummy",
        base_url="https://api.x.ai/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="desktop",
    )
    agent.api_mode = "codex_responses"
    monkeypatch.setattr(agent, "_emit_status", lambda *a, **k: None)
    monkeypatch.setattr(
        agent, "_compute_non_stream_stale_timeout", lambda *a, **k: 120.0
    )
    return agent


def test_event_idle_does_not_kill_reasoning_think_under_floor(tmp_path, monkeypatch):
    """After first SSE byte, a grok think shorter than the floor must complete."""
    from agent import chat_completion_helpers as h

    agent = _make_xai_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "0.5")
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "0")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent,
        "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent,
        "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    sentinel = SimpleNamespace(ok=True)

    def fake_stream(api_kwargs, client=None, on_first_delta=None):
        agent._codex_stream_last_event_ts = time.time()
        if on_first_delta:
            on_first_delta()
        time.sleep(1.2)  # > 0.5s base, << 600s floor
        return sentinel

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)

    large = "x" * 44_000
    resp = h.interruptible_api_call(
        agent, {"model": "grok-4.5", "input": large, "tools": [{"x": "y" * 1000}]}
    )
    assert resp is sentinel
    assert "codex_stream_idle_kill" not in closes


def test_event_idle_still_kills_non_reasoning_at_base(tmp_path, monkeypatch):
    """Non-reasoning models keep the short base idle (no reasoning floor)."""
    from agent import chat_completion_helpers as h

    agent = _make_xai_agent(tmp_path, monkeypatch, model="gpt-4o")
    agent.model = "gpt-4o"
    agent.provider = "openai"
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "0.4")
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "0")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(
        agent,
        "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent,
        "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    stop = {"flag": False}

    def fake_stream(api_kwargs, client=None, on_first_delta=None):
        agent._codex_stream_last_event_ts = time.time()
        if on_first_delta:
            on_first_delta()
        deadline = time.time() + 30
        while time.time() < deadline and not stop["flag"]:
            time.sleep(0.02)
        raise RuntimeError("connection closed")

    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)
    try:
        with pytest.raises(TimeoutError) as excinfo:
            h.interruptible_api_call(agent, {"model": "gpt-4o", "input": "hi"})
        msg = str(excinfo.value).lower()
        assert "no sse events" in msg or "threshold" in msg
        assert "codex_stream_idle_kill" in closes
    finally:
        stop["flag"] = True
