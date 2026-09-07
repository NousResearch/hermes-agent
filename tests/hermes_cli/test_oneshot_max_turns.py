"""Oneshot (-z) must forward the configured turn cap, like the chat and cron paths do.

``_run_agent`` builds its AIAgent directly and passed no ``max_iterations``, so a configured
``agent.max_turns`` was ignored and every -z run went unbounded. An unset cap stays unlimited,
which is the schema default the gateway and cron also resolve to.
"""

from __future__ import annotations

import sys
import types

import pytest

import hermes_cli.oneshot as oneshot


def _build_agent(monkeypatch, cfg) -> dict:
    """Run ``_run_agent`` against stubs and return the kwargs AIAgent was constructed with."""
    import hermes_cli.config
    import hermes_cli.mcp_startup
    import hermes_cli.runtime_provider
    import hermes_cli.tools_config
    import run_agent

    monkeypatch.setattr(hermes_cli.config, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(
        hermes_cli.runtime_provider, "resolve_runtime_provider", lambda **k: {"provider": "openai"}
    )
    monkeypatch.setattr(hermes_cli.tools_config, "_get_platform_tools", lambda *a, **k: set())
    monkeypatch.setattr(
        hermes_cli.mcp_startup, "ensure_mcp_discovery_before_agent_build", lambda **k: None
    )
    monkeypatch.setattr(oneshot, "get_fallback_chain", lambda *a, **k: [])
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr(oneshot, "_close_agent", lambda *a, **k: None)

    seen = {}

    def _fake_agent(**kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(
            run_conversation=lambda _prompt: {"final_response": "ok"},
            suppress_status_output=False,
            stream_delta_callback=None,
            tool_gen_callback=None,
        )

    monkeypatch.setattr(run_agent, "AIAgent", _fake_agent)

    oneshot._run_agent("hi")
    return seen


def test_max_turns_becomes_max_iterations(monkeypatch):
    seen = _build_agent(monkeypatch, {"model": {"default": "gpt-5"}, "agent": {"max_turns": 7}})
    assert seen["max_iterations"] == 7


def test_legacy_root_level_max_turns_still_applies(monkeypatch):
    """The root-level key is never migrated on disk; cron keeps the same fallback."""
    seen = _build_agent(monkeypatch, {"model": {"default": "gpt-5"}, "max_turns": 5})
    assert seen["max_iterations"] == 5


@pytest.mark.parametrize("cfg_agent", [{}, {"max_turns": None}])
def test_unset_max_turns_stays_unlimited(monkeypatch, cfg_agent):
    """null = unlimited is the documented schema default (caps truncated runs mid-task)."""
    seen = _build_agent(monkeypatch, {"model": {"default": "gpt-5"}, "agent": cfg_agent})
    assert seen["max_iterations"] == sys.maxsize


@pytest.mark.parametrize("spelling", ["none", "unlimited", 0])
def test_unlimited_spellings_stay_unbounded(monkeypatch, spelling):
    seen = _build_agent(
        monkeypatch, {"model": {"default": "gpt-5"}, "agent": {"max_turns": spelling}}
    )
    assert seen["max_iterations"] == sys.maxsize
