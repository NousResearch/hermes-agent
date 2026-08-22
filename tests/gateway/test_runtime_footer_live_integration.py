from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import agent.account_usage as account_usage
import gateway.run as gateway_run
from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from gateway.config import Platform
from plugins.memory.supermemory import resolve_supermemory_connection_settings


def _snapshot(provider: str, label: str) -> AccountUsageSnapshot:
    return AccountUsageSnapshot(
        provider=provider,
        source="test",
        fetched_at=datetime(2026, 8, 11, tzinfo=timezone.utc),
        windows=(AccountUsageWindow(label=label, used_percent=25),),
    )


def test_runtime_footer_context_uses_actual_agent_runtime():
    agent = SimpleNamespace(
        provider="openai-codex",
        base_url="https://chatgpt.example/codex",
        api_key="model-secret",
    )

    assert gateway_run._runtime_footer_context_from_agent(agent) == {
        "provider": "openai-codex",
        "base_url": "https://chatgpt.example/codex",
        "api_key": "model-secret",
    }


@pytest.mark.asyncio
async def test_live_footer_consumes_active_runtime_and_reenters_profile_scope(
    tmp_path,
    monkeypatch,
):
    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    (profile_home / ".env").write_text("SUPERMEMORY_API_KEY=profile-sm-secret\n")
    (profile_home / "supermemory.json").write_text(
        json.dumps({"base_url": "https://self-hosted.example/root"})
    )

    runner = SimpleNamespace(
        config=SimpleNamespace(multiplex_profiles=True),
        _resolve_profile_home_for_source=lambda source: profile_home,
    )
    source = SimpleNamespace(platform=Platform.TELEGRAM)
    agent_result = {
        "model": "openai/gpt-5.6",
        "last_prompt_tokens": 50,
        "context_length": 100,
        "_runtime_footer_context": {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.example/codex",
            "api_key": "model-secret",
        },
    }
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "display": {
                "runtime_footer": {
                    "enabled": True,
                    "fields": ["model", "quota"],
                }
            }
        },
    )

    calls = []

    def fake_fetch(provider, base_url=None, api_key=None):
        calls.append((provider, base_url, api_key))
        if provider == "supermemory":
            settings = resolve_supermemory_connection_settings()
            assert settings["api_key"] == "profile-sm-secret"
            assert settings["base_url"] == "https://self-hosted.example/root"
            assert base_url is None
            assert api_key is None
            return _snapshot("supermemory", "Supermemory credits")
        assert (provider, base_url, api_key) == (
            "openai-codex",
            "https://chatgpt.example/codex",
            "model-secret",
        )
        return _snapshot("openai-codex", "7d")

    monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)

    line = await gateway_run._build_runtime_footer_for_result(
        runner,
        source=source,
        session_key="telegram:123",
        agent_result=agent_result,
        turn_seconds=1.0,
    )

    assert line == "gpt-5.6\nQuota Used:\n7d - 25%\nSupermemory credits - 25%"
    assert set(calls) == {
        ("openai-codex", "https://chatgpt.example/codex", "model-secret"),
        ("supermemory", None, None),
    }
    assert "_runtime_footer_context" not in agent_result
