from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone

import pytest

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
import gateway.runtime_footer as runtime_footer


@pytest.mark.asyncio
async def test_quota_fetch_yields_event_loop_during_delayed_request(monkeypatch):
    ticked = asyncio.Event()
    saw_tick_from_worker: list[bool] = []

    monkeypatch.setattr(
        "plugins.memory.supermemory.resolve_supermemory_connection_settings",
        lambda: {"api_key": "test-key", "base_url": "https://api.supermemory.ai", "api_timeout": 10},
    )

    def delayed_fetch(provider, *, base_url=None, api_key=None):
        assert provider == "supermemory"
        time.sleep(0.05)
        saw_tick_from_worker.append(ticked.is_set())
        return None

    async def tick_while_fetching():
        await asyncio.sleep(0.01)
        ticked.set()

    monkeypatch.setattr("agent.account_usage.fetch_account_usage", delayed_fetch)

    snapshot, _ = await asyncio.gather(
        runtime_footer.fetch_runtime_footer_quota_snapshot("custom-local"),
        tick_while_fetching(),
    )

    assert snapshot is None
    assert saw_tick_from_worker == [True]


@pytest.mark.asyncio
async def test_quota_fetch_timeout_is_bounded_and_fail_open(monkeypatch):
    release_worker = threading.Event()

    def blocked_fetch(provider, *, base_url=None, api_key=None):
        release_worker.wait(timeout=1)
        return None

    monkeypatch.setattr("agent.account_usage.fetch_account_usage", blocked_fetch)
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        snapshot = await runtime_footer.fetch_runtime_footer_quota_snapshot(
            "custom-local",
            timeout_seconds=0.01,
        )
    finally:
        release_worker.set()

    assert snapshot is None
    assert loop.time() - started < 0.2


@pytest.mark.asyncio
async def test_quota_fetch_combines_model_and_supermemory_snapshots(monkeypatch):
    now = datetime(2026, 8, 11, tzinfo=timezone.utc)
    fetched = []

    monkeypatch.setattr(
        "plugins.memory.supermemory.resolve_supermemory_connection_settings",
        lambda: {"api_key": "test-key", "base_url": "https://api.supermemory.ai", "api_timeout": 10},
    )

    def fake_fetch(provider, *, base_url=None, api_key=None):
        fetched.append((provider, base_url, api_key))
        if provider == "openai-codex":
            return AccountUsageSnapshot(
                provider=provider,
                source="test",
                fetched_at=now,
                windows=(AccountUsageWindow(label="Session", used_percent=25),),
            )
        if provider == "supermemory":
            return AccountUsageSnapshot(
                provider=provider,
                source="test",
                fetched_at=now,
                windows=(
                    AccountUsageWindow(
                        label="Supermemory credits",
                        used_percent=80,
                        detail="$10.00 of $50.00 remaining",
                    ),
                ),
            )
        raise AssertionError(provider)

    monkeypatch.setattr("agent.account_usage.fetch_account_usage", fake_fetch)

    snapshot = await runtime_footer.fetch_runtime_footer_quota_snapshot(
        "openai-codex",
        base_url="https://chatgpt.example/codex",
        api_key="codex-token",
    )

    assert snapshot is not None
    assert snapshot.provider == "combined"
    assert [(window.label, window.used_percent) for window in snapshot.windows] == [
        ("Session", 25),
        ("Supermemory credits", 80),
    ]
    assert fetched == [
        ("openai-codex", "https://chatgpt.example/codex", "codex-token"),
        ("supermemory", None, None),
    ]


@pytest.mark.asyncio
async def test_quota_fetch_skips_supermemory_without_a_configured_key(monkeypatch):
    fetched = []

    monkeypatch.setattr(
        "plugins.memory.supermemory.resolve_supermemory_connection_settings",
        lambda: {"api_key": "", "base_url": "https://api.supermemory.ai", "api_timeout": 10},
    )

    def fake_fetch(provider, *, base_url=None, api_key=None):
        fetched.append((provider, base_url, api_key))
        return None

    monkeypatch.setattr("agent.account_usage.fetch_account_usage", fake_fetch)

    snapshot = await runtime_footer.fetch_runtime_footer_quota_snapshot("custom-local")

    assert snapshot is None
    assert fetched == []


@pytest.mark.asyncio
async def test_quota_fetch_timeout_includes_supermemory_config_resolution(monkeypatch):
    def slow_resolver():
        time.sleep(0.1)
        return {"api_key": "", "base_url": "https://api.supermemory.ai", "api_timeout": 10}

    monkeypatch.setattr(
        "plugins.memory.supermemory.resolve_supermemory_connection_settings",
        slow_resolver,
    )

    loop = asyncio.get_running_loop()
    started = loop.time()
    snapshot = await runtime_footer.fetch_runtime_footer_quota_snapshot(
        "custom-local",
        timeout_seconds=0.01,
    )

    assert snapshot is None
    assert loop.time() - started < 0.05


@pytest.mark.asyncio
async def test_async_footer_builder_skips_fetch_when_footer_disabled(monkeypatch):
    called = False

    async def forbidden_fetch(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("quota fetch must stay disabled")

    monkeypatch.setattr(
        runtime_footer,
        "fetch_runtime_footer_quota_snapshot",
        forbidden_fetch,
    )

    line = await runtime_footer.build_footer_line_async(
        user_config={"display": {"runtime_footer": {"enabled": False}}},
        platform_key="telegram",
        provider="openai-codex",
        base_url="https://chatgpt.example/codex",
        api_key="codex-token",
        model="openai/gpt-5.6",
        context_tokens=50,
        context_length=100,
    )

    assert line == ""
    assert called is False


@pytest.mark.asyncio
async def test_async_footer_builder_skips_fetch_when_quota_field_is_omitted(monkeypatch):
    called = False

    async def forbidden_fetch(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("quota fetch must stay hidden")

    monkeypatch.setattr(
        runtime_footer,
        "fetch_runtime_footer_quota_snapshot",
        forbidden_fetch,
    )

    line = await runtime_footer.build_footer_line_async(
        user_config={
            "display": {
                "runtime_footer": {"enabled": True, "fields": ["model"]}
            }
        },
        platform_key="telegram",
        provider="openai-codex",
        model="openai/gpt-5.6",
        context_tokens=50,
        context_length=100,
    )

    assert line == "gpt-5.6"
    assert called is False


@pytest.mark.asyncio
async def test_async_footer_builder_fetches_and_renders_enabled_quota(monkeypatch):
    snapshot = AccountUsageSnapshot(
        provider="supermemory",
        source="test",
        fetched_at=datetime(2026, 8, 11, tzinfo=timezone.utc),
        windows=(AccountUsageWindow(label="Supermemory credits", used_percent=80),),
    )

    async def fake_fetch(provider, *, base_url=None, api_key=None, timeout_seconds=5.0):
        assert (provider, base_url, api_key) == (
            "openai-codex",
            "https://chatgpt.example/codex",
            "codex-token",
        )
        return snapshot

    monkeypatch.setattr(
        runtime_footer,
        "fetch_runtime_footer_quota_snapshot",
        fake_fetch,
    )

    line = await runtime_footer.build_footer_line_async(
        user_config={
            "display": {
                "runtime_footer": {"enabled": True, "fields": ["model", "quota"]}
            }
        },
        platform_key="telegram",
        provider="openai-codex",
        base_url="https://chatgpt.example/codex",
        api_key="codex-token",
        model="openai/gpt-5.6",
        context_tokens=50,
        context_length=100,
    )

    assert line == "gpt-5.6\nQuota Used:\nSupermemory credits - 80%"
