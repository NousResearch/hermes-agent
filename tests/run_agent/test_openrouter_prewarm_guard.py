"""Regression test for issue #24651.

The OpenRouter metadata prewarm guard in run_agent.py used to be a
``threading.Event`` with a non-atomic ``is_set() + set()`` pair, which
could lose the race under concurrent ``AIAgent.__init__`` calls and
spawn duplicate ``fetch_model_metadata`` threads.  The guard is now a
``threading.Lock`` used as a one-shot gate via
``acquire(blocking=False)`` — atomic, so exactly one caller wins.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import run_agent


@pytest.fixture
def fresh_prewarm_lock(monkeypatch):
    """Reset the module-level one-shot gate so each test sees an unclaimed lock."""
    monkeypatch.setattr(run_agent, "_openrouter_prewarm_lock", threading.Lock())


def test_prewarm_guard_is_a_lock_not_an_event():
    """Regression: the gate must be a Lock (atomic), not an Event (TOCTOU race)."""
    expected_type = type(threading.Lock())
    assert isinstance(run_agent._openrouter_prewarm_lock, expected_type)


def test_concurrent_acquire_only_one_caller_wins(fresh_prewarm_lock):
    """With N threads racing on acquire(blocking=False), exactly one returns True."""
    lock = run_agent._openrouter_prewarm_lock
    n = 64
    barrier = threading.Barrier(n)
    wins = []
    wins_lock = threading.Lock()

    def attempt():
        barrier.wait()  # maximise the race window
        if lock.acquire(blocking=False):
            with wins_lock:
                wins.append(threading.get_ident())

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(lambda _: attempt(), range(n)))

    assert len(wins) == 1, f"expected exactly one winner, got {len(wins)}"


def test_aiagent_init_spawns_prewarm_only_once_under_concurrency(
    fresh_prewarm_lock, monkeypatch
):
    """End-to-end: N concurrent AIAgent(provider='openrouter') spawn 1 prewarm thread."""
    monkeypatch.setattr(run_agent, "fetch_model_metadata", lambda *a, **k: None)

    started = []
    started_lock = threading.Lock()
    real_thread = threading.Thread

    def recording_thread(*args, **kwargs):
        if kwargs.get("name") == "openrouter-prewarm":
            with started_lock:
                started.append(kwargs)
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(run_agent.threading, "Thread", recording_thread)

    n = 32
    barrier = threading.Barrier(n)
    errors = []

    def make_agent():
        try:
            barrier.wait()
            run_agent.AIAgent(
                model="openai/gpt-4o-mini",
                provider="openrouter",
                api_key="sk-dummy",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                platform="cli",
            )
        except Exception as exc:  # pragma: no cover - surface init failures
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(lambda _: make_agent(), range(n)))

    assert not errors, f"AIAgent.__init__ raised under concurrency: {errors[:3]}"
    assert len(started) == 1, (
        f"expected exactly one openrouter-prewarm thread spawn, got {len(started)}"
    )


def test_prewarm_skipped_for_non_openrouter_provider(fresh_prewarm_lock, monkeypatch):
    """Non-openrouter providers must not spawn the prewarm thread or burn the gate."""
    monkeypatch.setattr(run_agent, "fetch_model_metadata", lambda *a, **k: None)

    started = []
    real_thread = threading.Thread

    def recording_thread(*args, **kwargs):
        if kwargs.get("name") == "openrouter-prewarm":
            started.append(kwargs)
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(run_agent.threading, "Thread", recording_thread)

    run_agent.AIAgent(
        model="claude-sonnet-4-6",
        provider="anthropic",
        api_key="sk-dummy",
        base_url="https://api.anthropic.com",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )

    assert started == [], "non-openrouter provider must not spawn prewarm thread"
    # Gate must still be free — another openrouter caller can still claim it.
    assert run_agent._openrouter_prewarm_lock.acquire(blocking=False)
