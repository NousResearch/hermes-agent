"""Provider-scoped physical LLM request concurrency invariants (#109889)."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent import llm_concurrency, relay_llm, relay_runtime
from hermes_cli import config as config_module


@pytest.fixture(autouse=True)
def _provider_limit(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "load_config_readonly",
        lambda: {"providers": {"openrouter": {"max_in_flight": 1}}},
    )
    llm_concurrency._reset_provider_limiters()
    yield
    llm_concurrency._reset_provider_limiters()


@pytest.fixture
def managed_relay_turn(tmp_path, monkeypatch):
    pytest.importorskip("nemo_relay")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="limited-session", platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease, turn_id="limited-turn", task_id="limited-task")
    lease.host.retain_managed_execution("test.llm_concurrency")
    try:
        yield
    finally:
        lease.host.release_managed_execution("test.llm_concurrency")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


def test_sync_and_async_attempts_share_only_their_provider_budget():
    first_entered = threading.Event()
    release_first = threading.Event()
    async_submitted = threading.Event()
    async_entered = threading.Event()
    active = 0
    max_active = 0
    active_lock = threading.Lock()

    def enter(*, hold: bool = False):
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            if hold:
                # Codex-compatible transports can open a nested Relay stream from
                # inside the outer physical callback; this must reuse the slot.
                assert relay_llm.execute(
                    {}, lambda _request: "nested-ok",
                    name="openrouter", model_name="nested", session_id="",
                ) == "nested-ok"
                first_entered.set()
                assert release_first.wait(timeout=2)
            else:
                async_entered.set()
            return "ok"
        finally:
            with active_lock:
                active -= 1

    async def run_async_attempt():
        async_submitted.set()
        return await relay_llm.execute_async(
            {}, lambda _request: asyncio.to_thread(enter),
            name="openrouter", model_name="async-model", session_id="",
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(
            relay_llm.execute, {}, lambda _request: enter(hold=True),
            name="openrouter", model_name="sync-model", session_id="",
        )
        assert first_entered.wait(timeout=2)
        second = pool.submit(asyncio.run, run_async_attempt())
        assert async_submitted.wait(timeout=2)

        # An unrelated provider is not queued behind OpenRouter's occupied slot.
        assert relay_llm.execute(
            {}, lambda _request: "other-ok",
            name="anthropic", model_name="other-model", session_id="",
        ) == "other-ok"
        assert not async_entered.is_set()

        release_first.set()
        assert first.result(timeout=2) == "ok"
        assert second.result(timeout=2) == "ok"

    assert max_active == 1


def test_cancelled_sync_waiter_never_dispatches_after_slot_opens():
    holder_entered = threading.Event()
    release_holder = threading.Event()
    waiter_submitted = threading.Event()

    def hold(_request):
        holder_entered.set()
        assert release_holder.wait(timeout=2)
        return "held"

    class WaitingCallback:
        _interrupt_requested = False
        dispatched = False

        def call(self, _request):
            self.dispatched = True
            return "should-not-run"

    waiting = WaitingCallback()

    def wait_for_slot():
        waiter_submitted.set()
        return relay_llm.execute(
            {}, waiting.call, name="openrouter", model_name="waiting", session_id="")

    with ThreadPoolExecutor(max_workers=2) as pool:
        holder = pool.submit(
            relay_llm.execute, {}, hold,
            name="openrouter", model_name="holder", session_id="",
        )
        assert holder_entered.wait(timeout=2)
        waiter = pool.submit(wait_for_slot)
        assert waiter_submitted.wait(timeout=2)
        waiting._interrupt_requested = True
        release_holder.set()

        assert holder.result(timeout=2) == "held"
        with pytest.raises(InterruptedError, match="concurrency wait interrupted"):
            waiter.result(timeout=2)
        assert not waiting.dispatched

    assert relay_llm.execute(
        {}, lambda _request: "fresh",
        name="openrouter", model_name="fresh", session_id="",
    ) == "fresh"


def test_managed_relay_callback_uses_the_same_provider_budget(managed_relay_turn):
    managed_entered = threading.Event()
    release_managed = threading.Event()
    unmanaged_submitted = threading.Event()
    unmanaged_entered = threading.Event()

    def managed_callback(_request):
        managed_entered.set()
        assert release_managed.wait(timeout=2)
        return {"content": "managed"}

    def run_unmanaged():
        unmanaged_submitted.set()
        return relay_llm.execute(
            {}, lambda _request: unmanaged_entered.set() or "unmanaged",
            name="openrouter", model_name="unmanaged", session_id="",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        managed = pool.submit(
            relay_llm.execute, {}, managed_callback,
            name="openrouter", model_name="managed", session_id="limited-session",
        )
        assert managed_entered.wait(timeout=2)
        unmanaged = pool.submit(run_unmanaged)
        assert unmanaged_submitted.wait(timeout=2)
        assert relay_llm.execute(
            {}, lambda _request: "other",
            name="anthropic", model_name="other", session_id="",
        ) == "other"
        assert not unmanaged_entered.is_set()
        release_managed.set()

        assert managed.result(timeout=2) == {"content": "managed"}
        assert unmanaged.result(timeout=2) == "unmanaged"


@pytest.mark.parametrize("finish", ["exhaust", "close"])
def test_stream_holds_provider_slot_until_its_lifetime_ends(finish):
    second_submitted = threading.Event()
    second_opened = threading.Event()

    first = relay_llm.stream(
        {}, lambda _request: iter(("one", "two")),
        name="openrouter", model_name="primary", session_id="", finalizer=dict,
    )

    def open_auxiliary_stream():
        second_submitted.set()
        return relay_llm.stream_current(
            {}, lambda _request: second_opened.set() or iter(("aux",)),
            name="openrouter", model_name="auxiliary", finalizer=dict,
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(open_auxiliary_stream)
        assert second_submitted.wait(timeout=2)
        assert relay_llm.execute(
            {}, lambda _request: "unlimited",
            name="anthropic", model_name="other", session_id="",
        ) == "unlimited"
        assert not second_opened.is_set()

        if finish == "exhaust":
            assert list(first) == ["one", "two"]
        else:
            first.close()

        second = pending.result(timeout=2)
        assert second_opened.is_set()
        assert list(second) == ["aux"]
