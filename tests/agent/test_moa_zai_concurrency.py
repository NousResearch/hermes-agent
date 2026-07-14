"""Integration tests for Z.AI gating inside Mixture-of-Agents calls."""

import threading
import time
from types import SimpleNamespace

import pytest

from agent import moa_loop, zai_concurrency


def _response(content="ok"):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="glm-5.2")


def _zai_runtime(slot):
    return {
        "provider": "zai",
        "model": slot.get("model") or "glm-5.2",
        "base_url": "https://api.z.ai/api/coding/paas/v4",
        "api_key": "test-key",
    }


@pytest.fixture(autouse=True)
def _stable_gate():
    zai_concurrency._reset_for_tests(2, 0.0)
    yield
    zai_concurrency._reset_for_tests(2, 0.0)


def test_parallel_moa_references_share_process_cap(monkeypatch):
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def fake_call_llm(**kwargs):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.03)
        with lock:
            in_flight -= 1
        return _response(kwargs["model"])

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    monkeypatch.setattr(moa_loop, "_slot_runtime", _zai_runtime)
    monkeypatch.setattr(
        "agent.usage_pricing.estimate_usage_cost",
        lambda *args, **kwargs: SimpleNamespace(
            amount_usd=None,
            status="unknown",
            source=None,
        ),
    )

    refs = [{"provider": "zai", "model": f"glm-{index}"} for index in range(6)]
    out = moa_loop._run_references_parallel(
        refs,
        [{"role": "user", "content": "review"}],
    )

    assert len(out) == 6
    assert peak == 2


def test_zai_stream_holds_slot_until_iterator_finishes(monkeypatch):
    zai_concurrency._reset_for_tests(1, 0.0)
    release_first = threading.Event()
    first_created = threading.Event()
    calls = []
    lock = threading.Lock()

    class _Stream:
        def __init__(self, index):
            self.index = index
            self.done = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.done:
                raise StopIteration
            self.done = True
            if self.index == 1:
                release_first.wait(timeout=2)
            return self.index

    def fake_call_llm(**kwargs):
        with lock:
            calls.append(kwargs)
            index = len(calls)
        if index == 1:
            first_created.set()
        return _Stream(index)

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    runtime = _zai_runtime({"model": "glm-5.2"})
    outputs = []

    def consume():
        outputs.append(
            list(
                moa_loop._call_moa_model(
                    runtime=runtime,
                    task="moa_aggregator",
                    messages=[],
                    stream=True,
                )
            )
        )

    first = threading.Thread(target=consume)
    second = threading.Thread(target=consume)
    first.start()
    assert first_created.wait(timeout=1)
    second.start()
    time.sleep(0.05)

    assert len(calls) == 1
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(calls) == 2
    assert sorted(outputs) == [[1], [2]]


def test_moa_client_preserves_interrupt_check():
    interrupt_check = lambda: True
    client = moa_loop.MoAClient("review", interrupt_check=interrupt_check)
    assert client.chat.completions.interrupt_check is interrupt_check
