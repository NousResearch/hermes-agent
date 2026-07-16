"""Integration tests for Z.AI gating inside Mixture-of-Agents calls."""

import threading
import time
from types import SimpleNamespace

import pytest

from agent import agent_runtime_helpers, moa_loop, zai_concurrency


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


def test_switch_to_moa_preserves_interruptible_zai_wait(monkeypatch):
    zai_concurrency._reset_for_tests(1, 0.0)
    monkeypatch.setattr(zai_concurrency, "_ZAI_ACQUIRE_POLL_INTERVAL", 0.01)
    sem = zai_concurrency._gate._semaphore()
    assert sem.acquire()

    agent = SimpleNamespace(
        model="minimax-m3",
        provider="opencode-go",
        api_mode="anthropic_messages",
        api_key="old-key",
        base_url="https://old.example/v1",
        client=object(),
        _client_kwargs={"base_url": "https://old.example/v1"},
        _config_context_length=123456,
        _transport_cache={},
        _interrupt_requested=False,
        quiet_mode=True,
    )

    # The fake agent intentionally omits post-swap AIAgent helpers. The MoA
    # client is installed before that machinery runs, which is the live
    # switch_model path this regression covers.
    with pytest.raises(AttributeError):
        agent_runtime_helpers.switch_model(
            agent,
            new_model="review",
            new_provider="moa",
            api_key="moa-virtual-provider",
            base_url="moa://local",
            api_mode="chat_completions",
        )

    assert type(agent.client).__name__ == "MoAClient"
    assert agent.client.chat.completions.interrupt_check is not None

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"moa": {}})
    monkeypatch.setattr(
        "hermes_cli.moa_config.resolve_moa_preset",
        lambda *_args, **_kwargs: {
            "enabled": True,
            "reference_models": [],
            "aggregator": {"provider": "zai", "model": "glm-5.2"},
        },
    )
    monkeypatch.setattr(moa_loop, "_slot_runtime", _zai_runtime)
    provider_calls = []

    def _unexpected_provider_call(**kwargs):
        provider_calls.append(kwargs)
        return _response()

    monkeypatch.setattr(moa_loop, "call_llm", _unexpected_provider_call)

    errors = []
    started = threading.Event()

    def _run_switched_client():
        started.set()
        try:
            agent.client.chat.completions.create(
                messages=[{"role": "user", "content": "review"}],
            )
        except Exception as exc:  # capture the cross-thread interrupt
            errors.append(exc)

    worker = threading.Thread(target=_run_switched_client)
    worker.start()
    try:
        assert started.wait(timeout=1)
        time.sleep(0.05)
        assert worker.is_alive(), "the switched client never queued on the held slot"
        assert provider_calls == []

        agent._interrupt_requested = True
        worker.join(timeout=2)

        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], InterruptedError)
        assert provider_calls == []
        assert not sem.acquire(blocking=False)
    finally:
        agent._interrupt_requested = True
        worker.join(timeout=2)
        sem.release()


def test_one_shot_moa_forwards_interrupt_check(monkeypatch):
    interrupt_check = lambda: True
    observed = []

    def _fake_references(*args, **kwargs):
        observed.append(("references", kwargs.get("interrupt_check")))
        return [("advisor", "advice", None)]

    def _fake_call(*, interrupt_check=None, **kwargs):
        observed.append(("aggregator", interrupt_check))
        return _response("synthesis")

    monkeypatch.setattr(moa_loop, "_run_references_parallel", _fake_references)
    monkeypatch.setattr(moa_loop, "_call_moa_model", _fake_call)
    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {
            "provider": "zai",
            "model": "glm-5.2",
            "base_url": "https://api.z.ai/api/coding/paas/v4",
            "api_key": "test-key",
        },
    )

    result = moa_loop.aggregate_moa_context(
        user_prompt="review",
        api_messages=[{"role": "user", "content": "review"}],
        reference_models=[{"provider": "zai", "model": "glm-5.2"}],
        aggregator={"provider": "zai", "model": "glm-5.2"},
        interrupt_check=interrupt_check,
    )

    assert "synthesis" in result
    assert observed == [
        ("references", interrupt_check),
        ("aggregator", interrupt_check),
    ]
