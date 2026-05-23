from concurrent.futures import Future
from types import SimpleNamespace

from agent.context_compressor import ContextCompressor


class ImmediateExecutor:
    """Synchronous executor for deterministic tests."""

    def submit(self, fn, *args, **kwargs):
        fut = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # pragma: no cover - mirrors Future behavior
            fut.set_exception(exc)
        return fut


class HoldingExecutor:
    """Executor that records work without running it."""

    def __init__(self):
        self.future = Future()

    def submit(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        return self.future


def make_compressor(**overrides):
    compressor = ContextCompressor(
        model="test-model",
        threshold_percent=0.50,
        protect_first_n=1,
        protect_last_n=3,
        summary_target_ratio=0.10,
        quiet_mode=True,
        config_context_length=128_000,
        background_precompression={
            "enabled": True,
            "trigger_threshold": 0.20,
            "max_workers": 1,
        },
        **overrides,
    )
    compressor.tail_token_budget = 1_000
    compressor._precompression_executor = ImmediateExecutor()
    return compressor


def make_messages(extra_middle=""):
    messages = [{"role": "system", "content": "system"}]
    for idx in range(12):
        messages.append({"role": "user", "content": f"user turn {idx} {extra_middle}"})
        messages.append({"role": "assistant", "content": f"assistant turn {idx} {extra_middle}"})
    messages.append({"role": "user", "content": "latest user request"})
    return messages


def test_background_precompaction_cache_is_used_by_later_compress(monkeypatch):
    compressor = make_compressor()
    calls = []

    def fake_generate(turns, focus_topic=None, previous_summary_override=None, update_state=True):
        calls.append([m["content"] for m in turns])
        return "[CONTEXT SUMMARY]: background ready summary"

    monkeypatch.setattr(compressor, "_generate_summary_locked", fake_generate)
    messages = make_messages()

    assert compressor.maybe_start_background_precompression(messages, current_tokens=30_000) is True
    assert len(calls) == 1

    def fail_if_called(*args, **kwargs):  # pragma: no cover - should not run
        raise AssertionError("foreground compression should use ready background summary")

    monkeypatch.setattr(compressor, "_generate_summary", fail_if_called)
    compressed = compressor.compress(messages, current_tokens=70_000)

    assert any("background ready summary" in str(msg.get("content")) for msg in compressed)
    assert compressor._last_precompression_cache_hit is True


def test_background_precompaction_cache_is_rejected_when_messages_change(monkeypatch):
    compressor = make_compressor()
    monkeypatch.setattr(
        compressor,
        "_generate_summary_locked",
        lambda turns, focus_topic=None, previous_summary_override=None, update_state=True: "[CONTEXT SUMMARY]: stale background summary",
    )
    original_messages = make_messages()

    assert compressor.maybe_start_background_precompression(original_messages, current_tokens=30_000) is True

    changed_messages = make_messages(extra_middle="changed")
    monkeypatch.setattr(
        compressor,
        "_generate_summary",
        lambda turns, focus_topic=None: "[CONTEXT SUMMARY]: fresh foreground summary",
    )

    compressed = compressor.compress(changed_messages, current_tokens=70_000)

    rendered = "\n".join(str(msg.get("content")) for msg in compressed)
    assert "fresh foreground summary" in rendered
    assert "stale background summary" not in rendered
    assert compressor._last_precompression_cache_hit is False


def test_background_precompaction_does_not_mutate_live_summary_state(monkeypatch):
    compressor = make_compressor()
    compressor._previous_summary = "live foreground summary"
    compressor._summary_failure_cooldown_until = 123.0
    compressor._last_summary_error = "keep this foreground error"

    def fake_call_llm(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="background summary"))]
        )

    monkeypatch.setattr("agent.context_compressor.call_llm", fake_call_llm)

    assert compressor.maybe_start_background_precompression(make_messages(), current_tokens=30_000) is True

    assert compressor._previous_summary == "live foreground summary"
    assert compressor._summary_failure_cooldown_until == 123.0
    assert compressor._last_summary_error == "keep this foreground error"


def test_session_reset_cancels_inflight_background_precompaction():
    compressor = make_compressor()
    executor = HoldingExecutor()
    setattr(compressor, "_precompression_executor", executor)

    assert compressor.maybe_start_background_precompression(make_messages(), current_tokens=30_000) is True
    assert compressor._precompression_future is executor.future

    compressor.on_session_reset()

    assert executor.future.cancelled()
    assert compressor._precompression_future is None
    assert compressor._precompression_inflight_key is None
    assert compressor._precompression_inflight_generation is None
    assert compressor._precompression_cache == {}


def test_abort_on_summary_failure_still_aborts_after_background_failure(monkeypatch):
    compressor = make_compressor(abort_on_summary_failure=True)
    messages = make_messages()

    def fail_background(*args, **kwargs):
        raise RuntimeError("background summary failed")

    monkeypatch.setattr(compressor, "_generate_summary_locked", fail_background)
    assert compressor.maybe_start_background_precompression(messages, current_tokens=30_000) is True
    assert compressor._precompression_cache == {}

    monkeypatch.setattr(compressor, "_generate_summary", lambda *args, **kwargs: "")

    compressed = compressor.compress(messages, current_tokens=70_000)

    assert compressed == messages
    assert compressor._last_compress_aborted is True
    assert compressor._last_summary_fallback_used is False
    assert compressor._last_summary_dropped_count == 0
    assert compressor._last_precompression_cache_hit is False
