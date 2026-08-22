"""Contract tests for structured, operation-bound memory observations."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.memory_manager import MemoryManager, build_memory_context_block
from agent.memory_provider import (
    MAX_MEMORY_OBSERVATION_BYTES,
    MAX_MEMORY_OBSERVATIONS,
    MemoryObservation,
    MemoryPrefetchResult,
)
from tests.agent.test_memory_provider import FakeMemoryProvider


class StructuredMemoryProvider(FakeMemoryProvider):
    """Concrete fixture provider consuming the structured prefetch contract."""

    def __init__(self, name="structured", result=None):
        super().__init__(name=name)
        self.result = result or MemoryPrefetchResult()

    def prefetch(self, query, *, session_id=""):
        self.prefetch_queries.append((query, session_id))
        return self.result


class SessionStructuredProvider(FakeMemoryProvider):
    """Built-in-shaped provider used to exercise concurrent manager calls."""

    def __init__(self, barrier):
        super().__init__(name="builtin")
        self.barrier = barrier

    def prefetch(self, query, *, session_id=""):
        self.barrier.wait(timeout=5)
        return MemoryPrefetchResult(
            context=f"context:{session_id}",
            observations=(
                MemoryObservation(
                    source_kind="session_context",
                    schema="fixture.session_context",
                    version=1,
                    payload={"session_id": session_id},
                ),
            ),
        )


def _observation(payload=None, *, provider=""):
    return MemoryObservation(
        source_kind="fixture_context",
        schema="fixture.context",
        version=1,
        provider=provider,
        payload={"available": True} if payload is None else payload,
    )


def _capture_hook(monkeypatch, events):
    """Install a deterministic in-process memory observer for manager tests."""
    import hermes_cli.lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "memory_prefetch")

    def invoke_hook(name, **kwargs):
        assert name == "memory_prefetch"
        events.append(kwargs)
        return ["ignored transform"]

    monkeypatch.setattr(lifecycle, "invoke_hook", invoke_hook)


def _disable_hook(monkeypatch):
    import hermes_cli.lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "has_hook", lambda name: False)
    monkeypatch.setattr(
        lifecycle,
        "invoke_hook",
        lambda *args, **kwargs: pytest.fail("observer must not fire"),
    )


def test_memory_prefetch_is_registered_as_an_observer_hook():
    from hermes_cli.plugins import SHELL_UNSUPPORTED_HOOKS, VALID_HOOKS

    assert "memory_prefetch" in VALID_HOOKS
    assert "memory_prefetch" in SHELL_UNSUPPORTED_HOOKS


def test_legacy_string_context_and_injected_bytes_are_unchanged(monkeypatch):
    """The compatibility path preserves the exact formatted context bytes."""
    _disable_hook(monkeypatch)
    context = "Résumé from memory\n- keep spacing and punctuation"

    legacy = FakeMemoryProvider(name="builtin")
    legacy._prefetch_result = context
    structured = StructuredMemoryProvider(
        name="builtin",
        result=MemoryPrefetchResult(context=context),
    )

    legacy_manager = MemoryManager()
    legacy_manager.add_provider(legacy)
    legacy_text = build_memory_context_block(legacy_manager.prefetch_all("query"))
    structured_manager = MemoryManager()
    structured_manager.add_provider(structured)
    structured_text = build_memory_context_block(
        structured_manager.prefetch_all("query")
    )

    assert structured_manager.prefetch_all("query").encode("utf-8") == context.encode(
        "utf-8"
    )
    assert structured_text.encode("utf-8") == legacy_text.encode("utf-8")


def test_structured_context_merge_and_minimal_operation_event(monkeypatch):
    events = []
    _capture_hook(monkeypatch, events)
    manager = MemoryManager()
    first = StructuredMemoryProvider(
        name="builtin",
        result=MemoryPrefetchResult(
            context="first — résumé",
            observations=(_observation(),),
        ),
    )
    second = StructuredMemoryProvider(
        name="external",
        result=MemoryPrefetchResult(
            context="second",
            observations=(_observation({"component": "second"}),),
        ),
    )
    manager.add_provider(first)
    manager.add_provider(second)

    result = manager.prefetch_all_result(
        "question",
        session_id="session-a",
        task_id="task-a",
        turn_id="turn-a",
    )

    expected_context = "first — résumé\n\nsecond"
    expected_bytes = expected_context.encode("utf-8")
    assert result.context == expected_context
    assert result.observations
    assert [item.provider for item in result.observations] == ["builtin", "external"]
    assert events
    event = events[0]
    assert set(event) == {
        "query",
        "session_id",
        "task_id",
        "turn_id",
        "observations",
        "context_sha256",
        "context_byte_length",
    }
    assert "result" not in event
    assert event["observations"] is result.observations
    assert event["query"] == "question"
    assert event["session_id"] == "session-a"
    assert event["task_id"] == "task-a"
    assert event["turn_id"] == "turn-a"
    assert event["context_sha256"] == hashlib.sha256(expected_bytes).hexdigest()
    assert event["context_byte_length"] == len(expected_bytes)
    assert event["context_byte_length"] != len(expected_context)
    # The hook's return value is ignored; it cannot transform the public result.
    assert result.context == expected_context


def test_mixed_legacy_and_structured_context_keeps_legacy_bytes_out_of_hook(monkeypatch):
    events = []
    _capture_hook(monkeypatch, events)
    legacy_context = "legacy raw recall that must not cross the hook"
    legacy = FakeMemoryProvider(name="builtin")
    legacy._prefetch_result = legacy_context
    structured = StructuredMemoryProvider(
        name="external",
        result=MemoryPrefetchResult(
            context="structured recall",
            observations=(_observation({"source": "explicit"}),),
        ),
    )
    manager = MemoryManager()
    manager.add_provider(legacy)
    manager.add_provider(structured)

    result = manager.prefetch_all_result("exact query", session_id="session-a")

    assert result.context == f"{legacy_context}\n\nstructured recall"
    assert len(events) == 1
    event = events[0]
    assert "result" not in event
    assert legacy_context not in repr(event)
    assert event["observations"] == result.observations
    assert [item.provider for item in event["observations"]] == ["external"]


def test_hook_receives_exact_clean_query(monkeypatch):
    events = []
    _capture_hook(monkeypatch, events)
    expanded_query = (
        '[IMPORTANT: The user has invoked the "skill-creator" skill, indicating they want '
        "you to follow its instructions. The full skill content is loaded below.]\n\n"
        "Large skill body that must not be sent to recall.\n\n"
        "The user has provided the following instruction alongside the skill invocation: "
        "exact clean query"
    )
    manager = MemoryManager()
    manager.add_provider(
        StructuredMemoryProvider(
            name="builtin",
            result=MemoryPrefetchResult(
                context="context",
                observations=(_observation(),),
            ),
        )
    )

    manager.prefetch_all_result(expanded_query, session_id="session-a")

    assert events[0]["query"] == "exact clean query"
    assert "Large skill body" not in events[0]["query"]


def test_structured_context_without_observation_does_not_emit(monkeypatch):
    events = []
    _capture_hook(monkeypatch, events)
    manager = MemoryManager()
    manager.add_provider(
        StructuredMemoryProvider(
            name="builtin",
            result=MemoryPrefetchResult(context="structured but undisclosed"),
        )
    )

    result = manager.prefetch_all_result("question")

    assert result.context == "structured but undisclosed"
    assert result.observations == ()
    assert events == []


def test_direct_structured_prefetch_marks_turn_correlation_unavailable(monkeypatch):
    """Direct manager callers cannot silently acquire a synthetic turn id."""
    events = []
    _capture_hook(monkeypatch, events)
    manager = MemoryManager()
    manager.add_provider(
        StructuredMemoryProvider(
            name="builtin",
            result=MemoryPrefetchResult(
                context="context",
                observations=(_observation(),),
            ),
        )
    )

    manager.prefetch_all_result("question", session_id="session-a")

    assert events[0]["session_id"] == "session-a"
    assert events[0]["task_id"] is None
    assert events[0]["turn_id"] is None
    assert "api_request_id" not in events[0]


def test_malformed_observations_are_dropped_but_context_survives(monkeypatch):
    events = []
    _capture_hook(monkeypatch, events)
    malformed = StructuredMemoryProvider(
        name="builtin",
        result=MemoryPrefetchResult(
            context="usable context",
            observations=(
                _observation(object()),
                _observation("x" * (MAX_MEMORY_OBSERVATION_BYTES + 1)),
            ),
        ),
    )
    manager = MemoryManager()
    manager.add_provider(malformed)

    result = manager.prefetch_all_result("question")

    assert result.context == "usable context"
    assert result.observations == ()
    assert events == []


def test_observations_are_bounded_and_recursively_immutable(monkeypatch):
    _disable_hook(monkeypatch)
    observations = tuple(
        _observation({"nested": [{"value": index}]})
        for index in range(MAX_MEMORY_OBSERVATIONS + 3)
    )
    manager = MemoryManager()
    manager.add_provider(
        StructuredMemoryProvider(
            name="builtin",
            result=MemoryPrefetchResult(context="context", observations=observations),
        )
    )

    result = manager.prefetch_all_result("question")

    assert len(result.observations) <= MAX_MEMORY_OBSERVATIONS
    assert isinstance(result.observations, tuple)
    payload = result.observations[0].payload
    assert json.loads(json.dumps(payload))["nested"][0]["value"] == 0
    assert isinstance(payload["nested"], tuple)
    with pytest.raises(TypeError):
        payload["new"] = "nope"  # type: ignore[index]
    with pytest.raises(TypeError):
        payload["nested"][0]["value"] = "nope"  # type: ignore[index]


def test_no_event_for_string_only_prefetch(monkeypatch):
    events = []
    _capture_hook(monkeypatch, events)
    provider = FakeMemoryProvider(name="builtin")
    provider._prefetch_result = "plain context"
    manager = MemoryManager()
    manager.add_provider(provider)

    assert manager.prefetch_all("question") == "plain context"
    assert events == []


def test_hook_callback_errors_are_isolated(monkeypatch):
    """The real plugin registry continues after one observer raises."""
    from hermes_cli import plugins

    plugin_manager = plugins.PluginManager()
    plugin_manager._discovered = True
    seen = []

    def broken(**kwargs):
        raise RuntimeError("observer failure")

    def healthy(observations, **kwargs):
        seen.append(observations)

    ctx = plugins.PluginContext(plugins.PluginManifest(name="fixture"), plugin_manager)
    ctx.register_hook("memory_prefetch", broken)
    ctx.register_hook("memory_prefetch", healthy)
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: plugin_manager)

    manager = MemoryManager()
    expected = MemoryPrefetchResult(
        context="context",
        observations=(_observation(),),
    )
    manager.add_provider(StructuredMemoryProvider(name="builtin", result=expected))

    result = manager.prefetch_all_result("question", session_id="session")

    assert result.context == "context"
    assert seen == [result.observations]


def test_concurrent_operations_keep_session_observations_bound(monkeypatch):
    events = []
    event_lock = threading.Lock()
    import hermes_cli.lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "memory_prefetch")

    def capture_thread_safe(name, **kwargs):
        with event_lock:
            events.append(kwargs)
        return []

    # The deterministic test observer serializes capture so assertions do not
    # depend on callback timing.
    monkeypatch.setattr(lifecycle, "invoke_hook", capture_thread_safe)
    manager = MemoryManager()
    manager.add_provider(SessionStructuredProvider(threading.Barrier(2)))

    def run(session_id):
        return session_id, manager.prefetch_all_result(
            "question",
            session_id=session_id,
            task_id=f"task-{session_id}",
            turn_id=f"turn-{session_id}",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        returned = dict(pool.map(run, ("session-a", "session-b")))

    assert {item["session_id"] for item in events} == {"session-a", "session-b"}
    for session_id, result in returned.items():
        assert result.context == f"context:{session_id}"
        assert result.observations[0].payload["session_id"] == session_id
        matching = [event for event in events if event["session_id"] == session_id]
        assert len(matching) == 1
        event = matching[0]
        assert "result" not in event
        assert event["task_id"] == f"task-{session_id}"
        assert event["turn_id"] == f"turn-{session_id}"
        assert event["observations"][0].payload["session_id"] == session_id
        encoded = result.context.encode("utf-8")
        assert event["context_sha256"] == hashlib.sha256(encoded).hexdigest()
        assert event["context_byte_length"] == len(encoded)


def test_concurrent_same_session_turns_keep_explicit_ids_bound(monkeypatch):
    events = []
    event_lock = threading.Lock()
    import hermes_cli.lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "memory_prefetch")

    def capture_thread_safe(name, **kwargs):
        with event_lock:
            events.append(kwargs)
        return []

    monkeypatch.setattr(lifecycle, "invoke_hook", capture_thread_safe)
    manager = MemoryManager()
    manager.add_provider(SessionStructuredProvider(threading.Barrier(2)))

    def run(turn_number):
        return manager.prefetch_all_result(
            "question",
            session_id="same-session",
            task_id=f"task-{turn_number}",
            turn_id=f"turn-{turn_number}",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, (1, 2)))

    assert {(event["task_id"], event["turn_id"]) for event in events} == {
        ("task-1", "turn-1"),
        ("task-2", "turn-2"),
    }
    assert all(event["session_id"] == "same-session" for event in events)
    assert all(result.context == "context:same-session" for result in results)


def test_invalid_provider_return_keeps_existing_provider_isolation(monkeypatch):
    _disable_hook(monkeypatch)

    class InvalidProvider(FakeMemoryProvider):
        def prefetch(self, query, *, session_id=""):  # type: ignore[override]
            return object()

    manager = MemoryManager()
    manager.add_provider(InvalidProvider(name="builtin"))

    # A fixture that returns a non-string/non-result is treated like the old
    # provider exception: this provider contributes no context, but the manager
    # remains usable for later providers.
    assert manager.prefetch_all("question") == ""
