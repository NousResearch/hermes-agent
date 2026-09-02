"""Contract tests for structured, operation-bound memory observations."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.memory_manager import MemoryManager, build_memory_context_block
from agent.memory_provider import (
    MAX_MEMORY_OBSERVATION_BYTES,
    MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES,
    MAX_MEMORY_OBSERVATION_NODES,
    MAX_MEMORY_OBSERVATION_OPERATION_NODES,
    MAX_MEMORY_OBSERVATIONS,
    MemoryObservation,
    MemoryPrefetchResult,
    _freeze_memory_observation_payload,
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


class GuardedObservationTuple(tuple):
    """Tuple fixture that fails if the manager traverses past a bounded prefix."""

    max_index: int
    accessed_indices: list[int]

    def __new__(cls, values, *, max_index):
        instance = super().__new__(cls, values)
        instance.max_index = max_index
        instance.accessed_indices = []
        return instance

    def __iter__(self):
        for index in range(tuple.__len__(self)):
            if index > self.max_index:
                raise AssertionError("observation traversal exceeded its bound")
            self.accessed_indices.append(index)
            yield tuple.__getitem__(self, index)


def _observation(payload=None, *, provider=""):
    return MemoryObservation(
        source_kind="fixture_context",
        schema="fixture.context",
        version=1,
        provider=provider,
        payload={"available": True} if payload is None else payload,
    )


def test_prefetch_result_bounds_exact_builtin_list_to_cap_plus_lookahead():
    """Public result construction never eagerly copies an unbounded list."""
    exact = MemoryPrefetchResult(
        context="context",
        observations=[
            _observation({"index": index})
            for index in range(MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES)
        ],
    )
    cap_plus_one = MemoryPrefetchResult(
        context="context",
        observations=[
            _observation({"index": index})
            for index in range(MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES + 1)
        ],
    )
    very_large = MemoryPrefetchResult(
        context="context",
        observations=[_observation({"index": index}) for index in range(100_000)],
    )

    assert len(exact.observations) == MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES
    assert len(cap_plus_one.observations) == MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES + 1
    assert len(very_large.observations) == MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES + 1
    assert very_large.context == "context"


def test_prefetch_result_rejects_list_subclass_without_iteration():
    """Subclass iteration cannot bypass the bounded exact-list contract."""

    class IterationBomb(list):
        def __iter__(self):
            raise AssertionError("list subclass must not be iterated")

    with pytest.raises(TypeError, match="observations must be a list or tuple"):
        MemoryPrefetchResult(
            context="context",
            observations=IterationBomb([_observation()]),
        )


def _capture_hook(monkeypatch, events):
    """Install a deterministic in-process memory observer for manager tests."""
    from hermes_cli import plugins
    import agent.plugin_stream_hooks as dispatcher

    def enqueue_hook(name, **kwargs):
        assert name == "memory_prefetch"
        events.append(kwargs)
        return True

    monkeypatch.setattr(
        plugins,
        "has_hook",
        lambda name: name == "memory_prefetch",
    )
    monkeypatch.setattr(dispatcher, "enqueue_plugin_observer_hook", enqueue_hook)


def _disable_hook(monkeypatch):
    from hermes_cli import plugins

    monkeypatch.setattr(plugins, "iter_hook_callbacks", lambda _name: ())
    monkeypatch.setattr(plugins, "has_hook", lambda _name: False)


def _stub_direct_prefetch(monkeypatch):
    """Keep fixture fan-out tests off the external timeout transport path."""
    direct_calls = []

    def prefetch_provider(_manager, provider, query, *, session_id=""):
        direct_calls.append(provider.name)
        return provider.prefetch(query, session_id=session_id)

    monkeypatch.setattr(MemoryManager, "_prefetch_provider", prefetch_provider)
    return direct_calls


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
    _stub_direct_prefetch(monkeypatch)
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
    _stub_direct_prefetch(monkeypatch)
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
                _observation({"valid": True}),
            ),
        ),
    )
    manager = MemoryManager()
    manager.add_provider(malformed)

    result = manager.prefetch_all_result("question")

    assert result.context == "usable context"
    assert [item.payload for item in result.observations] == [{"valid": True}]
    assert events[0]["observations"] is result.observations


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


def test_prefetch_observation_skips_context_digest_without_consumer(monkeypatch):
    """No registered observer means merged context is never encoded."""
    from hermes_cli import plugins
    import agent.plugin_stream_hooks as dispatcher

    class ExplodingContext(str):
        def encode(self, *_args, **_kwargs):
            raise AssertionError("context digest should be gated off")

    monkeypatch.setattr(plugins, "has_hook", lambda _name: False)
    monkeypatch.setattr(
        dispatcher,
        "enqueue_plugin_observer_hook",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("enqueue should be gated off")
        ),
    )

    MemoryManager._emit_prefetch_observation(
        MemoryPrefetchResult(
            context=ExplodingContext("large merged context"),
            observations=(_observation(),),
        ),
        query="question",
        session_id="session",
        task_id="task",
        turn_id="turn",
    )


def test_prefetch_observation_digest_is_preserved_for_consumer(monkeypatch):
    """An opted-in consumer still receives the exact UTF-8 digest envelope."""
    from hermes_cli import plugins
    import agent.plugin_stream_hooks as dispatcher

    events = []
    monkeypatch.setattr(plugins, "has_hook", lambda name: name == "memory_prefetch")
    monkeypatch.setattr(
        dispatcher,
        "enqueue_plugin_observer_hook",
        lambda name, **kwargs: events.append((name, kwargs)) or True,
    )
    context = "résumé context"
    result = MemoryPrefetchResult(context=context, observations=(_observation(),))

    MemoryManager._emit_prefetch_observation(
        result,
        query="question",
        session_id="session",
        task_id="task",
        turn_id="turn",
    )

    encoded = context.encode("utf-8")
    assert len(events) == 1
    name, event = events[0]
    assert name == "memory_prefetch"
    assert event["context_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert event["context_byte_length"] == len(encoded)
    assert event["observations"] is result.observations


def test_hook_callback_errors_are_isolated(monkeypatch):
    """The real plugin registry continues after one observer raises."""
    from hermes_cli import plugins
    from agent.plugin_stream_hooks import shutdown_plugin_observer_dispatcher

    shutdown_plugin_observer_dispatcher()
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

    try:
        result = manager.prefetch_all_result("question", session_id="session")

        deadline = time.monotonic() + 1.0
        while not seen and time.monotonic() < deadline:
            time.sleep(0.01)

        assert result.context == "context"
        assert seen == [result.observations]
    finally:
        shutdown_plugin_observer_dispatcher()


def test_memory_prefetch_observer_is_async_and_preserves_operation_envelope(
    monkeypatch,
):
    from hermes_cli import plugins
    from agent.plugin_stream_hooks import shutdown_plugin_observer_dispatcher

    shutdown_plugin_observer_dispatcher()
    started = threading.Event()
    release = threading.Event()
    events = []

    def consumer(**kwargs):
        started.set()
        release.wait(timeout=1.0)
        events.append(kwargs)

    monkeypatch.setattr(
        plugins,
        "iter_hook_callbacks",
        lambda name: (consumer,) if name == "memory_prefetch" else (),
    )
    monkeypatch.setattr(
        plugins,
        "has_hook",
        lambda name: name == "memory_prefetch",
    )
    manager = MemoryManager()
    manager.add_provider(
        StructuredMemoryProvider(
            name="builtin",
            result=MemoryPrefetchResult(
                context="résumé context",
                observations=(_observation(),),
            ),
        )
    )

    try:
        started_at = time.monotonic()
        result = manager.prefetch_all_result(
            "question",
            session_id="session-a",
            task_id="task-a",
            turn_id="turn-a",
        )
        elapsed = time.monotonic() - started_at

        assert elapsed < 0.1
        assert started.wait(timeout=1.0)
        release.set()
        deadline = time.monotonic() + 1.0
        while not events and time.monotonic() < deadline:
            time.sleep(0.01)

        assert len(events) == 1
        event = events[0]
        assert event["observations"] is result.observations
        assert event["session_id"] == "session-a"
        assert event["task_id"] == "task-a"
        assert event["turn_id"] == "turn-a"
        assert event["query"] == "question"
        assert event["context_byte_length"] == len(result.context.encode("utf-8"))
        assert event["telemetry_schema_version"]
        with pytest.raises(TypeError):
            result.observations[0].payload["changed"] = True  # type: ignore[index]
    finally:
        release.set()
        shutdown_plugin_observer_dispatcher()


def test_operation_observation_count_budget_keeps_ordered_prefix_and_context(
    monkeypatch, caplog
):
    events = []
    _capture_hook(monkeypatch, events)
    _stub_direct_prefetch(monkeypatch)
    first_count = MAX_MEMORY_OBSERVATIONS // 2 + 1
    first = StructuredMemoryProvider(
        name="builtin",
        result=MemoryPrefetchResult(
            context="first provider context",
            observations=tuple(
                _observation({"index": index}) for index in range(first_count)
            ),
        ),
    )
    second = StructuredMemoryProvider(
        name="external",
        result=MemoryPrefetchResult(
            context="second provider context",
            observations=tuple(
                _observation({"index": index})
                for index in range(MAX_MEMORY_OBSERVATIONS)
            ),
        ),
    )
    manager = MemoryManager()
    manager.add_provider(first)
    manager.add_provider(second)
    with caplog.at_level("WARNING", logger="agent.memory_manager"):
        result = manager.prefetch_all_result("question")

    expected = [("builtin", index) for index in range(first_count)] + [
        ("external", index) for index in range(MAX_MEMORY_OBSERVATIONS - first_count)
    ]
    assert result.context == "first provider context\n\nsecond provider context"
    assert [
        (item.provider, item.payload["index"]) for item in result.observations
    ] == expected
    assert len(result.observations) == MAX_MEMORY_OBSERVATIONS
    warnings = [
        record
        for record in caplog.records
        if "observation count budget" in record.message
    ]
    assert len(warnings) == 1
    assert "dropping remaining observations" in warnings[0].message
    assert len(events) == 1
    assert events[0]["observations"] is result.observations
    assert isinstance(events[0]["observations"], tuple)


def test_operation_budget_bounds_provider_traversal_and_preserves_later_context(
    monkeypatch, caplog
):
    _disable_hook(monkeypatch)
    _stub_direct_prefetch(monkeypatch)
    first_observations = GuardedObservationTuple(
        (
            _observation({"index": index})
            for index in range(MAX_MEMORY_OBSERVATIONS + 1_000)
        ),
        max_index=MAX_MEMORY_OBSERVATIONS,
    )
    later_observations = GuardedObservationTuple((), max_index=-1)
    first_result = MemoryPrefetchResult(
        context="first provider context",
        observations=first_observations,
    )
    later_result = MemoryPrefetchResult(
        context="later provider context",
        observations=later_observations,
    )
    assert len(first_result.observations) == MAX_MEMORY_OBSERVATIONS + 1_000
    manager = MemoryManager()
    manager.add_provider(StructuredMemoryProvider(name="builtin", result=first_result))
    manager.add_provider(StructuredMemoryProvider(name="external", result=later_result))

    with caplog.at_level("WARNING", logger="agent.memory_manager"):
        result = manager.prefetch_all_result("question")

    assert result.context == "first provider context\n\nlater provider context"
    assert [item.payload["index"] for item in result.observations] == list(
        range(MAX_MEMORY_OBSERVATIONS)
    )
    # MAX + 1 is the only look-ahead needed to prove that the prefix is
    # truncated; the large tail is never visited or encoded.
    assert first_observations.accessed_indices == list(
        range(MAX_MEMORY_OBSERVATIONS + 1)
    )
    assert later_observations.accessed_indices == []
    warnings = [
        record
        for record in caplog.records
        if "observation count budget" in record.message
    ]
    assert len(warnings) == 1


def test_memory_manager_module_identity_survives_fresh_agent_fixture():
    """A fresh-agent fixture must not strand this module's imported class.

    ``test_empty_tool_name_loop_dampening.agent_env`` deliberately imports a
    fresh ``agent.*`` tree.  Its teardown must restore the tree because this
    module imported ``MemoryManager`` during collection; otherwise the
    aggregate-budget test patches a different module object than the class it
    exercises and a provider exception drops its context.
    """
    import agent.memory_manager as current_memory_manager

    assert current_memory_manager.MemoryManager is MemoryManager
    assert (
        MemoryManager._normalize_prefetch_result.__globals__
        is current_memory_manager.__dict__
    )


def test_operation_observation_batch_budget_is_aggregate_across_providers(
    monkeypatch, caplog
):
    import agent.memory_manager as memory_manager_module

    events = []
    _capture_hook(monkeypatch, events)
    direct_calls = _stub_direct_prefetch(monkeypatch)
    # Four bounded strings make each envelope large enough that one fits while
    # two exceed this deliberately small operation-level budget.
    large_payload = {"parts": ["x" * 3500] * 4}
    monkeypatch.setattr(
        memory_manager_module, "MAX_MEMORY_OBSERVATION_BATCH_BYTES", 20_000
    )
    second_observations = GuardedObservationTuple(
        (_observation(large_payload) for _ in range(1_000)),
        max_index=0,
    )
    manager = MemoryManager()
    manager.add_provider(
        StructuredMemoryProvider(
            name="builtin",
            result=MemoryPrefetchResult(
                context="first provider context",
                observations=(_observation(large_payload),),
            ),
        )
    )
    manager.add_provider(
        StructuredMemoryProvider(
            name="external",
            result=MemoryPrefetchResult(
                context="second provider context",
                observations=second_observations,
            ),
        )
    )

    with caplog.at_level("WARNING", logger="agent.memory_manager"):
        result = manager.prefetch_all_result("question")

    assert result.context == "first provider context\n\nsecond provider context"
    assert [item.provider for item in result.observations] == ["builtin"]
    assert second_observations.accessed_indices == [0]
    assert direct_calls == ["builtin", "external"]
    warnings = [
        record
        for record in caplog.records
        if "aggregate observation batch budget" in record.message
    ]
    assert len(warnings) == 1
    assert "dropping remaining observations" in warnings[0].message
    assert len(events) == 1
    assert events[0]["observations"] is result.observations
    assert isinstance(events[0]["observations"], tuple)


def test_malformed_observation_keeps_each_provider_context_and_result_tuple(
    monkeypatch, caplog
):
    events = []
    _capture_hook(monkeypatch, events)
    _stub_direct_prefetch(monkeypatch)
    manager = MemoryManager()
    manager.add_provider(
        StructuredMemoryProvider(
            name="builtin",
            result=MemoryPrefetchResult(
                context="malformed provider context",
                observations=(_observation(object()),),
            ),
        )
    )
    manager.add_provider(
        StructuredMemoryProvider(
            name="external",
            result=MemoryPrefetchResult(
                context="valid provider context",
                observations=(_observation({"valid": True}),),
            ),
        )
    )

    with caplog.at_level("WARNING", logger="agent.memory_manager"):
        result = manager.prefetch_all_result("question")

    assert result.context == "malformed provider context\n\nvalid provider context"
    assert [item.provider for item in result.observations] == ["external"]
    assert "malformed prefetch observation" in caplog.text
    assert len(events) == 1
    assert events[0]["observations"] is result.observations
    assert isinstance(events[0]["observations"], tuple)


def test_concurrent_operations_keep_session_observations_bound(monkeypatch):
    events = []
    event_lock = threading.Lock()
    import agent.plugin_stream_hooks as dispatcher

    def capture_thread_safe(name, **kwargs):
        assert name == "memory_prefetch"
        with event_lock:
            events.append(kwargs)
        return []

    # The deterministic test observer serializes capture so assertions do not
    # depend on callback timing.
    monkeypatch.setattr(dispatcher, "enqueue_plugin_observer_hook", capture_thread_safe)
    monkeypatch.setattr(
        "hermes_cli.plugins.has_hook",
        lambda name: name == "memory_prefetch",
    )
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
    import agent.plugin_stream_hooks as dispatcher

    def capture_thread_safe(name, **kwargs):
        assert name == "memory_prefetch"
        with event_lock:
            events.append(kwargs)
        return []

    monkeypatch.setattr(dispatcher, "enqueue_plugin_observer_hook", capture_thread_safe)
    monkeypatch.setattr(
        "hermes_cli.plugins.has_hook",
        lambda name: name == "memory_prefetch",
    )
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


class _NodeCountingList(list):
    """List that trips if payload freezing visits more than a caller-set cap.

    Each iteration bumps a shared counter, so a global cap can catch a
    pathological width/depth tree the moment freezing crosses the budget —
    without depending on wall clock or memory-allocation heuristics.
    """

    def __init__(self, values, *, counter, cap):
        super().__init__(values)
        self._counter = counter
        self._cap = cap

    def __iter__(self):
        for item in list.__iter__(self):
            self._counter[0] += 1
            if self._counter[0] > self._cap:
                raise AssertionError(
                    f"payload freezing visited more than {self._cap} nodes"
                )
            yield item


def _pathological_tree(depth, width, counter, cap):
    """Build a max-width tree that would explode without the node budget."""
    if depth == 0:
        return 0
    return _NodeCountingList(
        [_pathological_tree(depth - 1, width, counter, cap) for _ in range(width)],
        counter=counter,
        cap=cap,
    )


def test_payload_node_budget_stops_pathological_traversal_early():
    """Regression: width and depth alone don't bound the traversal — nodes do.

    Before the fix, freezing a width-8 depth-6 payload processed 262,144
    leaves before ``json.dumps`` rejected the encoded byte size. The node
    budget must terminate the recursion inside the small lookahead window
    around ``MAX_MEMORY_OBSERVATION_NODES``.
    """
    counter = [0]
    cap = MAX_MEMORY_OBSERVATION_NODES + 8
    payload = _pathological_tree(depth=6, width=8, counter=counter, cap=cap)
    with pytest.raises(ValueError, match="too many nodes"):
        _freeze_memory_observation_payload(payload)
    assert counter[0] <= cap


def test_payload_node_budget_boundary_exact_and_over():
    """Boundary check right at the node budget and one past it.

    Per-container width is capped at ``MAX_MEMORY_OBSERVATION_ITEMS`` (64),
    so a flat list can't reach the node budget alone. Build an outer list of
    K inner lists, each holding M integer leaves — the total node count is
    ``1 + K + K*M``. Pick K, M so this equals the budget exactly (fits) and
    then bump one dimension so it exceeds (must fail).
    """
    # 1 + 63 + 63*64 = 4096 nodes, at MAX_MEMORY_OBSERVATION_NODES.
    exact_k, exact_m = 63, 64
    assert 1 + exact_k + exact_k * exact_m == MAX_MEMORY_OBSERVATION_NODES
    exact = [list(range(exact_m)) for _ in range(exact_k)]
    _freeze_memory_observation_payload(exact)

    # 1 + 64 + 64*64 = 4161 nodes, past the budget by 65.
    over = [list(range(64)) for _ in range(64)]
    with pytest.raises(ValueError, match="too many nodes"):
        _freeze_memory_observation_payload(over)


def test_payload_node_budget_does_not_leak_between_calls():
    """Each payload gets its own budget — earlier calls do not shrink it."""
    payload = {"a": [1, 2, 3], "b": {"c": [4, 5]}}
    for _ in range(50):
        frozen, _ = _freeze_memory_observation_payload(payload)
        assert frozen["a"] == (1, 2, 3)


def _adversarial_malformed_payload():
    """Payload whose freeze walks ~MAX_MEMORY_OBSERVATION_NODES before failing.

    63 inner lists of 64 valid ints fill 4095 valid freeze nodes; the final
    inner list holds an ``object()`` leaf whose freeze raises ``TypeError``.
    Freezing this in isolation touches 4160+ nodes before it can be rejected —
    a fresh per-payload budget forces the full traversal for every candidate.
    """
    good_inner = [list(range(64)) for _ in range(63)]
    tail_inner = list(range(63)) + [object()]
    return good_inner + [tail_inner]


def test_malformed_observation_tail_shares_operation_traversal_budget(
    monkeypatch, caplog
):
    """A malformed observation tail cannot force fresh per-payload traversal.

    Regression: ``_normalize_prefetch_result`` bounded accepted count and
    bytes, but malformed candidates dropped in the except branch did not
    consume any operation budget. Because ``_freeze_json_value`` reset its
    4096-node budget per payload, a provider could return many malformed
    payloads and force ``N × MAX_MEMORY_OBSERVATION_NODES`` traversal work.

    The fix threads a shared operation traversal budget through every
    candidate so malformed and valid payloads compete for the same node
    allowance. Once the shared budget is exhausted, every subsequent
    candidate fails on its first budget decrement and the tail is dropped
    without deep recursion — while the earlier valid prefix stays admitted.
    """
    import agent.memory_provider as memory_provider_module

    _disable_hook(monkeypatch)
    _stub_direct_prefetch(monkeypatch)

    calls = [0]
    original_freeze = memory_provider_module._freeze_json_value

    def counting_freeze(value, *, depth=0, budget=None, operation_budget=None):
        calls[0] += 1
        return original_freeze(
            value,
            depth=depth,
            budget=budget,
            operation_budget=operation_budget,
        )

    monkeypatch.setattr(
        memory_provider_module, "_freeze_json_value", counting_freeze
    )

    valid_prefix = _observation({"index": 0})
    n_malformed = 200
    malformed_tail = tuple(
        _observation(_adversarial_malformed_payload()) for _ in range(n_malformed)
    )
    provider = StructuredMemoryProvider(
        name="builtin",
        result=MemoryPrefetchResult(
            context="usable context",
            observations=(valid_prefix,) + malformed_tail,
        ),
    )
    manager = MemoryManager()
    manager.add_provider(provider)

    with caplog.at_level("WARNING", logger="agent.memory_manager"):
        result = manager.prefetch_all_result("question")

    # Context and the valid ordered-prefix observation survive; every
    # malformed candidate is dropped after logging.
    assert result.context == "usable context"
    assert [item.payload["index"] for item in result.observations] == [0]
    assert "malformed prefetch observation" in caplog.text

    # Total freeze traversal across every malformed candidate is bounded by
    # the shared operation budget plus a small O(N) cost per candidate that
    # fails on the first budget decrement. Without the fix, this would be
    # ~n_malformed × MAX_MEMORY_OBSERVATION_NODES freeze calls.
    unfixed_lower_bound = n_malformed * MAX_MEMORY_OBSERVATION_NODES // 2
    assert calls[0] < unfixed_lower_bound
    # Concrete headroom: shared budget + one decrement per tail candidate +
    # a handful of frames for the valid prefix and container bookkeeping.
    assert calls[0] <= MAX_MEMORY_OBSERVATION_OPERATION_NODES + n_malformed + 64


def test_operation_budget_does_not_relax_per_payload_node_cap():
    """A single payload is still capped at MAX_MEMORY_OBSERVATION_NODES.

    Regression: threading the operation budget through as ``budget=`` used
    to replace the per-payload counter, letting one payload traverse up to
    MAX_MEMORY_OBSERVATION_OPERATION_NODES (16 × the intended cap) before
    the encoded-byte check could reject it. The two counters must be
    additive — supplying an operation budget must not raise the per-payload
    ceiling.
    """
    # Same shape as test_payload_node_budget_boundary_exact_and_over's
    # ``over`` case (1 + 64 + 64*64 = 4161 nodes) — 65 past the per-payload
    # budget but far under the 65536-node operation budget.
    over = [list(range(64)) for _ in range(64)]
    operation_budget = [MAX_MEMORY_OBSERVATION_OPERATION_NODES]
    with pytest.raises(ValueError, match="too many nodes"):
        _freeze_memory_observation_payload(over, operation_budget=operation_budget)
    # The per-payload cap fired first, so only ~MAX_MEMORY_OBSERVATION_NODES
    # nodes of the operation budget were spent — the operation counter still
    # has almost all of its allowance for subsequent payloads.
    spent = MAX_MEMORY_OBSERVATION_OPERATION_NODES - operation_budget[0]
    assert spent <= MAX_MEMORY_OBSERVATION_NODES + 2


def test_operation_budget_exhausted_across_many_valid_payloads():
    """A shared operation budget caps traversal across many valid payloads.

    Freezing enough max-node payloads with the same operation counter must
    exhaust it and raise ``ValueError`` on subsequent freezes — proving the
    operation cap really is enforced across candidates, not merely per one.
    """
    # 1 + 63 + 63*64 = 4096 nodes: the exact per-payload budget.
    max_payload = [list(range(64)) for _ in range(63)]
    operation_budget = [MAX_MEMORY_OBSERVATION_OPERATION_NODES]
    # Exactly MAX_MEMORY_OBSERVATIONS (16) full payloads fit under the
    # operation budget of 16 × MAX_MEMORY_OBSERVATION_NODES.
    for _ in range(MAX_MEMORY_OBSERVATIONS):
        _freeze_memory_observation_payload(
            max_payload, operation_budget=operation_budget
        )
    assert operation_budget[0] == 0
    # The very next node decrement drives the operation budget negative.
    with pytest.raises(ValueError, match="operation exhausted node budget"):
        _freeze_memory_observation_payload(
            {"anything": 1}, operation_budget=operation_budget
        )


def test_payload_encoded_size_boundary_is_exact_and_bounded():
    """The compact UTF-8 payload limit accepts exactly N bytes, not N+1."""
    exact = ["x" * 4092] * 4 + [10]
    frozen, encoded_size = _freeze_memory_observation_payload(exact)
    encoded = json.dumps(frozen, ensure_ascii=False, separators=(",", ":"))
    assert len(encoded.encode("utf-8")) == MAX_MEMORY_OBSERVATION_BYTES
    assert encoded_size == MAX_MEMORY_OBSERVATION_BYTES

    over = ["x" * 4092] * 4 + [100]
    with pytest.raises(ValueError, match="payload is too large"):
        _freeze_memory_observation_payload(over)


def test_ordered_valid_observations_unaffected_by_shared_operation_budget(
    monkeypatch,
):
    """Ordered valid observations from one provider still freeze normally.

    Sanity check that the shared operation budget threaded through the
    manager does not perturb the accepted-prefix contract for a well-behaved
    provider returning small, valid payloads.
    """
    _disable_hook(monkeypatch)
    _stub_direct_prefetch(monkeypatch)
    manager = MemoryManager()
    payloads = [{"index": i, "note": f"item-{i}"} for i in range(5)]
    observations = tuple(_observation(payload) for payload in payloads)
    provider = StructuredMemoryProvider(
        name="builtin",
        result=MemoryPrefetchResult(
            context="ctx",
            observations=observations,
        ),
    )
    manager.add_provider(provider)
    result = manager.prefetch_all_result("question")
    assert result.context == "ctx"
    # Ordered prefix, all admitted, unchanged shape.
    assert [item.payload["index"] for item in result.observations] == list(range(5))
    assert [item.payload["note"] for item in result.observations] == [
        f"item-{i}" for i in range(5)
    ]


class _InfiniteObservationTuple(tuple):
    """Tuple-shaped iterable that trips if the manager pulls past a hard cap.

    The manager only accepts ``list``/``tuple`` observation containers (see
    ``MemoryPrefetchResult.__post_init__``); a plain generator would be
    rejected before it reaches the traversal path we want to bound. This
    subclass pre-materializes a very large tuple of malformed candidates and
    tracks how many ``next()`` calls the manager made — an "infinite-like"
    shape for the manager's purposes because it exceeds every legitimate
    per-operation budget by orders of magnitude, and raises loudly if the
    manager keeps pulling once the inspected-candidate cap should have
    stopped it.
    """

    def __new__(cls, values, *, guard_cap):
        instance = super().__new__(cls, values)
        instance._guard_cap = guard_cap
        instance.pulled = 0
        return instance

    def __iter__(self):
        for index in range(tuple.__len__(self)):
            if self.pulled >= self._guard_cap:
                raise AssertionError(
                    "observation traversal exceeded the inspected-candidate "
                    f"guard cap of {self._guard_cap} pulls"
                )
            self.pulled += 1
            yield tuple.__getitem__(self, index)


def _assert_inspected_cap_bounds(
    tail: _InfiniteObservationTuple,
    caplog,
    *,
    malformed_pattern: str,
):
    """Shared assertions for the three inspected-cap regression fixtures."""
    # The manager stops the tail at (or a little past) the shared cap. The
    # small O(1) headroom covers the valid-prefix candidate plus one
    # count/bytes look-ahead when the accepted budget also fires.
    assert tail.pulled <= MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES + 2
    # Per-candidate malformed warnings are bounded by the same cap — not by
    # the size of the tail, which is orders of magnitude larger.
    malformed_warnings = [
        record for record in caplog.records if malformed_pattern in record.message
    ]
    assert len(malformed_warnings) <= MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES
    # Exactly one actionable truncation warning for this cap, no per-item spam.
    truncation_warnings = [
        record
        for record in caplog.records
        if "inspected-candidate cap" in record.message
    ]
    assert len(truncation_warnings) == 1
    assert "stopping tail traversal" in truncation_warnings[0].message


def test_wrong_type_observation_tail_is_bounded_by_inspected_cap(monkeypatch, caplog):
    """A wrong-type tail cannot force unbounded next()/logging work.

    Regression: wrong-type candidates fail their ``isinstance`` guard before
    freeze runs, so the operation node budget never decrements for them.
    Without an operation-wide inspected-candidate cap, a provider returning a
    huge (or infinite-like) iterable of non-``MemoryObservation`` objects
    forced one ``next()`` and one warning per element.
    """
    _disable_hook(monkeypatch)
    _stub_direct_prefetch(monkeypatch)
    valid_prefix = _observation({"index": 0})
    tail_size = MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES * 100
    guard_cap = MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES + 4
    tail = _InfiniteObservationTuple(
        (valid_prefix,) + tuple(object() for _ in range(tail_size)),
        guard_cap=guard_cap,
    )
    manager = MemoryManager()
    manager.add_provider(
        StructuredMemoryProvider(
            name="builtin",
            result=MemoryPrefetchResult(context="usable context", observations=tail),
        )
    )

    with caplog.at_level("WARNING", logger="agent.memory_manager"):
        result = manager.prefetch_all_result("question")

    assert result.context == "usable context"
    assert [item.payload["index"] for item in result.observations] == [0]
    _assert_inspected_cap_bounds(
        tail, caplog, malformed_pattern="malformed prefetch observation"
    )


def test_invalid_metadata_observation_tail_is_bounded_by_inspected_cap(
    monkeypatch, caplog
):
    """An invalid-metadata tail also cannot force unbounded work.

    Regression: candidates whose ``source_kind`` / ``schema`` / ``version``
    fields fail validation also short-circuit before ``_freeze_json_value``
    runs, so the operation node budget never decrements. The inspected cap
    is the only thing that bounds their pull/log cost.
    """
    _disable_hook(monkeypatch)
    _stub_direct_prefetch(monkeypatch)
    valid_prefix = _observation({"index": 0})
    invalid_meta = MemoryObservation(
        source_kind="",  # empty field fails the manager-side guard
        schema="fixture.context",
        version=1,
        payload={"whatever": True},
    )
    tail_size = MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES * 100
    guard_cap = MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES + 4
    tail = _InfiniteObservationTuple(
        (valid_prefix,) + tuple(invalid_meta for _ in range(tail_size)),
        guard_cap=guard_cap,
    )
    manager = MemoryManager()
    manager.add_provider(
        StructuredMemoryProvider(
            name="builtin",
            result=MemoryPrefetchResult(context="usable context", observations=tail),
        )
    )

    with caplog.at_level("WARNING", logger="agent.memory_manager"):
        result = manager.prefetch_all_result("question")

    assert result.context == "usable context"
    assert [item.payload["index"] for item in result.observations] == [0]
    _assert_inspected_cap_bounds(
        tail, caplog, malformed_pattern="malformed prefetch observation"
    )


def test_node_budget_exhausted_tail_is_bounded_by_inspected_cap(monkeypatch, caplog):
    """A tail that fails on operation-node budget exhaustion is also bounded.

    Regression: after enough freeze work spends the shared operation node
    budget, each subsequent candidate freeze fails on its first budget
    decrement — but the manager still pulled ``next()`` and logged one
    warning per element. The inspected cap must bound this path too.
    """
    _disable_hook(monkeypatch)
    _stub_direct_prefetch(monkeypatch)
    valid_prefix = _observation({"index": 0})
    # Each adversarial payload freeze walks near-full
    # MAX_MEMORY_OBSERVATION_NODES nodes before failing, so a handful drain
    # the shared operation-node budget. Remaining candidates then fail on
    # their first budget decrement — one next() and one log per element
    # without the inspected-candidate cap.
    tail_size = MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES * 3
    guard_cap = MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES + 4
    tail = _InfiniteObservationTuple(
        (valid_prefix,)
        + tuple(
            _observation(_adversarial_malformed_payload())
            for _ in range(tail_size)
        ),
        guard_cap=guard_cap,
    )
    manager = MemoryManager()
    manager.add_provider(
        StructuredMemoryProvider(
            name="builtin",
            result=MemoryPrefetchResult(context="usable context", observations=tail),
        )
    )

    with caplog.at_level("WARNING", logger="agent.memory_manager"):
        result = manager.prefetch_all_result("question")

    assert result.context == "usable context"
    assert [item.payload["index"] for item in result.observations] == [0]
    _assert_inspected_cap_bounds(
        tail, caplog, malformed_pattern="malformed prefetch observation"
    )
