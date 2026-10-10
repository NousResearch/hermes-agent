"""Trust-boundary tests for bounded memory observations."""

from __future__ import annotations

import threading
from pathlib import Path

from agent.memory_manager import MemoryManager
from agent.memory_provider import (
    MAX_MEMORY_OBSERVATION_INSPECTED_CANDIDATES,
    MemoryObservation,
    MemoryPrefetchResult,
)
from tests.agent.test_memory_observation import (
    StructuredMemoryProvider,
    _adversarial_malformed_payload,
    _disable_hook,
    _enable_external_prefetch_spill,
    _observation,
    _stub_direct_prefetch,
)


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


def test_observation_tuple_subclass_iterator_is_bypassed_and_context_is_kept(
    monkeypatch,
):
    """Read valid tuple storage without invoking the provider's iterator override."""
    _disable_hook(monkeypatch)
    iterator_called = threading.Event()

    class BrokenTuple(tuple):
        def __iter__(self):
            iterator_called.set()
            raise RuntimeError("broken observation iterator")

    manager = MemoryManager()
    manager.add_provider(
        StructuredMemoryProvider(
            name="builtin",
            result=MemoryPrefetchResult(
                context="usable context",
                observations=BrokenTuple((_observation({"kept": True}), _observation())),
            ),
        )
    )

    result = manager.prefetch_all_result("question")

    assert result.context == "usable context"
    assert [item.payload for item in result.observations] == [
        {"kept": True},
        {"available": True},
    ]
    assert not iterator_called.is_set()


def test_mapping_field_scan_is_bounded_and_prefetch_context_is_kept():
    from agent.memory_manager_prefetch import normalize_prefetch_result

    candidate = {
        **{f"extra-{index}": index for index in range(2_000)},
        "source_kind": "recall",
        "schema": "fixture.context",
        "version": 1,
        "payload": {"kept": True},
    }
    normalized = normalize_prefetch_result(
        StructuredMemoryProvider(name="external", result=MemoryPrefetchResult()),
        {"context": "usable context", "observations": [candidate]},
        remaining_bytes=64 * 1024,
    )

    assert normalized.result.context == "usable context"
    assert normalized.result.observations == ()


def test_structured_prefetch_context_subclass_is_normalized_before_spill(
    monkeypatch, tmp_path
):
    """A forged __len__ cannot skip spilling structured external context."""
    _disable_hook(monkeypatch)

    class LyingString(str):
        def __len__(self):
            return 1

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _enable_external_prefetch_spill(home)
    context = LyingString("x" * 10_100)
    manager = MemoryManager()
    manager.add_provider(
        StructuredMemoryProvider(
            name="external", result=MemoryPrefetchResult(context=context)
        )
    )

    result = manager.prefetch_all_result("question", session_id="lying-length")

    assert type(result.context) is str
    assert "full content saved to " in result.context
    saved = Path(result.context.split("full content saved to ", 1)[1].split("]", 1)[0])
    assert saved.read_text(encoding="utf-8") == str.__str__(context) + "\n"
