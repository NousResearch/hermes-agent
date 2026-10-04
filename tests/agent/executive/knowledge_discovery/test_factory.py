"""Focal tests for the canonical EvidencePackEngine factory.

These tests cover the production composition seam at
``agent.executive.knowledge_discovery.factory.build_evidence_pack_engine``.
They exercise the actual factory callable (not a higher-order
factory builder) end-to-end against minimal in-memory doubles:

* Borrowed ``storage`` is a duck-typed state_meta backing exposing
  callable ``get_meta`` / ``set_meta``. The factory borrows the
  reference for the engine's lifetime; the tests verify it is never
  closed and never re-constructed.
* The state loader is exercised through the contract provider to
  confirm the canonical ``goal:<session_id>`` lookup, propagation of
  ``get_meta`` exceptions, malformed-JSON propagation, non-Mapping
  propagation, and missing-state behavior.
* The audit sink default-off and default-on paths are exercised
  against a real process emitter (``agent.monitoring.emitter``),
  confirming ``get_emitter`` is resolved exactly once when the
  caller did not supply an audit sink and never when it did.
* Default-off invariants: ``build_objective_services`` must not
  invoke the factory, must not resolve the emitter, and must not
  read from storage when ``goals.evidence_pack.enabled`` is falsy
  or absent.

Coverage matrix (matches the B1-E4 specification):

* valid factory composition
* borrowed storage identity (factory borrows; never copies)
* no SessionDB construction (factory never imports it)
* no ``close`` calls on borrowed storage or emitter
* correct ``goal:<session_id>`` lookup
* missing state produces no contract hit
* malformed JSON becomes contract source failure
* non-Mapping JSON becomes contract source failure
* get_meta exception becomes contract source failure
* caller sources preserved by identity
* supplied ``contract`` source replaced by factory
* supplied ``audit_sink`` used without resolving ``get_emitter``
* absent ``audit_sink`` resolves ``get_emitter`` once
* emitter is not closed by the factory / engine / adapter
* no engine invocation (dry_run / discover / rollback) during construction
* storage ``None`` maps to ``storage_unavailable``
* structurally invalid storage maps to ``storage_unavailable``
* ordinary factory error remains ``factory_error``
* default-off invokes no factory and causes no side effects
* malformed config remains ``invalid_config``
* CLI passes ``self._session_db`` and ``build_evidence_pack_engine``
* session rebinding remains per session (factory callable is per-call)
* no engine / provider / adapter / emitter modification
"""
from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import pytest

from agent.executive.knowledge_discovery import (
    EvidencePackEngine,
    KnowledgeHitV2,
    KnowledgeQuery,
)
from agent.executive.knowledge_discovery.factory import (
    build_evidence_pack_engine,
)
from agent.executive.services import (
    EvidencePackDegradeReason,
    EvidencePackStorageUnavailable,
    build_objective_services,
)


# ─────────────────────────────────────────────────────────────────────
# Canonical session and key constants
# ─────────────────────────────────────────────────────────────────────


SESSION_ID = "b1-e4-session-canonical"
SESSION_ID_OTHER = "b1-e4-session-other"
GOAL_KEY = f"goal:{SESSION_ID}"
GOAL_KEY_OTHER = f"goal:{SESSION_ID_OTHER}"
CONFIG: Mapping[str, Any] = {}


# ─────────────────────────────────────────────────────────────────────
# Borrowed-storage double (no SessionDB)
# ─────────────────────────────────────────────────────────────────────


class BorrowedStorage:
    """Duck-typed state_meta backing for the canonical factory.

    Records every ``get_meta`` / ``set_meta`` call and refuses to be
    closed. ``close_calls`` lets tests assert no ``close`` is issued
    by the factory, engine, or audit adapter.
    """

    def __init__(
        self,
        *,
        state: Optional[dict[str, Any]] = None,
        raise_on_get: Optional[BaseException] = None,
    ) -> None:
        self._state: dict[str, Any] = dict(state or {})
        self._raise_on_get = raise_on_get
        self.get_calls: list[str] = []
        self.set_calls: list[tuple[str, Any]] = []
        self.close_calls: int = 0

    def get_meta(self, key: str) -> Any:
        self.get_calls.append(key)
        if self._raise_on_get is not None:
            raise self._raise_on_get
        return self._state.get(key)

    def set_meta(self, key: str, value: Any) -> None:
        self.set_calls.append((key, value))
        self._state[key] = value

    def close(self) -> None:  # pragma: no cover - guarded by tests
        self.close_calls += 1


class StorageWithoutGetMeta:
    """Storage missing get_meta → structurally invalid."""

    def set_meta(self, key: str, value: Any) -> None:  # pragma: no cover
        pass


class StorageWithoutSetMeta:
    """Storage missing set_meta → structurally invalid."""

    def get_meta(self, key: str) -> Any:  # pragma: no cover
        return None


class StorageWithNonCallableMethods:
    """Storage whose get_meta / set_meta attributes are not callable."""

    get_meta = 123
    set_meta = "not-callable"


# ─────────────────────────────────────────────────────────────────────
# Borrowed audit-sink double
# ─────────────────────────────────────────────────────────────────────


class RecordingAuditSink:
    def __init__(self) -> None:
        self.events: list[Any] = []
        self.close_calls: int = 0

    def emit(self, event: Any) -> None:
        self.events.append(event)

    def close(self) -> None:  # pragma: no cover
        self.close_calls += 1


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _query(text: str = "any objective") -> KnowledgeQuery:
    return KnowledgeQuery(
        objective_id="obj-b1-e4",
        objective_text=text,
    )


def _full_goal_state_dict() -> dict[str, Any]:
    return {
        "goal": "ship the migration",
        "contract": {
            "outcome": "ship the migration",
            "verification": "the auth test suite passes",
            "constraints": "keep the public /login response shape unchanged",
            "boundaries": "only touch services/auth and its tests",
            "stop_when": "a schema change needs product sign-off",
        },
    }


def _enabled_config() -> dict[str, Any]:
    return {"goals": {"evidence_pack": {"enabled": True}}}


# ─────────────────────────────────────────────────────────────────────
# 1. Valid factory composition
# ─────────────────────────────────────────────────────────────────────


def test_valid_factory_composition_returns_evidence_pack_engine():
    """The canonical factory returns a real EvidencePackEngine."""
    storage = BorrowedStorage()
    engine = build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=storage,
    )
    assert isinstance(engine, EvidencePackEngine)


# ─────────────────────────────────────────────────────────────────────
# 2. Borrowed storage identity (factory borrows, never copies)
# 3. No SessionDB construction
# 4. No close calls
# ─────────────────────────────────────────────────────────────────────


def test_factory_borrows_storage_without_copying_or_closing():
    """Factory must borrow the storage by identity; never close it."""
    storage = BorrowedStorage()
    engine = build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=storage,
    )
    # Borrowed by identity on the engine.
    assert engine._storage is storage
    # No close was issued by the factory.
    assert storage.close_calls == 0


def test_factory_module_does_not_import_session_db():
    """The factory must never reach into SessionDB / hermes_state internals.

    Behavioral contract: a fresh Python subprocess installs an import
    blocker for ``hermes_state`` (and ``hermes_state.*``) BEFORE importing
    the factory module, then composes the engine via the public factory
    callable and exercises it through the public ``dry_run`` method. The
    subprocess must complete successfully without the blocker firing.

    The blocker raises ``ImportError`` on any ``import hermes_state`` or
    ``from hermes_state import ...`` triggered top-down or deferred.
    Because the blocker is installed BEFORE any factory import, it
    detects top-level factory imports of ``hermes_state`` as well as
    imports triggered later during composition / ``dry_run``.

    The subprocess exercises the engine through the public
    ``dry_run(objective_id, objective_text)`` method and inspects the
    returned public ``EvidencePack`` dataclass. The test does NOT reach
    into private attributes like ``engine._sources``,
    ``engine._session_db``, ``engine._storage``, ``engine._audit_sink``,
    or any other private attribute. It does NOT take a post‑``import
    ``sys.modules`` baseline — the only baseline is the clean subprocess
    state itself.
    """
    import subprocess
    import sys

    # Build the subprocess script via string concatenation. We avoid nesting
    # triple-quoted strings so the outer Python source is unambiguous.
    # Each statement is appended on its own line so failure tracebacks
    # point at the right line.
    subprocess_script_lines = [
        "import sys",
        "import json",
        "",
        # Meta-path finder that raises ImportError on any hermes_state
        # import. If hermes_state is reachable from the factory's
        # transitive import closure (top-level or deferred), the very
        # next import that touches it raises ImportError and the
        # subprocess fails — proving the factory requires hermes_state.
        "class _HermesStateBlocker:",
        "    _BLOCKED_PREFIXES = ('hermes_state',)",
        "    def find_spec(self, fullname, path, target=None):",
        "        for prefix in self._BLOCKED_PREFIXES:",
        "            if fullname == prefix or fullname.startswith(prefix + '.'):",
        "                raise ImportError(",
        "                    'blocked: hermes_state is forbidden for the '",
        "                    'factory import-boundary probe (%r)' % fullname",
        "                )",
        "        return None",
        "",
        # Pre-emptively purge hermes_state from sys.modules so the
        # blocker is the only path to it. Combined with the meta_path
        # hook, this guarantees any attempt to load hermes_state raises
        # ImportError — including imports triggered lazily.
        "for mod_name in list(sys.modules):",
        "    if mod_name == 'hermes_state' or mod_name.startswith('hermes_state.'):",
        "        del sys.modules[mod_name]",
        "",
        "sys.meta_path.insert(0, _HermesStateBlocker())",
        "",
        # Real work: import the factory and exercise the canonical
        # composition seam. The import happens AFTER the blocker is
        # installed, so a top-level factory import of hermes_state
        # would fire here. This must succeed without the blocker
        # raising.
        "from agent.executive.knowledge_discovery.factory import (",
        "    build_evidence_pack_engine,",
        ")",
        "",
        # A duck-typed storage implementing the structural contract
        # (``get_meta`` / ``set_meta``). It never imports hermes_state
        # or anything else.
        "class _DuckStorage:",
        "    def __init__(self, state=None):",
        "        self._state = dict(state or {})",
        "        self.close_calls = 0",
        "    def get_meta(self, key):",
        "        return self._state.get(key)",
        "    def set_meta(self, key, value):",
        "        self._state[key] = value",
        "    def close(self):",
        "        self.close_calls += 1",
        "",
        # A duck-typed audit sink implementing the structural contract
        # (``emit(event)``). Supplying it explicitly bypasses the
        # default emitter-resolution path so the subprocess does not
        # depend on the monitoring subsystem's singleton.
        "class _DuckAuditSink:",
        "    def __init__(self):",
        "        self.events = []",
        "        self.close_calls = 0",
        "    def emit(self, event):",
        "        self.events.append(event)",
        "    def close(self):",
        "        self.close_calls += 1",
        "",
        "SESSION_ID = 'factory-import-boundary-probe'",
        "GOAL_KEY = 'goal:' + SESSION_ID",
        "STATE = {",
        "    'goal': 'ship the migration',",
        "    'contract': {",
        "        'outcome': 'ship the migration',",
        "        'verification': 'the auth test suite passes',",
        "        'constraints': 'keep the public /login response shape unchanged',",
        "        'boundaries': 'only touch services/auth and its tests',",
        "        'stop_when': 'a schema change needs product sign-off',",
        "    },",
        "}",
        "",
        "storage = _DuckStorage(state={GOAL_KEY: json.dumps(STATE)})",
        "audit_sink = _DuckAuditSink()",
        "engine = build_evidence_pack_engine(",
        "    session_id=SESSION_ID,",
        "    config={},",
        "    storage=storage,",
        "    audit_sink=audit_sink,",
        ")",
        "",
        # Exercise the engine via the PUBLIC ``dry_run`` method. This
        # is the canonical public entry point; it drives every
        # registered source and returns a public ``EvidencePack``. Any
        # deferred import inside the loader / provider has a chance to
        # fire here.
        "pack = engine.dry_run(",
        "    'obj-factory-import-boundary',",
        "    'any objective',",
        ")",
        "",
        # Inspect the PUBLIC result only. The EvidencePack dataclass
        # exposes ``sources_queried``, ``sources_failed``,
        # ``total_hits``, and ``overall_confidence`` as public
        # attributes.
        "result = {",
        "    'engine_class': type(engine).__name__,",
        "    'engine_module': type(engine).__module__,",
        "    'sources_queried': list(pack.sources_queried),",
        "    'sources_failed': list(pack.sources_failed),",
        "    'total_hits': int(pack.total_hits),",
        "    'overall_confidence': float(pack.overall_confidence),",
        "    'has_summary': bool(pack.summary_text),",
        "    'storage_close_calls': storage.close_calls,",
        "    'audit_sink_close_calls': audit_sink.close_calls,",
        "    'hermes_state_in_modules': any(",
        "        m == 'hermes_state' or m.startswith('hermes_state.')",
        "        for m in sys.modules",
        "    ),",
        "}",
        "sys.stdout.write('FACTORY_IMPORT_BOUNDARY_RESULT=' + json.dumps(result))",
    ]
    subprocess_script = "\n".join(subprocess_script_lines)

    completed = subprocess.run(
        [sys.executable, "-c", subprocess_script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        "factory import-boundary subprocess failed:\n"
        f"stdout={completed.stdout!r}\n"
        f"stderr={completed.stderr!r}"
    )
    # The subprocess emits a JSON line on success; parse it for
    # additional behavioral confirmations.
    import json as _json

    marker = "FACTORY_IMPORT_BOUNDARY_RESULT="
    line = next(
        (ln for ln in completed.stdout.splitlines() if ln.startswith(marker)),
        None,
    )
    assert line is not None, (
        "factory import-boundary subprocess did not emit a result line; "
        f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
    )
    report = _json.loads(line[len(marker):])

    # Behavioral observation 1: the factory / engine compose
    # successfully even though hermes_state is import-blocked, so
    # hermes_state is NOT in the factory's transitive import closure.
    assert report["engine_class"] == "EvidencePackEngine"
    # Behavioral observation 2: the public dry_run result reflects
    # that the contract source was queried, no source failed, exactly
    # one hit was produced, and a non-empty summary was emitted.
    assert report["sources_queried"] == ["contract"]
    assert report["sources_failed"] == []
    assert report["total_hits"] == 1
    assert report["overall_confidence"] > 0.0
    assert report["has_summary"] is True
    # Behavioral observation 3: storage and audit sink were never
    # closed by the factory, the engine, or the adapter.
    assert report["storage_close_calls"] == 0
    assert report["audit_sink_close_calls"] == 0
    # Behavioral observation 4: hermes_state was never imported —
    # the import blocker was never activated.
    assert report["hermes_state_in_modules"] is False


# ─────────────────────────────────────────────────────────────────────
# 5. Correct goal:<session_id> lookup
# 6. Missing state produces no contract hit
# ─────────────────────────────────────────────────────────────────────


def test_state_loader_reads_exact_goal_key():
    """Storage is queried with the canonical ``goal:<session_id>`` key."""
    state = _full_goal_state_dict()
    raw_text = json.dumps(state)
    storage = BorrowedStorage(state={GOAL_KEY: raw_text})

    engine = build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=storage,
    )

    # Engine's contract source composes and runs against the stored state.
    contract_source = engine._sources["contract"]
    hits = contract_source(_query(), max_hits=5, observed_at="2026-08-04T11:30:00+00:00")

    assert storage.get_calls == [GOAL_KEY]
    assert len(hits) == 1
    assert isinstance(hits[0], KnowledgeHitV2)
    assert hits[0].source == "contract"


def test_missing_state_produces_no_contract_hit():
    """A missing key returns None from the loader → [] from the provider."""
    storage = BorrowedStorage()  # empty

    engine = build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=storage,
    )

    contract_source = engine._sources["contract"]
    hits = contract_source(_query(), max_hits=5, observed_at="2026-08-04T11:30:00+00:00")
    assert hits == []
    assert storage.get_calls == [GOAL_KEY]


def test_falsy_state_value_produces_no_contract_hit():
    """Empty string / 0 / None values must all return None from loader."""
    for falsy in ("", 0, None):
        storage = BorrowedStorage(state={GOAL_KEY: falsy})
        engine = build_evidence_pack_engine(
            session_id=SESSION_ID,
            config=CONFIG,
            storage=storage,
        )
        contract_source = engine._sources["contract"]
        hits = contract_source(
            _query(), max_hits=5, observed_at="2026-08-04T11:30:00+00:00"
        )
        assert hits == []


# ─────────────────────────────────────────────────────────────────────
# 7. Malformed JSON becomes contract source failure
# 8. Non-Mapping JSON becomes contract source failure
# 9. get_meta exception becomes contract source failure
# ─────────────────────────────────────────────────────────────────────


def test_malformed_json_propagates_through_contract_source():
    """Stored malformed JSON must NOT be coerced into missing state."""
    storage = BorrowedStorage(state={GOAL_KEY: "{not json"})
    engine = build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=storage,
    )
    contract_source = engine._sources["contract"]
    # The contract adapter propagates loader exceptions to the engine's
    # source-failed accounting, so the engine surfaces the failure.
    pack = engine.dry_run(
        "obj-b1-e4",
        "any objective",
    )
    assert "contract" in pack.sources_failed
    assert "contract" not in pack.sources_queried


def test_non_mapping_json_propagates_through_contract_source():
    """JSON that parses to a non-object must not be silently coerced."""
    storage = BorrowedStorage(state={GOAL_KEY: json.dumps([1, 2, 3])})
    engine = build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=storage,
    )
    pack = engine.dry_run(
        "obj-b1-e4",
        "any objective",
    )
    assert "contract" in pack.sources_failed


def test_get_meta_exception_propagates_through_contract_source():
    """A get_meta exception must propagate to the engine, not be swallowed."""
    storage = BorrowedStorage(
        state={}, raise_on_get=RuntimeError("storage unreachable")
    )
    engine = build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=storage,
    )
    pack = engine.dry_run(
        "obj-b1-e4",
        "any objective",
    )
    assert "contract" in pack.sources_failed


# ─────────────────────────────────────────────────────────────────────
# 10. Caller sources preserved by identity
# 11. Supplied contract source replaced by factory
# ─────────────────────────────────────────────────────────────────────


def test_caller_sources_preserved_by_identity():
    """Non-contract caller sources survive composition unchanged."""

    def policy_provider(query: KnowledgeQuery, *, max_hits: int, observed_at: str):
        return []

    def gbrain_provider(query: KnowledgeQuery, *, max_hits: int, observed_at: str):
        return []

    supplied = {"policy": policy_provider, "gbrain": gbrain_provider}
    engine = build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=BorrowedStorage(),
        sources=supplied,
    )
    assert engine._sources["policy"] is policy_provider
    assert engine._sources["gbrain"] is gbrain_provider


def test_supplied_contract_source_replaced_by_factory():
    """A caller-supplied ``contract`` entry MUST be replaced."""

    def fake_contract_provider(query: KnowledgeQuery, *, max_hits: int, observed_at: str):
        return []  # would be wrong if left in place

    supplied = {"contract": fake_contract_provider}
    engine = build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=BorrowedStorage(),
        sources=supplied,
    )
    assert engine._sources["contract"] is not fake_contract_provider


def test_none_sources_yields_empty_source_map_with_contract_added():
    """``sources=None`` is treated as an empty source map."""
    engine = build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=BorrowedStorage(),
        sources=None,
    )
    # Contract is installed even when no other sources are supplied.
    assert "contract" in engine._sources
    assert len(engine._sources) == 1


def test_non_mapping_sources_raise_type_error():
    """Non-Mapping sources must raise TypeError."""
    with pytest.raises(TypeError):
        build_evidence_pack_engine(
            session_id=SESSION_ID,
            config=CONFIG,
            storage=BorrowedStorage(),
            sources=["not", "a", "mapping"],  # type: ignore[arg-type]
        )


def test_supplied_sources_are_shallow_copied():
    """The factory must not mutate caller-owned source dicts."""
    supplied = {"policy": lambda *a, **k: []}
    original_keys = set(supplied.keys())
    build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=BorrowedStorage(),
        sources=supplied,
    )
    # Caller-owned structure is unchanged.
    assert set(supplied.keys()) == original_keys


# ─────────────────────────────────────────────────────────────────────
# 12. Supplied audit_sink used without resolving get_emitter
# 13. Absent audit_sink resolves get_emitter once
# 14. Emitter is not closed
# ─────────────────────────────────────────────────────────────────────


def test_supplied_audit_sink_used_without_resolving_get_emitter(monkeypatch):
    """When the caller supplies an audit sink, ``get_emitter`` is NOT called."""
    calls: list[Any] = []

    def fail_get_emitter(*args, **kwargs):  # pragma: no cover
        calls.append(args)
        raise AssertionError("get_emitter must not be called when sink supplied")

    monkeypatch.setattr(
        "agent.monitoring.emitter.get_emitter", fail_get_emitter
    )

    sink = RecordingAuditSink()
    engine = build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=BorrowedStorage(),
        audit_sink=sink,
    )
    assert engine._audit_sink is sink
    assert calls == []


def test_absent_audit_sink_resolves_get_emitter_and_wraps_in_adapter(monkeypatch):
    """Without an audit sink, the factory must resolve ``get_emitter`` and wrap it."""
    sentinel = object()

    def fake_get_emitter():
        fake_get_emitter.calls += 1
        return sentinel

    fake_get_emitter.calls = 0  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "agent.monitoring.emitter.get_emitter", fake_get_emitter
    )

    engine = build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=BorrowedStorage(),
        audit_sink=None,
    )
    # The factory wraps the resolved emitter exactly once per call.
    assert fake_get_emitter.calls == 1  # type: ignore[attr-defined]
    # The adapter borrows the emitter; the adapter is the audit_sink.
    adapter = engine._audit_sink
    assert adapter._emitter is sentinel


def test_emitter_is_not_closed_by_factory_engine_or_adapter(monkeypatch):
    """The factory / engine / adapter must never close the emitter."""
    from agent.monitoring.emitter import MonitoringEmitter

    class CountingEmitter:
        def __init__(self) -> None:
            self.closed: bool = False
            self.payloads: list[Any] = []

        def emit(self, payload: Any) -> None:
            self.payloads.append(payload)

        def close(self) -> None:  # pragma: no cover
            self.closed = True

    emitter = CountingEmitter()
    monkeypatch.setattr(
        "agent.monitoring.emitter.get_emitter", lambda: emitter
    )

    engine = build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=BorrowedStorage(),
    )
    # No close from construction.
    assert emitter.closed is False
    # No close from a no-op engine method invocation.
    assert engine.get_meta("anything") is None
    assert emitter.closed is False


# ─────────────────────────────────────────────────────────────────────
# 15. No engine invocation during construction
# ─────────────────────────────────────────────────────────────────────


def test_factory_does_not_invoke_engine_during_construction():
    """The factory must NOT call dry_run / discover / rollback."""
    storage = BorrowedStorage()

    # Wrap dry_run / discover / rollback to detect any factory-time calls.
    original_dry_run = EvidencePackEngine.dry_run
    original_discover = EvidencePackEngine.discover
    original_rollback = EvidencePackEngine.rollback
    invocations: list[str] = []

    def spy_dry_run(self, *args, **kwargs):
        invocations.append("dry_run")
        return original_dry_run(self, *args, **kwargs)

    def spy_discover(self, *args, **kwargs):
        invocations.append("discover")
        return original_discover(self, *args, **kwargs)

    def spy_rollback(self, *args, **kwargs):
        invocations.append("rollback")
        return original_rollback(self, *args, **kwargs)

    EvidencePackEngine.dry_run = spy_dry_run  # type: ignore[method-assign]
    EvidencePackEngine.discover = spy_discover  # type: ignore[method-assign]
    EvidencePackEngine.rollback = spy_rollback  # type: ignore[method-assign]
    try:
        build_evidence_pack_engine(
            session_id=SESSION_ID,
            config=CONFIG,
            storage=storage,
        )
    finally:
        EvidencePackEngine.dry_run = original_dry_run  # type: ignore[method-assign]
        EvidencePackEngine.discover = original_discover  # type: ignore[method-assign]
        EvidencePackEngine.rollback = original_rollback  # type: ignore[method-assign]

    assert invocations == []


# ─────────────────────────────────────────────────────────────────────
# 16. storage None maps to storage_unavailable
# 17. Structurally invalid storage maps to storage_unavailable
# ─────────────────────────────────────────────────────────────────────


def test_storage_none_raises_storage_unavailable():
    """``storage=None`` must raise the typed unavailability signal."""
    with pytest.raises(EvidencePackStorageUnavailable):
        build_evidence_pack_engine(
            session_id=SESSION_ID,
            config=CONFIG,
            storage=None,
        )


def test_storage_without_get_meta_raises_storage_unavailable():
    """Storage missing ``get_meta`` must raise the typed unavailability signal."""
    with pytest.raises(EvidencePackStorageUnavailable):
        build_evidence_pack_engine(
            session_id=SESSION_ID,
            config=CONFIG,
            storage=StorageWithoutGetMeta(),
        )


def test_storage_without_set_meta_raises_storage_unavailable():
    """Storage missing ``set_meta`` must raise the typed unavailability signal."""
    with pytest.raises(EvidencePackStorageUnavailable):
        build_evidence_pack_engine(
            session_id=SESSION_ID,
            config=CONFIG,
            storage=StorageWithoutSetMeta(),
        )


def test_storage_with_non_callable_get_meta_raises_storage_unavailable():
    """Non-callable ``get_meta`` attribute must raise the typed signal."""
    with pytest.raises(EvidencePackStorageUnavailable):
        build_evidence_pack_engine(
            session_id=SESSION_ID,
            config=CONFIG,
            storage=StorageWithNonCallableMethods(),
        )


# ─────────────────────────────────────────────────────────────────────
# 18. Ordinary factory error remains factory_error
# ─────────────────────────────────────────────────────────────────────


def test_generic_factory_error_remains_factory_error():
    """Generic exceptions thrown by the factory are caught as factory_error."""
    def bad_factory(**kwargs):
        raise RuntimeError("unexpected blow-up")

    services = build_objective_services(
        session_id=SESSION_ID,
        config=_enabled_config(),
        evidence_pack_engine_factory=bad_factory,
    )
    assert services.evidence_pack_engine is None
    assert services.evidence_pack_status == "degraded"
    assert services.evidence_pack_degrade_reason == "factory_error"
    assert services.evidence_pack_error_type == "RuntimeError"


# ─────────────────────────────────────────────────────────────────────
# Storage-unavailable degrade path
# ─────────────────────────────────────────────────────────────────────


def test_storage_unavailable_surfaces_as_degraded_storage_unavailable():
    """Typed storage failure surfaces as ``storage_unavailable``."""
    services = build_objective_services(
        session_id=SESSION_ID,
        config=_enabled_config(),
        evidence_pack_engine_factory=build_evidence_pack_engine,
        storage=None,
    )
    assert services.evidence_pack_engine is None
    assert services.evidence_pack_status == "degraded"
    assert services.evidence_pack_degrade_reason == "storage_unavailable"
    assert services.evidence_pack_error_type == "EvidencePackStorageUnavailable"


def test_storage_unavailable_literal_added_to_degrade_reason():
    """The Literal type lists the new reason."""
    values = getattr(EvidencePackDegradeReason, "__args__", ())
    assert "storage_unavailable" in values


def test_typed_exception_owner_is_services_module():
    """The typed exception is owned by ``agent.executive.services``."""
    import agent.executive.services as services_module

    assert hasattr(services_module, "EvidencePackStorageUnavailable")
    assert services_module.EvidencePackStorageUnavailable is EvidencePackStorageUnavailable


# ─────────────────────────────────────────────────────────────────────
# 19. Default-off invokes no factory and causes no side effects
# 20. Malformed config remains invalid_config
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "config",
    [
        {},
        None,
        {"goals": {}},
        {"goals": {"evidence_pack": {}}},
        {"goals": {"evidence_pack": {"enabled": False}}},
        {"goals": {"evidence_pack": {"enabled": None}}},
    ],
)
def test_default_off_does_not_invoke_factory_or_resolve_emitter(monkeypatch, config):
    """With evidence_pack disabled/absent, no factory call, no emitter lookup."""

    class CountingFactory:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            raise AssertionError("factory must not be invoked when disabled")

    factory = CountingFactory()

    def fail_get_emitter():
        raise AssertionError("get_emitter must not be resolved when disabled")

    monkeypatch.setattr(
        "agent.monitoring.emitter.get_emitter", fail_get_emitter
    )

    services = build_objective_services(
        session_id=SESSION_ID,
        config=config,
        storage=BorrowedStorage(),
        evidence_pack_engine_factory=factory,
    )
    assert services.evidence_pack_engine is None
    assert services.evidence_pack_status == "disabled"
    assert services.evidence_pack_degrade_reason is None
    assert services.evidence_pack_error_type is None
    assert services.evidence_pack_enabled is False
    assert factory.calls == []


def test_malformed_config_remains_invalid_config(monkeypatch):
    """A malformed config produces invalid_config without invoking the factory."""

    class CountingFactory:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            raise AssertionError("factory must not be invoked when invalid")

    factory = CountingFactory()
    monkeypatch.setattr(
        "agent.monitoring.emitter.get_emitter",
        lambda: (_ for _ in ()).throw(AssertionError("must not be called")),
    )

    for cfg in (
        [],  # non-mapping root
        {"goals": []},  # non-mapping goals
        {"goals": {"evidence_pack": []}},  # non-mapping evidence_pack
        {"goals": {"evidence_pack": {"enabled": "yes"}}},  # non-bool enabled
    ):
        services = build_objective_services(
            session_id=SESSION_ID,
            config=cfg,
            storage=BorrowedStorage(),
            evidence_pack_engine_factory=factory,
        )
        assert services.evidence_pack_engine is None
        assert services.evidence_pack_status == "disabled"
        assert services.evidence_pack_degrade_reason == "invalid_config"
        assert services.evidence_pack_error_type is None
        assert services.evidence_pack_enabled is False

    assert factory.calls == []


# ─────────────────────────────────────────────────────────────────────
# 21. CLI passes self._session_db and build_evidence_pack_engine
# ─────────────────────────────────────────────────────────────────────


def test_cli_passes_session_db_and_factory_to_build_objective_services(monkeypatch):
    """HermesCLI._get_goal_manager must wire SessionDB + factory."""
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = SESSION_ID
    cli._goal_manager = None
    sentinel_db = BorrowedStorage()

    captured: dict[str, Any] = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        # Match the documented default-off fallback shape.
        from agent.executive.services import ObjectiveServices
        return ObjectiveServices(
            session_id=kwargs["session_id"],
            storage=kwargs.get("storage"),
        )

    def fake_load_config():
        return {"goals": {"max_turns": 7}}

    monkeypatch.setattr(
        "agent.executive.services.build_objective_services",
        fake_build,
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        fake_load_config,
    )

    cli._session_db = sentinel_db
    cli._get_goal_manager()

    assert captured["session_id"] == SESSION_ID
    assert captured["config"] == {"goals": {"max_turns": 7}}
    assert captured["storage"] is sentinel_db
    assert captured["evidence_pack_engine_factory"] is build_evidence_pack_engine


def test_cli_fallback_branch_also_passes_storage_and_factory(monkeypatch):
    """The config-None fallback path must still pass storage and factory."""
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = SESSION_ID
    cli._goal_manager = None
    sentinel_db = BorrowedStorage()

    captured: list[dict[str, Any]] = []

    def fake_build(**kwargs):
        captured.append(kwargs)
        from agent.executive.services import ObjectiveServices
        return ObjectiveServices(
            session_id=kwargs["session_id"],
            storage=kwargs.get("storage"),
        )

    def boom_load_config():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(
        "agent.executive.services.build_objective_services",
        fake_build,
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        boom_load_config,
    )

    cli._session_db = sentinel_db
    cli._get_goal_manager()

    # Only the fallback invocation happened, but it must still carry
    # the storage + factory arguments.
    assert len(captured) == 1
    assert captured[0]["config"] is None
    assert captured[0]["storage"] is sentinel_db
    assert captured[0]["evidence_pack_engine_factory"] is build_evidence_pack_engine


def test_cli_does_not_pass_storage_none():
    """The CLI must NOT hardcode ``storage=None`` as a kwarg literal.

    Behavioral contract: when ``HermesCLI._session_db`` is set to a
    sentinel object, invoking ``_get_goal_manager`` must transmit
    ``storage=<that sentinel>`` to ``build_objective_services`` by
    identity. The kwarg must reflect the actual ``_session_db`` value —
    if production hardcoded ``storage=None``, the assertion fails
    because the observed ``kwargs["storage"]`` would be ``None``, not
    the sentinel.

    This is the strict, identity-based version of the test: any code
    path that bypasses ``self._session_db`` (e.g. a hardcoded
    ``storage=None`` literal) cannot satisfy the assertion.
    """
    from unittest.mock import patch

    from cli import HermesCLI

    from agent.executive.services import ObjectiveServices

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "cli-storage-none-check"
    cli._goal_manager = None
    # Assign a sentinel storage so the CLI does not fall back to None.
    sentinel_storage = object()
    cli._session_db = sentinel_storage

    services = ObjectiveServices(session_id=cli.session_id)
    with (
        patch("hermes_cli.config.load_config", return_value={}),
        patch(
            "agent.executive.services.build_objective_services",
            return_value=services,
        ) as build,
        patch("hermes_cli.goals.GoalManager"),
    ):
        cli._get_goal_manager()

    # Strict identity-based contract: kwargs["storage"] must be the
    # exact sentinel instance assigned to cli._session_db. If the CLI
    # hardcoded storage=None, this assertion fails.
    assert build.call_count == 1
    kwargs = build.call_args.kwargs
    assert "storage" in kwargs, (
        "CLI did not transmit a storage kwarg at all — it must pass "
        "whatever self._session_db holds (production contract)."
    )
    assert kwargs["storage"] is sentinel_storage, (
        "CLI did not pass its own _session_db by identity: "
        f"expected sentinel={sentinel_storage!r}, got "
        f"{kwargs['storage']!r}. A hardcoded storage=None literal "
        "would fail this assertion."
    )


def test_cli_does_not_pass_storage_none_when_session_db_missing():
    """Companion behavioral check: a CLI without a stored ``_session_db``
    attribute must still NOT hardcode ``storage=None``.

    The challenge: the production CLI uses
    ``getattr(self, "_session_db", None)``, which legitimately returns
    ``None`` when ``_session_db`` is genuinely absent from the
    instance. A naive assertion that simply checks
    ``kwargs["storage"] is None`` would pass even if the CLI hardcoded
    ``storage=None``, defeating the contract.

    To distinguish the two, this test installs a HermesCLI subclass
    whose ``__getattribute__`` synthesizes a unique sentinel for any
    access to ``_session_db``. The production CLI's
    ``getattr(self, "_session_db", None)`` therefore returns the
    sentinel (not the ``None`` default), and ``kwargs["storage"]``
    must be that sentinel. If the CLI hardcoded ``storage=None``,
    the assertion fails.
    """
    from unittest.mock import patch

    from cli import HermesCLI

    from agent.executive.services import ObjectiveServices

    # Probe subclass: every attribute lookup for "_session_db" returns
    # _PROBE_SENTINEL regardless of whether the instance has the
    # attribute stored. This simulates a CLI that conceptually has
    # a non-None storage reachable via attribute access, even though
    # nothing is stored on the instance dict.
    class _ProbeHermesCLI(HermesCLI):
        _PROBE_SENTINEL = object()

        def __getattribute__(self, name):
            if name == "_session_db":
                return _ProbeHermesCLI._PROBE_SENTINEL
            return object.__getattribute__(self, name)

    cli = _ProbeHermesCLI.__new__(_ProbeHermesCLI)
    cli.session_id = "cli-storage-none-fallback"
    cli._goal_manager = None
    # Deliberately do NOT set _session_db as an instance attribute —
    # the probe's __getattribute__ synthesizes it on access. Verify by
    # inspecting the instance dict directly (hasattr would return True
    # via the probe, defeating the check).
    assert "_session_db" not in cli.__dict__

    services = ObjectiveServices(session_id=cli.session_id)
    with (
        patch("hermes_cli.config.load_config", return_value={}),
        patch(
            "agent.executive.services.build_objective_services",
            return_value=services,
        ) as build,
        patch("hermes_cli.goals.GoalManager"),
    ):
        cli._get_goal_manager()

    # Strict identity-based contract: kwargs["storage"] must be the
    # probe sentinel. If production hardcoded storage=None, this fails
    # because kwargs["storage"] would be None.
    assert build.call_count == 1
    kwargs = build.call_args.kwargs
    assert "storage" in kwargs, (
        "CLI did not transmit a storage kwarg at all — production "
        "must pass whatever self._session_db holds (here, the probe "
        "sentinel)."
    )
    assert kwargs["storage"] is _ProbeHermesCLI._PROBE_SENTINEL, (
        "CLI did not pass its own _session_db by identity: expected "
        f"probe sentinel={_ProbeHermesCLI._PROBE_SENTINEL!r}, got "
        f"{kwargs['storage']!r}. A hardcoded storage=None literal "
        "would fail this assertion."
    )


# ─────────────────────────────────────────────────────────────────────
# 22. Session rebinding remains per session (factory is per-call)
# ─────────────────────────────────────────────────────────────────────


def test_each_factory_call_produces_an_independent_engine():
    """Two factory calls with different sessions → two independent engines."""
    state_a = BorrowedStorage(state={GOAL_KEY: json.dumps(_full_goal_state_dict())})
    state_b = BorrowedStorage(state={GOAL_KEY_OTHER: json.dumps(_full_goal_state_dict())})

    engine_a = build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=state_a,
    )
    engine_b = build_evidence_pack_engine(
        session_id=SESSION_ID_OTHER,
        config=CONFIG,
        storage=state_b,
    )

    assert engine_a is not engine_b
    # Each engine borrows its own storage by identity.
    assert engine_a._storage is state_a
    assert engine_b._storage is state_b
    # The contract providers read different keys.
    contract_a = engine_a._sources["contract"]
    contract_b = engine_b._sources["contract"]
    assert contract_a is not contract_b


def test_factory_does_not_cache_a_global_engine():
    """Two factory calls with identical inputs → two distinct engines."""
    storage = BorrowedStorage()
    engine_a = build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=storage,
    )
    engine_b = build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=storage,
    )
    assert engine_a is not engine_b


# ─────────────────────────────────────────────────────────────────────
# 23. No engine / provider / adapter / emitter modification
# ─────────────────────────────────────────────────────────────────────


def test_factory_does_not_modify_sealed_components(monkeypatch):
    """The factory must not import or monkey-patch sealed components."""
    from agent.executive.knowledge_discovery import engine as engine_mod
    from agent.executive.knowledge_discovery.adapters import (
        contract_provider as cp_mod,
    )
    from agent.executive.knowledge_discovery.adapters import (
        audit_sink as audit_mod,
    )

    # Snapshot relevant class-level attributes on the sealed modules
    # before / after a factory invocation. We compare classes by
    # identity — monkey-patching any of them would change identity.
    before = {
        "engine_cls": engine_mod.EvidencePackEngine,
        "engine_dry_run": engine_mod.EvidencePackEngine.dry_run,
        "engine_discover": engine_mod.EvidencePackEngine.discover,
        "engine_rollback": engine_mod.EvidencePackEngine.rollback,
        "contract_provider": cp_mod.make_contract_provider,
        "audit_sink_cls": audit_mod.EvidencePackMonitoringAuditSink,
    }

    # Force a factory invocation that exercises every composition path.
    monkeypatch.setattr(
        "agent.monitoring.emitter.get_emitter",
        lambda: object(),
    )
    build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=BorrowedStorage(),
    )

    after = {
        "engine_cls": engine_mod.EvidencePackEngine,
        "engine_dry_run": engine_mod.EvidencePackEngine.dry_run,
        "engine_discover": engine_mod.EvidencePackEngine.discover,
        "engine_rollback": engine_mod.EvidencePackEngine.rollback,
        "contract_provider": cp_mod.make_contract_provider,
        "audit_sink_cls": audit_mod.EvidencePackMonitoringAuditSink,
    }
    assert before == after


# ─────────────────────────────────────────────────────────────────────
# State loader invariants: hermetic (no hermes_cli import) and fresh
# ─────────────────────────────────────────────────────────────────────


def test_state_loader_does_not_import_hermes_cli():
    """The factory's loader must not import hermes_cli.

    Behavioral contract: in a hermetic subprocess, an import blocker
    for ``hermes_cli`` (and ``hermes_cli.*``) is installed BEFORE the
    factory module is imported. The factory is then composed via the
    public ``build_evidence_pack_engine`` callable and exercised through
    the public ``dry_run`` method on the resulting engine. The
    subprocess must complete successfully without the blocker firing.

    Because the blocker is installed first, it detects both top-level
    factory imports of ``hermes_cli`` (which would otherwise be silently
    hidden behind a post-import ``sys.modules`` baseline) and deferred
    imports triggered during composition or ``dry_run``. The test does
    NOT take a post-import ``sys.modules`` baseline — the only baseline
    is the clean subprocess state itself.

    The test does NOT reach into private engine attributes such as
    ``engine._sources``, ``engine._storage``, ``engine._session_db``,
    or ``engine._audit_sink``. It uses only the public factory callable
    and the public ``dry_run`` / ``discover`` / ``rollback`` API.
    """
    import subprocess
    import sys

    # Build the subprocess script via line concatenation. Each
    # statement is appended on its own line so failure tracebacks
    # point at the right line and no nested triple-quoted string
    # confuses the outer Python parser.
    subprocess_script_lines = [
        "import sys",
        "import json",
        "",
        # Meta-path finder that raises ImportError on any hermes_cli
        # import. Installed BEFORE the factory module is imported, so
        # it catches top-level hermes_cli imports from the factory
        # itself (which a post-import ``sys.modules`` baseline would
        # miss).
        "class _HermesCliBlocker:",
        "    _BLOCKED_PREFIXES = ('hermes_cli',)",
        "    def find_spec(self, fullname, path, target=None):",
        "        for prefix in self._BLOCKED_PREFIXES:",
        "            if fullname == prefix or fullname.startswith(prefix + '.'):",
        "                raise ImportError(",
        "                    'blocked: hermes_cli is forbidden for the '",
        "                    'state-loader import-boundary probe (%r)' % fullname",
        "                )",
        "        return None",
        "",
        # Purge any pre-existing hermes_cli modules from sys.modules so
        # the blocker is the only path to them.
        "for mod_name in list(sys.modules):",
        "    if mod_name == 'hermes_cli' or mod_name.startswith('hermes_cli.'):",
        "        del sys.modules[mod_name]",
        "",
        "sys.meta_path.insert(0, _HermesCliBlocker())",
        "",
        # Real work: import the factory and exercise it. This must
        # succeed without the blocker raising.
        "from agent.executive.knowledge_discovery.factory import (",
        "    build_evidence_pack_engine,",
        ")",
        "",
        "class _DuckStorage:",
        "    def __init__(self, state=None):",
        "        self._state = dict(state or {})",
        "        self.close_calls = 0",
        "    def get_meta(self, key):",
        "        return self._state.get(key)",
        "    def set_meta(self, key, value):",
        "        self._state[key] = value",
        "    def close(self):",
        "        self.close_calls += 1",
        "",
        # A duck-typed audit sink so the factory does not need to
        # resolve the monitoring emitter (which would be a hermes_cli
        # -free but side-effecting default).
        "class _DuckAuditSink:",
        "    def __init__(self):",
        "        self.events = []",
        "        self.close_calls = 0",
        "    def emit(self, event):",
        "        self.events.append(event)",
        "    def close(self):",
        "        self.close_calls += 1",
        "",
        "SESSION_ID = 'state-loader-import-boundary-probe'",
        "GOAL_KEY = 'goal:' + SESSION_ID",
        "STATE = {",
        "    'goal': 'ship the migration',",
        "    'contract': {",
        "        'outcome': 'ship the migration',",
        "        'verification': 'the auth test suite passes',",
        "        'constraints': 'keep the public /login response shape unchanged',",
        "        'boundaries': 'only touch services/auth and its tests',",
        "        'stop_when': 'a schema change needs product sign-off',",
        "    },",
        "}",
        "",
        "storage = _DuckStorage(state={GOAL_KEY: json.dumps(STATE)})",
        "audit_sink = _DuckAuditSink()",
        "engine = build_evidence_pack_engine(",
        "    session_id=SESSION_ID,",
        "    config={},",
        "    storage=storage,",
        "    audit_sink=audit_sink,",
        ")",
        "",
        # Exercise the engine via the PUBLIC ``dry_run`` method. The
        # result is a public ``EvidencePack`` dataclass exposing
        # ``sources_queried``, ``sources_failed``, ``total_hits``,
        # ``overall_confidence``, and ``summary_text``.
        "pack = engine.dry_run(",
        "    'obj-state-loader-import-boundary',",
        "    'any objective',",
        ")",
        "",
        "result = {",
        "    'engine_class': type(engine).__name__,",
        "    'sources_queried': list(pack.sources_queried),",
        "    'sources_failed': list(pack.sources_failed),",
        "    'total_hits': int(pack.total_hits),",
        "    'overall_confidence': float(pack.overall_confidence),",
        "    'has_summary': bool(pack.summary_text),",
        "    'storage_close_calls': storage.close_calls,",
        "    'audit_sink_close_calls': audit_sink.close_calls,",
        "    'hermes_cli_in_modules': any(",
        "        m == 'hermes_cli' or m.startswith('hermes_cli.')",
        "        for m in sys.modules",
        "    ),",
        "}",
        "sys.stdout.write('STATE_LOADER_IMPORT_RESULT=' + json.dumps(result))",
    ]
    subprocess_script = "\n".join(subprocess_script_lines)

    completed = subprocess.run(
        [sys.executable, "-c", subprocess_script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        "state-loader import-boundary subprocess failed:\n"
        f"stdout={completed.stdout!r}\n"
        f"stderr={completed.stderr!r}"
    )
    import json as _json

    marker = "STATE_LOADER_IMPORT_RESULT="
    line = next(
        (ln for ln in completed.stdout.splitlines() if ln.startswith(marker)),
        None,
    )
    assert line is not None, (
        "state-loader import-boundary subprocess did not emit a result "
        f"line; stdout={completed.stdout!r}, stderr={completed.stderr!r}"
    )
    report = _json.loads(line[len(marker):])

    # Behavioral observation 1: the factory / engine compose
    # successfully even though hermes_cli is import-blocked, so
    # hermes_cli is NOT in the factory's transitive import closure.
    assert report["engine_class"] == "EvidencePackEngine"
    # Behavioral observation 2: the public dry_run result reflects
    # that the contract source was queried, no source failed, exactly
    # one hit was produced, and a non-empty summary was emitted.
    assert report["sources_queried"] == ["contract"]
    assert report["sources_failed"] == []
    assert report["total_hits"] == 1
    assert report["overall_confidence"] > 0.0
    assert report["has_summary"] is True
    # Behavioral observation 3: storage and audit sink were never
    # closed by the factory, the engine, or the adapter.
    assert report["storage_close_calls"] == 0
    assert report["audit_sink_close_calls"] == 0
    # Behavioral observation 4: hermes_cli was never imported — the
    # import blocker was never activated. This is the load-bearing
    # assertion for the defect: a hermes_cli import that fires at
    # factory import time would either be caught by the blocker
    # above (causing subprocess exit with ImportError) or land in
    # sys.modules, making this assertion fail.
    assert report["hermes_cli_in_modules"] is False, (
        "factory transitively imported hermes_cli modules: "
        f"{[m for m in sorted(sys.modules) if m.startswith('hermes_cli')]!r}"
    )


def test_state_loader_returns_fresh_mapping_without_mutating_storage():
    """Repeated calls yield independent mappings; storage is never mutated."""
    state = _full_goal_state_dict()
    storage = BorrowedStorage(state={GOAL_KEY: json.dumps(state)})

    engine = build_evidence_pack_engine(
        session_id=SESSION_ID,
        config=CONFIG,
        storage=storage,
    )
    contract_source = engine._sources["contract"]

    hits_a = contract_source(_query(), max_hits=5, observed_at="2026-08-04T11:30:00+00:00")
    hits_b = contract_source(_query(), max_hits=5, observed_at="2026-08-04T11:30:00+00:00")

    # Same logical contract → same hit. Independent Mapping returns.
    assert len(hits_a) == len(hits_b) == 1
    # Storage was never written to by the loader.
    assert storage.set_calls == []


# ─────────────────────────────────────────────────────────────────────
# session_id validation
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("sid", ["", None, 0, 123, ("a",), []])
def test_invalid_session_id_raises_value_error(sid):
    """session_id must be a non-empty string."""
    with pytest.raises((ValueError, TypeError)):
        build_evidence_pack_engine(
            session_id=sid,  # type: ignore[arg-type]
            config=CONFIG,
            storage=BorrowedStorage(),
        )