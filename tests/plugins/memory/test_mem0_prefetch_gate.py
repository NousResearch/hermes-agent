"""mem0 recall-prefetch gate: interactive prefetches, non-interactive does not.

Runnable directly (``python3 tests/plugins/memory/test_mem0_prefetch_gate.py``) or under pytest.

Why this exists: the prefetched recall block is injected into the transcript and re-sent on
EVERY later call of that session. A cron/subagent run has no user question for recall to answer,
so the round-trip is pure cost. The gate must fire for interactive turns and stay silent for
autonomous ones — and stay overridable, because "no recall in cron" is a default, not a law.

The silence case is the important one: an unset/blank setting that silently switched the gate
OFF (matching the literal string "None") shipped once during development and was caught here.
"""
from __future__ import annotations

from typing import Any, Tuple

from plugins.memory.mem0 import (
    _PREFETCH_SKIP_CONTEXTS,
    Mem0MemoryProvider,
    _parse_skip_contexts,
)

_FACT = "user prefers terse outcome-first reports"


class _StubBackend:
    """Records every search so a test can prove NO call was made, not just an empty body."""

    def __init__(self):
        self.queries: list[str] = []

    def search(self, query, filters=None, top_k=10, rerank=False):
        self.queries.append(query)
        return [{"memory": _FACT}]


def _provider(agent_context="primary", skip=None) -> Tuple[Mem0MemoryProvider, _StubBackend]:
    """A provider wired to a stub backend instead of hitting the Mem0 API."""
    p = Mem0MemoryProvider()
    stub = _StubBackend()
    p._backend = stub  # type: ignore[assignment]  # stand-in for a real backend in tests
    p._agent_context = agent_context
    p._prefetch_skip_contexts = _parse_skip_contexts(skip)
    return p, stub


def test_interactive_turn_prefetches():
    q = "what did we decide about the schema?"
    p, stub = _provider(agent_context="primary")
    p.on_turn_start(1, q)
    assert _FACT in p.prefetch(q)
    assert stub.queries == [q]


def test_non_interactive_contexts_make_no_mem0_call():
    q = "run the nightly ingest"
    for ctx in ("cron", "subagent", "flush"):
        p, stub = _provider(agent_context=ctx)
        p.on_turn_start(1, q)
        assert p.prefetch(q) == "", f"{ctx} should inject nothing"
        assert stub.queries == [], f"{ctx} should not call mem0 at all"


def test_gate_is_on_by_default():
    """The regression that motivated this file: unset must mean SKIP, not "prefetch everywhere"."""
    assert _parse_skip_contexts(None) == _PREFETCH_SKIP_CONTEXTS
    assert _parse_skip_contexts("") == _PREFETCH_SKIP_CONTEXTS
    assert _parse_skip_contexts([]) == _PREFETCH_SKIP_CONTEXTS


def test_missing_agent_context_defaults_to_interactive():
    """Fail-open for callers that predate the gate: an unknown context still prefetches."""
    q = "hello there friend"
    p, stub = _provider(agent_context="primary")
    assert _FACT in p.prefetch(q)


def test_override_none_prefetches_in_every_context():
    q = "run the nightly ingest"
    p, stub = _provider(agent_context="cron", skip="none")
    p.on_turn_start(1, q)
    assert _FACT in p.prefetch(q)
    assert stub.queries == [q]


def test_override_custom_list():
    q = "run the nightly ingest"
    p, _stub = _provider(agent_context="cron", skip="cron")
    p.on_turn_start(1, q)
    assert p.prefetch(q) == ""


def test_parse_skip_contexts_forms():
    assert _parse_skip_contexts(["CRON ", "subagent"]) == frozenset({"cron", "subagent"})
    assert _parse_skip_contexts("cron,flush") == frozenset({"cron", "flush"})
    assert _parse_skip_contexts("none") == frozenset()
    assert _parse_skip_contexts(" subagent ") == frozenset({"subagent"})


def test_typo_falls_back_to_default_not_silent_off():
    """Reviewer finding: a typo like 'crom' parsed to junk that matched nothing, silently
    switching the gate OFF. Unrecognized names must warn-and-fall-back to the default."""
    assert _parse_skip_contexts("crom") == _PREFETCH_SKIP_CONTEXTS
    assert _parse_skip_contexts(7) == _PREFETCH_SKIP_CONTEXTS          # int -> junk tokens
    assert _parse_skip_contexts({"a": 1}) == _PREFETCH_SKIP_CONTEXTS   # dict -> junk tokens
    assert _parse_skip_contexts("none,cron") == frozenset({"cron"})    # 'none' only counts alone


def test_gate_is_case_insensitive_on_context():
    """Reviewer finding: 'Cron' bypassed the gate because the comparison was verbatim."""
    q = "run the nightly ingest"
    p, stub = _provider(agent_context="Cron")
    p.on_turn_start(1, q)
    assert p.prefetch(q) == ""
    assert stub.queries == []


if __name__ == "__main__":
    import traceback

    _tests: list[Any] = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    _failed = 0
    for _t in _tests:
        try:
            _t()
            print(f"  PASS  {_t.__name__}")
        except Exception:
            _failed += 1
            print(f"  FAIL  {_t.__name__}")
            traceback.print_exc()
    print(f"\n{len(_tests) - _failed}/{len(_tests)} passed")
    raise SystemExit(1 if _failed else 0)
