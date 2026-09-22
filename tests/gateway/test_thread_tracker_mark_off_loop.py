"""The thread-participation tracker's persist must not block the event loop.

``gateway/platforms/helpers.ThreadParticipationTracker._save`` ends in
``atomic_json_write`` -> ``os.replace``, whose duration is unbounded under
filesystem pressure.  Every caller sits on an inbound-message coroutine --
Matrix ``_resolve_message_context`` and ``create_handoff_thread``, Discord
``_handle_message`` (x2) and ``_handle_thread_create_slash`` -- so that rename
was paid inline on the loop, stalling every other adapter's polling and every
in-flight turn for as long as it took.

The fix is a CHOKE-POINT one rather than five call-site ones: ``mark_async``
offloads only the persist via ``asyncio.to_thread``, and all five coroutine
call sites await it.  ``mark`` keeps its exact synchronous contract for the
non-loop callers and tests.

These tests are about OBSERVABLE BEHAVIOUR -- can the loop keep running while
the persist is in its slow path -- not about which API was called.  There are no
wall-clock thresholds: the rename is held open on a real barrier for as long as
the assertion needs, and a companion gate-proof forces the pre-fix inline shape
and asserts the very same barrier DOES starve the loop, so the liveness test
cannot pass vacuously.

They also pin the property the off-loop move would otherwise silently break.
Moving the persist to a worker thread removes the accidental serialization the
event loop used to provide, so the check/insert/trim/write sequence can now
interleave across two marks and lose an entry.  ``atomic_json_write`` makes each
write atomic; it does not make the sequence atomic.

And they pin the read-after-mark contract: the adapters gate on
``thread_id in self._threads`` immediately after marking, so the in-memory half
of ``mark_async`` has to be synchronous even though the persist is not.
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import os
import threading
from pathlib import Path

import pytest

from gateway.platforms import helpers
from gateway.platforms.helpers import ThreadParticipationTracker


@pytest.fixture()
def tracker(tmp_path, monkeypatch):
    """A tracker whose state file lives in an isolated directory."""
    path = tmp_path / "matrix_threads.json"
    monkeypatch.setattr(
        ThreadParticipationTracker, "_state_path", lambda self: path, raising=True
    )
    obj = ThreadParticipationTracker("matrix")
    obj.state_path = path  # convenience for assertions
    return obj


class _HeldReplace:
    """Replace ``os.replace`` with one that blocks until released."""

    def __init__(self, monkeypatch):
        self._gate = threading.Event()
        self._entered = threading.Event()
        self._real = os.replace
        monkeypatch.setattr(os, "replace", self._blocking, raising=True)

    def _blocking(self, src, dst, *a, **kw):
        self._entered.set()
        self._gate.wait(timeout=10.0)
        return self._real(src, dst, *a, **kw)

    def wait_until_entered(self, timeout=5.0):
        return self._entered.wait(timeout)

    def release(self):
        self._gate.set()


# ---------------------------------------------------------------------------
# Liveness: the loop keeps running while the persist is stalled.
# ---------------------------------------------------------------------------


def test_the_persist_does_not_block_the_loop(tracker, monkeypatch):
    """A stalled rename must not stop the loop from running other tasks.

    No stopwatch.  The witness is ORDERING, not elapsed time: the rename is
    held open on a barrier that only a background timer releases, and the
    assertion is that the sibling task ticked BEFORE that release.  On the
    pre-fix inline shape the loop is stuck inside the rename, so the sibling
    cannot possibly tick until the release happens -- which is exactly what
    ``test_gate_proof_the_sync_form_does_block_the_loop`` demonstrates with the
    identical barrier.
    """

    async def scenario():
        held = _HeldReplace(monkeypatch)
        released = threading.Event()
        order: dict[str, bool] = {}

        async def sibling():
            await asyncio.sleep(0)
            # Captured AT TICK TIME: was the rename still being held?
            order["ticked_before_release"] = not released.is_set()

        mark = asyncio.create_task(tracker.mark_async("!room:example.org"))
        sibling_task = asyncio.create_task(sibling())

        def _release_later():
            released.set()
            held.release()

        releaser = threading.Timer(1.0, _release_later)
        releaser.start()
        try:
            await asyncio.wait_for(mark, timeout=10.0)
        finally:
            releaser.cancel()

        await sibling_task
        assert held.wait_until_entered(), "the persist never reached os.replace"
        assert "ticked_before_release" in order, "the sibling task never ran"
        assert order["ticked_before_release"], (
            "the loop did not advance the sibling task until the held rename "
            "was released -- the persist is still blocking the event loop"
        )

        # Durability is preserved, not traded away for liveness.
        assert json.loads(tracker.state_path.read_text(encoding="utf-8")) == [
            "!room:example.org"
        ]

    asyncio.run(scenario())


def test_gate_proof_the_sync_form_does_block_the_loop(tracker, monkeypatch):
    """The barrier above is real: the PRE-FIX shape starves the loop with it.

    This is what stops the liveness test from passing vacuously.  Same tracker,
    same held rename, same sibling -- but calling the blocking ``mark``
    directly from the coroutine, as the Matrix/Discord sites used to.  The
    sibling must NOT have ticked before the release.
    """

    async def scenario():
        held = _HeldReplace(monkeypatch)
        released = threading.Event()
        order: dict[str, bool] = {}
        started = asyncio.Event()

        async def sibling():
            started.set()
            await asyncio.sleep(0)
            order["ticked_before_release"] = not released.is_set()

        asyncio.create_task(sibling())
        # Let the sibling START and park on its own yield point, so the only
        # thing standing between it and its tick is the loop itself.  Without
        # this the task would simply never have begun, and "it did not tick"
        # would prove nothing.
        await started.wait()

        def _release_later():
            released.set()
            held.release()

        releaser = threading.Timer(0.75, _release_later)
        releaser.start()
        try:
            # The pre-fix call shape, inline on the loop.
            tracker.mark("!room:example.org")
        finally:
            releaser.cancel()

        await asyncio.sleep(0)
        assert held.wait_until_entered(), "the persist never reached os.replace"
        assert "ticked_before_release" in order, "the sibling task never resumed"
        assert order["ticked_before_release"] is False, (
            "the sibling task ticked while the INLINE rename was held -- the "
            "barrier is not actually blocking, so the liveness test above "
            "would pass vacuously"
        )

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# The concurrency the off-loop move introduces.
# ---------------------------------------------------------------------------


def test_two_concurrent_marks_do_not_lose_an_entry(tracker, monkeypatch):
    """Off-loop marks run on worker threads, so the sequence must be locked.

    The event loop used to serialize every caller by accident.  With the
    persist on ``asyncio.to_thread`` two marks genuinely overlap, and a
    check/insert/trim/write sequence without a lock drops one of the two.  The
    oracle is the DURABLE FILE, not the in-memory dict.
    """

    entered = threading.Barrier(2, timeout=10.0)
    real_write = helpers.atomic_json_write

    def _synchronised_write(path, payload, *a, **kw):
        # Force maximum overlap: neither write proceeds until both have begun.
        try:
            entered.wait()
        except threading.BrokenBarrierError:  # pragma: no cover - timeout path
            pass
        return real_write(path, payload, *a, **kw)

    monkeypatch.setattr(helpers, "atomic_json_write", _synchronised_write, raising=True)

    async def scenario():
        await asyncio.gather(
            tracker.mark_async("!a:example.org"),
            tracker.mark_async("!b:example.org"),
        )

    asyncio.run(scenario())

    persisted = json.loads(tracker.state_path.read_text(encoding="utf-8"))
    assert sorted(persisted) == ["!a:example.org", "!b:example.org"], (
        "a concurrent mark was lost from the DURABLE file: "
        f"{persisted!r}. atomic_json_write makes each write atomic; it does "
        "not make check/insert/trim/write atomic."
    )


def test_a_mark_is_visible_in_memory_before_the_persist_completes(tracker, monkeypatch):
    """``thread_id in tracker`` must be true before the persist runs AT ALL.

    Both adapters gate on membership right after marking
    (``in_bot_thread = bool(thread_id and thread_id in self._threads)``), so
    deferring the in-memory insert to the worker thread alongside the write
    would reopen the mention-gating hole the tracker exists to close -- and
    would make membership depend on executor availability.

    The oracle is a handoff that NEVER RUNS: ``asyncio.to_thread`` is replaced
    by a coroutine that parks forever without ever invoking the callable.  That
    is what makes this non-vacuous.  Merely waiting for the worker to reach the
    rename -- the obvious shape -- also observes a True under the deferred
    implementation, because a deferred insert still happens before the write;
    this one cannot.
    """
    never_ran = asyncio.Event()

    async def _never(fn, *a, **kw):
        never_ran.set()
        await asyncio.Event().wait()  # park forever; fn is never called

    monkeypatch.setattr(asyncio, "to_thread", _never, raising=True)

    async def scenario():
        mark = asyncio.create_task(tracker.mark_async("!first:example.org"))
        await asyncio.wait_for(never_ran.wait(), timeout=5.0)
        try:
            assert "!first:example.org" in tracker, (
                "the mark is not in memory once the persist has been handed "
                "off -- the in-memory insert was deferred onto the worker "
                "thread, so membership now depends on executor availability "
                "and mention gating can re-prompt for an @mention in a thread "
                "the bot just joined"
            )
        finally:
            mark.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await mark

    asyncio.run(scenario())


def test_marking_the_same_thread_twice_persists_once(tracker, monkeypatch):
    """The dedupe short-circuit survives the async form.

    ``mark``'s "already known -> no write" behaviour is what keeps this off the
    hot path at all; losing it would put a rename on every single message in a
    tracked thread.
    """
    writes: list[list[str]] = []
    real_write = helpers.atomic_json_write

    def _counting_write(path, payload, *a, **kw):
        writes.append(list(payload))
        return real_write(path, payload, *a, **kw)

    monkeypatch.setattr(helpers, "atomic_json_write", _counting_write, raising=True)

    async def scenario():
        await tracker.mark_async("!same:example.org")
        await tracker.mark_async("!same:example.org")
        await tracker.mark_async("!same:example.org")

    asyncio.run(scenario())
    assert len(writes) == 1, (
        f"a repeat mark still persisted: {len(writes)} writes, {writes!r}. "
        "Every message in a tracked thread would pay a rename."
    )


# ---------------------------------------------------------------------------
# Class sweep: enforce the invariant, do not enumerate the sites.
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _adapter_source_files() -> list[Path]:
    repo = _repo_root()
    out: list[Path] = []
    for root in ("gateway", "plugins"):
        out.extend(sorted((repo / root).rglob("*.py")))
    return out


class _CoroutineCallVisitor(ast.NodeVisitor):
    """Collect ``<tracker>.mark(...)`` calls made from inside a coroutine.

    Tracks the enclosing function so a hit inside a plain ``def`` (which is
    allowed to block) is not reported.  Nested plain ``def``s inside a
    coroutine are treated as non-coroutine context, matching the runtime: such
    a helper is only blocking when something calls it, and that call site is
    itself visited.

    Also collects every call to the ASYNC form and whether it is awaited.  An
    un-awaited ``mark_async(...)`` never runs at all: the thread is silently not
    recorded, so mention gating re-prompts forever in a thread the bot joined,
    and -- because the in-memory insert lives inside the coroutine too -- not
    even the process-local state is updated.  The blocking form is gone from
    that call site, so the sweep above cannot see it.
    """

    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.hits: list[str] = []
        self.async_calls: list[str] = []
        self.unawaited_async_calls: list[str] = []
        self._depth = 0
        self._awaited: set[int] = set()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        saved, self._depth = self._depth, 0
        self.generic_visit(node)
        self._depth = saved

    def visit_Await(self, node: ast.Await) -> None:
        self._awaited.add(id(node.value))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        # Attribute-name match on ``_threads``: the tracker is always bound as
        # ``self._threads`` by convention (see the class docstring's usage
        # block), and matching on the bare method name ``mark`` would collide
        # with unrelated ``.mark()`` methods across the tree.
        if (
            isinstance(func, ast.Attribute)
            and func.attr in ("mark", "mark_async")
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "_threads"
        ):
            where = f"{self.rel}:{node.lineno}"
            if func.attr == "mark":
                if self._depth > 0:
                    self.hits.append(where)
            else:
                self.async_calls.append(where)
                if id(node) not in self._awaited:
                    self.unawaited_async_calls.append(where)
        self.generic_visit(node)


def test_no_coroutine_calls_the_blocking_thread_mark():
    """No ``async def`` may call the BLOCKING ``mark`` directly.

    This is the invariant, not a list: a newly added adapter that calls the
    sync form from its inbound handler fails here without anyone remembering to
    update an inventory.  ``mark`` itself stays public for the non-loop callers
    and the existing tests, so it cannot simply be deleted.

    SCOPE, stated exactly.  This is a LEXICAL sweep: it flags a call written
    inside an ``async def`` body.  A coroutine that reaches ``mark``
    INDIRECTLY -- through a plain ``def`` helper that itself calls it -- is not
    flagged here; catching that transitive class needs a call-graph walk, which
    this gate deliberately does not attempt.
    """
    repo = _repo_root()
    hits: list[str] = []
    unawaited: list[str] = []
    for path in _adapter_source_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        visitor = _CoroutineCallVisitor(str(path.relative_to(repo)))
        visitor.visit(tree)
        hits.extend(visitor.hits)
        unawaited.extend(visitor.unawaited_async_calls)

    assert not hits, (
        "coroutine(s) call the BLOCKING ThreadParticipationTracker.mark "
        "directly. That ends in atomic_json_write -> os.replace inline on the "
        "event loop, stalling every adapter and every in-flight turn for as "
        "long as the rename takes. Use `await self._threads.mark_async(...)`.\n"
        + "\n".join(f"  {h}" for h in hits)
    )

    assert not unawaited, (
        "call(s) to _threads.mark_async are NOT awaited. The call then "
        "evaluates to a coroutine object that is never run: the thread is "
        "neither persisted nor recorded in memory, so mention gating keeps "
        "demanding an @mention in a thread the bot has already joined -- and "
        "the sweep above cannot see it, because the blocking form is gone from "
        "the call site.\n"
        + "\n".join(f"  {h}" for h in unawaited)
    )


def test_the_sweep_is_not_vacuous():
    """The sweep must actually parse adapter code and see the async form.

    A green sweep over an empty file list, or over a tree where the async form
    does not exist, proves nothing.
    """
    repo = _repo_root()
    files = _adapter_source_files()
    assert len(files) >= 20, f"scanned only {len(files)} files; the glob is broken"

    async_call_sites: list[str] = []
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        visitor = _CoroutineCallVisitor(str(path.relative_to(repo)))
        visitor.visit(tree)
        async_call_sites.extend(visitor.async_calls)

    # Measured on this tree: 2 in the Matrix adapter, 3 in the Discord adapter.
    assert len(async_call_sites) >= 5, (
        "the sweep found fewer _threads.mark_async call sites than the 5 this "
        f"fix converted ({async_call_sites!r}); either the visitor stopped "
        "matching or the call sites regressed to the blocking form under a "
        "shape the sweep above does not recognise"
    )
