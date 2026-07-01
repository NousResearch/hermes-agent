"""Regression tests for #55924: cron scheduler SIGTERM/teardown RuntimeError guards.

The bug: during gateway SIGTERM teardown, three code paths in
``cron/scheduler.py`` call ``asyncio.run()`` / ``executor.submit()`` while
the Python interpreter is finalizing, raising:

    RuntimeError: cannot schedule new futures after interpreter shutdown

That traceback pollutes ``~/.hermes/logs/errors.log`` on every gateway
restart (``hermes update``, ``hermes gateway stop``, system shutdown). The
fix guards each call site with ``sys.is_finalizing()``.

These tests:

1. **Production-path RED proof**: demonstrate that ``ThreadPoolExecutor.submit``
   raises ``RuntimeError: cannot schedule new futures after interpreter
   shutdown`` once the interpreter is finalizing. We simulate this by
   monkeypatching ``ThreadPoolExecutor.submit`` to mirror CPython's real
   at-exit behavior (the actual CPython guard is internal; the simulated
   stub is the canonical reproduction pattern).

2. **Layer 1 + Layer 2 (_deliver_result)**: assert ``_deliver_result`` skips
   ``asyncio.run`` / ``pool.submit`` when ``sys.is_finalizing()`` returns
   True, and still attempts delivery (or short-circuits via the local
   branch) when it returns False.

3. **Layer 3 (_submit_with_guard)**: assert the new top-level helper
   ``_submit_with_guard`` (exposed by the fix) returns ``None`` during
   finalization without calling ``pool.submit``, and returns a real
   ``Future`` in the normal path.

Layer mapping (see LAYERS.md):
- Layer 1 — _deliver_result() standalone ``asyncio.run(coro)`` (~line 1474)
- Layer 2 — _deliver_result() RuntimeError fallback ``pool.submit(asyncio.run, ...)``
            (~line 1483)
- Layer 3 — ``_submit_with_guard`` helper (~line 2956) replacing the inline
            ``pool.submit(_run_and_release)`` in ``tick()``
"""

from __future__ import annotations

import concurrent.futures
import sys
from unittest.mock import patch

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME per test (matches sibling cron tests)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)
    return home


def _install_finalizing_guard(monkeypatch, value: bool):
    """Stub ``sys.is_finalizing`` to return ``value`` on both ``sys`` and
    ``cron.scheduler.sys`` so the scheduler module sees the same answer
    the production code does."""
    monkeypatch.setattr(sys, "is_finalizing", lambda: value, raising=False)
    import cron.scheduler as sched

    monkeypatch.setattr(sched.sys, "is_finalizing", lambda: value, raising=False)


# ---------------------------------------------------------------------------
# Production-path proof: pool.submit raises during interpreter finalization
# ---------------------------------------------------------------------------


def test_pool_submit_raises_after_finalize_marker(monkeypatch):
    """RED-phase proof: under interpreter finalization, pool.submit raises.

    We simulate CPython's at-exit guard by wrapping ``ThreadPoolExecutor.submit``
    to raise ``RuntimeError: cannot schedule new futures`` when
    ``sys.is_finalizing()`` returns True. This mirrors the real behavior at
    shutdown without depending on CPython's internal ``_python_exit`` flag,
    which is not part of the public API.
    """
    original_submit = concurrent.futures.ThreadPoolExecutor.submit

    def _stub_submit(self, fn, *args, **kwargs):
        if sys.is_finalizing():
            raise RuntimeError(
                "cannot schedule new futures after interpreter shutdown"
            )
        return original_submit(self, fn, *args, **kwargs)

    monkeypatch.setattr(
        "concurrent.futures.ThreadPoolExecutor.submit",
        _stub_submit,
    )
    _install_finalizing_guard(monkeypatch, value=True)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        with pytest.raises(RuntimeError, match="cannot schedule new futures"):
            pool.submit(lambda: None)
    finally:
        # Cannot call pool.shutdown(wait=) on an already-finalizing pool
        # without re-triggering the guard, so just drop the reference.
        del pool


def test_pool_submit_succeeds_when_not_finalizing(monkeypatch):
    """Counter-control: with ``is_finalizing=False``, the same wrapped
    submit must succeed normally (proves our stub isn't over-blocking)."""
    original_submit = concurrent.futures.ThreadPoolExecutor.submit

    def _stub_submit(self, fn, *args, **kwargs):
        if sys.is_finalizing():
            raise RuntimeError(
                "cannot schedule new futures after interpreter shutdown"
            )
        return original_submit(self, fn, *args, **kwargs)

    monkeypatch.setattr(
        "concurrent.futures.ThreadPoolExecutor.submit",
        _stub_submit,
    )
    _install_finalizing_guard(monkeypatch, value=False)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        fut = pool.submit(lambda: 42)
        assert fut.result(timeout=5) == 42
    finally:
        pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Layer 1 + Layer 2: _deliver_result() shutdown guard
# ---------------------------------------------------------------------------


def test_deliver_result_consults_shutdown_guard(monkeypatch, hermes_env):
    """The fix must consult ``sys.is_finalizing()`` in the standalone
    delivery path of ``_deliver_result`` (Layer 1) — BEFORE calling
    ``asyncio.run(coro)``.

    Verified by inspecting the function source: the fix must include at
    least one ``sys.is_finalizing()`` call inside the ``if not delivered``
    branch (where the standalone ``asyncio.run`` block lives). Pre-fix the
    function contains no such call (RED). Post-fix it does (GREEN).

    This is a structural regression test — it pins the bug-class fix to
    the function so a future refactor cannot silently remove the guard
    without breaking this test.
    """
    import inspect
    import cron.scheduler as sched

    src = inspect.getsource(sched._deliver_result)

    # The fix must add ``sys.is_finalizing()`` inside _deliver_result.
    assert "sys.is_finalizing()" in src, (
        "_deliver_result must consult sys.is_finalizing() to guard "
        "asyncio.run() during interpreter finalization (#55924). "
        "Function source does not contain a sys.is_finalizing() call."
    )

    # Guard must appear in the standalone delivery path, not just at the top.
    # Approximate: between "if not delivered" and the end of the function
    # (or the next top-level def) there must be at least one is_finalizing.
    standalone_block_marker = "if not delivered:"
    assert standalone_block_marker in src, (
        f"_deliver_result missing standalone-delivery marker "
        f"{standalone_block_marker!r} — function structure may have changed."
    )
    # Take the slice from the standalone marker onward.
    standalone_block = src[src.index(standalone_block_marker):]
    assert "sys.is_finalizing()" in standalone_block, (
        "sys.is_finalizing() must guard the standalone delivery block "
        "(Layer 1 — asyncio.run path) and the fallback recovery "
        "(Layer 2 — pool.submit path) inside _deliver_result (#55924)."
    )


def test_deliver_result_normal_path_does_not_short_circuit(monkeypatch, hermes_env):
    """Guard control: with ``is_finalizing=False``, ``_deliver_result`` MUST
    still take its normal control-flow path — it must not short-circuit on
    the shutdown guard. We verify by asserting that ``deliver='local'``
    still returns ``None`` (the documented local-only path) rather than
    the shutdown skip marker, and does not raise."""
    import cron.scheduler as sched

    _install_finalizing_guard(monkeypatch, value=False)

    job = {
        "id": "normal-deliver-test",
        "name": "Normal deliver test",
        "deliver": "local",
    }
    # With ``deliver='local'`` and no targets resolvable, _deliver_result
    # returns None via the local-only early-return. The shutdown guard
    # MUST NOT intercept this — that would be a false-positive shutdown.
    result = sched._deliver_result(job, "test content")
    assert result is None, (
        f"Normal-path deliver='local' must return None (local-only path), "
        f"got {result!r} — the shutdown guard may be over-blocking."
    )


# ---------------------------------------------------------------------------
# Layer 3: _submit_with_guard helper exposed by the fix
# ---------------------------------------------------------------------------


def test_submit_with_guard_module_helper_exists(hermes_env):
    """The fix must expose a top-level ``_submit_with_guard`` helper so it
    can be (a) unit-tested directly and (b) reused from any future tick site.

    Pre-fix this helper is defined as an inner closure inside ``tick()`` and
    is not module-callable, which is why the build/test layer cannot exercise
    it in isolation.
    """
    import cron.scheduler as sched

    assert hasattr(sched, "_submit_with_guard"), (
        "cron.scheduler must expose a module-level _submit_with_guard helper"
    )
    assert callable(sched._submit_with_guard)


def test_submit_with_guard_returns_none_when_finalizing(monkeypatch, hermes_env):
    """When ``sys.is_finalizing()`` is True, ``_submit_with_guard`` must
    return ``None`` without touching ``pool.submit``. This is the core of
    the Layer 3 fix."""
    import cron.scheduler as sched

    _install_finalizing_guard(monkeypatch, value=True)

    # Wrap ThreadPoolExecutor.submit to fail loudly if called (RED guard).
    submit_calls = []
    original_submit = concurrent.futures.ThreadPoolExecutor.submit

    def _spy_submit(self, fn, *args, **kwargs):
        submit_calls.append(fn)
        return original_submit(self, fn, *args, **kwargs)

    monkeypatch.setattr(
        "concurrent.futures.ThreadPoolExecutor.submit",
        _spy_submit,
    )

    job = {"id": "guard-finalizing", "name": "guard test"}
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        result = sched._submit_with_guard(job, pool)
        assert result is None, (
            f"_submit_with_guard must return None during finalization, got {result!r}"
        )
        assert submit_calls == [], (
            f"_submit_with_guard called pool.submit during finalization "
            f"(calls={submit_calls})"
        )
    finally:
        # Pool may already be tainted by the shutdown guard; drop it.
        del pool


def test_submit_with_guard_returns_future_when_normal(monkeypatch, hermes_env):
    """Counter-control: with ``is_finalizing=False``, ``_submit_with_guard``
    must submit the job and return a real ``concurrent.futures.Future``.
    The pre-fix inline code did this; the fix must preserve it."""
    import cron.scheduler as sched

    _install_finalizing_guard(monkeypatch, value=False)

    job = {"id": "guard-normal", "name": "normal guard test"}
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        result = sched._submit_with_guard(job, pool)
        assert isinstance(result, concurrent.futures.Future), (
            f"_submit_with_guard must return a Future in the normal path, got {type(result).__name__}"
        )
        # Wait for completion to keep the test deterministic.
        result.result(timeout=10)
    finally:
        pool.shutdown(wait=True)


def test_submit_with_guard_skips_already_running(monkeypatch, hermes_env):
    """The pre-fix inner-closure helper deduped against ``_running_job_ids``.
    The post-fix module-level helper must preserve that behavior: if the
    job is already in-flight, return ``None`` and skip the submit."""
    import cron.scheduler as sched

    _install_finalizing_guard(monkeypatch, value=False)

    job = {"id": "guard-dedup", "name": "dedup test"}
    sched._running_job_ids.add(job["id"])
    try:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            result = sched._submit_with_guard(job, pool)
            assert result is None, (
                f"_submit_with_guard must skip already-running jobs, got {result!r}"
            )
        finally:
            pool.shutdown(wait=False)
    finally:
        sched._running_job_ids.discard(job["id"])