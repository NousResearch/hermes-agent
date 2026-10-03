"""Regression test: two PROCESSES racing on suggestions.json lose a suggestion (W54-F038).

``cron/suggestions.py`` serializes its load->modify->save critical sections with a
process-local ``threading.Lock`` (:36), but writers span processes: the gateway
(background review fork / blueprints, ``cron/blueprints.py``) and ``hermes suggestions
accept|dismiss`` (a separate CLI process, ``hermes_cli/suggestions_cmd.py``) all RMW
the same ``suggestions.json``. A threading.Lock cannot exclude another process, so two
concurrent ``add_suggestion`` calls each read the pre-write snapshot and the last
``_save_raw`` wins — one process's suggestion is silently lost.

The fix takes ``fcntl.flock(LOCK_EX)`` on a sidecar ``.suggestions.lock`` in the
suggestions file's directory inside the existing threading-lock section, serializing
the whole read-modify-write span across processes too.

This test forces the race: two spawn'd processes rendezvous on a ``Barrier`` and both
run ``add_suggestion`` with the save stretched by a test-only pause inside the child's
critical section, making the both-load-before-either-saves overlap deterministic.
Pre-fix the second writer's load reads the pre-first-write file and its save clobbers
the first record. With the fix the flock serializes the whole span and the final file
holds both suggestions.
"""

import multiprocessing
import os
import sys
import time
from pathlib import Path

import pytest

from cron import suggestions as suggestions_mod

try:
    import fcntl  # noqa: F401  (availability gate only; the race fix uses it)

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover
    _HAS_FCNTL = False

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(suggestions_mod.__file__)))


def _suggestion_payload(title, key):
    return dict(
        title=title,
        description=f"desc {title}",
        source="catalog",
        job_spec={"prompt": "probe", "schedule": "0 0 * * *", "name": title, "deliver": "origin"},
        dedup_key=key,
    )


def _rmw_worker(suggestions_file, barrier, title, key, ok_flag, pause):
    """Spawn target: rendezvous, then one add_suggestion with a stretched save.

    The pause sits INSIDE add_suggestion's critical section (wrapped _save_raw), so
    with both workers released by the barrier together, both loads necessarily read
    the same pre-write snapshot unless a cross-process lock serializes the span.
    """
    sys.path.insert(0, _REPO_ROOT)
    from cron import suggestions as s

    s.SUGGESTIONS_FILE = Path(suggestions_file)
    if pause > 0:
        _real_save = s._save_raw

        def _slow_save(records):
            time.sleep(pause)
            _real_save(records)

        s._save_raw = _slow_save
    try:
        barrier.wait(timeout=30)
        record = s.add_suggestion(**_suggestion_payload(title, key))
        if record is None or record.get("title") != title:
            raise AssertionError(f"add_suggestion returned {record!r}")
        Path(ok_flag).write_text("ok")
    except Exception as exc:  # noqa: BLE001 — any failure must surface, not wedge the barrier
        try:
            Path(ok_flag).write_text(f"error: {exc!r}")
        except OSError:
            pass
        raise
    return 0


@pytest.mark.skipif(not _HAS_FCNTL, reason="POSIX fcntl/flock required")
def test_concurrent_adds_from_two_processes_both_persist(tmp_path, monkeypatch):
    suggestions_file = tmp_path / "suggestions.json"
    monkeypatch.setattr(suggestions_mod, "SUGGESTIONS_FILE", suggestions_file)

    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    procs, flags = [], []
    for i, title in enumerate(("alpha", "beta")):
        flag = tmp_path / f"child_{i}.result"
        flags.append(flag)
        procs.append(
            ctx.Process(
                target=_rmw_worker,
                args=(str(suggestions_file), barrier, title, f"w55-{title}", str(flag), 0.5),
            )
        )
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert not p.is_alive(), "worker process wedged in the critical section"
        assert p.exitcode == 0, f"worker crashed: {flags[procs.index(p)].read_text()}"

    # Each worker believed its own add succeeded; the file must still hold BOTH.
    # Pre-fix: last-writer-wins drops one of them and this assertion fails.
    on_disk = suggestions_mod.load_suggestions()
    assert {r["title"] for r in on_disk} == {"alpha", "beta"}
    assert {r["status"] for r in on_disk} == {"pending"}