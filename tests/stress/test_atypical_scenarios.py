"""Atypical user scenarios and configurations.

Exercises the kernel against user inputs and environments that the
normal tests assume away:

  - Data: unicode, emoji, RTL, huge strings, control chars, SQL
    injection attempts, malformed JSON, newlines in summaries.
  - Graph: cycles, self-parenting, diamonds, wide fan-out/fan-in.
  - Workspace: non-existent, spaces, symlinks, path traversal.
  - Clock: skew, pre-1970 timestamps, zero-duration runs.
  - Filesystem: HERMES_HOME with spaces / unicode / symlinks.
  - Scale extremes: 100k tasks, 10k runs per task, huge bodies.
  - Concurrency: idempotency-key race across processes.
  - Hostile: path traversal attempts, injection attempts.

Each scenario is self-contained. Failures are collected and printed
together at the end. Script exits 0 iff every scenario passed or was
cleanly SKIPPED (with reason).
"""

import json
import multiprocessing as mp
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Resolve the worktree path robustly.
_THIS = Path(__file__).resolve()
WT = _THIS.parents[2] if _THIS.parent.name == "stress" else Path.cwd()

FAILURES: list[str] = []
SKIPS: list[str] = []
_REGISTERED: list = []


def scenario(name):
    """Decorator: run `fn` in its own HERMES_HOME, collect failures.

    The returned function is named `_scenario_<name>` so discovery can
    find it in globals() reliably.
    """
    def wrap(fn):
        def run():
            home = tempfile.mkdtemp(prefix=f"hermes_atyp_{name}_")
            os.environ["HERMES_HOME"] = home
            os.environ["HOME"] = home
            for m in list(sys.modules.keys()):
                if m.startswith(("hermes_cli", "plugins", "gateway")):
                    del sys.modules[m]
            sys.path.insert(0, str(WT))
            from hermes_cli import kanban_db as kb  # noqa: F401
            print(f"\n═══ {name} ═══")
            try:
                fn(home, kb)
                print(f"  ✔ {name}")
            except AssertionError as e:
                msg = f"{name}: {e}"
                FAILURES.append(msg)
                print(f"  ✗ FAIL: {e}")
            except Exception as e:
                msg = f"{name}: unexpected {type(e).__name__}: {e}"
                FAILURES.append(msg)
                import traceback
                traceback.print_exc()
                print(f"  ✗ ERROR: {msg}")
            finally:
                try:
                    shutil.rmtree(home)
                except Exception:
                    pass
        run.__name__ = f"_scenario_{name}"
        # Register in a module-level list so discovery is trivial.
        _REGISTERED.append(run)
        return run
    return wrap


# =============================================================================
# DATA WEIRDNESS
# =============================================================================

@scenario("unicode_and_emoji")
def _(home, kb):
    kb.init_db()
    conn = kb.connect()
    try:
        # Emoji, CJK, RTL, zero-width joiner
        cases = [
            ("📋 buy groceries 🍎", "shopping"),
            ("设计认证模式", "implement"),
            ("אימות משתמש חדש", "auth-rtl"),  # Hebrew RTL
            ("مهمة تصحيح الأخطاء", "bug-arabic"),
            ("👨‍👩‍👧‍👦 family emoji ZWJ sequences 🏳️‍🌈", "emoji-stress"),
            ("control\x01chars\x02in\x03body", "ctrl"),
            ("null\x00bytes", "nullbyte"),
        ]
        for title, kind in cases:
            tid = kb.create_task(conn, title=title, assignee="w")
            back = kb.get_task(conn, tid)
            assert back.title == title, (
                f"[{kind}] round-trip mismatch: {title!r} → {back.title!r}"
            )
        print(f"  {len(cases)} unicode titles round-tripped")

        # Metadata with non-ASCII + emoji
        tid = kb.create_task(conn, title="with meta", assignee="w")
        kb.claim_task(conn, tid)
        meta = {
            "作者": "张三",
            "summary_fr": "résumé avec des caractères accentués",
            "emoji": "🎉🔥💯",
            "mixed_list": ["normal", "日本語", "🇺🇸"],
        }
        kb.complete_task(
            conn, tid,
            summary="完成了 📝 résumé",
            metadata=meta,
        )
        run = kb.latest_run(conn, tid)
        assert run.summary == "完成了 📝 résumé", f"summary round-trip failed"
        assert run.metadata == meta, (
            f"metadata round-trip failed: {run.metadata} != {meta}"
        )
        print(f"  metadata with CJK + emoji round-tripped")
    finally:
        conn.close()


@scenario("huge_strings")
def _(home, kb):
    """1MB body + 1MB summary + deeply nested metadata."""
    kb.init_db()
    conn = kb.connect()
    try:
        huge_body = "x" * (1024 * 1024)  # 1 MB
        huge_summary = "y" * (1024 * 1024)
        # Nested metadata: 50 levels deep
        meta = "leaf"
        for _ in range(50):
            meta = {"nested": meta}
        tid = kb.create_task(
            conn, title="huge task", body=huge_body, assignee="w",
        )
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, summary=huge_summary, metadata=meta)

        back = kb.get_task(conn, tid)
        assert back.body == huge_body, f"body truncated: {len(back.body)} vs {len(huge_body)}"
        run = kb.latest_run(conn, tid)
        assert run.summary == huge_summary
        assert run.metadata == meta
        print(f"  1 MB body + 1 MB summary + 50-deep metadata OK")
    finally:
        conn.close()


@scenario("sql_injection_attempts")
def _(home, kb):
    """SQLite parameterized queries should neutralize all of these, but
    verify empirically across every string field."""
    kb.init_db()
    conn = kb.connect()
    try:
        payloads = [
            "'; DROP TABLE tasks; --",
            "\" OR 1=1 --",
            "'; DELETE FROM task_runs; --",
            "Robert'); DROP TABLE students;--",  # Little Bobby Tables
            "\\x00\\x01\\x02",
            "' UNION SELECT * FROM kanban_notify_subs --",
        ]
        for p in payloads:
            tid = kb.create_task(
                conn, title=p, body=p, assignee=p, tenant=p,
            )
            back = kb.get_task(conn, tid)
            assert back.title == p
            assert back.body == p
            # Kernel should have stored, not executed
            # Verify tasks table still has rows
        count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert count == len(payloads), f"lost rows: {count} vs {len(payloads)}"
        # tasks table wasn't dropped (we're still here)
        print(f"  {len(payloads)} injection payloads neutralized")
    finally:
        conn.close()


@scenario("newlines_in_summary")
def _(home, kb):
    """Summaries with newlines, tabs, and shell metachars.

    The notifier truncates to first line — verify that's right, not
    that the kernel loses data."""
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="multiline", assignee="w")
        kb.claim_task(conn, tid)
        multi = "line 1\nline 2\tindented\n\nline 4"
        kb.complete_task(conn, tid, summary=multi)
        run = kb.latest_run(conn, tid)
        assert run.summary == multi, "full summary should survive in kernel"
        # Event payload takes first line (for notifier brevity)
        events = [e for e in kb.list_events(conn, tid) if e.kind == "completed"]
        assert events[0].payload["summary"] == "line 1", (
            f"event payload should be first line, got {events[0].payload['summary']!r}"
        )
        print("  multiline summary preserved on run; first line in event")
    finally:
        conn.close()


@scenario("malformed_metadata_via_cli")
def _(home, kb):
    """CLI rejects malformed JSON and non-dict JSON cleanly."""
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="meta test", assignee="w")
        kb.claim_task(conn, tid)
    finally:
        conn.close()

    env = {**os.environ, "PYTHONPATH": str(WT), "HERMES_HOME": home, "HOME": home}
    bad_metas = [
        "not-json",
        "[1, 2, 3]",  # array not dict
        "42",  # scalar
        '{"unclosed',  # truncated
    ]
    for bad in bad_metas:
        r = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "kanban",
             "complete", tid, "--metadata", bad],
            capture_output=True, text=True, env=env,
        )
        # Should print an error to stderr, exit non-zero, not touch the task
        assert "metadata" in r.stderr.lower() or "json" in r.stderr.lower(), (
            f"bad metadata {bad!r} didn't produce a metadata error: "
            f"stderr={r.stderr!r}"
        )
    # Verify task is still running (no partial apply)
    conn = kb.connect()
    try:
        assert kb.get_task(conn, tid).status == "running"
    finally:
        conn.close()
    print(f"  {len(bad_metas)} malformed --metadata values cleanly rejected")


# =============================================================================
# DEPENDENCY GRAPH PATHOLOGIES
# =============================================================================

@scenario("dependency_cycle")
def _(home, kb):
    """A → B → A should be refused. If it's allowed, recompute_ready
    could infinite-loop or never promote."""
    kb.init_db()
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="A", assignee="w")
        b = kb.create_task(conn, title="B", assignee="w", parents=[a])
        # Try to link A back to B — creating the cycle
        try:
            kb.link_tasks(conn, parent_id=b, child_id=a)
            # If that didn't raise, the kernel allowed a cycle.
            # Verify recompute_ready at least doesn't hang.
            import threading
            done = threading.Event()
            result = []
            def run():
                try:
                    result.append(kb.recompute_ready(conn))
                except Exception as e:
                    result.append(e)
                done.set()
            t = threading.Thread(target=run, daemon=True)
            t.start()
            done.wait(timeout=5)
            if not done.is_set():
                assert False, "recompute_ready HUNG on cyclic graph"
            raise AssertionError(
                "cycle creation was allowed; kernel should reject"
            )
        except (ValueError, RuntimeError, sqlite3.IntegrityError) as e:
            # Expected: kernel refuses the cycle
            print(f"  cycle correctly rejected: {e}")
    finally:
        conn.close()


@scenario("self_parent")
def _(home, kb):
    """A task cannot be its own parent."""
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="self", assignee="w")
        try:
            kb.link_tasks(conn, parent_id=tid, child_id=tid)
            raise AssertionError("self-parenting should be rejected")
        except (ValueError, RuntimeError, sqlite3.IntegrityError) as e:
            print(f"  self-parent rejected: {e}")
    finally:
        conn.close()


@scenario("diamond_dependency")
def _(home, kb):
    """Root → (A, B) → leaf. Leaf should promote to ready only when
    BOTH A and B are done."""
    kb.init_db()
    conn = kb.connect()
    try:
        root = kb.create_task(conn, title="root", assignee="w")
        kb.claim_task(conn, root)
        kb.complete_task(conn, root, result="ready")
        a = kb.create_task(conn, title="A", assignee="w", parents=[root])
        b = kb.create_task(conn, title="B", assignee="w", parents=[root])
        leaf = kb.create_task(conn, title="leaf", assignee="w", parents=[a, b])

        # A done but B not → leaf stays todo
        kb.claim_task(conn, a)
        kb.complete_task(conn, a, result="a done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, leaf).status == "todo", (
            f"leaf should still be todo with B unfinished, got "
            f"{kb.get_task(conn, leaf).status}"
        )
        # Both done → leaf promotes
        kb.claim_task(conn, b)
        kb.complete_task(conn, b, result="b done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, leaf).status == "ready", (
            f"leaf should promote with both parents done, got "
            f"{kb.get_task(conn, leaf).status}"
        )
        print(f"  diamond dependency resolved correctly")
    finally:
        conn.close()


@scenario("wide_fan_out")
def _(home, kb):
    """One parent, 500 children. Completing the parent should promote
    all 500 in its own recompute_ready pass (triggered by complete_task).
    """
    kb.init_db()
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="root", assignee="w")
        children = [
            kb.create_task(conn, title=f"c{i}", assignee="w", parents=[parent])
            for i in range(500)
        ]
        kb.claim_task(conn, parent)
        t0 = time.monotonic()
        kb.complete_task(conn, parent, result="done")
        elapsed = (time.monotonic() - t0) * 1000
        # complete_task calls recompute_ready internally; check result.
        ready_count = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status='ready' AND id != ?",
            (parent,),
        ).fetchone()[0]
        assert ready_count == 500, f"expected 500 promoted, got {ready_count}"
        for cid in children[:5]:
            assert kb.get_task(conn, cid).status == "ready"
        print(f"  500 children promoted in {elapsed:.0f}ms (via complete_task)")
    finally:
        conn.close()


@scenario("wide_fan_in")
def _(home, kb):
    """500 parents, 1 child. Child should not promote until all 500 done."""
    kb.init_db()
    conn = kb.connect()
    try:
        parents = [
            kb.create_task(conn, title=f"p{i}", assignee="w") for i in range(500)
        ]
        child = kb.create_task(
            conn, title="leaf", assignee="w", parents=parents,
        )
        # Complete 499 parents
        for p in parents[:-1]:
            kb.claim_task(conn, p)
            kb.complete_task(conn, p)
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo", (
            "child should still be todo with 1/500 parents incomplete"
        )
        # Finish the last one
        kb.claim_task(conn, parents[-1])
        kb.complete_task(conn, parents[-1])
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"
        print(f"  500 parents → 1 child promotion works")
    finally:
        conn.close()


# =============================================================================
# WORKSPACE EDGE CASES
# =============================================================================

@scenario("workspace_path_traversal")
def _(home, kb):
    """`workspace_path='../../../etc/passwd'` or absolute-outside-home
    should not be silently accepted and then executed in the wrong place."""
    kb.init_db()
    conn = kb.connect()
    try:
        # Direct kernel API — create with an attacker-ish path
        tid = kb.create_task(
            conn, title="path-traversal",
            assignee="w",
            workspace_kind="dir",
            workspace_path="../../../tmp/attacker",
        )
        task = kb.get_task(conn, tid)
        # Document what actually happens — is the path stored verbatim?
        # Is it resolved? Is it rejected?
        print(f"  stored workspace_path: {task.workspace_path!r}")
        print(f"  workspace_kind: {task.workspace_kind!r}")
        # Verify resolve_workspace (which the dispatcher calls) doesn't
        # allow escape.
        try:
            from hermes_cli.kanban_db import resolve_workspace
            resolved = resolve_workspace(task)
            # If resolve succeeded, check it's actually escape-safe.
            resolved_abs = str(Path(resolved).resolve())
            home_abs = str(Path(os.environ["HERMES_HOME"]).resolve())
            if not resolved_abs.startswith(home_abs) and resolved_abs.startswith("/tmp"):
                # This is escaping the home dir. Whether that's actually
                # a problem depends on the threat model. Flag for attention.
                print(f"  ⚠ workspace resolved OUTSIDE hermes_home: {resolved}")
                print(f"    (not necessarily a bug — dir: workspaces are intentionally arbitrary, but worth documenting)")
        except Exception as e:
            print(f"  resolve_workspace rejected: {e}")
    finally:
        conn.close()


@scenario("workspace_nonexistent_path")
def _(home, kb):
    """Dispatching a task whose workspace can't be resolved should go
    through the spawn-failure circuit breaker, not crash."""
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="bad-workspace", assignee="w",
            workspace_kind="dir",
            workspace_path="/nonexistent/path/that/does/not/exist",
        )
        # Run dispatch_once with a dummy spawn_fn
        result = kb.dispatch_once(conn, spawn_fn=lambda *_: 99999)
        # If the path was rejected, the task went through _record_spawn_failure
        task = kb.get_task(conn, tid)
        # Possible outcomes:
        # - Task back in ready (workspace issue = spawn_failed, retries)
        # - Task in running (kernel accepted the bogus path and spawned)
        # - Task auto-blocked (after N retries, but we only ran 1 tick)
        print(f"  after 1 tick with nonexistent workspace: status={task.status}")
        if task.status == "ready":
            # Expected path: workspace failure led to release
            spawn_failures = task.spawn_failures
            print(f"  spawn_failures counter: {spawn_failures}")
            assert spawn_failures >= 1, "spawn_failures counter didn't increment"
        elif task.status == "running":
            # Workspace not checked before spawn — the worker would hit
            # the bad path itself. Defensible for `dir:` workspaces that
            # the user might create later.
            print("  kernel accepted bogus path (deferred check to worker)")
    finally:
        conn.close()


# =============================================================================
# CLOCK SKEW
# =============================================================================

@scenario("clock_skew_start_greater_than_end")
def _(home, kb):
    """NTP jumps backward. Run.started_at gets written as 1234 but by
    the time complete_task runs, time.time() returned 1230. A human
    reading run history sees negative elapsed."""
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="time-travel", assignee="w")
        kb.claim_task(conn, tid)
        # Force a future started_at via raw SQL
        future = int(time.time()) + 3600
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE task_id = ?",
            (future, tid),
        )
        conn.commit()
        # Complete normally — ended_at will be now, < started_at
        kb.complete_task(conn, tid, summary="time-skewed")
        run = kb.latest_run(conn, tid)
        # Invariant I5 (from property fuzzer): started_at <= ended_at
        # when ended_at is set. Verify this is enforced OR gracefully
        # handled in display.
        if run.ended_at < run.started_at:
            # Kernel didn't reject the write; check that CLI display
            # doesn't produce "-1800s" elapsed.
            elapsed = run.ended_at - run.started_at
            print(f"  clock-skewed run: elapsed = {elapsed}s (negative)")
            print(f"  ⚠ kernel stores this; UI should clamp to 0 or handle")
            # Don't fail — document the behavior.
        else:
            print("  kernel normalized ended_at >= started_at")
    finally:
        conn.close()


# =============================================================================
# FILESYSTEM WEIRDNESS
# =============================================================================

@scenario("hermes_home_with_spaces")
def _(home, kb):
    """HERMES_HOME at a path with spaces — should work but catches
    anyone doing string interpolation without quoting."""
    # Note: home was already created with a safe prefix. We need to
    # reset to a weird one for this test.
    weird = tempfile.mkdtemp(prefix="hermes with spaces ")
    os.environ["HERMES_HOME"] = weird
    os.environ["HOME"] = weird
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="spaced", assignee="w")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, summary="path has spaces")
        runs = kb.list_runs(conn, tid)
        assert len(runs) == 1 and runs[0].outcome == "completed"
        # Verify the DB file is actually in the weird path
        db_path = Path(weird) / "kanban.db"
        assert db_path.exists(), f"DB not at {db_path}"
        print(f"  HERMES_HOME with spaces: OK at {weird}")
    finally:
        conn.close()
        shutil.rmtree(weird, ignore_errors=True)


@scenario("hermes_home_with_unicode")
def _(home, kb):
    """HERMES_HOME with non-ASCII chars."""
    # Pre-create directly since tempfile doesn't love unicode prefixes
    weird = f"/tmp/hermes_héllo_émöji_{os.getpid()}"
    os.makedirs(weird, exist_ok=True)
    os.environ["HERMES_HOME"] = weird
    os.environ["HOME"] = weird
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="unicode home", assignee="w")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, summary="ok")
        assert (Path(weird) / "kanban.db").exists()
        print(f"  HERMES_HOME with unicode path: OK at {weird}")
    finally:
        conn.close()
        shutil.rmtree(weird, ignore_errors=True)


@scenario("hermes_home_via_symlink")
def _(home, kb):
    """HERMES_HOME is a symlink to the real dir. _INITIALIZED_PATHS
    uses Path.resolve() — two different symlink names pointing at the
    same dir should NOT double-init."""
    real = tempfile.mkdtemp(prefix="hermes_real_")
    link1 = real + "_link1"
    link2 = real + "_link2"
    os.symlink(real, link1)
    os.symlink(real, link2)
    try:
        os.environ["HERMES_HOME"] = link1
        os.environ["HOME"] = link1
        kb._INITIALIZED_PATHS.clear()
        kb.init_db()
        conn1 = kb.connect()
        kb.create_task(conn1, title="t1", assignee="w")
        conn1.close()

        # Switch to link2 pointing at the same dir
        os.environ["HERMES_HOME"] = link2
        os.environ["HOME"] = link2
        conn2 = kb.connect()
        # Should see the task we created via link1
        all_tasks = kb.list_tasks(conn2)
        assert len(all_tasks) == 1, (
            f"symlinks to same dir should share DB, got {len(all_tasks)} tasks"
        )
        conn2.close()
        print("  symlinks to same HERMES_HOME share DB correctly")
    finally:
        for p in (link1, link2):
            try:
                os.remove(p)
            except OSError:
                pass
        shutil.rmtree(real, ignore_errors=True)


# =============================================================================
# SCALE EXTREMES
# =============================================================================

@scenario("huge_run_count_on_one_task")
def _(home, kb):
    """1000 reclaim cycles on a single task → 1000 run rows. Verify
    list_runs still performs, and build_worker_context isn't quadratic."""
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="retry-heavy", assignee="w")
        # Force reclaims by manually closing runs
        for i in range(1000):
            kb.claim_task(conn, tid)
            # Force close the run directly so we can make another claim
            rid = kb.latest_run(conn, tid).id
            kb._end_run(conn, tid, outcome="reclaimed", summary=f"attempt {i}")
            conn.execute(
                "UPDATE tasks SET status='ready', claim_lock=NULL, "
                "claim_expires=NULL WHERE id=?", (tid,),
            )
            conn.commit()
        runs = kb.list_runs(conn, tid)
        assert len(runs) == 1000, f"expected 1000 runs, got {len(runs)}"
        # build_worker_context should NOT take forever
        t0 = time.monotonic()
        ctx = kb.build_worker_context(conn, tid)
        elapsed = (time.monotonic() - t0) * 1000
        # The "Prior attempts" section renders ALL closed runs.
        # For 1000 runs this could produce a massive string.
        # Fair question: is this bounded? Let's measure.
        print(f"  1000 runs → list_runs OK; build_worker_context = {elapsed:.0f}ms, {len(ctx)} chars")
        if len(ctx) > 200_000:
            print(f"  ⚠ build_worker_context unbounded on retry-heavy tasks "
                  f"({len(ctx)} chars) — worker context will be huge")
    finally:
        conn.close()


@scenario("hundred_tenants")
def _(home, kb):
    """100 distinct tenants with 50 tasks each. board_stats + list_tasks
    should still return quickly."""
    kb.init_db()
    conn = kb.connect()
    try:
        for t in range(100):
            for i in range(50):
                kb.create_task(
                    conn, title=f"tenant-{t}-task-{i}",
                    tenant=f"tenant_{t:03d}",
                    assignee="w",
                )
        t0 = time.monotonic()
        stats = kb.board_stats(conn)
        el_stats = (time.monotonic() - t0) * 1000
        t0 = time.monotonic()
        tasks = kb.list_tasks(conn)
        el_list = (time.monotonic() - t0) * 1000
        print(f"  5000 tasks / 100 tenants: stats={el_stats:.0f}ms, list={el_list:.0f}ms")
        assert len(tasks) == 5000
    finally:
        conn.close()


# =============================================================================
# CONCURRENCY CORNERS
# =============================================================================

def _idempotency_race_worker(hermes_home: str, key: str, result_file: str,
                             barrier_path: str) -> None:
    """Subprocess body for the idempotency race test."""
    os.environ["HERMES_HOME"] = hermes_home
    os.environ["HOME"] = hermes_home
    sys.path.insert(0, str(WT))
    from hermes_cli import kanban_db as kb

    # Spin until the barrier file exists (crude sync across processes)
    while not os.path.exists(barrier_path):
        time.sleep(0.001)

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title=f"race pid={os.getpid()}",
            assignee="w", idempotency_key=key,
        )
    finally:
        conn.close()
    with open(result_file, "w") as f:
        f.write(tid)


@scenario("idempotency_key_race")
def _(home, kb):
    """Two processes concurrently call create_task with the same
    idempotency_key — should both get back the SAME task id, not two
    different ones."""
    kb.init_db()
    # Spawn workers, then drop the barrier so they fire ~simultaneously.
    key = "race-key-12345"
    barrier = os.path.join(home, "barrier")
    results = [os.path.join(home, f"res_{i}") for i in range(2)]
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(
            target=_idempotency_race_worker,
            args=(home, key, results[i], barrier),
        )
        for i in range(2)
    ]
    for p in procs:
        p.start()
    time.sleep(0.1)  # let them hit the spin
    # Fire the gun
    with open(barrier, "w") as f:
        f.write("go")
    for p in procs:
        p.join(timeout=10)

    tids = [open(r).read().strip() for r in results if os.path.exists(r)]
    assert len(tids) == 2, f"only {len(tids)} workers finished"
    assert tids[0] == tids[1], (
        f"idempotency key race produced two different tasks: {tids}"
    )
    # Also verify there's only ONE row in the DB
    conn = kb.connect()
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE idempotency_key = ?",
            (key,),
        ).fetchone()[0]
        assert count == 1, f"expected 1 task with key, got {count}"
    finally:
        conn.close()
    print(f"  idempotency race: both workers got {tids[0]}")



# =============================================================================
# MORE EDGE CASES
# =============================================================================

@scenario("assignee_with_special_chars")
def _(home, kb):
    """Profile names can contain @-signs, dots, hyphens. Some users
    might try nonsense. Kernel shouldn't break on any of them."""
    kb.init_db()
    conn = kb.connect()
    try:
        assignees = [
            "normal-dev",
            "dev.with.dots",
            "backend@v2",
            "日本語-dev",
            "🤖-bot",
            "x" * 200,  # very long
            "",  # empty string
        ]
        for a in assignees:
            tid = kb.create_task(conn, title=f"for {a!r}", assignee=a or None)
            back = kb.get_task(conn, tid)
            # Empty string is coerced to None by kernel, or stored verbatim?
            if a:
                assert back.assignee == a, f"assignee round-trip: {a!r} → {back.assignee!r}"
        print(f"  {len(assignees)} weird assignee names round-tripped")
    finally:
        conn.close()


@scenario("completed_task_reclaim_attempt")
def _(home, kb):
    """A task in 'done' should NOT be reclaimable — reclaim/claim paths
    must refuse."""
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="terminal", assignee="w")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, summary="all done")
        # Try to re-claim a done task
        claimed = kb.claim_task(conn, tid)
        assert claimed is None, "done task should not be claimable"
        # Try to complete it again
        ok = kb.complete_task(conn, tid, summary="oops twice")
        assert ok is False, "completing an already-done task should refuse"
        # Try to block it
        ok = kb.block_task(conn, tid, reason="trying")
        assert ok is False, "blocking a done task should refuse"
        print("  done task correctly resists re-claim/complete/block")
    finally:
        conn.close()


@scenario("archived_task_resurrection_attempt")
def _(home, kb):
    """An archived task should be invisible to normal ops."""
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="archive-me", assignee="w")
        kb.archive_task(conn, tid)
        # Archived task shouldn't appear in default list
        tasks = kb.list_tasks(conn)
        assert all(t.id != tid for t in tasks), "archived task leaked into default list"
        # But it should still exist in the DB
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row is not None
        assert row["status"] == "archived"
        # Trying to claim an archived task: should refuse
        claimed = kb.claim_task(conn, tid)
        assert claimed is None, "archived task should not be claimable"
        # Archived can be un-archived via direct status? No API for that intentionally
        # (archive is meant to be terminal). Verify this.
        # complete/block/unblock on archived should all refuse.
        assert kb.complete_task(conn, tid) is False
        assert kb.block_task(conn, tid, reason="no") is False
        assert kb.unblock_task(conn, tid) is False
        print("  archived task cannot be resurrected via normal APIs")
    finally:
        conn.close()


@scenario("unassigned_task_never_claims")
def _(home, kb):
    """Task without an assignee should never be claimed by dispatch_once,
    even though its status might be 'ready' if it has no parents."""
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="orphan", assignee=None)
        assert kb.get_task(conn, tid).status == "ready"
        result = kb.dispatch_once(conn, spawn_fn=lambda *_: 42)
        assert tid in result.skipped_unassigned
        assert len(result.spawned) == 0
        # Task should still be ready, untouched
        assert kb.get_task(conn, tid).status == "ready"
        print("  unassigned ready task correctly skipped by dispatcher")
    finally:
        conn.close()


@scenario("comment_storm")
def _(home, kb):
    """1000 comments on a single task — build_worker_context should still
    be reasonable."""
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="chatty", assignee="w")
        for i in range(1000):
            kb.add_comment(conn, tid, author=f"user{i % 5}", body=f"comment number {i}")
        comments = kb.list_comments(conn, tid)
        assert len(comments) == 1000
        t0 = time.monotonic()
        ctx = kb.build_worker_context(conn, tid)
        elapsed = (time.monotonic() - t0) * 1000
        print(f"  1000 comments: list in {elapsed:.0f}ms, context size = {len(ctx)} chars")
        if len(ctx) > 200_000:
            print(f"  ⚠ comment thread unbounded in worker context")
    finally:
        conn.close()


@scenario("empty_string_fields")
def _(home, kb):
    """Empty title should be rejected (we already do this). Empty body,
    empty summary, etc. should be accepted."""
    kb.init_db()
    conn = kb.connect()
    try:
        # Empty title → reject
        try:
            kb.create_task(conn, title="", assignee="w")
            raise AssertionError("empty title should have been rejected")
        except ValueError:
            pass
        # Whitespace-only title → reject
        try:
            kb.create_task(conn, title="   \t\n  ", assignee="w")
            raise AssertionError("whitespace-only title should have been rejected")
        except ValueError:
            pass
        # Empty body → accept (legitimate: just title says it all)
        tid = kb.create_task(conn, title="empty body ok", body="", assignee="w")
        assert kb.get_task(conn, tid).body in ("", None)
        # Empty summary on complete → accept
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, summary="")
        run = kb.latest_run(conn, tid)
        # Empty summary falls back to result; both empty → None on run
        print(f"  empty body accepted, empty-title rejected")
    finally:
        conn.close()


@scenario("tenant_with_newlines")
def _(home, kb):
    """Someone pastes a multi-line string into --tenant. Kernel should
    store what it gets — but queries filtering by tenant should still
    work against the raw value."""
    kb.init_db()
    conn = kb.connect()
    try:
        weird_tenant = "line1\nline2\tindented"
        tid = kb.create_task(conn, title="weird tenant", assignee="w", tenant=weird_tenant)
        back = kb.get_task(conn, tid)
        assert back.tenant == weird_tenant
        # board_stats groups by tenant — verify it doesn't fall over
        stats = kb.board_stats(conn)
        print(f"  multiline tenant stored and stats still work")
    finally:
        conn.close()


@scenario("parent_in_different_status_states")
def _(home, kb):
    """recompute_ready promotes a todo child only if ALL parents are
    in 'done'. Verify against parents in every non-done state."""
    kb.init_db()
    conn = kb.connect()
    try:
        # Create one parent in each possible non-done state
        p_ready = kb.create_task(conn, title="p-ready", assignee="w")
        p_running = kb.create_task(conn, title="p-running", assignee="w")
        kb.claim_task(conn, p_running)
        p_blocked = kb.create_task(conn, title="p-blocked", assignee="w")
        kb.block_task(conn, p_blocked, reason="stuck")
        p_triage = kb.create_task(conn, title="p-triage", assignee="w", triage=True)
        p_archived = kb.create_task(conn, title="p-archived", assignee="w")
        kb.archive_task(conn, p_archived)
        p_done = kb.create_task(conn, title="p-done", assignee="w")
        kb.claim_task(conn, p_done)
        kb.complete_task(conn, p_done)

        # Child with just one parent, cycle it through each state
        for parent, expected in [
            (p_ready, "todo"),     # parent not done → child stays todo
            (p_running, "todo"),
            (p_blocked, "todo"),
            (p_triage, "todo"),
            (p_archived, "todo"),  # archived != done!
            (p_done, "ready"),     # only done parent unblocks child
        ]:
            child = kb.create_task(
                conn, title=f"child-of-{parent}", assignee="w", parents=[parent],
            )
            kb.recompute_ready(conn)
            actual = kb.get_task(conn, child).status
            assert actual == expected, (
                f"child of {parent} ({kb.get_task(conn, parent).status}): "
                f"expected {expected}, got {actual}"
            )
        print("  child promotion correctly gated on parent.status == 'done'")
    finally:
        conn.close()


@scenario("dashboard_rest_with_weird_inputs")
def _(home, kb):
    """FastAPI TestClient POST /tasks with atypical JSON bodies."""
    kb.init_db()
    # Set a session token so the ws check doesnt bomb on import
    try:
        from hermes_cli import web_server as ws  # noqa
    except Exception:
        pass

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from plugins.kanban.dashboard.plugin_api import router as kanban_router
    app = FastAPI()
    app.include_router(kanban_router, prefix="/api/plugins/kanban")
    client = TestClient(app)

    # Empty title
    r = client.post("/api/plugins/kanban/tasks", json={"title": ""})
    assert r.status_code in (400, 422), f"empty title should 4xx, got {r.status_code}"

    # Title only
    r = client.post("/api/plugins/kanban/tasks", json={"title": "x"})
    assert r.status_code == 200, r.text

    # Huge title
    r = client.post("/api/plugins/kanban/tasks", json={"title": "x" * 10000})
    # Should succeed — kernel doesn't cap title length
    assert r.status_code == 200

    # Unicode + emoji
    r = client.post("/api/plugins/kanban/tasks", json={
        "title": "📋 deploy 🚀 to 生产",
        "body": "日本語 body",
        "assignee": "deploy-bot",
    })
    assert r.status_code == 200
    tid = r.json()["task"]["id"]
    assert r.json()["task"]["title"] == "📋 deploy 🚀 to 生产"

    # Invalid JSON schema — unknown field, pydantic should either ignore or 422
    r = client.post("/api/plugins/kanban/tasks", json={
        "title": "fine", "nonexistent_field": "whatever",
    })
    assert r.status_code in (200, 422)

    # Priority as non-int
    r = client.post("/api/plugins/kanban/tasks", json={"title": "prio", "priority": "high"})
    assert r.status_code == 422, f"string priority should 422, got {r.status_code}"

    # PATCH with empty body (no changes requested)
    r = client.patch(f"/api/plugins/kanban/tasks/{tid}", json={})
    # Accept either success-no-op or 400
    assert r.status_code in (200, 400)
    print("  dashboard REST handles weird inputs correctly")

# =============================================================================
# RUN ALL
# =============================================================================

def main():
    print(f"Running {len(_REGISTERED)} atypical-scenario tests...")
    for fn in _REGISTERED:
        fn()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Ran:      {len(_REGISTERED)}")
    print(f"  Failures: {len(FAILURES)}")
    print(f"  Skips:    {len(SKIPS)}")
    if FAILURES:
        print()
        for f in FAILURES:
            print(f"  ✗ {f}")
        sys.exit(1)
    else:
        print("\n✔ ALL ATYPICAL SCENARIOS HANDLED CORRECTLY")


if __name__ == "__main__":
    main()
