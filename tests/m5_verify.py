"""M5 集成验证：终结报告接入推送管线.

Tests:
  1. _all_descendants_terminal — all terminal / some active / no children
  2. build_pipeline_summary — 3-layer pipeline produces structured report
  3. _build_pipeline_report conditions — root+terminal→report, non-root→"",
     children-not-terminal→"", leaf→""

Run standalone:
    cd /home/zml/workspace/hermes-agent && python3 tests/m5_verify.py
"""
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, "/home/zml/workspace/hermes-agent")

from gateway.kanban_watchers import (
    _all_descendants_terminal,
    _PIPELINE_ACTIVE_STATUSES,
    _is_root_task,
    build_pipeline_summary,
)

passed = 0
failed = 0


def check(name, got, want):
    global passed, failed
    ok = got == want
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  {'PASS' if ok else 'FAIL'}: {name}: got={got!r}, want={want!r}")


def make_db():
    """Create an in-memory kanban DB with the real schema."""
    from hermes_cli.kanban_db import SCHEMA_SQL
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    return conn


def insert_task(conn, tid, title, status="done", assignee="coder"):
    conn.execute(
        "INSERT INTO tasks (id, title, status, assignee, created_at) "
        "VALUES (?,?,?,?,?)",
        (tid, title, status, assignee, 1000),
    )


def link(conn, parent, child):
    conn.execute(
        "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?,?)",
        (parent, child),
    )


def insert_run(conn, task_id, summary="all good", decisions=None, changed_files=None):
    import json
    cur = conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, outcome, summary, started_at, ended_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (task_id, "coder", "done", "completed", summary, 1000, 2000),
    )
    run_id = cur.lastrowid
    meta = {}
    if decisions:
        meta["decisions"] = decisions
    if changed_files:
        meta["changed_files"] = changed_files
    if meta:
        conn.execute(
            "UPDATE task_runs SET metadata=? WHERE id=?",
            (json.dumps(meta), run_id),
        )
    # Link the task to this run as its current run.
    conn.execute(
        "UPDATE tasks SET current_run_id=? WHERE id=?",
        (run_id, task_id),
    )


# ============================================================================
# 1. _all_descendants_terminal
# ============================================================================
print("\n=== 1. _all_descendants_terminal ===")

# 1a. All descendants done → True
conn = make_db()
insert_task(conn, "root", "Root", "done")
insert_task(conn, "c1", "Child 1", "done")
insert_task(conn, "c2", "Child 2", "done")
insert_task(conn, "g1", "Grandchild 1", "done")
link(conn, "root", "c1")
link(conn, "root", "c2")
link(conn, "c1", "g1")
conn.commit()
check("all descendants done → True",
    _all_descendants_terminal(conn, "root"), True)

# 1b. One child still running → False
conn.execute("UPDATE tasks SET status='running' WHERE id='c2'")
conn.commit()
check("one child running → False",
    _all_descendants_terminal(conn, "root"), False)

# 1c. One child blocked → False
conn.execute("UPDATE tasks SET status='blocked' WHERE id='c2'")
conn.commit()
check("one child blocked → False",
    _all_descendants_terminal(conn, "root"), False)

# 1d. Archived (terminal) → True
conn.execute("UPDATE tasks SET status='archived' WHERE id='c2'")
conn.commit()
check("archived child → True",
    _all_descendants_terminal(conn, "root"), True)

# 1e. No children → False (not a pipeline)
conn2 = make_db()
insert_task(conn2, "solo", "Solo task", "done")
conn2.commit()
check("no children → False",
    _all_descendants_terminal(conn2, "solo"), False)

# 1f. _PIPELINE_ACTIVE_STATUSES sanity
check("active set has 7 statuses",
    len(_PIPELINE_ACTIVE_STATUSES), 7)
check("done not in active set",
    "done" not in _PIPELINE_ACTIVE_STATUSES, True)
check("archived not in active set",
    "archived" not in _PIPELINE_ACTIVE_STATUSES, True)
check("running in active set",
    "running" in _PIPELINE_ACTIVE_STATUSES, True)


# ============================================================================
# 2. build_pipeline_summary — 3-layer pipeline
# ============================================================================
print("\n=== 2. build_pipeline_summary (3-layer pipeline) ===")

conn = make_db()
# A → B → C → V  (3 layers under root A)
insert_task(conn, "A", "Root Pipeline", "done", assignee="architect")
insert_task(conn, "B", "Build phase", "done", assignee="coder")
insert_task(conn, "C", "Test phase", "done", assignee="tester")
insert_task(conn, "V", "Verify phase", "done", assignee="reviewer")
link(conn, "A", "B")
link(conn, "B", "C")
link(conn, "C", "V")
insert_run(conn, "A", summary="pipeline complete", decisions=["use approach X"])
insert_run(conn, "B", summary="built feature", changed_files=["src/main.py"])
insert_run(conn, "C", summary="tests pass")
insert_run(conn, "V", summary="verified")
conn.commit()

check("A is root task", _is_root_task(conn, "A"), True)
check("B is NOT root task", _is_root_task(conn, "B"), False)
check("V is NOT root task", _is_root_task(conn, "V"), False)
check("all descendants of A terminal", _all_descendants_terminal(conn, "A"), True)

report = build_pipeline_summary("A", conn)
check("report is non-empty string", isinstance(report, str) and len(report) > 0, True)
check("report contains pipeline title",
    "Root Pipeline" in report, True)
check("report contains total task count",
    "总任务: 4" in report, True)
check("report contains pass count",
    "通过: 4" in report, True)
check("report contains 耗时", "耗时:" in report, True)
check("report contains 下一步建议", "下一步建议:" in report, True)
print(f"\n  --- Report preview ---\n{report}")


# ============================================================================
# 3. build_pipeline_summary edge cases
# ============================================================================
print("\n=== 3. build_pipeline_summary edge cases ===")

# 3a. Non-root task → empty string
report_nonroot = build_pipeline_summary("B", conn)
check("non-root task → empty string", report_nonroot, "")

# 3b. Leaf task → empty string
report_leaf = build_pipeline_summary("V", conn)
check("leaf task → empty string", report_leaf, "")

# 3c. Nonexistent task → empty string
report_missing = build_pipeline_summary("ZZZ", conn)
check("nonexistent task → empty string", report_missing, "")

# 3d. None task_id → empty string
report_none = build_pipeline_summary(None, conn)
check("None task_id → empty string", report_none, "")

# 3e. Summary with some failed tasks
conn3 = make_db()
insert_task(conn3, "root", "Failed Pipeline", "done")
insert_task(conn3, "ok", "Success child", "done")
insert_task(conn3, "bad", "Failed child", "archived")
link(conn3, "root", "ok")
link(conn3, "root", "bad")
conn3.commit()
report_mixed = build_pipeline_summary("root", conn3)
check("mixed pipeline report non-empty",
    isinstance(report_mixed, str) and len(report_mixed) > 0, True)
check("mixed report shows 1 failure",
    "失败: 1" in report_mixed, True)


# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*50}")
print(f"Total: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
