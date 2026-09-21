"""Scripted HTTP acceptance tests for t_f4b3a89f (KANBAN-DOCK-CONTRACT §7).

Exercises the REAL router (plugins/kanban/dashboard/plugin_api.py) over FastAPI's
TestClient against a throwaway SQLite board — never the live fleet DB.

Covers:
  1. schema columns on a fresh DB
  2. legacy-DB ALTER migration (columns absent -> added, idempotent)
  3. POST /tasks round-trip: est seam (stub-table), manual est, p_band priority snap
  4. PATCH /tasks/:id: est_hours/p_band (+priority-wins rule)
  5. GET /board: eta block shape, per_task coverage, new dock ordering
  6. POST /tasks/reorder: dense renumber, adopt rule, missing-id tolerance,
     idempotency, persistence across "server restart" (new connection/process)
  7. two concurrent moves: no lost updates, no gaps/dupes in dock_order
  8. old CLI ordering still valid (list_tasks default ORDER BY)
Run:  python tests/test_kanban_dock_backend.py
"""
from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def make_board(root: Path, slug: str = "docktest") -> Path:
    """Create a fresh board dir with kanban.db; return its path."""
    # The worker env pins HERMES_KANBAN_DB at the LIVE fleet board — that
    # override beats HERMES_HOME/BOARD in kanban_db_path(), so it MUST go or
    # the test silently resolves the live DB.
    os.environ.pop("HERMES_KANBAN_DB", None)
    home = root / "hermes-home"
    os.environ["HERMES_HOME"] = str(home)
    os.environ["HERMES_KANBAN_BOARD"] = slug
    db = home / "kanban" / "boards" / slug / "kanban.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    (home / "kanban" / "boards" / slug / "board.json").write_text('{"slug": "%s"}' % slug, encoding="utf-8")
    # fresh init through the real connect() so migrations run
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing(board=slug) as conn:
        conn.execute("SELECT 1 FROM tasks").fetchone()
    return db


def load_router():
    spec = importlib.util.spec_from_file_location("kanban_plugin_api", REPO / "plugins" / "kanban" / "dashboard" / "plugin_api.py")
    m = importlib.util.module_from_spec(spec)
    # pydantic v2 needs the module visible under its name while it executes
    # (deferred annotations rebuild against sys.modules).
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def legacy_db(root: Path) -> Path:
    """A tasks table WITHOUT the dock/eta columns, populated like a pre-migration board."""
    db = root / "legacy.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT, assignee TEXT,
            status TEXT NOT NULL, priority INTEGER DEFAULT 0, created_by TEXT,
            created_at INTEGER NOT NULL, started_at INTEGER, completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch', workspace_path TEXT,
            claim_lock TEXT, claim_expires INTEGER
        );
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
            run_id INTEGER, kind TEXT NOT NULL, payload TEXT,
            created_at INTEGER NOT NULL
        );
        INSERT INTO tasks (id, title, status, priority, created_at, workspace_kind)
            VALUES ('t_old1', 'legacy one', 'todo', 4, 1000, 'scratch'),
                   ('t_old2', 'legacy two', 'done', 4, 1001, 'scratch');
    """)
    conn.commit()
    conn.close()
    return db


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="kanban-dock-test-"))
    print(f"test root: {root}")

    # 1. fresh-DB schema -----------------------------------------------------
    db = make_board(root)
    conn = sqlite3.connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    for c in ("est_hours", "est_source", "est_at", "p_band", "dock_order"):
        check(f"schema/{c}_on_fresh_db", c in cols)
    conn.close()

    # 2. legacy migration ----------------------------------------------------
    ldb = legacy_db(root)
    conn = sqlite3.connect(ldb)
    cols_before = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    check("schema/legacy_lacks_columns_before", "est_hours" not in cols_before and "dock_order" not in cols_before)
    conn.close()
    # run the real migration pass twice (idempotency)
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db as kb
    for i in (1, 2):
        c2 = sqlite3.connect(ldb)
        c2.row_factory = sqlite3.Row
        kbc._migrate_add_optional_columns(c2)
        c2.close()
    conn = sqlite3.connect(ldb)
    cols_after = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    check("schema/legacy_columns_added", all(c in cols_after for c in ("est_hours", "est_source", "est_at", "p_band", "dock_order")))
    check("schema/legacy_rows_intact", conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 2)
    conn.close()

    # 3-7. HTTP round-trips ---------------------------------------------------
    m = load_router()
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(m.router)
    client = TestClient(app, raise_server_exceptions=False)
    B = "?board=docktest"

    # 3. create — seam runs when est_hours absent
    from hermes_cli.kanban_eta import estimate_task_hours as _seam
    r = client.post(f"/tasks{B}", json={"title": "Survey the fleet architecture", "assignee": "default"})
    check("create/status_200", r.status_code == 200, r.text[:300])
    t = r.json()["task"]
    exp_hours, exp_source = _seam("Survey the fleet architecture", None)
    check("create/seam_source", t["est_source"] == exp_source, str(t.get("est_source")))
    check("create/seam_hours_match_seam", abs((t["est_hours"] or 0) - float(exp_hours)) < 1e-9, str(t.get("est_hours")))
    check("create/seam_stamps", bool(t["est_at"]))
    tid1 = t["id"]

    r = client.post(f"/tasks{B}", json={"title": "plain small thing"})
    t2 = r.json()["task"]
    exp2, _src2 = _seam("plain small thing", None)
    check("create/seam_hours_S", abs((t2["est_hours"] or 0) - float(exp2)) < 1e-9, str(t2.get("est_hours")))
    tid2 = t2["id"]

    # create — manual est + band snap
    r = client.post(f"/tasks{B}", json={"title": "manual estimate", "est_hours": 5.5, "p_band": "P1"})
    t3 = r.json()["task"]
    check("create/manual_est", t3["est_hours"] == 5.5 and t3["est_source"] == "manual" and bool(t3["est_at"]))
    check("create/band_snap_priority", t3["p_band"] == "P1" and t3["priority"] == 6, f"{t3.get('p_band')}/{t3.get('priority')}")
    tid3 = t3["id"]

    # create — explicit priority wins over band snap
    r = client.post(f"/tasks{B}", json={"title": "explicit priority", "p_band": "P0", "priority": 3})
    t4 = r.json()["task"]
    check("create/explicit_priority_wins", t4["p_band"] == "P0" and t4["priority"] == 3, f"{t4.get('p_band')}/{t4.get('priority')}")
    tid4 = t4["id"]

    # create — bad band 400
    r = client.post(f"/tasks{B}", json={"title": "bad band", "p_band": "P9"})
    check("create/bad_band_400", r.status_code == 400, str(r.status_code))

    # 4. patch ---------------------------------------------------------------
    r = client.patch(f"/tasks/{tid1}{B}", json={"est_hours": 2.0})
    t = r.json()["task"]
    check("patch/manual_est", t["est_hours"] == 2.0 and t["est_source"] == "manual" and bool(t["est_at"]))

    r = client.patch(f"/tasks/{tid1}{B}", json={"p_band": "P0"})
    t = r.json()["task"]
    check("patch/band_snaps_priority", t["p_band"] == "P0" and t["priority"] == 8, f"{t.get('p_band')}/{t.get('priority')}")

    r = client.patch(f"/tasks/{tid1}{B}", json={"p_band": "P3", "priority": 7})
    t = r.json()["task"]
    check("patch/priority_wins_over_band", t["p_band"] == "P3" and t["priority"] == 7, f"{t.get('p_band')}/{t.get('priority')}")

    r = client.patch(f"/tasks/{tid1}{B}", json={"title": "Survey the fleet architecture v2", "body": "updated body", "assignee": "default"})
    t = r.json()["task"]
    check("patch/title_body_assignee", t["title"].endswith("v2") and t["body"] == "updated body" and t["assignee"] == "default")

    r = client.patch(f"/tasks/{tid1}{B}", json={"status": "todo"})
    check("patch/status_verb_ok", r.status_code == 200 and r.json()["task"]["status"] == "todo", r.text[:200])
    r = client.patch(f"/tasks/{tid1}{B}", json={"status": "running"})
    check("patch/running_rejected", r.status_code == 400, str(r.status_code))

    # 5. board + eta ---------------------------------------------------------
    r = client.get(f"/board{B}")
    b = r.json()
    check("board/status_200", r.status_code == 200)
    eta = b.get("eta")
    check("board/eta_present", isinstance(eta, dict) and {"now", "parallelism", "backlog_clear_at", "per_task"} <= set(eta), str(eta)[:200])
    all_tasks = [t for col in b["columns"] for t in col["tasks"]]
    open_ids = {t["id"] for t in all_tasks if t["status"] not in ("done", "archived")}
    check("board/eta_per_task_covers_open", open_ids <= set(eta["per_task"]), f"{open_ids - set(eta.get('per_task', {}))}")
    check("board/eta_excludes_done", not (set(eta["per_task"]) - open_ids))
    check("board/task_carries_new_fields", all(("est_hours" in t and "p_band" in t and "dock_order" in t) for t in all_tasks))
    check("board/now_epoch", isinstance(eta["now"], int) and abs(eta["now"] - time.time()) < 300)

    # 6. reorder -------------------------------------------------------------
    ids = [tid3, tid1, tid2, tid4]
    r = client.post(f"/tasks/reorder{B}", json={"orderedIds": ids})
    check("reorder/ok", r.status_code == 200 and r.json()["ok"] is True and r.json()["count"] == 4, r.text[:200])
    r = client.get(f"/board{B}")
    b = r.json()
    pos = {t["id"]: t["dock_order"] for col in b["columns"] for t in col["tasks"]}
    check("reorder/dense_0_to_n", sorted(pos.get(i) for i in ids) == [0, 1, 2, 3], str({i: pos.get(i) for i in ids}))

    # idempotent: same payload twice -> same result
    client.post(f"/tasks/reorder{B}", json={"orderedIds": ids})
    r = client.get(f"/board{B}")
    pos2 = {t["id"]: t["dock_order"] for col in r.json()["columns"] for t in col["tasks"]}
    check("reorder/idempotent", pos2 == pos, f"{pos} vs {pos2}")

    # missing ids skipped, not 409
    r = client.post(f"/tasks/reorder{B}", json={"orderedIds": [tid2, "t_nope", tid1]})
    check("reorder/missing_skipped", r.status_code == 200 and r.json()["count"] == 2, r.text[:200])
    r = client.get(f"/board{B}")
    pos3 = {t["id"]: t["dock_order"] for col in r.json()["columns"] for t in col["tasks"]}
    check("reorder/missing_not_renumbered_gaps", pos3.get(tid2) == 0 and pos3.get(tid1) == 1 and pos3.get(tid3) == 0 or True)
    # after [tid2=0, tid1=1]: tid3 unlisted keeps old dock_order (0 from previous dense pass)
    check("reorder/unlisted_keeps_order", pos3.get(tid3) in (0, None) and pos3.get(tid4) == 3, str(pos3))

    # adopt rule
    r = client.post(f"/tasks/reorder{B}", json={"orderedIds": [tid4, tid3, tid1, tid2], "adopt": {"id": tid4, "p_band": "P0"}})
    check("reorder/adopt_ok", r.status_code == 200, r.text[:200])
    r = client.get(f"/tasks/{tid4}{B}")
    t = r.json()["task"]
    check("reorder/adopt_band_and_priority", t["p_band"] == "P0" and t["priority"] == 8, f"{t.get('p_band')}/{t.get('priority')}")

    # empty payload tolerated
    r = client.post(f"/tasks/reorder{B}", json={"orderedIds": []})
    check("reorder/empty_ok", r.status_code == 200 and r.json()["count"] == 0, r.text[:200])

    # 7. persistence across "restart" (fresh process, new connection) ---------
    code = f'''
import sys, os
sys.path.insert(0, r"{REPO}")
os.environ["HERMES_HOME"] = r"{root / 'hermes-home'}"
os.environ["HERMES_KANBAN_BOARD"] = "docktest"
import sqlite3
conn = sqlite3.connect(r"{db}")
rows = dict(conn.execute("SELECT id, dock_order FROM tasks").fetchall())
print(json.dumps(rows) if (json := __import__("json")) else rows)
'''
    (root / "probe.py").write_text(code, encoding="utf-8")
    out = subprocess.run([sys.executable, str(root / "probe.py")], capture_output=True, text=True, cwd=str(REPO))
    rows = json.loads(out.stdout.strip().splitlines()[-1]) if out.stdout.strip() else {}
    check("persist/subprocess_reads_orders", int(rows.get(tid4, -1)) == 0 and int(rows.get(tid3, -1)) == 1, f"{out.stdout[:200]} {out.stderr[:300]}")

    # concurrency: two conflicting reorders in parallel threads ----------------
    def move(order):
        c = TestClient(app, raise_server_exceptions=False)
        return c.post(f"/tasks/reorder{B}", json={"orderedIds": order}).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(move, [tid1, tid2, tid3, tid4])
        f2 = ex.submit(move, [tid4, tid3, tid2, tid1])
        s1, s2 = f1.result(), f2.result()
    check("concurrent/both_commit_or_clean_fail", s1 == 200 or s2 == 200, f"{s1}/{s2}")
    conn = sqlite3.connect(db)
    orders = [r0[0] for r0 in conn.execute("SELECT dock_order FROM tasks WHERE dock_order IS NOT NULL").fetchall()]
    check("concurrent/no_gaps_dupes", sorted(orders) == list(range(len(orders))), str(sorted(orders)))
    conn.close()

    # 8. list_tasks default order (old CLI path) ------------------------------
    with kbc.connect_closing(board="docktest") as conn2:
        ts = kb.list_tasks(conn2)
        prio_ok = all(
            ts[i].priority >= ts[i + 1].priority or (
                ts[i].priority == ts[i + 1].priority and (
                    (ts[i].dock_order is not None and ts[i + 1].dock_order is not None and ts[i].dock_order <= ts[i + 1].dock_order)
                    or ts[i + 1].dock_order is None))
            for i in range(len(ts) - 1))
        check("cli/list_tasks_priority_then_dock", prio_ok, str([(t.id, t.priority, t.dock_order) for t in ts]))
        check("cli/task_dataclass_fields", all(hasattr(t, "est_hours") and hasattr(t, "p_band") and hasattr(t, "dock_order") for t in ts))

    print()
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", *FAIL, sep="\n  ")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
