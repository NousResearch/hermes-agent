"""Credential-free SQLite integration probe; run with --repo PATH."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if not args.child:
        with tempfile.TemporaryDirectory(prefix="kanban-identity-") as home:
            env = {"HOME": home, "HERMES_HOME": home + "/hermes", "PATH": os.defpath,
                   "LANG": "C.UTF-8", "TZ": "UTC", "PYTHONDONTWRITEBYTECODE": "1"}
            return subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--repo", str(args.repo), "--child"],
                cwd=args.repo, env=env, check=False,
            ).returncode
    sys.path.insert(0, str(args.repo))
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from hermes_cli.kanban_db_dispatch import dispatch_once
    from tools.kanban_tools import _handle_create
    from hermes_cli.kanban_decompose import _apply_fanout, _Routing
    graph = importlib.import_module("hermes_cli.kanban_db_graph") if (args.repo / "hermes_cli/kanban_db_graph.py").exists() else kb
    decompose = graph.decompose_triage_task
    specs = [{"title": "work", "assignee": "default"}]
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="prerequisite", tenant="business-a")
        root = kb.create_task(conn, title="root", triage=True, tenant="business-a", parents=[parent])
        downstream = kb.create_task(conn, title="downstream", parents=[root], tenant="business-a")
        child = kb.create_task(conn, title="manual child", parents=[parent])
        tool = json.loads(_handle_create({"title": "tool child", "assignee": "default", "parents": [parent]}))
        assert tool["ok"]
        out = {"db_tenant": kb.get_task(conn, child).tenant,
               "tool_tenant": kb.get_task(conn, tool["task_id"]).tenant}
        first = decompose(conn, root, root_assignee="default", children=specs)
        assert first and kb.get_task(conn, first[0]).tenant == "business-a"
        assert conn.execute("SELECT 1 FROM task_links WHERE parent_id=? AND child_id=?", (parent, root)).fetchone()
        assert conn.execute("SELECT 1 FROM task_links WHERE parent_id=? AND child_id=?", (root, downstream)).fetchone()
        out["legitimate_prelinked_first_fanout"] = True
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (root,))
        out["repeat_refused"] = decompose(conn, root, root_assignee="default", children=specs) is None
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (root,))
            conn.execute("UPDATE task_events SET created_at=0 WHERE task_id=?", (root,))
        kb.gc_events(conn, older_than_seconds=1)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (root,))
        out["retention_repeat_refused"] = decompose(conn, root, root_assignee="default", children=specs) is None
        ids = list(conn.execute("SELECT id FROM tasks ORDER BY id"))
        for _ in range(3):
            dispatch_once(conn, max_spawn=0)
        out["three_dispatch_ticks_stable"] = ids == list(conn.execute("SELECT id FROM tasks ORDER BY id"))
        ingress_root = kb.create_task(conn, title="ingress", tenant="business-a", triage=True)
        routing = _Routing("default", "default", True, [], {"default"})
        assert _apply_fanout(ingress_root, {"tasks": specs}, routing, "probe").ok
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (ingress_root,))
        out["fanout_ingress_repeat_refused"] = not _apply_fanout(ingress_root, {"tasks": specs}, routing, "probe").ok
        race_root = kb.create_task(conn, title="race", tenant="business-a", triage=True)
    barrier = threading.Barrier(2)
    def race():
        with kbc.connect_closing() as conn:
            barrier.wait(timeout=10)
            return decompose(conn, race_root, root_assignee="default", children=specs)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: race(), range(2)))
    out["concurrent_winners"] = sum(value is not None for value in outcomes)
    with kbc.connect_closing() as conn:
        out["integrity"] = conn.execute("PRAGMA integrity_check").fetchone()[0]
    assert out["three_dispatch_ticks_stable"] and out["concurrent_winners"] == 1 and out["integrity"] == "ok"
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
