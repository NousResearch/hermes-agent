"""Helpers for the incremental ``GET /board`` tests: the plugin router, seeded cards, the reference client
merge and the reference board build (``GET /board`` as it was before the cache)."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb

BOARD = "/api/plugins/kanban/board"
T0 = 1_800_000_000
PLUGIN_API = Path(__file__).resolve().parents[3] / "plugins" / "kanban" / "dashboard" / "plugin_api.py"


class Clock:
    def __init__(self, start: int) -> None:
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def load_plugin(name: str = "kanban_board_delta_test"):
    spec = importlib.util.spec_from_file_location(name, PLUGIN_API)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def client_for(plugin) -> TestClient:
    app = FastAPI()
    app.include_router(plugin.router, prefix="/api/plugins/kanban")
    return TestClient(app)


def connect():
    from hermes_cli import kanban_db_connect as kbc
    return kbc.connect()


def done_id(i: int) -> str:
    return f"t_done{i:05d}"


def seed_done(conn, n: int) -> list[str]:
    """``n`` finished cards of realistic weight (a 2 KB body, a 1.5 KB summary, three events)."""
    body = "Goal and acceptance text. " * 80
    summary = "What was done and verified, with the evidence. " * 32
    ids = [done_id(i) for i in range(n)]
    cur = conn.cursor()
    cur.execute("BEGIN")
    for i, tid in enumerate(ids):
        created = T0 - 86_400 * 5 + i * 20
        done_at = created + 600
        cur.execute(
            "INSERT INTO tasks (id, title, body, assignee, status, priority, created_by, created_at, "
            "started_at, completed_at, workspace_kind, result) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (tid, f"done card {i}", body, "worker", "done", 0, "user", created, created + 5, done_at,
             "scratch", None),
        )
        cur.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome, summary) "
            "VALUES (?,?,?,?,?,?,?)", (tid, "worker", "done", created + 5, done_at, "completed", summary),
        )
        run_id = cur.lastrowid
        for kind, ts in (("created", created), ("claimed", created + 5), ("completed", done_at)):
            cur.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?,?,?,?,?)",
                (tid, run_id if kind != "created" else None, kind, None, ts),
            )
    cur.execute("COMMIT")
    return ids


def get(env, params: str = "", headers: dict | None = None):
    return env.client.get(BOARD + (("?" + params) if params else ""), headers=headers or {})


def poll(env, held: dict):
    """The desktop's own request: the held board's ``since`` and ``etag`` in the URL."""
    r = get(env, f"since={held['latest_event_id']}&etag={held['etag']}")
    assert r.status_code == 200
    return r, r.json()


def cards_of(board: dict) -> dict[str, dict]:
    return {t["id"]: t for col in board["columns"] for t in col["tasks"]}


def refresh_ages(board: dict) -> None:
    """The contract: ``age`` is a function of the response clock, so a merged board refreshes it."""
    now = board["now"]
    for col in board["columns"]:
        for t in col["tasks"]:
            c, s, co = t["created_at"], t["started_at"], t["completed_at"]
            t["age"] = {
                "created_age_seconds": now - c if c is not None else None,
                "started_age_seconds": now - s if s is not None else None,
                "time_to_complete_seconds": co - (s or c) if co is not None else None,
            }


def merge_delta(held: dict, delta: dict) -> dict:
    """Reference client: the merge rule the contract states, exactly."""
    out = copy.deepcopy(held)
    out["now"], out["etag"], out["latest_event_id"] = delta["now"], delta["etag"], delta["latest_event_id"]
    if delta.get("unchanged"):
        refresh_ages(out)
        return out
    cards = cards_of(held)
    for tid in delta["removed"]:
        cards.pop(tid, None)
    for c in delta["cards"]:
        cards[c["id"]] = copy.deepcopy(c)
    out["columns"] = [{"name": co["name"], "tasks": [cards[i] for i in co["ids"]]} for co in delta["column_order"]]
    out["tenants"] = delta["tenants"]
    out["assignees"] = delta["assignees"]
    refresh_ages(out)
    return out


def sent_ids(delta: dict) -> set[str]:
    return {c["id"] for c in delta.get("cards", [])}


def reference_board(plugin, *, tenant=None, include_archived=False, board=None,
                    workflow_template_id=None, current_step_key=None) -> dict:
    """``GET /board`` as it was before the cache, verbatim: every card rebuilt on every call."""
    with plugin._board_conn(board) as (board, conn):
        tasks = kb.list_tasks(
            conn, tenant=tenant, include_archived=include_archived,
            workflow_template_id=workflow_template_id, current_step_key=current_step_key)
        link_counts: dict = {}
        for row in conn.execute("SELECT parent_id, child_id FROM task_links").fetchall():
            link_counts.setdefault(row["parent_id"], {"parents": 0, "children": 0})["children"] += 1
            link_counts.setdefault(row["child_id"], {"parents": 0, "children": 0})["parents"] += 1
        comment_counts = {
            r["task_id"]: r["n"] for r in conn.execute("SELECT task_id, COUNT(*) AS n FROM task_comments GROUP BY task_id")}
        progress: dict = {}
        for row in conn.execute(
                "SELECT l.parent_id AS pid, t.status AS cstatus FROM task_links l JOIN tasks t ON t.id = l.child_id").fetchall():
            p = progress.setdefault(row["pid"], {"done": 0, "total": 0})
            p["total"] += 1
            p["done"] += row["cstatus"] == "done"
        diagnostics_per_task = plugin._compute_task_diagnostics(conn, task_ids=None)
        latest_event_id = conn.execute("SELECT COALESCE(MAX(id), 0) AS m FROM task_events").fetchone()["m"]
        columns = {c: [] for c in plugin.BOARD_COLUMNS}
        if include_archived:
            columns["archived"] = []
        summary_map = kb.latest_summaries(conn, [t.id for t in tasks])
        run_start_map = kb.current_run_started_ats(conn, [t.id for t in tasks])
        for t in tasks:
            full = summary_map.get(t.id)
            d = plugin._task_dict(t, latest_summary=(full[:200] if full else None),
                                  current_run_started_at=run_start_map.get(t.id))
            d["link_counts"] = link_counts.get(t.id, {"parents": 0, "children": 0})
            d["comment_count"] = comment_counts.get(t.id, 0)
            d["progress"] = progress.get(t.id)
            plugin._attach_diagnostics(d, diagnostics_per_task.get(t.id))
            columns[t.status if t.status in columns else "todo"].append(d)
        columns["done"].sort(key=lambda d: d["id"], reverse=True)
        columns["done"].sort(key=lambda d: (d["completed_at"] is None, -(d["completed_at"] or 0)))
        tenants = [r["tenant"] for r in conn.execute("SELECT DISTINCT tenant FROM tasks WHERE tenant IS NOT NULL ORDER BY tenant")]
        assignees = [e["name"] for e in kb.known_assignees(conn)]
        out = {
            "columns": [{"name": n, "tasks": columns[n]} for n in columns], "tenants": tenants,
            "assignees": assignees, "latest_event_id": int(latest_event_id), "now": int(time.time())}
    return json.loads(json.dumps(out))


def without_etag(board: dict) -> dict:
    return {k: v for k, v in board.items() if k != "etag"}


def diff_boards(a: dict, b: dict) -> list[str]:
    """Human-readable differences; empty means identical."""
    out: list[str] = []
    for key in sorted(set(a) | set(b)):
        if key == "columns":
            continue
        if a.get(key) != b.get(key):
            out.append(f"top-level {key}: {str(a.get(key))[:80]} != {str(b.get(key))[:80]}")
    ca = {c["name"]: c["tasks"] for c in a["columns"]}
    cb = {c["name"]: c["tasks"] for c in b["columns"]}
    if list(ca) != list(cb):
        out.append(f"column names {list(ca)} != {list(cb)}")
    for name, la in ca.items():
        lb = cb.get(name, [])
        if [t["id"] for t in la] != [t["id"] for t in lb]:
            out.append(f"column {name}: id order differs ({len(la)} vs {len(lb)})")
            continue
        for x, y in zip(la, lb):
            if x != y:
                keys = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
                out.append(f"card {x['id']} in {name}: fields {keys}")
    return out


def diag_shape(card: dict) -> list:
    return [(d["kind"], d["severity"], d.get("count", 1)) for d in card.get("diagnostics") or []]


def modulo_clock_text(held: dict, ref: dict) -> dict:
    """``held`` with the one thing a delta-held client may hold older than a fresh snapshot put right:
    the clock text of a diagnostic (title, detail, data, ``*_seen_at``, ``warnings.latest_at``) on a card
    whose diagnostics are the same in kind, severity, count and actions. Any other difference stays, so
    the comparison against the reference still fails on it (see test_clock_drift.py)."""
    out = copy.deepcopy(held)
    ref_cards = cards_of(ref)
    for col in out["columns"]:
        for i, card in enumerate(col["tasks"]):
            want = ref_cards.get(card["id"])
            if want is None or card == want:
                continue
            rest = {k for k in set(card) | set(want) if card.get(k) != want.get(k)}
            if not rest <= {"diagnostics", "warnings"}:
                continue

            def state(c):
                return [(d["kind"], d["severity"], d.get("count", 1), d.get("actions")) for d in c.get("diagnostics") or []]

            def summary(c):
                w = dict(c.get("warnings") or {})
                w.pop("latest_at", None)
                return w

            if state(card) == state(want) and summary(card) == summary(want):
                col["tasks"][i] = want
    return out
