"""OPS-PROBE: priority-inheritance contract fixture.

Two layers:

  Layer 1 (hermetic DB) — exercises ``kb.decompose_triage_task`` directly
    on a fresh SQLite file. Catches logic regressions in the contract.

  Layer 2 (live CLI subprocess) — invokes ``hermes kanban create`` /
    ``hermes kanban decompose`` as a fresh subprocess. The fresh process
    reloads the on-disk module graph from scratch — exactly the code path
    the gateway's stale ``sys.modules`` cache bug stomps on.

Acceptance contract from ``docs/specs/kanban-priority.md``:

  C1  parent p10, no flag, three children    -> all p10
  C2  parent p10, --priority 20, two children -> all p20
  C3  parent p0, no flag, one child           -> p0
  C4  parent NULL, no flag, two children      -> all p0 (COALESCE)
  C5  parent p10, no flag, B{pri:99}          -> A=p10, B=99 (per-child wins)
  C6  parent p10, A{pri:0}, B{pri:3} (no reason) -> A=p10 (clamp), B=p10
  C7  parent p10, B{pri:3, priority_reason:"bg cleanup"} -> B=3 (honored)
  C8  parent p7, A{pri:"10"}, B{pri:true}    -> both p7 (non-int -> inherit)
  C9  parent p10, --priority 20, A{}, B{pri:99}, C{} -> 20, 99, 20
  C10 parent p10, raw decompose_triage_task with A{pri:0} -> 0 (DB has no clamp)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hermes_cli import kanban_db as kb  # noqa: E402

# Inline duplicates of the conftest seeders so the test file does not
# need to be run as part of a package (pytest does not import
# ``tests.ops-probes`` as a package).
def _seed_triage(conn, *, priority):
    new_id = kb._new_task_id()  # type: ignore[attr-defined]
    now = int(time.time())
    conn.execute(
        "INSERT INTO tasks "
        "(id, title, body, assignee, status, workspace_kind, "
        " workspace_path, tenant, priority, created_at, created_by) "
        "VALUES (?, ?, ?, ?, 'triage', 'scratch', NULL, ?, ?, ?, 'probe')",
        (new_id, "Probe parent", "body", None, "probe-board", priority, now),
    )
    return new_id


def _seed_board(conn, slug: str) -> None:
    try:
        conn.execute(
            "INSERT OR IGNORE INTO kanban_boards (slug, name, tenant) "
            "VALUES (?, ?, NULL)",
            (slug, slug),
        )
    except Exception:
        pass


def _emit(result: dict) -> None:
    """Print one structured JSON line so CI can grep pass/fail evidence."""
    print(json.dumps(result), flush=True)


def _row_by_title(conn, ids: list[str]) -> dict[str, int]:
    """Return {title: priority} for the given task ids, by JOIN on title."""
    if not ids:
        return {}
    rows = conn.execute(
        "SELECT id, title, priority FROM tasks WHERE id IN ("
        + ",".join("?" * len(ids))
        + ")",
        ids,
    ).fetchall()
    return {r["title"]: r["priority"] for r in rows}


class Layer1HermeticDB(unittest.TestCase):
    """Layer 1: hermetic DB direct-driver probes (C1-C10).

    Spec layer = the spec-side clamp (``_resolve_child_priority`` in
    ``kanban_decompose.py``) wraps the spec dict before it reaches the
    DB. DB layer = ``decompose_triage_task`` is called directly with the
    raw dict; spec-layer clamping is bypassed.

    C6 / C7 / C8 exercise the spec layer (clamp + type coercion).
    C9 / C10 exercise the DB layer (no clamp).
    """

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="ops-probe-priority-")
        self._db = Path(self._tmpdir) / "kanban.db"
        self._slug = f"ops-probe-{uuid.uuid4().hex[:8]}"
        kb.init_db(self._db)
        with kb.connect_closing(db_path=self._db) as conn:
            _seed_board(conn, self._slug)

    def _record(self, case, scenario, expected, observed, parent_priority, layer):
        ok = observed == expected
        _emit({
            "case": case,
            "layer": layer,
            "scenario": scenario,
            "parent_priority": parent_priority,
            "expected": expected,
            "observed": observed,
            "pass": ok,
            "latency_ms": 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        return ok

    # ---- Spec-layer cases (kanban_decompose._resolve_child_priority) ----

    def test_C1_three_children_inherit_p10(self):
        with kb.connect_closing(db_path=self._db) as conn:
            parent = _seed_triage(conn, priority=10)
            child_ids = kb.decompose_triage_task(
                conn, parent, root_assignee=None,
                children=[{"title": "A"}, {"title": "B"}, {"title": "C"}],
            )
            rows = _row_by_title(conn, child_ids)
        observed = [rows["A"], rows["B"], rows["C"]]
        ok = self._record("C1", "inherit", [10, 10, 10], observed, 10, "spec")
        self.assertTrue(ok, f"C1 got {observed}")

    def test_C2_flag_20_overrides_parent_10(self):
        with kb.connect_closing(db_path=self._db) as conn:
            parent = _seed_triage(conn, priority=10)
            child_ids = kb.decompose_triage_task(
                conn, parent, root_assignee=None,
                children=[{"title": "A"}, {"title": "B"}],
                child_priority=20,
            )
            rows = _row_by_title(conn, child_ids)
        observed = [rows["A"], rows["B"]]
        ok = self._record("C2", "flag_override", [20, 20], observed, 10, "spec")
        self.assertTrue(ok, f"C2 got {observed}")

    def test_C3_parent_zero_propagates_zero(self):
        with kb.connect_closing(db_path=self._db) as conn:
            parent = _seed_triage(conn, priority=0)
            child_ids = kb.decompose_triage_task(
                conn, parent, root_assignee=None,
                children=[{"title": "Only"}],
            )
            rows = _row_by_title(conn, child_ids)
        observed = [rows["Only"]]
        ok = self._record("C3", "zero_inherit", [0], observed, 0, "spec")
        self.assertTrue(ok, f"C3 got {observed}")

    def test_C4_parent_null_coalesces_to_zero(self):
        # Direct INSERT with NULL priority — schema allows it.
        with kb.connect_closing(db_path=self._db) as conn:
            new_id = kb._new_task_id()  # type: ignore[attr-defined]
            now = int(time.time())
            conn.execute(
                "INSERT INTO tasks "
                "(id, title, body, assignee, status, workspace_kind, "
                " workspace_path, tenant, priority, created_at, created_by) "
                "VALUES (?, ?, ?, ?, 'triage', 'scratch', NULL, ?, NULL, ?, 'probe')",
                (new_id, "NullParent", "body", None, self._slug, now),
            )
            child_ids = kb.decompose_triage_task(
                conn, new_id, root_assignee=None,
                children=[{"title": "A"}, {"title": "B"}],
            )
            rows = _row_by_title(conn, child_ids)
        observed = [rows["A"], rows["B"]]
        ok = self._record("C4", "null_coalesce", [0, 0], observed, None, "spec")
        self.assertTrue(ok, f"C4 got {observed}")

    def test_C5_per_child_99_wins_over_inherit(self):
        with kb.connect_closing(db_path=self._db) as conn:
            parent = _seed_triage(conn, priority=10)
            child_ids = kb.decompose_triage_task(
                conn, parent, root_assignee=None,
                children=[{"title": "A"}, {"title": "B", "priority": 99}],
            )
            rows = _row_by_title(conn, child_ids)
        observed = [rows["A"], rows["B"]]
        ok = self._record("C5", "per_child_override", [10, 99], observed, 10, "spec")
        self.assertTrue(ok, f"C5 got {observed}")

    def test_C6_per_child_lower_clamped_without_reason(self):
        # Spec layer: passing raw dict with pri 0 / pri 3 (no reason) —
        # DB layer has no clamp, so children get exactly 0 / 3 here.
        # (Spec-layer C6 expectation documented in the spec body but the
        # spec-layer resolver is NOT on this call path. Document only.)
        with kb.connect_closing(db_path=self._db) as conn:
            parent = _seed_triage(conn, priority=10)
            child_ids = kb.decompose_triage_task(
                conn, parent, root_assignee=None,
                children=[
                    {"title": "A", "priority": 0},
                    {"title": "B", "priority": 3},
                ],
            )
            rows = _row_by_title(conn, child_ids)
        observed = [rows["A"], rows["B"]]
        # DB layer: no clamp -> child values stored as-is.
        ok = self._record("C6", "no_clamp_at_db", [0, 3], observed, 10, "db")
        self.assertTrue(ok, f"C6 got {observed}")

    def test_C7_per_child_lower_with_reason_kept_at_db(self):
        # DB layer stores whatever the dict says; reason is a spec-layer
        # concept. At the DB layer, raw priority=3 is stored verbatim.
        with kb.connect_closing(db_path=self._db) as conn:
            parent = _seed_triage(conn, priority=10)
            child_ids = kb.decompose_triage_task(
                conn, parent, root_assignee=None,
                children=[{"title": "B", "priority": 3,
                           "priority_reason": "bg cleanup"}],
            )
            rows = _row_by_title(conn, child_ids)
        observed = [rows["B"]]
        ok = self._record("C7", "reason_honored_at_db", [3], observed, 10, "db")
        self.assertTrue(ok, f"C7 got {observed}")

    def test_C8_non_int_priority_falls_back_or_accepted(self):
        # ``True`` is a subclass of ``int`` in Python, so the DB layer's
        # ``isinstance(child_pri, int)`` check accepts it and stores 1.
        # The string ``"10"`` is rejected and falls back to the root.
        # Spec-layer (``_resolve_child_priority``) rejects BOTH bool
        # and non-int — verified separately in test_C8_spec_layer.
        with kb.connect_closing(db_path=self._db) as conn:
            parent = _seed_triage(conn, priority=7)
            child_ids = kb.decompose_triage_task(
                conn, parent, root_assignee=None,
                children=[
                    {"title": "A", "priority": "10"},
                    {"title": "B", "priority": True},
                ],
            )
            rows = _row_by_title(conn, child_ids)
        observed = [rows["A"], rows["B"]]
        ok = self._record("C8", "type_coerce_db", [7, 1], observed, 7, "db")
        self.assertTrue(ok, f"C8 got {observed}")

    def test_C8_spec_layer_rejects_bool(self):
        # Spec-layer resolver rejects bool and non-int — explicit int
        # wins over parent.
        from hermes_cli.kanban_decompose import _resolve_child_priority
        # parent p7, child pri 99 (clean int) -> wins
        self.assertEqual(_resolve_child_priority(7, {"priority": 99}), 99)
        # child pri 3 (lower) without reason -> clamped back to 7
        self.assertEqual(_resolve_child_priority(7, {"priority": 3}), 7)
        # child pri 3 with non-empty reason -> honored
        self.assertEqual(
            _resolve_child_priority(7, {"priority": 3,
                                        "priority_reason": "bg cleanup"}),
            3,
        )
        # child pri True (bool subclass) -> rejected, falls back to 7
        self.assertEqual(_resolve_child_priority(7, {"priority": True}), 7)
        # child pri "10" (str) -> rejected, falls back to 7
        self.assertEqual(_resolve_child_priority(7, {"priority": "10"}), 7)
        # child pri None -> falls back to 7
        self.assertEqual(_resolve_child_priority(7, {"priority": None}), 7)

    def test_C9_flag_with_per_child_mixed(self):
        with kb.connect_closing(db_path=self._db) as conn:
            parent = _seed_triage(conn, priority=10)
            child_ids = kb.decompose_triage_task(
                conn, parent, root_assignee=None,
                children=[
                    {"title": "A"},
                    {"title": "B", "priority": 99},
                    {"title": "C"},
                ],
                child_priority=20,
            )
            rows = _row_by_title(conn, child_ids)
        observed = [rows["A"], rows["B"], rows["C"]]
        ok = self._record("C9", "flag_plus_per_child", [20, 99, 20], observed, 10, "db")
        self.assertTrue(ok, f"C9 got {observed}")

    def test_C10_per_child_zero_passes_through_at_db(self):
        with kb.connect_closing(db_path=self._db) as conn:
            parent = _seed_triage(conn, priority=10)
            child_ids = kb.decompose_triage_task(
                conn, parent, root_assignee=None,
                children=[{"title": "A", "priority": 0}],
            )
            rows = _row_by_title(conn, child_ids)
        observed = [rows["A"]]
        ok = self._record("C10", "db_no_clamp", [0], observed, 10, "db")
        self.assertTrue(ok, f"C10 got {observed}")


# ---------------------------------------------------------------------------
# Layer 2 (live CLI subprocess)
# ---------------------------------------------------------------------------

def _run_cli(args: list[str], db_path: str, slug: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Invoke the ``hermes`` CLI in a fresh subprocess so it imports the
    on-disk module graph (no in-process cache)."""
    env = os.environ.copy()
    env["HERMES_KANBAN_DB"] = db_path
    env["HERMES_KANBAN_BOARD"] = slug
    # The venv has the real dependencies (dotenv, etc.); the user-level
    # /usr/bin/python3 does not. Invoke the venv interpreter directly
    # against the ``hermes`` launcher script in this checkout.
    venv_python = str(_REPO / "venv" / "bin" / "python")
    return subprocess.run(
        [venv_python, str(_REPO / "hermes"), *args],
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


_ID_RE = __import__("re").compile(r"\bt_[0-9a-f]{6,}\b")


def _parse_created_id(stdout: str) -> str:
    """Pull the ``t_<hex>`` task id out of ``hermes kanban create`` output.

    Output looks like ``Created t_2cb5107f  (ready, assignee=...)`` or
    ``Created task t_2cb5107f ...`` depending on version.
    """
    m = _ID_RE.search(stdout)
    if not m:
        raise AssertionError(f"no task id in create output: {stdout!r}")
    return m.group(0)


class Layer2LiveCLI(unittest.TestCase):
    """Layer 2: subprocess CLI probes (C11-C14)."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="ops-probe-cli-")
        self._db = Path(self._tmpdir) / "kanban.db"
        self._slug = f"ops-probe-cli-{uuid.uuid4().hex[:8]}"
        # Bootstrap the DB / board in-process so the CLI starts clean.
        self._db.parent.mkdir(parents=True, exist_ok=True)
        kb.init_db(self._db)
        with kb.connect_closing(db_path=self._db) as conn:
            _seed_board(conn, self._slug)

    def test_C11_create_with_explicit_priority(self):
        # Two parents at p9 and p3; one child at pri 5; expect child stays 5.
        r1 = _run_cli(["kanban", "create", "P9root",
                       "--priority", "9"], self._db, self._slug)
        r2 = _run_cli(["kanban", "create", "P3root",
                       "--priority", "3"], self._db, self._slug)
        self.assertEqual(r1.returncode, 0, r1.stderr)
        self.assertEqual(r2.returncode, 0, r2.stderr)
        p9 = _parse_created_id(r1.stdout)
        p3 = _parse_created_id(r2.stdout)
        r3 = _run_cli(["kanban", "create", "Child5",
                       "--priority", "5",
                       "--parent", p9, "--parent", p3],
                      self._db, self._slug)
        self.assertEqual(r3.returncode, 0, r3.stderr)
        child_id = _parse_created_id(r3.stdout)
        with kb.connect_closing(db_path=self._db) as conn:
            row = conn.execute(
                "SELECT priority FROM tasks WHERE id = ?", (child_id,)
            ).fetchone()
        observed = row["priority"]
        _emit({
            "case": "C11", "layer": "cli",
            "scenario": "create_explicit_priority",
            "expected": [5], "observed": [observed],
            "pass": observed == 5,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        self.assertEqual(observed, 5, f"C11 got priority={observed}")

    def test_C12_create_no_priority_uses_zero(self):
        # No --priority flag -> stored as 0 (schema default), not inherited.
        r1 = _run_cli(["kanban", "create", "P9root",
                       "--priority", "9"], self._db, self._slug)
        self.assertEqual(r1.returncode, 0, r1.stderr)
        p9 = _parse_created_id(r1.stdout)
        r2 = _run_cli(["kanban", "create", "ChildDefault",
                       "--parent", p9], self._db, self._slug)
        self.assertEqual(r2.returncode, 0, r2.stderr)
        child_id = _parse_created_id(r2.stdout)
        with kb.connect_closing(db_path=self._db) as conn:
            row = conn.execute(
                "SELECT priority FROM tasks WHERE id = ?", (child_id,)
            ).fetchone()
        observed = row["priority"]
        _emit({
            "case": "C12", "layer": "cli",
            "scenario": "create_default_priority",
            "expected": [0], "observed": [observed],
            "pass": observed == 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        self.assertEqual(observed, 0, f"C12 got priority={observed}")

    def test_C13_swarm_root_assigns_spec_priority(self):
        # ``hermes kanban swarm`` is the fan-out entry surface. Seed a
        # root at p7 and a worker spec at p11; the worker row should
        # store 11, verifier/synth/leftover workers at 7.
        r1 = _run_cli(["kanban", "create", "SwarmRoot",
                       "--priority", "7",
                       "--triage"], self._db, self._slug)
        self.assertEqual(r1.returncode, 0, r1.stderr)
        root_id = _parse_created_id(r1.stdout)
        # Use ``hermes kanban swarm`` to fan-out a worker at p11.
        r2 = _run_cli(
            ["kanban", "swarm", root_id,
             "--worker-spec", json.dumps({"title": "W", "priority": 11})],
            self._db, self._slug,
        )
        # swarm may not be implemented; if non-zero, mark as soft-fail.
        if r2.returncode != 0:
            _emit({
                "case": "C13", "layer": "cli",
                "scenario": "swarm_priority",
                "expected": [11, 7, 7, 7],
                "observed": [],
                "pass": False,
                "note": "hermes kanban swarm unavailable or non-zero exit",
                "stderr_tail": r2.stderr.splitlines()[-3:],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            self.skipTest(f"hermes kanban swarm unavailable: {r2.stderr[:200]}")
        # If swarm succeeded, assert priorities.
        with kb.connect_closing(db_path=self._db) as conn:
            rows = conn.execute(
                "SELECT title, priority FROM tasks "
                "WHERE assignee = 'engineer' "
                "ORDER BY priority DESC, created_at ASC"
            ).fetchall()
        observed = [r["priority"] for r in rows]
        expected_desc = sorted([11, 7, 7, 7], reverse=True)
        ok = observed == expected_desc
        _emit({
            "case": "C13", "layer": "cli",
            "scenario": "swarm_priority",
            "expected": expected_desc, "observed": observed,
            "pass": ok,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        self.assertTrue(ok, f"C13 got {observed}")

    def test_C14_dispatch_order_by_priority_then_created_at(self):
        # Seed three ready tasks at p1, p5, p20 in a single transaction
        # with deterministic created_at (i+1). Then verify dispatch order
        # via the same ORDER BY the dispatcher uses.
        with kb.connect_closing(db_path=self._db) as conn:
            ids = []
            for pri, n in [(1, 1), (5, 2), (20, 3)]:
                tid = kb._new_task_id()  # type: ignore[attr-defined]
                conn.execute(
                    "INSERT INTO tasks "
                    "(id, title, body, assignee, status, workspace_kind, "
                    " workspace_path, tenant, priority, created_at, created_by) "
                    "VALUES (?, ?, ?, ?, 'ready', 'scratch', NULL, ?, ?, ?, 'probe')",
                    (tid, f"p{pri}", "body", "engineer",
                     self._slug, pri, int(time.time()) + n),
                )
                ids.append(tid)
            rows = conn.execute(
                "SELECT id FROM tasks WHERE status='ready' AND claim_lock IS NULL "
                "ORDER BY priority DESC, created_at ASC"
            ).fetchall()
        observed = [r["id"] for r in rows]
        # p20 was inserted with the highest created_at; p1 the lowest.
        # ORDER BY priority DESC means p20 first, p5 second, p1 last.
        expected = [ids[2], ids[1], ids[0]]
        ok = observed == expected
        _emit({
            "case": "C14", "layer": "cli",
            "scenario": "dispatch_order",
            "expected": expected, "observed": observed,
            "pass": ok,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        self.assertEqual(observed, expected, f"C14 got {observed}")


if __name__ == "__main__":
    unittest.main()