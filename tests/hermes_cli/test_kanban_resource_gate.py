"""Kanban dispatcher resource gate — ``tasks.resources`` column, derived lock.

A card's ``resources`` are exclusive physical keys (``kind:identifier``, e.g.
``harmony-device:x1``) that must not be shared by two concurrent workers: the
09-10 02:31 incident let one tick fan three cards out onto the same HarmonyOS
device. ``running`` status IS the lock — no lease table, no fencing: a ready
card whose resource is held by any running card stays ``ready`` (deferred to a
later tick, ``skipped_resource_held``), never blocked, never failed.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home_with_profiles(monkeypatch):
    """Spin up a fresh HERMES_HOME with kanban DB + alpha/beta profiles."""
    test_home = tempfile.mkdtemp(prefix="kanban_resource_gate_test_")
    for prof in ("alpha", "beta", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db


def _fake_spawn(*args, **kwargs):
    return 12345


def _connect(kb):
    from hermes_cli import kanban_db_connect as kbc
    return kbc.connect_closing()


def _make_board(kb, *task_kwargs_tuple):
    """Create the default board (idempotent for the fixture's lifetime)."""
    with _connect(kb) as conn:
        kb.create_board(slug="default", name="Test")


# ---------------------------------------------------------------------------
# Storage: schema, migration, Task.from_row, parse/normalize helpers
# ---------------------------------------------------------------------------


def test_new_board_has_resources_column(isolated_kanban_home_with_profiles):
    kb = isolated_kanban_home_with_profiles
    _make_board(kb)
    with _connect(kb) as conn:
        cols = {row["name"]: (row["type"] or "").upper()
                for row in conn.execute("PRAGMA table_info(tasks)")}
    assert "resources" in cols
    assert cols["resources"] == "TEXT"


def test_legacy_board_migration_adds_resources(isolated_kanban_home_with_profiles):
    """A board whose DB predates the column gains it on the next init pass and
    keeps its rows (additive migration, NULL default)."""
    kb = isolated_kanban_home_with_profiles
    _make_board(kb)
    db_path = kb.kanban_db_path()
    # Roll the schema back to pre-resources, with one surviving row.
    raw = sqlite3.connect(str(db_path))
    raw.execute("ALTER TABLE tasks DROP COLUMN resources")
    raw.execute(
        "INSERT INTO tasks (id, title, status, created_at) VALUES ('legacy-1', 'L', 'done', 1000)"
    )
    raw.commit()
    raw.close()
    # init_db always re-runs the migration pass (connect() caches first init).
    kb.init_db()
    with _connect(kb) as conn:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
        row = conn.execute("SELECT status FROM tasks WHERE id = 'legacy-1'").fetchone()
    assert "resources" in cols
    assert row is not None and row["status"] == "done"


def test_task_from_row_resources_four_states(isolated_kanban_home_with_profiles):
    """NULL -> None; '[]' -> []; valid JSON -> list; corrupt JSON -> None."""
    kb = isolated_kanban_home_with_profiles
    _make_board(kb)
    with _connect(kb) as conn:
        with kb.write_txn(conn):
            for tid, raw in (("t-null", None), ("t-empty", "[]"),
                             ("t-one", '["a:b"]'), ("t-bad", "not json")):
                conn.execute(
                    "INSERT INTO tasks (id, title, status, created_at, resources) "
                    "VALUES (?, ?, 'ready', 1000, ?)",
                    (tid, tid, raw),
                )
        assert kb.get_task(conn, "t-null").resources is None
        assert kb.get_task(conn, "t-empty").resources == []
        assert kb.get_task(conn, "t-one").resources == ["a:b"]
        assert kb.get_task(conn, "t-bad").resources is None


def test_parse_task_resources_fails_open(isolated_kanban_home_with_profiles):
    """Bad data on the read path never blocks dispatch and never raises."""
    kb = isolated_kanban_home_with_profiles
    assert kb.parse_task_resources(None) == ()
    assert kb.parse_task_resources("") == ()
    assert kb.parse_task_resources("not json") == ()
    assert kb.parse_task_resources('{"a": 1}') == ()  # valid JSON, not a list
    assert kb.parse_task_resources("[]") == ()
    assert kb.parse_task_resources('["a:b", "", "c:d"]') == ("a:b", "c:d")
    assert kb.parse_task_resources(["a:b", 2]) == ("a:b", "2")  # already-parsed list


def test_normalize_resources_normalizes(isolated_kanban_home_with_profiles):
    kb = isolated_kanban_home_with_profiles
    assert kb.normalize_resources(None) is None
    assert kb.normalize_resources([]) is None
    assert kb.normalize_resources(["  "]) is None  # only blanks -> nothing to hold
    assert kb.normalize_resources([" a:b ", "a:b", ""]) == ["a:b"]  # strip + dedupe, keep order
    assert kb.normalize_resources(["c:d", "a:b"]) == ["c:d", "a:b"]
    # Case-preserved identifiers: device serials must round-trip exactly.
    assert kb.normalize_resources(["harmony-device:DEV1"]) == ["harmony-device:DEV1"]
    assert kb.normalize_resources(
        ["harmony-device:6HQ0226318000078"]
    ) == ["harmony-device:6HQ0226318000078"]
    assert kb.normalize_resources(["  usb:K1  ", "usb:K1", ""]) == ["usb:K1"]
    assert kb.normalize_resources(["usb:K1:L2"]) == ["usb:K1:L2"]  # identifier may contain ':'


def test_normalize_resources_rejects_instead_of_rewriting(isolated_kanban_home_with_profiles):
    """Uppercase kind, injection characters, missing ':', empty kind/identifier
    and non-str are refused (ValueError naming the offending token and the
    allowed charset) — never silently rewritten into a different identity."""
    kb = isolated_kanban_home_with_profiles
    for bad in (["Harmony-Device:X1"], ["usb:K1;rm -rf /"], ["nocolon"],
                [":K1"], ["usb:"], ["a b"], [123]):
        with pytest.raises(ValueError) as excinfo:
            kb.normalize_resources(bad)
        msg = str(excinfo.value)
        assert "resources" in msg
        assert "a-z" in msg  # allowed-charset hint


def _running_holder(kb, title: str, resources, assignee: str = "alpha") -> str:
    """Create a card and claim it — a healthy ``running`` holder with real
    claim bookkeeping (a hand-made running row with NULL claim fields is a
    zombie the dispatcher's reconcile pass would requeue before the gate)."""
    with _connect(kb) as conn:
        tid = kb.create_task(conn, title=title, assignee=assignee, resources=resources)
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None and claimed.status == "running"
    return tid


def test_running_task_resources_unions_running_only(isolated_kanban_home_with_profiles):
    """Only ``running`` holds; blocked/review/ready/done never do."""
    kb = isolated_kanban_home_with_profiles
    _make_board(kb)
    _running_holder(kb, "holder-a", ["dev:a", "dev:b"])
    with _connect(kb) as conn:
        kb.create_task(conn, title="ready-b", assignee="alpha", resources=["dev:b"])
        kb.create_task(conn, title="blocked-c", assignee="alpha",
                       resources=["dev:c"], initial_status="blocked")
        done_id = kb.create_task(conn, title="done-d", assignee="alpha", resources=["dev:d"])
        review_id = kb.create_task(conn, title="rev-e", assignee="alpha", resources=["dev:e"])
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (done_id,))
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (review_id,))
        assert kb.running_task_resources(conn) == frozenset({"dev:a", "dev:b"})


# ---------------------------------------------------------------------------
# create_task / set_task_resources
# ---------------------------------------------------------------------------


def test_create_task_persists_resources(isolated_kanban_home_with_profiles):
    kb = isolated_kanban_home_with_profiles
    _make_board(kb)
    with _connect(kb) as conn:
        tid = kb.create_task(conn, title="needs device", assignee="alpha",
                             resources=["a:b", "c:d"])
        task = kb.get_task(conn, tid)
        raw = conn.execute("SELECT resources FROM tasks WHERE id = ?", (tid,)).fetchone()
        events = [e for e in kb.list_events(conn, tid) if e.kind == "created"]
    assert task.resources == ["a:b", "c:d"]
    assert json.loads(raw["resources"]) == ["a:b", "c:d"]
    assert events and events[0].payload.get("resources") == ["a:b", "c:d"]


def test_create_task_empty_resources_stores_null(isolated_kanban_home_with_profiles):
    """Empty list means "no resources" and is stored as NULL, not '[]'."""
    kb = isolated_kanban_home_with_profiles
    _make_board(kb)
    with _connect(kb) as conn:
        tid = kb.create_task(conn, title="plain", assignee="alpha", resources=[])
        raw = conn.execute("SELECT resources FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert raw["resources"] is None
        assert kb.get_task(conn, tid).resources is None


def test_set_task_resources_set_clear_missing_archived(isolated_kanban_home_with_profiles):
    kb = isolated_kanban_home_with_profiles
    _make_board(kb)
    with _connect(kb) as conn:
        tid = kb.create_task(conn, title="t", assignee="alpha")
        assert kb.set_task_resources(conn, tid, ["x:y"]) is True
        assert kb.get_task(conn, tid).resources == ["x:y"]
        # Clearing with an empty list stores NULL.
        assert kb.set_task_resources(conn, tid, []) is True
        assert kb.get_task(conn, tid).resources is None
        # Unknown task -> False.
        assert kb.set_task_resources(conn, "t-missing", ["x:y"]) is False
        # Archived task is refused.
        arch = kb.create_task(conn, title="arch", assignee="alpha")
        assert kb.archive_task(conn, arch) is True
        with pytest.raises(RuntimeError):
            kb.set_task_resources(conn, arch, ["x:y"])
        # Invalid token rejected before any write.
        with pytest.raises(ValueError):
            kb.set_task_resources(conn, tid, ["BAD KEY"])
        assert kb.get_task(conn, tid).resources is None


# ---------------------------------------------------------------------------
# The gate in the dispatcher tick
# ---------------------------------------------------------------------------


def _two_cards_same_resource(kb, resource="dev:x"):
    """Two ready alpha cards needing ``resource``; the first has priority so
    lane order is deterministic."""
    with _connect(kb) as conn:
        first = kb.create_task(conn, title="first", assignee="alpha",
                               resources=[resource], priority=5)
        second = kb.create_task(conn, title="second", assignee="alpha",
                                resources=[resource], priority=1)
    return first, second


def test_same_tick_resource_conflict_deferred(isolated_kanban_home_with_profiles):
    """Two cards sharing one resource never spawn in the same tick — the exact
    09-10 cross-contamination shape (dry-run variant)."""
    kb = isolated_kanban_home_with_profiles
    first, second = _two_cards_same_resource(kb)
    from hermes_cli import kanban_db_dispatch as kbd
    with _connect(kb) as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
    assert [s[0] for s in res.spawned] == [first]
    assert res.skipped_resource_held == [(second, "dev:x")]


def test_real_spawn_rolling_hold_blocks_same_tick(isolated_kanban_home_with_profiles):
    """Non-dry-run: the first card claims (-> running) mid-tick; the second
    must still be deferred even though the tick-start held-set was empty.
    Stay-ready contract: no block, no failure, no block_recurrences."""
    kb = isolated_kanban_home_with_profiles
    first, second = _two_cards_same_resource(kb)
    from hermes_cli import kanban_db_dispatch as kbd
    with _connect(kb) as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
    assert [s[0] for s in res.spawned] == [first]
    assert res.skipped_resource_held == [(second, "dev:x")]
    with _connect(kb) as conn:
        row = conn.execute(
            "SELECT status, consecutive_failures, block_recurrences FROM tasks WHERE id = ?",
            (second,),
        ).fetchone()
    assert row["status"] == "ready"
    assert row["consecutive_failures"] == 0
    assert row["block_recurrences"] == 0


def test_holder_release_on_next_tick(isolated_kanban_home_with_profiles):
    """The deferred card is picked up once the holder leaves running."""
    kb = isolated_kanban_home_with_profiles
    first, second = _two_cards_same_resource(kb)
    from hermes_cli import kanban_db_dispatch as kbd
    with _connect(kb) as conn:
        res1 = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
    assert res1.skipped_resource_held == [(second, "dev:x")]
    with _connect(kb) as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done', claim_lock = NULL WHERE id = ?", (first,)
            )
    with _connect(kb) as conn:
        res2 = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
    assert [s[0] for s in res2.spawned] == [second]
    assert res2.skipped_resource_held == []


def test_multi_resource_any_intersection_blocks(isolated_kanban_home_with_profiles):
    """Any single shared key out of several blocks; disjoint keys don't."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_dispatch as kbd
    _running_holder(kb, "holder", ["dev:a"])
    with _connect(kb) as conn:
        overlap = kb.create_task(conn, title="overlap", assignee="alpha",
                                 resources=["dev:b", "dev:a"], priority=5)
        disjoint = kb.create_task(conn, title="disjoint", assignee="alpha",
                                  resources=["dev:b", "dev:c"], priority=1)
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
    assert res.skipped_resource_held == [(overlap, "dev:a")]
    assert [s[0] for s in res.spawned] == [disjoint]


def test_no_resources_dispatch_unchanged(isolated_kanban_home_with_profiles):
    """Cards without resources are untouched by the gate."""
    kb = isolated_kanban_home_with_profiles
    with _connect(kb) as conn:
        a = kb.create_task(conn, title="a", assignee="alpha")
        b = kb.create_task(conn, title="b", assignee="alpha")
    from hermes_cli import kanban_db_dispatch as kbd
    with _connect(kb) as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
    assert sorted(s[0] for s in res.spawned) == sorted([a, b])
    assert res.skipped_resource_held == []


def test_per_profile_cap_and_resource_gate_stack(isolated_kanban_home_with_profiles):
    """Both gates apply: per-profile cap is checked first, the resource gate
    only sees cards that cleared the cap."""
    kb = isolated_kanban_home_with_profiles
    with _connect(kb) as conn:
        first = kb.create_task(conn, title="first", assignee="alpha",
                               resources=["dev:x"], priority=5)
        second = kb.create_task(conn, title="second", assignee="alpha",
                                resources=["dev:x"], priority=1)
    from hermes_cli import kanban_db_dispatch as kbd
    with _connect(kb) as conn:
        res1 = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False,
                                 max_in_progress_per_profile=1)
    assert [s[0] for s in res1.spawned] == [first]
    # Cap fires before the resource gate, so the deferral lands in the cap bucket.
    assert [c[0] for c in res1.skipped_per_profile_capped] == [second]
    assert res1.skipped_resource_held == []
    with _connect(kb) as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done', claim_lock = NULL WHERE id = ?", (first,)
            )
    with _connect(kb) as conn:
        res2 = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False,
                                 max_in_progress_per_profile=1)
    assert [s[0] for s in res2.spawned] == [second]


# ---------------------------------------------------------------------------
# _has_spawnable resource awareness (health telemetry)
# ---------------------------------------------------------------------------


def test_has_spawnable_ready_resource_aware(isolated_kanban_home_with_profiles):
    """A ready card whose only key is held by a running card is NOT spawnable
    ("correctly idle", not "stuck"); with the key free it is."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_dispatch as kbd
    _running_holder(kb, "holder", ["dev:x"])
    with _connect(kb) as conn:
        blocked_key = kb.create_task(conn, title="needs held key", assignee="alpha",
                                     resources=["dev:x"])
        assert kbd.has_spawnable_ready(conn) is False
        kb.set_task_resources(conn, blocked_key, ["dev:y"])
        assert kbd.has_spawnable_ready(conn) is True


# ---------------------------------------------------------------------------
# Watcher spawn log
# ---------------------------------------------------------------------------


def test_watcher_log_reports_resource_held(caplog):
    from gateway import kanban_watchers_dispatcher as kwd
    from hermes_cli.kanban_db_dispatch import DispatchResult

    res = DispatchResult(
        spawned=[("t1", "alpha", "/ws")],
        skipped_resource_held=[("t2", "dev:x"), ("t3", "dev:y")],
    )
    with caplog.at_level(logging.INFO):
        assert kwd._log_spawn_results([("slug", res)]) is True
    assert "skipped_resource_held=2" in caplog.text
    # None result rows are tolerated as before.
    assert kwd._log_spawn_results([("slug", None)]) is False


# ---------------------------------------------------------------------------
# Diagnostics: stranded-in-ready exemption
# ---------------------------------------------------------------------------


def _stranded_task(resources_raw):
    return {
        "id": "t-strand", "title": "stranded?", "status": "ready",
        "assignee": "alpha", "claim_lock": None, "created_at": 1000,
        "resources": resources_raw,
    }


def _stranded_events(now):
    return [{"kind": "created", "created_at": now - 7200}]  # ready for 2h


def test_stranded_rule_exempt_when_resource_held(isolated_kanban_home_with_profiles):
    from hermes_cli import kanban_diagnostics as kd

    now = 10_000_000
    cfg = {"stranded_threshold_seconds": 60}
    diags = kd.compute_task_diagnostics(
        _stranded_task('["dev:x"]'), _stranded_events(now), [],
        now=now, config=cfg, held_resources=frozenset({"dev:x"}),
    )
    assert [d.kind for d in diags] == []
    # Without the held key the same card IS stranded.
    diags = kd.compute_task_diagnostics(
        _stranded_task('["dev:x"]'), _stranded_events(now), [],
        now=now, config=cfg, held_resources=frozenset({"dev:other"}),
    )
    assert "stranded_in_ready" in [d.kind for d in diags]


def test_stranded_rule_default_held_set_is_empty(isolated_kanban_home_with_profiles):
    """Callers that don't pass held_resources keep today's behavior (no crash,
    no wrongful exemption) — the ``&``/``or`` precedence trap."""
    from hermes_cli import kanban_diagnostics as kd

    now = 10_000_000
    diags = kd.compute_task_diagnostics(
        _stranded_task('["dev:x"]'), _stranded_events(now), [],
        now=now, config={"stranded_threshold_seconds": 60},
    )
    assert "stranded_in_ready" in [d.kind for d in diags]


def test_stranded_rule_bad_resources_json_does_not_exempt(isolated_kanban_home_with_profiles):
    """Corrupt resources on the row fail open: the card stays flaggable."""
    from hermes_cli import kanban_diagnostics as kd

    now = 10_000_000
    diags = kd.compute_task_diagnostics(
        _stranded_task("not json"), _stranded_events(now), [],
        now=now, config={"stranded_threshold_seconds": 60},
        held_resources=frozenset({"dev:x"}),
    )
    assert "stranded_in_ready" in [d.kind for d in diags]


def test_stranded_detail_mentions_held_resource(isolated_kanban_home_with_profiles):
    from hermes_cli import kanban_diagnostics as kd

    now = 10_000_000
    diags = kd.compute_task_diagnostics(
        _stranded_task(None), _stranded_events(now), [],
        now=now, config={"stranded_threshold_seconds": 60},
    )
    detail = next(d for d in diags if d.kind == "stranded_in_ready").detail
    assert "waiting on a held resource" in detail


# ---------------------------------------------------------------------------
# CLI: create --resources, update subcommand, display
# ---------------------------------------------------------------------------


def _parse(*argv):
    """Build the real ``hermes kanban`` parser (one subparser group, as the CLI
    wires it) and parse a ``kanban`` argv tail."""
    from hermes_cli.kanban_parser import build_parser
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    build_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["kanban", *argv])


def test_cli_create_with_resources(isolated_kanban_home_with_profiles, capsys):
    kb = isolated_kanban_home_with_profiles
    _make_board(kb)
    from hermes_cli import kanban as kc
    args = _parse("create", "needs device", "--assignee", "alpha",
                  "--resources", "a:b, c:d")
    rc = kc._cmd_create(args)
    assert rc == 0
    with _connect(kb) as conn:
        tasks = kb.list_tasks(conn)
    assert len(tasks) == 1
    assert tasks[0].resources == ["a:b", "c:d"]


def test_cli_create_rejects_invalid_resource(isolated_kanban_home_with_profiles, capsys):
    kb = isolated_kanban_home_with_profiles
    _make_board(kb)
    from hermes_cli import kanban as kc
    args = _parse("create", "bad", "--assignee", "alpha",
                  "--resources", "Harmony-Device:X1")
    rc = kc._cmd_create(args)
    assert rc != 0
    err = capsys.readouterr().err
    assert "resources" in err
    with _connect(kb) as conn:
        assert kb.list_tasks(conn) == []  # no card created


def test_cli_update_subcommand(isolated_kanban_home_with_profiles, capsys):
    kb = isolated_kanban_home_with_profiles
    _make_board(kb)
    from hermes_cli import kanban as kc
    with _connect(kb) as conn:
        tid = kb.create_task(conn, title="t", assignee="alpha")
    args = _parse("update", tid, "--resources", "x:y,z:w")
    assert kc._cmd_update(args) == 0
    with _connect(kb) as conn:
        assert kb.get_task(conn, tid).resources == ["x:y", "z:w"]
    # Empty string clears.
    args = _parse("update", tid, "--resources", "")
    assert kc._cmd_update(args) == 0
    with _connect(kb) as conn:
        assert kb.get_task(conn, tid).resources is None
    # Unknown task -> non-zero.
    args = _parse("update", "t-missing", "--resources", "x:y")
    assert kc._cmd_update(args) != 0


def test_cli_show_and_list_display_resources(isolated_kanban_home_with_profiles, capsys):
    kb = isolated_kanban_home_with_profiles
    _make_board(kb)
    from hermes_cli import kanban as kc
    with _connect(kb) as conn:
        tid = kb.create_task(conn, title="t", assignee="alpha", resources=["a:b", "c:d"])
    out = kc.run_slash(f"show {tid}")
    assert "a:b, c:d" in out
    out = kc.run_slash("list")
    assert f"[res:a:b,c:d]" in out
    assert out.count("[res:") == 1  # compact bracket only when non-empty
    payload = json.loads(kc.run_slash("list --json"))
    assert payload[0]["resources"] == ["a:b", "c:d"]


# ---------------------------------------------------------------------------
# Model tool surface
# ---------------------------------------------------------------------------


def test_kanban_create_schema_declares_resources():
    from tools.kanban_tools_schemas import KANBAN_CREATE_SCHEMA
    props = KANBAN_CREATE_SCHEMA["parameters"]["properties"]
    assert props["resources"]["type"] == "array"
    assert props["resources"]["items"] == {"type": "string"}


def test_tool_kanban_create_with_resources(isolated_kanban_home_with_profiles):
    kb = isolated_kanban_home_with_profiles
    _make_board(kb)
    from tools.kanban_tools import _handle_create
    out = _handle_create({"title": "t", "assignee": "alpha", "resources": ["dev:x"]})
    payload = json.loads(out)
    assert payload["ok"] is True
    with _connect(kb) as conn:
        assert kb.get_task(conn, payload["task_id"]).resources == ["dev:x"]
    # A bare string is NOT coerced (unlike skills/parents): deterministic error,
    # no card — a stringified key would silently gate nothing.
    out = _handle_create({"title": "t2", "assignee": "alpha", "resources": "usb:K1"})
    assert "resources" in out
    assert json.loads(out).get("ok") is not True
    # Non-str element inside a list is a deterministic error too, no card.
    out = _handle_create({"title": "t3", "assignee": "alpha", "resources": ["dev:x", 5]})
    assert "resources" in out
    # Invalid token -> structured error mentioning resources, no card.
    out = _handle_create({"title": "t4", "assignee": "alpha", "resources": ["BAD KEY"]})
    assert "resources" in out
    with _connect(kb) as conn:
        assert [t.title for t in kb.list_tasks(conn)] == ["t"]
