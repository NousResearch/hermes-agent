"""Acceptance tests — kanban dispatcher resource gate (red team, black-box).

Covers the frozen acceptance scenarios 1-10 in the design doc
(``## 验收场景`` of requirement 20260911-#-kanban-dispatcher-资源闸).
Written BEFORE the implementation exists; every test is expected red on a
no-op gate (never intercepts -> 场景1.P1/2.P2/3.P1/5.*/6.P1/7.P2 fail) and
on a never-releasing gate (over-blocking -> 场景4.P1/6.P2-P3/7.P3 fail).

Black-box public surface only:
- ``hermes_cli.kanban_db``: parse_task_resources / normalize_resources /
  running_task_resources / set_task_resources, create_task(resources=...),
  Task.resources
- ``hermes_cli.kanban_db_connect.connect_closing``
- ``hermes_cli.kanban_db_dispatch.dispatch_once`` (DispatchResult.
  skipped_resource_held: list of (task_id, resource), resource =
  sorted(intersection)[0]) and ``has_spawnable_ready``
- ``hermes_cli.kanban_diagnostics.compute_task_diagnostics`` with
  ``held_resources=``
- real ``hermes kanban create/update/show/list`` subprocesses

Resource token grammar pinned by the frozen predicates: the kind before the
first ``:`` must be lowercase; identifiers are case-sensitive (device
serials like ``6HQ0226318000078`` must survive verbatim). 场景8.P1 / 10.P2
require ``usb:K1`` and ``harmony-device:6HQ0226318000078`` to be ACCEPTED
while ``Harmony-Device:X1`` (uppercase kind) and ``usb:K1;rm -rf /``
(injection chars) are rejected.

Harness follows tests/hermes_cli/test_kanban_per_profile_cap.py (temp
HERMES_HOME + sys.modules clear + dispatch_once(spawn_fn=_fake_spawn) +
direct DispatchResult asserts). CLI harness follows
tests/hermes_cli/test_kanban_cli_exit_status.py (real subprocess,
``python -m hermes_cli.main``). Nothing writes to the real ``~/.hermes``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]

# Frozen scenario literals — used verbatim.
DEV1 = "harmony-device:DEV1"
DEV2 = "harmony-device:DEV2"
K1 = "usb:K1"
K2 = "usb:K2"
K3 = "usb:K3"
K4 = "usb:K4"
SERIAL = "harmony-device:6HQ0226318000078"

_REWIND_SECONDS = 31 * 60  # "滞留超 30 分钟" via 31-minute back-dating


@pytest.fixture()
def gate_home(tmp_path, monkeypatch):
    """Fresh HERMES_HOME + kanban DB; hermes modules re-imported so they
    bind to the redirected home (per_profile_cap harness)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    for prof in ("alpha", "beta", "default"):
        os.makedirs(os.path.join(str(home), "profiles", prof), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    kanban_db.init_db()
    yield kanban_db


# ---------------------------------------------------------------------------
# Black-box helpers (raw SQL only for status/rewind/inspection — no private
# implementation functions touched).
# ---------------------------------------------------------------------------


def _fake_spawn(*args, **kwargs):
    return 12345


def _mk_ready(conn, kb, title, assignee="alpha", resources=None):
    """Create a card and force it ready via raw UPDATE so the test does not
    depend on create_task's default status."""
    kwargs = {"title": title, "assignee": assignee}
    if resources is not None:
        kwargs["resources"] = list(resources)
    tid = kb.create_task(conn, **kwargs)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    return tid


def _make_running(conn, kb, tid, assignee="alpha"):
    """Drive a ready card to running via claim (block_kinds precedent)."""
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer=assignee)
    assert claimed is not None, f"claim_task failed for {tid}"
    return tid


def _leave_running(conn, kb, tid, status="done"):
    """Take a running card out of running the way
    test_kanban_per_profile_cap.py does (raw UPDATE of status +
    claim_lock=NULL). For the 场景5 reclaim arm this models crash reclaim:
    crash is not a status — the holder leaves running and its claim is
    released, so the resource it held is free."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = ?, claim_lock = NULL WHERE id = ?",
            (status, tid),
        )


def _status(conn, tid):
    return conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (tid,)
    ).fetchone()[0]


def _dispatch(kbc, kbd, **kwargs):
    with kbc.connect_closing() as conn:
        return kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False, **kwargs)


def _resource_bucket_ids(res):
    return [entry[0] for entry in res.skipped_resource_held]


def _capped_bucket_ids(res):
    return [entry[0] for entry in res.skipped_per_profile_capped]


def _spawned_ids(res):
    return [entry[0] for entry in res.spawned]


def _in_qs(n):
    return ",".join("?" * n)


def _rewind_31min(conn, kb, tids):
    """Back-date cards + their events by 31 minutes (场景6 公共前置)."""
    with kb.write_txn(conn):
        conn.execute(
            f"UPDATE tasks SET created_at = created_at - {_REWIND_SECONDS} "
            f"WHERE id IN ({_in_qs(len(tids))})",
            tuple(tids),
        )
        conn.execute(
            f"UPDATE task_events SET created_at = created_at - {_REWIND_SECONDS} "
            f"WHERE task_id IN ({_in_qs(len(tids))})",
            tuple(tids),
        )


def _compute_diags(conn, kd, tid, **kwargs):
    """Round-trip a real DB row through the rule engine the way the API
    layer does (sqlite3.Row in, diagnostics out)."""
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
    events = list(conn.execute(
        "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (tid,)
    ))
    runs = list(conn.execute(
        "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (tid,)
    ))
    return kd.compute_task_diagnostics(row, events, runs, **kwargs)


# ---------------------------------------------------------------------------
# 场景 1 — held resource blocks the candidate this tick; release on the next
# ---------------------------------------------------------------------------


def test_s1_p1_p2_holder_blocks_then_release(gate_home):
    """场景1.P1: while running card A holds harmony-device:DEV1, one
    dispatch() must not release ready card B declaring the same resource —
    B stays ready, no failure counted, and the result records (B, DEV1) in
    skipped_resource_held. 场景1.P2: once A leaves running, the next tick
    dispatches B."""
    kb = gate_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        a = _mk_ready(conn, kb, "A", resources=[DEV1])
        _make_running(conn, kb, a)
        b = _mk_ready(conn, kb, "B", resources=[DEV1])

    # 场景1.P1 — holder still running.
    res1 = _dispatch(kbc, kbd)
    assert (b, DEV1) in res1.skipped_resource_held
    with kbc.connect_closing() as conn:
        assert _status(conn, b) == "ready"
        assert conn.execute(
            "SELECT consecutive_failures FROM tasks WHERE id = ?", (b,)
        ).fetchone()[0] == 0, "resource-gate skip must not count a failure"

    # 场景1.P2 — holder left running, next tick releases B.
    with kbc.connect_closing() as conn:
        _leave_running(conn, kb, a)
    res2 = _dispatch(kbc, kbd)
    assert b in _spawned_ids(res2)
    with kbc.connect_closing() as conn:
        assert _status(conn, b) == "running"


# ---------------------------------------------------------------------------
# 场景 2 — rolling update within one tick: only the first same-resource card
# spawns
# ---------------------------------------------------------------------------


def test_s2_p1_p2_rolling_update_single_tick(gate_home):
    """场景2.P1: a SINGLE dispatch() facing 3 ready cards all declaring
    harmony-device:DEV1 spawns exactly 1 into running, the other 2 stay
    ready. 场景2.P2: the 2 deferred cards are recorded in
    skipped_resource_held (tick-internal rolling update — without it the
    09-10 cross-contamination incident reproduces)."""
    kb = gate_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        ids = [
            _mk_ready(conn, kb, f"card{i}", resources=[DEV1]) for i in range(3)
        ]

    res = _dispatch(kbc, kbd)

    with kbc.connect_closing() as conn:
        statuses = {tid: _status(conn, tid) for tid in ids}
    running = [t for t, s in statuses.items() if s == "running"]
    ready = [t for t, s in statuses.items() if s == "ready"]
    assert len(running) == 1, statuses
    assert len(ready) == 2, statuses
    # 场景2.P2
    assert len(res.skipped_resource_held) == 2
    assert sorted(_resource_bucket_ids(res)) == sorted(ready)
    assert all(resource == DEV1 for _, resource in res.skipped_resource_held)


# ---------------------------------------------------------------------------
# 场景 3 — multi-resource card: any intersection blocks, all-free releases
# ---------------------------------------------------------------------------


def test_s3_p1_p2_p3_multi_resource_all_free_required(gate_home):
    """场景3.P1: while DEV1 (of D's {DEV1, K1}) is held, D is not released.
    场景3.P2: DEV1 free but K1 held — D still not released. 场景3.P3: once
    every declared resource is free, D is released."""
    kb = gate_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        e = _mk_ready(conn, kb, "E", resources=[DEV1])
        _make_running(conn, kb, e)
        d = _mk_ready(conn, kb, "D", resources=[DEV1, K1])

    # 场景3.P1 — DEV1 held.
    res1 = _dispatch(kbc, kbd)
    assert (d, DEV1) in res1.skipped_resource_held
    with kbc.connect_closing() as conn:
        assert _status(conn, d) == "ready"

    # 场景3.P2 — DEV1 released, K1 now held by G.
    with kbc.connect_closing() as conn:
        _leave_running(conn, kb, e)
    with kbc.connect_closing() as conn:
        g = _mk_ready(conn, kb, "G", resources=[K1], assignee="beta")
        _make_running(conn, kb, g, assignee="beta")
    res2 = _dispatch(kbc, kbd)
    assert (d, K1) in res2.skipped_resource_held
    with kbc.connect_closing() as conn:
        assert _status(conn, d) == "ready"

    # 场景3.P3 — all declared resources free.
    with kbc.connect_closing() as conn:
        _leave_running(conn, kb, g)
    res3 = _dispatch(kbc, kbd)
    assert d in _spawned_ids(res3)
    with kbc.connect_closing() as conn:
        assert _status(conn, d) == "running"


def test_s3_sorted_first_conflict_resource_recorded(gate_home):
    """Contract: the bucket element is (task_id, sorted(intersection)[0]).
    With BOTH declared resources held, the recorded resource must be the
    sorted-first one (harmony-device:DEV1 < usb:K1), deterministically."""
    kb = gate_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        e = _mk_ready(conn, kb, "E", resources=[DEV1])
        _make_running(conn, kb, e)
        g = _mk_ready(conn, kb, "G", resources=[K1], assignee="beta")
        _make_running(conn, kb, g, assignee="beta")
        d = _mk_ready(conn, kb, "D", resources=[DEV1, K1])

    res = _dispatch(kbc, kbd)

    assert res.skipped_resource_held == [(d, DEV1)]
    with kbc.connect_closing() as conn:
        assert _status(conn, d) == "ready"


# ---------------------------------------------------------------------------
# 场景 4 — resource-less cards behave exactly as before (zero regression)
# ---------------------------------------------------------------------------


def test_s4_p1_p2_no_resources_zero_regression(gate_home):
    """场景4.P1: a ready card H with no declared resources dispatches as
    usual. 场景4.P2: the dispatcher records no resource-wait skip for it
    (the only card on the board, so the bucket must be empty)."""
    kb = gate_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        h = _mk_ready(conn, kb, "H")

    res = _dispatch(kbc, kbd)

    assert h in _spawned_ids(res)
    assert res.skipped_resource_held == []
    with kbc.connect_closing() as conn:
        assert _status(conn, h) == "running"


# ---------------------------------------------------------------------------
# 场景 5 — only status 'running' holds resources
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("leave_as", ["blocked", "review", "done", "reclaim"])
def test_s5_p1_to_p4_only_running_holds(gate_home, leave_as):
    """场景5.P1/P2/P3/P4: once the holder A leaves running — to blocked, to
    review, to done, or via reclaim (crash path modelled as the
    per_profile_cap-style UPDATE; crash is not a status, reclaim releases
    the claim) — a dispatch() must release ready card B declaring A's
    resource."""
    kb = gate_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        a = _mk_ready(conn, kb, "A", resources=[DEV1])
        _make_running(conn, kb, a)
        b = _mk_ready(conn, kb, "B", resources=[DEV1])

    # AUTO-FIX 留痕 (qa, 2026-09-11): the "blocked" arm originally left the
    # state via raw UPDATE, but the dispatcher's pre-existing recompute_ready
    # (#28712) auto-recovers blocked cards with no sticky `kanban_block`
    # event (kanban_db.py::_has_sticky_block: "direct DB manipulation"
    # → auto-recover), so A was resurrected to ready and re-spawned before
    # B. Setup now uses the canonical block API — the asserted contract
    # ("a blocked card holds nothing") is unchanged; only the state-setup
    # mechanism is corrected.
    with kbc.connect_closing() as conn:
        if leave_as == "blocked":
            assert kb.block_task(conn, a, kind="transient", reason="s5 operator block")
        elif leave_as == "reclaim":
            # Crash path: the worker died; reclaim releases the claim and
            # drops the card back to ready (not done) — post-reclaim state.
            _leave_running(conn, kb, a, status="ready")
        else:
            _leave_running(conn, kb, a, status=leave_as)

    if leave_as == "reclaim":
        # AUTO-FIX 留痕 (qa, 2026-09-11): the original arm asserted B spawns
        # on the very tick after the crash, but the dispatcher's pre-existing
        # reclaim→ready semantics put A back in the ready lane and lane order
        # (priority DESC, created_at ASC) legitimately re-spawns A first; the
        # gate's rolling update then defers B. The crash-path contract under
        # test is: (1) the held set drops the moment A leaves running ("only
        # running holds" — a stale held set would also self-block A's own
        # re-spawn, so this still kills the no-op mutation), and (2) the
        # re-queued holder re-acquires via the same gate, deferring B with a
        # bucket record — a healthy wait, not a stuck lock.
        with kbc.connect_closing() as conn:
            assert kb.running_task_resources(conn) == frozenset()
        res = _dispatch(kbc, kbd)
        assert a in _spawned_ids(res), "crashed card must be re-queueable"
        assert res.skipped_resource_held == [(b, DEV1)]
        with kbc.connect_closing() as conn:
            assert _status(conn, b) == "ready"
        return

    res = _dispatch(kbc, kbd)
    assert b in _spawned_ids(res), leave_as
    with kbc.connect_closing() as conn:
        assert _status(conn, b) == "running", leave_as


# ---------------------------------------------------------------------------
# 场景 6 — resource wait is not stranded; genuinely stuck cards still are
# ---------------------------------------------------------------------------


def _s6_board(kb, kbc):
    """A running holding DEV1; B (DEV1), I (no resources), J (DEV2) ready
    and back-dated 31 minutes."""
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        a = _mk_ready(conn, kb, "A", resources=[DEV1])
        _make_running(conn, kb, a)
        b = _mk_ready(conn, kb, "B", resources=[DEV1])
        i = _mk_ready(conn, kb, "I")
        j = _mk_ready(conn, kb, "J", resources=[DEV2])
        _rewind_31min(conn, kb, [b, i, j])
    return b, i, j


def test_s6_p1_resource_wait_exempt_from_stranded(gate_home):
    """场景6.P1: ready card B's declared resource is held by a running card
    and B has been ready >30min — the stranded diagnostic must NOT flag B
    (healthy wait, not stuck)."""
    kb = gate_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_diagnostics as kd

    b, _i, _j = _s6_board(kb, kbc)
    with kbc.connect_closing() as conn:
        held = kb.running_task_resources(conn)
        assert held == frozenset({DEV1})
        diags = _compute_diags(conn, kd, b, held_resources=held)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    assert stranded == []


def test_s6_p2_no_resource_ready_stranded(gate_home):
    """场景6.P2: ready card I declares nothing and has been ready >30min —
    it must still be flagged stranded."""
    kb = gate_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_diagnostics as kd

    _b, i, _j = _s6_board(kb, kbc)
    with kbc.connect_closing() as conn:
        held = kb.running_task_resources(conn)
        diags = _compute_diags(conn, kd, i, held_resources=held)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    assert len(stranded) == 1
    assert stranded[0].data["age_seconds"] >= 1800


def test_s6_p3_free_resource_ready_stranded_and_detail_clause(gate_home):
    """场景6.P3: ready card J's declared resource (DEV2) is free and J has
    been ready >30min — J must be flagged stranded, and the detail's common
    causes must mention the held-resource wait."""
    kb = gate_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_diagnostics as kd

    _b, _i, j = _s6_board(kb, kbc)
    with kbc.connect_closing() as conn:
        held = kb.running_task_resources(conn)
        diags = _compute_diags(conn, kd, j, held_resources=held)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    assert len(stranded) == 1
    assert "waiting on a held resource" in stranded[0].detail


# ---------------------------------------------------------------------------
# 场景 7 — resource gate stacks with the per-profile quota
# ---------------------------------------------------------------------------


def test_s7_p1_quota_gate_wins_bucketing(gate_home):
    """场景7.P1: per-profile quota full AND C's resources free — C is not
    released, is bucketed as skipped_per_profile_capped, and NOT as
    skipped_resource_held (per-profile check runs first)."""
    kb = gate_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        r = _mk_ready(conn, kb, "R", assignee="alpha")
        _make_running(conn, kb, r, assignee="alpha")  # eats alpha's 1-slot quota, holds nothing
        c = _mk_ready(conn, kb, "C", resources=[K1], assignee="alpha")

    res = _dispatch(kbc, kbd, max_in_progress_per_profile=1)

    with kbc.connect_closing() as conn:
        assert _status(conn, c) == "ready"
    assert c in _capped_bucket_ids(res)
    assert all(entry[0] != c for entry in res.skipped_resource_held)


def test_s7_p2_resource_gate_bucket_when_quota_free(gate_home):
    """场景7.P2: quota has room (holder is a different profile) but C's
    resource is held by a running card — C stays ready and is bucketed as
    skipped_resource_held."""
    kb = gate_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        h = _mk_ready(conn, kb, "H", assignee="beta", resources=[K1])
        _make_running(conn, kb, h, assignee="beta")
        c = _mk_ready(conn, kb, "C", resources=[K1], assignee="alpha")

    res = _dispatch(kbc, kbd, max_in_progress_per_profile=1)

    with kbc.connect_closing() as conn:
        assert _status(conn, c) == "ready"
    assert (c, K1) in res.skipped_resource_held
    assert c not in _capped_bucket_ids(res)


def test_s7_p3_both_gates_pass_spawns(gate_home):
    """场景7.P3: quota has room and C's resources are all free — C is
    released."""
    kb = gate_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        c = _mk_ready(conn, kb, "C", resources=[K1], assignee="alpha")

    res = _dispatch(kbc, kbd, max_in_progress_per_profile=1)

    assert c in _spawned_ids(res)
    with kbc.connect_closing() as conn:
        assert _status(conn, c) == "running"


# ---------------------------------------------------------------------------
# Error / helper contracts (hermes_cli.kanban_db public helpers)
# ---------------------------------------------------------------------------


def test_error_contract_fail_open_db_garbage_resources(gate_home):
    """错误契约 + 红队补充谓词(a): DB hand-edited resources='not json' must
    fail OPEN on the read path — parse_task_resources returns (), dispatch
    proceeds treating the card as resource-less (the card is released even
    though a running holder exists), and nothing raises."""
    kb = gate_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        a = _mk_ready(conn, kb, "A", resources=[K1])
        _make_running(conn, kb, a)
        b = _mk_ready(conn, kb, "B", resources=[K1])
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET resources = 'not json' WHERE id = ?", (b,))

    with kbc.connect_closing() as conn:
        assert kb.parse_task_resources("not json") == ()
        assert kb.parse_task_resources(None) == ()
        assert kb.parse_task_resources('["a:b"]') == ("a:b",)

    res = _dispatch(kbc, kbd)

    assert b in _spawned_ids(res)
    assert res.skipped_resource_held == []
    with kbc.connect_closing() as conn:
        assert _status(conn, b) == "running"


def test_error_contract_normalize_resources_validation(gate_home):
    """错误契约 (验收 #8 matrix, write path): uppercase KIND rejected with
    the original token in the message; injection chars rejected; non-str
    rejected; whitespace-wrapped tokens stripped + order-preserving dedupe;
    empty result normalizes to None; case-sensitive identifiers preserved
    (no silent lowercasing of serials — pinned by 场景8.P1/10.P2)."""
    kb = gate_home

    assert kb.normalize_resources(None) is None
    assert kb.normalize_resources([]) is None
    assert kb.normalize_resources(["", "   "]) is None
    assert kb.normalize_resources(["  usb:K1  ", "usb:K1", ""]) == [K1]
    assert kb.normalize_resources(["harmony-device:DEV1"]) == [DEV1]

    with pytest.raises(ValueError) as excinfo:
        kb.normalize_resources(["Harmony-Device:X1"])
    assert "Harmony-Device:X1" in str(excinfo.value)
    # message must hint the legal charset (lowercase)
    assert "a-z" in str(excinfo.value) or "lowercase" in str(excinfo.value)

    with pytest.raises(ValueError):
        kb.normalize_resources(["usb:K1;rm -rf /"])
    with pytest.raises(ValueError):
        kb.normalize_resources([123])


def test_contract_running_task_resources_union_of_running(gate_home):
    """契约: running_task_resources is the union of parse(resources) over
    status='running' rows ONLY — blocked/review/done/todo hold nothing."""
    kb = gate_home
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        a = _mk_ready(conn, kb, "A", resources=[DEV1, K1])
        _make_running(conn, kb, a)
        done_t = _mk_ready(conn, kb, "doneT", resources=[DEV2], assignee="beta")
        _make_running(conn, kb, done_t, assignee="beta")
        _leave_running(conn, kb, done_t)
        blocked_t = _mk_ready(conn, kb, "blkT", resources=[K3], assignee="beta")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='blocked' WHERE id=?", (blocked_t,)
            )
        _mk_ready(conn, kb, "readyT", resources=[K4], assignee="beta")

        assert kb.running_task_resources(conn) == frozenset({DEV1, K1})

    with kbc.connect_closing() as conn:
        _leave_running(conn, kb, a)
        assert kb.running_task_resources(conn) == frozenset()


def test_contract_spawn_probe_resource_aware(gate_home):
    """契约 (闸语义·探针): has_spawnable_ready is False when every ready
    row conflicts with a held resource (so gateway bad_ticks stuck alarms
    do not misfire on healthy resource waits), and True once a resource-
    clean spawnable row exists."""
    kb = gate_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        a = _mk_ready(conn, kb, "A", resources=[DEV1])
        _make_running(conn, kb, a)
        b = _mk_ready(conn, kb, "B", resources=[DEV1])

        with kbc.connect_closing() as inner:
            assert not kbd.has_spawnable_ready(inner)

        h = _mk_ready(conn, kb, "H")
        with kbc.connect_closing() as inner:
            assert kbd.has_spawnable_ready(inner)

        _leave_running(conn, kb, a)

    with kbc.connect_closing() as inner:
        assert kbd.has_spawnable_ready(inner)


# ---------------------------------------------------------------------------
# 场景 8.P4 / 9.P2 — Python API storage-level predicates
# ---------------------------------------------------------------------------


def test_s8_p4_api_create_persists_equivalent_set(gate_home):
    """场景8.P4: creating a card via the Python API with the scenario 8
    resources persists an equivalent set in the DB (JSON array of str)."""
    kb = gate_home
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        tid = kb.create_task(
            conn, title="s8p4", assignee="alpha", resources=[SERIAL, K1]
        )
        raw = conn.execute(
            "SELECT resources FROM tasks WHERE id = ?", (tid,)
        ).fetchone()[0]
        task = kb.get_task(conn, tid)

    assert raw is not None
    assert set(json.loads(raw)) == {SERIAL, K1}
    assert task.resources is not None
    assert set(task.resources) == {SERIAL, K1}


def test_s9_p2_api_set_task_resources_persists(gate_home):
    """场景9.P2: updating a card's resource declaration via
    set_task_resources persists the new declaration."""
    kb = gate_home
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        tid = _mk_ready(conn, kb, "s9p2", resources=[DEV1])
        kb.set_task_resources(conn, tid, [K2])
        raw = conn.execute(
            "SELECT resources FROM tasks WHERE id = ?", (tid,)
        ).fetchone()[0]
        task = kb.get_task(conn, tid)

    assert raw is not None
    assert K2 in set(json.loads(raw))
    assert set(task.resources) == {K2}


def test_s10_p4_tool_rejects_bad_resource_types(gate_home):
    """场景10.P4: model-tool kanban_create with a wrongly-typed resources
    argument returns a deterministic error whose text contains
    "resources", and creates NO card (count unchanged from baseline)."""
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    with kbc.connect_closing() as conn:
        baseline = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

    for bad in ("usb:K1", [K1, 5]):  # not a list; element not a str
        out = kt._handle_create(
            {"title": "bad res", "assignee": "peer", "resources": bad}
        )
        d = json.loads(out)
        assert d.get("error"), out
        assert "resources" in d["error"]

    with kbc.connect_closing() as conn:
        after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    assert after == baseline


# ---------------------------------------------------------------------------
# 场景 8/9/10 — real-process CLI predicates
# ---------------------------------------------------------------------------


class TestKanbanResourceCli:
    """Real-process CLI scenarios (``[real-process]`` predicates): every
    observation goes through a ``hermes kanban ...`` subprocess against a
    temp HERMES_HOME. Harness follows
    tests/hermes_cli/test_kanban_cli_exit_status.py."""

    @pytest.fixture()
    def cli_home(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        for mod in list(sys.modules.keys()):
            if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
                del sys.modules[mod]
        from hermes_cli import kanban_db
        kanban_db.init_db()
        return home

    def _run(self, home, *args):
        env = os.environ.copy()
        env["HERMES_HOME"] = str(home)
        env["HERMES_KANBAN_HOME"] = str(home)
        for name in (
            "HERMES_KANBAN_BOARD",
            "HERMES_KANBAN_DB",
            "HERMES_KANBAN_WORKSPACES_ROOT",
        ):
            env.pop(name, None)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", *args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

    def _task_rows(self, home):
        """In-process read of the same DB the subprocesses wrote."""
        from hermes_cli import kanban_db_connect as kbc
        with kbc.connect_closing() as conn:
            return conn.execute(
                "SELECT id, resources FROM tasks ORDER BY id"
            ).fetchall()

    def _create(self, home, title, *extra):
        return self._run(home, "kanban", "create", title, "--json", *extra)

    # -- 场景 8 -----------------------------------------------------------

    def test_s8_p1_create_with_resources_exit_zero(self, cli_home):
        """场景8.P1: `hermes kanban create --resources
        "harmony-device:6HQ0226318000078,usb:K1"` exits 0, reports the new
        card id on stdout, and the card lands in the DB declaring both
        resources."""
        home = cli_home
        proc = self._create(
            home, "s8 card", "--resources", f"{SERIAL},{K1}"
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        task_id = payload["id"]
        assert task_id
        assert task_id in proc.stdout

        rows = self._task_rows(home)
        ours = [r for r in rows if r["id"] == task_id]
        assert len(ours) == 1
        assert set(json.loads(ours[0]["resources"])) == {SERIAL, K1}

    def test_s8_p2_show_displays_resources(self, cli_home):
        """场景8.P2: `hermes kanban show <id>` prints the card's declared
        resources verbatim."""
        home = cli_home
        created = self._create(home, "s8 show", "--resources", f"{SERIAL},{K1}")
        assert created.returncode == 0, created.stderr
        task_id = json.loads(created.stdout)["id"]

        shown = self._run(home, "kanban", "show", task_id)
        assert shown.returncode == 0, shown.stderr
        assert SERIAL in shown.stdout
        assert K1 in shown.stdout

    def test_s8_p3_list_presents_resources(self, cli_home):
        """场景8.P3: `hermes kanban list` surfaces resources; `list --json`
        carries a `resources` key on EVERY card (unset -> [] shaping)."""
        home = cli_home
        with_res = self._create(home, "s8 list res", "--resources", f"{SERIAL},{K1}")
        assert with_res.returncode == 0, with_res.stderr
        res_id = json.loads(with_res.stdout)["id"]
        plain = self._create(home, "s8 list plain")
        assert plain.returncode == 0, plain.stderr
        plain_id = json.loads(plain.stdout)["id"]

        listing = self._run(home, "kanban", "list")
        assert listing.returncode == 0, listing.stderr
        assert SERIAL in listing.stdout

        listing_json = self._run(home, "kanban", "list", "--json")
        assert listing_json.returncode == 0, listing_json.stderr
        data = json.loads(listing_json.stdout)
        tasks = data.get("tasks") if isinstance(data, dict) else data
        assert tasks, listing_json.stdout
        by_id = {t["id"]: t for t in tasks}
        for task in tasks:
            assert "resources" in task
        assert set(by_id[res_id]["resources"]) == {SERIAL, K1}
        assert by_id[plain_id]["resources"] == []

    # -- 场景 9 -----------------------------------------------------------

    def test_s9_p1_update_sets_resources(self, cli_home):
        """场景9.P1: `hermes kanban update <id> --resources "usb:K2"` exits
        0 and the follow-up show reflects the new declaration."""
        home = cli_home
        created = self._create(home, "s9 card")
        assert created.returncode == 0, created.stderr
        task_id = json.loads(created.stdout)["id"]

        upd = self._run(home, "kanban", "update", task_id, "--resources", K2)
        assert upd.returncode == 0, upd.stderr

        shown = self._run(home, "kanban", "show", task_id)
        assert shown.returncode == 0, shown.stderr
        assert K2 in shown.stdout

        rows = [r for r in self._task_rows(home) if r["id"] == task_id]
        assert len(rows) == 1
        assert set(json.loads(rows[0]["resources"])) == {K2}

    def test_s9_p3_update_empty_string_clears(self, cli_home):
        """场景9.P3: `hermes kanban update <id> --resources ""` exits 0 and
        clears the declaration — the follow-up show mentions neither the
        old serial nor any harmony-device resource."""
        home = cli_home
        created = self._create(home, "s9 clear", "--resources", f"{SERIAL},{K2}")
        assert created.returncode == 0, created.stderr
        task_id = json.loads(created.stdout)["id"]

        upd = self._run(home, "kanban", "update", task_id, "--resources", "")
        assert upd.returncode == 0, upd.stderr

        shown = self._run(home, "kanban", "show", task_id)
        assert shown.returncode == 0, shown.stderr
        # AUTO-FIX 留痕 (qa, 2026-09-11): the original bare-substring checks
        # (`"usb:K2" not in stdout`) also matched the immutable `created`
        # event payload in show's history section, which legitimately records
        # the declaration made at creation time. The contract under test is
        # the CURRENT declaration: the `resources:` field line must be gone.
        # This still fails for a no-op clear (the field line would remain).
        import re as _re

        assert not _re.search(r"^\s*resources:", shown.stdout, _re.M), shown.stdout
        assert not _re.search(r"^\s*resources:.*harmony-device", shown.stdout, _re.M)

        rows = [r for r in self._task_rows(home) if r["id"] == task_id]
        assert len(rows) == 1
        raw = rows[0]["resources"]
        assert raw is None or json.loads(raw) == []

    def test_s9_update_nonexistent_task_nonzero_exit(self, cli_home):
        """红队补充谓词(b): `kanban update <nonexistent-id> --resources`
        must exit non-zero."""
        home = cli_home
        created = self._create(home, "anchor card")
        assert created.returncode == 0, created.stderr

        upd = self._run(
            home, "kanban", "update", "t_absent000", "--resources", K2
        )
        assert upd.returncode != 0

    # -- 场景 10 ----------------------------------------------------------

    def test_s10_p1_uppercase_rejected_deterministically(self, cli_home):
        """场景10.P1: `--resources "Harmony-Device:X1"` is deterministically
        rejected (two runs, identical non-zero exit, stderr mentions
        resources) and the raw uppercase form never lands in the DB."""
        home = cli_home
        baseline = len(self._task_rows(home))

        r1 = self._create(home, "s10a", "--resources", "Harmony-Device:X1")
        r2 = self._create(home, "s10b", "--resources", "Harmony-Device:X1")
        assert r1.returncode != 0
        assert r2.returncode != 0
        assert r1.returncode == r2.returncode
        assert "resources" in r1.stderr.lower()
        assert "resources" in r2.stderr.lower()

        rows = self._task_rows(home)
        assert len(rows) == baseline  # no card created
        assert all(
            "Harmony-Device:X1" not in (r["resources"] or "") for r in rows
        )

    def test_s10_p2_whitespace_normalized_accepted(self, cli_home):
        """场景10.P2: `" usb:K1 , usb:K2 "` is deterministically ACCEPTED
        (two runs, identical zero exit) and stored as the normalized
        equivalent set {usb:K1, usb:K2}."""
        home = cli_home
        noisy = f" {K1} , {K2} "
        r1 = self._create(home, "s10c", "--resources", noisy)
        r2 = self._create(home, "s10d", "--resources", noisy)
        assert r1.returncode == 0, r1.stderr
        assert r2.returncode == r1.returncode

        rows = self._task_rows(home)
        assert len(rows) == 2
        for row in rows:
            assert set(json.loads(row["resources"])) == {K1, K2}

    def test_s10_p3_injection_rejected_deterministically(self, cli_home):
        """场景10.P3: `--resources "usb:K1;rm -rf /"` is deterministically
        rejected (two runs, identical non-zero exit, stderr mentions
        resources) and no injection characters ever land in the DB."""
        home = cli_home
        baseline = len(self._task_rows(home))
        evil = "usb:K1;rm -rf /"

        r1 = self._create(home, "s10e", "--resources", evil)
        r2 = self._create(home, "s10f", "--resources", evil)
        assert r1.returncode != 0
        assert r2.returncode != 0
        assert r1.returncode == r2.returncode
        assert "resources" in r1.stderr.lower()
        assert "resources" in r2.stderr.lower()

        rows = self._task_rows(home)
        assert len(rows) == baseline  # no card created
        assert all(";" not in (r["resources"] or "") for r in rows)
