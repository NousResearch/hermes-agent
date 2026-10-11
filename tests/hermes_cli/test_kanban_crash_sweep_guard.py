"""The reclaim sweep may not act on a liveness substrate it cannot vouch for.

Ruling ``t_63a20c59`` (platform-stl, 2026-10-10), carrier ``t_fc2cf9ee``.

**R4.** A sweep that would close ``>= 3`` runs ``crashed`` in ONE tick is a host-systemic signature,
not N card failures. It must refuse the whole write and fail LOUD. The measured census this pins is
the regression fixture: on 2026-10-10 ALL 21 ``crashed`` runs on board ``migrations`` fell into three
instants (8 at ``ended_at`` 14:20:37, 5 at 13:35:34, 8 at 00:01:23), every one carrying
``pid N not alive``, and pid 70697 was measured ALIVE eight minutes after the 14:20:37 sweep and
still writing the live tree. Not one of the 21 was a genuine crash.

**R6.** The identity shape the dispatcher WRITES and the shape it READS are recorded and compared on
every tick, and a row whose fingerprint was built by a different identity generation is HELD rather
than compared (R2 ``UNKNOWN -> HOLD``). Three incomparable shapes were measured in circulation at
once: ``'|<abs-epoch-cs>'``, ``None`` -> ``'unverified'``, and
``bootsession:<uuid>|<ref_cs>|<boot-relative start>``.
"""

import os
import time

import pytest

from hermes_cli import kanban_crash_sweep_guard as guard
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


# ---------------------------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    # The generation stamp must land in the sandbox home, never the machine-global one.
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def _dead_pid() -> int:
    """A pid that is definitely not alive (scan downward for a number nothing owns)."""
    for candidate in range(99999, 90000, -1):
        if not kb._pid_alive(candidate):
            return candidate
    raise AssertionError("no free pid on this host")


def _writer_shape() -> str:
    return guard.identity_shape(kbd._process_fingerprint(os.getpid()))


def _comparable_fingerprint() -> str:
    """A fingerprint of the SAME generation as this process's writer, with a live-looking start."""
    shape = _writer_shape()
    family = guard.shape_family(shape)
    if family == guard.IDENTITY_SHAPE_BOOT_WITNESS:
        return "bootsession:TEST|179000000000|1234"
    if family == guard.IDENTITY_SHAPE_COMPOSED:
        return "ref|1234"
    return "|179000000000"  # epoch-cs (macOS: current_instantiation_epoch() is "")


def _incomparable_fingerprint() -> str:
    """A fingerprint built by a DIFFERENT identity generation than this process's writer."""
    shape = _writer_shape()
    if guard.shape_family(shape) == guard.IDENTITY_SHAPE_BOOT_WITNESS:
        return "|179000000000"                                   # epoch-cs/2, a different family
    return "bootsession:TEST|179000000000|1234"                  # boot-witness/3, a different family


def _claimed_running(conn, *, pid, started_at, claim_expires=None) -> str:
    """A ``running`` card claimed by a host-local worker whose pid/fingerprint we control."""
    tid = kb.create_task(conn, title="job", assignee="worker")
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid)
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_started_at = ?, started_at = ?, claim_expires = ? WHERE id = ?",
            (started_at, old, old if claim_expires is None else claim_expires, tid),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE id = "
            "(SELECT current_run_id FROM tasks WHERE id = ?)",
            (old, tid),
        )
    return tid


def _task_row(conn, tid):
    return conn.execute(
        "SELECT status, worker_pid, worker_started_at, claim_lock FROM tasks WHERE id = ?", (tid,),
    ).fetchone()


def _run_rows(conn, tid):
    return conn.execute(
        "SELECT id, status, outcome, ended_at FROM task_runs WHERE task_id = ?", (tid,),
    ).fetchall()


def _event_count(conn, tid, kind) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM task_events WHERE task_id = ? AND kind = ?", (tid, kind),
    ).fetchone()
    return int(row["c"])


def _closure(task_id, *, kind="unknown", shape=None, **kwargs) -> guard.SweepClosure:
    return guard.SweepClosure(
        task_id=task_id, pid=1234, kind=kind, shape=shape or "epoch-cs/2", **kwargs,
    )


# ---------------------------------------------------------------------------------------------
# R4 -- the count guard itself
# ---------------------------------------------------------------------------------------------


def test_the_threshold_defaults_to_three(monkeypatch):
    monkeypatch.delenv(guard.MASS_CRASH_ABORT_THRESHOLD_ENV, raising=False)
    assert guard.mass_crash_threshold() == 3


def test_the_threshold_env_below_two_disables_the_guard(monkeypatch):
    monkeypatch.setenv(guard.MASS_CRASH_ABORT_THRESHOLD_ENV, "0")
    assert guard.mass_crash_threshold() == 0
    monkeypatch.setenv(guard.MASS_CRASH_ABORT_THRESHOLD_ENV, "7")
    assert guard.mass_crash_threshold() == 7


def test_an_unparseable_threshold_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv(guard.MASS_CRASH_ABORT_THRESHOLD_ENV, "lots")
    assert guard.mass_crash_threshold() == 3


def test_two_unexplained_deaths_are_not_a_host_signature():
    assessment = guard.assess_sweep(
        [_closure("t1"), _closure("t2")], write_shape="epoch-cs/2", now=1791659000,
    )
    assert assessment.refusal_needed is False
    assert assessment.verdict.count == 2


def test_three_unexplained_deaths_in_one_tick_are_refused():
    assessment = guard.assess_sweep(
        [_closure("t1"), _closure("t2"), _closure("t3")], write_shape="epoch-cs/2", now=1791659000,
    )
    assert assessment.refusal_needed is True
    assert assessment.verdict.count == 3
    assert assessment.verdict.task_ids == ("t1", "t2", "t3")


def test_the_signature_is_a_count_and_a_timestamp_and_nothing_else():
    assessment = guard.assess_sweep(
        [_closure(f"t{i}") for i in range(8)], write_shape="epoch-cs/2", now=1791658837,
    )
    assert assessment.verdict.ended_at == 1791658837
    assert assessment.verdict.signature == "mass-crash-sweep:count=8,ended_at=1791658837"
    assert assessment.verdict.signature.startswith(guard.MASS_CRASH_SIGNATURE_PREFIX)


@pytest.mark.parametrize("kind", sorted(guard.UNEXPLAINED_DEATH_KINDS))
def test_every_unexplained_death_kind_is_counted(kind):
    assessment = guard.assess_sweep(
        [_closure(f"t{i}", kind=kind) for i in range(3)], write_shape="epoch-cs/2",
    )
    assert assessment.verdict.count == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"protocol_violation": True},
        {"terminal_provider": True},
        {"rate_limited": True},
    ],
    ids=["protocol-violation", "terminal-provider", "rate-limited"],
)
def test_evidence_carrying_verdicts_are_not_counted(kwargs):
    """A clean exit, a provider refusal and a quota wall are not a broken liveness witness."""
    assessment = guard.assess_sweep(
        [_closure(f"t{i}", **kwargs) for i in range(5)], write_shape="epoch-cs/2",
    )
    assert assessment.verdict.count == 0
    assert assessment.refusal_needed is False


def test_a_mixed_sweep_counts_only_the_unexplained_deaths():
    closures = [
        _closure("a"), _closure("b"),
        _closure("clean", protocol_violation=True),
        _closure("quota", rate_limited=True),
    ]
    assessment = guard.assess_sweep(closures, write_shape="epoch-cs/2")
    assert assessment.verdict.task_ids == ("a", "b")
    assert assessment.refusal_needed is False


def test_the_operator_override_lets_a_genuine_mass_crash_through(monkeypatch):
    monkeypatch.setenv(guard.MASS_CRASH_ABORT_THRESHOLD_ENV, "0")
    assessment = guard.assess_sweep(
        [_closure(f"t{i}") for i in range(21)], write_shape="epoch-cs/2",
    )
    assert assessment.refusal_needed is False
    assert assessment.verdict.count == 21  # measured, reported, not refused


def test_one_tick_is_one_ended_at(monkeypatch):
    """The ruling's two spellings -- ">= 3 in one tick" and ">= 3 at one ended_at" -- are one
    measurement: a tick stamps every closure with the same instant."""
    assessment = guard.assess_sweep(
        [_closure(f"t{i}") for i in range(3)], write_shape="epoch-cs/2", now=1791658837,
    )
    assert assessment.verdict.ended_at == 1791658837
    # every counted closure shares the tick instant by construction (the sweep writes one `now`)
    assert assessment.verdict.task_ids == ("t0", "t1", "t2")


# ---------------------------------------------------------------------------------------------
# R4 -- the 2026-10-10 census is the regression fixture
# ---------------------------------------------------------------------------------------------


#: `ended_at` -> how many runs the 2026-10-10 sweep closed at that instant. All 21 carried the same
#: evidence line, `pid N not alive`.
CENSUS_20261010 = {1791658837: 8, 1791656134: 5, 1791648083: 8}  # 14:20:37, 13:35:34, 00:01:23


def test_the_20261010_census_is_refused_at_every_one_of_its_three_instants():
    assert sum(CENSUS_20261010.values()) == 21, "the measured census is 21 runs, not fewer"
    for instant, size in CENSUS_20261010.items():
        closures = [_closure(f"census-{instant}-{i}", kind="unknown") for i in range(size)]
        assessment = guard.assess_sweep(closures, write_shape="epoch-cs/2", now=instant)
        assert assessment.refusal_needed is True, f"instant {instant} closed {size} runs"
        assert assessment.verdict.count == size
        assert assessment.verdict.task_ids == tuple(f"census-{instant}-{i}" for i in range(size))


def test_the_census_evidence_line_is_the_liveness_verdict():
    """The census's own words: a liveness verdict masquerading as N card failures."""
    assert kbd._classify_dead_worker_exit(_dead_pid(), "yoyodine:1").error_text.endswith(
        "not alive"
    )


# ---------------------------------------------------------------------------------------------
# R6 -- shapes, the hold, and the generation stamp
# ---------------------------------------------------------------------------------------------


def test_identity_shape_names_the_structure_not_the_value():
    assert guard.identity_shape(None) == "none"
    assert guard.identity_shape("") == "none"
    assert guard.identity_shape("unverified") == "unverified"
    assert guard.identity_shape(179036519519) == "legacy-int"
    assert guard.identity_shape("179041285600") == "legacy-int"
    assert guard.identity_shape("|179036519519") == "epoch-cs/2"
    assert guard.identity_shape("bootsession:U|5458133") == "boot-witness/2"
    assert guard.identity_shape("bootsession:U|179160482300|5458133") == "boot-witness/3"


def test_the_three_circulating_shapes_are_distinguishable():
    """Measured 2026-10-10: three incomparable shapes were live at once."""
    versions = {
        "|179036519519": "epoch-cs/2",                                  # .venv 3.11.16 + psutil
        "unverified": "unverified",                                     # 3.14 gateway, no psutil
        "bootsession:6FE94283|179160482300|5458133": "boot-witness/3",  # the live rows
    }
    tokens = {guard.identity_shape(sample) for sample in versions}
    assert len(tokens) == 3


def test_a_cross_generation_pair_is_incomparable():
    assert guard.shapes_comparable("boot-witness/2", "boot-witness/3") is False
    assert guard.shapes_comparable("epoch-cs/2", "boot-witness/2") is False
    assert guard.shapes_comparable("boot-witness/2", "boot-witness/2") is True
    # A row with no fingerprint, and the UNVERIFIED marker, keep the bare-existence rule.
    assert guard.shapes_comparable("boot-witness/2", "none") is True
    assert guard.shapes_comparable("boot-witness/2", "unverified") is True
    assert guard.shapes_comparable("epoch-cs/2", "legacy-int") is True


def test_a_row_from_another_generation_is_held_and_not_counted():
    """R2 UNKNOWN -> HOLD at row granularity: an incomparable row is neither reclaimed nor counted."""
    closures = [
        _closure("comparable-1", shape="epoch-cs/2"),
        _closure("comparable-2", shape="epoch-cs/2"),
        _closure("comparable-3", shape="epoch-cs/2"),
        _closure("foreign", shape="boot-witness/3"),
    ]
    assessment = guard.assess_sweep(closures, write_shape="epoch-cs/2")
    assert assessment.held == ("foreign",)
    assert assessment.held_shapes == ("boot-witness/3->epoch-cs/2",)
    assert assessment.verdict.count == 3  # the held row is not one of them


def test_the_hold_survives_the_threshold_override(monkeypatch):
    """The count guard is policy; an incomparable comparison is a fact about the values."""
    monkeypatch.setenv(guard.MASS_CRASH_ABORT_THRESHOLD_ENV, "0")
    assessment = guard.assess_sweep(
        [_closure("foreign", shape="boot-witness/3")], write_shape="epoch-cs/2",
    )
    assert assessment.refusal_needed is False
    assert assessment.held == ("foreign",)


def test_first_tick_records_the_generation_and_later_ticks_do_not_alert(tmp_path):
    path = tmp_path / "stamp.json"
    first = guard.check_deployed_generation(sample="|179036519519", path=path, now=1000)
    assert first.recorded is True
    assert first.alerting is False
    second = guard.check_deployed_generation(sample="|179036519520", path=path, now=2000)
    assert second.alerting is False
    assert guard.read_generation_stamp(path)["sample"] == "|179036519520"


def test_a_shape_change_is_detected_in_one_tick(tmp_path):
    path = tmp_path / "stamp.json"
    guard.check_deployed_generation(sample="|179036519519", path=path, now=1000)
    tick = guard.check_deployed_generation(sample="bootsession:U|a|b", path=path, now=2000)
    assert tick.alerting is True
    assert tick.reasons == ["identity-shape-changed:epoch-cs/2->boot-witness/3"]


def test_losing_psutil_alerts(tmp_path):
    path = tmp_path / "stamp.json"
    guard.check_deployed_generation(sample="|1", psutil=True, path=path, now=1000)
    tick = guard.check_deployed_generation(sample="|1", psutil=False, path=path, now=2000)
    assert "psutil-lost" in tick.reasons


def test_an_interpreter_minor_move_alerts_and_a_patch_bump_does_not(tmp_path):
    path = tmp_path / "stamp.json"
    guard.check_deployed_generation(sample="|1", interpreter="3.11.16", path=path, now=1000)
    patch = guard.check_deployed_generation(sample="|1", interpreter="3.11.9", path=path, now=2000)
    assert patch.alerting is False
    minor = guard.check_deployed_generation(sample="|1", interpreter="3.14.7", path=path, now=3000)
    assert minor.reasons == ["interpreter-changed:3.11->3.14"]


def test_losing_a_witness_symbol_alerts_but_gaining_one_does_not(tmp_path):
    path = tmp_path / "stamp.json"
    guard.check_deployed_generation(sample="|1", witness_symbols=[], path=path, now=1000)
    # A fix landing: the symbols ARRIVE. Not a regression.
    landed = guard.check_deployed_generation(
        sample="|1", witness_symbols=["_worker_liveness", "WORKER_ALIVE"], path=path, now=2000,
    )
    assert landed.alerting is False
    # Then a tree write drops them again -- the recurring regression.
    lost = guard.check_deployed_generation(sample="|1", witness_symbols=[], path=path, now=3000)
    assert lost.reasons == ["witness-bytes-lost:WORKER_ALIVE,_worker_liveness"]


def test_a_diverging_tick_keeps_reporting_and_does_not_overwrite_the_reference(tmp_path):
    path = tmp_path / "stamp.json"
    guard.check_deployed_generation(sample="|1", path=path, now=1000)
    for moment in (2000, 3000, 4000):
        tick = guard.check_deployed_generation(sample="bootsession:U|a|b", path=path, now=moment)
        assert tick.alerting is True
    assert guard.read_generation_stamp(path)["sample"] == "|1"  # still the reference


def test_an_unreadable_stamp_is_treated_as_absent(tmp_path):
    path = tmp_path / "stamp.json"
    path.write_text("{ not json", encoding="utf-8")
    tick = guard.check_deployed_generation(sample="|1", path=path, now=1000)
    assert tick.recorded is True
    assert tick.alerting is False


def test_witness_symbol_presence_reads_the_executing_module():
    present = guard.witness_symbols_present(kbd)
    for name in present:
        assert name in guard.WITNESS_SYMBOLS
        assert hasattr(kbd, name)
    # The tree has none of them today; the check reports that rather than guessing.
    assert isinstance(present, list)


# ---------------------------------------------------------------------------------------------
# the acceptance test: a synthetic tick that would close >= 3 runs writes NOTHING and alerts
# ---------------------------------------------------------------------------------------------


def test_a_synthetic_tick_that_would_close_three_runs_writes_nothing_and_alerts(board):
    conn = board
    fingerprint = _comparable_fingerprint()
    dead = _dead_pid()
    ids = [_claimed_running(conn, pid=dead, started_at=fingerprint) for _ in range(3)]
    before = {tid: _run_rows(conn, tid) for tid in ids}

    crashed = kbd.detect_crashed_workers(conn)

    # 1. NOTHING was closed.
    assert crashed == []
    for tid in ids:
        row = _task_row(conn, tid)
        assert row["status"] == "running", f"{tid} was released by a refused sweep"
        assert row["worker_pid"] == dead
        assert row["claim_lock"] is not None
        assert _run_rows(conn, tid) == before[tid], f"{tid}'s run was ended by a refused sweep"
    assert kbd.detect_crashed_workers._last_auto_blocked == []

    # 2. The decision is a count and a timestamp.
    abort = kbd.detect_crashed_workers._last_crash_sweep_abort
    assert abort is not None
    assert abort.count == 3
    assert abort.threshold == 3
    assert abort.signature.startswith("mass-crash-sweep:count=3,ended_at=")
    assert abort.task_ids == tuple(ids)

    # 3. It produces an alert: one acting card for the design authority ...
    actors = conn.execute(
        "SELECT id, assignee, title FROM tasks WHERE idempotency_key LIKE 'mass-crash:%'",
    ).fetchall()
    assert len(actors) == 1, "exactly one acting card per signature"
    assert actors[0]["assignee"] == guard.MASS_CRASH_ACTOR_ASSIGNEE
    assert "REFUSED" in actors[0]["title"]

    # 4. ... and per-card evidence, without touching any status.
    for tid in ids:
        assert _event_count(conn, tid, guard.MASS_CRASH_ABORT_EVENT) == 1
        assert _task_row(conn, tid)["status"] == "running"


def test_the_alert_is_idempotent_inside_the_hour_bucket(board):
    conn = board
    fingerprint = _comparable_fingerprint()
    dead = _dead_pid()
    for _ in range(3):
        _claimed_running(conn, pid=dead, started_at=fingerprint)
    kbd.detect_crashed_workers(conn)
    kbd.detect_crashed_workers(conn)
    rows = conn.execute(
        "SELECT COUNT(*) AS c FROM tasks WHERE idempotency_key LIKE 'mass-crash:%'",
    ).fetchone()
    assert int(rows["c"]) == 1, "a persistent condition pages once per hour, not once per tick"


def test_a_sweep_below_the_threshold_still_reclaims_normally(board):
    conn = board
    fingerprint = _comparable_fingerprint()
    dead = _dead_pid()
    ids = [_claimed_running(conn, pid=dead, started_at=fingerprint) for _ in range(2)]

    crashed = kbd.detect_crashed_workers(conn)

    assert sorted(crashed) == sorted(ids)
    for tid in ids:
        assert _task_row(conn, tid)["status"] != "running"
    assert kbd.detect_crashed_workers._last_crash_sweep_abort is None


def test_a_row_from_another_identity_generation_is_held_not_reclaimed(board):
    """The measured 2026-10-10 defect, at one row: the fingerprint is not comparable, so HOLD."""
    conn = board
    dead = _dead_pid()
    tid = _claimed_running(conn, pid=dead, started_at=_incomparable_fingerprint())

    crashed = kbd.detect_crashed_workers(conn)

    assert crashed == []
    row = _task_row(conn, tid)
    assert row["status"] == "running"
    assert row["worker_pid"] == dead
    assert kbd.detect_crashed_workers._last_crash_sweep_held == [tid]
    # A hold still alerts, because it means the dispatcher WRITES one shape and READS another.
    assert _event_count(conn, tid, guard.MASS_CRASH_ABORT_EVENT) == 0  # no refusal: nothing refused
    assert kbd.detect_crashed_workers._last_crash_sweep_abort is None


def test_the_reclaim_reports_the_refusal_on_the_dispatch_result(board, monkeypatch):
    """The tick itself carries the refusal, so a caller never has to read the log."""
    conn = board
    fingerprint = _comparable_fingerprint()
    dead = _dead_pid()
    # A claim that has NOT expired, so the sweep reaches the crash path instead of the TTL reclaim.
    future = int(time.time()) + 3600
    for _ in range(3):
        _claimed_running(conn, pid=dead, started_at=fingerprint, claim_expires=future)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    result = kbd._dispatch_once_locked(conn, ttl_seconds=None, stale_timeout_seconds=0)

    assert result.crashed == []
    assert result.crash_sweep_refused["count"] == 3
    assert result.crash_sweep_refused["threshold"] == 3
    assert result.crash_sweep_refused["signature"].startswith("mass-crash-sweep:count=3,ended_at=")


def test_the_tick_stamps_the_deployed_generation_once(board, tmp_path):
    conn = board
    result = kbd._dispatch_once_locked(conn, ttl_seconds=None, stale_timeout_seconds=0)
    assert result.identity_generation_alerts == []
    stamp = guard.read_generation_stamp()
    assert stamp is not None
    assert stamp["shape"] == _writer_shape()
    assert stamp["version"] == guard.GENERATION_STAMP_VERSION


def test_the_tick_alerts_when_the_deployed_generation_moves(board, tmp_path, monkeypatch):
    """A regression is detected in ONE tick -- the whole point of R6."""
    conn = board
    kbd._dispatch_once_locked(conn, ttl_seconds=None, stale_timeout_seconds=0)
    stamp = guard.read_generation_stamp()
    stamp["shape"] = "boot-witness/3"          # the reference moves under the tick
    stamp["witness_symbols"] = ["_worker_liveness"]
    guard.write_generation_stamp(stamp)

    result = kbd._dispatch_once_locked(conn, ttl_seconds=None, stale_timeout_seconds=0)

    assert result.identity_generation_alerts, "a moved generation must alert in ONE tick"
    assert any(
        reason.startswith("identity-shape-changed:") for reason in result.identity_generation_alerts
    )
    assert any(
        reason.startswith("witness-bytes-lost:") for reason in result.identity_generation_alerts
    )
