"""Dispatch-plane freshness: a shipped edit to a dispatch module must not stay invisible.

Covers kanban t_50d090c0 — the two 2026-09-24/25 incidents where ``hermes_cli/kanban_db_dispatch.py``
(and the goal-mode policy beside it) were edited, reviewed and verified while the live gateway kept
serving the module it imported at boot.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from gateway import dispatch_freshness as df


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A fake checkout with one watched module, plus a clean process-local probe state."""
    root = tmp_path / "repo"
    (root / "hermes_cli").mkdir(parents=True)
    module = root / "hermes_cli" / "kanban_db_dispatch.py"
    module.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(df, "_PROJECT_ROOT", root)
    monkeypatch.setattr(df, "WATCHED_SOURCES", ("hermes_cli/kanban_db_dispatch.py",))
    monkeypatch.setattr(df, "_router_script_path", lambda: None)
    monkeypatch.setattr(df, "_boot", None)
    df.reset_state()
    return root, module


def _bump(path: Path, *, seconds: float = 5.0) -> None:
    """Push a file's mtime into the future — the only thing this check keys on."""
    future = time.time() + seconds
    os.utime(path, (future, future))


def _fake_log(caplog):
    return logging.getLogger("test.dispatch_freshness")


# ---- matching (the positive case: criterion 1) ----------------------------------------------------

def test_recorded_snapshot_then_edit_is_stale_and_named(tree):
    _root, module = tree
    df.record_boot()
    assert df.detect_stale() is None, "a process that just snapshotted its own code is clean"

    _bump(module)
    detected = df.detect_stale()
    assert detected is not None
    _boot_mtime_ns, current, newer = detected
    assert newer == ("hermes_cli/kanban_db_dispatch.py",)
    assert current["newest"] == "hermes_cli/kanban_db_dispatch.py"


def test_edit_before_boot_is_not_stale(tree):
    """Criterion 2, unit-level: a clean boot after the last edit must produce no detection."""
    _root, module = tree
    _bump(module)
    df.record_boot()
    assert df.detect_stale() is None


def test_record_boot_is_idempotent(tree):
    """A later edit must not be absorbed into the boot snapshot (that is the whole detection)."""
    _root, module = tree
    first = df.record_boot()
    _bump(module)
    second = df.record_boot()
    assert first == second, "an mtime bump after boot must not silently re-snapshot"
    assert df.detect_stale() is not None


def test_new_watched_module_counts_as_stale(tmp_path, monkeypatch):
    """A module that did not exist at boot (goal_mode's companion) is stale once it appears."""
    root = tmp_path / "repo"
    (root / "hermes_cli").mkdir(parents=True)
    monkeypatch.setattr(df, "_PROJECT_ROOT", root)
    monkeypatch.setattr(df, "WATCHED_SOURCES", ("hermes_cli/kanban_db_dispatch.py", "hermes_cli/kanban_goal_policy.py"))
    monkeypatch.setattr(df, "_router_script_path", lambda: None)
    monkeypatch.setattr(df, "_boot", None)
    df.reset_state()
    (root / "hermes_cli" / "kanban_db_dispatch.py").write_text("value = 1\n", encoding="utf-8")
    df.record_boot()

    (root / "hermes_cli" / "kanban_goal_policy.py").write_text("POLICY = 'default'\n", encoding="utf-8")
    detected = df.detect_stale()
    assert detected is not None
    assert detected[2] == ("hermes_cli/kanban_goal_policy.py",)


# ---- the serving process's probe ------------------------------------------------------------------

def test_probe_needs_two_consecutive_probes_then_warns_once(tree, caplog):
    _root, module = tree
    df.record_boot()
    _bump(module)

    assert df.probe_and_act(_fake_log(caplog), auto_reload=False) == "unstable"
    with caplog.at_level(logging.INFO, logger="test.dispatch_freshness"):
        assert df.probe_and_act(_fake_log(caplog), auto_reload=False) == "warned"
        assert df.settled() is True
        assert df.probe_and_act(_fake_log(caplog), auto_reload=False) == "settled"

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, warnings
    assert "STALE DISPATCH PLANE" in warnings[0]
    assert r"hermes_cli/kanban_db_dispatch.py" in warnings[0]
    assert "hermes gateway restart" in warnings[0]


def test_probe_is_silent_on_a_clean_boot(tree, caplog):
    df.record_boot()
    with caplog.at_level(logging.DEBUG):
        for _ in range(3):
            assert df.probe_and_act(_fake_log(caplog), auto_reload=True) == "clean"
    assert caplog.records == []


def test_probe_never_reloads_onto_uncompilable_code(tree, caplog):
    """A half-written module has a fresh mtime; restarting onto it would take the gateway down."""
    _root, module = tree
    df.record_boot()
    module.write_text("def broken(:\n", encoding="utf-8")
    _bump(module)
    requested = []

    with caplog.at_level(logging.INFO, logger="test.dispatch_freshness"):
        assert df.probe_and_act(_fake_log(caplog), running_workers=lambda: 0,
                                auto_reload=True, request_restart=lambda: requested.append(1)) == "unstable"
        assert df.probe_and_act(_fake_log(caplog), running_workers=lambda: 0,
                                auto_reload=True, request_restart=lambda: requested.append(1)) == "broken"
        assert df.probe_and_act(_fake_log(caplog), running_workers=lambda: 0,
                                auto_reload=True, request_restart=lambda: requested.append(1)) == "broken"

    assert requested == [], "an unimportable module must never become a restart target"
    assert df.settled() is False, "the file may still be mid-write; the check stays live"
    assert len([r for r in caplog.records if "does not compile" in r.getMessage()]) == 1


# ---- the deferred restart (criterion 4) -----------------------------------------------------------

def test_auto_reload_defers_until_no_worker_runs(tree, caplog):
    _root, module = tree
    df.record_boot()
    _bump(module)
    requested = []
    active = {"n": 2}

    def _request():
        requested.append(1)
        return True

    with caplog.at_level(logging.INFO, logger="test.dispatch_freshness"):
        assert df.probe_and_act(_fake_log(caplog), running_workers=lambda: active["n"],
                                auto_reload=True, request_restart=_request) == "unstable"
        assert df.probe_and_act(_fake_log(caplog), running_workers=lambda: active["n"],
                                auto_reload=True, request_restart=_request) == "deferred"
        assert requested == [], "never bounce the gateway out from under a running worker"

        active["n"] = 1
        assert df.probe_and_act(_fake_log(caplog), running_workers=lambda: active["n"],
                                auto_reload=True, request_restart=_request) == "deferred"

        active["n"] = 0
        assert df.probe_and_act(_fake_log(caplog), running_workers=lambda: active["n"],
                                auto_reload=True, request_restart=_request) == "requested"
        assert requested == [1]
        assert df.settled() is True

    deferred = [r.getMessage() for r in caplog.records if "restart deferred" in r.getMessage()]
    assert len(deferred) == 2, deferred  # one per distinct active count, not one per tick
    assert "waiting on 2 active kanban worker(s)" in deferred[0]
    assert "waiting on 1 active kanban worker(s)" in deferred[1]
    assert any("dispatch-plane reload requested" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("workers", [lambda: None, lambda: (_ for _ in ()).throw(RuntimeError("db gone"))])
def test_auto_reload_treats_an_unknown_worker_count_as_busy(tree, caplog, workers):
    _root, module = tree
    df.record_boot()
    _bump(module)
    requested = []
    with caplog.at_level(logging.INFO, logger="test.dispatch_freshness"):
        df.probe_and_act(_fake_log(caplog), running_workers=workers, auto_reload=True,
                         request_restart=lambda: requested.append(1) or True)
        assert df.probe_and_act(_fake_log(caplog), running_workers=workers, auto_reload=True,
                                request_restart=lambda: requested.append(1) or True) == "deferred"
    assert requested == []
    assert any("waiting on unknown active kanban worker(s)" in r.getMessage() for r in caplog.records)


def test_auto_reload_off_warns_without_acting(tree, caplog):
    _root, module = tree
    df.record_boot()
    _bump(module)
    requested = []
    with caplog.at_level(logging.INFO, logger="test.dispatch_freshness"):
        df.probe_and_act(_fake_log(caplog), running_workers=lambda: 0, auto_reload=False,
                         request_restart=lambda: requested.append(1) or True)
        assert df.probe_and_act(_fake_log(caplog), running_workers=lambda: 0, auto_reload=False,
                                request_restart=lambda: requested.append(1) or True) == "warned"
    assert requested == [], "auto-reload off must never request a restart"


# ---- judging a persisted gateway record (the ``hermes doctor`` side) ------------------------------

def test_boot_stamp_fields_empty_before_a_snapshot(tree):
    assert df.boot_stamp_fields() == {}


def test_boot_stamp_fields_after_snapshot(tree):
    _root, module = tree
    df.record_boot()
    stamp = df.boot_stamp_fields()
    assert stamp["dispatch_code_files"] == 1
    assert stamp["dispatch_code_newest"] == "hermes_cli/kanban_db_dispatch.py"
    assert stamp["dispatch_code_max_mtime_ns"] == int(module.stat().st_mtime_ns)


def test_judge_record_uses_the_boot_snapshot_stamp(tree):
    _root, module = tree
    df.record_boot()
    stamp = df.boot_stamp_fields()
    assert df.judge_record(stamp).status == "fresh"

    _bump(module)
    verdict = df.judge_record(stamp)
    assert verdict.status == "stale"
    assert verdict.source == "stamp"
    assert verdict.stale_files == ("hermes_cli/kanban_db_dispatch.py",)


def test_judge_record_falls_back_to_the_process_start_time(tree):
    _root, module = tree
    # ns -> epoch centiseconds, rounded UP: the derived boot instant is a rounded stamp, so a file
    # written within that same centisecond must still read as loaded.
    mtime_ns = int(module.stat().st_mtime_ns)
    booted = -(-mtime_ns // 10_000_000)
    assert df.judge_record({"start_time": booted}).status == "fresh"

    _bump(module)
    verdict = df.judge_record({"start_time": booted})
    assert (verdict.status, verdict.source) == ("stale", "start_time")


@pytest.mark.parametrize("record", [{}, {"start_time": 4_200}, {"start_time": None}, {"start_time": True}])
def test_judge_record_is_unknown_without_a_usable_boot_instant(tree, record):
    """Linux keeps a /proc tick counter in ``start_time``; an unreadable boot instant warns about nothing."""
    verdict = df.judge_record(record)
    assert verdict.status == "unknown"
    assert "restart the gateway" in verdict.reason


def test_judge_record_unknown_when_nothing_is_watchable(tmp_path, monkeypatch):
    monkeypatch.setattr(df, "_PROJECT_ROOT", tmp_path / "empty")
    monkeypatch.setattr(df, "_router_script_path", lambda: None)
    assert df.judge_record({}).status == "unknown"


# ---- config surfaces ------------------------------------------------------------------------------

def test_auto_reload_defaults_on_and_honours_both_overrides(monkeypatch):
    monkeypatch.delenv(df._AUTO_RELOAD_ENV, raising=False)
    assert df.auto_reload_enabled({}) is True
    assert df.auto_reload_enabled({"dispatch_auto_reload": False}) is False
    assert df.auto_reload_enabled({"dispatch_auto_reload": "false"}) is False
    assert df.auto_reload_enabled({"dispatch_auto_reload": True}) is True
    monkeypatch.setenv(df._AUTO_RELOAD_ENV, "0")
    assert df.auto_reload_enabled({"dispatch_auto_reload": True}) is False
    monkeypatch.setenv(df._AUTO_RELOAD_ENV, "1")
    assert df.auto_reload_enabled({"dispatch_auto_reload": False}) is True
    assert df._AUTO_RELOAD_CONFIG_KEY in df.reload_opt_out_hint()



# ---- the boot -> gateway_state.json link the doctor check reads -----------------------------------

def test_status_identity_fields_carry_the_boot_snapshot(tree):
    """The whole mechanism is only visible to `hermes doctor` if the snapshot reaches the record."""
    from gateway import status as gw_status

    _root, module = tree
    assert df.boot_stamp_fields() == {}, "a process that never snapshotted must not claim freshness"

    df.record_boot()
    fields = df.boot_stamp_fields()
    assert set(fields) == {"dispatch_code_max_mtime_ns", "dispatch_code_files", "dispatch_code_newest"}
    assert fields["dispatch_code_files"] == len(df.watched_paths())

    identity = gw_status._get_code_identity_fields()
    assert identity["dispatch_code_max_mtime_ns"] == fields["dispatch_code_max_mtime_ns"]
    assert identity["dispatch_code_files"] == fields["dispatch_code_files"]
    assert identity["dispatch_code_newest"] == fields["dispatch_code_newest"]

    # ...and an edit after boot must show up in the very next record the gateway writes.
    _bump(module)
    assert gw_status._get_code_identity_fields()["dispatch_code_max_mtime_ns"] < module.stat().st_mtime_ns
    assert df.judge_record(gw_status._get_code_identity_fields()).status == "stale"


def test_router_script_is_watched_from_the_dispatcher_constant(tmp_path, monkeypatch):
    """The Jev router lives outside the package; its path is read from the dispatcher, never duplicated."""
    script = tmp_path / "jev_router.py"
    script.write_text("print('{}')\n", encoding="utf-8")
    monkeypatch.setattr(df, "_ROUTER_SCRIPT_ATTR", ("hermes_cli.kanban_db_dispatch", "_JEV_ROUTER_SCRIPT"))
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._JEV_ROUTER_SCRIPT", str(script), raising=False)
    assert df._router_script_path() == script
    assert ("jev_router.py", script) in df.watched_paths()


# ---- the gate's data source: a real claim is a real running worker --------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (mirrors tests/hermes_cli/test_kanban_goal_policy.py)."""
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_running_worker_count_is_taken_from_the_live_board(kanban_home):
    """Criterion 4's signal, end to end: a claimed card counts, and an unknown count is never 0."""
    from gateway.kanban_watchers import _running_kanban_workers
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    assert _running_kanban_workers(kb) == 0

    conn = kbc.connect()
    try:
        card = kb.create_task(conn, title="busy worker", assignee="hephaestus")
        kb.claim_task(conn, card)
    finally:
        conn.close()
    assert _running_kanban_workers(kb) == 1

    conn = kbc.connect()
    try:
        kb.complete_task(conn, card, summary="done")
    finally:
        conn.close()
    assert _running_kanban_workers(kb) == 0
