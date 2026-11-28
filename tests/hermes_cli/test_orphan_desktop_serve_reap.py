"""Orphan Desktop-local ``hermes serve`` reap at backend start.

When Desktop dies uncleanly, local ``serve --host 127.0.0.1 --port 0``
children can be reparented to pid 1 and keep full MCP trees alive. The next
boot must clear those corpses without touching intentional fixed-port serves
(e.g. ``--port 9119`` remote dashboards).
"""

from __future__ import annotations

import os
from unittest.mock import patch

from hermes_cli.dashboard_procs import (
    _is_desktop_local_serve_cmdline,
    _reap_orphaned_desktop_local_serves,
)


def test_reap_refuses_indeterminate_lock_ownership():
    with patch("hermes_cli.dashboard_procs._scan_dashboard_processes") as scan:
        result = _reap_orphaned_desktop_local_serves(
            lock_owned_pids_fn=lambda: (_ for _ in ()).throw(OSError("unreadable")),
            sleep_fn=lambda _: None,
        )
    assert result["matched"] == []
    scan.assert_not_called()


def test_reap_refuses_unreadable_default_lock_directory():
    from hermes_cli import dashboard_procs
    class Unreadable:
        def is_dir(self):
            return True
        def iterdir(self):
            raise PermissionError("denied")
    with (
        patch.object(dashboard_procs, "_remote_lock_roots", return_value=[Unreadable()]),
        patch.object(dashboard_procs, "_scan_dashboard_processes") as scan,
    ):
        assert _reap_orphaned_desktop_local_serves(sleep_fn=lambda _: None)["matched"] == []
    scan.assert_not_called()


def test_reap_refuses_existing_malformed_lock(tmp_path):
    from hermes_cli import dashboard_procs
    import pytest
    lock_root = tmp_path / "desktop-ssh"
    owner = lock_root / ("a" * 32)
    owner.mkdir(parents=True)
    (owner / "backend.lock.json").write_text("{bad json")
    assert dashboard_procs._lock_owned_serve_pids(lock_root) == set()  # non-destructive reader
    with pytest.raises(ValueError, match="uncertain remote backend lock"):
        dashboard_procs._lock_owned_serve_pids(lock_root, strict=True)
    # An existing but unreadable file has the same fail-closed behavior.
    with patch.object(dashboard_procs, "read_valid_backend_lock", return_value=None):
        with pytest.raises(ValueError, match="uncertain remote backend lock"):
            dashboard_procs._lock_owned_serve_pids(lock_root, strict=True)
    with (
        patch.object(dashboard_procs, "_remote_lock_roots", return_value=[lock_root]),
        patch.object(dashboard_procs, "_scan_dashboard_processes") as scan,
    ):
        assert _reap_orphaned_desktop_local_serves(sleep_fn=lambda _: None)["matched"] == []
    scan.assert_not_called()


def test_reap_refuses_pid_recycled_during_candidate_selection():
    from gateway import status
    births = {123456: 42}
    signals = []
    with (
        patch("hermes_cli.dashboard_procs._scan_dashboard_processes", return_value=[
            (123456, "hermes serve --host 127.0.0.1 --port 0")]),
        patch("hermes_cli.dashboard_procs._process_ppid", return_value=1),
        patch("hermes_cli.dashboard_procs._posix_descendants", return_value={}),
        patch.object(status, "get_process_start_time", side_effect=lambda pid: births.get(pid)),
        patch("os.kill", side_effect=lambda pid, sig: signals.append((pid, sig))),
    ):
        def age(pid):
            births[pid] = 43  # PID was recycled after scan, before the candidate is selected
            return 600
        result = _reap_orphaned_desktop_local_serves(
            lock_owned_pids_fn=lambda: set(), process_age_seconds_fn=age,
            sleep_fn=lambda _: None,
        )
    assert result["matched"] == [123456]
    assert result["failed"] == [123456]
    assert signals == []


def test_reap_refuses_recycled_root_and_late_lock():
    from gateway import status
    births = {123456: 42}
    signals = []
    lock_scans = iter((set(), set(), set(), {123456}))
    with (
        patch("hermes_cli.dashboard_procs._scan_dashboard_processes", return_value=[
            (123456, "hermes serve --host 127.0.0.1 --port 0")]),
        patch("hermes_cli.dashboard_procs._process_ppid", return_value=1),
        patch("hermes_cli.dashboard_procs._posix_descendants", return_value={}),
        patch.object(status, "get_process_start_time", side_effect=lambda pid: births.get(pid)),
        patch("os.kill", side_effect=lambda pid, sig: signals.append((pid, sig))),
    ):
        result = _reap_orphaned_desktop_local_serves(
            lock_owned_pids_fn=lambda: next(lock_scans),
            process_age_seconds_fn=lambda _: 600,
            sleep_fn=lambda _: births.update({123456: 43}),
        )
    assert result["matched"] == [123456]
    assert [(pid, sig) for pid, sig in signals if sig != 0] == [(123456, 15)]
    assert result["killed"] == []


def test_reap_refuses_lock_appearing_before_term():
    from gateway import status
    signals = []
    locks = iter((set(), set(), {123456}))
    with (
        patch("hermes_cli.dashboard_procs._scan_dashboard_processes", return_value=[
            (123456, "hermes serve --host 127.0.0.1 --port 0")]),
        patch("hermes_cli.dashboard_procs._process_ppid", return_value=1),
        patch("hermes_cli.dashboard_procs._posix_descendants", return_value={}),
        patch.object(status, "get_process_start_time", return_value=42),
        patch("os.kill", side_effect=lambda pid, sig: signals.append((pid, sig))),
    ):
        result = _reap_orphaned_desktop_local_serves(
            lock_owned_pids_fn=lambda: next(locks),
            process_age_seconds_fn=lambda _: 600,
            sleep_fn=lambda _: None,
        )
    assert signals == []
    assert result["failed"] == [123456]


def test_desktop_local_serve_shape_matches_ephemeral_loopback():
    assert _is_desktop_local_serve_cmdline(
        "python -m hermes_cli.main serve --host 127.0.0.1 --port 0"
    )
    assert _is_desktop_local_serve_cmdline(
        "hermes serve --isolated --host 127.0.0.1 --port 0 --ssh-owner-nonce abc"
    )
    assert _is_desktop_local_serve_cmdline(
        "/venv/bin/hermes serve --host=127.0.0.1 --port=0"
    )


def test_desktop_local_serve_shape_spares_fixed_port_and_non_serve():
    assert not _is_desktop_local_serve_cmdline(
        "hermes serve --host 100.106.105.2 --port 9119 --skip-build"
    )
    assert not _is_desktop_local_serve_cmdline(
        "hermes serve --host 127.0.0.1 --port 9119"
    )
    assert not _is_desktop_local_serve_cmdline("hermes gateway run --replace")
    # "serve" inside another token is not the serve subcommand (token matcher, not substring).
    assert not _is_desktop_local_serve_cmdline(
        "hermes kanban --preserve-cache --host 127.0.0.1 --port 0"
    )


def test_reap_only_kills_ppid1_local_serves():
    scanned = [
        (111, "hermes serve --host 127.0.0.1 --port 0"),  # orphan local
        (222, "hermes serve --host 127.0.0.1 --port 0"),  # still has parent
        (333, "hermes serve --host 100.1.2.3 --port 9119"),  # fixed remote
        (444, "hermes serve --isolated --host 127.0.0.1 --port 0"),  # orphan isolated
    ]
    ppids = {111: 1, 222: 50, 333: 1, 444: 1}
    terms: list[int] = []
    live = {111, 222, 333, 444}

    def fake_kill(pid, sig):
        if sig == 0:
            if pid in live:
                return
            raise ProcessLookupError()
        if sig == 15:
            terms.append(pid)
            live.discard(pid)
            return
        if sig == 9:
            live.discard(pid)
            return
        return

    with (
        patch(
            "hermes_cli.dashboard_procs._scan_dashboard_processes",
            return_value=scanned,
        ),
        patch(
            "hermes_cli.dashboard_procs._process_ppid",
            side_effect=lambda pid: ppids.get(pid),
        ),
        patch("os.kill", side_effect=fake_kill),
        patch("gateway.status.get_process_start_time", side_effect=lambda pid: 42 if pid in live else None),
    ):
        os.environ.pop("HERMES_DESKTOP_CHILD_PID", None)
        result = _reap_orphaned_desktop_local_serves(
            sleep_fn=lambda _s: None,
            signal_term=15,
            signal_kill=9,
            process_age_seconds_fn=lambda _pid: 600.0,
        )

    assert set(result["matched"]) == {111, 444}
    assert set(terms) == {111, 444}
    assert set(result["killed"]) == {111, 444}
    assert 222 not in terms
    assert 333 not in terms


def test_reap_passes_child_pid_exclude_to_scan():
    with (
        patch(
            "hermes_cli.dashboard_procs._scan_dashboard_processes",
            return_value=[],
        ) as scan,
        patch.dict(os.environ, {"HERMES_DESKTOP_CHILD_PID": "999,111"}, clear=False),
    ):
        result = _reap_orphaned_desktop_local_serves(sleep_fn=lambda _s: None)

    assert result["matched"] == []
    exclude = scan.call_args.kwargs["exclude_pids"]
    assert 111 in exclude
    assert 999 in exclude


# ---------------------------------------------------------------------------
# Regression: a legitimately lock-owned SSH remote backend (started by another
# client/machine) must survive the boot reap even though it matches the
# Desktop-local serve shape and is orphaned at ppid 1. Production incident:
# the local Desktop app's reboot reap killed a Mac Mini backend that a MacBook
# had launched over SSH — its PID was the recorded owner in backend.lock.json.
# ---------------------------------------------------------------------------

import json
from hermes_cli.dashboard_procs import (
    _lock_owned_serve_pids,
    _valid_lockfile_payload,
)


def _valid_lock_payload(pid: int, ownership_id: str, spawn_nonce: str) -> dict:
    """A minimally-valid backend.lock.json body matching remote-lifecycle.ts."""
    return {
        "schemaVersion": 2,
        "protocolVersion": 1,
        "ownershipId": ownership_id,
        "spawnNonce": spawn_nonce,
        "tokenFingerprint": "a" * 32,
        "pid": pid,
        "port": 0,
        "profile": "default",
        "hermesPath": "/opt/hermes/bin/hermes",
        "hermesHome": "~/.hermes",
        "logPath": f"~/.hermes/desktop-ssh/{ownership_id}/{spawn_nonce}.log",
        "startedAt": "2026-08-04T20:00:00Z",
    }


def test_lock_owned_serve_pids_reads_valid_backend_lock(tmp_path):
    oid = "f" * 32
    nonce = "d" * 16
    lock_root = tmp_path / "desktop-ssh"
    (lock_root / oid).mkdir(parents=True)
    (lock_root / oid / "backend.lock.json").write_text(
        json.dumps(_valid_lock_payload(7777, oid, nonce))
    )
    # A second, malformed lock (bad schemaVersion) must contribute nothing.
    other_oid = "e" * 32
    (lock_root / other_oid).mkdir(parents=True)
    (lock_root / other_oid / "backend.lock.json").write_text(
        json.dumps({**_valid_lock_payload(8888, other_oid, nonce), "schemaVersion": 99})
    )
    assert _lock_owned_serve_pids(base_dir=lock_root) == {7777}


def test_lock_owned_serve_pids_sees_root_home_locks_from_a_profile_home(tmp_path, monkeypatch):
    """A profile backend (``HERMES_HOME=<root>/profiles/<name>``) must still see the Desktop's SSH
    locks, which live under ``<root>/desktop-ssh`` — otherwise its reaper kills the sibling
    profile's live SSH backend on every profile switch (#89811)."""
    import hermes_constants

    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "flocki"
    profile_home.mkdir(parents=True)
    oid = "f" * 32
    nonce = "d" * 16
    (root / "desktop-ssh" / oid).mkdir(parents=True)
    (root / "desktop-ssh" / oid / "backend.lock.json").write_text(
        json.dumps(_valid_lock_payload(7777, oid, nonce))
    )
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: root)
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None, raising=False)

    assert _lock_owned_serve_pids() == {7777}


def test_valid_lockfile_payload_rejects_wrong_owner_and_shape():
    oid = "f" * 32
    nonce = "d" * 16
    good = _valid_lockfile_payload(_valid_lock_payload(1, oid, nonce), oid)
    assert good is True
    # ownershipId must match the directory it lives in.
    assert _valid_lockfile_payload(
        _valid_lock_payload(1, "e" * 32, nonce), oid
    ) is False
    # pid out of range.
    bad_pid = _valid_lock_payload(1, oid, nonce)
    bad_pid["pid"] = 0
    assert _valid_lockfile_payload(bad_pid, oid) is False
    # port out of range.
    bad_port = _valid_lock_payload(1, oid, nonce)
    bad_port["port"] = 70000
    assert _valid_lockfile_payload(bad_port, oid) is False
    # spawnNonce wrong length.
    bad_nonce = _valid_lock_payload(1, oid, nonce)
    bad_nonce["spawnNonce"] = "z" * 15
    assert _valid_lockfile_payload(bad_nonce, oid) is False
    # logPath not ending in <oid>/<nonce>.log.
    bad_log = _valid_lock_payload(1, oid, nonce)
    bad_log["logPath"] = f"~/.hermes/desktop-ssh/{oid}/other.log"
    assert _valid_lockfile_payload(bad_log, oid) is False


def test_reap_spare_lock_owned_ssh_remote_backend_of_foreign_client():
    """The exact production-incident shape: a foreign-client SSH remote backend
    matches the Desktop-local serve shape and is orphaned at ppid 1, but a valid
    backend.lock.json owns its PID. The reap must NOT kill it."""
    scanned = [
        (555, "hermes serve --host 127.0.0.1 --port 0"),  # lock-owned remote
        (666, "hermes serve --host 127.0.0.1 --port 0"),  # genuine orphan
    ]
    ppids = {555: 1, 666: 1}
    terms: list[int] = []
    live = {555, 666}

    def fake_kill(pid, sig):
        if sig == 0:
            if pid in live:
                return
            raise ProcessLookupError()
        if sig == 15:
            terms.append(pid)
            live.discard(pid)
            return
        if sig == 9:
            live.discard(pid)
            return
        return

    # 555 is claimed by a valid backend.lock.json; 666 is not.
    lock_owned = {555}

    with (
        patch(
            "hermes_cli.dashboard_procs._scan_dashboard_processes",
            return_value=scanned,
        ),
        patch(
            "hermes_cli.dashboard_procs._process_ppid",
            side_effect=lambda pid: ppids.get(pid),
        ),
        patch("os.kill", side_effect=fake_kill),
        patch("gateway.status.get_process_start_time", side_effect=lambda pid: 42 if pid in live else None),
    ):
        os.environ.pop("HERMES_DESKTOP_CHILD_PID", None)
        result = _reap_orphaned_desktop_local_serves(
            sleep_fn=lambda _s: None,
            signal_term=15,
            signal_kill=9,
            lock_owned_pids_fn=lambda: lock_owned,
            process_age_seconds_fn=lambda _pid: 600.0,
        )

    # Only the genuine orphan (666) is reaped; the lock-owned remote (555) lives.
    assert set(result["matched"]) == {666}
    assert set(terms) == {666}
    assert 555 not in terms
    assert set(result["killed"]) == {666}


def test_reap_spares_young_backend_until_desktop_can_write_lock():
    """A concurrently-starting sibling has no lock yet but is not an orphan."""
    scanned = [(777, "hermes serve --isolated --host 127.0.0.1 --port 0")]
    terms: list[int] = []

    def fake_kill(pid, sig):
        if sig == 0:
            return
        if sig == 15:
            terms.append(pid)
        return

    with (
        patch(
            "hermes_cli.dashboard_procs._scan_dashboard_processes",
            return_value=scanned,
        ),
        patch("hermes_cli.dashboard_procs._process_ppid", return_value=1),
        patch("os.kill", side_effect=fake_kill),
    ):
        result = _reap_orphaned_desktop_local_serves(
            sleep_fn=lambda _s: None,
            signal_term=15,
            signal_kill=9,
            process_age_seconds_fn=lambda _pid: 2.0,
        )

    assert result["matched"] == []
    assert result["killed"] == []
    assert terms == []


def test_reap_spares_backend_when_process_age_is_unknown():
    scanned = [(778, "hermes serve --isolated --host 127.0.0.1 --port 0")]
    terms: list[int] = []

    def fake_age(_pid):
        raise RuntimeError("process disappeared during age probe")

    with (
        patch(
            "hermes_cli.dashboard_procs._scan_dashboard_processes",
            return_value=scanned,
        ),
        patch("hermes_cli.dashboard_procs._process_ppid", return_value=1),
        patch("os.kill", side_effect=lambda pid, sig: terms.append(pid) if sig == 15 else None),
    ):
        result = _reap_orphaned_desktop_local_serves(
            sleep_fn=lambda _s: None,
            signal_term=15,
            signal_kill=9,
            process_age_seconds_fn=fake_age,
        )

    assert result["matched"] == []
    assert result["killed"] == []
    assert terms == []


def test_reap_age_boundary_makes_180_second_orphan_eligible():
    scanned = [
        (779, "hermes serve --isolated --host 127.0.0.1 --port 0"),
        (780, "hermes serve --isolated --host 127.0.0.1 --port 0"),
    ]
    terms: list[int] = []
    live = {779, 780}

    def fake_kill(pid, sig):
        if sig == 0:
            if pid in live:
                return
            raise ProcessLookupError()
        if sig == 15:
            terms.append(pid)
            live.discard(pid)
        return

    ages = {779: 179.999, 780: 180.0}
    with (
        patch(
            "hermes_cli.dashboard_procs._scan_dashboard_processes",
            return_value=scanned,
        ),
        patch("hermes_cli.dashboard_procs._process_ppid", return_value=1),
        patch("os.kill", side_effect=fake_kill),
    ):
        result = _reap_orphaned_desktop_local_serves(
            sleep_fn=lambda _s: None,
            signal_term=15,
            signal_kill=9,
            process_age_seconds_fn=lambda pid: ages[pid],
        )

    assert result["matched"] == [780]
    assert result["killed"] == [780]
    assert terms == [780]


def test_reap_spare_lock_owned_backend_even_without_exclude_match(tmp_path):
    """End-to-end through the real lock scanner: a backend.lock.json on disk
    spares a matching orphaned serve even when HERMES_DESKTOP_CHILD_PID is
    unset (foreign-client backend, not ours by env either)."""
    oid = "a" * 32
    nonce = "b" * 16
    lock_root = tmp_path / "desktop-ssh"
    (lock_root / oid).mkdir(parents=True)
    (lock_root / oid / "backend.lock.json").write_text(
        json.dumps(_valid_lock_payload(4242, oid, nonce))
    )

    scanned = [(4242, "hermes serve --host 127.0.0.1 --port 0")]
    terms: list[int] = []

    def fake_kill(pid, sig):
        if sig == 15:
            terms.append(pid)

    with (
        patch(
            "hermes_cli.dashboard_procs._scan_dashboard_processes",
            return_value=scanned,
        ),
        patch(
            "hermes_cli.dashboard_procs._process_ppid",
            return_value=1,
        ),
        patch("os.kill", side_effect=fake_kill),
    ):
        os.environ.pop("HERMES_DESKTOP_CHILD_PID", None)
        result = _reap_orphaned_desktop_local_serves(
            sleep_fn=lambda _s: None,
            signal_term=15,
            signal_kill=9,
            lock_owned_pids_fn=lambda: _lock_owned_serve_pids(base_dir=lock_root),
        )

    assert terms == []
    assert result["matched"] == []


def test_reap_kills_descendants_of_killed_roots_but_spares_a_failed_roots_subtree():
    """#112631: a SIGKILLed backend never ran PTY_REGISTRY.close_all(), so the reaper sweeps its
    surviving hosted-TUI children — but only for roots it actually killed. A root whose own kill
    raised (EPERM: not ours) keeps its subtree intact instead of being orphaned half-way."""
    scanned = [
        (111, "hermes serve --host 127.0.0.1 --port 0"),  # ours: killed, child 1111 swept
        (444, "hermes serve --host 127.0.0.1 --port 0"),  # not ours: EPERM, child 4444 spared
    ]
    descendants = {1111: (111, 5.0), 4444: (444, 6.0)}
    start_times = {111: 4.0, 444: 4.0, 1111: 5.0, 4444: 6.0}
    sent: list[tuple[int, int]] = []

    def fake_kill(pid, sig):
        if pid == 444:
            raise PermissionError("Operation not permitted")
        sent.append((pid, sig))

    with (
        patch("hermes_cli.dashboard_procs._scan_dashboard_processes", return_value=scanned),
        patch("hermes_cli.dashboard_procs._process_ppid", return_value=1),
        patch("hermes_cli.dashboard_procs._posix_descendants", return_value=descendants) as snap,
        patch("gateway.status.get_process_start_time", side_effect=start_times.get),
        patch("psutil.pid_exists", return_value=True),
        patch("os.kill", side_effect=fake_kill),
    ):
        os.environ.pop("HERMES_DESKTOP_CHILD_PID", None)
        result = _reap_orphaned_desktop_local_serves(
            sleep_fn=lambda _s: None, signal_term=15, signal_kill=9,
            process_age_seconds_fn=lambda _pid: 600.0,
        )

    snap.assert_called_once_with([111, 444])  # snapshotted before the kill reparents them
    assert result["killed"] == [111] and result["failed"] == [444]
    assert (1111, 9) in sent, "surviving child of the killed backend was not swept"
    assert not any(pid == 4444 for pid, _ in sent), "child of a root we could not kill was swept"
