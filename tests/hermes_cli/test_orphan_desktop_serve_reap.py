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
    assert not _is_desktop_local_serve_cmdline(
        "vim notes about hermes serve --port 0"
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
                return None
            raise ProcessLookupError()
        if sig == 15:
            terms.append(pid)
            live.discard(pid)
            return None
        if sig == 9:
            live.discard(pid)
            return None
        return None

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
        patch("sys.platform", "darwin"),
    ):
        os.environ.pop("HERMES_DESKTOP_CHILD_PID", None)
        result = _reap_orphaned_desktop_local_serves(
            sleep_fn=lambda _s: None,
            signal_term=15,
            signal_kill=9,
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
        patch("sys.platform", "darwin"),
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
    bad_log["logPath"] = "~/.hermes/desktop-ssh/{oid}/other.log".format(oid=oid)
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
                return None
            raise ProcessLookupError()
        if sig == 15:
            terms.append(pid)
            live.discard(pid)
            return None
        if sig == 9:
            live.discard(pid)
            return None
        return None

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
        patch("sys.platform", "darwin"),
    ):
        os.environ.pop("HERMES_DESKTOP_CHILD_PID", None)
        result = _reap_orphaned_desktop_local_serves(
            sleep_fn=lambda _s: None,
            signal_term=15,
            signal_kill=9,
            lock_owned_pids_fn=lambda: lock_owned,
        )

    # Only the genuine orphan (666) is reaped; the lock-owned remote (555) lives.
    assert set(result["matched"]) == {666}
    assert set(terms) == {666}
    assert 555 not in terms
    assert set(result["killed"]) == {666}


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
        return None

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
        patch("sys.platform", "darwin"),
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
