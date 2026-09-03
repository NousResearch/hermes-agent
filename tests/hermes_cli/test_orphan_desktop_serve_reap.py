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
            return None
        if sig == 15:
            terms.append(pid)
        return None

    with (
        patch(
            "hermes_cli.dashboard_procs._scan_dashboard_processes",
            return_value=scanned,
        ),
        patch("hermes_cli.dashboard_procs._process_ppid", return_value=1),
        patch("os.kill", side_effect=fake_kill),
        patch("sys.platform", "darwin"),
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
        patch("sys.platform", "darwin"),
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
                return None
            raise ProcessLookupError()
        if sig == 15:
            terms.append(pid)
            live.discard(pid)
        return None

    ages = {779: 179.999, 780: 180.0}
    with (
        patch(
            "hermes_cli.dashboard_procs._scan_dashboard_processes",
            return_value=scanned,
        ),
        patch("hermes_cli.dashboard_procs._process_ppid", return_value=1),
        patch("os.kill", side_effect=fake_kill),
        patch("sys.platform", "darwin"),
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


# ── #101626: a lock claim is not a heartbeat ────────────────────────────────
# `backend.lock.json` records that a Desktop client once claimed a PID. Only
# that client retires the claim, and a client whose machine stops executing
# (laptop asleep, waking for Power Nap and sleeping again) never does. The
# claim then grants its SSH-isolated backend permanent immunity while the
# backend keeps writing the HERMES_HOME the next backend is about to open.


def _reap_with_claims(scanned, lock_owned, claim_disproved, terms):
    live = {pid for pid, _ in scanned}

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
        patch("hermes_cli.dashboard_procs._process_ppid", return_value=1),
        patch("os.kill", side_effect=fake_kill),
        patch("sys.platform", "darwin"),
    ):
        os.environ.pop("HERMES_DESKTOP_CHILD_PID", None)
        return _reap_orphaned_desktop_local_serves(
            sleep_fn=lambda _s: None,
            signal_term=15,
            signal_kill=9,
            lock_owned_pids_fn=lambda: set(lock_owned),
            process_age_seconds_fn=lambda _pid: 600.0,
            claim_disproved_fn=claim_disproved,
        )


class _FakeConn:
    def __init__(self, status):
        self.status = status


class _FakeProcess:
    """Minimal psutil.Process stand-in for the claim-disproof probes."""

    def __init__(self, *, ppid, age, cmdline, statuses):
        import time as _time

        self._ppid = ppid
        self._create_time = _time.time() - age
        self._cmdline = cmdline
        self._statuses = statuses

    def ppid(self):
        return self._ppid

    def create_time(self):
        return self._create_time

    def cmdline(self):
        return self._cmdline

    def net_connections(self, kind="inet"):
        return [_FakeConn(status) for status in self._statuses]


SSH_SERVE_CMD = (
    "hermes serve --isolated --host 127.0.0.1 --port 0 "
    "--ssh-session-token-file /home/h/.hermes/desktop-ssh/a/b.token "
    "--ssh-owner-nonce 0123456789abcdef"
)


def test_reap_withdraws_immunity_from_a_disproved_claim():
    """A sleeping laptop's claim keeps a second writer alive on our database."""
    terms: list[int] = []
    result = _reap_with_claims(
        scanned=[(555, SSH_SERVE_CMD)],
        lock_owned={555},
        claim_disproved=lambda pid: pid == 555,
        terms=terms,
    )

    assert set(result["matched"]) == {555}
    assert set(terms) == {555}


def test_reap_keeps_immunity_when_the_claim_is_not_disproved():
    """Indeterminate or live: the claim stands, exactly as before (#78872)."""
    terms: list[int] = []
    result = _reap_with_claims(
        scanned=[(555, SSH_SERVE_CMD)],
        lock_owned={555},
        claim_disproved=lambda _pid: False,
        terms=terms,
    )

    assert result["matched"] == []
    assert terms == []


def test_reap_never_widens_when_the_claim_probe_raises():
    """A probe that blows up must not be read as 'claim disproved'."""

    def boom(_pid):
        raise RuntimeError("psutil exploded")

    terms: list[int] = []
    result = _reap_with_claims(
        scanned=[(555, SSH_SERVE_CMD)],
        lock_owned={555},
        claim_disproved=boom,
        terms=terms,
    )

    assert result["matched"] == []
    assert terms == []


def test_claim_disproved_requires_the_same_hermes_home():
    """#94032 inverse: a backend on another database is never a candidate."""
    from hermes_cli.dashboard_procs import _lock_claim_is_disproved

    with (
        patch("sys.platform", "linux"),
        patch(
            "hermes_cli.dashboard_procs._hermes_home_for_pid",
            return_value="/home/h/.hermes/profiles/other",
        ),
    ):
        assert _lock_claim_is_disproved(555, self_home="/home/h/.hermes") is False


def test_claim_disproved_is_false_when_the_home_is_unreadable():
    from hermes_cli.dashboard_procs import _lock_claim_is_disproved

    with (
        patch("sys.platform", "linux"),
        patch("hermes_cli.dashboard_procs._hermes_home_for_pid", return_value=None),
    ):
        assert _lock_claim_is_disproved(555, self_home="/home/h/.hermes") is False


def test_claim_disproved_spares_a_backend_with_an_established_client(tmp_path):
    """A tunnel still carrying a client keeps its immunity."""
    from hermes_cli.dashboard_procs import _lock_claim_is_disproved

    home = str(tmp_path)
    proc = _FakeProcess(
        ppid=1,
        age=6000.0,
        cmdline=["hermes", "serve", "--isolated", "--host", "127.0.0.1", "--port", "0"],
        statuses=["LISTEN", "ESTABLISHED"],
    )

    with (
        patch("sys.platform", "linux"),
        patch("hermes_cli.dashboard_procs._hermes_home_for_pid", return_value=home),
        patch("psutil.Process", return_value=proc),
    ):
        assert _lock_claim_is_disproved(555, self_home=home) is False


def test_claim_disproved_for_a_clientless_orphan_on_our_home(tmp_path):
    """The #101626 shape: ppid 1, old, ephemeral, no client, our database."""
    from hermes_cli.dashboard_procs import _lock_claim_is_disproved

    home = str(tmp_path)
    proc = _FakeProcess(
        ppid=1,
        age=6000.0,
        cmdline=["hermes", "serve", "--isolated", "--host", "127.0.0.1", "--port", "0"],
        statuses=["LISTEN"],
    )

    with (
        patch("sys.platform", "linux"),
        patch("hermes_cli.dashboard_procs._hermes_home_for_pid", return_value=home),
        patch("psutil.Process", return_value=proc),
    ):
        assert _lock_claim_is_disproved(555, self_home=home) is True


def test_claim_disproved_spares_a_fixed_port_operator_serve(tmp_path):
    """An operator-managed `--port 9119` serve is not a Desktop ephemeral."""
    from hermes_cli.dashboard_procs import _lock_claim_is_disproved

    home = str(tmp_path)
    proc = _FakeProcess(
        ppid=1,
        age=6000.0,
        cmdline=["hermes", "serve", "--host", "127.0.0.1", "--port", "9119"],
        statuses=["LISTEN"],
    )

    with (
        patch("sys.platform", "linux"),
        patch("hermes_cli.dashboard_procs._hermes_home_for_pid", return_value=home),
        patch("psutil.Process", return_value=proc),
    ):
        assert _lock_claim_is_disproved(555, self_home=home) is False
