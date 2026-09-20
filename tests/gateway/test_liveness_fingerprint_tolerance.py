"""Estate-wide liveness-fingerprint tolerance (2026-09-19 RCA follow-up).

Same root cause as tests/cron/test_owner_liveness_fingerprint.py: processes on one
host read the same pid's start time with ~1s disagreement, so bit-exact compares in
LIVENESS verdicts declared live owners dead/recycled. Kill-permission gates
(``_start_times_agree`` force-kill, ``pid_is_hermes`` taskkill) stay exact on
purpose: a refused kill is fail-safe.
"""
from __future__ import annotations

import os
from unittest import mock

import gateway.status as G
import gateway.delivery_ledger as L
import hermes_cli.kanban_db_dispatch as K


DRIFT = 100          # centiseconds: the exact incident delta (+1.00s)
FAR = 720_000        # 2h: a recycled pid differs by minutes-to-days


# ---------------------------------------------------------------- gateway.status

def test_start_times_match_unit():
    assert G.start_times_match(1000, 1000)
    assert G.start_times_match(1000, 1000 + DRIFT)
    assert G.start_times_match(1000, 1000 - DRIFT)
    assert not G.start_times_match(1000, 1000 + FAR)
    assert not G.start_times_match(None, 1000)
    assert not G.start_times_match(1000, "junk")


def test_scoped_lock_owner_state_tolerates_drift():
    # Polarity vs pre-fix code: a drifted LIVE owner was declared "exited" → the lock
    # could be taken while the owner still held it.
    with mock.patch.object(G, "_pid_exists", return_value=True), \
         mock.patch.object(G, "_get_process_start_time", return_value=1000):
        assert G._scoped_lock_owner_state(123, 1000 + DRIFT) == "same"
        assert G._scoped_lock_owner_state(123, 1000 - DRIFT) == "same"
        assert G._scoped_lock_owner_state(123, 1000 + FAR) == "exited"   # reuse → exited
    with mock.patch.object(G, "_pid_exists", return_value=True), \
         mock.patch.object(G, "_get_process_start_time", return_value=None):
        assert G._scoped_lock_owner_state(123, 1000) == "unknown"        # unprobeable ≠ dead
    with mock.patch.object(G, "_pid_exists", return_value=False):
        assert G._scoped_lock_owner_state(123, 1000) == "exited"


def test_pid_marker_names_self_tolerates_drift():
    with mock.patch.object(G, "_get_process_start_time", return_value=1000):
        assert G._pid_marker_names_self(os.getpid(), 1000 + DRIFT)
        assert G._pid_marker_names_self(os.getpid(), None)              # unprobeable target
        assert G._pid_marker_names_self(os.getpid() + 1, 1000) is False  # not our pid


# --------------------------------------------------------- gateway.delivery_ledger

def test_owner_alive_tolerates_drift_and_keeps_junk_alive():
    with mock.patch.object(L, "_start_time", return_value=1000):
        assert L._owner_alive(123, 1000 + DRIFT)
        assert not L._owner_alive(123, 1000 + FAR)
        assert L._owner_alive(123, None)     # no fingerprint recorded → alive
        assert L._owner_alive(123, "junk")   # junk recorded value: fail-safe alive (pre-existing)


# ------------------------------------------------------ hermes_cli.kanban_db_dispatch

def test_pid_recycled_integer_row_tolerates_drift():
    # Polarity vs pre-fix code: a drifted LIVE worker was declared recycled → its
    # claim could be released / termination refused while it kept running.
    with mock.patch("gateway.status.get_process_start_time", return_value=1000):
        assert not K._pid_recycled(123, 1000 + DRIFT)          # drifted live worker → not recycled
        assert K._pid_recycled(123, 1000 + FAR)                 # reused pid → recycled
        assert K._pid_recycled(123, 1000) is False               # exact match → not recycled
        assert K._pid_recycled(123, None) is False              # no fingerprint recorded
        assert K._pid_recycled(123, "junk")                     # junk → recycled (pre-existing)


def test_pid_recycled_string_row_tolerates_drift():
    recorded = "1789801702|1000"
    with mock.patch.object(K, "_process_fingerprint", return_value="1789801702|1100"):
        assert not K._pid_recycled(123, recorded)               # drifted live → not recycled
    with mock.patch.object(K, "_process_fingerprint", return_value="1789801702|1000"):
        assert not K._pid_recycled(123, recorded)               # exact → not recycled
    with mock.patch.object(K, "_process_fingerprint", return_value="9999|1000"):
        assert K._pid_recycled(123, recorded)                   # cross-boot epoch → recycled
    with mock.patch.object(K, "_process_fingerprint", return_value="1789801702|9999"):
        assert K._pid_recycled(123, recorded)                   # far start → recycled (reuse)
    with mock.patch.object(K, "_process_fingerprint", return_value=None):
        assert K._pid_recycled(123, recorded)                   # unreadable → stranger-conservative
    assert K._pid_recycled(123, K.UNVERIFIED_WORKER_FINGERPRINT)  # UNVERIFIED marker is foreign → recycled


def test_pid_recycled_far_string_row_is_recycled():
    recorded = "1789801702|1000"
    with mock.patch.object(K, "_process_fingerprint", return_value="1789801702|" + str(1000 + FAR)):
        assert K._pid_recycled(123, recorded)


# ------------------------------------------------ reviewer follow-up: boundary + gap tests

def test_start_times_match_boundary_is_inclusive():
    # ±200 native units inclusive at both edges; ±201 fails (reuse stays detectable).
    assert G.start_times_match(1000, 1200)
    assert G.start_times_match(1000, 800)
    assert not G.start_times_match(1000, 1201)
    assert not G.start_times_match(1000, 799)


def test_pid_marker_names_self_far_polarity():
    # A marker start far from ours must NOT name self (pid-reuse guard intact),
    # even though drift within ±200 does.
    with mock.patch.object(G, "_get_process_start_time", return_value=1000):
        assert G._pid_marker_names_self(os.getpid(), 1000 + DRIFT)
        assert G._pid_marker_names_self(os.getpid(), 1000)          # exact still fine
        assert G._pid_marker_names_self(os.getpid(), 1000 + FAR) is False  # reuse → not self
        assert G._pid_marker_names_self(os.getpid(), "junk") is False     # junk → not self


def test_validated_scoped_lock_gateway_owner_tolerates_drift():
    # Direct cover of the widened corroborating-compare (status.py): the lock
    # record's start_time vs the target home's gateway.pid start_time may drift
    # ±2s between processes on one host; far apart → pid reuse → rejected.
    import gateway.status as G2

    def run(lock_start, pidrec_start):
        lock_record = {
            "pid": 1234,
            "start_time": lock_start,
            "hermes_home": "/tmp/does-not-matter",
            "gateway": True,
        }
        pid_record = {"pid": 1234, "start_time": pidrec_start, "hermes_home": "/tmp/does-not-matter"}
        with mock.patch.object(G2, "_pid_exists", return_value=True), \
             mock.patch.object(G2, "_get_process_start_time", return_value=lock_start), \
             mock.patch.object(G2, "_record_looks_like_gateway", return_value=True), \
             mock.patch.object(G2, "_pid_from_record", return_value=1234), \
             mock.patch.object(G2, "_read_process_cmdline", return_value=None), \
             mock.patch.object(G2, "_read_json_file", return_value=pid_record), \
             mock.patch.object(G2, "_same_hermes_home", return_value=True):
            return G2._validated_scoped_lock_gateway_owner(lock_record)

    assert run(1000 + DRIFT, 1000) is not None   # drifted corroboration → still the owner
    assert run(1000, 1000) is not None            # exact → owner
    assert run(1000 + FAR, 1000) is None          # reuse → rejected (fails closed)
