"""A host record that cannot prove its own incarnation must not vouch for a recycled pid.

Observed failure (2026-10): a gateway died without cleaning up its host record; the record carried
no createTime and no startTime (written by a Hermes from before those fields existed). The dead
gateway's pid was later taken over by an unrelated process (a monitoring daemon). Because
``_same_incarnation`` treats a missing create_time as a match, the liveness probe proved the
recycled pid "alive", every new gateway dialed a control socket nobody owned, refused to start
("A gateway already owns this host and has not published its served set yet"), and the supervisor
loop made 223 restart attempts over two hours — with the cron scheduler dead the whole time.

The fix bounds the "owner exists, served set unknown" verdict for unfalsifiable records: once the
boot-race grace on the record has expired with the socket still silent, the owner is treated as
gone and the caller may take the host over. Records WITH incarnation proof keep the old behaviour
(a live-but-wedged owner holds the host lock and cannot be replaced anyway), and a fresh
unprovable record keeps the boot-race verdict (the record lands a moment before the socket binds).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

from gateway import host_attach
from gateway.host_attach import HOST_RECORD_BOOT_GRACE_S, HostGateway, host_gateway
from gateway.host_rendezvous import HostRecord


def _unprovable_record(*, age_s: float, pid: int | None = None) -> HostRecord:
    """A record with NO process-incarnation proof, ``age_s`` old — the legacy format's shape."""
    return HostRecord(
        role="gateway",
        pid=pid if pid is not None else os.getpid(),
        create_time=None,
        start_time=None,
        host="",
        port=None,
        protocol_version=1,
        token_fingerprint="",
        profiles=(),
        updated_at=(datetime.now(UTC) - timedelta(seconds=age_s)).isoformat(),
        home="/fake/home",
    )


def _probe(monkeypatch, record: HostRecord, *, answers: dict | None = None) -> HostGateway | None:
    """Run one probe against ``record``; the control socket answers ``answers`` or nothing."""
    import gateway.host_rendezvous as hr

    monkeypatch.setattr(hr, "read_record", lambda role: record)
    monkeypatch.setattr(host_attach, "_identify", lambda home: answers)
    host_attach.invalidate_host_gateway_cache()
    try:
        return host_attach._probe_host_gateway(wait_for_channel=0.0)
    finally:
        host_attach.invalidate_host_gateway_cache()


def test_recycled_pid_behind_unprovable_record_is_treated_as_gone(monkeypatch):
    """The outage: an old record with no incarnation proof + a silent socket = no owner.

    The pid is alive (this test process), which is exactly what fooled the liveness probe when the
    dead gateway's pid had been recycled by an unrelated daemon.
    """
    record = _unprovable_record(age_s=HOST_RECORD_BOOT_GRACE_S * 100)
    assert _probe(monkeypatch, record) is None


def test_unreadable_record_age_is_treated_as_stale(monkeypatch):
    """A record whose age cannot be parsed must not vouch for itself forever."""
    record = _unprovable_record(age_s=0)
    record = HostRecord(**{**record.__dict__, "updated_at": "not-a-timestamp"})
    assert _probe(monkeypatch, record) is None


def test_fresh_unprovable_record_keeps_the_boot_race_verdict(monkeypatch):
    """A record published seconds ago may simply predate the socket bind: still transient."""
    record = _unprovable_record(age_s=1.0)
    gateway = _probe(monkeypatch, record)
    assert gateway is not None
    assert gateway.served_known is False
    assert gateway.pid == record.pid


def test_record_with_incarnation_proof_keeps_the_old_verdict(monkeypatch):
    """A provable record (live owner, silent socket) stays a transient verdict: a wedged owner

    holds the host lock, so a takeover cannot succeed anyway — the supervisor must keep retrying.
    """
    import psutil

    record = HostRecord(
        role="gateway",
        pid=os.getpid(),
        create_time=psutil.Process(os.getpid()).create_time(),
        start_time=None,
        host="",
        port=None,
        protocol_version=1,
        token_fingerprint="",
        profiles=(),
        updated_at=(datetime.now(UTC) - timedelta(hours=2)).isoformat(),
        home="/fake/home",
    )
    gateway = _probe(monkeypatch, record)
    assert gateway is not None
    assert gateway.served_known is False


def test_answering_socket_still_attaches(monkeypatch):
    """The happy path is untouched: a live owner that answers keeps its served set."""
    record = _unprovable_record(age_s=HOST_RECORD_BOOT_GRACE_S * 100)
    identity = {"pid": record.pid, "hermes_home": "/fake/home", "served_profiles": ["default"]}
    gateway = _probe(monkeypatch, record, answers=identity)
    assert gateway is not None
    assert gateway.served_known is True
    assert gateway.serves("default")


def test_decide_starts_instead_of_refusing_forever(monkeypatch):
    """The user-visible outcome: decide() returns START (takeover), not the REFUSE loop.

    The recorded pid must be a live process OTHER than this one: decide() starts outright when the
    record names the caller itself, which would make this pass on the unfixed code too.
    """
    from pathlib import Path

    from gateway.host_attach import START, decide

    record = _unprovable_record(age_s=HOST_RECORD_BOOT_GRACE_S * 100, pid=os.getppid())
    import gateway.host_rendezvous as hr

    monkeypatch.setattr(hr, "read_record", lambda role: record)
    monkeypatch.setattr(host_attach, "_identify", lambda home: None)
    host_attach.invalidate_host_gateway_cache()
    try:
        decision = decide(Path("/fake/home"))
    finally:
        host_attach.invalidate_host_gateway_cache()
    assert decision.outcome == START
