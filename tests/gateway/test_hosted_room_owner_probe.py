"""The strict owner-incarnation probe: only verified cessation may answer ``ended``.

Every other outcome is ``unknown``, which authorizes nothing. These are pure in-process checks:
no process is started, signalled or killed, and the only live pid used is this test's own.
"""

from __future__ import annotations

import os

import pytest

from gateway import hosted_room_owner_probe as probe


pytest.importorskip("psutil")


@pytest.fixture
def domain():
    captured = probe.capture_local_domain()
    if captured is None:
        pytest.skip("this host cannot supply the Linux descriptor domain")
    return captured


def test_capture_describes_this_running_incarnation(domain):
    assert domain["pid"] == os.getpid()
    assert domain["descriptor_version"] == probe.DESCRIPTOR_VERSION
    assert domain["boot_id"] and domain["pid_ns"] and domain["host"]
    assert domain["process_start"] > 0


def test_this_process_is_alive(domain):
    assert probe.probe_owner_incarnation(domain) == probe.ALIVE


def test_a_verified_start_time_difference_is_a_reused_pid_and_therefore_ended(domain):
    """Positive evidence, not a refusal: the recorded incarnation is provably gone."""
    reused = {**domain, "process_start": domain["process_start"] - 5.0}

    assert probe.probe_owner_incarnation(reused) == probe.ENDED


def test_an_absent_process_is_ended(domain):
    import psutil

    free_pid = next(
        pid for pid in range(4_000_000, 4_100_000) if not psutil.pid_exists(pid))

    assert probe.probe_owner_incarnation(
        {**domain, "process_start": 1.0, "pid": free_pid}) == probe.ENDED


@pytest.mark.parametrize(
    "override, why",
    [
        ({"boot_id": "0000-different-boot"}, "a database copied across a reboot"),
        ({"pid_ns": "pid:[999999]"}, "a descriptor from another pid namespace"),
        ({"descriptor_version": 999}, "an unknown descriptor version"),
        ({"descriptor_version": True}, "a bool is not version 1 even though True == 1"),
        ({"descriptor_version": 1.0}, "a float is not version 1 even though 1.0 == 1"),
        ({"pid": True}, "a bool is not a pid even though True == 1"),
        ({"pid": 0}, "pid 0 is not a process"),
        ({"pid": -1}, "a negative pid"),
        ({"process_start": float("inf")}, "an infinite start time would differ from every real one"),
        ({"process_start": float("-inf")}, "a negative infinity start time"),
        ({"process_start": float("nan")}, "NaN is not a start time"),
        ({"process_start": 0}, "a zero start time"),
        ({"process_start": -1.0}, "a negative start time"),
        ({"process_start": True}, "a bool is not a start time"),
        ({"process_start": "yesterday"}, "a non-numeric start time"),
        ({"boot_id": None}, "a missing boot id"),
        ({"pid_ns": 12}, "a non-string pid namespace"),
    ])
def test_a_descriptor_outside_the_validated_domain_is_unknown(domain, override, why):
    assert probe.probe_owner_incarnation({**domain, **override}) == probe.UNKNOWN, why


@pytest.mark.parametrize("value", [None, {}, [], "descriptor", 3])
def test_a_missing_or_malformed_descriptor_is_unknown(value):
    """Legacy NULL rows and corrupt values authorize nothing."""
    assert probe.probe_owner_incarnation(value) == probe.UNKNOWN


def test_an_unreadable_process_is_unknown_not_ended(domain, monkeypatch):
    """AccessDenied and unexpected errors are refusals: they have not shown anything ended."""
    import psutil

    class Denied:
        def __init__(self, pid):
            raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", Denied)
    assert probe.probe_owner_incarnation(domain) == probe.UNKNOWN

    class Broken:
        def __init__(self, pid):
            raise OSError("an unexpected /proc read failure")

    monkeypatch.setattr(psutil, "Process", Broken)
    assert probe.probe_owner_incarnation(domain) == probe.UNKNOWN


def test_a_zombie_is_ended(domain, monkeypatch):
    import psutil

    class Zombie:
        def __init__(self, pid):
            pass

        def status(self):
            return psutil.STATUS_ZOMBIE

    monkeypatch.setattr(psutil, "Process", Zombie)
    assert probe.probe_owner_incarnation(domain) == probe.ENDED


def test_native_descriptor_validation_never_returns_a_partial_map(domain):
    complete = {
        **domain, "home": "/tmp/home", "profile": "ops",
        "runtime_session_id": "runtime-1", "stored_session_key": "stored-1"}

    assert probe.validate_native_descriptor(complete) == complete
    for field in ("home", "profile", "runtime_session_id", "stored_session_key"):
        assert probe.validate_native_descriptor({**complete, field: ""}) is None, field
        assert probe.validate_native_descriptor(
            {k: v for k, v in complete.items() if k != field}) is None, field
    assert probe.validate_native_descriptor({**complete, "boot_id": "other"}) is None
    assert probe.validate_native_descriptor({**complete, "host": 7}) is None
    assert probe.validate_native_descriptor(None) is None
