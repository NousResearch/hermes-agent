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


@pytest.mark.parametrize("legacy", [True, False], ids=["v1-uncertain", "v2-reuse"])
def test_only_boot_relative_start_difference_proves_pid_reuse(domain, legacy):
    reused = dict(domain)
    if legacy:
        reused["descriptor_version"] = 1
        reused.pop("process_start_ticks")
        reused["process_start"] -= 5.0
    else:
        reused["process_start_ticks"] += 1

    assert probe.probe_owner_incarnation(reused) == (probe.UNKNOWN if legacy else probe.ENDED)


@pytest.mark.parametrize("legacy", [True, False], ids=["v1", "v2"])
def test_an_absent_process_is_ended(domain, legacy):
    import psutil

    free_pid = next(
        pid for pid in range(4_000_000, 4_100_000) if not psutil.pid_exists(pid))

    descriptor = {**domain, "process_start": 1.0, "pid": free_pid}
    if legacy:
        descriptor["descriptor_version"] = 1
        descriptor.pop("process_start_ticks")
    assert probe.probe_owner_incarnation(descriptor) == probe.ENDED


@pytest.mark.parametrize(
    "override, why",
    [
        ({"boot_id": "0000-different-boot"}, "a database copied across a reboot"),
        ({"pid_ns": "pid:[999999]"}, "a descriptor from another pid namespace"),
        ({"descriptor_version": 999}, "an unknown descriptor version"),
        ({"descriptor_version": True}, "a bool is not version 1 even though True == 1"),
        ({"descriptor_version": 1.0}, "a float is not version 1 even though 1.0 == 1"),
        ({"descriptor_version": 2.0}, "a float is not version 2"),
        ({"descriptor_version": 1}, "a v1 descriptor cannot carry a v2 tick stamp"),
        ({"process_start_ticks": None}, "missing boot-relative evidence"),
        ({"process_start_ticks": True}, "a bool is not a tick stamp"),
        ({"process_start_ticks": 1.0}, "a float is not a tick stamp"),
        ({"process_start_ticks": "1"}, "a string is not a tick stamp"),
        ({"process_start_ticks": 0}, "a zero tick stamp"),
        ({"process_start_ticks": -1}, "a negative tick stamp"),
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


@pytest.mark.linux_only
@pytest.mark.parametrize("failure", ["denied", "readonly", "missing", "malformed"])
def test_unreadable_kernel_ticks_never_authorize_death(domain, monkeypatch, failure):
    import errno
    import io
    import hermes_state_common

    def unreadable(path, mode):
        assert path == f"/proc/{os.getpid()}/stat" and mode == "rb"
        if failure == "malformed":
            return io.BytesIO(b"unparseable stat")
        code = {"denied": errno.EACCES, "readonly": errno.EROFS, "missing": errno.ENOENT}[failure]
        raise OSError(code, "simulated kernel read failure")

    # Affect only the reused helper's read, not psutil's independent live-process observation.
    monkeypatch.setattr(hermes_state_common, "open", unreadable, raising=False)
    assert hermes_state_common._proc_start_ticks(os.getpid()) is None
    assert probe.probe_owner_incarnation(domain) == probe.UNKNOWN
    assert probe.capture_local_domain() is None


def test_a_zombie_is_ended(domain, monkeypatch):
    import psutil

    class Zombie:
        def __init__(self, pid):
            pass

        def status(self):
            return psutil.STATUS_ZOMBIE

    monkeypatch.setattr(psutil, "Process", Zombie)
    assert probe.probe_owner_incarnation(domain) == probe.ENDED


@pytest.mark.parametrize("legacy", [True, False], ids=["v1", "v2"])
def test_native_descriptor_validation_never_returns_a_partial_map(domain, legacy):
    complete = {
        **domain, "home": "/tmp/home", "profile": "ops",
        "runtime_session_id": "runtime-1", "stored_session_key": "stored-1"}
    if legacy:
        complete["descriptor_version"] = 1
        complete.pop("process_start_ticks")

    assert probe.validate_native_descriptor(complete) == complete
    for field in ("home", "profile", "runtime_session_id", "stored_session_key"):
        assert probe.validate_native_descriptor({**complete, field: ""}) is None, field
        assert probe.validate_native_descriptor(
            {k: v for k, v in complete.items() if k != field}) is None, field
    assert probe.validate_native_descriptor({**complete, "boot_id": "other"}) is None
    assert probe.validate_native_descriptor({**complete, "host": 7}) is None
    assert probe.validate_native_descriptor(None) is None
