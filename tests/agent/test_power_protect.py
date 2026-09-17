"""Crash/recovery invariants for macOS Power Protect ownership."""

import json

import pytest

from agent.power_protect import (
    PowerProtectError,
    PowerProtectLease,
    recover_stale_power_protect,
)


def test_last_owner_restores_previous_value(tmp_path):
    sleep = [0]
    writes = []

    def read():
        return sleep[0]

    def set_sleep(value):
        writes.append(value)
        sleep[0] = value

    first = PowerProtectLease(read, set_sleep, root=tmp_path)
    second = PowerProtectLease(read, set_sleep, root=tmp_path)
    second._pid = first._pid + 100000
    second._start_time = None
    first.acquire()
    second.acquire()
    assert writes == [1]

    first.release()
    assert sleep == [1]
    second.release()
    assert sleep == [0]
    assert writes == [1, 0]


def test_release_preserves_a_user_change_during_the_lease(tmp_path):
    sleep = [0]
    writes = []

    def set_sleep(value):
        writes.append(value)
        sleep[0] = value

    lease = PowerProtectLease(lambda: sleep[0], set_sleep, root=tmp_path)
    lease.acquire()
    sleep[0] = 0  # The user changed the setting while Hermes was running.
    lease.release()

    assert writes == [1]
    assert sleep == [0]
    assert not (tmp_path / "stay-awake-power.json").exists()


def test_failed_enable_removes_pending_journal(tmp_path):
    def fail_enable(_value):
        raise RuntimeError("pmset failed")

    lease = PowerProtectLease(lambda: 0, fail_enable, root=tmp_path)
    with pytest.raises(RuntimeError, match="pmset failed"):
        lease.acquire()

    assert not (tmp_path / "stay-awake-power.json").exists()


def test_recover_stale_owner_restores_previous_value(tmp_path):
    state_path = tmp_path / "stay-awake-power.json"
    state_path.write_text(
        json.dumps({
            "version": 1,
            "previous_sleep_disabled": 0,
            "owners": {"999999": {"pid": 999999, "start_time": None}},
        }),
        encoding="utf-8",
    )
    sleep = [1]
    writes = []

    def set_recovered(value):
        writes.append(value)
        sleep[0] = value

    recovered = recover_stale_power_protect(
        lambda: sleep[0],
        set_recovered,
        root=tmp_path,
    )

    assert recovered is True
    assert writes == [0]
    assert sleep == [0]
    assert not state_path.exists()


def test_unknown_sleep_disabled_is_not_touched_without_hermes_state(tmp_path):
    writes = []
    assert recover_stale_power_protect(lambda: 1, writes.append, root=tmp_path) is False
    assert writes == []


def test_malformed_owner_fails_closed(tmp_path):
    (tmp_path / "stay-awake-power.json").write_text(
        json.dumps({
            "version": 1,
            "previous_sleep_disabled": 0,
            "owners": {"bad": {"pid": "not-a-pid", "start_time": None}},
        }),
        encoding="utf-8",
    )
    with pytest.raises(PowerProtectError):
        recover_stale_power_protect(lambda: 1, lambda _value: None, root=tmp_path)


def test_corrupt_state_fails_closed(tmp_path):
    (tmp_path / "stay-awake-power.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(PowerProtectError):
        PowerProtectLease(lambda: 0, lambda _value: None, root=tmp_path).acquire()
