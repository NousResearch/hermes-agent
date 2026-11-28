"""A gateway.pid left over from before a reboot must not count as a live gateway.

``start_time`` is clock ticks since boot, so after a reboot an unrelated process (headless Chrome on
a forge-help VM) can hold the recorded PID. Records carry the boot id; a record from another boot,
or a legacy record with no identity whose PID now runs a non-gateway command, is stale.
"""
import json
import os

from gateway import status


def _record(**overrides):
    record = {"pid": os.getpid(), "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
              "start_time": status._get_process_start_time(os.getpid()), "boot_id": "boot-A"}
    record.update(overrides)
    return record


def test_pid_record_carries_boot_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(status, "_live_boot_id", lambda: "boot-A")
    status.write_pid_file()
    assert json.loads((tmp_path / "gateway.pid").read_text())["boot_id"] == "boot-A"


def test_record_from_earlier_boot_is_stale_even_when_pid_and_start_time_match(monkeypatch):
    monkeypatch.setattr(status, "_live_boot_id", lambda: "boot-B")  # rebooted since boot-A
    assert status._live_pid_from_record(_record()) is None


def test_record_from_current_boot_is_live(monkeypatch):
    monkeypatch.setattr(status, "_live_boot_id", lambda: "boot-A")
    assert status._live_pid_from_record(_record()) == os.getpid()


def test_legacy_record_naming_a_non_gateway_process_is_stale(monkeypatch):
    monkeypatch.setattr(status, "_live_boot_id", lambda: "boot-B")
    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: "chrome --headless")
    assert status._live_pid_from_record({"pid": os.getpid()}) is None


def test_legacy_record_naming_a_gateway_stays_live(monkeypatch):
    monkeypatch.setattr(status, "_live_boot_id", lambda: "boot-B")
    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: "python -m hermes_cli.main gateway run")
    assert status._live_pid_from_record({"pid": os.getpid()}) == os.getpid()


def test_get_running_pid_clears_pid_file_from_before_reboot(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(status, "_live_boot_id", lambda: "boot-B")
    pid_path = tmp_path / "gateway.pid"
    pid_path.write_text(json.dumps(_record()))
    assert status.get_running_pid(pid_path) is None
    assert not pid_path.exists()


def test_scoped_lock_record_carries_boot_id_and_pre_reboot_lock_is_stale(monkeypatch):
    # acquire_scoped_lock writes ``{**_build_pid_record(), "scope": ...}``; stale-lock recovery and
    # runtime_ownership judge it through _live_pid_from_record.
    monkeypatch.setattr(status, "_live_boot_id", lambda: "boot-A")
    record = {**status._build_pid_record(), "scope": "telegram-bot-token"}
    assert record["boot_id"] == "boot-A"
    assert status._live_pid_from_record(record) == os.getpid()
    monkeypatch.setattr(status, "_live_boot_id", lambda: "boot-B")  # rebooted
    assert status._live_pid_from_record(record) is None


def test_gateway_runtime_lock_record_from_before_reboot_is_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(status, "_live_boot_id", lambda: "boot-B")
    lock_path = tmp_path / "gateway.lock"
    lock_path.write_text(json.dumps(_record()))
    assert status._live_pid_from_record(status._read_gateway_lock_record(lock_path)) is None
