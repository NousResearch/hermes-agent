"""Kernel OOM-kill evidence in the lifecycle ledger (#148527).

A kernel OOM kill is a SIGKILL, so no exit path runs.  After an unclean death the next boot
compares the OOM-kill counters in the dead life's last heartbeat with the current ones: growth
shows the kernel OOM-killed some process in that scope (the gateway's cgroup, or the host), not
which one.  The /proc and cgroup files are tmp fixtures (Linux text format) wired in through the
path constants and ``cgroup_cleanup``'s ``/proc/self/cgroup`` reader.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, UTC
from pathlib import Path

import pytest

from gateway import cgroup_cleanup, lifecycle_ledger
from gateway.lifecycle_ledger import detect_unclean_exit, get_lifecycle_sentinel_path, record_startup, sample_oom_kills
from gateway.memory_status import collect_memory_status
from gateway.shutdown_watchdog import get_loop_heartbeat_path, write_loop_heartbeat

_DEAD_PID = 2 ** 22 + 12345  # beyond default pid_max on Linux; never alive
_BOOT = "6f1c0d6e-2a7b-4c1e-9d3f-1b2a3c4d5e6f"
# Kernel formats: /proc/vmstat is "key value" per line, memory.events likewise.
_VMSTAT = "nr_free_pages 123456\npgfault 99\noom_kill {n}\nunevictable_pgs_culled 0\n"
_EVENTS = "low 0\nhigh 0\nmax 12\noom 3\noom_kill {n}\noom_group_kill 0\n"
_CGROUP_V2 = "0::/system.slice/hermes-gateway.service\n"
_CGROUP_V1 = "12:memory:/system.slice/hermes-gateway.service\n11:cpu,cpuacct:/system.slice\n"


class _Kernel:
    """Fixture /proc + /sys/fs/cgroup the ledger reads through its path constants."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.vmstat = root / "proc" / "vmstat"
        self.self_cgroup = root / "proc" / "self" / "cgroup"
        self.cgroup_root = root / "sys" / "fs" / "cgroup"
        self.events = self.cgroup_root / "system.slice" / "hermes-gateway.service" / "memory.events"
        self.self_cgroup.parent.mkdir(parents=True)
        self.events.parent.mkdir(parents=True)
        self.boot_id: str | None = _BOOT
        monkeypatch.setattr(lifecycle_ledger, "_VMSTAT_PATH", str(self.vmstat))
        monkeypatch.setattr(lifecycle_ledger, "_CGROUP_ROOT", self.cgroup_root)
        real_path = cgroup_cleanup.Path
        monkeypatch.setattr(cgroup_cleanup, "Path",
                            lambda p: self.self_cgroup if p == "/proc/self/cgroup" else real_path(p))
        monkeypatch.setattr("hermes_cli.process_identity._live_boot_id", lambda: self.boot_id)

    def set(self, *, host: int | None, cgroup: int | None, cgroup_line: str = _CGROUP_V2) -> None:
        if host is not None:
            self.vmstat.write_text(_VMSTAT.format(n=host), encoding="utf-8")
        self.self_cgroup.write_text(cgroup_line, encoding="utf-8")
        lifecycle_ledger._cgroup_v2_events_path.cache_clear()  # a new membership: as if another process
        if cgroup is not None:
            self.events.write_text(_EVENTS.format(n=cgroup), encoding="utf-8")

    def recreate_cgroup(self) -> None:
        """A systemd unit restart: same path, new directory (new inode), counter back at 0."""
        old = self.events.parent
        old.rename(old.with_name("retired"))
        old.mkdir()
        self.events.write_text(_EVENTS.format(n=0), encoding="utf-8")


@pytest.fixture
def kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    lifecycle_ledger._cgroup_v2_events_path.cache_clear()
    yield _Kernel(tmp_path / "kernel", monkeypatch)
    lifecycle_ledger._cgroup_v2_events_path.cache_clear()


def _died_with_heartbeat(home: Path, oom: dict | None, *, heartbeat_pid: int = _DEAD_PID, age_s: float = 5.0,
                         claimed_at: float = 1000.0) -> None:
    """A previous life that was SIGKILLed (phase still running, dead pid) ``age_s`` after its last
    heartbeat sampled ``oom``; ``claimed_at`` is that life's sentinel ``start_time``."""
    sentinel = get_lifecycle_sentinel_path(home)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({"phase": "running", "pid": _DEAD_PID, "start_time": claimed_at,
                                    "started_at": "2026-11-04T08:00:00+00:00"}), encoding="utf-8")
    written = (datetime.now(UTC) - timedelta(seconds=age_s)).isoformat()
    hb: dict = {"pid": heartbeat_pid, "updated_at": written, "monotonic": 1.0,
                "mem": {"rss_kib": 900_000, "mem_total_kib": 4_000_000, "mem_available_kib": 2_000_000}}
    if oom is not None:
        hb["oom"] = oom
    get_loop_heartbeat_path(home).write_text(json.dumps(hb), encoding="utf-8")


def test_sample_reads_boot_id_host_and_cgroup_counters(kernel: _Kernel) -> None:
    kernel.set(host=7, cgroup=2)
    sample = sample_oom_kills()
    assert sample == {"boot_id": _BOOT, "host_oom_kills": 7, "cgroup_oom_kills": 2,
                      "cgroup_id": kernel.events.parent.stat().st_ino}


def test_sample_omits_what_it_cannot_read(kernel: _Kernel) -> None:
    kernel.boot_id = None
    kernel.set(host=None, cgroup=None, cgroup_line=_CGROUP_V1)  # no vmstat, cgroup v1 only
    assert sample_oom_kills() == {}
    kernel.set(host=4, cgroup=None)  # v2 line but no memory controller file
    assert sample_oom_kills() == {"host_oom_kills": 4}
    kernel.vmstat.write_text("oom_kill banana\n", encoding="utf-8")
    assert sample_oom_kills() == {}


def test_heartbeat_carries_the_oom_sample(tmp_path: Path, kernel: _Kernel) -> None:
    kernel.set(host=7, cgroup=2)
    write_loop_heartbeat(home=tmp_path)
    payload = json.loads(get_loop_heartbeat_path(tmp_path).read_text(encoding="utf-8"))
    assert payload["oom"] == sample_oom_kills()
    assert payload["oom"]["cgroup_oom_kills"] == 2


def test_cgroup_counter_increase_is_cgroup_scoped_oom_kill_activity(tmp_path: Path, kernel: _Kernel) -> None:
    kernel.set(host=7, cgroup=2)
    _died_with_heartbeat(tmp_path, sample_oom_kills())
    kernel.set(host=8, cgroup=3)  # the kernel OOM-killed a process in the gateway's cgroup

    evidence = record_startup(home=tmp_path)

    assert evidence is not None and evidence["oom_kill_seen"] is True
    assert evidence["oom_kill_scope"] == "cgroup"
    sentinel = json.loads(get_lifecycle_sentinel_path(tmp_path).read_text(encoding="utf-8"))
    assert sentinel["prior_oom_kill_seen"] is True and sentinel["prior_oom_kill_scope"] == "cgroup"
    assert collect_memory_status(tmp_path)["last_boot_oom_kill"] == "cgroup"
    [record] = [json.loads(line) for line in
                (tmp_path / "logs" / "gateway-exit-diag.log").read_text(encoding="utf-8").splitlines()]
    assert record["tag"] == "gateway.previous_unclean_exit" and record["oom_kill_scope"] == "cgroup"


def test_cgroup_counter_wins_over_another_process_raising_the_host_counter(tmp_path: Path, kernel: _Kernel) -> None:
    kernel.set(host=7, cgroup=2)
    _died_with_heartbeat(tmp_path, sample_oom_kills())
    kernel.set(host=9, cgroup=2)  # some OTHER process on the host was OOM-killed

    evidence = detect_unclean_exit(tmp_path)

    assert evidence is not None and "oom_kill_seen" not in evidence


def test_host_counter_is_the_fallback_without_a_cgroup_counter(tmp_path: Path, kernel: _Kernel) -> None:
    kernel.set(host=7, cgroup=None, cgroup_line=_CGROUP_V1)
    _died_with_heartbeat(tmp_path, sample_oom_kills())
    kernel.set(host=8, cgroup=None, cgroup_line=_CGROUP_V1)

    evidence = detect_unclean_exit(tmp_path)

    assert evidence is not None and evidence["oom_kill_seen"] is True and evidence["oom_kill_scope"] == "host"


def test_recreated_cgroup_falls_back_to_the_host_counter(tmp_path: Path, kernel: _Kernel) -> None:
    kernel.set(host=7, cgroup=2)
    _died_with_heartbeat(tmp_path, sample_oom_kills())
    kernel.recreate_cgroup()  # unit restarted: new cgroup at 0, not comparable
    kernel.set(host=8, cgroup=None)

    evidence = record_startup(home=tmp_path)

    assert evidence is not None and evidence["oom_kill_scope"] == "host"
    assert collect_memory_status(tmp_path)["last_boot_oom_kill"] == "host"


def test_unchanged_counters_make_no_claim(tmp_path: Path, kernel: _Kernel) -> None:
    kernel.set(host=7, cgroup=2)
    _died_with_heartbeat(tmp_path, sample_oom_kills())

    evidence = record_startup(home=tmp_path)

    assert evidence is not None and evidence["prior_pid"] == _DEAD_PID and "oom_kill_seen" not in evidence
    sentinel = json.loads(get_lifecycle_sentinel_path(tmp_path).read_text(encoding="utf-8"))
    assert sentinel["prior_unclean_exit"] is True and "prior_oom_kill_seen" not in sentinel
    assert collect_memory_status(tmp_path)["last_boot_oom_kill"] is None


def test_a_reboot_between_lives_makes_no_claim(tmp_path: Path, kernel: _Kernel) -> None:
    kernel.set(host=7, cgroup=2)
    _died_with_heartbeat(tmp_path, sample_oom_kills())
    kernel.boot_id = "a-later-boot"
    kernel.set(host=50, cgroup=9)

    evidence = detect_unclean_exit(tmp_path)

    assert evidence is not None and "oom_kill_seen" not in evidence


@pytest.mark.parametrize("oom", [
    pytest.param(None, id="heartbeat-without-oom"),
    pytest.param({"host_oom_kills": 1, "cgroup_oom_kills": 0}, id="no-boot-id"),
    pytest.param({"boot_id": _BOOT, "host_oom_kills": "1"}, id="malformed-counter"),
    pytest.param(["not", "a", "dict"], id="not-a-dict"),
])
def test_missing_or_malformed_heartbeat_data_makes_no_claim(tmp_path: Path, kernel: _Kernel, oom) -> None:
    kernel.set(host=9, cgroup=9)
    _died_with_heartbeat(tmp_path, oom)

    evidence = detect_unclean_exit(tmp_path)

    assert evidence is not None and "oom_kill_seen" not in evidence


def test_missing_kernel_files_now_make_no_claim_and_do_not_raise(tmp_path: Path, kernel: _Kernel) -> None:
    kernel.set(host=7, cgroup=2)
    _died_with_heartbeat(tmp_path, sample_oom_kills())
    kernel.vmstat.unlink()
    kernel.events.unlink()
    kernel.self_cgroup.unlink()

    evidence = detect_unclean_exit(tmp_path)

    assert evidence is not None and "oom_kill_seen" not in evidence


@pytest.mark.platforms("linux")
def test_sample_reads_this_hosts_counters() -> None:
    sample = sample_oom_kills()
    assert isinstance(sample.get("host_oom_kills"), int)
    assert sample.get("boot_id") == Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()


@pytest.mark.platforms("macos")
def test_sample_is_empty_off_linux() -> None:
    assert sample_oom_kills() == {}


def test_clean_previous_life_never_reads_as_oom_kill_activity(tmp_path: Path, kernel: _Kernel) -> None:
    kernel.set(host=7, cgroup=2)
    sentinel = get_lifecycle_sentinel_path(tmp_path)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({"phase": "exited", "pid": os.getpid() + 1, "exit_reason": "graceful_shutdown"}),
                        encoding="utf-8")
    assert record_startup(home=tmp_path) is None
    assert collect_memory_status(tmp_path)["last_boot_oom_kill"] is None


@pytest.mark.parametrize(("age_s", "claimed"), [
    pytest.param(lifecycle_ledger._OOM_WINDOW_S - 20, True, id="inside-window"),
    pytest.param(lifecycle_ledger._OOM_WINDOW_S + 20, False, id="past-window"),
    pytest.param(7 * 24 * 3600.0, False, id="week-old"),
])
def test_counter_growth_is_compared_only_within_the_window(tmp_path: Path, kernel: _Kernel, age_s: float,
                                                           claimed: bool) -> None:
    kernel.set(host=7, cgroup=2)
    _died_with_heartbeat(tmp_path, sample_oom_kills(), age_s=age_s)
    kernel.set(host=8, cgroup=3)

    evidence = detect_unclean_exit(tmp_path)

    assert evidence is not None and evidence.get("oom_kill_seen", False) is claimed


def test_a_heartbeat_from_another_life_makes_no_oom_claim(tmp_path: Path, kernel: _Kernel) -> None:
    """The replacement died before its first heartbeat: the heartbeat left behind is an older
    life's (another pid), so its counters say nothing about this death."""
    kernel.set(host=10, cgroup=0)
    _died_with_heartbeat(tmp_path, sample_oom_kills(), heartbeat_pid=_DEAD_PID - 1)
    kernel.set(host=11, cgroup=1)

    evidence = record_startup(home=tmp_path)

    assert evidence is not None and "oom_kill_seen" not in evidence
    sentinel = json.loads(get_lifecycle_sentinel_path(tmp_path).read_text(encoding="utf-8"))
    assert sentinel["prior_unclean_exit"] is True and "prior_oom_kill_seen" not in sentinel


def test_a_heartbeat_older_than_the_lifes_claim_makes_no_oom_claim(tmp_path: Path, kernel: _Kernel) -> None:
    kernel.set(host=10, cgroup=0)
    _died_with_heartbeat(tmp_path, sample_oom_kills(), age_s=60.0, claimed_at=time.time() - 30)  # same pid reused
    kernel.set(host=11, cgroup=1)

    evidence = detect_unclean_exit(tmp_path)

    assert evidence is not None and "oom_kill_seen" not in evidence


def test_non_utf8_cgroup_path_still_records_the_unclean_exit(tmp_path: Path, kernel: _Kernel) -> None:
    kernel.set(host=7, cgroup=2)
    _died_with_heartbeat(tmp_path, sample_oom_kills())
    kernel.self_cgroup.write_bytes(b"0::/system.slice/hermes-\xff.service\n")
    lifecycle_ledger._cgroup_v2_events_path.cache_clear()

    assert cgroup_cleanup._own_cgroup_path() == "/system.slice/hermes-\udcff.service"
    evidence = record_startup(home=tmp_path)

    assert evidence is not None
    sentinel = json.loads(get_lifecycle_sentinel_path(tmp_path).read_text(encoding="utf-8"))
    assert sentinel["prior_unclean_exit"] is True


def test_a_failing_oom_enrichment_keeps_the_unclean_verdict(tmp_path: Path, kernel: _Kernel,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    kernel.set(host=7, cgroup=2)
    _died_with_heartbeat(tmp_path, sample_oom_kills())

    def _boom() -> dict:
        raise RuntimeError("reader bug")

    monkeypatch.setattr(lifecycle_ledger, "sample_oom_kills", _boom)
    evidence = record_startup(home=tmp_path)

    assert evidence is not None and evidence["prior_pid"] == _DEAD_PID and "oom_kill_seen" not in evidence
    sentinel = json.loads(get_lifecycle_sentinel_path(tmp_path).read_text(encoding="utf-8"))
    assert sentinel["prior_unclean_exit"] is True


def test_the_cgroup_path_is_resolved_once_per_process(kernel: _Kernel, monkeypatch: pytest.MonkeyPatch) -> None:
    kernel.set(host=7, cgroup=2)
    calls: list[int] = []
    real = cgroup_cleanup._own_cgroup_path
    monkeypatch.setattr(cgroup_cleanup, "_own_cgroup_path", lambda: calls.append(1) or real())

    first, second = sample_oom_kills(), sample_oom_kills()

    assert first == second and first["cgroup_oom_kills"] == 2
    assert calls == [1]
