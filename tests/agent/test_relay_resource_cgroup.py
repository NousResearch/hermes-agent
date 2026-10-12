"""Local identity fencing and measured CPU intervals never invent unavailable data."""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from agent import relay_resource_cgroup as resources

NATIVE_ID = "a" * 64


@pytest.fixture
def source(tmp_path, monkeypatch):
    proc, cgroup = tmp_path / "proc", tmp_path / "cgroup"
    boot = proc / "sys/kernel/random"
    boot.mkdir(parents=True)
    (boot / "boot_id").write_text("f52c11f3-a5ea-4f79-9fce-cf40640df87a\n")
    process = proc / "123"
    process.mkdir()
    fields = ["S", *["0"] * 18, "900", "0"]
    (process / "stat").write_text("123 (worker (with parentheses)) " + " ".join(fields))
    (process / "cgroup").write_text(f"0::/docker/{NATIVE_ID}\n")
    counters = cgroup / "docker" / NATIVE_ID
    counters.mkdir(parents=True)
    (counters / "cpu.stat").write_text("usage_usec 1500000\nuser_usec 1000000\nsystem_usec 500000\n")
    (counters / "memory.current").write_text("4096\n")
    (counters / "cpu.pressure").write_text("some avg10=0.00 avg60=0.00 avg300=0.00 total=150\n")
    monkeypatch.setattr(resources, "_inspect", lambda *_: (123, "2026-10-12T00:00:00Z"))
    clock = itertools.count(1_000_000, 1000)
    monkeypatch.setattr(resources.time, "monotonic_ns", lambda: next(clock))
    monkeypatch.setattr(resources.time, "time_ns", lambda: 100_000_000_000)
    return proc, cgroup, counters


def _sample(source):
    proc, cgroup, _ = source
    return resources.read_sample("docker", NATIVE_ID, proc_root=proc, cgroup_root=cgroup)


@pytest.fixture
def reading():
    return resources.CgroupSample(
        "available", source_identity=("boot", "started", 123, 900, 1, 2),
        read_started_ns=1000, read_ended_ns=7000, cpu_started_ns=2000, cpu_ended_ns=3000,
        cpu_usage_us=1500000, cpu_user_us=1000000, cpu_system_us=500000, cpu_pressure_some_us=150,
        memory_started_ns=4000, memory_ended_ns=5000, memory_current_bytes=4096,
        source_end_unix_ns=100_000_000_000, source_end_monotonic_ns=6500, clock_mapping_uncertainty_ns=500,
    )


@pytest.mark.platforms("posix")
def test_cpu_without_quota_and_memory_have_independent_source_windows(source):
    sample = _sample(source)
    assert sample.status == "available"
    assert (sample.cpu_usage_us, sample.cpu_user_us, sample.cpu_system_us) == (1500000, 1000000, 500000)
    assert sample.cpu_throttled_us is None
    assert sample.cpu_pressure_some_us == 150
    assert sample.cpu_started_ns < sample.cpu_ended_ns < sample.memory_started_ns < sample.memory_ended_ns
    assert sample.clock_mapping_uncertainty_ns == 500
    data = resources.sample_data(sample)
    assert data["cpu_interval_status"] == "baseline"
    assert data["memory_current_bytes"] == 4096
    assert data["source_read_duration_ns"] == 6000
    assert data["memory_window"]["window_duration_ns"] == 1000
    assert data["scheduler_switches_status"] == "not_collected"
    assert NATIVE_ID not in json.dumps(data)
    assert str(source[0]) not in json.dumps(data)


@pytest.mark.platforms("posix")
def test_optional_memory_pressure_and_throttle_failures_do_not_fail_cpu(source):
    _, _, counters = source
    (counters / "memory.current").unlink()
    (counters / "cpu.pressure").write_text("some malformed\n")
    sample = _sample(source)
    assert sample.status == "available"
    assert sample.memory_current_bytes is None
    assert sample.cpu_pressure_some_us is None
    data = resources.sample_data(sample)
    assert data["memory_status"] == "unavailable"
    assert "memory_current_bytes" not in data
    assert "cpu_cumulative_pressure_some_us" not in data


def test_intervals_use_monotonic_duration_and_calibrated_unix_windows(reading):
    previous = reading
    current = replace(previous, cpu_started_ns=previous.cpu_started_ns + 1_000_000_000,
                      cpu_ended_ns=previous.cpu_ended_ns + 1_000_000_000,
                      cpu_usage_us=previous.cpu_usage_us + 2_000_000,
                      cpu_user_us=previous.cpu_user_us + 1_750_000,
                      cpu_system_us=previous.cpu_system_us + 250_000,
                      cpu_pressure_some_us=previous.cpu_pressure_some_us + 700_000)
    data = resources.sample_data(current, previous)
    assert data["cpu_usage_cores"] == 2.0
    assert data["window_duration_ns"] == 1_000_000_000
    assert data["window_end_unix_ns"] - data["window_start_unix_ns"] == 1_000_000_000
    assert data["cpu_pressure_some_delta_us"] == 700_000
    assert data["interval_measurement_uncertainty_ns"] == 1000
    assert "cpu_throttled_delta_us" not in data


@pytest.mark.parametrize("change", ["boot", "container", "pid", "counter", "pressure", "clock"])
def test_restarts_counter_resets_and_nonpositive_intervals_begin_new_baseline(reading, change):
    previous = reading
    current = replace(previous, cpu_started_ns=previous.cpu_started_ns + 10_000,
                      cpu_ended_ns=previous.cpu_ended_ns + 10_000, cpu_usage_us=2_000_000)
    if change in ("boot", "container", "pid"):
        identity = list(current.source_identity)
        identity[{"boot": 0, "container": 1, "pid": 3}[change]] = "replacement"
        current = replace(current, source_identity=tuple(identity))
    elif change == "counter":
        current = replace(current, cpu_usage_us=1)
    elif change == "pressure":
        current = replace(current, cpu_pressure_some_us=1)
    else:
        current = replace(current, cpu_started_ns=previous.cpu_started_ns, cpu_ended_ns=previous.cpu_ended_ns)
    data = resources.sample_data(current, previous)
    assert data["cpu_interval_status"] == "baseline"
    assert "cpu_usage_delta_us" not in data
    assert "cpu_usage_cores" not in data


def test_failed_previous_sample_does_not_poison_new_baseline(reading):
    assert resources.sample_data(reading, resources.CgroupSample("source_unavailable"))[
        "cpu_interval_status"] == "baseline"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("path", ["/", "/docker/" + "b" * 64, "/docker/prefix-" + NATIVE_ID,
                                 f"/docker/{NATIVE_ID}/../{NATIVE_ID}", f"/docker//{NATIVE_ID}"])
def test_remote_pid_collisions_root_and_unsafe_paths_are_not_sampled(source, path):
    proc, _, _ = source
    (proc / "123/cgroup").write_text(f"0::{path}\n")
    data = resources.sample_data(_sample(source))
    assert data == {"schema_version": "1", "source": "linux_local_cgroup_v2",
                    "status": "source_unavailable", "scheduler_switches_status": "not_collected"}


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("target", ["root", "directory", "counter"])
def test_symlink_sources_are_rejected(source, target):
    proc, root, counters = source
    chosen = {"root": root, "directory": counters, "counter": counters / "cpu.stat"}[target]
    moved = chosen.with_name(chosen.name + "-real")
    chosen.rename(moved)
    chosen.symlink_to(moved, target_is_directory=target != "counter")
    sample = resources.read_sample("docker", NATIVE_ID, proc_root=proc, cgroup_root=root)
    assert sample.status == "source_unavailable"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("change", ["pid", "container", "directory", "cgroup"])
def test_identity_must_remain_stable_during_source_reads(source, monkeypatch, change):
    proc, _, counters = source
    if change == "pid":
        starts = iter([900, 901])
        monkeypatch.setattr(resources, "_pid_start", lambda *_: next(starts))
    elif change == "container":
        inspections = iter([(123, "before"), (123, "after")])
        monkeypatch.setattr(resources, "_inspect", lambda *_: next(inspections))
    else:
        original = resources._read_counters

        def replacement(fd):
            values = original(fd)
            if change == "directory":
                counters.rename(counters.with_name("replaced"))
                counters.mkdir()
            else:
                (proc / "123/cgroup").write_text(f"0::/replacement/{NATIVE_ID}\n")
            return values

        monkeypatch.setattr(resources, "_read_counters", replacement)
    assert _sample(source).status == "source_unavailable"


def test_inspection_binds_exact_docker_identity_and_never_exports_diagnostics(monkeypatch):
    from tools.environments import remote_common
    from subprocess import CompletedProcess

    calls = []

    def capture(argv, *, timeout):
        calls.append((argv, timeout))
        return CompletedProcess(argv, 0, stdout=json.dumps(["b" * 64, 123, "started", True]))

    monkeypatch.setattr(remote_common, "run_capture", capture)
    with pytest.raises(ValueError):
        resources._inspect("/custom/docker", NATIVE_ID, 2.0)
    assert calls[0][0][-1] == NATIVE_ID
    assert calls[0][1] == 2.0


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("text", ["usage_usec -1\n", "user_usec 1\n", "usage_usec broken\n"])
def test_invalid_mandatory_counter_is_unknown_not_zero(source, text):
    (source[2] / "cpu.stat").write_text(text)
    assert _sample(source).status == "source_unavailable"


def test_unavailable_host_and_invalid_id_have_explicit_unknown_statuses(tmp_path):
    assert resources.read_sample("docker", NATIVE_ID, proc_root=tmp_path).status == "unsupported_host"
    assert resources.read_sample("docker", "short-id", proc_root=tmp_path).status == "identity_unverified"


@pytest.mark.platforms("linux")
def test_live_linux_proc_pid_binding_reads_real_kernel_start_ticks():
    start = resources._pid_start(Path("/proc"), os.getpid())
    assert isinstance(start, int) and start > 0
    boot_id = (Path("/proc") / "sys/kernel/random/boot_id").read_text().strip()
    assert len(boot_id) == 36
