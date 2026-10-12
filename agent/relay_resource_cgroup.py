"""Content-free, host-side cgroup-v2 readings for verified local Docker targets.

No process runs inside the measured container. Docker inspect's PID alone is not
trusted: its local cgroup must carry the exact full container ID, and process,
container and directory identities must remain stable across each reading.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

_CONTAINER_ID = re.compile(r"[0-9a-f]{64}\Z")
_COUNTERS = ("cpu_usage_us", "cpu_user_us", "cpu_system_us", "cpu_pressure_some_us", "cpu_throttled_us")


@dataclass(frozen=True)
class CgroupSample:
    status: str
    source_identity: tuple | None = None
    read_started_ns: int | None = None
    read_ended_ns: int | None = None
    cpu_started_ns: int | None = None
    cpu_ended_ns: int | None = None
    cpu_usage_us: int | None = None
    cpu_user_us: int | None = None
    cpu_system_us: int | None = None
    cpu_pressure_some_us: int | None = None
    cpu_throttled_us: int | None = None
    memory_started_ns: int | None = None
    memory_ended_ns: int | None = None
    memory_current_bytes: int | None = None
    source_end_unix_ns: int | None = None
    source_end_monotonic_ns: int | None = None
    clock_mapping_uncertainty_ns: int | None = None


def _inspect(docker_exe: str, container_id: str, timeout: float) -> tuple[int, str]:
    from tools.environments.remote_common import run_capture

    result = run_capture([
        docker_exe, "inspect", "--format",
        "[{{json .Id}},{{.State.Pid}},{{json .State.StartedAt}},{{.State.Running}}]",
        container_id,
    ], timeout=timeout)
    identity, pid, started, running = json.loads(result.stdout)
    if result.returncode or identity != container_id or running is not True or type(pid) is not int or pid <= 0:
        raise ValueError("Unverified Docker identity")
    if not isinstance(started, str) or not started:
        raise ValueError("Missing Docker start identity")
    return pid, started


def _pid_start(proc_root: Path, pid: int) -> int:
    text = (proc_root / str(pid) / "stat").read_text(encoding="ascii")
    prefix, _, tail = text.rpartition(")")
    if int(prefix.split("(", 1)[0]) != pid:
        raise ValueError("Unverified process identity")
    return int(tail.split()[19])


def _cgroup_parts(proc_root: Path, pid: int, container_id: str) -> tuple[str, ...]:
    rows = (proc_root / str(pid) / "cgroup").read_text(encoding="ascii").splitlines()
    [path] = [row[3:] for row in rows if row.startswith("0::")]
    parts = path.split("/")[1:]
    allowed = {container_id, f"docker-{container_id}.scope"}
    if not path.startswith("/") or not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError("Unsafe cgroup identity")
    if not any(part in allowed for part in parts):
        raise ValueError("Docker PID is not a verified local target")
    return tuple(parts)


@contextlib.contextmanager
def _directory(root: Path, parts: tuple[str, ...]):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(root, flags)
    try:
        for part in parts:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def _text_at(fd: int, name: str) -> str:
    with os.fdopen(os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd), encoding="ascii") as source:
        value = source.read(16385)
    if len(value) > 16384:
        raise ValueError("Oversized counter source")
    return value


def _number(value: str | None) -> int | None:
    if value is None:
        return None
    number = int(value)
    if number < 0:
        raise ValueError("Negative cumulative counter")
    return number


def _optional(fd: int, name: str) -> str | None:
    try:
        return _text_at(fd, name)
    except (OSError, UnicodeError, ValueError):
        return None


def _pressure(text: str | None) -> int | None:
    if text is not None:
        for line in text.splitlines():
            if line.startswith("some "):
                values = dict(item.split("=", 1) for item in line.split()[1:])
                return _number(values.get("total"))
    return None


def _read_counters(fd: int) -> dict:
    cpu_started = time.monotonic_ns()
    cpu = dict(line.split() for line in _text_at(fd, "cpu.stat").splitlines())
    cpu_values = {"cpu_usage_us": _number(cpu["usage_usec"]),
                  "cpu_user_us": _number(cpu.get("user_usec")),
                  "cpu_system_us": _number(cpu.get("system_usec")),
                  "cpu_throttled_us": _number(cpu.get("throttled_usec"))}
    with contextlib.suppress(ValueError):
        cpu_values["cpu_pressure_some_us"] = _pressure(_optional(fd, "cpu.pressure"))
    cpu_ended = time.monotonic_ns()
    memory_started = time.monotonic_ns()
    memory = None
    with contextlib.suppress(ValueError):
        memory = _number(_optional(fd, "memory.current"))
    memory_ended = time.monotonic_ns()
    return {**cpu_values, "cpu_started_ns": cpu_started, "cpu_ended_ns": cpu_ended,
            "memory_started_ns": memory_started, "memory_ended_ns": memory_ended,
            "memory_current_bytes": memory}


def _verified_sample(docker_exe: str, container_id: str, proc_root: Path, cgroup_root: Path,
                     timeout: float, boot_id: str, read_started: int) -> CgroupSample:
    pid, started = _inspect(docker_exe, container_id, timeout)
    pid_start = _pid_start(proc_root, pid)
    parts = _cgroup_parts(proc_root, pid, container_id)
    with _directory(cgroup_root, parts) as fd:
        stat = os.fstat(fd)
        values = _read_counters(fd)
        if _pid_start(proc_root, pid) != pid_start or _cgroup_parts(proc_root, pid, container_id) != parts:
            raise ValueError("Source process changed during reading")
        with _directory(cgroup_root, parts) as checked_fd:
            checked = os.fstat(checked_fd)
            if (stat.st_dev, stat.st_ino) != (checked.st_dev, checked.st_ino):
                raise ValueError("Cgroup changed during reading")
    if _inspect(docker_exe, container_id, timeout) != (pid, started):
        raise ValueError("Container restarted during reading")
    before = time.monotonic_ns()
    unix = time.time_ns()
    after = time.monotonic_ns()
    return CgroupSample(
        "available", source_identity=(boot_id, started, pid, pid_start, stat.st_dev, stat.st_ino),
        read_started_ns=read_started, read_ended_ns=after, **values,
        source_end_unix_ns=unix, source_end_monotonic_ns=(before + after) // 2,
        clock_mapping_uncertainty_ns=(after - before + 1) // 2,
    )


def read_sample(docker_exe: str, container_id: str, *, proc_root: Path = Path("/proc"),
                cgroup_root: Path = Path("/sys/fs/cgroup"), timeout: float = 2.0) -> CgroupSample:
    """Return unknown statuses, never substituted zeroes, on unsupported/failed sources."""
    read_started = time.monotonic_ns()
    if not isinstance(container_id, str) or _CONTAINER_ID.fullmatch(container_id) is None:
        return CgroupSample("identity_unverified")
    try:
        boot_id = str(uuid.UUID((proc_root / "sys/kernel/random/boot_id").read_text(encoding="ascii").strip()))
    except (OSError, ValueError):
        return CgroupSample("unsupported_host")
    try:
        return _verified_sample(docker_exe, container_id, proc_root, cgroup_root, timeout, boot_id, read_started)
    except (OSError, ValueError, UnicodeError, KeyError, IndexError, subprocess.SubprocessError):
        # Docker diagnostics and local paths never enter the exported schema.
        return CgroupSample("source_unavailable")


def _deltas(current: CgroupSample, previous: CgroupSample | None) -> dict | None:
    if (previous is None or previous.status != "available" or current.source_identity is None
            or current.source_identity != previous.source_identity):
        return None
    deltas = {}
    for name in _COUNTERS:
        now, then = getattr(current, name), getattr(previous, name)
        if now is not None and then is not None:
            if now < then:
                return None
            deltas[name] = now - then
    return deltas


def _window(sample: CgroupSample, start: int, end: int) -> dict:
    anchor = sample.source_end_unix_ns - sample.source_end_monotonic_ns
    return {"window_start_unix_ns": anchor + start, "window_end_unix_ns": anchor + end,
            "window_duration_ns": end - start,
            "clock_mapping_uncertainty_ns": sample.clock_mapping_uncertainty_ns}


def sample_data(current: CgroupSample, previous: CgroupSample | None = None) -> dict:
    """Describe real source windows; intervals reset on process/counter discontinuities."""
    data = {"schema_version": "1", "source": "linux_local_cgroup_v2", "status": current.status,
            "scheduler_switches_status": "not_collected"}
    if current.status != "available":
        return data
    data["source_read_duration_ns"] = current.read_ended_ns - current.read_started_ns
    data["cpu_read_duration_ns"] = current.cpu_ended_ns - current.cpu_started_ns
    data["cpu_read_window"] = _window(current, current.cpu_started_ns, current.cpu_ended_ns)
    for name in _COUNTERS:
        value = getattr(current, name)
        if value is not None:
            data[name.replace("cpu_", "cpu_cumulative_", 1)] = value
    data.update(source_end_unix_ns=current.source_end_unix_ns,
                source_end_monotonic_ns=current.source_end_monotonic_ns,
                clock_mapping_uncertainty_ns=current.clock_mapping_uncertainty_ns)
    deltas = _deltas(current, previous)
    end = (current.cpu_started_ns + current.cpu_ended_ns) // 2
    start = end if deltas is None else (previous.cpu_started_ns + previous.cpu_ended_ns) // 2
    if deltas is None or end <= start or "cpu_usage_us" not in deltas:
        data["cpu_interval_status"] = "baseline"
    else:
        data.update(_window(current, start, end), cpu_interval_status="available")
        data["interval_measurement_uncertainty_ns"] = (
            current.cpu_ended_ns - current.cpu_started_ns + previous.cpu_ended_ns - previous.cpu_started_ns
        ) // 2
        data["cpu_usage_delta_us"] = deltas["cpu_usage_us"]
        data["cpu_usage_cores"] = deltas["cpu_usage_us"] * 1000 / (end - start)
        for name in _COUNTERS[1:]:
            if name in deltas:
                data[name.removesuffix("_us") + "_delta_us"] = deltas[name]
    data["memory_status"] = "unavailable" if current.memory_current_bytes is None else "available"
    if current.memory_current_bytes is not None:
        data["memory_current_bytes"] = current.memory_current_bytes
        data["memory_window"] = _window(current, current.memory_started_ns, current.memory_ended_ns)
    return data
