"""Default -j/--jobs worker count must fit a cgroup MemoryMax cap.

kanban workers run scripts/run_tests.sh -> run_tests_parallel.py wrapped in
tools/process_registry.py's ``systemd-run --user --scope --property
MemoryMax=<cap>`` (effectively 4GiB on a typical host). The historical
default worker count is ``os.cpu_count() * 2`` — 112 on a 56-core host —
which spawns 112 concurrent ``python -m pytest <file>`` interpreters and
blows straight through a 4GiB scope. The kernel cgroup OOM killer then
kills the scope's top-level process (the kanban worker itself, not a test
subprocess), silently ending the agent turn with no exception surfaced.

_default_job_count() must clamp the default to fit a detected cgroup
memory.max, while still honoring an explicit HERMES_TEST_WORKERS override
verbatim (the caller stated intent) and never *raising* the default above
cpu_count()*2 just because headroom exists.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_PATH = REPO_ROOT / "scripts" / "run_tests_parallel.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_tests_parallel", _RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_cgroup_info_falls_back_to_cpu_count_times_two(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_runner()
    monkeypatch.delenv("HERMES_TEST_WORKERS", raising=False)
    monkeypatch.setattr(mod, "_cgroup_memory_max_bytes", lambda: None)
    monkeypatch.setattr(mod.os, "cpu_count", lambda: 56)
    assert mod._default_job_count() == 112


def test_tight_memory_cap_clamps_default_below_cpu_count_times_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduces the OOM incident: 56 cores, 4GiB scope -> naive default
    112 workers, clamped default must fit inside the cap."""
    mod = _load_runner()
    monkeypatch.delenv("HERMES_TEST_WORKERS", raising=False)
    monkeypatch.setattr(mod.os, "cpu_count", lambda: 56)
    four_gib = 4 * 1024 * 1024 * 1024
    monkeypatch.setattr(mod, "_cgroup_memory_max_bytes", lambda: four_gib)

    clamped = mod._default_job_count()

    assert clamped < 112
    assert clamped * mod._ASSUMED_WORKER_RSS_BYTES <= four_gib


def test_generous_memory_cap_does_not_raise_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cgroup with ample headroom must not push the default ABOVE the
    historical cpu_count()*2 ceiling."""
    mod = _load_runner()
    monkeypatch.delenv("HERMES_TEST_WORKERS", raising=False)
    monkeypatch.setattr(mod.os, "cpu_count", lambda: 4)
    huge = 512 * 1024 * 1024 * 1024
    monkeypatch.setattr(mod, "_cgroup_memory_max_bytes", lambda: huge)

    assert mod._default_job_count() == 8


def test_explicit_env_override_wins_verbatim_even_under_a_tight_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HERMES_TEST_WORKERS is an explicit, stated intent and must never be
    silently overridden by the cgroup clamp."""
    mod = _load_runner()
    monkeypatch.setenv("HERMES_TEST_WORKERS", "200")
    monkeypatch.setattr(mod.os, "cpu_count", lambda: 56)
    monkeypatch.setattr(mod, "_cgroup_memory_max_bytes", lambda: 4 * 1024 * 1024 * 1024)

    assert mod._default_job_count() == 200


def test_cgroup_memory_max_bytes_reads_real_proc_cgroup_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise the actual /proc/self/cgroup + /sys/fs/cgroup/.../memory.max
    parsing path (not just the monkeypatched seam above)."""
    mod = _load_runner()
    cgroup_file = tmp_path / "cgroup"
    cgroup_file.write_text("0::/user.slice/hermes-worker.scope\n", encoding="utf-8")
    sys_fs_cgroup = tmp_path / "sys_fs_cgroup"
    scope_dir = sys_fs_cgroup / "user.slice" / "hermes-worker.scope"
    scope_dir.mkdir(parents=True)
    (scope_dir / "memory.max").write_text("4294967296\n", encoding="utf-8")

    real_read_text = Path.read_text

    def fake_read_text(self: Path, *args, **kwargs):
        if str(self) == "/proc/self/cgroup":
            return real_read_text(cgroup_file, *args, **kwargs)
        if str(self).startswith("/sys/fs/cgroup/"):
            relative = str(self)[len("/sys/fs/cgroup/"):]
            return real_read_text(sys_fs_cgroup / relative, *args, **kwargs)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    assert mod._cgroup_memory_max_bytes() == 4294967296


def test_cgroup_memory_max_bytes_unlimited_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A "max" (unlimited) memory.max value must not be parsed as a cap."""
    mod = _load_runner()
    cgroup_file = tmp_path / "cgroup"
    cgroup_file.write_text("0::/user.slice/foo.scope\n", encoding="utf-8")
    sys_fs_cgroup = tmp_path / "sys_fs_cgroup"
    scope_dir = sys_fs_cgroup / "user.slice" / "foo.scope"
    scope_dir.mkdir(parents=True)
    (scope_dir / "memory.max").write_text("max\n", encoding="utf-8")

    real_read_text = Path.read_text

    def fake_read_text(self: Path, *args, **kwargs):
        if str(self) == "/proc/self/cgroup":
            return real_read_text(cgroup_file, *args, **kwargs)
        if str(self).startswith("/sys/fs/cgroup/"):
            relative = str(self)[len("/sys/fs/cgroup/"):]
            return real_read_text(sys_fs_cgroup / relative, *args, **kwargs)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    assert mod._cgroup_memory_max_bytes() is None
