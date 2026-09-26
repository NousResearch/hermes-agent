"""The runner's default -j must fit a cgroup-v2 memory.max cap.

Inside a container, CI job or systemd scope with a finite memory.max, one
pytest subprocess per CPU on a many-core host can overrun the cap, and the
kernel OOM-kills the cgroup's top-level process (the caller, not a test).
An explicit HERMES_TEST_WORKERS is stated intent and is never clamped.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_RUNNER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_tests_parallel.py"
_GIB = 1024 * 1024 * 1024


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_tests_parallel", _RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def runner(monkeypatch):
    mod = _load_runner()
    monkeypatch.delenv("HERMES_TEST_WORKERS", raising=False)
    monkeypatch.setattr(mod.os, "cpu_count", lambda: 64)
    return mod


def test_default_fits_a_tight_cap_and_never_exceeds_cpu_count(runner, monkeypatch):
    monkeypatch.setattr(runner, "_cgroup_memory_max_bytes", lambda: 4 * _GIB)
    clamped = runner._default_job_count()
    assert 1 <= clamped < 64
    assert clamped * runner._ASSUMED_WORKER_RSS_BYTES <= 4 * _GIB

    monkeypatch.setattr(runner, "_cgroup_memory_max_bytes", lambda: 512 * _GIB)
    assert runner._default_job_count() == 64
    monkeypatch.setattr(runner, "_cgroup_memory_max_bytes", lambda: None)
    assert runner._default_job_count() == 64


def test_explicit_env_override_is_never_clamped(runner, monkeypatch):
    monkeypatch.setenv("HERMES_TEST_WORKERS", "200")
    monkeypatch.setattr(runner, "_cgroup_memory_max_bytes", lambda: 4 * _GIB)
    assert runner._default_job_count() == 200


@pytest.mark.parametrize("raw,expected", [("4294967296\n", 4294967296), ("max\n", None)])
def test_reads_memory_max_of_the_process_cgroup(monkeypatch, tmp_path, raw, expected):
    mod = _load_runner()
    cgroup_file = tmp_path / "cgroup"
    cgroup_file.write_text("0::/user.slice/job.scope\n", encoding="utf-8")
    sys_fs_cgroup = tmp_path / "sys_fs_cgroup"
    (sys_fs_cgroup / "user.slice" / "job.scope").mkdir(parents=True)
    (sys_fs_cgroup / "user.slice" / "job.scope" / "memory.max").write_text(raw, encoding="utf-8")
    real_read_text = Path.read_text

    def fake_read_text(self, *args, **kwargs):
        if str(self) == "/proc/self/cgroup":
            return real_read_text(cgroup_file, *args, **kwargs)
        if str(self).startswith("/sys/fs/cgroup/"):
            return real_read_text(sys_fs_cgroup / str(self)[len("/sys/fs/cgroup/"):], *args, **kwargs)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    assert mod._cgroup_memory_max_bytes() == expected
