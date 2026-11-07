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


def _fake_cgroup_tree(tmp_path, limits):
    """Build /proc/self/cgroup + /sys/fs/cgroup stand-ins under tmp_path.

    ``limits`` maps a cgroup path relative to the root ("" is the root) to
    its memory.max text; the process sits in the deepest path.
    """
    root = tmp_path / "sys_fs_cgroup"
    for rel, raw in limits.items():
        level = root.joinpath(*rel.split("/")) if rel else root
        level.mkdir(parents=True, exist_ok=True)
        level.joinpath("memory.max").write_text(raw, encoding="utf-8")
    leaf = max(limits, key=lambda rel: rel.count("/") + bool(rel))
    proc_cgroup = tmp_path / "cgroup"
    proc_cgroup.write_text(f"0::/{leaf}\n", encoding="utf-8")
    return proc_cgroup, root


@pytest.mark.parametrize(
    "limits,expected",
    [
        ({"user.slice": "max\n", "user.slice/job.scope": "4294967296\n"}, 4 * _GIB),
        ({"user.slice": "max\n", "user.slice/job.scope": "max\n"}, None),
        # cgroup-v2 limits are hierarchical: a leaf of "max" under a finite
        # ancestor is still capped by that ancestor.
        ({"user.slice": "4294967296\n", "user.slice/job.scope": "max\n"}, 4 * _GIB),
        ({"user.slice": "2147483648\n", "user.slice/job.scope": "8589934592\n"}, 2 * _GIB),
        # A cgroup-namespaced container sees its own cap at the visible root.
        ({"": "4294967296\n", "job.scope": "max\n"}, 4 * _GIB),
    ],
)
def test_effective_memory_max_is_the_tightest_visible_ancestor(tmp_path, limits, expected):
    mod = _load_runner()
    proc_cgroup, root = _fake_cgroup_tree(tmp_path, limits)
    assert mod._cgroup_memory_max_bytes(proc_cgroup, root) == expected
