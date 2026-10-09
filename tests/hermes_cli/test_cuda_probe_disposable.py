"""The CUDA driver-pool probe must not make its caller a CUDA client (#133426).

``cuInit`` keeps device handles open for the rest of the calling process's life, and a CUDA
client that never allocates still blocks the GPU's runtime suspend. The hardware endpoint
that reaches this probe is polled every few seconds by the Desktop backend, so the driver
work runs in a disposable child (``cuda_probe``) and only plain data crosses back.

The child's JSON contract is asserted against the real module in a real subprocess — the
property under test is process lifetime, which a mock cannot reproduce. The parent half is
driven through a stub spawn so its parsing and failure contract are exercised without a
driver present.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import hermes_cli.local_runtime.cuda_probe as cuda_probe
import hermes_cli.local_runtime.hardware as hw

_PY = sys.executable


def _run_child() -> subprocess.CompletedProcess:
    return subprocess.run(
        [_PY, "-I", str(Path(cuda_probe.__file__).resolve())],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )


# ── the child's JSON contract ────────────────────────────────


def test_child_prints_one_json_line_even_without_a_driver():
    """No driver on the host is a legitimate answer: one JSON line, nulls, exit 0 — never a
    traceback the parent has to parse around. On a host WITH a driver the same line carries
    real values; either shape must satisfy the contract."""
    result = _run_child()
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert set(payload) == {"pool", "integrated"}
    if payload["pool"] is None:
        assert payload["integrated"] is None
    else:
        assert isinstance(payload["pool"], int) and payload["pool"] > 0
        assert isinstance(payload["integrated"], bool)


# ── the parent half: delegation, parsing, and failure contract ──


def _spawn_result(monkeypatch, *, stdout: str, returncode: int = 0, raises=None):
    class _Completed:
        def __init__(self):
            self.stdout = stdout
            self.returncode = returncode

    def fake_run(*args, **_kwargs):
        cmd = args[0]
        assert cmd[1] == "-I", "the child must run isolated so ambient modules cannot leak in"
        assert str(Path(cuda_probe.__file__).resolve()) in cmd, "python must run this module"
        if raises is not None:
            raise raises
        return _Completed()

    monkeypatch.setattr(cuda_probe.subprocess, "run", fake_run)
    return cuda_probe.probe_pool()


def test_probe_pool_parses_a_discrete_answer(monkeypatch):
    got = _spawn_result(monkeypatch, stdout='{"pool": 8589934592, "integrated": false}\n')
    assert got == (8589934592, False)


def test_probe_pool_parses_an_integrated_answer(monkeypatch):
    got = _spawn_result(monkeypatch, stdout='{"pool": 49821620632, "integrated": true}\n')
    assert got == (49821620632, True)


def test_probe_pool_reads_the_last_line_only(monkeypatch):
    """A driver that chats on stdout before the JSON must not corrupt the read."""
    got = _spawn_result(
        monkeypatch, stdout='libcuda: warning\n\n{"pool": 8589934592, "integrated": null}\n')
    assert got == (8589934592, None)


def test_probe_pool_null_pool_is_none(monkeypatch):
    assert _spawn_result(monkeypatch, stdout='{"pool": null, "integrated": null}\n') is None


def test_probe_pool_garbage_is_none(monkeypatch):
    assert _spawn_result(monkeypatch, stdout="not json at all\n") is None


def test_probe_pool_nonzero_exit_is_none(monkeypatch):
    assert _spawn_result(
        monkeypatch, stdout='{"pool": 1, "integrated": false}\n', returncode=1) is None


def test_probe_pool_spawn_failure_is_none(monkeypatch):
    assert _spawn_result(monkeypatch, stdout="", raises=OSError("no exec")) is None


def test_probe_pool_timeout_is_none(monkeypatch):
    assert _spawn_result(
        monkeypatch, stdout="",
        raises=subprocess.TimeoutExpired(cmd="cuda_probe",
                                         timeout=cuda_probe._PROBE_TIMEOUT_S)) is None


def test_probe_pool_missing_integrated_key_keeps_the_pool(monkeypatch):
    """A partial child answer keeps the pool and degrades the classification to unknown,
    which is what the attribute-less engine fallback already means."""
    got = _spawn_result(monkeypatch, stdout='{"pool": 8589934592}\n')
    assert got == (8589934592, None)


# ── hardware.py delegates to the child seam ───────────────────


def test_hardware_delegates_to_the_disposable_probe(monkeypatch):
    calls = []

    def fake_probe():
        calls.append(1)
        return (1234, False)

    monkeypatch.setattr(cuda_probe, "probe_pool", fake_probe)
    assert hw._cuda_driver_pool() == (1234, False)
    assert calls == [1]
