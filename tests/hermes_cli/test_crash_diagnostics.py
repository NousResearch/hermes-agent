"""Tests for crash-cause diagnostics (``hermes_cli.crash_diagnostics``).

Pins the cause inference (signal + faulting backtrace → human-readable cause), the
cross-platform dispatch (never raises into the caller), and the one-line summary format.
The OS-specific readers (.ips / coredumpctl / WER) are exercised only through the
graceful-degradation path here — they're integration-tested on their respective platforms.
"""
from __future__ import annotations

from hermes_cli import crash_diagnostics as cd


def test_infer_cause_gpu_metal():
    cause = cd._infer_cause("SIGABRT", ["libmlx.dylib: x", "Metal: command buffer"])
    assert "GPU error (Metal/MLX)" in cause


def test_infer_cause_gpu_cuda():
    cause = cd._infer_cause("SIGABRT", ["libcudart.so: cudaMalloc", "libtorch_cuda.so"])
    assert "GPU error (CUDA/NVIDIA)" in cause


def test_infer_cause_segfault():
    assert "segmentation fault" in cd._infer_cause("SIGSEGV", ["libpython: PyEval"])


def test_infer_cause_abort():
    assert "aborted" in cd._infer_cause("SIGABRT", ["libc++abi: __abort_message"])


def test_infer_cause_oom_killer():
    assert "killed" in cd._infer_cause("SIGKILL", [])


def test_recent_crashes_returns_list_and_never_raises():
    # A bogus filter yields no matches on any platform; the call must still return a list,
    # not raise, even when the platform's crash tools are missing.
    out = cd.recent_crashes(name_filter="no-such-process-zzz", since_hours=1)
    assert isinstance(out, list)


def test_summarize_format():
    rec = {
        "when": 0, "process": "python", "signal": "SIGABRT",
        "cause": "aborted (SIGABRT)", "backtrace": ["a.dylib: foo", "b.dylib: bar"],
    }
    summary = cd.summarize(rec)
    assert "python (SIGABRT)" in summary
    assert "aborted" in summary
    assert "a.dylib: foo" in summary  # backtrace surfaced


def test_summarize_no_backtrace_is_single_line():
    rec = {"when": 0, "process": "node", "signal": "WER", "cause": "application fault", "backtrace": []}
    assert "\n" not in cd.summarize(rec)


def test_os_kind_is_known_bucket():
    assert cd.os_kind() in ("macos", "linux", "windows", "unknown")


def test_doctor_section_returns_list():
    assert isinstance(cd.doctor_section(), list)


# ── restart_notice: the suffix appended to the gateway "back online" message ──

def test_restart_notice_empty_on_clean_restart(monkeypatch):
    # No recent crash record (clean shutdown) → no suffix.
    monkeypatch.setattr(cd, "recent_crashes", lambda **kw: [])
    assert cd.restart_notice() == ""


def test_restart_notice_names_the_cause_on_unclean_restart(monkeypatch):
    monkeypatch.setattr(cd, "recent_crashes", lambda **kw: [
        {"process": "python", "signal": "SIGABRT",
         "cause": "GPU error (Metal/MLX) → SIGABRT (often GPU memory pressure / OOM)"}
    ])
    notice = cd.restart_notice()
    assert notice.startswith("\n\n⚠️")
    assert "previous run ended unexpectedly" in notice
    assert "python (SIGABRT)" in notice
    assert "GPU error (Metal/MLX)" in notice


def test_restart_notice_swallows_errors(monkeypatch):
    # Must never raise into the restart/notification path.
    def boom(**kw):
        raise RuntimeError("crash reader blew up")
    monkeypatch.setattr(cd, "recent_crashes", boom)
    assert cd.restart_notice() == ""
