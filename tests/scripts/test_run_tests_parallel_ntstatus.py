"""Windows NTSTATUS exits are interpreter crashes, not "no tests ran" (#137198).

The runner's crash classifier keyed on faulthandler banners and POSIX signals
only, so a loader-failure exit (positive NTSTATUS rc, no output) fell through
to the collection-error bucket while the CRASHED section stayed empty.

The rc values are plain integers here, so the classification is pinned on every
host; the real-exit end-to-end test lives in test_run_tests_parallel.py
(windows-marked, since POSIX exit codes are mod 256 and cannot carry NTSTATUS).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_PATH = REPO_ROOT / "scripts" / "run_tests_parallel.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_tests_parallel", _RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ntstatus_no_output_exit_is_described_as_a_crash():
    mod = _load_runner()
    dll_init_failed = mod._describe_interpreter_crash(0xC0000142, "")
    assert dll_init_failed is not None
    assert "0xc0000142" in dll_init_failed
    assert "STATUS_DLL_INIT_FAILED" in dll_init_failed

    commitment_limit = mod._describe_interpreter_crash(0xC000012D, "")
    assert commitment_limit is not None
    assert "STATUS_COMMITMENT_LIMIT" in commitment_limit

    # A sign-wrapped DWORD (0xC000012D as a C int) must classify identically,
    # not fall into the POSIX-signal branch.
    signed = mod._describe_interpreter_crash(0xC000012D - (1 << 32), "")
    assert signed is not None and "STATUS_COMMITMENT_LIMIT" in signed, signed

    # An unrecognized NTSTATUS is still a fault exit, never a collection error.
    unknown = mod._describe_interpreter_crash(0xC000013A, "")
    assert unknown is not None and "0xc000013a" in unknown


def test_plain_nonzero_exit_stays_out_of_the_crash_classifier():
    """pytest's own failure exits keep their current bucket — the classifier
    only widens to fault-shaped codes, not to every nonzero rc."""
    mod = _load_runner()
    for rc in (1, 2, 3, 4, 5, 124, 255):
        assert mod._describe_interpreter_crash(rc, "") is None, rc
    # A real POSIX signal is a small negative; 0xFFFFFFF5 also carries the
    # severity bits, so the classifier must separate them by magnitude.
    sigsegv = mod._describe_interpreter_crash(-11, "")
    assert sigsegv is not None and "SIGSEGV" in sigsegv, sigsegv
    # A hex code quoted in captured text is not an exit status: classify rc,
    # never the output.
    embedded = mod._describe_interpreter_crash(1, "bash dofork: 0xc0000142\n1 failed in 0.01s")
    assert embedded is None


def test_ntstatus_child_counts_as_crashed_not_no_ran(tmp_path, monkeypatch):
    """End-to-end through _run_one_file_once: the silent NTSTATUS child is
    diagnosed as crashed, not left for the no-tests-ran bucket."""
    mod = _load_runner()
    child = SimpleNamespace(
        pid=123, returncode=0xC0000142, communicate=lambda **kwargs: ("", None),
    )
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(mod, "_kill_tree", lambda *args, **kwargs: None)

    _, code, output, counts, _ = mod._run_one_file_once(tmp_path / "test.py", [], tmp_path, 30)

    assert code == 0xC0000142
    assert counts.get("crashed") == 1, f"expected crashed bucket, got {counts}"
    assert output.startswith("(interpreter crashed:")
    assert "0xc0000142" in output
