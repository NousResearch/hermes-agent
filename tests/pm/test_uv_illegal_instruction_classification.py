"""uv cannot run at all on some CPUs: a build compiled against a newer x86-64
baseline dies with EXCEPTION_ILLEGAL_INSTRUCTION (0xC000001D) before printing
anything (#72518).

Every install step shells out to that same binary -- venv creation, the resolve,
`uv pip install`, `uv pip check` -- so the failure has to be recognized from the
exit code alone and explained in terms the user can act on, instead of leaking a
bare NTSTATUS through the generic uv error.
"""

from __future__ import annotations

import subprocess

import pytest

from pm.environment import (
    UV_UNSUPPORTED_CPU_REMEDY,
    BuildFailure,
    PythonEnvironment,
    ResolutionConflict,
    UvUnsupportedCpu,
    classify_uv_failure,
)
from pm.package import InstallError

# Windows spells the same NTSTATUS two ways depending on the caller.
SIGNED = -1073741795
UNSIGNED = 3221225501


@pytest.mark.parametrize("code", [SIGNED, UNSIGNED])
def test_illegal_instruction_is_classified_by_exit_code(code):
    """A crash prints nothing, so classification cannot depend on output text."""
    error = classify_uv_failure("sync", code, "")

    assert isinstance(error, UvUnsupportedCpu)
    assert isinstance(error, InstallError)
    assert "EXCEPTION_ILLEGAL_INSTRUCTION" in error.cause
    assert "0xC000001D" in error.cause
    assert error.remedy == UV_UNSUPPORTED_CPU_REMEDY


@pytest.mark.parametrize("code", [SIGNED, UNSIGNED])
def test_illegal_instruction_survives_a_verbose_tail(code):
    """Streamed verbose output must not push the verdict back to a generic error."""
    assert isinstance(classify_uv_failure("sync", code, "N" * 4000), UvUnsupportedCpu)


def test_access_violation_stays_a_plain_install_error():
    """Only EXCEPTION_ILLEGAL_INSTRUCTION is a CPU-capability verdict.

    -1073741819 is 0xC0000005, an access violation: a different crash that must
    not end up telling the user to replace their CPU.
    """
    error = classify_uv_failure("sync", -1073741819, "")

    assert not isinstance(error, UvUnsupportedCpu)
    assert type(error) is InstallError


def test_existing_output_markers_are_not_shadowed():
    """The new exit-code branch runs first, but only matches its own two codes."""
    assert isinstance(
        classify_uv_failure("lock", 1, "error: No solution found for the requested dependencies"),
        ResolutionConflict,
    )
    assert isinstance(
        classify_uv_failure("sync", 1, "error: Failed to build; the build backend returned an error"),
        BuildFailure,
    )


def _env(tmp_path) -> PythonEnvironment:
    return PythonEnvironment(
        uv=tmp_path / "uv",
        python=tmp_path / "python",
        destination=tmp_path / "venv",
        cache=tmp_path / "cache",
        env={},
    )


def test_sync_surfaces_the_cpu_verdict(tmp_path, monkeypatch):
    def fake_run(self, args, *, cwd, timeout):
        return subprocess.CompletedProcess(args, SIGNED, stdout="", stderr="")

    monkeypatch.setattr(PythonEnvironment, "_run", fake_run)
    with pytest.raises(UvUnsupportedCpu):
        _env(tmp_path).sync(tmp_path)


def test_venv_creation_surfaces_the_cpu_verdict(tmp_path, monkeypatch):
    """`uv venv` is the first uv call on a fresh install, so it needs the same
    verdict -- otherwise the user sees `uv venv failed: ` with an empty cause."""
    def fake_run(self, args, *, cwd, timeout):
        return subprocess.CompletedProcess(args, SIGNED, stdout="", stderr="")

    monkeypatch.setattr(PythonEnvironment, "_run", fake_run)
    with pytest.raises(UvUnsupportedCpu):
        _env(tmp_path).create()


def test_venv_creation_keeps_its_message_for_other_failures(tmp_path, monkeypatch):
    def fake_run(self, args, *, cwd, timeout):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="permission denied")

    monkeypatch.setattr(PythonEnvironment, "_run", fake_run)
    with pytest.raises(InstallError) as caught:
        _env(tmp_path).create()
    assert not isinstance(caught.value, UvUnsupportedCpu)
    assert "uv venv failed" in caught.value.cause
