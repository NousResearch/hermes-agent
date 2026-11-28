"""A stdout-less compose probe refuses instead of permitting the build.

The probe runs with ``capture_output=True, encoding="utf-8",
errors="replace"``, so a naturally ``None`` stdout is not reachable through
real pipes — this is fault injection (leonininder's harness shape, #124990
review), pinning the refusal contract if the capture contract ever changes:
a missing capture is UNKNOWN state, and an empty names list is what
PERMITS the mutating build phases, so ``None`` must refuse, never coerce.
POSIX-only: the injected probe never spawns, and the marker-writing build
avoids Windows command-string quoting for no extra contract coverage.
"""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from agent.verify.recipes import Recipe
from agent.verify.runner import run_verify


def _inject_probe(monkeypatch, *, stdout, returncode=0, stderr=""):
    """Intercept only the ``docker compose ps`` probe; every other subprocess
    (the build phase itself) runs for real, so the build marker is an honest
    witness of whether the mutating phases were permitted."""
    import agent.verify.runner as runner

    real_run = subprocess.run

    def fake_run(argv, **kwargs):
        if list(argv[:3]) == ["docker", "compose", "ps"]:
            return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr=stderr)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)


def _recipe_with_marker_build(marker: Path) -> Recipe:
    code = f"from pathlib import Path; Path({str(marker)!r}).write_text('built')"
    return Recipe(name="probe-none fixture", kind="compose",
                  build=[f"{sys.executable} -c {shlex.quote(code)}"])


@pytest.mark.platforms("posix")
def test_probe_stdout_none_refuses_mutating_build(tmp_path, monkeypatch):
    marker = tmp_path / "build-ran"
    _inject_probe(monkeypatch, stdout=None)

    result = run_verify(tmp_path, _recipe_with_marker_build(marker),
                        phases=("build",), skip_start=True)

    assert not result.ok
    assert len(result.phases) == 1 and result.phases[0].exit_code == 1
    refusal = result.phases[0].output_tail
    assert "Refusing to run" in refusal
    assert "no captured stdout" in refusal
    assert "live containers cannot be ruled out" in refusal
    assert not marker.exists(), "None stdout must refuse, never fall through to the build"


@pytest.mark.platforms("posix")
def test_probe_known_empty_stdout_still_permits_build(tmp_path, monkeypatch):
    marker = tmp_path / "build-ran"
    _inject_probe(monkeypatch, stdout="")

    result = run_verify(tmp_path, _recipe_with_marker_build(marker),
                        phases=("build",), skip_start=True)

    assert result.ok, result.phases and result.phases[0].output_tail
    assert marker.exists(), "a successfully captured EMPTY probe output means no live containers"
