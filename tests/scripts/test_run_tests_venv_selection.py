"""Behavioral coverage for scripts/run_tests.sh venv selection.

Regression context: the canonical runner used to accept the first candidate
venv with a ``bin/activate`` file. In checkouts where ``.venv`` exists but
has no pytest (created without pip, or site-packages pruned) the runner
selected it and every test file died with ``No module named pytest`` —
blocking the canonical suite even when a later candidate (``venv``) was
fully usable. Candidates are now import-checked for pytest (the same guard
the HERMES_PYTHON fallback always applied) and skipped candidates are named
on stderr.

These tests pin the selection contract end-to-end by executing the real
``scripts/run_tests.sh`` in a disposable fake repo root:

* a pytest-less candidate is skipped and the next candidate is selected
* the documented candidate order (.venv, venv, $HOME fallback) is preserved
* the $HOME fallback candidate is used when local venvs are unusable
* HERMES_PYTHON is only used when no local candidate is usable
* with no usable venv anywhere the script exits 1 with an accurate error

POSIX-only: drives bash, chmod, and /bin/sh shims.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_TESTS_SH = REPO_ROOT / "scripts" / "run_tests.sh"

# The stub stands in for scripts/run_tests_parallel.py: instead of running
# the suite it reports which interpreter the wrapper selected, via a marker
# env var each fake "good" venv shim exports before exec'ing real python.
_STUB_RUNNER = textwrap.dedent(
    """
    import os
    print(f"SELECTED_MARKER={os.environ.get('HERMES_TEST_VENV_MARKER', '')}")
    """
).strip()

# Shim for a USABLE candidate venv: forwards everything to the real
# interpreter (so `-c 'import pytest'` succeeds and the stub runner
# executes), tagging the environment so the stub can report which venv won.
_GOOD_PYTHON_SHIM = """\
#!/bin/sh
export HERMES_TEST_VENV_MARKER={marker}
exec {real_python} "$@"
"""

# Shim for an UNUSABLE candidate venv: any invocation fails, so the
# `import pytest` probe rejects it.
_BAD_PYTHON_SHIM = """\
#!/bin/sh
exit 1
"""


def _make_venv(root: Path, name: str, *, usable: bool) -> Path:
    """Create a fake venv at ``root/name`` with bin/activate + bin/python."""
    venv = root / name
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text("# fake activate\n")
    python_shim = venv / "bin" / "python"
    if usable:
        python_shim.write_text(
            _GOOD_PYTHON_SHIM.format(marker=name, real_python=sys.executable)
        )
    else:
        python_shim.write_text(_BAD_PYTHON_SHIM)
    python_shim.chmod(0o755)
    return venv


def _make_fake_repo(tmp_path: Path) -> Path:
    """Fake repo root with the real run_tests.sh and a stub parallel runner."""
    fake_root = tmp_path / "repo"
    (fake_root / "scripts").mkdir(parents=True)
    shutil.copy(RUN_TESTS_SH, fake_root / "scripts" / "run_tests.sh")
    (fake_root / "scripts" / "run_tests_parallel.py").write_text(_STUB_RUNNER + "\n")
    return fake_root


def _run_wrapper(fake_root: Path, home: Path) -> subprocess.CompletedProcess:
    env = {
        # Deliberately minimal: no HERMES_PYTHON unless a test sets it, no
        # credential vars, HOME pointed at a temp dir so the real
        # ~/.hermes/hermes-agent/venv candidate never leaks in.
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
    }
    return subprocess.run(
        ["bash", str(fake_root / "scripts" / "run_tests.sh")],
        cwd=fake_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )


def _selected_marker(proc: subprocess.CompletedProcess) -> str:
    for line in proc.stdout.splitlines():
        if line.startswith("SELECTED_MARKER="):
            return line.split("=", 1)[1]
    raise AssertionError(
        f"stub runner never reported a selection; "
        f"rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only bash/sh shims")
@pytest.mark.live_system_guard_bypass
def test_venv_without_pytest_is_skipped_for_next_candidate(tmp_path: Path) -> None:
    fake_root = _make_fake_repo(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    _make_venv(fake_root, ".venv", usable=False)
    _make_venv(fake_root, "venv", usable=True)

    proc = _run_wrapper(fake_root, home)

    assert proc.returncode == 0, proc.stderr
    assert _selected_marker(proc) == "venv"
    # The skipped candidate must be named on stderr so the skip is visible.
    assert "skipping venv without pytest" in proc.stderr
    assert str(fake_root / ".venv") in proc.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only bash/sh shims")
@pytest.mark.live_system_guard_bypass
def test_candidate_order_preserved_when_first_is_usable(tmp_path: Path) -> None:
    fake_root = _make_fake_repo(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    _make_venv(fake_root, ".venv", usable=True)
    _make_venv(fake_root, "venv", usable=True)

    proc = _run_wrapper(fake_root, home)

    assert proc.returncode == 0, proc.stderr
    assert _selected_marker(proc) == ".venv"
    assert "skipping venv without pytest" not in proc.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only bash/sh shims")
@pytest.mark.live_system_guard_bypass
def test_home_fallback_candidate_used_when_local_venvs_unusable(tmp_path: Path) -> None:
    fake_root = _make_fake_repo(tmp_path)
    home = tmp_path / "home"
    _make_venv(fake_root, ".venv", usable=False)
    _make_venv(home / ".hermes" / "hermes-agent", "venv", usable=True)

    proc = _run_wrapper(fake_root, home)

    assert proc.returncode == 0, proc.stderr
    assert _selected_marker(proc) == "venv"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only bash/sh shims")
@pytest.mark.live_system_guard_bypass
def test_hermes_python_used_only_when_no_local_candidate_usable(tmp_path: Path) -> None:
    fake_root = _make_fake_repo(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    _make_venv(fake_root, ".venv", usable=False)

    hermes_python = tmp_path / "hermes-python"
    hermes_python.write_text(
        _GOOD_PYTHON_SHIM.format(marker="hermes_python", real_python=sys.executable)
    )
    hermes_python.chmod(0o755)

    env_hermes = hermes_python  # alias for readability below
    proc = subprocess.run(
        ["bash", str(fake_root / "scripts" / "run_tests.sh")],
        cwd=fake_root,
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(home),
            "HERMES_PYTHON": str(env_hermes),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stderr
    assert _selected_marker(proc) == "hermes_python"
    assert "using Nix dev venv via HERMES_PYTHON" in proc.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only bash/sh shims")
@pytest.mark.live_system_guard_bypass
def test_no_usable_venv_exits_nonzero_with_accurate_error(tmp_path: Path) -> None:
    fake_root = _make_fake_repo(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    _make_venv(fake_root, ".venv", usable=False)

    proc = _run_wrapper(fake_root, home)

    assert proc.returncode == 1
    assert "no virtualenv with pytest found" in proc.stderr
    # The error must name what was skipped so the fix path is obvious.
    assert str(fake_root / ".venv") in proc.stderr
