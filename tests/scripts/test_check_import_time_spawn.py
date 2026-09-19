"""Behavioral tests for the import-time process-spawn CI guard.

The invariant under test: a spawn that runs while pytest is *collecting* is
flagged, and a spawn that runs when a test or fixture body runs is not. The
distinction is not "does a `def` enclose it" -- a decorator, a default argument
and an annotation all sit inside a `FunctionDef` node yet are evaluated where
the `def` sits.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check_import_time_spawn.py"
)


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def _write(root: Path, name: str, body: str) -> Path:
    path = root / name
    path.write_text(body, encoding="utf-8")
    return path


def test_empty_tree_passes(tmp_path):
    assert _run(tmp_path).returncode == 0


def test_skipif_decorator_probe_is_flagged(tmp_path):
    """The real #94383/#111048 shape: a class-method decorator, not module scope."""
    _write(
        tmp_path,
        "test_probe.py",
        "import subprocess\n"
        "import pytest\n"
        "\n"
        "\n"
        "class TestThing:\n"
        "    @pytest.mark.skipif(\n"
        '        subprocess.run(["which", "rg"], capture_output=True).returncode != 0,\n'
        '        reason="ripgrep not installed",\n'
        "    )\n"
        "    def test_it(self):\n"
        "        pass\n",
    )

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "test_probe.py:7" in result.stdout
    assert "subprocess.run() runs at import" in result.stdout


def test_module_and_class_scope_spawns_are_flagged(tmp_path):
    _write(
        tmp_path,
        "test_scopes.py",
        "import os\n"
        "import subprocess\n"
        "\n"
        'HAVE_GIT = subprocess.call(["git", "--version"]) == 0\n'
        "\n"
        "\n"
        "class TestThing:\n"
        '    OUT = subprocess.check_output(["uname"])\n'
        "\n"
        "    def test_it(self):\n"
        "        pass\n"
        "\n"
        "\n"
        'RC = os.system("true")\n',
    )

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "test_scopes.py:4" in result.stdout
    assert "test_scopes.py:8" in result.stdout
    assert "os.system() runs at import" in result.stdout


def test_default_argument_and_annotation_probes_are_flagged(tmp_path):
    """Both are evaluated at `def` time, so both abort collection."""
    _write(
        tmp_path,
        "test_signature.py",
        "import subprocess\n"
        "\n"
        "\n"
        'def test_default(out=subprocess.check_output(["uname"])):\n'
        "    pass\n",
    )

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "test_signature.py:4" in result.stdout


def test_spawn_inside_a_body_is_not_flagged(tmp_path):
    """Collection has already succeeded by then; a raise is one contained failure."""
    _write(
        tmp_path,
        "test_bodies.py",
        "import subprocess\n"
        "\n"
        "import pytest\n"
        "\n"
        "\n"
        "@pytest.fixture\n"
        "def repo(tmp_path):\n"
        '    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)\n'
        "    return tmp_path\n"
        "\n"
        "\n"
        "def test_it(repo):\n"
        '    assert subprocess.run(["git", "status"], cwd=repo).returncode == 0\n'
        "\n"
        "\n"
        "def _later():\n"
        '    return lambda: subprocess.check_output(["uname"])\n',
    )

    assert _run(tmp_path).returncode == 0


def test_total_probe_is_not_flagged(tmp_path):
    """The prescribed replacement must pass, or the message sends people nowhere."""
    _write(
        tmp_path,
        "test_total.py",
        "import shutil\n"
        "\n"
        "import pytest\n"
        "\n"
        'RG = shutil.which("rg")\n'
        "\n"
        "\n"
        '@pytest.mark.skipif(RG is None, reason="ripgrep not installed")\n'
        "def test_it():\n"
        "    pass\n",
    )

    assert _run(tmp_path).returncode == 0


def test_opt_out_comment_silences_one_call(tmp_path):
    _write(
        tmp_path,
        "test_optout.py",
        "import subprocess\n"
        "\n"
        'CHEAP = subprocess.run(["true"]).returncode  # import-time-exec: ok -- cannot raise\n'
        'LOUD = subprocess.run(["which", "rg"]).returncode\n',
    )

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "test_optout.py:4" in result.stdout
    assert "test_optout.py:3" not in result.stdout


def test_non_test_files_are_ignored_but_conftest_is_not(tmp_path):
    _write(tmp_path, "helper.py", 'import subprocess\nX = subprocess.run(["true"])\n')
    assert _run(tmp_path).returncode == 0

    _write(tmp_path, "conftest.py", 'import subprocess\nX = subprocess.run(["true"])\n')
    result = _run(tmp_path)

    assert result.returncode == 1
    assert "conftest.py:2" in result.stdout


def test_unparseable_file_is_skipped_not_fatal(tmp_path):
    _write(tmp_path, "test_broken.py", "def (:\n")

    assert _run(tmp_path).returncode == 0


def test_missing_root_is_a_usage_error(tmp_path):
    result = _run(tmp_path / "nope")

    assert result.returncode == 2
    assert "no such directory" in result.stderr


@pytest.mark.parametrize("subdir", ["tools", "nested/deeper"])
def test_walks_subdirectories(tmp_path, subdir):
    target = tmp_path / subdir
    target.mkdir(parents=True)
    _write(target, "test_deep.py", 'import subprocess\nX = subprocess.run(["which", "rg"])\n')

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "test_deep.py:2" in result.stdout


def test_real_test_tree_is_clean():
    """The tree this guard was written for must be green, or the guard is noise."""
    repo_root = Path(__file__).resolve().parents[2]

    result = _run(repo_root / "tests")

    assert result.returncode == 0, result.stdout
