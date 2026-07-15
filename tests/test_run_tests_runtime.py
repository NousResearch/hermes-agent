"""Regression tests for the canonical runner's linked-worktree contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_tests.sh"


def _make_probe(
    tmp_path: Path, *, should_fail: bool = False
) -> tuple[Path, Path]:
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    output = tmp_path / "probe-result.json"
    failure = "\n    assert False, 'intentional runner failure'" if should_fail else ""
    probe = probe_dir / "test_runtime_probe.py"
    probe.write_text(
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "import hermes_cli\n\n"
        "def test_probe():\n"
        "    payload = {\n"
        "        'hermes_cli': str(Path(hermes_cli.__file__).resolve()),\n"
        "        'sys_path': [str(Path(item).resolve()) for item in sys.path if item],\n"
        "        'pycache_prefix': sys.pycache_prefix,\n"
        "        'pycache_env': os.environ.get('PYTHONPYCACHEPREFIX'),\n"
        "        'pythonpath': os.environ.get('PYTHONPATH'),\n"
        "        'python_no_user_site': os.environ.get('PYTHONNOUSERSITE'),\n"
        "    }\n"
        f"    Path({str(output)!r}).write_text(json.dumps(payload))\n"
        f"{failure}\n"
    )
    return probe, output


def _make_editable_primary(tmp_path: Path) -> Path:
    """Create a disposable primary-checkout stand-in; never touch real Hermes."""
    primary = tmp_path / "editable-primary"
    scripts = primary / "scripts"
    package = primary / "hermes_cli"
    scripts.mkdir(parents=True)
    package.mkdir()
    shutil.copy2(REPO_ROOT / "scripts" / "run_tests.sh", scripts / "run_tests.sh")
    shutil.copy2(
        REPO_ROOT / "scripts" / "run_tests_parallel.py",
        scripts / "run_tests_parallel.py",
    )
    (package / "__init__.py").write_text("PRIMARY_SENTINEL = True\n")
    return primary


def _run_runner(
    probe: Path,
    tmp_path: Path,
    *,
    executable: Path = RUNNER,
    inherited_pythonpath: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "HERMES_TEST_WORKERS": "1",
            "TMPDIR": str(tmp_path),
            # This deliberately points at a disposable primary stand-in. The
            # canonical runner must not trust inherited import configuration.
            "PYTHONPATH": str(inherited_pythonpath or tmp_path / "poison"),
        }
    )
    return subprocess.run(
        [str(executable), "--files", str(probe), "-q"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=90,
    )


def _read_probe(output: Path) -> dict[str, Any]:
    assert output.exists(), f"probe did not write result: {output}"
    return json.loads(output.read_text())


def test_runner_uses_current_worktree_and_safe_runtime_paths(tmp_path: Path) -> None:
    """A linked-worktree cwd wins over a primary/editable runner path."""
    probe, output = _make_probe(tmp_path)
    primary = _make_editable_primary(tmp_path)

    # Invoke the disposable primary runner while cwd is the actual linked
    # worktree. This reproduces an editable/primary path accidentally owning
    # repo-root resolution without ever mutating the real primary checkout.
    proc = _run_runner(
        probe,
        tmp_path,
        executable=primary / "scripts" / "run_tests.sh",
        inherited_pythonpath=primary,
    )

    assert proc.returncode == 0, proc.stdout
    payload = _read_probe(output)
    hermes_cli_path = Path(payload["hermes_cli"])
    assert hermes_cli_path.is_relative_to(REPO_ROOT), payload
    assert not hermes_cli_path.is_relative_to(primary), payload
    assert str(REPO_ROOT) in payload["sys_path"]
    pythonpath_entries = payload["pythonpath"].split(os.pathsep)
    assert pythonpath_entries[0] == str(REPO_ROOT)
    assert str(primary) not in pythonpath_entries
    assert payload["python_no_user_site"] == "1"
    assert payload["pycache_env"]
    assert payload["pycache_prefix"] == payload["pycache_env"]
    assert Path(payload["pycache_prefix"]).is_relative_to(tmp_path)

    # Duration state belongs to the actual linked worktree, never the
    # disposable primary/editable source path. The latter must stay clean.
    assert (REPO_ROOT / "test_durations.json").exists()
    assert not (primary / "test_durations.json").exists()
    assert not list(primary.rglob("*.pyc"))


def test_runner_preserves_nonzero_failure_status(tmp_path: Path) -> None:
    probe, _output = _make_probe(tmp_path, should_fail=True)
    proc = _run_runner(probe, tmp_path)

    assert proc.returncode != 0, proc.stdout
    assert "intentional runner failure" in proc.stdout


def test_runner_success_status_is_zero(tmp_path: Path) -> None:
    probe, _output = _make_probe(tmp_path)
    proc = _run_runner(probe, tmp_path)

    assert proc.returncode == 0, proc.stdout
