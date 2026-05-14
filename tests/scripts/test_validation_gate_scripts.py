from __future__ import annotations

import importlib.util
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_windows_footguns_module():
    module_path = REPO_ROOT / "scripts" / "check-windows-footguns.py"
    spec = importlib.util.spec_from_file_location("check_windows_footguns", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_windows_footgun_checker_exits_nonzero_for_bare_open(tmp_path):
    module = _load_windows_footguns_module()
    target = tmp_path / "bad.py"
    target.write_text("with open('notes.txt') as f:\n    print(f.read())\n", encoding="utf-8")

    assert module.main([str(target)]) == 1


def test_windows_footgun_checker_allows_explicit_encoding(tmp_path):
    module = _load_windows_footguns_module()
    target = tmp_path / "good.py"
    target.write_text(
        "with open('notes.txt', encoding='utf-8') as f:\n    print(f.read())\n",
        encoding="utf-8",
    )

    assert module.main([str(target)]) == 0


def test_check_sh_rejects_whitespace_errors_in_committed_diff(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)

    scripts_dir = repo / "scripts"
    scripts_dir.mkdir()
    shutil.copy2(REPO_ROOT / "scripts" / "check.sh", scripts_dir / "check.sh")
    (scripts_dir / "run_tests.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (scripts_dir / "check-windows-footguns.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    for script in scripts_dir.iterdir():
        script.chmod(script.stat().st_mode | stat.S_IXUSR)

    venv_bin = repo / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "activate").write_text("", encoding="utf-8")
    python_wrapper = venv_bin / "python"
    python_wrapper.write_text(
        f"""#!/usr/bin/env python3
import os
import sys

if sys.argv[1:3] == ["-m", "ruff"]:
    raise SystemExit(0)

os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
""",
        encoding="utf-8",
    )
    python_wrapper.chmod(python_wrapper.stat().st_mode | stat.S_IXUSR)

    bin_dir = repo / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)

    good_file = repo / "example.txt"
    good_file.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()

    subprocess.run(["git", "checkout", "-qb", "feature"], cwd=repo, check=True)
    good_file.write_text("bad trailing whitespace \n", encoding="utf-8")
    subprocess.run(["git", "add", "example.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "add whitespace error"], cwd=repo, check=True)

    env = {
        **os.environ,
        "HERMES_CHECK_BASE_REF": base,
        "HERMES_CHECK_PYTHON": f"{sys.version_info.major}.{sys.version_info.minor}",
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
    }
    result = subprocess.run(
        ["bash", "scripts/check.sh"],
        cwd=repo,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode != 0
    assert "git diff --check" in result.stdout
    assert "trailing whitespace" in result.stdout
