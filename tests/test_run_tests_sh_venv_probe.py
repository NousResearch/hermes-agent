"""Regression tests for scripts/run_tests.sh virtualenv probing."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path


REQUIRED_TEST_IMPORTS = {"pytest", "croniter", "psutil", "pytest_asyncio"}


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _make_fake_python(venv_dir: Path, modules: set[str]) -> Path:
    bin_dir = venv_dir / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "activate").write_text("# fake activate\n", encoding="utf-8")
    (bin_dir / "modules.txt").write_text("\n".join(sorted(modules)), encoding="utf-8")
    python_bin = bin_dir / "python"
    _write_executable(
        python_bin,
        textwrap.dedent(
            f"""
            #!{sys.executable}
            from __future__ import annotations

            import sys
            from pathlib import Path

            modules = set((Path(__file__).with_name("modules.txt")).read_text().splitlines())
            args = sys.argv[1:]
            if args and args[0] == "-":
                sys.stdin.read()
                for module_name in args[1:]:
                    if module_name not in modules:
                        print(module_name)
                raise SystemExit(0)
            if args[:2] == ["-m", "compileall"]:
                raise SystemExit(0)
            if args and args[0].endswith("run_tests_parallel.py"):
                print(f"SELECTED:{{Path(__file__).as_posix()}}")
                raise SystemExit(0)
            raise SystemExit(f"unexpected fake-python args: {{args!r}}")
            """
        ).lstrip(),
    )
    return python_bin


def test_run_tests_sh_skips_drifted_venv_and_reports_missing_imports(
    tmp_path: Path,
) -> None:
    """A pytest-only venv must not launch the suite with missing deps."""
    repo_root = Path(__file__).resolve().parents[1]
    test_repo = tmp_path / "repo"
    scripts_dir = test_repo / "scripts"
    scripts_dir.mkdir(parents=True)
    run_tests = scripts_dir / "run_tests.sh"
    run_tests.write_text(
        (repo_root / "scripts" / "run_tests.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    _make_fake_python(test_repo / ".venv", {"pytest"})
    selected_python = _make_fake_python(test_repo / "venv", REQUIRED_TEST_IMPORTS)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "git", "#!/usr/bin/env sh\nexit 0\n")

    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"

    proc = subprocess.run(
        ["bash", str(run_tests), "tests/cron/test_due_stale_cron_edit.py"],
        cwd=test_repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout
    assert "skipping venv missing test imports:" in proc.stdout
    assert (
        f"{test_repo / '.venv'}(missing:croniter,psutil,pytest_asyncio)" in proc.stdout
    )
    assert f"SELECTED:{selected_python.as_posix()}" in proc.stdout
