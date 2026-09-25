"""install.ps1 native calls survive Windows PowerShell 5.1, and reruns keep local work.

Windows PowerShell 5.1 turns a native command's redirected stderr into an
ErrorRecord that terminates a script running under
``$ErrorActionPreference = "Stop"``. These run the real installer under
``powershell.exe`` (5.1) -- dot-sourced for its helpers, or as the real
``-Stage repository -Json`` entry against local git repositories.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.platforms("windows")
INSTALLER = Path(__file__).resolve().parents[3] / "scripts" / "install.ps1"


def _powershell() -> str:
    powershell = shutil.which("powershell")
    assert powershell, "Windows PowerShell 5.1 is part of every supported Windows"
    return powershell


def _dot_sourced(body: str) -> subprocess.CompletedProcess:
    script = f'$ErrorActionPreference = "Stop"; . "{INSTALLER}"; {body}'
    return subprocess.run([_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                          capture_output=True, text=True, timeout=120)


def test_native_stderr_under_stop_leaves_the_exit_code_to_the_caller():
    result = _dot_sourced('Invoke-Native { cmd /c "echo soft-failure 1>&2 & exit 3" 2>$null }; '
                          '"exit=$LASTEXITCODE preference=$ErrorActionPreference"')
    assert result.returncode == 0, result.stdout + result.stderr
    assert "exit=3 preference=Stop" in result.stdout


def test_path_uv_must_run_and_meet_the_pin(tmp_path):
    for name, body in {"old": "@echo uv 0.6.17", "new": "@echo uv 99.0.0 (abc 2099-01-01)",
                       "broken": "@echo boom 1>&2 & exit /b 1"}.items():
        (tmp_path / f"{name}.cmd").write_text(body + "\r\n")
    checks = "; ".join(f'"{n}=$(Test-UvAtLeastPin \'{tmp_path / (n + ".cmd")}\')"' for n in ("old", "new", "broken"))
    result = _dot_sourced(checks)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.split() == ["old=False", "new=True", "broken=False"]


def test_uv_find_output_survives_a_non_utf8_console_code_page(tmp_path):
    """`uv python find` answers in UTF-8; a captured native command's output is decoded with the
    console's OEM code page (gb2312 here) unless the capture runs under a UTF-8 window, so a
    non-ASCII install path arrived mangled and the dependency stage failed."""
    emit = tmp_path / "emit.py"
    emit.write_text('import sys\nsys.stdout.write("C:/Users/\\u5f20\\u4e09/temp/file.txt")\n', encoding="utf-8")
    result_file = tmp_path / "result.txt"
    body = (f"$py = '{sys.executable}'; $emit = '{emit}'; $mark = [string][char]0x5f20; "
            "$ambient = [Console]::OutputEncoding.WebName; "
            "$raw = ((& $py -X utf8 $emit) -join ' ').Trim(); "
            "$fixed = ((Invoke-NativeUtf8 { & $py -X utf8 $emit }) -join ' ').Trim(); "
            "foreach ($v in @($ambient, $raw.Contains($mark), $fixed.Contains($mark))) "
            f"{{ Add-Content -Path '{result_file}' -Value $v }}")
    result = _dot_sourced(body)
    assert result.returncode == 0, result.stdout + result.stderr
    ambient, raw_has, fixed_has = result_file.read_text(encoding="utf-8").split()
    assert fixed_has == "True", f"ambient={ambient}: Invoke-NativeUtf8 mangled the path"
    if ambient != "utf-8":
        assert raw_has == "False", f"ambient={ambient}: the plain capture stopped mangling"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def _stage(origin: Path, home: Path, *extra: str) -> tuple[subprocess.CompletedProcess, dict]:
    env = dict(os.environ, HERMES_REPO_URL=str(origin), HERMES_HOME=str(home))
    result = subprocess.run([_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(INSTALLER),
                             "-Stage", "repository", "-Json", *extra],
                            env=env, capture_output=True, text=True, timeout=180)
    frames = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(frames) == 1, result.stdout + result.stderr
    return result, frames[0]


def test_rerun_parks_local_work_and_pins_only_branch_commits(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    (origin / "README").write_text("one")
    _git(origin, "add", "README")
    _git(origin, "commit", "-qm", "one")
    home = tmp_path / "home"
    install = home / "hermes-agent"
    assert _stage(origin, home)[1]["ok"] is True
    (install / "README").write_text("local edit")
    (origin / "README").write_text("two")
    _git(origin, "commit", "-qam", "two")
    _git(origin, "checkout", "-qb", "side")
    (origin / "README").write_text("side")
    _git(origin, "commit", "-qam", "side")
    off_branch = _git(origin, "rev-parse", "HEAD")
    _git(origin, "checkout", "-q", "main")

    result, frame = _stage(origin, home)
    assert frame["ok"] is True, result.stdout + result.stderr
    assert (install / "README").read_text() == "two"
    assert "local edit" in _git(install, "stash", "show", "-p", "stash@{0}")

    result, frame = _stage(origin, home, "-Commit", off_branch)
    assert frame["ok"] is False
    assert "is not on branch main" in frame["reason"]
