"""Cron script jobs on native Windows (``hermes cron create --script`` + ``hermes cron run``).

A no-agent script job runs its script through an interpreter picked by extension and
delivers stdout as the job result. ``hermes cron run <id>`` executes it synchronously
through ``cron.scheduler.run_job`` (the ticker's code path) when no gateway owns the store.

The ``.sh`` case launches Hermes with the PATH a NATIVE Windows process has (Start menu,
Scheduled Task, Desktop): Git's ``cmd`` dir only, no ``Git\\bin`` / ``Git\\usr\\bin``, so
the only ``bash`` a bare PATH lookup finds is the WSL stub in System32 (or none). Git for
Windows is installed at its standard location the whole time, which is exactly the host
the canonical resolver (``tools/environments/local.py::_find_bash``) handles.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pytest

from tests.e2e.core.windows._helpers import WinHome, expect, hermes, known, make_home, nonce

pytestmark = [pytest.mark.windows_only, pytest.mark.integration]

KNOWN: dict[str, str] = {
    "sh_script": "#120504 cron .sh scripts resolve bash via bare PATH lookup (WSL stub), not Git Bash",
}

_JOB_ID = re.compile(r"Created job: (\S+)")


def _native_process_path() -> str:
    """PATH minus every directory that ships a real bash.exe (Git\\bin, Git\\usr\\bin, MSYS2...);
    System32 (with its WSL stub, when the feature is present) stays, as for a native process."""
    kept = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        norm = os.path.normcase(os.path.normpath(entry)) if entry else ""
        if not entry:
            continue
        stub_dir = "system32" in norm or "windowsapps" in norm
        if not stub_dir and os.path.isfile(os.path.join(entry, "bash.exe")):
            continue
        kept.append(entry)
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    if not any(os.path.normcase(os.path.normpath(e)) == os.path.normcase(system32) for e in kept):
        kept.append(system32)
    return os.pathsep.join(kept)


def _run_script_job(home: WinHome, script: str, env_extra: dict[str, str] | None = None) -> tuple[dict, str]:
    created = hermes(home, "cron", "create", "1d", "--name", "win-script", "--script", script,
                     "--no-agent", "--deliver", "local", cwd=home.profile, env_extra=env_extra)
    match = _JOB_ID.search(created.stdout)
    assert created.returncode == 0 and match, created.tail()
    job_id = match.group(1)
    ran = hermes(home, "cron", "run", job_id, cwd=home.profile, env_extra=env_extra)
    assert ran.returncode == 0 and "Ran now:" in ran.stdout, f"job did not run synchronously:\n{ran.tail()}"
    jobs = json.loads((home.hermes_home / "cron" / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    job = next(j for j in jobs if j["id"] == job_id)
    outputs = sorted((home.hermes_home / "cron" / "output" / job_id).glob("*.md"))
    return job, outputs[-1].read_text(encoding="utf-8") if outputs else ""


def test_python_script_job_delivers_stdout(tmp_path: Path) -> None:
    marker = nonce("PYJOB")
    home = make_home(tmp_path, "http://127.0.0.1:9/v1")  # no-agent: the model is never called
    scripts = home.hermes_home / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "report.py").write_text(f"import sys\nprint('{marker}', sys.platform)\n", encoding="utf-8")
    job, output = _run_script_job(home, "report.py")
    assert job["last_status"] == "ok", f"python script job failed: {job.get('last_error')}\n{output}"
    assert f"{marker} win32" in output, f"script stdout not delivered:\n{output}"


@known("sh_script", KNOWN)
def test_sh_script_job_runs_under_git_bash(tmp_path: Path) -> None:
    git_bash = Path(os.environ.get("ProgramFiles", r"C:\Program Files"), "Git", "bin", "bash.exe")
    assert git_bash.is_file(), f"precondition: Git for Windows installed at {git_bash}"
    marker = nonce("SHJOB")
    home = make_home(tmp_path, "http://127.0.0.1:9/v1")
    scripts = home.hermes_home / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "report.sh").write_bytes(f'#!/usr/bin/env bash\necho "{marker}" "$(uname -s)"\n'.encode())
    native_path = _native_process_path()
    job, output = _run_script_job(home, "report.sh", env_extra={"PATH": native_path})
    bare = shutil.which("bash", path=native_path)
    expect(job["last_status"] == "ok" and marker in output,
           f"bare PATH lookup finds {bare!r}; job status {job['last_status']!r}: {job.get('last_error')}\n{output}")
    assert "_NT-" in output, f".sh job ran, but not under Git Bash (MSYS/MinGW uname):\n{output}"
