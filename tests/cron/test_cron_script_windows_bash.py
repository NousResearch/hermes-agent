"""A cron ``.sh`` job runs under the same bash the terminal tool resolves, not whatever
``bash`` PATH hits first. On Windows that is the WSL stub (System32\\bash.exe), which drops the
backslashes from the script path and exits 127 (the job never runs)."""
import os
import subprocess

import pytest

from cron.scheduler_script import _script_argv


@pytest.mark.platforms("windows")
def test_sh_job_runs_when_wsl_stub_is_first_on_path(tmp_path, monkeypatch):
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    if not os.path.isfile(os.path.join(system32, "bash.exe")):
        pytest.skip("no WSL bash stub on this host")
    monkeypatch.setenv("PATH", system32 + os.pathsep + os.environ.get("PATH", ""))

    script = tmp_path / "job.sh"
    script.write_text("echo cron-sh-ok\n", encoding="utf-8")

    argv, _env, err = _script_argv(script)
    assert err is None
    result = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "cron-sh-ok"
