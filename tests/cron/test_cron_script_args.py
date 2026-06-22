"""Tests for cron script_args feature — B class tests.

Covers _run_job_script() argv construction, create_job() validation,
and _format_job() echo-back of the script_args field.

See: /Users/x/.hermes/tmp/cc-fix-plan-dual-bug-v2.md
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    (hermes_home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    import cron.scheduler as sched_mod
    monkeypatch.setattr(sched_mod, "_hermes_home", hermes_home)

    return hermes_home


def _fake_subprocess_run(returncode=0, stdout="ok", stderr=""):
    """Return a side_effect for subprocess.run that captures argv."""
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)

    return captured, fake_run


class TestRunJobScriptArgs:
    """B: _run_job_script argv construction with script_args."""

    def test_no_args(self, cron_env):
        """script_args not passed → argv contains only interpreter + script."""
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "test.py"
        script.write_text('print("hello")\n')

        captured, fake_run = _fake_subprocess_run()
        with patch("subprocess.run", side_effect=fake_run):
            success, output = _run_job_script("test.py")

        assert success is True
        assert output == "ok"
        argv = captured["argv"]
        assert argv[0] == sys.executable
        assert argv[1].endswith("test.py")
        assert len(argv) == 2

    def test_single_arg(self, cron_env):
        """script_args=["--mode", "close"] → both appended to argv."""
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "test.py"
        script.write_text('print("hello")\n')

        captured, fake_run = _fake_subprocess_run()
        with patch("subprocess.run", side_effect=fake_run):
            success, output = _run_job_script("test.py", script_args=["--mode", "close"])

        assert success is True
        argv = captured["argv"]
        assert argv[-2] == "--mode"
        assert argv[-1] == "close"

    def test_multiple_args(self, cron_env):
        """Multiple items in script_args → all appended in order."""
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "test.py"
        script.write_text('print("hello")\n')

        captured, fake_run = _fake_subprocess_run()
        with patch("subprocess.run", side_effect=fake_run):
            success, output = _run_job_script(
                "test.py",
                script_args=["--verbose", "--output", "/tmp/out", "--retries", "3"],
            )

        assert success is True
        argv = captured["argv"]
        args_slice = argv[2:]  # after interpreter + script
        assert args_slice == ["--verbose", "--output", "/tmp/out", "--retries", "3"]

    def test_script_none_script_args(self):
        """script=None → early return with error, no subprocess call."""
        from cron.scheduler import _run_job_script

        with patch("cron.scheduler.subprocess.run") as mock_run:
            success, output = _run_job_script(None)

        assert success is False
        assert "Script path is empty or None" == output
        mock_run.assert_not_called()

    def test_script_empty_string(self):
        """script="" → early return with error, no subprocess call."""
        from cron.scheduler import _run_job_script

        with patch("cron.scheduler.subprocess.run") as mock_run:
            success, output = _run_job_script("")

        assert success is False
        assert "Script path is empty or None" == output
        mock_run.assert_not_called()

    def test_empty_script_args_list(self, cron_env):
        """script_args=[] → same argv length as no script_args."""
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "test.py"
        script.write_text('print("hello")\n')

        captured, fake_run = _fake_subprocess_run()
        with patch("subprocess.run", side_effect=fake_run):
            success, output = _run_job_script("test.py", script_args=[])

        assert success is True
        argv = captured["argv"]
        assert len(argv) == 2  # just interpreter + script

    def test_script_path_with_spaces(self, cron_env):
        """Filename with spaces is NOT split — script stays one path, not two tokens."""
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "my script.py"
        script.write_text('print("ok")\n')

        captured, fake_run = _fake_subprocess_run()
        with patch("subprocess.run", side_effect=fake_run):
            success, output = _run_job_script("my script.py", script_args=["--flag"])

        assert success is True
        argv = captured["argv"]
        # The script path must be a single argv entry, not split on space
        assert argv[1].endswith("my script.py")
        assert "--flag" in argv

    def test_path_traversal_rejected(self, cron_env):
        """Path traversal (../../etc/passwd) is blocked by containment check."""
        from cron.scheduler import _run_job_script

        success, output = _run_job_script("../../etc/passwd")
        assert success is False
        assert "blocked" in output.lower()

    def test_shell_script_with_args(self, cron_env):
        """.sh script uses bash interpreter and appends script_args correctly."""
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "backup.sh"
        script.write_text("#!/bin/bash\necho done\n")

        captured, fake_run = _fake_subprocess_run()
        with patch("subprocess.run", side_effect=fake_run):
            success, output = _run_job_script("backup.sh", script_args=["--force", "--dest", "/tmp/out"])

        assert success is True
        assert output == "ok"
        argv = captured["argv"]
        # argv[0] must be bash, not python
        assert argv[0].endswith("bash")
        assert argv[1].endswith("backup.sh")
        args_slice = argv[2:]
        assert args_slice == ["--force", "--dest", "/tmp/out"]

    def test_bash_script_with_args(self, cron_env):
        """.bash script also uses bash interpreter and appends script_args."""
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "deploy.bash"
        script.write_text("#!/bin/bash\necho done\n")

        captured, fake_run = _fake_subprocess_run()
        with patch("subprocess.run", side_effect=fake_run):
            success, output = _run_job_script("deploy.bash", script_args=["--env", "prod"])

        assert success is True
        argv = captured["argv"]
        assert argv[0].endswith("bash")
        assert argv[1].endswith("deploy.bash")
        assert argv[2:] == ["--env", "prod"]


class TestCreateJobScriptArgsValidation:
    """B: create_job() script_args input validation."""

    def test_non_list_rejected(self, cron_env):
        """script_args must be a list — non-list values raise TypeError."""
        from cron.jobs import create_job

        with pytest.raises(TypeError, match="script_args must be a list"):
            create_job(
                prompt="test",
                schedule="every 1h",
                script="test.py",
                script_args="not_a_list",
            )

    def test_non_list_rejected_in_cronjob_update(self, cron_env, monkeypatch):
        """cronjob() update with non-list script_args returns error."""
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        create_res = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="test",
            script="test.py",
        ))
        job_id = create_res["job_id"]

        update_res = json.loads(cronjob(
            action="update",
            job_id=job_id,
            script_args="not_a_list",
        ))
        assert update_res["success"] is False
        assert "must be a list" in update_res["error"].lower()

    def test_update_script_args_success(self, cron_env, monkeypatch):
        """cronjob() update with valid script_args persists and echoes them back."""
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        create_res = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="test",
            script="test.py",
        ))
        job_id = create_res["job_id"]

        update_res = json.loads(cronjob(
            action="update",
            job_id=job_id,
            script_args=["--mode", "full", "--verbose"],
        ))
        assert update_res["success"] is True
        assert update_res["job"]["script_args"] == ["--mode", "full", "--verbose"]

        # Verify persisted by reading back via list
        list_res = json.loads(cronjob(action="list"))
        for j in list_res["jobs"]:
            if j["job_id"] == job_id:
                assert j["script_args"] == ["--mode", "full", "--verbose"]
                break
        else:
            pytest.fail(f"Job {job_id} not found in list")

    def test_update_script_args_clear(self, cron_env, monkeypatch):
        """cronjob() update with script_args=[] clears the field."""
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        create_res = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="test",
            script="test.py",
            script_args=["--original"],
        ))
        job_id = create_res["job_id"]

        update_res = json.loads(cronjob(
            action="update",
            job_id=job_id,
            script_args=[],
        ))
        assert update_res["success"] is True
        assert update_res["job"]["script_args"] == []

        # Verify persisted as empty
        list_res = json.loads(cronjob(action="list"))
        for j in list_res["jobs"]:
            if j["job_id"] == job_id:
                assert j["script_args"] == []
                break
        else:
            pytest.fail(f"Job {job_id} not found in list")

    def test_update_script_args_retain(self, cron_env, monkeypatch):
        """cronjob() update without script_args field retains existing value."""
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        create_res = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="test",
            script="test.py",
            script_args=["--original"],
        ))
        job_id = create_res["job_id"]

        # Update only the name — script_args should be retained
        update_res = json.loads(cronjob(
            action="update",
            job_id=job_id,
            name="renamed-job",
        ))
        assert update_res["success"] is True
        assert update_res["job"]["script_args"] == ["--original"]

    def test_non_string_elements_normalized(self, cron_env):
        """script_args with int, bool, float → all normalised to str via str()."""
        from cron.jobs import create_job
        from tools.cronjob_tools import _format_job

        job = create_job(
            prompt="test",
            schedule="every 1h",
            script="test.py",
            script_args=[1, True, 3.14, 0, False],
        )

        formatted = _format_job(job)
        assert formatted["script_args"] == ["1", "True", "3.14", "0", "False"]


class TestFormatJobScriptArgs:
    """B: _format_job() echo-back of script_args."""

    def test_format_job_includes_script_args(self, cron_env):
        """_format_job echoes script_args when present on the job."""
        from cron.jobs import create_job
        from tools.cronjob_tools import _format_job

        job = create_job(
            prompt="analyze",
            schedule="every 1h",
            script="collector.py",
            script_args=["--mode", "full", "--since", "yesterday"],
        )

        formatted = _format_job(job)
        assert "script_args" in formatted
        assert formatted["script_args"] == ["--mode", "full", "--since", "yesterday"]

    def test_format_job_no_script_args(self, cron_env):
        """_format_job echoes empty script_args list when not supplied."""
        from cron.jobs import create_job
        from tools.cronjob_tools import _format_job

        job = create_job(
            prompt="analyze",
            schedule="every 1h",
            script="collector.py",
        )

        formatted = _format_job(job)
        assert "script_args" in formatted
        assert formatted["script_args"] == []
