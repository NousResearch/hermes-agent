"""A cron script must be able to tell a real fire from a manual run.

Watchdog scripts commonly fire once and then delete their own state file to
disarm. That makes them NOT idempotent: running one by hand to check on it
consumes the pending alert -- output goes to the operator's terminal, the
state file is gone, and the next scheduled run has nothing left to report.

Nothing in the subprocess environment distinguished the two callers, so a
script had no way to defend itself. ``_run_job_script`` now exports
``HERMES_CRON_RUN=1``; absent means "invoked by hand, don't mutate state".

The scripts here are real files executed by the real runner -- the contract
under test is what a script actually observes in its environment.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import _run_job_script  # noqa: E402


@pytest.fixture
def scripts_dir(tmp_path, monkeypatch):
    """Real HERMES_HOME layout: the runner only executes files under scripts/."""
    home = tmp_path / ".hermes"
    (home / "scripts").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home / "scripts"


def _script(where: Path, name: str, body: str) -> Path:
    p = where / name
    p.write_text(body)
    p.chmod(0o755)
    return p


class TestCronRunMarker:
    def test_scheduled_run_sets_the_marker(self, scripts_dir):
        s = _script(
            scripts_dir,
            "probe.py",
            "import os\nprint(os.environ.get('HERMES_CRON_RUN', 'ABSENT'))\n",
        )
        ok, out = _run_job_script(str(s))
        assert ok, out
        assert out.strip() == "1"

    def test_marker_reaches_shell_scripts_too(self, scripts_dir):
        s = _script(scripts_dir, "probe.sh", 'echo "${HERMES_CRON_RUN:-ABSENT}"\n')
        ok, out = _run_job_script(str(s))
        assert ok, out
        assert out.strip() == "1"

    def test_manual_invocation_does_not_see_the_marker(self, tmp_path, monkeypatch):
        """The whole point: running the script directly must look different.

        This is the case that silently ate an alert -- a human running the
        same file gets no marker, so a guarded script leaves its state alone.
        """
        monkeypatch.delenv("HERMES_CRON_RUN", raising=False)
        s = _script(
            tmp_path,
            "probe.py",
            "import os\nprint(os.environ.get('HERMES_CRON_RUN', 'ABSENT'))\n",
        )
        import subprocess

        r = subprocess.run(
            [sys.executable, str(s)], capture_output=True, text=True, timeout=60
        )
        assert r.stdout.strip() == "ABSENT"

    def test_one_shot_watchdog_keeps_its_alert_on_a_manual_run(self, tmp_path):
        """End-to-end on the real failure shape: fire-once-then-disarm.

        Same script, both callers. Under cron it announces and disarms; run by
        hand it must announce nothing and leave the state file intact so the
        next scheduled run can still deliver.
        """
        state = tmp_path / "pending.flag"
        state.write_text("armed")
        watchdog = _script(
            tmp_path,
            "watchdog.sh",
            'STATE="$1"\n'
            'if [ -f "$STATE" ]; then\n'
            '  if [ "${HERMES_CRON_RUN:-}" != "1" ]; then\n'
            '    echo "[dry-run] would announce" >&2\n'
            "    exit 0\n"
            "  fi\n"
            '  rm -f "$STATE"\n'
            '  echo "ALERT"\n'
            "fi\n",
        )
        import subprocess

        # Manual: no marker -> must not consume the alert.
        env = dict(os.environ)
        env.pop("HERMES_CRON_RUN", None)
        r = subprocess.run(
            ["bash", str(watchdog), str(state)],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        assert "ALERT" not in r.stdout
        assert state.exists(), "manual run disarmed the watchdog -- alert lost"

        # Scheduled: marker present -> announces and disarms exactly once.
        env["HERMES_CRON_RUN"] = "1"
        r = subprocess.run(
            ["bash", str(watchdog), str(state)],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        assert "ALERT" in r.stdout
        assert not state.exists()

    def test_marker_does_not_clobber_existing_env(self, scripts_dir):
        """HERMES_HOME and friends must still arrive intact."""
        s = _script(
            scripts_dir,
            "probe.py",
            "import os\nprint('HOME_OK' if os.environ.get('HERMES_HOME') else 'HOME_MISSING')\n",
        )
        ok, out = _run_job_script(str(s))
        assert ok, out
        assert "HOME_OK" in out
