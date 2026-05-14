"""Regression test for dynamic cron paths after profile override (issue #25295)."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_cron_jobs_use_active_profile_after_hermes_home_switch(monkeypatch, tmp_path):
    """Cron writes under the active profile directory after HERMES_HOME changes."""
    hermes_root = tmp_path / ".hermes"
    profile_dir = hermes_root / "profiles" / "coder"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (hermes_root / "active_profile").write_text("coder")

    # Import cron jobs while HERMES_HOME still points to the hermes root.
    monkeypatch.setenv("HERMES_HOME", str(hermes_root))
    import cron.jobs as jobs_mod
    importlib.reload(jobs_mod)

    # Simulate process startup profile resolution.
    from hermes_cli.main import _apply_profile_override

    _apply_profile_override()

    # Any write should now target the profile dir, not the root path.
    job = jobs_mod.create_job(prompt="status check", schedule="every 1h")
    assert job["id"]

    active_profile_jobs = profile_dir / "cron" / "jobs.json"
    root_jobs = hermes_root / "cron" / "jobs.json"
    assert active_profile_jobs.exists()
    assert not root_jobs.exists()

    # Also validate the prompt-context helper follows the same runtime path.
    _get_output_dir = jobs_mod._get_output_dir()
    assert _get_output_dir == profile_dir / "cron" / "output"
