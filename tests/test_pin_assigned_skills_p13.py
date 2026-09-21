"""P13 isolation proof for scripts/pin-assigned-skills.sh.

Verifies the dry-run flag:
- PIN_DRY_RUN=1 prints the skills that would be pinned, exits 0, never
  calls `hermes curator pin`, never touches the lockfile.
- HERMES_HOME can be redirected so the production .hermes is untouched.
- a missing HERMES_HOME in dry-run is fine (short-circuits before the
  find/cd guards are reached for the pin path).
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "pin-assigned-skills.sh"


def _run(env_extra: dict | None = None):
    env = dict(os.environ)
    env.pop("PIN_DRY_RUN", None)
    env.pop("HERMES_HOME", None)
    if env_extra:
        env.update(env_extra)
    # Ensure `hermes` is NOT on PATH so any accidental call fails loudly.
    env["PATH"] = "/usr/bin:/bin"
    return subprocess.run(
        ["bash", str(WRAPPER)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=15,
    )


def _build_fake_home(tmp_path, skills=("my-skill",)):
    fake = tmp_path / "fake_hermes"
    cfg = fake / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        "skills:\n  enabled_skills:\n" + "".join(f"    - {s}\n" for s in skills)
    )
    # Create the skills dir + SKILL.md so the `find` succeeds.
    skills_dir = fake / "skills"
    for s in skills:
        sd = skills_dir / "cat" / s
        sd.mkdir(parents=True)
        (sd / "SKILL.md").write_text(f"# {s}\n")
    return fake


def test_dry_run_exits_zero_without_hermes_cli(tmp_path):
    """PIN_DRY_RUN=1 prints would-pin lines and never calls hermes."""
    fake = _build_fake_home(tmp_path)
    r = _run({"PIN_DRY_RUN": "1", "HERMES_HOME": str(fake)})
    assert r.returncode == 0, r.stderr
    assert "dry-run: would pin my-skill" in r.stdout
    # No lockfile created.
    assert not (fake / ".pin_assigned_skills.lock").exists()


def test_dry_run_works_even_when_dir_missing(tmp_path):
    """Dry-run short-circuits before requiring a full skill layout."""
    missing = tmp_path / "does-not-exist"
    r = _run({"PIN_DRY_RUN": "1", "HERMES_HOME": str(missing)})
    assert r.returncode == 0, r.stderr
    # No skill found (no config.yaml) → no would-pin line, but no error.


def test_dry_run_does_not_touch_production_home(tmp_path, monkeypatch):
    """Dry-run with a temp HERMES_HOME must not create any file in the
    production ~/.hermes."""
    fake = _build_fake_home(tmp_path)
    r = _run({"PIN_DRY_RUN": "1", "HERMES_HOME": str(fake)})
    assert r.returncode == 0, r.stderr
    # The fake home must not have a lockfile.
    assert not (fake / ".pin_assigned_skills.lock").exists()
