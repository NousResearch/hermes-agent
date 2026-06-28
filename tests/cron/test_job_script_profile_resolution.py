"""Regression tests for cron script resolution using job's profile field (#54288).

When a cron job specifies `"profile": "X"`, its script should resolve against
`~/.hermes/profiles/X/scripts/` — not the active HERMES_HOME/scripts/ which
depends on the last-active profile context.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def profile_layout(tmp_path):
    """Create a multi-profile scripts layout.

    Returns (default_home, profile_a, profile_b) where each has a scripts/ dir.
    """
    default_home = tmp_path / ".hermes"
    profile_a = default_home / "profiles" / "alpha"
    profile_b = default_home / "profiles" / "bravo"

    for d in [default_home / "scripts", profile_a / "scripts", profile_b / "scripts"]:
        d.mkdir(parents=True)

    # Scripts in different profiles
    (default_home / "scripts" / "global.sh").write_text("#!/bin/bash\necho global")
    (profile_a / "scripts" / "alpha.sh").write_text("#!/bin/bash\necho alpha")
    (profile_b / "scripts" / "bravo.sh").write_text("#!/bin/bash\necho bravo")

    return default_home, profile_a, profile_b


def _patch_hermes_home(monkeypatch, hermes_home: Path):
    """Patch _get_hermes_home and get_default_hermes_root for testing."""
    monkeypatch.setattr(
        "cron.scheduler._get_hermes_home",
        lambda: hermes_home,
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: hermes_home.parent if hermes_home.parent.name == "profiles" else hermes_home,
    )


def test_script_resolves_to_job_profile(profile_layout, monkeypatch):
    """Script should be found in the job's profile scripts dir."""
    default_home, profile_a, profile_b = profile_layout

    # Active HERMES_HOME is profile_b (simulating wrong profile context)
    _patch_hermes_home(monkeypatch, profile_b)
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: default_home,
    )

    from cron.scheduler import _run_job_script

    # Job says profile=alpha, script=alpha.sh
    # Should find alpha.sh in profiles/alpha/scripts/, not bravo's scripts/
    ok, output = _run_job_script("alpha.sh", job_profile="alpha")
    assert ok, f"Script should be found: {output}"
    assert "alpha" in output


def test_script_not_found_in_wrong_profile(profile_layout, monkeypatch):
    """Script from one profile should NOT be found when job specifies another."""
    default_home, profile_a, profile_b = profile_layout

    _patch_hermes_home(monkeypatch, profile_b)
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: default_home,
    )

    from cron.scheduler import _run_job_script

    # Job says profile=alpha, script=bravo.sh
    # bravo.sh exists in bravo/scripts/ but NOT in alpha/scripts/
    ok, output = _run_job_script("bravo.sh", job_profile="alpha")
    assert not ok
    assert "not found" in output.lower()


def test_no_profile_uses_active_hermes_home(profile_layout, monkeypatch):
    """Without job_profile, scripts resolve against active HERMES_HOME."""
    default_home, profile_a, profile_b = profile_layout

    _patch_hermes_home(monkeypatch, profile_a)
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: default_home,
    )

    from cron.scheduler import _run_job_script

    # No job_profile → uses active HERMES_HOME (profile_a)
    ok, output = _run_job_script("alpha.sh")
    assert ok, f"Script should be found: {output}"
    assert "alpha" in output


def test_path_traversal_blocked_in_profile_mode(profile_layout, monkeypatch):
    """Path traversal must still be blocked when using job_profile."""
    default_home, profile_a, profile_b = profile_layout

    _patch_hermes_home(monkeypatch, profile_b)
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: default_home,
    )

    from cron.scheduler import _run_job_script

    ok, output = _run_job_script("../../../etc/passwd", job_profile="alpha")
    assert not ok
    assert "blocked" in output.lower() or "outside" in output.lower()


def test_absolute_path_in_profile_mode(profile_layout, monkeypatch):
    """Absolute paths must still be validated against the scripts dir."""
    default_home, profile_a, profile_b = profile_layout

    _patch_hermes_home(monkeypatch, profile_b)
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: default_home,
    )

    from cron.scheduler import _run_job_script

    # Absolute path to a file outside the scripts dir
    ok, output = _run_job_script("/tmp/evil.sh", job_profile="alpha")
    assert not ok
