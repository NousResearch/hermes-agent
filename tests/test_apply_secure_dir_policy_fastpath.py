"""Regression tests for the steady-state fast path in apply_secure_dir_policy (#138105).

The multiplex cron ticker secures cron dirs once per profile per cycle; with dozens of
mostly-empty profiles the per-call container probe (reads /proc/1/cgroup and
/proc/self/mountinfo) dominated the cron-scheduler thread and starved the Discord heartbeat
under the GIL. A directory already at the policy's default mode with no operator overrides
must skip that probe entirely; anything else must keep the full policy.
"""
import os
import stat
from unittest.mock import patch

import pytest

from hermes_constants import apply_secure_dir_policy


def _isolate_policy_env(monkeypatch):
    monkeypatch.delenv("HERMES_HOME_MODE", raising=False)
    monkeypatch.delenv("HERMES_UID", raising=False)
    monkeypatch.delenv("HERMES_GID", raising=False)
    monkeypatch.setattr("hermes_constants.get_managed_system", lambda home=None: None)


@pytest.mark.platforms("posix")  # POSIX file modes
def test_owner_only_dir_skips_container_probe_entirely(tmp_path, monkeypatch):
    _isolate_policy_env(monkeypatch)
    d = tmp_path / "cron"
    d.mkdir()
    os.chmod(d, 0o700)

    def _probe_must_not_run():
        raise AssertionError("already-owner-only dir must not re-run container detection")

    monkeypatch.setattr("hermes_constants._container_or_chmod_skipped", _probe_must_not_run)
    apply_secure_dir_policy(d)  # returns through the fast path without raising


@pytest.mark.platforms("posix")  # POSIX file modes
def test_wrong_mode_dir_still_runs_full_policy(tmp_path, monkeypatch):
    _isolate_policy_env(monkeypatch)
    d = tmp_path / "cron"
    d.mkdir()
    os.chmod(d, 0o755)

    probed: list[int] = []
    monkeypatch.setattr(
        "hermes_constants._container_or_chmod_skipped", lambda: probed.append(1) or True,
    )
    apply_secure_dir_policy(d)
    assert probed == [1]
    # Container-skip branch leaves the operator's mode exactly as found.
    assert stat.S_IMODE(os.stat(d).st_mode) == 0o755


@pytest.mark.platforms("posix")  # POSIX file modes
def test_hermes_uid_set_bypasses_fast_path(tmp_path, monkeypatch):
    """HERMES_UID deployments still need the chown on every call, so the fast path must not
    fire even when the mode already matches."""
    _isolate_policy_env(monkeypatch)
    monkeypatch.setenv("HERMES_UID", "1000")
    d = tmp_path / "cron"
    d.mkdir()
    os.chmod(d, 0o700)

    monkeypatch.setattr("hermes_constants._container_or_chmod_skipped", lambda: True)
    with patch.object(os, "chown") as mock_chown:
        apply_secure_dir_policy(d)
    mock_chown.assert_called_once_with(d, 1000, -1)
