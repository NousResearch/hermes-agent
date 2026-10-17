"""Native Windows namespace controls for profile-local Cron custody."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from hermes_cron_export_custody import (
    CronCustodyRefused,
    cron_capture_custody,
    cron_writer_custody,
)


def _junction(link: Path, target: Path) -> None:
    subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        check=True,
        capture_output=True,
    )


@pytest.mark.platforms("windows")
def test_profile_root_junction_refuses_without_creating_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    link = tmp_path / "profile-junction"
    _junction(link, physical)
    monkeypatch.setenv("HERMES_HOME", str(link))
    try:
        with pytest.raises(CronCustodyRefused):
            with cron_capture_custody(timeout_seconds=2.0):
                pass
        assert not tuple(physical.iterdir())
    finally:
        link.rmdir()


@pytest.mark.platforms("windows")
def test_parent_junction_retarget_invalidates_lease_and_releases_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for parent in (first, second):
        (parent / "profile").mkdir(parents=True)
    link = tmp_path / "parent-junction"
    _junction(link, first)
    monkeypatch.setenv("HERMES_HOME", str(link / "profile"))
    manager = cron_capture_custody(timeout_seconds=2.0)
    lease = manager.__enter__()
    try:
        lease.assert_current()
        link.rmdir()
        _junction(link, second)
        with pytest.raises(CronCustodyRefused):
            lease.assert_current()
    finally:
        try:
            with pytest.raises(CronCustodyRefused):
                manager.__exit__(None, None, None)
        finally:
            link.rmdir()
    # The old physical generation's handle must be released, rather than
    # merely allowing acquisition against the replacement target.
    monkeypatch.setenv("HERMES_HOME", str(first / "profile"))
    with cron_writer_custody(timeout_seconds=2.0):
        pass
    assert not (first / "profile" / "cron").exists()
