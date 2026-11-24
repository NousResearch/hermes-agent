"""The printed release date follows the running commit, not a frozen literal (#159217)."""

import datetime
import time

import pytest

import hermes_cli
from hermes_cli.version_info import VersionInfo


def _info(commit_date: int | None) -> VersionInfo:
    return VersionInfo("0.23.3", "0.23.3+4.gabcdef0", 4, "a" * 40, "main", "git", False, commit_date)


@pytest.mark.platforms("posix")  # time.tzset() is POSIX-only
def test_banner_prints_the_running_commit_date_in_compact_calver(monkeypatch):
    from hermes_cli import banner

    # 00:30 UTC: a local-time render shows the previous day west of UTC; the date must be UTC everywhere.
    info = _info(int(datetime.datetime(2026, 9, 4, 0, 30, tzinfo=datetime.UTC).timestamp()))
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    monkeypatch.setattr("hermes_cli.steward.read_install_stamp", lambda project_root: {})
    monkeypatch.setattr("hermes_cli.version_info.get_version_info", lambda: info)
    monkeypatch.setattr(banner, "get_version_info", lambda: info)
    monkeypatch.setattr(banner, "get_git_banner_state", lambda repo_dir=None: None)
    monkeypatch.setattr("hermes_cli.update_channel.resolve_update_channel", lambda *args: "main")
    monkeypatch.setattr(hermes_cli, "__release_date__", "2000.1.1")

    try:
        assert "Hermes Agent v0.23.3+4.gabcdef0 (2026.9.4)" in banner.format_banner_version_label()
    finally:
        monkeypatch.undo()
        time.tzset()


def test_release_date_falls_back_to_the_literal_without_provenance(monkeypatch):
    from hermes_cli.version_info import release_date_text

    monkeypatch.setattr("hermes_cli.version_info.get_version_info", lambda: _info(None))
    monkeypatch.setattr(hermes_cli, "__release_date__", "2031.2.3")

    assert release_date_text() == "2031.2.3"
