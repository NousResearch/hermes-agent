"""The version label's parenthetical date comes from the running build's identity.

A checked-in literal can never track tag-based releases: the release flow cuts a
tag without a repo commit, so no step bumps a date in the tree (#135217). Every
surface that prints the date therefore resolves it through ``version_info``, the
single identity authority — the existing literal stays only as the no-provenance
fallback.
"""

from datetime import UTC, datetime
from unittest.mock import Mock

from hermes_cli import banner, version_info


def _info(commit_date: int | None = None) -> version_info.VersionInfo:
    return version_info.VersionInfo(
        "0.22.5", "0.22.5", 0, "a" * 40, None, "git", False, commit_date
    )


def _epoch(*args: int) -> int:
    return int(datetime(*args, tzinfo=UTC).timestamp())


def test_release_date_is_utc_ymd_without_padding():
    # The label has always read unpadded (2026.9.24); keep that shape.
    assert _info(_epoch(2026, 10, 8)).release_date == "2026.10.8"
    assert _info(_epoch(2026, 1, 5)).release_date == "2026.1.5"


def test_release_date_of_a_late_utc_commit_stays_on_its_utc_day():
    # 23:30Z is already the next day in Asia/Shanghai: the label must not follow local time.
    assert _info(_epoch(2026, 10, 23, 23, 30)).release_date == "2026.10.23"


def test_release_date_is_empty_without_a_commit_date():
    assert _info().release_date == ""


def test_get_release_date_prefers_identity_over_the_literal(monkeypatch):
    import hermes_cli

    monkeypatch.setattr(version_info, "_cached_version_info", _info(_epoch(2026, 10, 8)))

    assert version_info.get_release_date() == "2026.10.8"
    assert version_info.get_release_date() != hermes_cli.__release_date__


def test_get_release_date_keeps_the_literal_without_provenance(monkeypatch):
    import hermes_cli

    monkeypatch.setattr(
        version_info,
        "_cached_version_info",
        version_info.VersionInfo("unknown", "unknown", None, None, None, "unknown"),
    )

    assert version_info.get_release_date() == hermes_cli.__release_date__


def test_banner_label_dates_the_build_from_identity(tmp_path, monkeypatch):
    from hermes_cli import config

    root = tmp_path / "source"
    root.mkdir()
    (root / ".git").mkdir()
    monkeypatch.setattr(config, "get_project_root", lambda: root)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"update": {"channel": "main"}})
    monkeypatch.setattr(version_info, "_cached_version_info", _info(_epoch(2026, 10, 8)))
    monkeypatch.setattr(banner, "get_git_banner_state", Mock(return_value=None))

    label = banner.format_banner_version_label()

    assert label == "Hermes Agent v0.22.5 (2026.10.8) · main" or label.startswith(
        "Hermes Agent v0.22.5 (2026.10.8)"
    )
