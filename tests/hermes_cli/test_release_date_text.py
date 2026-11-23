"""The printed release date follows the running commit, not a frozen literal (#159217).

``hermes --version`` and the banner used to print ``__release_date__``
(``2026.9.24``) no matter which release was installed. The authoritative date
already exists beside the version in ``VersionInfo.commit_date``; these tests
pin the three surfaces that render it plus the fallback and the format.
"""

import datetime
from types import SimpleNamespace

import pytest

import hermes_cli
from hermes_cli.version_info import VersionInfo

COMMIT_EPOCH = int(datetime.datetime(2026, 1, 2, 12, 0).timestamp())


def _expected(epoch: int) -> str:
    """``YYYY.M.D`` for the process-local date, per the product's format."""
    stamp = datetime.datetime.fromtimestamp(epoch)
    return f"{stamp.year}.{stamp.month}.{stamp.day}"


def _info(commit_date: int | None) -> VersionInfo:
    return VersionInfo(
        "0.23.3", "0.23.3+4.gabcdef0", 4, "a" * 40, "main", "git", False, commit_date,
    )


@pytest.fixture
def pinned_identity(monkeypatch):
    """Pin the resolved identity on every module that aliases the resolver."""
    info = _info(COMMIT_EPOCH)
    monkeypatch.setattr("hermes_cli.version_info.get_version_info", lambda: info)
    yield info


def test_release_date_text_uses_commit_date(pinned_identity):
    from hermes_cli.version_info import release_date_text

    assert release_date_text() == _expected(COMMIT_EPOCH)


def test_legacy_version_flag_prints_commit_date(monkeypatch, capsys):
    from agent import legacy_cli

    monkeypatch.setattr("hermes_cli.version_info.get_version_info", lambda: _info(COMMIT_EPOCH))
    monkeypatch.setattr(hermes_cli, "__version__", "0.23.3", raising=False)

    with pytest.raises(SystemExit) as exit_info:
        legacy_cli._build_parser().parse_args(["--version"])

    assert exit_info.value.code == 0
    printed = capsys.readouterr().out
    assert f"Hermes Agent v0.23.3 ({_expected(COMMIT_EPOCH)})" in printed
    assert "2026.9.24" not in printed


def test_banner_label_prints_commit_date(monkeypatch):
    from hermes_cli import banner

    info = _info(COMMIT_EPOCH)
    monkeypatch.setattr("hermes_cli.version_info.get_version_info", lambda: info)
    monkeypatch.setattr(banner, "get_version_info", lambda: info)
    monkeypatch.setattr(banner, "get_git_banner_state", lambda repo_dir=None: None)
    monkeypatch.setattr("hermes_cli.update_channel.resolve_update_channel", lambda *args: "main")

    label = banner.format_banner_version_label()

    assert f"Hermes Agent v0.23.3+4.gabcdef0 ({_expected(COMMIT_EPOCH)})" in label
    assert "2026.9.24" not in label


def test_release_date_text_falls_back_without_provenance(monkeypatch):
    from hermes_cli.version_info import release_date_text

    monkeypatch.setattr("hermes_cli.version_info.get_version_info", lambda: _info(None))
    monkeypatch.setattr(hermes_cli, "__release_date__", "2031.2.3")

    assert release_date_text() == "2031.2.3"


def test_release_date_text_keeps_compact_calver_format(monkeypatch):
    from hermes_cli.version_info import release_date_text

    epoch = int(datetime.datetime(2026, 9, 4, 12, 0).timestamp())
    monkeypatch.setattr("hermes_cli.version_info.get_version_info", lambda: _info(epoch))

    text = release_date_text()

    assert text == "2026.9.4"
    assert text.count(".") == 2
    assert not any(part.startswith("0") and len(part) > 1 for part in text.split("."))


def test_session_info_keeps_the_tui_release_date_contract(pinned_identity, monkeypatch):
    """The TUI gateway keeps advertising the resolved date; no surface drift."""
    from tui_gateway import server

    monkeypatch.setattr(hermes_cli, "__release_date__", "2026.9.23")

    info = server._session_info(None, {})

    assert info["version"] == "0.23.3"
    assert info["release_date"] == _expected(COMMIT_EPOCH)
