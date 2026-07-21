"""Tests for Hermes Doctor secret-file permission diagnostics."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from hermes_cli import doctor


pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="POSIX mode bits not enforced on Windows",
)


def _write_with_mode(path: Path, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("SECRET=value\n", encoding="utf-8")
    os.chmod(path, mode)


def test_secret_permission_check_flags_group_or_other_bits(tmp_path: Path):
    env_path = tmp_path / ".env"
    auth_path = tmp_path / "auth.json"
    _write_with_mode(env_path, 0o600)
    _write_with_mode(auth_path, 0o644)

    results = doctor._collect_secret_file_permission_checks(
        hermes_home=tmp_path,
        home=tmp_path / "home",
    )

    by_label = {result.label: result for result in results}
    assert by_label["Hermes .env"].secure is True
    assert by_label["Hermes auth store"].secure is False
    assert by_label["Hermes auth store"].mode == 0o644
    assert by_label["Hermes auth store"].recommended_mode == 0o600


def test_secret_permission_check_ignores_missing_files(tmp_path: Path):
    results = doctor._collect_secret_file_permission_checks(
        hermes_home=tmp_path,
        home=tmp_path / "home",
    )

    assert results == []


def test_secret_permission_check_includes_github_hosts(tmp_path: Path):
    gh_hosts = tmp_path / "home" / ".config" / "gh" / "hosts.yml"
    _write_with_mode(gh_hosts, 0o660)

    results = doctor._collect_secret_file_permission_checks(
        hermes_home=tmp_path / ".hermes",
        home=tmp_path / "home",
    )

    assert len(results) == 1
    assert results[0].label == "GitHub CLI hosts.yml"
    assert results[0].secure is False
    assert results[0].mode == 0o660


def test_render_secret_permission_checks_adds_manual_issue_without_secret_content(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    env_path = tmp_path / ".env"
    _write_with_mode(env_path, 0o644)
    manual_issues: list[str] = []

    doctor._render_secret_file_permission_checks(
        manual_issues,
        hermes_home=tmp_path,
        home=tmp_path / "home",
        display_home="<hermes-home>",
    )

    output = capsys.readouterr().out
    assert "Secret File Permissions" in output
    assert "Hermes .env" in output
    assert "0o644" in output
    assert "SECRET=value" not in output
    assert manual_issues == [f"Restrict Hermes .env permissions: chmod 600 {env_path}"]


def test_secret_permission_check_accepts_owner_execute_only_for_directories(tmp_path: Path):
    # Files should not need execute bits; this test guards against accidentally
    # treating 0o700 as the target file mode.
    env_path = tmp_path / ".env"
    _write_with_mode(env_path, stat.S_IRUSR | stat.S_IWUSR)

    [result] = doctor._collect_secret_file_permission_checks(
        hermes_home=tmp_path,
        home=tmp_path / "home",
    )

    assert result.secure is True
    assert result.mode == 0o600
