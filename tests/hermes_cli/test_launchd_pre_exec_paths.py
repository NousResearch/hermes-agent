"""launchd's own cwd/stdout/stderr handles must stay on the boot volume.

Regression guard: with a HERMES_HOME on a removable/external volume, macOS denies xpcproxy the
plist's ``WorkingDirectory``/``StandardOutPath``/``StandardErrorPath`` before the job's osascript
identity exists, so the gateway parks at ``spawn failed`` (EX_CONFIG 78) with no gateway log line
at all. Only those three paths may move; the job's own redirect stays in HERMES_HOME so
``hermes logs`` keeps working.
"""

from pathlib import Path

import pytest

from hermes_cli import gateway_launchd


@pytest.mark.platforms("macos")
def test_pre_exec_dir_moves_off_a_non_boot_volume(monkeypatch, tmp_path):
    desired = tmp_path / "external" / "logs"
    desired.mkdir(parents=True)

    monkeypatch.setattr(gateway_launchd, "_on_boot_volume", lambda path: False)

    assert gateway_launchd._launchd_pre_exec_dir(desired, tmp_path / "internal") == tmp_path / "internal"


@pytest.mark.platforms("macos")
def test_pre_exec_dir_is_unchanged_on_the_boot_volume(monkeypatch, tmp_path):
    desired = tmp_path / "logs"
    desired.mkdir()

    monkeypatch.setattr(gateway_launchd, "_on_boot_volume", lambda path: True)

    assert gateway_launchd._launchd_pre_exec_dir(desired, tmp_path / "internal") == desired


@pytest.mark.platforms("macos")
def test_pre_exec_dir_creates_the_fallback_it_returns(monkeypatch, tmp_path):
    desired = tmp_path / "external" / "logs"
    desired.mkdir(parents=True)
    fallback = tmp_path / "internal" / "Hermes"

    monkeypatch.setattr(gateway_launchd, "_on_boot_volume", lambda path: False)

    assert gateway_launchd._launchd_pre_exec_dir(desired, fallback).is_dir()


def test_boot_volume_probe_answers_for_this_host(tmp_path):
    assert gateway_launchd._on_boot_volume(Path("/")) is True
    assert gateway_launchd._on_boot_volume(tmp_path) is True


def test_boot_volume_probe_keeps_the_configured_path_when_unreadable():
    assert gateway_launchd._on_boot_volume(Path("/nonexistent-hermes-probe")) is True
