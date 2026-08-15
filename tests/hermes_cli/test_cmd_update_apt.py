"""APT-managed Hermes installs must never fall through to the git updater."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.main import _cmd_update_check, cmd_update


def test_apt_stamp_is_detected_and_recommends_pkg_upgrade(tmp_path):
    from hermes_cli.config import detect_install_method, recommended_update_command_for_method

    (tmp_path / ".install_method").write_text("apt\n", encoding="utf-8")
    assert detect_install_method(project_root=tmp_path) == "apt"
    assert recommended_update_command_for_method("apt") == "pkg upgrade hermes-agent"


@patch("hermes_cli.config.is_managed", return_value=False)
@patch("hermes_cli.config.detect_install_method", return_value="apt")
@patch("subprocess.run")
def test_cmd_update_apt_prints_pkg_guidance_without_git(
    mock_run, _mock_method, _mock_managed, capsys
):
    with pytest.raises(SystemExit) as excinfo:
        cmd_update(SimpleNamespace(check=False))

    assert excinfo.value.code == 1
    assert "pkg upgrade hermes-agent" in capsys.readouterr().out
    assert mock_run.call_args_list == []


@patch("hermes_cli.config.detect_install_method", return_value="apt")
@patch("subprocess.run")
def test_cmd_update_check_apt_prints_pkg_guidance_without_git(
    mock_run, _mock_method, capsys
):
    with pytest.raises(SystemExit) as excinfo:
        _cmd_update_check()

    assert excinfo.value.code == 1
    assert "pkg upgrade hermes-agent" in capsys.readouterr().out
    assert mock_run.call_args_list == []
