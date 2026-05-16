import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from rl_cli import check_tinker_atropos


def test_check_tinker_atropos_success(tmp_path):
    """Test successful case where submodule and environments dir exist."""
    # Setup mock file structure
    mock_parent = tmp_path
    tinker_path = mock_parent / "tinker-atropos"
    envs_path = tinker_path / "tinker_atropos" / "environments"
    envs_path.mkdir(parents=True)

    # Create some mock environment files
    (envs_path / "env1.py").touch()
    (envs_path / "env2.py").touch()
    (envs_path / "_hidden.py").touch()  # Should be ignored

    with patch("rl_cli.Path") as MockPath:
        mock_file = MagicMock()
        mock_file.parent = mock_parent
        MockPath.return_value = mock_file

        ok, result = check_tinker_atropos()

        assert ok is True
        assert result["path"] == str(tinker_path)
        assert result["environments_count"] == 2


def test_check_tinker_atropos_no_submodule(tmp_path):
    """Test failure when tinker-atropos submodule is missing."""
    mock_parent = tmp_path

    with patch("rl_cli.Path") as MockPath:
        mock_file = MagicMock()
        mock_file.parent = mock_parent
        MockPath.return_value = mock_file

        ok, result = check_tinker_atropos()

        assert ok is False
        assert "submodule not found" in result


def test_check_tinker_atropos_no_environments_dir(tmp_path):
    """Test failure when environments directory is missing."""
    mock_parent = tmp_path
    tinker_path = mock_parent / "tinker-atropos"
    tinker_path.mkdir()

    with patch("rl_cli.Path") as MockPath:
        mock_file = MagicMock()
        mock_file.parent = mock_parent
        MockPath.return_value = mock_file

        ok, result = check_tinker_atropos()

        assert ok is False
        assert "environments directory not found" in result
