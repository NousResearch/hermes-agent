"""Regression tests for #49665 — venv directory resolution must honor ``.venv``.

``_venv_scripts_dir()`` previously hardcoded ``PROJECT_ROOT / "venv"`` and
returned None for installs using the ``.venv`` convention (the name
``python -m venv`` suggests). That silently disabled the Windows update
quarantine and concurrent-instance detection that key off it.
"""

from __future__ import annotations

from unittest.mock import patch

import hermes_cli.main as main


def test_resolve_finds_plain_venv(tmp_path):
    (tmp_path / "venv").mkdir()
    with patch.object(main, "PROJECT_ROOT", tmp_path):
        assert main._resolve_project_venv_dir() == tmp_path / "venv"


def test_resolve_finds_dot_venv(tmp_path):
    (tmp_path / ".venv").mkdir()
    with patch.object(main, "PROJECT_ROOT", tmp_path):
        assert main._resolve_project_venv_dir() == tmp_path / ".venv"


def test_resolve_prefers_plain_venv_over_dot_venv(tmp_path):
    (tmp_path / "venv").mkdir()
    (tmp_path / ".venv").mkdir()
    with patch.object(main, "PROJECT_ROOT", tmp_path):
        assert main._resolve_project_venv_dir() == tmp_path / "venv"


def test_resolve_none_when_absent(tmp_path):
    with patch.object(main, "PROJECT_ROOT", tmp_path):
        assert main._resolve_project_venv_dir() is None


def test_scripts_dir_resolves_for_dot_venv(tmp_path):
    scripts = tmp_path / ".venv" / "bin"
    scripts.mkdir(parents=True)
    with patch.object(main, "PROJECT_ROOT", tmp_path), patch.object(
        main, "_is_windows", return_value=False
    ):
        assert main._venv_scripts_dir() == scripts


def test_scripts_dir_resolves_for_dot_venv_windows(tmp_path):
    scripts = tmp_path / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    with patch.object(main, "PROJECT_ROOT", tmp_path), patch.object(
        main, "_is_windows", return_value=True
    ):
        assert main._venv_scripts_dir() == scripts


def test_scripts_dir_none_when_no_venv(tmp_path):
    with patch.object(main, "PROJECT_ROOT", tmp_path):
        assert main._venv_scripts_dir() is None
