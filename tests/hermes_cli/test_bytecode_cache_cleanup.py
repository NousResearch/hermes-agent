"""Tests for bytecode cache cleanup helpers in hermes_cli.main."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main


def _make_pycache(root: Path, *parts: str) -> Path:
    """Create a __pycache__ directory under root/parts."""
    p = root.joinpath(*parts, "__pycache__")
    p.mkdir(parents=True)
    (p / "foo.cpython-311.pyc").write_text("old")
    return p


def test_clear_bytecode_cache_removes_pycache_and_skips_ignored_dirs(tmp_path):
    """__pycache__ directories are removed; venv/node_modules/.git are skipped."""
    keep = [
        _make_pycache(tmp_path, "venv"),
        _make_pycache(tmp_path, "node_modules", "pkg"),
        _make_pycache(tmp_path, ".git"),
        _make_pycache(tmp_path, ".worktrees", "wt"),
    ]
    remove = [
        _make_pycache(tmp_path, "hermes_cli"),
        _make_pycache(tmp_path, "tools", "registry"),
    ]

    removed = cli_main._clear_bytecode_cache(tmp_path)

    assert removed == len(remove)
    for p in remove:
        assert not p.exists()
    for p in keep:
        assert p.exists()


def test_clear_hermes_home_bytecode_cache_uses_get_hermes_home(tmp_path):
    """Home-wide cleanup delegates to _clear_bytecode_cache using HERMES_HOME."""
    home_cache = _make_pycache(tmp_path, "profiles", "operator", "scripts")
    src_cache = _make_pycache(tmp_path, "hermes-agent", "hermes_cli")
    venv_cache = _make_pycache(tmp_path, "hermes-agent", "venv", "lib")

    with patch("hermes_cli.main.get_hermes_home", return_value=tmp_path):
        removed = cli_main._clear_hermes_home_bytecode_cache()

    assert removed == 2
    assert not home_cache.exists()
    assert not src_cache.exists()
    assert venv_cache.exists()


def test_clear_hermes_home_bytecode_cache_no_home(tmp_path):
    """If get_hermes_home returns a non-existent path, cleanup is a no-op."""
    missing = tmp_path / "missing"
    with patch("hermes_cli.main.get_hermes_home", return_value=missing):
        assert cli_main._clear_hermes_home_bytecode_cache() == 0