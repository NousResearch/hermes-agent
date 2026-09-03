"""Focused tests for _canonical_config_path_key symlink-loop fallback."""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.config import _canonical_config_path_key


def test_canonical_config_path_key_returns_string_for_normal_path(tmp_path):
    """Normal path resolves without error and returns a non-empty string."""
    p = tmp_path / "config.yaml"
    key = _canonical_config_path_key(p)
    assert isinstance(key, str)
    assert key  # non-empty


def test_canonical_config_path_key_falls_back_on_oserror(monkeypatch):
    """OSError during resolve → absolute path string, no crash."""
    p = Path("/some/nonexistent/path/config.yaml")

    def _raising_resolve(*_args, **_kwargs):
        raise OSError("No such file or directory")

    monkeypatch.setattr(Path, "resolve", _raising_resolve)

    key = _canonical_config_path_key(p)
    assert isinstance(key, str)
    assert "config.yaml" in key, "fallback key should contain the filename"


def test_canonical_config_path_key_falls_back_on_runtime_error(monkeypatch):
    """RuntimeError (symlink loop) during resolve → absolute path, no crash."""
    p = Path("/symlink/loop/config.yaml")

    def _raising_resolve(*_args, **_kwargs):
        raise RuntimeError("Symlink loop detected")

    monkeypatch.setattr(Path, "resolve", _raising_resolve)

    key = _canonical_config_path_key(p)
    assert isinstance(key, str)
    assert "config.yaml" in key, (
        "_canonical_config_path_key must return an absolute fallback "
        "identity on RuntimeError, not propagate the exception"
    )
