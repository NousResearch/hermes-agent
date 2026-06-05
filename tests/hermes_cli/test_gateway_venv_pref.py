"""Regression: venv fallback must prefer installer-canonical ``venv/`` over ``.venv/``."""
import sys
import hermes_cli.gateway as gw


def _force_fallback(monkeypatch, root):
    # Pretend we're NOT inside a venv so the directory-probe fallback runs.
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(gw, "PROJECT_ROOT", root)


def test_prefers_canonical_venv_when_both_exist(tmp_path, monkeypatch):
    (tmp_path / "venv").mkdir()
    (tmp_path / ".venv").mkdir()
    _force_fallback(monkeypatch, tmp_path)
    assert gw._detect_venv_dir() == tmp_path / "venv"


def test_falls_back_to_dotvenv_when_only_dotvenv(tmp_path, monkeypatch):
    (tmp_path / ".venv").mkdir()
    _force_fallback(monkeypatch, tmp_path)
    assert gw._detect_venv_dir() == tmp_path / ".venv"
