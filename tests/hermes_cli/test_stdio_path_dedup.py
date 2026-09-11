"""Regression tests for PATH dedup normalization in ``hermes_cli.stdio``.

On Windows the same directory can appear on PATH under several spellings — case
variants, ``/`` vs ``\\`` separators, trailing separators (MSYS auto-translation
artifacts). ``_augment_path_with_known_tools`` used to dedup with a bare
``str.lower()`` comparison, which missed those spellings, so every Python
startup prepended the venv ``Scripts`` dir again — issue #108508 observed 70+
consecutive duplicates in one bash session snapshot. The dedup now compares
``_path_key()`` keys (``ntpath`` semantics + ``casefold``), which equate all of
those spellings. The key function is pure string math, so these tests run on
any host.
"""

from __future__ import annotations

import os

import hermes_cli.stdio as stdio
from hermes_cli.stdio import _path_key


def test_path_key_case_insensitive():
    assert _path_key("C:\\Hermes\\venv\\Scripts") == _path_key("c:\\hermes\\VENV\\scripts")


def test_path_key_forward_and_back_slashes_equivalent():
    assert _path_key("C:\\Hermes\\venv\\Scripts") == _path_key("C:/Hermes/venv/Scripts")


def test_path_key_trailing_separator_ignored():
    assert _path_key("C:\\Hermes\\venv\\Scripts\\") == _path_key("C:\\Hermes\\venv\\Scripts")


def test_path_key_dot_segments_resolved():
    assert _path_key("C:\\Hermes\\venv\\..\\venv\\Scripts") == _path_key("C:\\Hermes\\venv\\Scripts")


def test_path_key_different_dirs_stay_different():
    assert _path_key("C:\\a\\Scripts") != _path_key("C:\\a\\Scripts2")


def test_augment_skips_dir_already_on_path_in_another_spelling(monkeypatch, tmp_path):
    """The issue-#108508 snowball: PATH already holds the venv Scripts dir in a
    trailing-separator / mixed-case variant, so the old ``str.lower()`` dedup
    missed it and prepended a fresh duplicate on every startup."""
    monkeypatch.setattr(stdio, "is_windows", lambda: True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    scripts = tmp_path / "hermes" / "hermes-agent" / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    variant = str(scripts).replace("Scripts", "SCRIPTS") + os.sep
    monkeypatch.setenv("PATH", variant)

    stdio._augment_path_with_known_tools()

    # Nothing new to prepend: PATH must come back exactly as it went in.
    assert os.environ["PATH"] == variant


def test_augment_still_prepends_a_missing_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(stdio, "is_windows", lambda: True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    scripts = tmp_path / "hermes" / "hermes-agent" / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    monkeypatch.setenv("PATH", "/baseline/bin")

    stdio._augment_path_with_known_tools()

    entries = os.environ["PATH"].split(os.pathsep)
    assert str(scripts) in entries
    # Prepend semantics: Hermes-managed dirs win collisions with the ambient PATH.
    assert entries.index(str(scripts)) < entries.index("/baseline/bin")


def test_augment_keeps_case_only_match_deduped(monkeypatch, tmp_path):
    """A plain case variant was already caught by the old ``str.lower()`` dedup;
    the normalized key must keep that behaviour."""
    monkeypatch.setattr(stdio, "is_windows", lambda: True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    scripts = tmp_path / "hermes" / "hermes-agent" / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    variant = str(scripts).replace("Scripts", "scripts")
    monkeypatch.setenv("PATH", variant)

    stdio._augment_path_with_known_tools()

    assert os.environ["PATH"] == variant
