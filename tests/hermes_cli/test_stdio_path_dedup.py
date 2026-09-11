"""PATH dedup in ``hermes_cli/stdio.py`` must treat spelling variants of the same
Windows directory as equal.

``_augment_path_with_known_tools`` runs at every Python startup and prepends the
Hermes-managed tool dirs when they are missing from PATH. The old membership
check compared entries via ``entry.lower()``, so ``C:\\hermes\\venv\\Scripts`` and
``C:\\hermes\\venv\\Scripts\\`` (or an MSYS-translated ``c:/hermes/venv/Scripts``)
counted as different directories: every startup prepended another copy, and the
bash session snapshot accumulated dozens of consecutive duplicates
(issue #108508).

The fakes below are string-level on purpose: ``_augment_path_with_known_tools``
is exercised with Windows-shaped values (``LOCALAPPDATA`` with backslashes, a
PATH mixing separators and case) while ``os.path.isdir`` is stubbed, because the
behavior under test is pure string membership, not filesystem semantics.
"""

import os

import pytest

from hermes_cli import stdio as stdio_mod
from hermes_cli.stdio import _augment_path_with_known_tools, _windows_path_key

_FAKE_APPDATA = "C:\\HermesTest"


class TestWindowsPathKey:
    def test_spelling_variants_share_one_key(self):
        assert _windows_path_key("C:\\Foo\\") == _windows_path_key("c:/foo")
        assert _windows_path_key("c:\\FOO") == _windows_path_key("C:/foo/")
        assert _windows_path_key(r"C:\Foo\Bar") == _windows_path_key("C:\\FOO\\bar")

    def test_distinct_directories_stay_distinct(self):
        assert _windows_path_key("C:\\hermes\\git\\bin") != _windows_path_key(
            "C:\\hermes\\git\\usr\\bin"
        )

    def test_empty_is_identity(self):
        assert _windows_path_key("") == ""


class TestAugmentPathDedup:
    """Repeated startups must not stack spelling-variant duplicates."""

    @pytest.fixture
    def fake_windows(self, monkeypatch):
        monkeypatch.setattr(stdio_mod, "is_windows", lambda: True)
        # Windows-shaped PATH semantics while running on any host: ";" as the
        # entry separator and case/backslash-insensitive comparison keys.
        monkeypatch.setattr(os, "pathsep", ";")
        monkeypatch.setenv("LOCALAPPDATA", _FAKE_APPDATA)
        monkeypatch.setattr(
            os.path,
            "isdir",
            lambda p: (
                "venv\\scripts"
                in str(p).lower().replace("venv/scripts", "venv\\scripts")
            ),
        )

    def _venv_entries(self):
        return [
            e
            for e in os.environ["PATH"].split(os.pathsep)
            if "scripts" in e.lower() and "hermestest" in e.lower()
        ]

    def test_no_prepend_when_variants_already_present(self, fake_windows, monkeypatch):
        variants = os.pathsep.join([
            _FAKE_APPDATA + "\\hermes\\hermes-agent\\venv\\Scripts\\",
            "c:/hermestest/hermes/hermes-agent/venv/Scripts",
        ])
        monkeypatch.setenv("PATH", variants)
        _augment_path_with_known_tools()
        assert len(self._venv_entries()) == 2  # исходные варианты, ничего не добавлено

    def test_repeated_startups_do_not_accumulate(self, fake_windows, monkeypatch):
        monkeypatch.setenv("PATH", "c:/HERMESTEST/hermes/hermes-agent/venv/Scripts")
        for _ in range(5):
            _augment_path_with_known_tools()
        assert len(self._venv_entries()) == 1

    def test_missing_dir_is_prepended_once(self, fake_windows, monkeypatch):
        monkeypatch.setenv("PATH", "C:\\Windows\\System32")
        _augment_path_with_known_tools()
        _augment_path_with_known_tools()
        entries = self._venv_entries()
        assert len(entries) == 1
        assert os.environ["PATH"].startswith(entries[0])
