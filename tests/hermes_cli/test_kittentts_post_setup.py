"""Regression tests for the KittenTTS post-setup install path (PR #82891).

PR #82891 pinned the kittentts wheel's sha256 so a tampered third-party
release can't be installed silently. The reviewer found the new download
call raised ``UnboundLocalError: cannot access local variable 'subprocess'``
on the first-time install path: ``_run_post_setup`` binds ``subprocess`` as
a *local* name (several branches ``import subprocess`` locally), but the
``kittentts`` branch never imported it before calling ``subprocess.run(...)``.

These tests exercise the missing-module path end to end and pin both sides
of the hash gate: mismatch -> refuse to install; valid hash -> install the
wheel. Before the fix every one of them raised UnboundLocalError, which is
exactly how the bug stayed masked.
"""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# The sha256 pinned in hermes_cli/tools_config.py for the official 0.8.1
# wheel. Tests must NOT depend on it matching real bytes — the fake download
# writes arbitrary bytes and the fake hashlib returns this digest on demand.
PIN_HEX = "482a436c4f1f3192153710376e459ff3689517ebcda7c2b051e2fd4187b41851"


@pytest.fixture
def no_kittentts_installed(monkeypatch):
    """Force the 'kittentts not installed' path through __import__."""
    import hermes_cli.tools_config as tc

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "kittentts":
            raise ImportError(f"No module named {name!r} (test stub)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return tc


def _fake_download(monkeypatch, tc, content: bytes, returncode: int = 0) -> list:
    """Stub subprocess.run as the curl download; writes content to -o dest."""
    calls: list = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        i = cmd.index("-o")
        Path(cmd[i + 1]).write_bytes(content)
        return SimpleNamespace(returncode=returncode, stdout="", stderr="curl stub")

    monkeypatch.setattr(tc.subprocess, "run", fake_run)
    return calls


class TestKittenTtsPostSetup:
    def test_download_failure_returns_without_install(
        self, monkeypatch, capsys, no_kittentts_installed
    ):
        """A failed download must warn and return without calling pip."""
        tc = no_kittentts_installed
        pip_calls: list = []
        monkeypatch.setattr(
            tc, "_pip_install",
            lambda *a, **k: pip_calls.append(a) or SimpleNamespace(returncode=0, stderr=""),
        )
        _fake_download(monkeypatch, tc, b"", returncode=7)

        tc._run_post_setup("kittentts")

        captured = capsys.readouterr()
        text = captured.out + captured.err
        assert "download failed" in text
        assert pip_calls == []

    def test_tampered_wheel_refused_before_install(
        self, monkeypatch, capsys, no_kittentts_installed
    ):
        """Regression: first-time install must reach the hash gate (no
        UnboundLocalError) and a tampered wheel must be refused without
        calling _pip_install."""
        tc = no_kittentts_installed
        pip_calls: list = []
        monkeypatch.setattr(
            tc, "_pip_install",
            lambda *a, **k: pip_calls.append(a) or SimpleNamespace(returncode=0, stderr=""),
        )
        _fake_download(
            monkeypatch, tc, b"TAMPERED WHEEL BYTES - not the real 0.8.1 artifact"
        )

        tc._run_post_setup("kittentts")

        captured = capsys.readouterr()
        text = captured.out + captured.err
        assert "sha256 mismatch" in text
        assert "refusing to install" in text
        assert pip_calls == []

    def test_valid_hash_installs_wheel(
        self, monkeypatch, capsys, no_kittentts_installed
    ):
        """A wheel whose hash matches the pin must be handed to pip."""
        tc = no_kittentts_installed

        # The helper does `import hashlib as _hl`; point it at a stub whose
        # sha256() returns the pinned digest for both wheels (kittentts AND
        # soundfile) so the valid-hash path is taken for the whole install.
        import hashlib as _real_hl_mod

        KITTENTTS_PIN = "482a436c4f1f3192153710376e459ff3689517ebcda7c2b051e2fd4187b41851"
        SOUNDFILE_PIN = "8ba81ae3a89fd5ab3bef8a8eb481fbbe794e806309675a89b4df48b8d31908a8"

        class _FakeSha256:
            def __init__(self, data=b""):
                self._data = data

            def hexdigest(self):
                if b"verified kittentts bytes" in self._data:
                    return KITTENTTS_PIN
                if b"verified soundfile bytes" in self._data:
                    return SOUNDFILE_PIN
                return _real_hl_mod.sha256(self._data).hexdigest()

        monkeypatch.setattr(
            _real_hl_mod, "sha256", lambda data=b"": _FakeSha256(data)
        )

        pip_calls: list = []

        def fake_pip(args, *, timeout=300):
            pip_calls.append(args)
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr(tc, "_pip_install", fake_pip)

        def fake_run(cmd, **kwargs):
            i = cmd.index("-o")
            dest = Path(cmd[i + 1])
            dest.write_bytes(
                b"verified soundfile bytes"
                if "soundfile" in dest.name
                else b"verified kittentts bytes"
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(tc.subprocess, "run", fake_run)

        tc._run_post_setup("kittentts")

        captured = capsys.readouterr()
        text = captured.out + captured.err
        assert "wheels sha256 verified" in text
        assert pip_calls, "expected _pip_install to run for a valid hash"
        # _pip_install(["-U", wheel_path, soundfile_path, "--quiet"], ...)
        assert pip_calls[0][1].endswith(".whl")
        assert "soundfile" in pip_calls[0][2]

    def test_already_installed_skips_download(self, monkeypatch, capsys):
        """If kittentts is importable, no download/hash/install work runs."""
        import hermes_cli.tools_config as tc

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "kittentts":
                return MagicMock()
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        run_calls: list = []
        monkeypatch.setattr(
            tc.subprocess, "run",
            lambda *a, **k: run_calls.append(a)
            or SimpleNamespace(returncode=0, stdout="", stderr=""),
        )

        tc._run_post_setup("kittentts")

        assert run_calls == []
        captured = capsys.readouterr()
        assert "already installed" in (captured.out + captured.err)
