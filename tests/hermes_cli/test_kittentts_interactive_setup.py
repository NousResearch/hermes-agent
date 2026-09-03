"""Regression tests for the interactive KittenTTS setup path (PR #82891).

The reviewer (egilewski, 2026-08-14) found two remaining supply-chain gaps:

- [P1] The ordinary interactive TTS setup (`hermes setup tts` ->
  ``_install_kittentts_deps``) installed the remote release directly,
  bypassing the wheel sha256 gate added to ``hermes tools post-setup
  kittentts``. A replaced/compromised release could execute arbitrary package
  code through the normal setup path.
- [P2] The verified install resolved ``soundfile`` by bare name — no version
  or hash constraint — so a compromised package index remained a
  supply-chain path.

Fix: both setup flows now route through the shared fail-closed installer
``hermes_cli.tools_config._install_kittentts_verified``, which downloads
kittentts AND soundfile from pinned URLs and refuses to install unless BOTH
sha256 digests match. These tests prove the interactive caller can never hand
unverified bytes to pip.
"""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# The sha256 pinned in hermes_cli/tools_config.py for the official 0.8.1
# kittentts wheel and soundfile 0.14.0 wheel.
KITTENTTS_PIN = "482a436c4f1f3192153710376e459ff3689517ebcda7c2b051e2fd4187b41851"
SOUNDFILE_PIN = "8ba81ae3a89fd5ab3bef8a8eb481fbbe794e806309675a89b4df48b8d31908a8"


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


class TestInteractiveKittenTtsSetup:
    def test_tampered_kittentts_wheel_never_reaches_pip(
        self, monkeypatch, capsys, no_kittentts_installed
    ):
        """The interactive caller must be as fail-closed as post-setup: a
        tampered wheel (real hashlib, arbitrary bytes) is refused before
        _pip_install is ever invoked."""
        tc = no_kittentts_installed
        pip_calls: list = []
        monkeypatch.setattr(
            tc, "_pip_install",
            lambda *a, **k: pip_calls.append(a) or SimpleNamespace(returncode=0, stderr=""),
        )
        _fake_download(
            monkeypatch, tc, b"TAMPERED WHEEL BYTES - not the real 0.8.1 artifact"
        )

        from hermes_cli.setup import _install_kittentts_deps

        ok = _install_kittentts_deps()

        captured = capsys.readouterr()
        text = captured.out + captured.err
        assert ok is False
        assert "sha256 mismatch" in text
        assert "refusing to install" in text
        assert pip_calls == []

    def test_tampered_soundfile_wheel_never_reaches_pip(
        self, monkeypatch, capsys, no_kittentts_installed
    ):
        """P2: soundfile is installed from a pinned wheel, and a tampered
        soundfile wheel also fails closed — nothing reaches pip."""
        tc = no_kittentts_installed
        pip_calls: list = []
        monkeypatch.setattr(
            tc, "_pip_install",
            lambda *a, **k: pip_calls.append(a) or SimpleNamespace(returncode=0, stderr=""),
        )

        # First download (kittentts) returns bytes whose hash matches the pin;
        # second download (soundfile) returns tampered bytes.
        real_sha256 = __import__("hashlib").sha256
        download_content = {"kittentts": b"", "soundfile": b""}

        def fake_run(cmd, **kwargs):
            i = cmd.index("-o")
            dest = cmd[i + 1]
            label = "soundfile" if "soundfile" in dest else "kittentts"
            if label == "kittentts":
                # bytes whose sha256 equals the pinned digest
                Path(dest).write_bytes(b"fake-verified-kittentts")
            else:
                Path(dest).write_bytes(b"TAMPERED SOUNDFILE BYTES")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(tc.subprocess, "run", fake_run)

        # Force hashlib.sha256 to return the pinned digest for the kittentts
        # file only; the tampered soundfile bytes hash to something else.
        import hashlib as _hl_mod

        real_sha256 = _hl_mod.sha256

        class _FakeSha256:
            def __init__(self, data=b""):
                self._data = data

            def hexdigest(self):
                if b"fake-verified-kittentts" in self._data:
                    return KITTENTTS_PIN
                return real_sha256(self._data).hexdigest()

        monkeypatch.setattr(_hl_mod, "sha256", lambda data=b"": _FakeSha256(data))

        from hermes_cli.setup import _install_kittentts_deps

        ok = _install_kittentts_deps()

        captured = capsys.readouterr()
        text = captured.out + captured.err
        assert ok is False
        assert "soundfile wheel sha256 mismatch" in text
        assert pip_calls == []

    def test_valid_wheels_reach_pip_as_verified_local_paths(
        self, monkeypatch, capsys, no_kittentts_installed
    ):
        """Both wheels verified -> pip receives local .whl paths, and
        soundfile is NOT resolved by bare name (P2)."""
        tc = no_kittentts_installed
        pip_calls: list = []

        def fake_pip(args, *, timeout=300):
            pip_calls.append(args)
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr(tc, "_pip_install", fake_pip)

        import hashlib as _hl_mod

        real_sha256 = _hl_mod.sha256

        class _FakeSha256:
            def __init__(self, data=b""):
                self._data = data

            def hexdigest(self):
                if b"VERIFIED-KITTENTTS" in self._data:
                    return KITTENTTS_PIN
                if b"VERIFIED-SOUNDFILE" in self._data:
                    return SOUNDFILE_PIN
                return real_sha256(self._data).hexdigest()

        monkeypatch.setattr(_hl_mod, "sha256", lambda data=b"": _FakeSha256(data))

        def fake_run(cmd, **kwargs):
            i = cmd.index("-o")
            dest = cmd[i + 1]
            label = "soundfile" if "soundfile" in dest else "kittentts"
            Path(dest).write_bytes(
                b"VERIFIED-SOUNDFILE" if label == "soundfile" else b"VERIFIED-KITTENTTS"
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(tc.subprocess, "run", fake_run)

        from hermes_cli.setup import _install_kittentts_deps

        ok = _install_kittentts_deps()

        captured = capsys.readouterr()
        text = captured.out + captured.err
        assert ok is True
        assert "wheels sha256 verified" in text
        assert pip_calls, "expected _pip_install to run for verified wheels"
        args = pip_calls[0]
        assert args[0] == "-U"
        # Both install args are LOCAL verified .whl paths, never a bare URL
        # or a bare package name.
        assert args[1].endswith(".whl") and "kittentts" in args[1]
        assert args[2].endswith(".whl") and "soundfile" in args[2]
        assert not any(a == "soundfile" for a in args)
