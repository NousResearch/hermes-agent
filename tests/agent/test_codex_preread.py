"""Codex native-binary page-cache pre-read at host startup.

The pre-read must follow the npm launcher to the native executable it really spawns (the
``codex`` on PATH is usually a Node script), and must read nothing for profiles that never
selected the Codex app-server runtime.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent import codex_preread
from agent.codex_preread import start_codex_preread

NATIVE = b"\x7fELF" + b"\0" * (3 << 20)  # spans several read chunks


def _npm_install(root: Path) -> tuple[Path, Path]:
    """``<prefix>/bin/codex -> lib/node_modules/@openai/codex/bin/codex.js`` + its vendor binary."""
    pkg = root / "lib" / "node_modules" / "@openai" / "codex"
    launcher = pkg / "bin" / "codex.js"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/usr/bin/env node\n", encoding="utf-8")
    launcher.chmod(0o755)
    native = (pkg / "node_modules" / "@openai" / "codex-linux-x64" / "vendor"
              / "x86_64-unknown-linux-musl" / "bin" / "codex")
    native.parent.mkdir(parents=True)
    native.write_bytes(NATIVE)
    native.chmod(0o755)
    shim = root / "bin" / "codex"
    shim.parent.mkdir()
    shim.symlink_to(os.path.relpath(launcher, shim.parent))
    return shim, native


def _write_config(home: Path, *, runtime: str, codex_bin: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"model:\n  provider: openai-codex\n  openai_runtime: {runtime}\n  codex_bin: {codex_bin}\n", encoding="utf-8")


@pytest.fixture
def reads(monkeypatch):
    """Paths the pre-read thread opened for reading, with the bytes it pulled from each."""
    import builtins
    seen: dict[str, int] = {}
    real_open = builtins.open

    class _Counting:
        def __init__(self, fh, path):
            self._fh, self._path = fh, path

        def read(self, *args):
            data = self._fh.read(*args)
            seen[self._path] = seen.get(self._path, 0) + len(data)
            return data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._fh.close()

    def counting_open(file, mode="r", *args, **kwargs):
        fh = real_open(file, mode, *args, **kwargs)
        return _Counting(fh, os.path.realpath(file)) if "b" in mode and "r" in mode else fh

    monkeypatch.setattr(codex_preread, "open", counting_open, raising=False)
    return seen


@pytest.mark.platforms("posix")
def test_startup_reads_the_native_binary_behind_the_npm_launcher(tmp_path, monkeypatch, reads):
    shim, native = _npm_install(tmp_path / "prefix")
    home = Path(os.environ["HERMES_HOME"])
    _write_config(home, runtime="codex_app_server", codex_bin=shim)

    start_codex_preread(multiplex=False).join(timeout=30)

    assert reads.get(str(native.resolve()), 0) >= len(NATIVE)  # every byte, plus the format probe


@pytest.mark.platforms("posix")
def test_profiles_off_the_codex_runtime_read_nothing(tmp_path, monkeypatch, reads):
    shim, native = _npm_install(tmp_path / "prefix")
    home = Path(os.environ["HERMES_HOME"])
    _write_config(home, runtime="auto", codex_bin=shim)

    start_codex_preread(multiplex=True).join(timeout=30)

    assert str(native.resolve()) not in reads
