"""Windows binary resolution must prefer the runnable ``.cmd``/``.exe``/``.bat`` wrapper over npm's
POSIX ``#!/bin/sh`` shim, which shares the bare name and fails ``CreateProcess`` with WinError 193."""

import os
import subprocess
from pathlib import Path

import pytest

from agent.lsp import install, servers


def _npm_shim_pair(bin_dir: Path, name: str) -> Path:
    """npm's Windows layout: POSIX shim under the bare name, runnable ``.cmd`` beside it (both X_OK)."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = bin_dir / name
    shim.write_text("#!/bin/sh\nexec node \"$0.js\" \"$@\"\n", encoding="utf-8")
    cmd = bin_dir / f"{name}.cmd"
    cmd.write_text("@echo off\r\necho wrapper-ran\r\n", encoding="utf-8")
    for p in (shim, cmd):
        os.chmod(p, 0o755)
    return cmd


def test_windows_resolution_picks_cmd_wrapper_from_npm_bin_over_posix_shim(tmp_path: Path, monkeypatch):
    """Drives the production entry with ``is_windows`` as data (no sys.platform patch): the pair
    lives where npm leaves it (``lsp/node_modules/.bin``), and pyright's langserver sibling keeps
    the resolved suffix instead of falling onto the bare shim."""
    monkeypatch.setattr(install, "hermes_lsp_bin_dir", lambda: tmp_path / "bin")
    (tmp_path / "bin").mkdir()
    cmd = _npm_shim_pair(tmp_path / "node_modules" / ".bin", "tool")

    assert install._existing_binary("tool", is_windows=True) == str(cmd)
    # POSIX resolution never looks in npm's bin dir and never appends wrappers.
    assert install._existing_binary("tool", is_windows=False) is None

    pyright_cmd = _npm_shim_pair(tmp_path / "node_modules" / ".bin", "pyright")
    langserver_cmd = _npm_shim_pair(tmp_path / "node_modules" / ".bin", "pyright-langserver")
    ctx = servers.ServerContext(workspace_root=str(tmp_path), binary_overrides={"pyright": [str(pyright_cmd)]})
    spec = servers._spawn_pyright(str(tmp_path), ctx)
    assert spec is not None and spec.command[0] == str(langserver_cmd)


@pytest.mark.platforms("windows")
def test_existing_binary_resolves_runnable_cmd_over_posix_shim(tmp_path: Path, monkeypatch):
    """Live: staging dir holds npm's shim AND its .cmd; the resolved path must actually run."""
    monkeypatch.setattr(install, "hermes_lsp_bin_dir", lambda: tmp_path)
    shim = tmp_path / "tool"
    shim.write_text("#!/bin/sh\nexec node \"$0.js\" \"$@\"\n", encoding="utf-8")
    (tmp_path / "tool.cmd").write_text("@echo off\r\necho wrapper-ran\r\n", encoding="utf-8")

    resolved = install._existing_binary("tool")

    assert resolved == str(tmp_path / "tool.cmd")
    out = subprocess.run([str(resolved)], capture_output=True, text=True, encoding="utf-8", errors="replace",
                         check=True, stdin=subprocess.DEVNULL)
    assert "wrapper-ran" in out.stdout


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("suffix", (".cmd", ".exe", ".bat"))
def test_existing_binary_prefers_native_candidate_across_bases(tmp_path: Path, monkeypatch, suffix):
    """A native npm wrapper must outrank a staged extensionless POSIX shim."""
    monkeypatch.setattr(install, "hermes_lsp_bin_dir", lambda: tmp_path / "lsp" / "bin")
    staged = tmp_path / "lsp" / "bin" / "tool"
    staged.parent.mkdir(parents=True)
    staged.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    os.chmod(staged, 0o755)
    npm_bin = tmp_path / "lsp" / "node_modules" / ".bin"
    npm_bin.mkdir(parents=True)
    native = npm_bin / f"tool{suffix}"
    native.write_text("wrapper", encoding="utf-8")

    assert install._existing_binary("tool") == str(native)


@pytest.mark.platforms("posix")
def test_posix_resolution_keeps_staging_base_precedence(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(install, "hermes_lsp_bin_dir", lambda: tmp_path / "lsp" / "bin")
    staged = tmp_path / "lsp" / "bin" / "tool"
    staged.parent.mkdir(parents=True)
    staged.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    os.chmod(staged, 0o755)
    npm_bin = tmp_path / "lsp" / "node_modules" / ".bin"
    npm_bin.mkdir(parents=True)
    (npm_bin / "tool").write_text("npm", encoding="utf-8")

    assert install._existing_binary("tool") == str(staged)
