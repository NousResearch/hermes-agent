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


def test_windows_stale_shim_in_staging_dir_is_quarantined_not_returned(tmp_path: Path, monkeypatch):
    """Regression for #116947: an installer that staged the bare
    ``node_modules/.bin/<name>`` file leaves a POSIX ``#!/bin/sh`` shim in ``lsp/bin/``.  On
    Windows that shim must never be returned (``CreateProcess`` fails with WinError 193); it is
    renamed aside so the runnable wrapper (or a fresh reinstall) takes over."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setattr(install, "hermes_lsp_bin_dir", lambda: bin_dir)
    shim = bin_dir / "pyright-langserver"
    shim.write_text("#!/bin/sh\nexec node \"$0.js\" \"$@\"\n", encoding="utf-8")
    cmd = bin_dir / "pyright-langserver.cmd"
    cmd.write_text("@echo off\r\necho wrapper-ran\r\n", encoding="utf-8")
    for p in (shim, cmd):
        os.chmod(p, 0o755)

    assert install._existing_binary("pyright-langserver", is_windows=True) == str(cmd)
    assert not shim.exists()
    assert (bin_dir / "pyright-langserver.quarantined-shim").exists()


def test_windows_stale_shim_with_no_wrapper_reports_missing(tmp_path: Path, monkeypatch):
    """The legacy shim alone (no wrapper beside it) resolves to missing on Windows so the
    installer runs again instead of handing ``CreateProcess`` a shell script."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setattr(install, "hermes_lsp_bin_dir", lambda: bin_dir)
    (bin_dir / "pyright-langserver").write_text("#!/bin/sh\nexec node \"$0.js\" \"$@\"\n", encoding="utf-8")

    assert install._existing_binary("pyright-langserver", is_windows=True) is None
    assert install.detect_status("pyright") == "missing"


def test_posix_bare_binary_without_shebang_still_resolves(tmp_path: Path, monkeypatch):
    """The quarantine only triggers on the ``#!`` prefix: a genuine extensionless executable
    (ELF/Mach-O/PE) keeps resolving, on both platforms."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setattr(install, "hermes_lsp_bin_dir", lambda: bin_dir)
    native = bin_dir / "gopls"
    native.write_bytes(b"\x7fELFfake-binary")
    os.chmod(native, 0o755)

    assert install._existing_binary("gopls", is_windows=True) == str(native)
    assert install._existing_binary("gopls", is_windows=False) == str(native)


def test_run_installer_never_raises(monkeypatch):
    """The installer runs on a gateway background thread: ANY escaping exception silences the
    process with only the ``npm install --prefix ...`` line in the log (#116947)."""
    monkeypatch.setattr(install.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert install._run_installer("npm", "pyright", ["npm", "install"], timeout=1) is False


@pytest.mark.windows_only
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
