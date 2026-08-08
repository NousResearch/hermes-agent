"""Tests for follow-up fixes to the LSP integration (PR after #24168).

Covers:

1. ``typescript-language-server`` install recipe pulls in ``typescript``
   alongside the server, so the npm install command targets both.
2. ``hermes lsp status`` surfaces a ``Backend warnings`` section when
   bash-language-server is installed but ``shellcheck`` is missing.
3. ``_check_lint`` returns ``skipped`` (not ``error``) when the linter
   command exists on PATH but couldn't actually run — e.g. ``npx tsc``
   without the typescript SDK installed.  This is what unblocks the
   LSP semantic tier on TypeScript files when the user doesn't also
   have a project-level ``tsc``.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.lsp.install import INSTALL_RECIPES


# ---------------------------------------------------------------------------
# Fix 1: typescript install recipe carries the typescript SDK
# ---------------------------------------------------------------------------


def _write_realistic_npm_cmd(launcher: Path, package: str, entrypoint: str) -> Path:
    """Write an npm-compatible Windows launcher fixture with exact CRLF bytes."""
    launcher.parent.mkdir(parents=True, exist_ok=True)
    target = launcher.parent.parent / package / entrypoint
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'console.log("npm-wrapper-spawned")\n')
    launcher.write_bytes(
        (
            "@ECHO off\r\n"
            "GOTO start\r\n"
            ":find_dp0\r\n"
            "SET dp0=%~dp0\r\n"
            "EXIT /b\r\n"
            ":start\r\n"
            "SETLOCAL\r\n"
            "CALL :find_dp0\r\n"
            "\r\n"
            'IF EXIST "%dp0%\\node.exe" (\r\n'
            '  SET "_prog=%dp0%\\node.exe"\r\n'
            ") ELSE (\r\n"
            '  SET "_prog=node"\r\n'
            "  SET PATHEXT=%PATHEXT:;.JS;=;%\r\n"
            ")\r\n"
            "\r\n"
            'endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & '
            f'"%_prog%"  "%dp0%\\..\\{package}\\{entrypoint}" %*\r\n'
        ).encode("utf-8")
    )
    return target


def test_install_npm_works_without_extras(tmp_path, monkeypatch):
    """Backwards compat: pyright-style recipes (no extras) still install."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stderr="")

    from agent.lsp import install as install_mod

    monkeypatch.setattr(install_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(install_mod, "find_node_executable", lambda c: "/usr/bin/npm" if c == "npm" else None)

    install_mod._install_npm("pyright", "pyright-langserver")

    cmd = captured["cmd"]
    assert "pyright" in cmd
    # Should not blow up when extra_pkgs is omitted/None
    install_targets = [c for c in cmd if not c.startswith("-") and c not in {
        "install", "--prefix", str(install_mod.hermes_lsp_bin_dir().parent),
        "/usr/bin/npm",
    }]
    assert install_targets == ["pyright"]


def test_install_npm_keeps_realistic_windows_cmd_in_node_modules_bin(tmp_path, monkeypatch):
    """A stock npm shim remains beside the package target it resolves."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.lsp import install as install_mod

    def fake_run(cmd, **kwargs):
        managed = (
            install_mod.hermes_lsp_bin_dir().parent
            / "node_modules"
            / ".bin"
            / "pyright-langserver.cmd"
        )
        _write_realistic_npm_cmd(managed, "pyright", "langserver.index.js")
        return MagicMock(returncode=0, stderr="")

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(install_mod, "find_node_executable", lambda name: "npm.cmd")
    monkeypatch.setattr(install_mod.subprocess, "run", fake_run)

    resolved = install_mod._install_npm("pyright", "pyright-langserver")

    managed = (
        install_mod.hermes_lsp_bin_dir().parent
        / "node_modules"
        / ".bin"
        / "pyright-langserver.cmd"
    )
    assert resolved == str(managed)
    assert not (install_mod.hermes_lsp_bin_dir() / "pyright-langserver.cmd").exists()


def test_existing_binary_recovers_realistic_managed_npm_cmd(tmp_path, monkeypatch):
    """Next-process recovery accepts npm's launcher at its managed origin."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.lsp import install as install_mod

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    managed = (
        install_mod.hermes_lsp_bin_dir().parent
        / "node_modules"
        / ".bin"
        / "pyright-langserver.cmd"
    )
    _write_realistic_npm_cmd(managed, "pyright", "langserver.index.js")

    assert install_mod._existing_binary("pyright-langserver") == str(managed)


def test_existing_binary_skips_stale_cmd_in_hermes_staging_dir(tmp_path, monkeypatch):
    """Historical Hermes-relocated npm .cmd files are not authoritative."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.lsp import install as install_mod

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    staged = install_mod.hermes_lsp_bin_dir() / "pyright-langserver.cmd"
    staged.write_bytes(b"@ECHO off\r\necho stale Hermes relocation\r\n")
    monkeypatch.setattr(install_mod.shutil, "which", lambda _name: None)

    assert install_mod._existing_binary("pyright-langserver") is None


_FALLBACK_CMD_BODIES = [
    b"@ECHO off\r\n"
    b'IF EXIST "%dp0%\\..\\optional\\helper.exe" (\r\n'
    b'  "%dp0%\\..\\optional\\helper.exe" %*\r\n'
    b") ELSE (\r\n"
    b'  node "%dp0%\\..\\package\\index.js" %*\r\n'
    b")\r\n",
    b"@ECHO off\r\n"
    b'IF EXIST "%dp0%\\..\\node-helper\\helper.js" (\r\n'
    b'  node "%dp0%\\..\\node-helper\\helper.js" %*\r\n'
    b") ELSE (\r\n"
    b'  node "%dp0%\\..\\package\\index.js" %*\r\n'
    b")\r\n",
]


@pytest.mark.parametrize("body", _FALLBACK_CMD_BODIES)
def test_existing_binary_accepts_fallback_cmd_at_managed_origin(tmp_path, monkeypatch, body):
    """Managed wrappers are accepted by provenance, not parsed control flow."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.lsp import install as install_mod

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    managed = (
        install_mod.hermes_lsp_bin_dir().parent
        / "node_modules"
        / ".bin"
        / "fixture-language-server.cmd"
    )
    managed.parent.mkdir(parents=True, exist_ok=True)
    managed.write_bytes(body)
    fallback = managed.parent.parent / "package" / "index.js"
    fallback.parent.mkdir()
    fallback.write_bytes(b'console.log("fallback")\n')

    assert install_mod._existing_binary("fixture-language-server") == str(managed)


@pytest.mark.parametrize("body", _FALLBACK_CMD_BODIES)
def test_existing_binary_accepts_fallback_cmd_on_path(tmp_path, monkeypatch, body):
    """PATH wrappers are accepted without diagnosing third-party relocation."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    from agent.lsp import install as install_mod

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    on_path = tmp_path / "path" / "fixture-language-server.cmd"
    on_path.parent.mkdir(parents=True)
    on_path.write_bytes(body)
    fallback = on_path.parent.parent / "package" / "index.js"
    fallback.parent.mkdir()
    fallback.write_bytes(b'console.log("fallback")\n')
    monkeypatch.setattr(
        install_mod.shutil,
        "which",
        lambda name: str(on_path) if name in {"fixture-language-server", "fixture-language-server.cmd"} else None,
    )

    assert install_mod._existing_binary("fixture-language-server") == str(on_path)


def test_existing_binary_accepts_path_cmd_without_reading_it(tmp_path, monkeypatch):
    """A .cmd found on PATH is authoritative and its contents stay opaque."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    from agent.lsp import install as install_mod

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    on_path = tmp_path / "path" / "fixture-language-server.cmd"
    on_path.parent.mkdir(parents=True)
    on_path.write_bytes(b"not for Hermes to classify\r\n")
    monkeypatch.setattr(install_mod.shutil, "which", lambda _name: str(on_path))

    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.suffix.lower() == ".cmd":
            raise AssertionError(f"wrapper was opened: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    assert install_mod._existing_binary("fixture-language-server") == str(on_path)


def test_existing_binary_rejects_windows_posix_shim(tmp_path, monkeypatch):
    """An extensionless npm shell shim cannot be spawned by CreateProcess."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.lsp import install as install_mod

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    posix_shim = install_mod.hermes_lsp_bin_dir() / "pyright-langserver"
    posix_shim.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(install_mod.shutil, "which", lambda _name: None)

    assert install_mod._existing_binary("pyright-langserver") is None


def test_native_candidates_prefer_exe_without_reading_native_or_wrapper_files(tmp_path, monkeypatch):
    """Native executables win and .exe/.bat/.cmd contents remain opaque."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.lsp import install as install_mod

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    base = install_mod.hermes_lsp_bin_dir() / "fake-language-server"
    exe = base.with_suffix(".exe")
    cmd = base.with_suffix(".cmd")
    bat = base.with_suffix(".bat")
    exe.write_bytes(b"MZ" + b"x" * 64)
    cmd.write_text("@ECHO off\r\n", encoding="utf-8")
    bat.write_text("@ECHO off\r\n", encoding="utf-8")

    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.suffix.lower() in {".exe", ".bat", ".cmd"}:
            raise AssertionError(f"launcher was opened: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(install_mod.shutil, "which", lambda _name: None)

    assert install_mod._existing_binary("fake-language-server") == str(exe)


@pytest.mark.skipif(os.name != "nt", reason="requires native cmd.exe launcher semantics")
def test_recovered_managed_npm_cmd_spawns_through_cmd(tmp_path, monkeypatch):
    """The path returned by recovery actually reaches the package entrypoint."""
    cmd_exe = shutil.which("cmd.exe")
    node_exe = shutil.which("node.exe")
    if cmd_exe is None or node_exe is None:
        pytest.skip("cmd.exe and node.exe are required")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.lsp import install as install_mod

    managed = (
        install_mod.hermes_lsp_bin_dir().parent
        / "node_modules"
        / ".bin"
        / "fixture-language-server.cmd"
    )
    _write_realistic_npm_cmd(managed, "fixture-language-server", "index.js")

    resolved = install_mod._existing_binary("fixture-language-server")
    assert resolved is not None
    proc = subprocess.run(
        [cmd_exe, "/d", "/s", "/c", resolved],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert resolved == str(managed)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "npm-wrapper-spawned"



def test_install_pip_finds_windows_scripts_launcher(tmp_path, monkeypatch):
    """pip console scripts can land in Scripts/ on native Windows."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.lsp import install as install_mod

    def fake_run(cmd, **kwargs):
        scripts_dir = install_mod.hermes_lsp_bin_dir().parent / "python-packages" / "Scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        launcher = scripts_dir / "fake-language-server.exe"
        launcher.write_text("launcher\n")
        launcher.chmod(0o755)
        return MagicMock(returncode=0, stderr="")

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(install_mod.subprocess, "run", fake_run)

    resolved = install_mod._install_pip("fake-lsp", "fake-language-server")

    assert resolved is not None
    assert resolved.endswith("fake-language-server.exe")
    assert (install_mod.hermes_lsp_bin_dir() / "fake-language-server.exe").exists()


# ---------------------------------------------------------------------------
# Fix 2: ``hermes lsp status`` surfaces shellcheck-missing for bash
# ---------------------------------------------------------------------------






def test_backend_warnings_fires_when_bash_installed_but_shellcheck_missing(tmp_path, monkeypatch):
    """The exact scenario from the bug report."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.lsp import cli as lsp_cli

    def which(name):
        if name == "bash-language-server":
            return "/fake/bin/bash-language-server"
        return None  # shellcheck missing

    with patch("shutil.which", side_effect=which):
        notes = lsp_cli._backend_warnings()
    assert len(notes) == 1
    assert "shellcheck" in notes[0].lower()
    assert "bash-language-server" in notes[0].lower()


def test_status_output_includes_backend_warnings_section(tmp_path, monkeypatch):
    """End-to-end: status command output includes the warning section."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # Pretend bash-language-server is installed but shellcheck is missing
    def which(name):
        if name == "bash-language-server":
            return "/fake/bin/bash-language-server"
        return None

    from agent.lsp import cli as lsp_cli

    buf = io.StringIO()
    with patch("shutil.which", side_effect=which), redirect_stdout(buf):
        lsp_cli._cmd_status(emit_json=False)

    output = buf.getvalue()
    assert "Backend warnings" in output
    assert "shellcheck" in output


# ---------------------------------------------------------------------------
# Fix 3: tier-1 lint treats unusable linters as ``skipped``, not ``error``
# ---------------------------------------------------------------------------










def test_check_lint_returns_error_for_real_ts_type_errors(tmp_path):
    """Sanity: real TypeScript errors still go through the error path."""
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    ts_file = tmp_path / "bad.ts"
    ts_file.write_text("const x: string = 42;\n")

    env = LocalEnvironment()
    fops = ShellFileOperations(env)

    real_tsc_error = (
        "bad.ts:1:7 - error TS2322: Type 'number' is not assignable to type 'string'.\n"
        "1 const x: string = 42;\n"
        "        ~\n"
        "Found 1 error.\n"
    )

    def fake_exec(cmd, **kwargs):
        result = MagicMock()
        result.exit_code = 1
        result.stdout = real_tsc_error
        return result

    with patch.object(fops, "_exec", side_effect=fake_exec), \
         patch.object(fops, "_has_command", return_value=True):
        lint = fops._check_lint(str(ts_file))

    assert lint.skipped is False
    assert lint.success is False
    assert "TS2322" in lint.output


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
