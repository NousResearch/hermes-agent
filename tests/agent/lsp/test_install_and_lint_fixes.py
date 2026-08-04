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
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import pytest

from agent.lsp.install import INSTALL_RECIPES


# ---------------------------------------------------------------------------
# Fix 1: typescript install recipe carries the typescript SDK
# ---------------------------------------------------------------------------






def test_install_npm_works_without_extras(tmp_path, monkeypatch):
    """Backwards compat: pyright-style recipes (no extras) still install."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return MagicMock(returncode=0, stderr="")

    from agent.lsp import install as install_mod

    monkeypatch.setattr(install_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(install_mod.shutil, "which", lambda c: "/usr/bin/npm" if c == "npm" else None)

    install_mod._install_npm("pyright", "pyright-langserver")

    cmd = captured["cmd"]
    assert "pyright" in cmd
    # Should not blow up when extra_pkgs is omitted/None
    install_targets = [c for c in cmd if not c.startswith("-") and c not in {
        "install", "--prefix", str(install_mod.hermes_lsp_bin_dir().parent),
        "/usr/bin/npm",
    }]
    assert install_targets == ["pyright"]




@pytest.mark.windows_only
def test_install_pip_finds_windows_scripts_launcher(tmp_path, monkeypatch):
    """pip console scripts can land in Scripts/ on native Windows.

    ``windows_only``: the ``Scripts/`` layout and the ``.exe`` launcher are
    what pip actually produces on Windows. Faking ``_is_windows()`` on Linux
    made the test assert against a directory tree the test itself created, on
    a host where pip would never lay it out that way.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.lsp import install as install_mod

    def fake_run(cmd, **kwargs):
        scripts_dir = install_mod.hermes_lsp_bin_dir().parent / "python-packages" / "Scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        launcher = scripts_dir / "fake-language-server.exe"
        launcher.write_text("launcher\n")
        launcher.chmod(0o755)
        return MagicMock(returncode=0, stderr="")

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
