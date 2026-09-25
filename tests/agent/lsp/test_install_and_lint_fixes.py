"""Tests for follow-up fixes to the LSP integration (PR after #24168).

Covers:

1. ``hermes lsp status`` surfaces a ``Backend warnings`` section when
   bash-language-server is installed but ``shellcheck`` is missing.
2. ``_check_lint`` returns ``skipped`` (not ``error``) when the linter
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


def test_install_python_server_uses_pm_tool_environment(tmp_path, monkeypatch):
    import pm
    from agent.lsp import install as install_mod

    binary = tmp_path / "environment" / "fake-language-server"
    calls = []
    selected = []

    def ensure(name, requirements, executable, **kwargs):
        calls.append((name, requirements, executable, kwargs))
        selected.append(binary)
        return binary

    monkeypatch.setattr(pm, "ensure_python_tool", ensure)
    monkeypatch.setattr(pm, "python_tool", lambda *a, **kw: selected[0] if selected else None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(install_mod, "INSTALL_RECIPES", {
        "fake-lsp": {"strategy": "pip", "pkg": "fake-lsp==1.0", "bin": "fake-language-server"},
    })
    monkeypatch.setattr(install_mod, "_install_results", {})
    monkeypatch.setattr(install_mod.shutil, "which", lambda *a, **kw: None)
    assert install_mod.try_install("fake-lsp") == str(binary)
    assert calls == [("lsp-fake-language-server", ["fake-lsp==1.0"], "fake-language-server", {"timeout": 300})]
    assert install_mod.detect_status("fake-lsp") == "installed"




def test_check_lint_returns_error_for_real_ts_type_errors(tmp_path, monkeypatch):
    """Sanity: real TypeScript errors still go through the error path."""
    from pathlib import Path

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
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


def test_lsp_package_manager_config_selects_installer_argv_and_never_falls_back_silently(tmp_path, monkeypatch):
    """``lsp.package_manager`` picks the Node installer (staging-dir semantics kept); a configured manager
    that is missing or unknown skips the install instead of quietly using npm (a typo must not bypass policy)."""
    from unittest.mock import MagicMock

    from agent.lsp import install as install_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    staging = str(install_mod.hermes_lsp_bin_dir().parent)
    cfg = {"lsp": {}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)
    runs = []
    monkeypatch.setattr(install_mod.subprocess, "run", lambda cmd, **kw: (runs.append(cmd), MagicMock(returncode=0, stderr=""))[1])
    present = {"npm": "/usr/bin/npm", "pnpm": "/usr/bin/pnpm", "yarn": "/usr/bin/yarn"}
    monkeypatch.setattr(install_mod, "find_node_executable", lambda name: present.get(name))

    cfg["lsp"] = {"package_manager": "pnpm"}
    install_mod._install_npm("pyright", "pyright-langserver")
    assert runs[-1] == ["/usr/bin/pnpm", "add", "--dir", staging, "pyright"]

    cfg["lsp"] = {"package_manager": "yarn"}  # global --cwd: valid on Yarn Classic and Berry
    install_mod._install_npm("pyright", "pyright-langserver")
    assert runs[-1] == ["/usr/bin/yarn", "--cwd", staging, "add", "pyright"]

    cfg["lsp"] = {"package_manager": "pnmp"}  # unknown (typo) → fail closed, no npm run
    assert install_mod._install_npm("pyright", "pyright-langserver") is None
    del present["yarn"]
    cfg["lsp"] = {"package_manager": "yarn"}  # configured but absent → no install, no npm run
    assert install_mod._install_npm("pyright", "pyright-langserver") is None
    assert len(runs) == 2


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
