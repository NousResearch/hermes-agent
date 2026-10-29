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

from agent.lsp.install import INSTALL_RECIPES
import json
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

    # Local .ts lint runs PM's npx directly, not through the terminal shell's _exec.
    with patch.object(fops, "_run_managed_node_linter", side_effect=lambda ext, path: fake_exec(path)), \
         patch.object(fops, "_local_workspace_untrusted", return_value=False):
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


@pytest.mark.platforms("windows")
def test_install_pip_finds_windows_scripts_launcher(tmp_path, monkeypatch):
    """The LSP installer keeps PM's native Windows launcher as its executable."""
    import pm
    from agent.lsp import install as install_mod

    launcher = tmp_path / "managed" / "Scripts" / "fake-language-server.exe"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("launcher\n", encoding="utf-8")
    calls = []
    def ensure(name, requirements, executable, **kwargs):
        calls.append((name, requirements, executable))
        return launcher
    monkeypatch.setattr(pm, "ensure_python_tool", ensure)
    resolved = install_mod._install_pip("fake-lsp", "fake-language-server")
    assert resolved == str(launcher)
    assert calls == [("lsp-fake-language-server", ["fake-lsp"], "fake-language-server")]


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


def test_status_json_keeps_current_main_service_contract(monkeypatch):
    from agent import lsp as lsp_module
    from agent.lsp import cli as lsp_cli
    from agent.lsp.manager import LSPService, _ClientEntry

    svc = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=5.0,
        install_strategy="manual",
        idle_timeout=0,
    )
    client = MagicMock()
    client.server_id = "pyright"
    client.workspace_root = "/owner-process"
    client.state = "running"
    client.is_running = True
    client.workspace_folders = ["/owner-process"]
    svc._clients[("pyright", "/owner-process")] = _ClientEntry(
        client=client,
        generation=7,
        leases=3,
        retiring=True,
        retire_reason="idle timeout",
        retirement_error="cleanup blocked",
    )

    monkeypatch.setattr(lsp_module, "get_service", lambda: svc)
    monkeypatch.setattr("agent.lsp.install.detect_status", lambda _pkg: "missing")

    buf = io.StringIO()
    with redirect_stdout(buf):
        assert lsp_cli._cmd_status(emit_json=True) == 0

    service = json.loads(buf.getvalue())["service"]
    assert service["enabled"] is False
    assert service["broken_retry_seconds"] == 0
    assert service["warmup_timeout"] == 0
    assert service["trusted_workspaces"] == []
    assert service["untrusted_skipped"] == []
    assert service["clients"] == [
        {
            "server_id": "pyright",
            "workspace_root": "/owner-process",
            "workspace_folders": ["/owner-process"],
            "state": "running",
            "running": True,
        }
    ]
