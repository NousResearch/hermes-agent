"""Tests for the PowerShellEditorServices (PSES) server registration.

PSES is unusual among the registry entries: it's a PowerShell module
bundle (GitHub release zip) driven by a ``pwsh`` bootstrap script, not a
single binary on PATH. These tests cover the registry wiring plus the
two-prerequisite spawn logic (pwsh host + module bundle).
"""
from __future__ import annotations

import os

import agent.lsp.servers as srv
from agent.lsp.install import detect_status
from agent.lsp.servers import (
    ServerContext,
    find_server_for_file,
    language_id_for,
)


def test_powershell_extensions_route_to_pses():
    for ext in ("script.ps1", "module.psm1", "manifest.psd1"):
        s = find_server_for_file(ext)
        assert s is not None, ext
        assert s.server_id == "powershell"










def _make_fake_bundle(root) -> str:
    bundle = root / "PowerShellEditorServices"
    inner = bundle / "PowerShellEditorServices"
    inner.mkdir(parents=True)
    (inner / "Start-EditorServices.ps1").write_text("# fake")
    return str(bundle)


def test_spawn_builds_command_with_bundle_via_env(monkeypatch, tmp_path):
    monkeypatch.setattr(srv, "_which", lambda *names: "/usr/bin/pwsh")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    bundle = _make_fake_bundle(tmp_path)
    monkeypatch.setenv("PSES_BUNDLE_PATH", bundle)

    ctx = ServerContext(workspace_root=str(tmp_path), install_strategy="manual")
    spec = srv._spawn_powershell_es(str(tmp_path), ctx)
    assert spec is not None
    assert spec.command[0] == "/usr/bin/pwsh"
    assert "-Stdio" in spec.command[-1]
    assert "Start-EditorServices.ps1" in spec.command[-1]
    assert bundle in spec.command[-1]
    # -NonInteractive / -NoProfile keep the host from hanging on a prompt.
    assert "-NonInteractive" in spec.command
    assert "-NoProfile" in spec.command




def test_bundle_path_init_override_not_leaked_into_init_options(monkeypatch, tmp_path):
    monkeypatch.setattr(srv, "_which", lambda *names: "/usr/bin/pwsh")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.delenv("PSES_BUNDLE_PATH", raising=False)
    bundle = _make_fake_bundle(tmp_path)

    ctx = ServerContext(
        workspace_root=str(tmp_path),
        install_strategy="manual",
        init_overrides={"powershell": {"bundlePath": bundle, "foo": "bar"}},
    )
    spec = srv._spawn_powershell_es(str(tmp_path), ctx)
    assert spec is not None
    # bundlePath is a Hermes-internal resolution key — it must not be sent
    # to the server as an LSP initializationOption.
    assert "bundlePath" not in spec.initialization_options
    assert spec.initialization_options.get("foo") == "bar"
