"""The Desktop dependency installer repairs npm's macOS get-windows omission."""

import subprocess
from types import SimpleNamespace

import pytest

import hermes_cli.main as main
import hermes_cli.main_desktop as main_desktop
import hermes_cli.main_web_build as main_web_build


@pytest.fixture()
def darwin(monkeypatch):
    monkeypatch.setattr(main_desktop.sys, "platform", "darwin")


def test_get_windows_install_failure_is_narrow(darwin):
    assert main_desktop._get_windows_install_failure(
        subprocess.CompletedProcess(
            [], 1, stdout="npm error get-windows: node-pre-gyp ERR! install response status 404", stderr=""
        )
    )
    assert not main_desktop._get_windows_install_failure(
        subprocess.CompletedProcess([], 1, stdout="npm error electron dist is missing", stderr="")
    )


def test_get_windows_install_failure_is_disabled_off_macos(monkeypatch):
    monkeypatch.setattr(main_desktop.sys, "platform", "linux")
    assert not main_desktop._get_windows_install_failure(
        subprocess.CompletedProcess([], 1, stdout="get-windows node-pre-gyp 404", stderr="")
    )


def test_repair_get_windows_uses_workspace_install_without_scripts(tmp_path, monkeypatch):
    package = tmp_path / "apps" / "desktop" / "node_modules" / "get-windows" / "package.json"
    package.parent.mkdir(parents=True)
    package.write_text('{"name":"get-windows","version":"9.3.0"}\n', encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        main_desktop.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr("hermes_constants.with_hermes_node_path", lambda env=None: env or {})
    monkeypatch.setattr(main_desktop, "_npm_lifecycle_env", lambda env: env)

    assert main_desktop._repair_get_windows_install("/fake/npm", tmp_path, {})
    command, kwargs = calls[0]
    assert command == [
        "/fake/npm",
        "install",
        "--workspace",
        "apps/desktop",
        "--ignore-scripts",
        "--include=optional",
        "--no-save",
        "get-windows@9.3.0",
    ]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["check"] is False


def test_desktop_install_repairs_only_matching_macos_failure(tmp_path, monkeypatch, darwin):
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main_desktop, "_nixos_build_env", lambda: None)
    monkeypatch.setattr("hermes_constants.with_hermes_node_path", lambda env=None: env or {})
    monkeypatch.setattr(main_desktop, "_npm_lifecycle_env", lambda env: env)
    monkeypatch.setattr(
        main_web_build,
        "_run_npm_install_deterministic",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="npm error get-windows node-pre-gyp ERR! install response status 404",
            stderr="",
        ),
    )
    repaired = []
    monkeypatch.setattr(
        main_desktop,
        "_repair_get_windows_install",
        lambda *args: repaired.append(args) or True,
    )

    main_desktop._install_desktop_workspace_deps("/fake/npm", {})
    assert len(repaired) == 1


def test_desktop_install_keeps_unrelated_failure_fatal(tmp_path, monkeypatch, darwin):
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main_desktop, "_nixos_build_env", lambda: None)
    monkeypatch.setattr("hermes_constants.with_hermes_node_path", lambda env=None: env or {})
    monkeypatch.setattr(
        main_web_build,
        "_run_npm_install_deterministic",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="npm error unable to resolve Electron",
            stderr="",
        ),
    )
    monkeypatch.setattr(main_desktop, "_electron_pkg_staged_missing_dist", lambda _root: False)
    monkeypatch.setattr(main_desktop, "_repair_get_windows_install", lambda *args: pytest.fail("unexpected repair"))

    with pytest.raises(SystemExit) as exc:
        main_desktop._install_desktop_workspace_deps("/fake/npm", {})
    assert exc.value.code == 1
