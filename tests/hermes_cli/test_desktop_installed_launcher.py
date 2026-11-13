"""``hermes desktop`` on an install that ships the app instead of its source.

A Nix (or vendored) install has no ``apps/desktop`` to build from: the
Electron app is its own artifact, launched as ``hermes-desktop``. ``cmd_gui``
must hand over to that app instead of dead-ending with
"Desktop GUI source not found" — and must still refuse to fake a build when a
build flag was asked for, or when the tree is a checkout.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main
from hermes_cli import main_desktop as cli_desktop


def _site_packages_install(tmp_path: Path, *, steward: str | None = None) -> Path:
    """An install whose code lives in site-packages, like pip and Nix do."""
    root = tmp_path / "lib" / "python3.14" / "site-packages"
    root.mkdir(parents=True)
    (root / "hermes_cli").mkdir()
    if steward is not None:
        (root / "install-stamp.json").write_text(
            json.dumps({"schemaVersion": 2, "distribution": steward, "updateMechanism": "external"}),
            encoding="utf-8",
        )
    return root


def _checkout(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    (root / "hermes_cli").mkdir(parents=True)
    (root / ".git").mkdir()
    return root


def _installed_app(directory: Path) -> Path:
    launcher = directory / "hermes-desktop"
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return launcher


def _args(**overrides) -> SimpleNamespace:
    base = dict(
        source=False,
        build_only=False,
        fake_boot=False,
        ignore_existing=False,
        hermes_root=None,
        cwd=None,
        skip_build=False,
        force_build=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _run(monkeypatch, args, *, root: Path, app_dir: Path | None = None):
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root, raising=False)
    monkeypatch.setenv("PATH", str(app_dir) if app_dir is not None else str(root.parent / "empty-bin"))
    launches: list[list[str]] = []

    def record_popen(argv, **kwargs):
        launches.append([str(a) for a in argv])
        return SimpleNamespace(pid=4242)

    with patch("hermes_cli.bundled_app.subprocess.Popen", side_effect=record_popen), pytest.raises(
        SystemExit
    ) as exit_info:
        cli_desktop.cmd_gui(args)
    return exit_info.value.code, launches


def _app_dir(tmp_path: Path) -> Path:
    app_dir = tmp_path / "bin"
    app_dir.mkdir()
    _installed_app(app_dir)
    return app_dir


def test_launches_the_installed_app_instead_of_failing(tmp_path, monkeypatch, capsys):
    root = _site_packages_install(tmp_path, steward="nix")
    app_dir = _app_dir(tmp_path)

    code, launches = _run(monkeypatch, _args(), root=root, app_dir=app_dir)

    assert code == 0
    assert launches == [[str(app_dir / "hermes-desktop")]]
    assert f"Launched the installed Hermes Desktop app: {app_dir / 'hermes-desktop'}" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["source", "build_only", "force_build"])
def test_a_build_flag_still_reports_the_missing_source(tmp_path, monkeypatch, capsys, flag):
    """A build was asked for; handing over to another app would silently drop it."""
    root = _site_packages_install(tmp_path, steward="nix")

    code, launches = _run(monkeypatch, _args(**{flag: True}), root=root, app_dir=_app_dir(tmp_path))

    assert code == 1
    assert launches == []
    assert "Desktop GUI source not found at:" in capsys.readouterr().out


def test_a_checkout_missing_its_source_never_delegates(tmp_path, monkeypatch, capsys):
    """A checkout that lost apps/desktop is broken; its error names its own path."""
    root = _checkout(tmp_path)

    code, launches = _run(monkeypatch, _args(), root=root, app_dir=_app_dir(tmp_path))

    assert code == 1
    assert launches == []
    out = capsys.readouterr().out
    assert f"Desktop GUI source not found at: {root / 'apps' / 'desktop'}" in out


def test_a_nix_install_names_its_own_desktop_package(tmp_path, monkeypatch, capsys):
    """The remedy must be the one this install can perform, not a pip/Homebrew one."""
    root = _site_packages_install(tmp_path, steward="nix")

    code, launches = _run(monkeypatch, _args(), root=root)

    assert code == 1
    assert launches == []
    out = capsys.readouterr().out
    assert "runs from the Nix store" in out
    assert "programs.hermes-agent.desktop.enable = true" in out
    assert "came from a Python package" not in out


def test_a_plain_pip_install_keeps_the_python_package_guidance(tmp_path, monkeypatch, capsys):
    root = _site_packages_install(tmp_path)

    code, _launches = _run(monkeypatch, _args(), root=root)

    assert code == 1
    out = capsys.readouterr().out
    assert "came from a Python package" in out
    assert "Nix store" not in out