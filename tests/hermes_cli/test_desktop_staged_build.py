"""Lifecycle contracts for staged Desktop package builds."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main
from hermes_cli.subcommands.gui import build_gui_parser
from hermes_cli.subcommands.update import build_update_parser


def _parser(builder, handler=lambda _args: None):
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    keyword = "cmd_gui" if builder is build_gui_parser else "cmd_update"
    builder(subs, **{keyword: handler})
    return parser


def _ns(**overrides):
    values = {
        "build_only": True,
        "output_dir": None,
        "source": False,
        "skip_build": False,
        "force_build": False,
        "fake_boot": False,
        "ignore_existing": False,
        "hermes_root": None,
        "cwd": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _desktop_tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    desktop = root / "apps" / "desktop"
    desktop.mkdir(parents=True)
    (desktop / "package.json").write_text("{}", encoding="utf-8")
    return root, desktop


def test_internal_update_skip_flag_parses_but_is_hidden():
    parser = _parser(build_update_parser)
    args = parser.parse_args(["update", "--skip-desktop-build"])
    assert args.skip_desktop_build is True
    assert "--skip-desktop-build" not in parser.format_help()


def test_desktop_output_dir_parser_propagates_absolute_path(tmp_path):
    parser = _parser(build_gui_parser)
    stage = tmp_path / "transaction"
    args = parser.parse_args(["desktop", "--build-only", "--output-dir", str(stage)])
    assert args.output_dir == str(stage)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"build_only": False}, "--build-only"),
        ({"source": True}, "--source"),
        ({"skip_build": True}, "--skip-build"),
        ({"output_dir": "relative/stage"}, "absolute"),
    ],
)
def test_staged_output_rejects_invalid_combinations(tmp_path, overrides, match):
    _root, desktop = _desktop_tree(tmp_path)
    values = {"output_dir": str(tmp_path / "stage"), **overrides}
    args = _ns(**values)
    with pytest.raises(ValueError, match=match):
        cli_main._resolve_desktop_output_dir(args, desktop)


def test_staged_output_rejects_live_release_and_aliases(tmp_path):
    _root, desktop = _desktop_tree(tmp_path)
    release = desktop / "release"
    release.mkdir()

    for unsafe in (release, release / "nested", desktop, desktop.parent):
        with pytest.raises(ValueError, match="unsafe"):
            cli_main._resolve_desktop_output_dir(_ns(output_dir=str(unsafe)), desktop)

    alias = tmp_path / "release-alias"
    alias.symlink_to(release, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        cli_main._resolve_desktop_output_dir(_ns(output_dir=str(alias)), desktop)


def test_staged_build_forces_builder_output_and_does_not_stamp_live_state(tmp_path, monkeypatch):
    root, desktop = _desktop_tree(tmp_path)
    stage = tmp_path / "stage"
    exe = stage / "mac-arm64" / "Hermes.app" / "Contents" / "MacOS" / "Hermes"
    exe.parent.mkdir(parents=True)
    exe.write_text("binary", encoding="utf-8")

    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    build_ok = subprocess.CompletedProcess(["npm"], 0)

    with patch("hermes_cli.main._resolve_node_runtime_npm", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main._desktop_build_needed") as build_needed, \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._write_desktop_build_stamp") as stamp, \
         patch("hermes_cli.main.subprocess.run", side_effect=[build_ok, build_ok]) as run:
        cli_main.cmd_gui(_ns(output_dir=str(stage)))

    build_needed.assert_not_called()
    stamp.assert_not_called()
    assert run.call_args_list[0].args[0] == ["/usr/bin/npm", "run", "build"]
    assert run.call_args_list[1].args[0] == [
        "/usr/bin/npm",
        "run",
        "builder",
        "--",
        "--dir",
        f"-c.directories.output={stage.resolve()}",
    ]


def test_normal_desktop_build_keeps_default_pack_and_stamp(tmp_path, monkeypatch):
    root, _desktop = _desktop_tree(tmp_path)
    exe = root / "apps" / "desktop" / "release" / "mac-arm64" / "Hermes.app" / "Contents" / "MacOS" / "Hermes"
    exe.parent.mkdir(parents=True)
    exe.write_text("binary", encoding="utf-8")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")

    ok = subprocess.CompletedProcess(["npm"], 0)
    with patch("hermes_cli.main._resolve_node_runtime_npm", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=ok), \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._desktop_processes_reading_release", return_value=[]), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._write_desktop_build_stamp") as stamp, \
         patch("hermes_cli.main.subprocess.run", return_value=ok) as run:
        cli_main.cmd_gui(_ns(output_dir=None))

    assert run.call_args_list[0].args[0] == ["/usr/bin/npm", "run", "pack"]
    stamp.assert_called_once()


def test_normal_macos_build_refuses_to_mutate_release_with_live_reader(tmp_path, monkeypatch):
    root, _desktop = _desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)

    with patch("hermes_cli.main._resolve_node_runtime_npm", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._desktop_processes_reading_release", return_value=[1234]), \
         patch("hermes_cli.main.subprocess.run") as run:
        with pytest.raises(SystemExit) as exc:
            cli_main.cmd_gui(_ns(output_dir=None))

    assert exc.value.code == 2
    run.assert_not_called()


def test_normal_macos_build_refuses_when_reader_scan_is_unknown(tmp_path, monkeypatch):
    root, _desktop = _desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)

    with patch("hermes_cli.main._resolve_node_runtime_npm", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._desktop_processes_reading_release", return_value=None), \
         patch("hermes_cli.main.subprocess.run") as run:
        with pytest.raises(SystemExit) as exc:
            cli_main.cmd_gui(_ns(output_dir=None))

    assert exc.value.code == 2
    run.assert_not_called()


def test_desktop_reader_scan_returns_none_when_psutil_missing(monkeypatch, tmp_path):
    root, desktop = _desktop_tree(tmp_path)
    (desktop / "release").mkdir()
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert cli_main._desktop_processes_reading_release(desktop) is None
