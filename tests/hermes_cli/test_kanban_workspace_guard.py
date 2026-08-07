from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main


@pytest.fixture
def kanban_env(monkeypatch, tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_guard")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    return workspace


def _ns(**kw):
    defaults = dict(
        skip_build=False,
        build_only=False,
        force_build=False,
        source=False,
        fake_boot=False,
        ignore_existing=False,
        hermes_root=None,
        cwd=None,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_guard_allows_targets_inside_workspace(kanban_env: Path) -> None:
    target = kanban_env / "repo" / "apps" / "desktop"
    target.mkdir(parents=True)
    cli_main._assert_kanban_workspace_owns_path(target, action="desktop npm/build")


def test_guard_rejects_targets_outside_workspace(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = tmp_path / "live-root" / "apps" / "desktop"
    target.mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_guard")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))

    with pytest.raises(SystemExit) as exc:
        cli_main._assert_kanban_workspace_owns_path(target, action="desktop npm/build")

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "outside the active kanban workspace" in err
    assert str(workspace) in err
    assert str(target) in err


def test_make_tui_argv_rejects_live_checkout_path(
    kanban_env: Path, monkeypatch, tmp_path: Path, capsys
) -> None:
    tui_dir = tmp_path / "live-root" / "ui-tui"
    tui_dir.mkdir(parents=True)
    (tui_dir / "package.json").write_text("{}")

    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setattr(cli_main, "_ensure_tui_node", lambda: None)
    monkeypatch.setattr(cli_main, "_tui_need_npm_install", lambda _root: False)
    monkeypatch.setattr(cli_main, "_tui_need_rebuild", lambda _root: False)

    with pytest.raises(SystemExit) as exc:
        cli_main._make_tui_argv(tui_dir, tui_dev=False)

    assert exc.value.code == 1
    assert "outside the active kanban workspace" in capsys.readouterr().err


def test_build_web_ui_rejects_live_checkout_path(
    kanban_env: Path, tmp_path: Path, capsys
) -> None:
    web_dir = tmp_path / "live-root" / "web"
    web_dir.mkdir(parents=True)
    (web_dir / "package.json").write_text("{}")

    with pytest.raises(SystemExit) as exc:
        cli_main._build_web_ui(web_dir)

    assert exc.value.code == 1
    assert "outside the active kanban workspace" in capsys.readouterr().err


def test_cmd_gui_rejects_live_checkout_path(
    kanban_env: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    root = tmp_path / "live-root"
    desktop_dir = root / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "package.json").write_text("{}")

    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(cli_main, "_desktop_build_needed", lambda *_args, **_kwargs: True)

    with patch("hermes_constants.find_node_executable", return_value="/usr/bin/npm"), \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 1
    assert "outside the active kanban workspace" in capsys.readouterr().err
