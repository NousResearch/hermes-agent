"""The real uninstall parser must select the data path, never code removal."""
from __future__ import annotations

import argparse
import pytest

from hermes_cli import main, uninstall
from hermes_cli.subcommands.uninstall import build_uninstall_parser
from tests.hermes_cli.test_data_uninstall import layout  # noqa: F401 — isolated home


@pytest.mark.parametrize("entry", ["cli", "module", "cli-dry-run"])
def test_data_entrypoints_preserve_runtime(layout, monkeypatch, entry):
    _, witnesses, data = layout

    def no_code_removal(*args, **kwargs):
        pytest.fail("data-only dispatch reached code removal")

    monkeypatch.setattr(uninstall, "run_uninstall", no_code_removal)
    monkeypatch.setattr(uninstall, "run_gui_uninstall", no_code_removal)
    monkeypatch.setattr(main, "_require_tty", no_code_removal)
    if entry == "module":
        assert uninstall.main(["--mode", "data"]) == 0
    else:
        parser = argparse.ArgumentParser()
        build_uninstall_parser(parser.add_subparsers(dest="command"), cmd_uninstall=main.cmd_uninstall)
        args = parser.parse_args(["uninstall", "--data", "--dry-run" if entry == "cli-dry-run" else "--yes"])
        args.func(args)
    assert all(path.exists() for path in witnesses)
    assert all(path.exists() is (entry == "cli-dry-run") for path in data)


@pytest.mark.parametrize("other", ["--full", "--gui"])
def test_conflicting_data_mode_is_rejected_before_dispatch(other):
    parser = argparse.ArgumentParser()
    build_uninstall_parser(parser.add_subparsers(dest="command"), cmd_uninstall=main.cmd_uninstall)
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["uninstall", "--data", other, "--yes"])
    assert error.value.code == 2


def test_module_subprocess_removes_only_the_disposable_home(layout, tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    home, witnesses, data = layout
    root = Path(uninstall.__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items() if key.upper() in {
        "PATH", "SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "TEMP", "TMP",
    }}
    env.update(HERMES_HOME=str(home), HERMES_RUNTIME_DIR=str(home / "machine" / "tool-store"),
               HOME=str(tmp_path), USERPROFILE=str(tmp_path), APPDATA=str(tmp_path),
               LOCALAPPDATA=str(tmp_path), PYTHONPATH=str(root), PYTHONUTF8="1")
    result = subprocess.run([sys.executable, "-m", "hermes_cli.uninstall", "--mode", "data"],
                            cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    # This process's actual source is outside the temp home, unlike layout's
    # declared source. Managed tools, launchers and sibling profiles survive.
    assert all(path.exists() for path in witnesses[1:])
    assert all(not path.exists() for path in data)
    assert not (home / "skills").exists()
    assert not (home / "SOUL.md").exists()


def test_module_uses_the_desktop_instances_explicit_data_path(layout, tmp_path):
    _, _, _ = layout
    ordinary = tmp_path / "user" / "desktop-data"
    variant = tmp_path / "variant-desktop-data"
    for directory in (ordinary, variant):
        directory.mkdir()
        (directory / "preferences.json").write_text("{}", encoding="utf-8")
    assert uninstall.main(["--mode", "data", "--desktop-userdata", str(variant)]) == 0
    assert not variant.exists()
    assert (ordinary / "preferences.json").read_text(encoding="utf-8") == "{}"
