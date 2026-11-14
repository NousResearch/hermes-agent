"""`hermes plugins update` with no name: offer every plugin that has an update, never move one silently."""
from __future__ import annotations

import argparse

import pytest

from hermes_cli.plugins_updates import CheckResult


def _parse(argv):
    from hermes_cli.plugins_cmd import plugins_command
    from hermes_cli.subcommands.plugins import build_plugins_parser

    parser = argparse.ArgumentParser(prog="hermes")
    build_plugins_parser(parser.add_subparsers(dest="command"), cmd_plugins=plugins_command)
    return parser.parse_args(["plugins", *argv])


@pytest.fixture
def fleet(monkeypatch, tmp_path):
    """Four installed rows: two the update path can move, a PyPI row it never upgrades (#132494) and a pin."""
    from hermes_cli import plugins_cmd, plugins_cmd_update, plugins_updates

    rows = [
        CheckResult("cat", "catalog", current="a" * 40, latest="b" * 40, update_available=True),
        CheckResult("gitty", "git", current="c" * 40, latest="d" * 40, update_available=True),
        CheckResult("pypkg", "pip", current="1.0", latest="1.1", update_available=True),
        CheckResult("pinned", "git", current="e" * 40, update_available=False, reason="pinned @ eeee"),
    ]
    calls = []

    def fake_update(name, target, console, *, interactive=True):
        calls.append((name, interactive))
        if name == "cat":
            raise TimeoutError("catalog pin refused")  # not a typed refusal: the batch must still go on
        return True

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(plugins_updates, "run_checks", lambda plugins_dir, **kw: rows)
    monkeypatch.setattr(plugins_cmd, "_user_installed_plugin_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(plugins_cmd_update, "update_one", fake_update)
    return calls


def test_no_terminal_refuses_without_yes_and_updates_only_movable_rows_with_it(fleet, monkeypatch, capsys):
    from hermes_cli import plugins_cmd

    monkeypatch.setattr(plugins_cmd, "_is_tty", lambda: False)
    args = _parse(["update"])
    with pytest.raises(SystemExit) as refused:
        plugins_cmd.plugins_command(args)
    assert refused.value.code == 1 and fleet == []
    assert "--yes" in capsys.readouterr().out

    # --yes: every movable plugin goes through the single-plugin path non-interactively; the failure
    # of the first does not stop the second, and the pip row / pin are never attempted.
    with pytest.raises(SystemExit) as done:
        plugins_cmd.plugins_command(_parse(["update", "--yes"]))
    assert done.value.code == 1
    assert fleet == [("cat", False), ("gitty", False)]
    out = capsys.readouterr().out
    assert "catalog pin refused" in out.split("Summary", 1)[1]
    assert "skipped pypkg" in out and "skipped pinned" in out


def test_terminal_asks_once_and_cancel_moves_nothing(fleet, monkeypatch, capsys):
    from hermes_cli import plugins_cmd

    monkeypatch.setattr(plugins_cmd, "_is_tty", lambda: True)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or "n")
    plugins_cmd.plugins_command(_parse(["update"]))
    assert prompts == ["\nUpdate all 2? [Y/n]: "] and fleet == []
