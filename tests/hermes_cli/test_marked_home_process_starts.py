"""A profile a plugin runs elsewhere names no program that this host starts.

#156725 says a host process starts no command the profile's own files name: the backend the plugin runs
elsewhere writes ``config.yaml`` and most of the profile home. These are the places where a host process
started such a program without building an agent, each through one file a backend can write. For a profile
that runs here every one of them starts its program as before, and each test says so.

The stand-in programs here only append a line to a file under ``tmp_path``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import profiles_run_elsewhere as pre

PLUGIN = "sandbox-plugin"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A Hermes root with the named profile ``ops`` as this process's home (not marked yet)."""
    root = tmp_path / ".hermes"
    ops = root / "profiles" / "ops"
    ops.mkdir(parents=True)
    (root / "config.yaml").write_text("{}\n", encoding="utf-8")
    (ops / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(ops))
    return root


def _mark(root: Path) -> None:
    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=root)


def _runs_here_again(root: Path) -> None:
    assert pre.clear_profile_run_elsewhere("ops", PLUGIN, root=root)


def _program(folder: Path, name: str, says: str = "") -> tuple[Path, Path]:
    """A program the profile's backend could have written: it appends a line to ``<name>.ran`` (and prints
    *says*)."""
    ran = folder / f"{name}.ran"
    if os.name == "nt":
        program = folder / f"{name}.cmd"
        program.write_text(f'@echo off\r\necho ran>>"{ran}"\r\n' + (f"echo {says}\r\n" if says else ""),
                           encoding="utf-8")
    else:
        program = folder / name
        program.write_text(f'#!/bin/sh\necho ran >> "{ran}"\n' + (f"echo {says}\n" if says else ""),
                           encoding="utf-8")
        program.chmod(0o755)
    return program, ran


# -- <home>/.container-mode -----------------------------------------------------------------------------

def _hermes(monkeypatch, *argv: str) -> int:
    """Run ``hermes <argv>`` through the CLI's entry point in this process; its exit code."""
    import sys

    from hermes_cli import main as hermes_main

    monkeypatch.setattr(sys, "argv", ["hermes", *argv])
    try:
        hermes_main.main()
    except SystemExit as exit_:
        return 0 if exit_.code is None else exit_.code
    return 0


def test_a_marked_profiles_container_mode_file_is_not_used(home, tmp_path, monkeypatch, capsys):
    """``<home>/.container-mode`` names the container runtime (``backend=``, a name or a path) that every
    ``hermes`` command is handed to before it is parsed. In a marked home the file is the backend's own, so the
    command stays a host command: no program is started for it and it runs to its normal end."""
    import hermes_constants
    from hermes_cli import main_container

    program, ran = _program(tmp_path, "runtime")
    ops = home / "profiles" / "ops"
    (ops / ".container-mode").write_text(f"backend={program}\ncontainer_name=stand-in\n", encoding="utf-8")
    monkeypatch.delenv("HERMES_DEV", raising=False)
    monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
    handed_over: list[list[str]] = []  # the real call replaces this process with the program
    monkeypatch.setattr(main_container.os, "execvp", lambda file, argv: handed_over.append(list(argv)))
    _mark(home)
    assert _hermes(monkeypatch, "config", "path") == 0
    assert str(ops / "config.yaml") in capsys.readouterr().out
    assert not ran.exists() and not handed_over
    _runs_here_again(home)
    _hermes(monkeypatch, "config", "path")
    assert ran.exists() and Path(handed_over[0][0]).name == program.name and handed_over[0][-2:] == ["config", "path"]
