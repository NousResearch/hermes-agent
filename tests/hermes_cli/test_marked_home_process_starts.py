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


def _program(folder: Path, name: str) -> tuple[Path, Path]:
    """A program the profile's backend could have written: it appends a line to ``<name>.ran``."""
    ran = folder / f"{name}.ran"
    if os.name == "nt":
        program = folder / f"{name}.cmd"
        program.write_text(f'@echo off\r\necho ran>>"{ran}"\r\n', encoding="utf-8")
    else:
        program = folder / name
        program.write_text(f'#!/bin/sh\necho ran >> "{ran}"\n', encoding="utf-8")
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


@pytest.mark.parametrize("left", ["bytes that are not text", "a folder"])
def test_a_marked_profiles_container_mode_file_is_not_opened(home, monkeypatch, capsys, left):
    """What the backend leaves under ``<home>/.container-mode`` need not be a file the host can read: bytes
    that are not UTF-8, or a folder of that name. In a marked home the host does not open it, so the command
    runs to its normal end."""
    import hermes_constants

    ops = home / "profiles" / "ops"
    if left == "a folder":
        (ops / ".container-mode").mkdir()
    else:
        (ops / ".container-mode").write_bytes(b"backend=\xff\xfe\n")
    monkeypatch.delenv("HERMES_DEV", raising=False)
    monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
    _mark(home)
    assert _hermes(monkeypatch, "config", "path") == 0
    assert str(ops / "config.yaml") in capsys.readouterr().out


def test_a_container_mode_file_written_during_the_start_is_not_used(home, tmp_path, monkeypatch, capsys):
    """The backend of a marked profile writes its home at any moment, between two steps of one start too.
    Here ``.container-mode`` is gone at every look the host takes and back before every read. It is the
    backend's all the same: no program is started for it and the command runs to its normal end."""
    import hermes_constants
    from hermes_cli import main_container

    program, ran = _program(tmp_path, "runtime")
    ops = home / "profiles" / "ops"
    mode = ops / ".container-mode"
    looked = os.path.lexists

    def gone_at_the_look_and_back_before_the_read(path) -> bool:
        if Path(path) != mode:
            return looked(path)
        mode.unlink(missing_ok=True)
        there = looked(path)
        mode.write_text(f"backend={program}\ncontainer_name=stand-in\n", encoding="utf-8")
        return there

    monkeypatch.setattr(os.path, "lexists", gone_at_the_look_and_back_before_the_read)
    monkeypatch.delenv("HERMES_DEV", raising=False)
    monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
    handed_over: list[list[str]] = []
    monkeypatch.setattr(main_container.os, "execvp", lambda file, argv: handed_over.append(list(argv)))
    _mark(home)
    assert _hermes(monkeypatch, "config", "path") == 0
    assert str(ops / "config.yaml") in capsys.readouterr().out
    assert mode.exists(), "the host took no look, so the stand-in backend wrote nothing"
    assert not ran.exists() and not handed_over


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(("argv", "ends_with"), [(("config", "path"), 0), (("serve",), 78)], ids=["config path", "serve"])
def test_a_marked_profiles_container_mode_fifo_is_not_opened(home, monkeypatch, argv, ends_with):
    """A FIFO under ``<home>/.container-mode`` holds whoever opens it until a writer comes. In a marked home no
    host step opens it, the fast chat and serve start-up paths included, so the command runs to its normal end:
    ``config path`` prints the path, ``serve`` refuses the marked profile."""
    import threading

    import hermes_constants

    fifo = home / "profiles" / "ops" / ".container-mode"
    os.mkfifo(fifo)
    for name in ("HERMES_DEV", "HERMES_DISABLE_FAST_CHAT_LAUNCH", "HERMES_DISABLE_FAST_SERVE_LAUNCH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
    _mark(home)
    finished, released = threading.Event(), []

    def release_blocked_readers() -> None:  # a regression fails the test instead of hanging it
        while not finished.wait(5):
            try:
                writer = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
            except OSError:  # ENXIO: nothing has it open for reading
                continue
            released.append(True)  # before the close lets the blocked reader go on
            os.close(writer)

    threading.Thread(target=release_blocked_readers, daemon=True).start()
    try:
        code = _hermes(monkeypatch, *argv)
    finally:
        finished.set()
    assert not released, "a host step opened the FIFO"
    assert code == ends_with
