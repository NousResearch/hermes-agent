"""Passive lookup of the managed whisper.cpp executable; never provisions it."""

from pathlib import Path


def whisper_cpp_binary() -> Path | None:
    from pm import installed_package

    installed = installed_package("whispercpp-cpu")
    return installed.binary if installed else None
