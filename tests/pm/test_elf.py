"""pm.elf rewrites a RUNPATH where it sits, so a staged binary keeps its size and layout."""
from __future__ import annotations

import pytest

from pm.elf import ElfError, runpath, set_runpath
from tests.termux_fixtures import elf_with_runpath

OLD = "/data/data/com.termux/files/usr/lib:/data/data/com.termux/files/usr/lib"


def test_a_shorter_runpath_replaces_the_old_one_and_moves_nothing(tmp_path):
    binary = tmp_path / "python3.14"
    before = elf_with_runpath(OLD)
    binary.write_bytes(before)
    assert runpath(binary) == OLD

    set_runpath(binary, "$ORIGIN/../lib")

    after = binary.read_bytes()
    assert runpath(binary) == "$ORIGIN/../lib"
    start = before.index(OLD.encode())
    assert after[:start] == before[:start]
    assert after[start + len(OLD):] == before[start + len(OLD):]


def test_a_runpath_that_does_not_fit_leaves_the_file_alone(tmp_path):
    binary = tmp_path / "python3.14"
    before = elf_with_runpath("/short")
    binary.write_bytes(before)
    with pytest.raises(ElfError, match="does not fit"):
        set_runpath(binary, "/a/much/longer/path")
    assert binary.read_bytes() == before


def test_a_file_that_is_not_an_elf_is_refused(tmp_path):
    binary = tmp_path / "python3.14"
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    with pytest.raises(ElfError):
        runpath(binary)


def _damaged():
    good = elf_with_runpath(OLD)
    at = good.index(OLD.encode())
    return {
        "cut inside the header": good[:40],
        "program headers past the end": good[:0x20] + (10**6).to_bytes(8, "little") + good[0x28:],
        "runpath with no terminator": good[:at + len(OLD)],
        "runpath that is not text": good.replace(OLD.encode()[:4], b"/\xff\xfe/"),
    }


@pytest.mark.parametrize("damage", sorted(_damaged()))
def test_a_damaged_elf_is_reported_as_one(tmp_path, damage):
    binary = tmp_path / "python3.14"
    binary.write_bytes(_damaged()[damage])
    with pytest.raises(ElfError):
        runpath(binary)


def test_a_runpath_with_a_nul_is_refused(tmp_path):
    binary = tmp_path / "python3.14"
    before = elf_with_runpath(OLD)
    binary.write_bytes(before)
    with pytest.raises(ElfError):
        set_runpath(binary, "$ORIGIN/../lib\0/elsewhere")
    assert binary.read_bytes() == before
