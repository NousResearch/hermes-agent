"""A native musl userland resolves PM to musl-native runtime artifacts (#123682)."""

from __future__ import annotations

from pathlib import Path
import struct

import pytest

from pm.lock import Lockfile
from pm.registry import all_packages, source_install_packages, walk

pytestmark = pytest.mark.platforms("linux")

REPO_ROOT = Path(__file__).resolve().parents[2]


def _elf_with_interpreter(path, loader, offset, bits=64, endian="<", interpreter_index=0):
    ident = b"\x7fELF" + bytes([2 if bits == 64 else 1, 1 if endian == "<" else 2, 1]) + bytes(9)
    count = interpreter_index + 1
    if bits == 64:
        header = struct.pack(endian + "HHIQQQIHHHHHH", 3, 0, 1, 0, 64, 0, 0, 64, 56, count, 0, 0, 0)
        program = struct.pack(endian + "IIQQQQQQ", 3, 4, offset, 0, 0, len(loader), len(loader), 1)
        program_size = 56
    else:
        header = struct.pack(endian + "HHIIIIIHHHHHH", 3, 0, 1, 0, 52, 0, 0, 52, 32, count, 0, 0, 0)
        program = struct.pack(endian + "IIIIIIII", 3, offset, 0, 0, len(loader), len(loader), 4, 1)
        program_size = 32
    prefix = ident + header + bytes(program_size * interpreter_index) + program
    image = bytearray(offset + len(loader))
    image[:len(prefix)] = prefix
    image[offset:] = loader
    path.write_bytes(image)


@pytest.mark.parametrize("bits,endian", [(32, "<"), (32, ">"), (64, "<"), (64, ">")])
@pytest.mark.parametrize("loader,is_musl", [(b"/lib64/ld-linux-x86-64.so.2\0", False), (b"/lib/ld-musl-x86_64.so.1\0", True)])
@pytest.mark.parametrize("interpreter_index", [0, 33])
def test_elf_loader_follows_interpreter_segment(tmp_path, bits, endian, loader, is_musl, interpreter_index):
    from pm import store

    binary = tmp_path / "shell"
    _elf_with_interpreter(binary, loader, 0x19000, bits, endian, interpreter_index)
    assert store._elf_loader_is_musl(binary) is is_musl


@pytest.mark.parametrize("damage", ["ident", "entry-size", "table", "interpreter"])
def test_elf_loader_returns_unknown_for_invalid_segments(tmp_path, damage):
    from pm import store

    binary = tmp_path / "shell"
    _elf_with_interpreter(binary, b"/lib64/ld-linux-x86-64.so.2\0", 0x19000)
    image = bytearray(binary.read_bytes())
    if damage == "ident":
        image[5] = 9
    elif damage == "entry-size":
        struct.pack_into("<H", image, 54, 1)
    elif damage == "table":
        image = image[:70]
    else:
        image = image[:-2]
    binary.write_bytes(image)
    assert store._elf_loader_is_musl(binary) is None


def test_musl_userland_gets_a_satisfiable_native_musl_closure(monkeypatch, tmp_path):
    from pm import store

    # /bin/sh's ELF interpreter is musl; the bootstrap Python (this glibc test
    # host's) says otherwise and must not win.
    fake_sh = tmp_path / "sh"
    _elf_with_interpreter(fake_sh, b"/lib/ld-musl-x86_64.so.1\0", 0x100)
    monkeypatch.setattr(store, "_native_linux_uses_musl",
                        lambda: store._elf_loader_is_musl(fake_sh), raising=False)
    monkeypatch.setattr(store, "_native_machine", lambda: "x86_64")
    monkeypatch.setattr(store, "_is_bionic_libc", lambda: False)

    target = store.current_target()
    assert target == "linux-x64-musl"

    lock = Lockfile(REPO_ROOT / "pm" / "lock.json")
    closure = walk(source_install_packages(all_packages()))
    assert {"python", "uv", "node"} <= {package.name for package in closure}
    for package in closure:
        assert package.missing_reason(target) is None, package.name
        if lock.version(package.name):
            assert lock.artifacts(package.name, target), f"{package.name} has no {target} artifact"
    for name in ("python", "uv", "node"):
        assert "musl" in lock.artifacts(name, target)[0]["url"], name
