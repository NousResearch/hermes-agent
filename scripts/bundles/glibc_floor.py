"""Fail a packaged Linux desktop app that needs a newer glibc than its arch's floor.

Every ELF file under the unpacked app (Electron, its libraries and the agent
payload) declares the GLIBC_* and GLIBCXX_* symbol versions it needs in its
``.gnu.version_r`` section. The loader refuses to start a binary whose host
lacks one of them, so a single newer requirement drops a whole distribution.

Floors: x64 runs on Ubuntu 22.04 and Debian 12, arm64 on Ubuntu 24.04 and
Debian 13.

    python -m scripts.bundles.glibc_floor --arch x64 apps/desktop/release/linux-unpacked
"""
from __future__ import annotations

import argparse
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

Version = tuple[int, ...]

FLOORS: dict[str, dict[str, Version]] = {
    "x64": {"GLIBC": (2, 35), "GLIBCXX": (3, 4, 30)},
    "arm64": {"GLIBC": (2, 39), "GLIBCXX": (3, 4, 33)},
}

# Upstream builds the x64 CUDA and ROCm llama.cpp engines on Ubuntu 24.04, so
# those engines run only on 24.04 and newer; every other file keeps the floor.
EXCEPTIONS: dict[str, tuple[tuple[re.Pattern[str], dict[str, Version]], ...]] = {
    "x64": ((re.compile(r"(^|/)agent-payload/tools/llamacpp-(cuda|hip)-[^/]+-linux-x64/"),
             {"GLIBC": (2, 38), "GLIBCXX": (3, 4, 32)}),),
}

_SHT_GNU_VERNEED = 0x6FFFFFFE
_VERSION = re.compile(r"(GLIBC|GLIBCXX)_(\d+(?:\.\d+)*)")


@dataclass(frozen=True)
class Violation:
    """A need above its ceiling, or (ceiling None) a file whose needs can't be read, with ``need`` saying why."""
    path: str
    need: str
    ceiling: str | None

    def __str__(self) -> str:
        if self.ceiling is None:
            return f"{self.path}: can't read its version needs ({self.need})"
        return f"{self.path} needs {self.need} (above {self.ceiling})"


def parse_need(name: str) -> tuple[str, Version] | None:
    """``GLIBCXX_3.4.30`` -> ``("GLIBCXX", (3, 4, 30))``; other version names -> None."""
    match = _VERSION.fullmatch(name)
    if match is None:
        return None
    return match[1], tuple(int(part) for part in match[2].split("."))


def _read(handle: BinaryIO, offset: int, size: int) -> bytes:
    handle.seek(offset)
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("truncated ELF file")
    return data


def _sections(handle: BinaryIO, ident: bytes) -> tuple[str, list[tuple[int, ...]]]:
    """The byte order and every section header as (name, type, flags, addr, offset, size, link, info, ...)."""
    wide = {1: False, 2: True}.get(ident[4])
    order = {1: "<", 2: ">"}.get(ident[5])
    if wide is None or order is None:
        raise ValueError("unknown ELF class or byte order")
    header = _read(handle, 0, 64 if wide else 52)
    if wide:
        shoff, = struct.unpack_from(order + "Q", header, 0x28)
        shentsize, shnum = struct.unpack_from(order + "HH", header, 0x3A)
        layout = order + "IIQQQQIIQQ"
    else:
        shoff, = struct.unpack_from(order + "I", header, 0x20)
        shentsize, shnum = struct.unpack_from(order + "HH", header, 0x2E)
        layout = order + "IIIIIIIIII"
    # The loader finds needs through the dynamic segment, so "no section table" is not "no needs".
    if shoff == 0:
        raise ValueError("no ELF section headers")
    size = struct.calcsize(layout)
    if shentsize != size:
        raise ValueError("unexpected ELF section header size")
    count = shnum or struct.unpack(layout, _read(handle, shoff, size))[5]
    return order, [struct.unpack(layout, _read(handle, shoff + index * size, size)) for index in range(count)]


def _needed_names(handle: BinaryIO, order: str, section: tuple[int, ...], strings: tuple[int, ...]) -> set[str]:
    table = _read(handle, strings[4], strings[5])
    body = _read(handle, section[4], section[5])
    names, cursor = set(), 0
    for _ in range(section[7]):
        _version, aux_count, _file, aux, following = struct.unpack_from(order + "HHIII", body, cursor)
        item = cursor + aux
        for _ in range(aux_count):
            _hash, _flags, _other, name, step = struct.unpack_from(order + "IHHII", body, item)
            names.add(table[name:table.index(b"\0", name)].decode("ascii"))
            item += step
        if not following:
            break
        cursor += following
    return names


def version_needs(path: Path) -> set[str] | None:
    """Every version name the file needs from its libraries, or None for a non-ELF file."""
    with path.open("rb") as handle:
        ident = handle.read(16)
        if len(ident) < 16 or ident[:4] != b"\x7fELF":
            return None
        order, sections = _sections(handle, ident)
        names: set[str] = set()
        for section in sections:
            if section[1] == _SHT_GNU_VERNEED:
                names |= _needed_names(handle, order, section, sections[section[6]])
        return names


def _ceilings(relative: str, arch: str) -> dict[str, Version]:
    for pattern, ceilings in EXCEPTIONS.get(arch, ()):
        if pattern.search(relative):
            return ceilings
    return FLOORS[arch]


def _text(version: Version) -> str:
    return ".".join(map(str, version))


def scan(app: Path, arch: str) -> tuple[int, list[Violation]]:
    """(ELF files scanned, needs above the ceiling that applies to each file)."""
    scanned, violations = 0, []
    for directory, _dirs, files in os.walk(app):
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(app).as_posix()
            try:
                needs = version_needs(path)
            except (ValueError, IndexError, struct.error) as error:
                scanned += 1
                violations.append(Violation(relative, str(error), None))
                continue
            if needs is None:
                continue
            scanned += 1
            ceilings = _ceilings(relative, arch)
            for need in sorted(needs):
                parsed = parse_need(need)
                if parsed is not None and parsed[1] > ceilings[parsed[0]]:
                    violations.append(Violation(relative, need, f"{parsed[0]}_{_text(ceilings[parsed[0]])}"))
    return scanned, violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", required=True, choices=sorted(FLOORS))
    parser.add_argument("app", type=Path, help="electron-builder's unpacked Linux app directory")
    args = parser.parse_args(argv)
    if not args.app.is_dir():
        parser.error(f"{args.app} is not a directory")
    scanned, violations = scan(args.app, args.arch)
    floor = ", ".join(f"{family}_{_text(version)}" for family, version in FLOORS[args.arch].items())
    print(f"{scanned} ELF files under {args.app}; linux-{args.arch} floor {floor}")
    for violation in violations:
        print(f"::error::{violation}")
    if violations:
        print(f"{len(violations)} problem(s) against the linux-{args.arch} glibc floor", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
