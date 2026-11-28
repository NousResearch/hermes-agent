"""Read and rewrite an ELF's DT_RUNPATH in place, without executing or relinking the file."""
from __future__ import annotations

import struct
from pathlib import Path

_PT_LOAD, _PT_DYNAMIC = 1, 2
_DT_NULL, _DT_STRTAB, _DT_RUNPATH = 0, 5, 29


class ElfError(ValueError):
    """The file is not an ELF this module can read, or the edit does not fit."""


def _runpath_span(data: bytes | bytearray) -> tuple[int, int]:
    """File offset and length of the DT_RUNPATH string."""
    try:
        return _parse_runpath_span(data)
    except ElfError:
        raise
    except (struct.error, ValueError) as exc:
        # A truncated or damaged file: offsets past the end, a string with no terminator.
        raise ElfError(f"damaged ELF: {exc}") from exc


def _parse_runpath_span(data: bytes | bytearray) -> tuple[int, int]:
    if data[:6] != b"\x7fELF\x02\x01":
        raise ElfError("not a little-endian ELF64")
    table = struct.unpack_from("<Q", data, 0x20)[0]
    size, count = struct.unpack_from("<HH", data, 0x36)
    # p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz
    segments = [struct.unpack_from("<IIQQQQ", data, table + i * size) for i in range(count)]
    dynamic = next(((offset, length) for kind, _, offset, _, _, length in segments if kind == _PT_DYNAMIC), None)
    if dynamic is None:
        raise ElfError("no dynamic segment")
    entries: dict[int, int] = {}
    for at in range(dynamic[0], dynamic[0] + dynamic[1], 16):
        tag, value = struct.unpack_from("<qQ", data, at)
        if tag == _DT_NULL:
            break
        entries.setdefault(tag, value)
    if _DT_RUNPATH not in entries or _DT_STRTAB not in entries:
        raise ElfError("no DT_RUNPATH")
    strings = entries[_DT_STRTAB]
    for kind, _, offset, address, _, length in segments:
        if kind == _PT_LOAD and address <= strings < address + length:
            start = offset + strings - address + entries[_DT_RUNPATH]
            return start, data.index(b"\0", start) - start
    raise ElfError("string table outside every loaded segment")


def runpath(binary: Path) -> str:
    data = binary.read_bytes()
    start, length = _runpath_span(data)
    try:
        return data[start:start + length].decode()
    except UnicodeDecodeError as exc:
        raise ElfError(f"RUNPATH is not text: {exc}") from exc


def set_runpath(binary: Path, value: str) -> None:
    """Overwrite the DT_RUNPATH string where it sits; the bytes left over become NULs.

    Nothing in the file moves, so the value must fit in the one it replaces. That is the write
    patchelf itself makes for a RUNPATH that fits; this one needs no tool on the staging host.
    """
    data = bytearray(binary.read_bytes())
    start, length = _runpath_span(data)
    encoded = value.encode()
    if b"\0" in encoded:
        raise ElfError("RUNPATH cannot hold a NUL")
    if len(encoded) > length:
        raise ElfError(f"RUNPATH of {len(encoded)} bytes does not fit the existing {length}")
    data[start:start + length] = encoded.ljust(length, b"\0")
    binary.write_bytes(data)
