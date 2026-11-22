"""install.sh selects the uv artifact for the native userland's libc (#123682)."""

import json
import struct
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("linux")

ROOT = Path(__file__).resolve().parents[3]
INSTALL_SH = ROOT / "scripts" / "install.sh"


def _elf64(interp: str, *, interp_offset: int = 0x2E0, decoy: bytes = b"") -> bytes:
    """Minimal ELF64 little-endian image with one PT_INTERP segment."""
    ehsize, phentsize, phnum = 64, 56, 2
    data = interp.encode() + b"\x00"
    ehdr = bytearray(64)
    ehdr[0:4] = b"\x7fELF"
    ehdr[4] = 2  # ELFCLASS64
    ehdr[5] = 1  # ELFDATA2LSB
    struct.pack_into("<Q", ehdr, 0x20, ehsize)  # e_phoff
    struct.pack_into("<HH", ehdr, 0x36, phentsize, phnum)
    phdr = bytearray(phentsize * phnum)
    struct.pack_into("<I", phdr, 0, 1)  # PT_LOAD
    struct.pack_into("<I", phdr, phentsize, 3)  # PT_INTERP
    struct.pack_into("<Q", phdr, phentsize + 8, interp_offset)  # p_offset
    struct.pack_into("<Q", phdr, phentsize + 32, len(data))  # p_filesz
    body = bytearray(bytes(ehdr) + bytes(phdr))
    body.extend(b"\x00" * (interp_offset - len(body)))
    body[0x100:0x100 + len(decoy)] = decoy
    return bytes(body + data)


def _target(tmp_path: Path, *probes: bytes) -> list[str]:
    paths = []
    for index, blob in enumerate(probes):
        path = tmp_path / f"probe{index}"
        path.write_bytes(blob)
        paths.append(str(path))
    script = r"""
source "$1" --manifest >/dev/null
target="$(uv_bootstrap_target "${@:2}")"
uv_bootstrap_pin "$target"
printf '%s\n%s\n%s\n' "$target" "$UV_PIN_URL" "$UV_PIN_SHA256"
"""
    result = subprocess.run(
        ["bash", "-c", script, "_", str(INSTALL_SH), *paths],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.splitlines()


def _artifact(target: str) -> list[str]:
    row = json.loads((ROOT / "pm" / "lock.json").read_text())["packages"]["uv"]["artifacts"][target]
    return [target, row["url"], row["sha256"]]


def test_musl_elf_interpreter_selects_musl_uv(tmp_path):
    # The ELF interpreter of /bin/sh decides, as in pm/store.py; a glibc ldd
    # (musl installed as a secondary toolchain, or a gcompat shim) must not.
    assert _target(tmp_path, _elf64("/lib/ld-musl-x86_64.so.1")) == _artifact("linux-x64-musl")


def test_glibc_elf_wins_even_when_interp_lies_past_8k_and_file_mentions_musl(tmp_path):
    # Fedora 44 places PT_INTERP at 0x10a000; a fixed head window misses it and
    # the musl path embedded in the same binary must not decide the target.
    blob = _elf64(
        "/lib64/ld-linux-x86-64.so.2",
        interp_offset=0x200000,
        decoy=b"/lib/ld-musl-x86_64.so.1",
    )
    assert _target(tmp_path, blob) == _artifact("linux-x64")


def test_first_classifiable_probe_wins(tmp_path):
    glibc = _elf64("/lib64/ld-linux-x86-64.so.2")
    musl = _elf64("/lib/ld-musl-x86_64.so.1")
    assert _target(tmp_path, musl, glibc) == _artifact("linux-x64-musl")
    assert _target(tmp_path, b"not an elf", glibc) == _artifact("linux-x64")
