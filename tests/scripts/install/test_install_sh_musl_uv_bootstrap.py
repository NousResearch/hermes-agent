"""install.sh bootstraps musl uv when the native userland is musl (#123682)."""

import json
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("linux")

ROOT = Path(__file__).resolve().parents[3]


def test_musl_elf_interpreter_selects_musl_uv_even_when_ldd_says_glibc():
    # The ELF interpreter of /bin/sh decides, as in pm/store.py; a glibc ldd
    # (musl installed as a secondary toolchain, or a gcompat shim) must not.
    script = r"""
source "$1" --manifest >/dev/null
uname() { case "$1" in -m) echo x86_64 ;; -s) echo Linux ;; esac; }
head() { printf '\177ELF\0\0\0/lib/ld-musl-x86_64.so.1\0'; }
ldd() { echo "ldd (GNU libc) 2.39"; }
target="$(uv_bootstrap_target)"
uv_bootstrap_pin "$target"
printf '%s\n%s\n%s\n' "$target" "$UV_PIN_URL" "$UV_PIN_SHA256"
"""
    result = subprocess.run(
        ["bash", "-c", script, "_", str(ROOT / "scripts" / "install.sh")],
        check=True, capture_output=True, text=True,
    )
    row = json.loads((ROOT / "pm" / "lock.json").read_text())["packages"]["uv"]["artifacts"]["linux-x64-musl"]
    assert result.stdout.splitlines() == ["linux-x64-musl", row["url"], row["sha256"]]
