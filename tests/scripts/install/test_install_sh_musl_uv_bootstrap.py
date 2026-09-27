"""The standalone shell installer selects the lockfile's native uv pin."""

import json
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("linux")

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "libc,loader_present,suffix",
    [
        ("musl libc", False, "-musl"),
        ("GNU libc", False, ""),
        ("absent", True, "-musl"),
        ("BusyBox", True, "-musl"),
        ("GNU libc", True, ""),
        ("absent", False, ""),
    ],
)
def test_bootstrap_selects_native_uv_pin(libc, loader_present, suffix):
    machine = subprocess.check_output(["uname", "-m"], text=True).strip()
    arch = {"x86_64": "x64", "aarch64": "arm64"}[machine]
    target = f"linux-{arch}{suffix}"
    script = """
source "$1" --manifest
libc_output="$2"
loader_present="$3"
ldd() {
    [ "$libc_output" != absent ] || return 127
    printf '%s\\n' "$libc_output"
}
compgen() {
    if [ "$1" = -G ] && [ "$2" = '/lib/ld-musl-*.so.1' ]; then
        [ "$loader_present" = yes ]
    else
        builtin compgen "$@"
    fi
}
target="$(uv_bootstrap_target)"
uv_bootstrap_pin "$target"
printf '%s\\n%s\\n%s\\n' "$target" "$UV_PIN_URL" "$UV_PIN_SHA256"
"""
    result = subprocess.run(
        ["bash", "-c", script, "_", str(ROOT / "scripts" / "install.sh"), libc,
         "yes" if loader_present else "no"],
        check=True, capture_output=True, text=True,
    )
    row = json.loads((ROOT / "pm" / "lock.json").read_text())["packages"]["uv"]["artifacts"][target]
    assert result.stdout.splitlines() == [target, row["url"], row["sha256"]]
