"""The standalone shell installer selects the lockfile's native uv pin."""

import json
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("linux")

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("libc,suffix", [("musl libc", "-musl"), ("GNU libc", "")])
def test_bootstrap_selects_native_uv_pin(libc, suffix):
    machine = subprocess.check_output(["uname", "-m"], text=True).strip()
    arch = {"x86_64": "x64", "aarch64": "arm64"}[machine]
    target = f"linux-{arch}{suffix}"
    script = """
source "$1" --manifest
libc_output="$2"
ldd() { printf '%s\\n' "$libc_output"; }
target="$(uv_bootstrap_target)"
uv_bootstrap_pin "$target"
printf '%s\\n%s\\n%s\\n' "$target" "$UV_PIN_URL" "$UV_PIN_SHA256"
"""
    result = subprocess.run(
        ["bash", "-c", script, "_", str(ROOT / "scripts" / "install.sh"), libc],
        check=True, capture_output=True, text=True,
    )
    row = json.loads((ROOT / "pm" / "lock.json").read_text())["packages"]["uv"]["artifacts"][target]
    assert result.stdout.splitlines() == [target, row["url"], row["sha256"]]
