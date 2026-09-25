"""The bootstrap never runs a uv it found on PATH; it stages the pin (#101269).

A uv on PATH at least as new as the pin must not become the bootstrap's uv:
that hands byte authority over the whole install to a user-controlled binary,
leaves the store slot empty, and leaves pm / doctor / MCP resolving a uv that
was never staged. The pin is the contract — the staged copy is sha256-verified
against ``pm/lock.json``, and a rerun hits it and fetches nothing.
"""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
INSTALL_SH = ROOT / "scripts" / "install.sh"
pytestmark = pytest.mark.platforms("posix")


def test_ensure_uv_stages_the_pin_even_when_a_newer_uv_is_on_path(tmp_path: Path) -> None:
    bash = shutil.which("bash")
    assert bash, "the shell bootstrap requires bash"

    home = tmp_path / "home" / ".hermes"
    marker = tmp_path / "path-uv-was-executed"
    user_bin = tmp_path / "user-bin"
    user_bin.mkdir()
    # A uv that claims to be NEWER than the pin and records being executed.
    (user_bin / "uv").write_text(
        f"#!{bash}\n"
        f"touch {shlex.quote(str(marker))}\n"
        'case "$1" in --version) echo "uv 99.0.0 (fake 2099-01-01)"; exit 0 ;; esac\n'
        "exit 0\n",
        encoding="utf-8",
    )
    (user_bin / "uv").chmod(0o755)

    env = {**os.environ, "HOME": str(tmp_path / "home"), "HERMES_HOME": str(home)}
    env["PATH"] = f"{user_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    env.pop("HERMES_RUNTIME_DIR", None)

    # Pre-seed the store slot: ensure_uv must find the pin and never touch the
    # network, so the only variable under test is which uv it chooses.
    script = (
        'source "$1" --manifest || exit 1\n'
        'target="$(uv_bootstrap_target)" || exit 1\n'
        'entry="${HERMES_RUNTIME_DIR:-$HERMES_HOME/tools}/uv-$UV_PIN_VERSION-$target"\n'
        'mkdir -p "$entry" || exit 1\n'
        'printf \'#!/usr/bin/env sh\\necho "uv 0.12.3 (fixture)"\\n\' > "$entry/uv" || exit 1\n'
        'chmod +x "$entry/uv" || exit 1\n'
        'ensure_uv || exit 1\n'
        'printf "UV_CMD=%s\\nENTRY=%s\\n" "$UV_CMD" "$entry"\n'
    )
    result = subprocess.run(
        [bash, "-c", script, "test", str(INSTALL_SH)],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    values = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    uv_cmd, entry = values["UV_CMD"], values["ENTRY"]

    assert uv_cmd == os.path.join(entry, "uv"), (
        f"ensure_uv picked {uv_cmd}, expected the staged pin {os.path.join(entry, 'uv')}"
    )
    assert entry.startswith(str(home / "tools" / "uv-")), entry
    assert not marker.exists(), (
        "ensure_uv executed the uv on PATH — the pin is the byte authority, "
        "not a user-installed binary"
    )
