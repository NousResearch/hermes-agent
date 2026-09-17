"""One-time macOS setup for Hermes closed-display Power Protect."""

from __future__ import annotations

import json
import os
import platform
import stat
import subprocess
import tempfile
from pathlib import Path

_RULE = """Cmnd_Alias PMSET_HERMES_POWER = /usr/bin/pmset -a disablesleep 1, /usr/bin/pmset -a disablesleep 0
%admin ALL=(ALL) NOPASSWD: PMSET_HERMES_POWER
"""
_TARGET = "/etc/sudoers.d/hermes-power-protect"


def _apple_script_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _run_native_install(source: Path) -> subprocess.CompletedProcess[str]:
    source_literal = _apple_script_string(str(source))
    script = (
        f"set p to quoted form of {source_literal}\n"
        'do shell script "/usr/sbin/visudo -c -f " & p & " && '
        f'/usr/bin/install -o root -g wheel -m 0440 " & p & " {_TARGET}" '
        "with administrator privileges"
    )
    return subprocess.run(
        ["/usr/bin/osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )


def _verify_rule() -> bool:
    result = subprocess.run(
        ["/usr/bin/sudo", "-n", "-l"],
        check=False,
        capture_output=True,
        text=True,
    )
    output = f"{result.stdout}\n{result.stderr}"
    return (
        result.returncode == 0
        and "NOPASSWD" in output
        and "/usr/bin/pmset -a disablesleep 1" in output
        and "/usr/bin/pmset -a disablesleep 0" in output
    )


def _is_installed() -> bool:
    try:
        metadata = os.stat(_TARGET, follow_symlinks=False)
    except OSError:
        return False
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o440
    ):
        return False
    return _verify_rule()


def run_power_setup(_args=None) -> int:
    """Install the narrowly scoped rule using macOS's native admin prompt."""
    if platform.system() != "Darwin":
        print("Hermes closed-display Power Protect is only available on macOS.")
        return 2

    if _is_installed():
        print(f"Hermes Power Protect is already installed: {_TARGET}")
        return 0

    with tempfile.TemporaryDirectory(prefix="hermes-power-protect-") as temp_dir:
        source = Path(temp_dir) / "hermes-power-protect"
        source.write_text(_RULE, encoding="utf-8")
        result = _run_native_install(source)
        if result.returncode != 0:
            detail = (
                result.stderr or result.stdout or "administrator setup was cancelled"
            ).strip()
            print(f"Hermes Power Protect was not installed: {detail}")
            return 1

    if not _verify_rule():
        print("Hermes Power Protect installation could not be verified.")
        return 1

    print(f"Hermes Power Protect installed: {_TARGET}")
    return 0
