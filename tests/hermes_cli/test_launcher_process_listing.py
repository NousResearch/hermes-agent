"""
Launcher arguments remain compatible with x11vnc authorization discovery.
"""

import os
from pathlib import Path
import subprocess

import pytest

from hermes_cli import _launchers
from hermes_platform.resolver.core import locate_command


# The FINDDISPLAY filter from x11vnc's src/ssltools.h, fed synthetic ps output only.
_FIND_DISPLAY = (
    r"grep '/X.* :[0-9][0-9]* .*-auth' | egrep -v 'startx|xinit' | "
    r"sed -e 's,^.*/X.* \(:[0-9][0-9]*\) .* -auth \([^ ][^ ]*\).*$,\1\,\2,' | sort -u"
)


def _has_discovery_tools() -> bool:
    grep = locate_command("grep").command
    if not locate_command("dash").command or not grep:
        return False
    version = subprocess.run(
        [*grep, "--version"], capture_output=True, check=False, timeout=10, env={**os.environ, "LC_ALL": "C"},
    )
    return version.returncode == 0 and b"GNU grep" in version.stdout


@pytest.mark.platforms("linux")
@pytest.mark.skipif(not _has_discovery_tools(), reason="x11vnc regression requires dash and GNU grep")
@pytest.mark.parametrize("entry_point", _launchers.ENTRY_POINTS)
def test_launcher_process_listing_preserves_x11vnc_auth_discovery(tmp_path: Path, entry_point: str) -> None:
    dash = locate_command("dash").command
    probe = subprocess.run([*dash, "-c", r'echo "\0"'], capture_output=True, check=True, timeout=10)
    assert probe.stdout == b"\0\n", "the selected shell must interpret echo escapes"

    script = _launchers._launcher_script(entry_point, tmp_path / "checkout", None)
    compile(script, "<generated launcher>", "exec")
    script = script.replace("\n", "?")  # procps renders embedded newlines as question marks.
    # Keep the NUL-producing launcher before Xorg, independent of grep's read buffer size.
    listing = (
        f"user 102 python3 -I -c {script}\n"
        "root 101 /usr/lib/xorg/Xorg -core :0 -seat seat0 "
        "-auth /run/lightdm/root/:0 -nolisten tcp vt7\n"
    ).encode()
    for emitter in ("printf '%s\\n' \"$psout\"", 'echo "$psout"'):
        result = subprocess.run(
            [*dash, "-c", "psout=$(cat)\n" + emitter + " | " + _FIND_DISPLAY],
            input=listing, capture_output=True, check=False, timeout=10, env={**os.environ, "LC_ALL": "C"},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == b":0,/run/lightdm/root/:0\n", (emitter, result.stdout, result.stderr)
        assert b"binary file matches" not in result.stderr, (emitter, result.stderr)
