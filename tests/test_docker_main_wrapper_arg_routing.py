"""main-wrapper.sh must route hermes global flags to `hermes <args>`.

Regression for #43295: `docker run <image> -p viewer gateway run` lost the
`-p viewer` and started a default `hermes gateway run`. The cause is the
executable-detection probe — `command -v "$1"` parses a leading-dash first arg
(`-p`) as an *option to `command`* and reports success, so the args were routed
to the bare-executable branch (`exec "$@"`) instead of `hermes "$@"`. The fix
is `command -v -- "$1"`, which stops option parsing.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_WRAPPER = REPO_ROOT / "docker" / "main-wrapper.sh"


def test_wrapper_uses_end_of_options_for_command_v():
    text = MAIN_WRAPPER.read_text(encoding="utf-8")
    assert 'command -v -- "$1"' in text, (
        "main-wrapper.sh must use `command -v -- \"$1\"` so a leading-dash CMD "
        "arg (e.g. `-p <profile>`) routes to `hermes <args>`"
    )
    # Guard against regression to the unsafe bare form.
    assert 'command -v "$1"' not in text


def test_command_v_dashdash_routes_leading_dash_to_hermes():
    """Behavioral check in a real POSIX shell: a hermes flag is NOT detected as
    an executable (so it falls through to the `hermes <args>` branch), while a
    genuine bare executable still is."""
    # `-p` (a hermes global flag) must NOT match -> routes to `hermes -p ...`.
    assert subprocess.run(["sh", "-c", 'command -v -- "-p" >/dev/null 2>&1']).returncode != 0
    # A real executable still matches -> routes around hermes (sleep/bash/sh).
    assert subprocess.run(["sh", "-c", 'command -v -- "sh" >/dev/null 2>&1']).returncode == 0
    # And the OLD bare form is what mis-detected `-p` as "found" (documents the bug).
    assert subprocess.run(["sh", "-c", 'command -v "-p" >/dev/null 2>&1']).returncode == 0
