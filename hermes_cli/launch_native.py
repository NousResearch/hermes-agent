"""Opt-in native TUI launch for `_launch_tui`.

Ink (`ui-tui/`) stays the default. Set HERMES_TUI_NATIVE=1 or pass --native
to run `hermes-tui-native` with the same env `_launch_tui` already builds.

If the binary is missing, return None so the caller falls back to Ink.
Dashboard PTY embeds (`HERMES_TUI_DASHBOARD=1`) always stay on Ink.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Mapping


def wants_native(
    argv: list[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> bool:
    if env is None:
        env = os.environ
    if env.get("HERMES_TUI_NATIVE") == "1":
        return True
    if argv is None:
        argv = sys.argv[1:]
    return "--native" in argv


def native_tui_bin(env: Mapping[str, str] | None = None) -> str | None:
    if env is None:
        env = os.environ
    explicit = (env.get("HERMES_TUI_NATIVE_BIN") or "").strip()
    if explicit and os.path.isfile(explicit) and os.access(explicit, os.X_OK):
        return explicit
    path = shutil.which("hermes-tui-native", path=env.get("PATH"))
    if path:
        return path
    return None


def native_tui_argv(env: Mapping[str, str]) -> list[str] | None:
    """Return argv for the native client, or None to fall back to Ink."""
    if env.get("HERMES_TUI_DASHBOARD") == "1":
        return None
    if not wants_native(env=env):
        return None
    bin_path = native_tui_bin(env)
    if not bin_path:
        print(
            "HERMES_TUI_NATIVE=1 but hermes-tui-native was not found.\n"
            "Build it: cargo install --path crates/tui\n"
            "Or set HERMES_TUI_NATIVE_BIN=/path/to/hermes-tui-native\n"
            "Falling back to the Ink TUI.",
            file=sys.stderr,
        )
        return None
    argv = [bin_path]
    resume = env.get("HERMES_TUI_RESUME") or os.environ.get("HERMES_TUI_RESUME")
    if resume:
        argv.extend(["--resume", resume])
    title = env.get("HERMES_TUI_TITLE") or os.environ.get("HERMES_TUI_TITLE")
    if title:
        argv.extend(["--title", title])
    return argv
