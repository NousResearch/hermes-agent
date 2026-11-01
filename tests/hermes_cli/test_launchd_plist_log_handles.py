# -*- coding: utf-8 -*-
"""The launchd job must not point its own log handles at the files the wrapper owns.

``launchd_program_arguments`` deliberately appends the gateway's stdout/stderr inside
the shell command "because ``system()`` otherwise inherits osascript's plist log
handles" — the wrapper is meant to be the sole owner of those two files.

Generating ``StandardOutPath`` / ``StandardErrorPath`` aimed at the same paths breaks
that contract: launchd opens each log once, the shell inside ``osascript`` opens it
again, and the gateway exits within seconds with ``EX_CONFIG 78`` without having
written a line. ``KeepAlive`` parks 78 on purpose, so the job latched into a respawn
loop, ``hermes gateway start`` reported success, and the Telegram intake stayed down
with nothing in ``gateway.log``, ``gateway.error.log`` or the unified log.
"""
import plistlib

import pytest

from hermes_cli.gateway_launchd import generate_launchd_plist

# plistlib and the osascript wrapper are the real contract; the scan reads
# sys.platform directly, so this runs on a real macOS host.
pytestmark = pytest.mark.platforms("macos")


def _generated_job() -> dict:
    return plistlib.loads(generate_launchd_plist().encode("utf-8"))


def test_launchd_does_not_reopen_the_logs_the_wrapper_already_owns():
    job = _generated_job()
    handles = {key: job[key] for key in ("StandardOutPath", "StandardErrorPath") if key in job}
    assert not handles, (
        f"launchd handles {handles} are opened a second time by the wrapper's own "
        "redirect; the gateway then dies with EX_CONFIG 78 writing no diagnostics"
    )


def test_the_wrapper_still_owns_both_logs():
    # Removing launchd's handles must not silence the gateway: the redirect inside
    # ProgramArguments is what actually carries the output.
    job = _generated_job()
    script = job["ProgramArguments"][-1]
    for path in ("gateway.log", "gateway.error.log"):
        assert path in script, f"{path} must stay in the wrapper redirect"
