"""Profile scanning must terminate without dropping long commands (#129281)."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("profile", ["default", "sibling"])
def test_long_profile_scan_preserves_targeting(profile):
    # A process timeout can interrupt a C regex holding the GIL; a thread cannot.
    script = r'''
from cron import lifecycle_guard as guard

guard._current_profile_name = lambda: "default"
profile = PROFILE
expected = profile == "default"
for selector in (f"-p {profile}", f"--profile {profile}", f"--profile={profile}"):
    for before, after in ((58, 0), (0, 58), (58, 58)):
        command = "hermes " + "--quiet " * before + selector + " " + "--quiet " * after + "gateway stop"
        assert guard.contains_gateway_lifecycle_command(command) is expected
        assert guard.contains_gateway_lifecycle_command_or_referenced_script(command) is expected
    for before, after in (("--usage-file - ", ""), ("", "--usage-file - ")):
        command = f"hermes {before}{selector} {after}gateway stop"
        assert guard.contains_gateway_lifecycle_command(command) is expected
        assert guard.contains_gateway_lifecycle_command_or_referenced_script(command) is expected
'''.replace("PROFILE", repr(profile))
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_nonmatching_flag_runs_finish():
    script = r'''
from cron import lifecycle_guard as guard

for command in (
    "rclone lsf $HOME/.hermes --recursive --files-only " + "--exclude " * 58,
    "hermes " + "--quiet " * 1000,
    "hermes -p default " + "--quiet " * 1000 + "gateway status",
    "hermes " + ("--option" + " " * 100 + "value " ) * 58,
):
    assert guard.contains_gateway_lifecycle_command(command) is False
'''
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
