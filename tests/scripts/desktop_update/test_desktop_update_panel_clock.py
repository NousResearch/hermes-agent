"""The macOS status panel shows the stage AND the elapsed time, like ui.html.

#90194 gave the hand-off window a clock so a long update can be told apart from
a hung one; it lives in ui.html and is fed by serve-ui.py from the shim's
STARTED_AT. macOS never renders ui.html: posix.sh draws update-panel.js (JXA)
instead, and that port only showed the stage. A 2m20s `git merge` on a treeless
install then sat under an unchanging "Updating code and dependencies" and the
user reopened Hermes from the Dock thinking the update had hung.

These run the real panel script in its real runtime (osascript), with only its
`run` handler replaced, and the real posix.sh spawn.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SHIM_DIR = REPO_ROOT / "scripts" / "desktop-update"
PANEL = SHIM_DIR / "update-panel.js"

pytestmark = pytest.mark.platforms("macos")


def _running_lines(tmp_path: Path, cases: list[tuple[str, float | None, float]]) -> list[str]:
    """Evaluate the panel's running line for (stage, startedAt, now) triples.

    The panel source runs as-is; only its `run` handler (the window) is swapped
    for one that returns the lines, so osascript prints them and exits.
    """
    harness = tmp_path / "panel-harness.js"
    harness.write_text(
        PANEL.read_text(encoding="utf-8")
        + "\nrun = function (argv) {\n"
        "  return JSON.stringify(JSON.parse(argv[0]).map(\n"
        "    ([stage, startedAt, now]) => runningLine(stage, startedAt === null ? NaN : startedAt, now)))\n"
        "}\n",
        encoding="utf-8",
    )
    out = subprocess.run(
        ["/usr/bin/osascript", "-l", "JavaScript", str(harness), json.dumps(cases)],
        capture_output=True, text=True, timeout=60, check=True,
    )
    return json.loads(out.stdout)


def test_running_line_counts_like_ui_html(tmp_path):
    stage = "Applying code changes"

    assert _running_lines(tmp_path, [
        (stage, 1000, 1000.4),
        (stage, 1000, 1059.9),
        (stage, 1000, 1060),
        (stage, 1000, 1000 + 62 * 60 + 3),
    ]) == [
        f"{stage}\n0s elapsed",
        f"{stage}\n59s elapsed",
        f"{stage}\n1m 0s elapsed",
        f"{stage}\n62m 3s elapsed",
    ]


def test_no_valid_clock_shows_the_stage_alone(tmp_path):
    """An older spawner passes no start; an empty or future start is not a clock either."""
    stage = "Updating code and dependencies"

    assert _running_lines(tmp_path, [(stage, None, 1000), (stage, 0, 1000), (stage, 2000, 1000)]) == [
        stage, stage, stage]


def _panel_argv(tmp_path: Path) -> list[str]:
    ps = subprocess.run(["ps", "-axww", "-o", "command="], capture_output=True, text=True, check=False)
    for line in ps.stdout.splitlines():
        if "/usr/bin/osascript" in line and str(PANEL) in line and str(tmp_path) in line:
            return line.split(str(PANEL), 1)[1].split()
    return []


@pytest.mark.skipif(not os.path.exists("/usr/bin/osascript"), reason="the panel is JXA")
def test_handoff_gives_the_panel_its_own_clock(tmp_path):
    """posix.sh hands the panel the same STARTED_AT that serve-ui.py receives."""
    install = tmp_path / "hermes-agent"
    install.mkdir()
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
        "HERMES_SELFTEST_HOLD_SECONDS": "4",
    }
    env.pop("HERMES_SELFTEST_FAIL", None)
    before = int(time.time())

    proc = subprocess.Popen(
        ["bash", str(SHIM_DIR / "posix.sh"), "--install-root", str(install), "--self-test-ui"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    argv: list[str] = []
    try:
        while proc.poll() is None and not argv:
            argv = _panel_argv(tmp_path)
            time.sleep(0.2)
        proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    assert len(argv) == 2, argv  # <status-file> <started-at>
    status, started_at = argv
    assert Path(status).name.startswith("hermes-update-status.")
    assert started_at.isdigit() and before - 5 <= int(started_at) <= int(time.time())
