"""The POSIX hand-off records creation times with a dot under any numeric locale (#135183).

mawk, the default awk on Debian and Ubuntu, prints printf's %f with LC_NUMERIC's decimal
separator. Under a comma-decimal locale proc_ct rendered the hand-off's own creation time as
``1791471938,040``; the marker grammar (MARKER_CT_RE, update_lock, update-marker-judge.ts) accepts
a dot only, so the hand-off read its own claim as a previous incarnation and every Desktop update
aborted with exit 3. Both cases run the real marker.sh against real processes, under a real
comma-decimal locale and an awk that honours it.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess

import pytest

from tests.scripts.desktop_update.test_desktop_update_posix_marker import POSIX, _ct

pytestmark = pytest.mark.platforms("linux")  # /proc creation times

MARKER_SH = POSIX.with_name("marker.sh")


@pytest.fixture(scope="module")
def comma_numeric_env(tmp_path_factory) -> dict[str, str]:
    """A comma-decimal LC_NUMERIC and an awk that honours it the way mawk does, or skip.

    it_IT is compiled into a private LOCPATH: hosts ship the definitions, rarely the compiled
    locale. gawk keeps a dot unless --use-lc-numeric; in that mode it prints %f with the locale's
    separator and still reads -v assignments with a dot, exactly as mawk does."""
    root = tmp_path_factory.mktemp("comma-locale")
    localedef = shutil.which("localedef")
    if localedef:
        subprocess.run([localedef, "-i", "it_IT", "-f", "UTF-8", str(root / "it_IT.UTF-8")],
                       capture_output=True, timeout=120, check=False)
    bin_dir = root / "bin"
    bin_dir.mkdir()
    awk = bin_dir / "awk"
    if mawk := shutil.which("mawk"):
        awk.symlink_to(mawk)
    elif gawk := shutil.which("gawk"):
        awk.write_text(f'#!/bin/sh\nexec {shlex.quote(gawk)} --use-lc-numeric "$@"\n', encoding="utf-8")
        awk.chmod(0o755)
    env = {**os.environ, "LOCPATH": str(root), "LC_ALL": "it_IT.UTF-8",
           "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
    probe = subprocess.run(["bash", "-c", "awk 'BEGIN{printf \"%.1f\", 1.5}'"], env=env,
                           capture_output=True, text=True, timeout=30, check=False)
    if probe.stdout != "1,5":
        pytest.skip(f"no comma-decimal locale with an awk that honours it here ({probe.stdout!r})")
    return env


@pytest.fixture
def sleeper():
    proc = subprocess.Popen(["sleep", "60"])
    yield proc.pid
    proc.kill()
    proc.wait()


def test_proc_ct_writes_a_dot_under_a_comma_decimal_locale(comma_numeric_env, sleeper):
    script = f"log() {{ :; }}; . {shlex.quote(str(MARKER_SH))}; proc_ct {sleeper}"
    result = subprocess.run(["bash", "-c", script], env=comma_numeric_env,
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.stdout.strip() == _ct(sleeper), result.stderr


def test_handoff_keeps_its_claim_and_publishes_the_update_child(tmp_path, comma_numeric_env, sleeper):
    """The reported exit 3: our own claim must stay ours, or the update child is never published."""
    marker = tmp_path / ".hermes-update-in-progress"
    script = (f"log() {{ printf '%s\\n' \"$*\" >&2; }}; MARKER={shlex.quote(str(marker))} "
              f"INSTALL_ROOT={shlex.quote(str(tmp_path))} DESKTOP_PID=0 HANDOFF_RUN='' "
              f"STARTED_AT=$(date +%s) MARKER_CLAIMED=0; . {shlex.quote(str(MARKER_SH))}; "
              f"marker_claim && marker_add_delegate {sleeper}")
    result = subprocess.run(["bash", "-c", script], env=comma_numeric_env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=30, check=False)

    assert result.returncode == 0, result.stderr
    lines = marker.read_text(encoding="utf-8").splitlines()
    assert re.fullmatch(r"ct:\d+\.\d{3}", lines[2]), lines
    assert lines[3] == f"delegate:{sleeper} ct:{_ct(sleeper)}"
