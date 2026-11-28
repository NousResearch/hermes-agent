"""A child the suite spawns with a rebuilt environment still refuses lazy installs.

Tests that boot a real gateway or CLI scrub ``HERMES_*`` or rebuild the env from PATH/LANG/TZ.
Without the fence that child boots like a user's fresh home and spawns the detached default-plugins
installer: a real clone and toolchain download that outlives the test and writes into its temp home
while it is removed.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROBE = "from pm.install import lazy_installs_allowed; print(lazy_installs_allowed())"


def _child(tmp_path, **extra) -> str:
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(tmp_path / "home"), PYTHONPATH=str(REPO), **extra)
    done = subprocess.run([sys.executable, "-c", PROBE], cwd=REPO, env=env, capture_output=True,
                          text=True, timeout=120, check=True)
    return done.stdout.strip().splitlines()[-1]


def test_rebuilt_child_env_keeps_the_no_fetch_rule(tmp_path):
    assert _child(tmp_path) == "False"


@pytest.mark.parametrize("lifted", ["in-parent", "in-child"])
def test_a_test_that_lifts_the_rule_still_gets_a_fetching_child(tmp_path, monkeypatch, lifted):
    if lifted == "in-parent":
        monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS")
        assert _child(tmp_path) == "True"
    else:
        assert _child(tmp_path, HERMES_DISABLE_LAZY_INSTALLS="0") == "True"
