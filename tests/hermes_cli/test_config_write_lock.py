"""config.yaml read-modify-write cycles hold a cross-process lock (#96571).

Two real processes write distinct keys at once; without the lock each replace started from a
snapshot that lacked the other's keys and about half the writes vanished.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_CHILD = r"""
import contextlib, io, sys
tag, n, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
from hermes_cli import config as c
for i in range(n):
    if mode == "set":
        with contextlib.redirect_stdout(io.StringIO()):
            c.set_config_value(f"race.{tag}{i}", "v", force=True)
    else:
        import utils
        utils.atomic_roundtrip_yaml_update(c.get_config_path(), f"race.{tag}{i}", "v")
"""


@pytest.mark.parametrize("mode", ["set", "roundtrip_update"])
def test_concurrent_writers_keep_every_key(tmp_path, mode):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("model:\n  default: x/y\n", encoding="utf-8")
    env = dict(os.environ, HERMES_HOME=str(home),
               PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH", "")])))
    n = 40
    procs = [subprocess.Popen([sys.executable, "-c", _CHILD, tag, str(n), mode], env=env, cwd=str(tmp_path))
             for tag in ("a", "b")]
    assert [p.wait(timeout=240) for p in procs] == [0, 0]

    from ruamel.yaml import YAML
    data = YAML(typ="safe").load((home / "config.yaml").read_text(encoding="utf-8-sig"))
    assert sorted(data["race"]) == sorted(f"{t}{i}" for t in "ab" for i in range(n))
    assert data["model"] == {"default": "x/y"}
