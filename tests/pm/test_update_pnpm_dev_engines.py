"""A pnpm re-pin carries root package.json's devEngines pin with it.

scripts/build/node-deps.mjs refuses any pnpm other than the devEngines one, so
`pm update pnpm` or `pm lock --bump pnpm` rewriting only pm/lock.json left every
install failing until someone edited package.json by hand.
"""

import json
import shutil
from argparse import Namespace
from pathlib import Path

import pytest

from pm import cli
from pm.lock import Lockfile
from pm.store import ALL_TARGETS
from pm.update import Resolved

REPO = Path(__file__).resolve().parents[2]
VERSION = "99.0.0"


def _update(lock):
    targets = [t for t in ALL_TARGETS if t != "linux-arm64-bionic"]
    decision = Resolved("pnpm", lock.version("pnpm"), "semver", version=VERSION,
                        per_target=dict.fromkeys(targets, VERSION))
    return cli._apply_pins([decision], lock)


def _bump(lock):
    return cli.cmd_lock(Namespace(name="pnpm", version=VERSION))


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    shutil.copytree(REPO / "pm", tmp_path / "pm", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(REPO / "package.json", tmp_path / "package.json")
    lock = Lockfile(tmp_path / "pm" / "lock.json")
    monkeypatch.setattr(cli, "_lockfile", lambda: lock)
    monkeypatch.setattr(cli, "hash_url", lambda url: "0" * 64)
    monkeypatch.setattr(cli, "_install_names", lambda names: 0)
    monkeypatch.setattr(cli, "_sync_venv_step", lambda: True)
    return tmp_path, lock


@pytest.mark.parametrize("repin", [_update, _bump], ids=["update", "bump"])
def test_pnpm_repin_moves_the_dev_engines_pin(repin, checkout):
    root, lock = checkout
    before = json.loads((root / "package.json").read_text(encoding="utf-8"))

    assert repin(lock) == 0

    after = json.loads((root / "package.json").read_text(encoding="utf-8"))
    assert after["devEngines"]["packageManager"]["version"] == lock.version("pnpm") == VERSION
    before["devEngines"]["packageManager"]["version"] = VERSION
    assert after == before, "only the pin moves"
