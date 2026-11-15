"""An installer rerun keeps a checkout on the update channel it chose (#137714).

Hermes-Setup reruns install.sh with its baked ``--branch main``. A checkout whose
config.yaml record (written by Hermes itself) names ``stable`` must stay where
``hermes update`` put it; a ``main`` record still follows the branch.
"""
from pathlib import Path
import os
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.platforms("posix")


def _git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        cwd=cwd, check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL,
    ).stdout.strip()


@pytest.mark.parametrize("channel,moves", [("stable", False), ("main", True)])
def test_rerun_keeps_a_chosen_channel_checkout(tmp_path, monkeypatch, channel, moves):
    from hermes_cli.update_channel import set_install_channel

    origin, home = tmp_path / "origin", tmp_path / "home"
    install = home / "hermes-agent"
    origin.mkdir()
    home.mkdir()
    _git(origin, "init", "-b", "main")
    for label in ("release", "newer"):
        _git(origin, "commit", "--allow-empty", "-m", label)
    release, tip = _git(origin, "rev-parse", "HEAD~1"), _git(origin, "rev-parse", "HEAD")
    _git(tmp_path, "clone", "-q", str(origin), str(install))
    _git(install, "checkout", "-q", "--detach", release)
    monkeypatch.setenv("HERMES_HOME", str(home))
    set_install_channel(channel, install)

    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "install.sh"), "--stage", "repository",
         "--branch", "main", "--dir", str(install), "--hermes-home", str(home), "--non-interactive"],
        check=False, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120,
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "NO_COLOR": "1"},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(install, "rev-parse", "HEAD") == (tip if moves else release), result.stdout
