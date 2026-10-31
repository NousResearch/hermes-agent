"""Each profile gets its own boot-time scratch prune (#134896)."""

import os
import subprocess
import sys
import time
from pathlib import Path

from tests.waiting import wait_until


def _seed_scratch(home: Path) -> tuple[Path, Path]:
    scratch = home / "cache" / "scratch"
    scratch.mkdir(parents=True)
    idle, fresh = scratch / "idle.txt", scratch / "fresh.txt"
    idle.write_text("old scratch", encoding="utf-8")
    fresh.write_text("active scratch", encoding="utf-8")
    ancient = time.time() - 30 * 3600
    os.utime(idle, (ancient, ancient))
    return idle, fresh


def _boot(home: Path, code: str, *args: Path) -> None:
    env = {key: value for key, value in os.environ.items()
           if key not in ("TMPDIR", "TMP", "TEMP", "HERMES_SCRATCH_DIR")}
    env["HERMES_HOME"] = str(home)
    subprocess.run(
        [sys.executable, "-c", code, *(str(arg) for arg in args)],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, encoding="utf-8", check=True, timeout=30,
    )


def test_bootstrap_then_profile_rehome_prunes_each_home(tmp_path: Path) -> None:
    homes = [tmp_path / "default", tmp_path / "profiles" / "one", tmp_path / "profiles" / "two"]
    entries = [_seed_scratch(home) for home in homes]
    _boot(homes[0], """
import os, sys
from pathlib import Path
import hermes_bootstrap
from hermes_constants import export_scratch_tmp_env
for home in sys.argv[1:]:
    os.environ['HERMES_HOME'] = home
    assert export_scratch_tmp_env()
    assert Path(os.environ['TMPDIR']) == Path(home) / 'cache' / 'scratch'
""", *homes[1:])
    # the due pass runs in a detached child, so wait for it instead of the boot process
    for idle, _ in entries:
        wait_until(lambda idle=idle: not idle.exists(), timeout=60, what="the detached prune removing the idle entry")
    for _, fresh in entries:
        assert fresh.exists()


def test_recent_stamp_and_once_per_root_guards_remain_independent(tmp_path: Path, monkeypatch) -> None:
    import hermes_constants
    import hermes_constants_scratch_detach as detach

    spawned: list[Path] = []
    monkeypatch.setattr(detach, "spawn_prune", spawned.append)
    monkeypatch.setattr(hermes_constants, "_scratch_pruned_roots", set())
    default, profile = tmp_path / "default", tmp_path / "profiles" / "one"
    default_scratch = hermes_constants.get_scratch_dir(default, prune=False)
    profile_scratch = hermes_constants.get_scratch_dir(profile, prune=False)
    (default_scratch / ".last_prune").touch()

    hermes_constants._prune_scratch_dir_once(default_scratch)
    assert spawned == [], "a recent stamp must still suppress pruning"
    hermes_constants._prune_scratch_dir_once(profile_scratch)
    assert spawned == [profile_scratch], "another root's stamp must not suppress this profile"
    hermes_constants._prune_scratch_dir_once(profile_scratch / ".." / "scratch")
    assert spawned == [profile_scratch], "the same root must not be pruned twice in a process"
