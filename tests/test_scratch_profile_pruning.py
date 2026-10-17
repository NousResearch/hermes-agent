"""Each profile gets its own boot-time scratch prune (#134896)."""

import os
import subprocess
import sys
import time
from pathlib import Path


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
    for idle, fresh in entries:
        assert not idle.exists()
        assert fresh.exists()


def test_recent_stamp_and_once_per_root_guards_remain_independent(tmp_path: Path) -> None:
    default, profile = tmp_path / "default", tmp_path / "profiles" / "one"
    default_idle, _ = _seed_scratch(default)
    profile_idle, _ = _seed_scratch(profile)
    (default_idle.parent / ".last_prune").touch()
    _boot(default, """
import os, sys, time
from pathlib import Path
import hermes_bootstrap
from hermes_constants import get_scratch_dir
home = Path(sys.argv[1])
scratch = get_scratch_dir(home)
late = scratch / 'late.txt'
late.write_text('old scratch', encoding='utf-8')
(scratch / '.last_prune').touch()
ancient = time.time() - 30 * 3600
for path in (late, scratch / '.last_prune'):
    os.utime(path, (ancient, ancient))
get_scratch_dir(home / 'cache' / '..')
assert late.exists(), 'the same root must not be pruned twice in a process'
""", profile)
    assert default_idle.exists(), "a recent stamp must still suppress pruning"
    assert not profile_idle.exists(), "another root's stamp must not suppress this profile"
    assert (profile_idle.parent / "late.txt").exists()
    _boot(profile, "import hermes_bootstrap")
    assert not (profile_idle.parent / "late.txt").exists()
