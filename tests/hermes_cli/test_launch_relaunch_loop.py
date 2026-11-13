"""A launch whose dependencies stay stale boots; it never re-execs itself forever.

Each test runs real launches as separate processes through the real decision chain the
bootstrap runs (``prepare_launch`` -> ``relaunch_command`` -> ``os.execv``). Only PM's sync and
currency answers and the store interpreter are stubbed, to model an install whose repair keeps
failing (or whose sync never makes it current). Every exec'd generation is counted.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GENERATION_CAP = 8

_LAUNCH = r'''
import os, sys
from pathlib import Path

repo, root, scratch, sync_mode, launch = sys.argv[1:6]
import json
command_args = json.loads(os.environ["LAUNCH_COMMAND"])  # an argv naming `update` alarms the test guard
sys.path.insert(0, repo)
import pm, pm.client
from hermes_cli import _launchers, venv_sync

generations = Path(scratch, f"generations-{launch}")
with generations.open("a") as fh:
    fh.write(f"{os.getpid()}\n")
if len(generations.read_text().split()) > int(os.environ["GENERATION_CAP"]):
    raise SystemExit(99)

state = Path(scratch, "state")

def sync(*_args, **_kwargs):
    with Path(scratch, "syncs").open("a") as fh:
        fh.write(f"{launch}\n")
    if sync_mode == "fail":
        raise RuntimeError("sync failed")
    state.write_text("current")

# "restore": PM's answer follows the sync; otherwise it never says current.
pm.venv_is_current = lambda **_kw: sync_mode == "restore" and state.read_text() == "current"
pm.sync_venv = sync
pm.client.ensure_tools_for_sync = lambda: None
_launchers.resolve_store_python = lambda *_a, **_k: Path(sys.executable)
venv_sync.publish_launchers = lambda *_a, **_k: None

if sync_mode == "restore" and getattr(sys, "_hermes_relaunched", False) and not Path(scratch, "restored").exists():
    # The relaunched child restores a torn update: the restored tree's dependencies are stale again.
    Path(scratch, "restored").write_text("1")
    state.write_text("stale")
    from hermes_cli import _early_recovery
    _early_recovery.relaunch_after_restore()

try:
    python = venv_sync.prepare_launch(Path(root), command_args)
    if python is not None:
        os.execv(str(python), venv_sync.relaunch_command(python, Path(repo), sys.argv, sys.orig_argv, None))
except RuntimeError as exc:
    print(f"degraded: {exc}", file=sys.stderr)
print("BOOTED")
'''


def _launch(tmp_path: Path, root: Path, sync_mode: str, launch: int, argv: list[str]):
    if not (tmp_path / "state").exists():
        (tmp_path / "state").write_text("stale", encoding="utf-8")
    script = tmp_path / "launch.py"
    script.write_text(_LAUNCH, encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "HERMES_HOME": str(tmp_path / "home"),
           "LAUNCH_COMMAND": json.dumps(argv), "GENERATION_CAP": str(GENERATION_CAP)}
    result = subprocess.run(
        [sys.executable, str(script), str(REPO), str(root), str(tmp_path), sync_mode, str(launch)],
        env=env, capture_output=True, text=True, timeout=120, check=False,
    )
    generations = len((tmp_path / f"generations-{launch}").read_text().split())
    return result, generations


def _syncs(tmp_path: Path) -> int:
    path = tmp_path / "syncs"
    return len(path.read_text().split()) if path.exists() else 0


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("argv", [[], ["update"], ["update", "--check"]], ids=["launch", "update", "update-check"])
def test_backed_off_failed_sync_boots_on_the_previous_dependencies(tmp_path, monkeypatch, argv):
    """#139952 records a failed launch sync so relaunches back off. A launch inside that backoff
    (or past the cap) has run no repair, so relaunching it would re-enter the same decision and
    exec again forever; it must boot on the previous dependencies, and `hermes update` (the
    remedy every backed-off launch names) still tries the repair; its report-only `--check` does not."""
    from hermes_cli import venv_sync

    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='example'\n", encoding="utf-8")
    (root / "install-stamp.json").write_text(json.dumps({"updateMechanism": "self"}), encoding="utf-8")

    for launch in range(1, 5):
        result, generations = _launch(tmp_path, root, "fail", launch, argv)
        assert result.returncode == 0, f"launch {launch} looped: {generations} generations\n{result.stderr}"
        assert result.stdout.strip() == "BOOTED"
        assert generations == 1
    expected = 4 if argv == ["update"] else venv_sync.COMPLETION_RETRY_BACKOFF_ATTEMPTS
    assert _syncs(tmp_path) == expected, "a backed-off launch re-ran the failing repair"

    if not argv:
        # Past the cap, at any age: still one generation, and one pointer at `hermes update`.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        record = venv_sync._completion_attempts_path(root)
        assert record.is_file(), "the failed syncs were not recorded where completion_retry_state reads"
        record.write_text(f"{venv_sync.COMPLETION_RETRY_MAX_ATTEMPTS}\n", encoding="utf-8")
        result, generations = _launch(tmp_path, root, "fail", 5, argv)
        assert (result.returncode, generations) == (0, 1), result.stderr
        assert result.stderr.count("could not be finished automatically") == 1


def _committed_checkout(tmp_path: Path) -> Path:
    from hermes_cli.source_stamp import write_source_stamp

    root = tmp_path / "checkout"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='example'\n", encoding="utf-8")
    (root / ".gitignore").write_text("install-stamp.json\n", encoding="utf-8")
    (root / "install-stamp.json").write_text(json.dumps({"updateMechanism": "self"}), encoding="utf-8")
    git_env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    for args in (["init"], ["config", "user.email", "test@example.com"], ["config", "user.name", "test"],
                 ["add", "-A"], ["commit", "-m", "init"]):
        subprocess.run(["git", *args], cwd=root, env=git_env, check=True, capture_output=True, timeout=30)
    assert write_source_stamp(root)["dirty"] is False
    return root


@pytest.mark.platforms("posix")
def test_relaunched_child_never_relaunches_into_itself(tmp_path):
    """A sync that never makes the install current (the PM answer disagrees with what the sync
    did) must not turn every launch into an exec chain: the relaunched child boots."""
    root = _committed_checkout(tmp_path)
    result, generations = _launch(tmp_path, root, "ok", 1, [])
    assert result.returncode == 0, f"looped: {generations} generations\n{result.stderr}"
    assert result.stdout.strip() == "BOOTED"
    assert generations == 2, "the first relaunch into the store interpreter is still taken"
    assert "relaunching did not leave this install current" in result.stderr


@pytest.mark.platforms("posix")
def test_marker_kept_by_a_restore_does_not_refuse_a_relaunch_that_worked(tmp_path):
    """A restore re-execs ``sys.orig_argv``, so the restored process still carries the relaunch
    marker. When its sync makes the install current, it relaunches once more and that child
    boots current; it never claims the relaunch left the install stale."""
    root = _committed_checkout(tmp_path)
    result, generations = _launch(tmp_path, root, "restore", 1, [])
    assert result.returncode == 0, f"looped: {generations} generations\n{result.stderr}"
    assert result.stdout.strip() == "BOOTED"
    assert "relaunching did not leave this install current" not in result.stderr
    # launch -> relaunched child (restores) -> restored process (syncs) -> its relaunch boots
    assert generations == 4
