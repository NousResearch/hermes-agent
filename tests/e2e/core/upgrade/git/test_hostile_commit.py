"""The source swap of ``hermes update`` is one crash-safe commit point (lane LP-COMMIT).

Each cell drives a real install (HEAD's own ``scripts/install.sh`` in a bwrap sandbox) through a
real ``hermes update`` against a local origin, breaks it at one named point with a real SIGKILL or a
real git failure, and then judges the next real launch from the files the updater itself owns:

* ``kill_after_tree_moved``: git finished the fast-forward of a release with NO dependency change
  and the updater is SIGKILLed before the completion child starts. The launcher tail (launchers,
  builds, config migration, ``install-stamp.json``) must still be owed and finished by the next
  launch — a tail armed only by the child is lost for good.
* ``torn_tree_on_git_failure``: git exits non-zero halfway through writing the release (a
  read-only directory: ``unable to unlink old ...``). The checkout must end whole at its pre-update
  commit, never torn with the interrupted-pull marker already dropped.
* ``kill_during_branch_switch``: the checkout is parked on a fully merged branch, so the update
  first switches it to ``main`` (CP0); the updater dies while git is rewriting files. The next
  launch must put the parked tree back.
* ``kill_mid_zip_swap``: the ZIP swap is SIGKILLed between renames. The next launch must finish or
  roll back the swap, leaving no ``*.hermes-update-staging``/``-old`` sibling to wedge a retry.
* ``kill_mid_zip_swap_with_hermes_cli_moved_aside``: the swap renamed ``hermes_cli/`` (the recovery
  code itself) aside and died before its replacement landed. The next launch must still restore.
* ``syntax_error_target``: a release whose startup module does not compile is refused BEFORE
  HEAD moves (no fast-forward in the reflog), not merely rolled back afterwards.
* ``failed_upstream_sync_after_origin_pull``: a fork checkout whose origin pull already moved the
  tree, then the upstream fast-forward fails (read-only directory) and the updater is killed. The
  tail owed for the moved tree must survive the failed second move.
* ``kill_mid_zip_swap_over_leftover_backup``: the only copy of an entry is a leftover
  ``*.hermes-update-old`` the swap heals before swapping; a kill mid-swap must never delete both.
* ``kill_during_branch_switch_without_local_branch``: the parked checkout has no local ``main``
  (only ``origin/main``); git's checkout guess would create it and rewrite the tree unmarked.
* ``kill_during_syntax_rollback``: a broken release past the preflight is rolled back and the
  updater dies mid-rollback; the next launch must land on the pre-update commit, not the broken one.
* ``git_killed_inside_rollback_reset``: the rollback's own ``git reset -q <pre>`` is SIGKILLed the
  moment its ``index.lock`` exists (HEAD still on the broken release, the lock left behind). The
  updater must keep the interrupted-pull marker (never report the rollback complete), and the next
  launch must reclaim the dead lock, redo the rollback and land on the pre-update commit whole.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.pm import _pm as P
from tests.e2e.core.upgrade.test_upgrade_path import _RETRY_PREFIX
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
]

REAL_GIT = shutil.which("git") or "git"
ARTIFACT_SUFFIXES = (".hermes-update-staging", ".hermes-update-old")


@pytest.fixture(scope="module")
def provider():
    with FakeLLMServer(default_text="fake reply for the hostile commit suite") as srv:
        yield srv


@pytest.fixture()
def world(tmp_path_factory, provider):
    root = tmp_path_factory.mktemp("hostile-commit")
    sb, origin = P.install_head(root)
    P.configure(sb, provider.base_url)
    shim = sb.root / "wrap" / "git"
    world = {"sb": sb, "origin": origin, "root": root, "shim": shim, "shim_text": shim.read_text(encoding="utf-8-sig"), "n": 0}
    yield world
    shim.write_text(world["shim_text"], encoding="utf-8")


def _hostile_git(world, body: str) -> None:
    """Prefix the sandbox's git shim with ``body`` (bash; ``$REAL`` is the real git, ``$PPID`` the
    updater process that spawned git)."""
    text = world["shim_text"]
    head, _, rest = text.partition("\n")
    world["shim"].write_text(f'{head}\nREAL="{REAL_GIT}"\n{body}\n{rest}', encoding="utf-8")


def _release(world, files: dict[str, str]) -> str:
    world["n"] += 1
    return I.publish_commit(world["origin"], world["root"], f"release: hostile commit {world['n']}", files)


def _head(sb) -> str:
    return I.git("rev-parse", "HEAD", cwd=sb.checkout)


def _tracked_dirty(sb) -> str:
    return I.git("status", "--porcelain", "--untracked-files=no", cwd=sb.checkout)


def _stamp_commit(sb) -> str:
    try:
        return json.loads((sb.checkout / "install-stamp.json").read_text(encoding="utf-8-sig")).get("commit") or ""
    except (OSError, ValueError):
        return ""


def _launch(sb, marker: str) -> subprocess.CompletedProcess:
    """A later real launch by the user (fresh pid, lazy installs on: the real launch path)."""
    return P.run_env(sb, [*_RETRY_PREFIX, sb.hermes, "-z", marker], P.lazy_env(sb), timeout=P.UPDATE_TIMEOUT)


def _update(sb) -> subprocess.CompletedProcess:
    return P.run_env(sb, [*_RETRY_PREFIX, sb.hermes, "update", "--yes", "--branch", "main", "--no-gateway-restart"],
                     sb.env, timeout=P.UPDATE_TIMEOUT)


def _artifacts(sb) -> list[str]:
    return sorted(p.name for p in sb.checkout.iterdir() if p.name.endswith(ARTIFACT_SUFFIXES))


def test_kill_after_tree_moved_still_owes_the_tail(world):
    sb = world["sb"]
    pre = _head(sb)
    assert _stamp_commit(sb) == pre and not P.pending_marker(sb).exists(), "harness: install not settled"
    target = _release(world, {"e2e_hostile_release.py": "RELEASE = 1\n"})
    # git completes the fast-forward, then the updater dies before it can spawn the completion child.
    _hostile_git(world, 'case " $* " in *" merge --ff-only "*) "$REAL" "$@"; rc=$?; kill -KILL $PPID; exit $rc;; esac')
    killed = _update(sb)
    _hostile_git(world, "")
    assert _head(sb) == target, "harness: the kill did not land after the tree moved\n" + I.describe(killed)
    owed = P.pending_marker(sb).exists()

    launch = _launch(sb, "first launch after a kill past the swap")
    assert owed, ("the tree moved to the release but no source-completion tail was owed: the kill "
                  "before the completion child lost it\n" + P.diagnostics(sb, killed, launch))
    assert not P.pending_marker(sb).exists() and _stamp_commit(sb) == target, (
        "the next launch did not finish the owed tail for the moved tree\n" + P.diagnostics(sb, killed, launch))


def test_torn_tree_on_git_failure_is_restored(world):
    sb = world["sb"]
    # Release 1 adds a file under a directory; the user makes that directory read-only.
    ok = _release(world, {"aaa_e2e_first.py": "V = 1\n", "zzz_e2e_locked/mod.py": "V = 1\n"})
    P.ok(_update(sb), "harness: plain update to the first release failed")
    assert _head(sb) == ok
    _release(world, {"aaa_e2e_first.py": "V = 2\n", "zzz_e2e_locked/mod.py": "V = 2\n"})
    locked = sb.checkout / "zzz_e2e_locked"
    locked.chmod(0o555)
    try:
        failed = _update(sb)
    finally:
        locked.chmod(0o755)
    assert failed.returncode != 0, "harness: git did not fail on the read-only directory\n" + I.describe(failed)
    launch = _launch(sb, "first launch after a torn pull")
    assert _head(sb) == ok and not _tracked_dirty(sb), (
        "git failed mid-pull and the checkout was left torn (marker dropped, nothing restores it):\n"
        + _tracked_dirty(sb) + "\n" + P.diagnostics(sb, failed, launch))


def test_kill_during_branch_switch_is_restored(world):
    sb = world["sb"]
    base = _head(sb)
    _release(world, {"e2e_cp0_a.py": "A = 1\n", "e2e_cp0_b.py": "B = 1\n"})
    P.ok(_update(sb), "harness: plain update failed")
    # Park the checkout on a fully merged branch one release behind main.
    I.git("checkout", "-q", "-b", "e2e-parked", base, cwd=sb.checkout)
    parked = _head(sb)
    _release(world, {"e2e_cp0_c.py": "C = 1\n"})
    # CP0: `git checkout main` dies after rewriting one file of main's tree.
    _hostile_git(world, 'if [ "${@: -1}" = main ] && case " $* " in *" checkout "*) true;; *) false;; esac; then '
                        '"$REAL" show main:e2e_cp0_a.py > e2e_cp0_a.py; kill -KILL $PPID; exit 137; fi')
    killed = _update(sb)
    _hostile_git(world, "")
    assert (sb.checkout / "e2e_cp0_a.py").is_file() and _head(sb) == parked, (
        "harness: the kill did not land mid-switch\n" + I.describe(killed))
    launch = _launch(sb, "first launch after a kill during the branch switch")
    assert _head(sb) == parked and not (sb.checkout / "e2e_cp0_a.py").exists(), (
        "a kill during the CP0 branch switch left main's files in the parked checkout\n"
        + P.diagnostics(sb, killed, launch))


def test_syntax_error_target_is_refused_before_head_moves(world):
    sb = world["sb"]
    pre = _head(sb)
    constants = I.git("show", "main:hermes_constants.py", cwd=world["origin"])
    _release(world, {"hermes_constants.py": constants + "\ndef broken(:\n"})
    reflog_before = I.git("reflog", "-n", "20", "--format=%H %gs", cwd=sb.checkout)
    refused = _update(sb)
    reflog_after = I.git("reflog", "-n", "20", "--format=%H %gs", cwd=sb.checkout)
    assert refused.returncode != 0 and _head(sb) == pre, I.describe(refused)
    assert reflog_after == reflog_before, (
        "HEAD moved onto the uncompilable release before the syntax guard refused it:\n"
        + reflog_after.replace(reflog_before, "").strip() + "\n" + I.describe(refused))


_ZIP_DRIVER = r"""
import os, signal, sys
sys.path.insert(0, sys.argv[3])  # the install's checkout (the venv's own path points at its workspace)
from hermes_cli import update_cmd_zip as z
import hermes_cli.main as m
assert str(m.PROJECT_ROOT) == sys.argv[3], m.PROJECT_ROOT
swaps, kill_at = [0], sys.argv[2]
def killing(real):
    def move(src, dst):
        real(src, dst)
        if kill_at.startswith("after:"):  # right after the named entry's replacement landed
            if os.path.basename(str(src)) == kill_at[6:] + ".hermes-update-staging":
                os.kill(os.getpid(), signal.SIGKILL)
        elif kill_at == "hermes_cli-moved-aside":  # the live package renamed away, its replacement not yet in
            if os.path.basename(str(src)) == "hermes_cli" and str(dst).endswith(".hermes-update-old"):
                os.kill(os.getpid(), signal.SIGKILL)
        elif str(src).endswith(".hermes-update-staging"):  # one entry swapped in
            swaps[0] += 1
            if swaps[0] >= int(kill_at):
                os.kill(os.getpid(), signal.SIGKILL)
    return move
os.rename, os.replace = killing(os.rename), killing(os.replace)
z._download_and_swap_zip("main", sys.argv[1])
"""


def _zip_killed_at(world, kill_at: str, files: dict[str, str] | None = None):
    """A ZIP update of a fresh release, SIGKILLed at ``kill_at`` (N-th entry swapped in, or a named point)."""
    sb = world["sb"]
    target = _release(world, {"e2e_zip_release.py": f"Z = {world['n']}\n", "hermes_cli/e2e_zip_marker.py": "M = 1\n",
                              **(files or {})})
    archive = world["root"] / f"release-{world['n']}.zip"
    I.git("archive", "--format=zip", "--prefix=hermes-agent-main/", "-o", str(archive), target, cwd=world["origin"])
    driver = world["root"] / "zip_driver.py"
    driver.write_text(_ZIP_DRIVER, encoding="utf-8")
    return P.run_env(sb, [sb.python, str(driver), archive.as_uri(), kill_at, str(sb.checkout)], sb.env,
                     timeout=P.UPDATE_TIMEOUT)


def test_kill_mid_zip_swap_is_settled_by_the_next_launch(world):
    sb = world["sb"]
    pre = _head(sb)
    # Killed right after a TRACKED file's new bytes landed: the swap order follows the archive's
    # directory listing, so a "kill at the N-th swap" can land on N unchanged entries only.
    agents = I.git("show", "HEAD:AGENTS.md", cwd=world["origin"]) + "\n<!-- e2e zip swap -->\n"
    killed = _zip_killed_at(world, "after:AGENTS.md", {"AGENTS.md": agents})
    at_kill = {"artifacts": _artifacts(sb), "dirty": _tracked_dirty(sb),
               "new_entry": (sb.checkout / "e2e_zip_release.py").exists()}
    assert at_kill["artifacts"] and at_kill["dirty"], (
        f"harness: the ZIP swap was not killed mid-rename ({at_kill['dirty']!r})\n" + I.describe(killed))
    launch = _launch(sb, "first launch after a kill mid ZIP swap")
    leftovers = _artifacts(sb)
    assert not leftovers, (f"{len(leftovers)} staging/backup siblings leaked and wedge the next update: "
                           f"{leftovers[:5]} (at kill: {len(at_kill['artifacts'])} siblings, "
                           f"tracked changes {at_kill['dirty']!r})\n" + I.describe(launch))
    assert _head(sb) == pre and not _tracked_dirty(sb), (
        "the interrupted ZIP swap left a mixed tree:\n" + _tracked_dirty(sb) + "\n" + I.describe(launch))


def test_kill_mid_zip_swap_with_hermes_cli_moved_aside_reaches_recovery(world):
    """The one window where the recovery code itself is gone: the swap renamed ``hermes_cli/`` aside
    and died before the replacement landed. The next launch must still reach the journal-driven
    restore (from the moved-aside copy) and run, not die on ``No module named 'hermes_cli'``."""
    sb = world["sb"]
    pre = _head(sb)
    killed = _zip_killed_at(world, "hermes_cli-moved-aside")
    assert not (sb.checkout / "hermes_cli").exists() and (sb.checkout / "hermes_cli.hermes-update-old").is_dir(), (
        "harness: the ZIP swap was not killed with hermes_cli/ moved aside\n" + I.describe(killed))
    launch = _launch(sb, "first launch after a kill with hermes_cli moved aside")
    assert launch.returncode == 0 and (sb.checkout / "hermes_cli" / "main.py").is_file(), (
        "the launch after a kill with hermes_cli/ moved aside could not reach recovery\n" + I.describe(launch))
    assert not _artifacts(sb) and _head(sb) == pre and not _tracked_dirty(sb), (
        f"left {_artifacts(sb)[:5]} / tracked changes {_tracked_dirty(sb)!r}\n" + I.describe(launch))
    P.ok(_update(sb), "the update after the restored swap failed")


def test_failed_upstream_sync_after_origin_pull_keeps_the_tail_owed(world):
    """Fork checkout: the origin pull moves the tree (committed), then the upstream fast-forward fails
    on a read-only directory and is put back to the post-origin commit. That failure must not hand
    back the obligations the origin pull armed: a kill right after it still owes the tail."""
    sb, root = world["sb"], world["root"]
    first = _release(world, {"zzz_e2e_fork/mod.py": "V = 1\n"})
    P.ok(_update(sb), "harness: plain update to the first release failed")
    assert _head(sb) == first and _stamp_commit(sb) == first
    moved = _release(world, {"e2e_fork_origin.py": "B = 1\n"})
    upstream = root / "upstream.git"
    I.git("clone", "-q", "--bare", "--shared", str(world["origin"]), str(upstream), cwd=root)
    up = I.publish_commit(upstream, root, "upstream: touches the locked dir", {"zzz_e2e_fork/mod.py": "V = 2\n"})
    I.git("remote", "add", "upstream", str(upstream), cwd=sb.checkout)
    flag = sb.checkout / ".git" / "e2e-upstream-ff-failed"
    # origin reads as a fork (the sandbox shim otherwise reports the official URL). Kill the updater at
    # its next branch check once the upstream merge has failed (and been settled).
    _hostile_git(world, 'case " $* " in *" remote get-url origin "*) echo https://github.com/e2e-fork/hermes-agent.git; '
                        'exit 0;; esac\n'
                        # The sync fast-forwards to the upstream commit it resolved (one SHA, m2).
                        f'case " $* " in *" merge --ff-only "*"{up} "*) "$REAL" "$@"; rc=$?; '
                        f'[ $rc != 0 ] && touch "{flag}"; exit $rc;; esac\n'
                        f'if [ -e "{flag}" ]; then case " $* " in *" rev-parse --abbrev-ref HEAD "*) '
                        'kill -KILL $PPID; exit 137;; esac; fi')
    locked = sb.checkout / "zzz_e2e_fork"
    locked.chmod(0o555)
    try:
        killed = _update(sb)
    finally:
        locked.chmod(0o755)
        _hostile_git(world, "")
    assert flag.exists() and killed.returncode != 0 and _head(sb) == moved and not _tracked_dirty(sb), (
        "harness: the upstream ff did not fail after the origin pull moved the tree\n" + I.describe(killed))
    owed = P.pending_marker(sb).exists()
    launch = _launch(sb, "first launch after a failed upstream sync")
    assert owed, ("the origin pull moved the tree but the failed upstream ff disarmed its owed tail\n"
                  + P.diagnostics(sb, killed, launch))
    assert _stamp_commit(sb) == moved and not P.pending_marker(sb).exists(), (
        "the next launch did not finish the tail for the moved tree\n" + P.diagnostics(sb, killed, launch))


def test_kill_mid_zip_swap_over_leftover_backup_keeps_the_entry(world):
    """A pre-journal crash left ``e2e_zip_lost.hermes-update-old`` as the ONLY copy of a tracked dir.
    The next swap heals it, swaps it, and is killed right after its replacement landed: recovery
    must put the old dir back, never drop both copies."""
    sb = world["sb"]
    pre = _release(world, {"e2e_zip_lost/m.py": "OLD = 1\n"})
    P.ok(_update(sb), "harness: plain update failed")
    lost = sb.checkout / "e2e_zip_lost"
    lost.rename(sb.checkout / "e2e_zip_lost.hermes-update-old")
    killed = _zip_killed_at(world, "after:e2e_zip_lost", {"e2e_zip_lost/m.py": "NEW = 1\n"})
    assert (lost / "m.py").read_text(encoding="utf-8-sig") == "NEW = 1\n" and (sb.checkout / "e2e_zip_lost.hermes-update-old").is_dir(), (
        "harness: the swap was not killed right after e2e_zip_lost landed\n" + I.describe(killed))
    launch = _launch(sb, "first launch after a kill over a leftover backup")
    assert (lost / "m.py").is_file() and (lost / "m.py").read_text(encoding="utf-8-sig") == "OLD = 1\n", (
        "recovery deleted both copies of an entry whose backup was healed before the swap: "
        f"{sorted(p.name for p in sb.checkout.iterdir() if p.name.startswith('e2e_zip'))}\n" + I.describe(launch))
    assert not _artifacts(sb) and _head(sb) == pre and not _tracked_dirty(sb), (
        f"left {_artifacts(sb)[:5]} / tracked changes {_tracked_dirty(sb)!r}\n" + I.describe(launch))


def test_kill_during_branch_switch_without_local_branch_is_restored(world):
    """The parked checkout has no local ``main``: only ``origin/main``. Whichever checkout actually
    moves the tree to main must run under the marker."""
    sb = world["sb"]
    base = _head(sb)
    _release(world, {"e2e_dwim_a.py": "A = 1\n"})
    P.ok(_update(sb), "harness: plain update failed")
    I.git("checkout", "-q", "-b", "e2e-parked", base, cwd=sb.checkout)
    I.git("branch", "-q", "-D", "main", cwd=sb.checkout)
    parked = _head(sb)
    _release(world, {"e2e_dwim_c.py": "C = 1\n"})
    # Die after writing one of main's files, in the checkout that moves the tree: a local main
    # exists, `-B` was given, or git may guess (no --no-guess) and create main from origin/main.
    # `-B main` names the resolved commit, not the ref (review O2), so it moves whatever comes last.
    _hostile_git(world, 'case " $* " in *" checkout "*) moves=0; case " $* " in *" -B main "*) moves=1;; esac; '
                        'case "${@: -1}" in main|origin/main) '
                        '"$REAL" rev-parse -q --verify refs/heads/main >/dev/null && moves=1; '
                        'case " $* " in *" --no-guess "*) ;; *) moves=1;; esac;; esac; '
                        'if [ $moves = 1 ]; then "$REAL" show origin/main:e2e_dwim_a.py > e2e_dwim_a.py; '
                        'kill -KILL $PPID; exit 137; fi;; esac')
    killed = _update(sb)
    _hostile_git(world, "")
    assert (sb.checkout / "e2e_dwim_a.py").is_file() and _head(sb) == parked, (
        "harness: the kill did not land mid-switch\n" + I.describe(killed))
    launch = _launch(sb, "first launch after a kill during a guessed branch switch")
    assert _head(sb) == parked and not (sb.checkout / "e2e_dwim_a.py").exists(), (
        "a kill while git created main from origin/main left main's files in the parked checkout "
        "(no interrupted-pull marker)\n" + P.diagnostics(sb, killed, launch))


def test_kill_during_syntax_rollback_lands_on_the_pre_update_commit(world):
    """A broken release the preflight cannot see (its target read is answered with the old file)
    is rolled back by the post-pull guard; the updater dies while ``reset --hard`` is writing the old
    files back. The next launch must finish on the pre-update commit, not restore the broken one."""
    sb = world["sb"]
    pre = _head(sb)
    constants = I.git("show", "main:hermes_constants.py", cwd=world["origin"])
    broken = _release(world, {"hermes_constants.py": constants + "\ndef broken(:\n",
                              "e2e_rollback_extra.py": "X = 1\n"})
    _hostile_git(world, 'case " $* " in *" show "*":hermes_constants.py "*) "$REAL" show HEAD:hermes_constants.py; '
                        'exit $?;; '
                        '*" cat-file --batch "*) sed "s#^[^ ]*:hermes_constants.py\\$#HEAD:hermes_constants.py#" | "$REAL" "$@"; exit $?;; '
                        '*" reset --hard "*) "$REAL" show "${@: -1}:hermes_constants.py" > hermes_constants.py; '
                        'kill -KILL $PPID; exit 137;; esac')
    killed = _update(sb)
    _hostile_git(world, "")
    assert "syntax error" in (killed.stdout or "") + (killed.stderr or ""), (
        "harness: the post-pull guard did not fire\n" + I.describe(killed))
    assert (sb.checkout / "e2e_rollback_extra.py").exists(), "harness: the kill did not land mid-rollback\n" + I.describe(killed)
    launch = _launch(sb, "first launch after a kill during the syntax rollback")
    assert _head(sb) == pre and not _tracked_dirty(sb) and not (sb.checkout / "e2e_rollback_extra.py").exists(), (
        f"the launch after a kill mid-rollback did not land on {pre[:10]} (HEAD {_head(sb)[:10]}, broken "
        f"{broken[:10]}; tracked changes {_tracked_dirty(sb)!r})\n" + P.diagnostics(sb, killed, launch))


# Run by the git shim inside the sandbox: start the real git under an inotify watch on its git dir and
# SIGKILL it the moment ``index.lock`` is created (git then holds the lock and has not moved HEAD).
_GIT_KILLER = r"""
import ctypes, os, signal, struct, subprocess, sys
real, args = sys.argv[1], sys.argv[2:]
git_dir = subprocess.run([real, "rev-parse", "--absolute-git-dir"], capture_output=True, text=True).stdout.strip()
libc = ctypes.CDLL(None, use_errno=True)
fd = libc.inotify_init1(0)
libc.inotify_add_watch(fd, git_dir.encode(), 0x100)  # IN_CREATE
git = subprocess.Popen([real, *args])
while git.poll() is None:
    buf, i = os.read(fd, 4096), 0
    while i < len(buf):
        length = struct.unpack_from("iIII", buf, i)[3]
        name, i = buf[i + 16:i + 16 + length].rstrip(b"\0"), i + 16 + length
        if name == b"index.lock":
            os.kill(git.pid, signal.SIGKILL)
            git.wait()
            sys.exit(137)
sys.exit(git.returncode)
"""


def test_git_killed_inside_the_rollback_reset_is_finished_by_the_next_launch(world):
    sb = world["sb"]
    pre = _head(sb)
    # A critical module the launcher imports only after the launch-time repair ran (the launcher's own
    # first import, hermes_constants, is a documented limit: a broken copy dies before any repair).
    tools = I.git("show", "main:model_tools.py", cwd=world["origin"])
    broken = _release(world, {"model_tools.py": tools + "\ndef broken(:\n",
                              "e2e_rollback_reset_extra.py": "X = 1\n"})
    killer = world["root"] / "git_killer.py"
    killer.write_text(_GIT_KILLER, encoding="utf-8")
    once = sb.root / "rollback-reset-killed"  # the sandbox can write only under its own root
    # The preflight reads the old file (so the broken release passes it); the rollback's first step,
    # `reset -q <pre>` (HEAD and index, no file), runs once under the killer.
    _hostile_git(world, 'case " $* " in *" show "*":model_tools.py "*) "$REAL" show HEAD:model_tools.py; '
                        'exit $?;; '
                        '*" cat-file --batch "*) sed "s#^[^ ]*:model_tools.py\\$#HEAD:model_tools.py#" | "$REAL" "$@"; exit $?;; '
                        f'*" reset -q {pre} "*) if [ ! -e "{once}" ]; then : > "{once}"; '
                        f'exec "{sb.python}" "{killer}" "$REAL" "$@"; fi;; esac')
    killed = _update(sb)
    _hostile_git(world, "")
    output = (killed.stdout or "") + (killed.stderr or "")
    lock = sb.checkout / ".git" / "index.lock"
    assert "syntax error" in output and once.exists(), "harness: the rollback reset never ran\n" + I.describe(killed)
    assert lock.exists() and _head(sb) == broken, (
        "harness: the kill did not land while git held index.lock\n" + I.describe(killed))
    assert killed.returncode != 0 and "Rollback complete" not in output, (
        "the updater reported a rollback that never happened\n" + I.describe(killed))
    assert (sb.checkout / ".git" / "hermes-update-pull").is_file(), (
        "the killed rollback erased its own recovery record while HEAD is still the broken release\n"
        + I.describe(killed))

    launch = _launch(sb, "first launch after git was killed inside the rollback")
    assert _head(sb) == pre and not _tracked_dirty(sb) and not lock.exists(), (
        f"the launch did not finish the rollback to {pre[:10]} (HEAD {_head(sb)[:10]}, broken {broken[:10]}, "
        f"lock {lock.exists()}, tracked changes {_tracked_dirty(sb)!r})\n" + P.diagnostics(sb, killed, launch))
    assert not (sb.checkout / "e2e_rollback_reset_extra.py").exists()
    assert not (sb.checkout / ".git" / "hermes-update-pull").exists()
