"""The single commit point of a source ``hermes update``: everything the tree move needs armed first.

Before git (or the ZIP swap) writes the first file of the new code, three things are durable:

* the interrupted-pull marker (``.git/hermes-update-pull``) naming the commit git moves to, for EVERY
  tree move (branch switch, upstream fork pull, the pull itself) so a launch after any exit that left
  a torn tree puts the old one back (``_early_recovery.restore_interrupted_pull``);
* the source-completion tail (``venv_sync.arm_completion``) and the host fleet-restart obligation,
  so a kill after the move but before the completion child starts still leaves the tail owed to the
  next launch (``venv_sync.prepare_launch``) and the restart owed to the next ``hermes update``.

Failures before the move are no-ops: ``disarm_commit_obligations`` puts both obligations back the way
this run found them once the tree is verified at its pre-update commit again.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Optional

from hermes_cli._early_recovery import (
    RECOVERY_CLOSURE,
    RECOVERY_CLOSURE_INIT,
    RECOVERY_CLOSURE_MANIFEST,
    blob_id,
    _lock_identity,
    interrupted_pull_marker,
    is_object_id,
    recovery_closure_dir,
    recovery_closure_verified,
    restore_interrupted_pull,
    write_durable_text,
)
from hermes_cli.update_cmd_common import _record_stop
from hermes_cli.update_custody import run_git

logger = logging.getLogger("hermes_cli.update_cmd")
# What this run found before it armed anything: {path: bytes or None}. None = nothing armed yet.
_armed_snapshot: Optional[dict[Path, Optional[bytes]]] = None
# What this run's latest arm left at each path: disarm hands back only records still exactly these.
_armed_bytes: dict[Path, Optional[bytes]] = {}
# (git_cmd, (HEAD, branch)) when this run reached its checkout phase, and the checkout it names.
_run_start: Optional[tuple[list, tuple[str, str]]] = None
_obligation_root: Optional[Path] = None
# This run's stake in the shared host record (``owners``): disarm removes only this one.
_owner: str = ""


def _owns_live_checkout(root: Path) -> bool:
    from hermes_cli.update_cmd import _m

    return _m()._pytest_owns_live_checkout(root)


def _obligation_paths(root: Path) -> list[Path]:
    from hermes_cli.update_cmd_fleet import _fleet_restart_pending_marker_path
    from hermes_cli.update_host_obligation import host_obligation_path
    from hermes_cli.venv_sync import completion_pending_path

    return [completion_pending_path(root), _fleet_restart_pending_marker_path(), host_obligation_path()]


def _read_or_none(path: Path) -> Optional[bytes]:
    """The record's bytes, None when absent. Any other read error raises: custody is never guessed."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def arm_commit_obligations(root: Path, expected_sha: str) -> None:
    """Owe the completion tail and the fleet restart for ``expected_sha`` BEFORE the tree moves.

    Idempotent within a run (the first call snapshots what to restore on a no-op failure). An
    unwritable install state, an unreadable record or a fleet restart neither store accepted
    raises: nothing has moved yet, and moving without the obligation is exactly the
    tail-never-runs state this exists to prevent.
    """
    global _armed_snapshot, _owner
    import uuid

    from hermes_cli.update_cmd_fleet import _write_fleet_restart_pending_marker
    from hermes_cli.venv_sync import arm_completion

    root = Path(root)
    if _owns_live_checkout(root):
        return
    if _armed_snapshot is None:
        _armed_snapshot = {path: _read_or_none(path) for path in _obligation_paths(root)}
        _owner = f"{root}|{os.getpid()}|{uuid.uuid4().hex}"
    try:
        arm_completion(root)
        owed = _write_fleet_restart_pending_marker(expected_sha=expected_sha or "", owner=_owner)
    finally:  # even a half-done arm: disarm must still recognise what this run wrote
        _armed_bytes.update({path: _read_or_none(path) for path in _armed_snapshot})
    if not owed:
        raise OSError("could not record the pending gateway-restart obligation")


def disarm_commit_obligations() -> None:
    """Restore both obligations to what this run found: the tree never left its pre-update commit.

    Off the commit this run STARTED from nothing is handed back: a failed later move (CP1 refused
    or failed after the CP0 branch switch, the upstream fork ff after the origin pull) settles on a
    commit that is already new code, so the obligations are owed for THAT head instead of the
    failed move's target, which the checkout does not contain and no restart could discharge.
    """
    global _armed_snapshot
    if _run_start is not None:
        git_cmd, (start_head, _branch) = _run_start
        root = Path(_obligation_root) if _obligation_root is not None else None
        head = head_and_branch(git_cmd, root)[0] if root is not None else ""
        if root is None or not start_head or head != start_head:
            if root is not None:
                _owe_for(root, head)
            return
    from hermes_cli.update_host_obligation import host_obligation_path, release_host_obligation, replace_bytes

    snapshot, _armed_snapshot = _armed_snapshot, None
    armed = dict(_armed_bytes)
    _armed_bytes.clear()
    host = host_obligation_path()
    for path, data in (snapshot or {}).items():
        try:
            if path == host:
                # Shared by every install of the OS user: same-SHA arms join one record, so bytes
                # cannot tell whose it is; the ``owners`` stake can (review C2).
                release_host_obligation(_owner)
                continue
            if path not in armed or _read_or_none(path) != armed[path]:
                continue  # rewritten since this run armed it (another install's update): theirs now
            if data is None:
                path.unlink(missing_ok=True)
            else:
                # An unpredictable exclusive temp, never a fixed name: nothing at any guessable
                # sibling is written through or deleted first (reviews N05, O1).
                replace_bytes(path, data)
        except OSError:
            pass  # an owed tail/restart left armed is a retry, never a lost obligation


def owe_restore_of(root: Path, pre: str | None) -> None:
    """A failed move left a torn tree under its marker: the next launch's restore puts ``pre`` back,
    so the obligations name ``pre``, never the target the checkout will not hold."""
    _owe_for(Path(root), pre or "")


def _owe_for(root: Path, sha: str) -> None:
    """Re-arm what this run armed for ``sha``, the commit the checkout is (or is restored) on."""
    if _armed_snapshot is None or not is_object_id(sha):
        return  # nothing armed by this run, or nothing names the code: keep what stands
    try:
        arm_commit_obligations(root, sha)
    except OSError as exc:
        logger.warning("Could not re-arm the update obligations for %s: %s", sha[:10], exc)
        print(f"  ⚠ Could not record the restart this update still owes for {sha[:10]} ({exc}); "
              "run `hermes gateway restart` once the update is done.", file=sys.stderr)


def debt_sha_for_move(pre: str, target: str) -> str:
    """The commit a fast-forward ``pre`` -> ``target`` arms the obligations for.

    A later move of this run (the upstream sync after the origin pull) owes ``pre``: the commit the
    earlier move already owes, which every landing of a fast-forward contains, and the fleet
    discharge holds the gateways to the checkout's actual HEAD whenever it contains the debt. Its
    failure then needs no retarget write, which could itself fail and leave the debt naming a commit
    the checkout never reached (review O4). The run's first move owes its ``target``: its failure
    hands the obligations back whole (``disarm_commit_obligations``).
    """
    start = _run_start[1][0] if _run_start is not None else ""
    return pre if start and pre != start else target


def arm_commit_point(git_cmd, root: Path, expected_sha: str, **move) -> str | None:
    """Arm the obligations, then the tree-move marker (``arm_tree_move(**move)``): no marker, no move.

    ``None`` when both are durable; otherwise why not, with the obligations handed back
    (``disarm_commit_obligations``), and the caller must stop before git writes a file.
    Linux/macOS gateways are paused first: this is every route's first checkout move.
    """
    from hermes_cli.update_cmd_posix_pause import pause_at_commit_point
    if refused := pause_at_commit_point():
        return refused
    try:
        arm_commit_obligations(root, expected_sha)
        arm_tree_move(git_cmd, root, **move)
    except OSError as exc:
        disarm_commit_obligations()
        head = head_and_branch(git_cmd, root)[0] if _run_start is not None else ""
        if head and head != _run_start[1][0]:
            # A later move (CP1 after the CP0 switch): THIS step moved nothing, an earlier one did.
            return (f"could not arm the update ({exc}); this step changed nothing, but an earlier step "
                    f"of this update already moved the checkout to {head[:10]}")
        return f"could not arm the update ({exc}); the checkout was not changed"
    return None


def arm_tree_move(git_cmd, root: Path, *, pre: str | None, target: str, stash: str | None,
                  rollback: str | None = None) -> Path:
    """Write the interrupted-pull marker for one git tree move (pre -> target).

    ``rollback`` (``branch``/``detach``): a syntax rollback that moves HEAD and the index back to
    ``pre`` before any file; a kill before that step finds HEAD still on ``target`` and the restore
    redoes it first, only while HEAD still names the ``ref`` recorded here (empty: detached).
    """
    marker = interrupted_pull_marker(root)
    ref = ""
    if rollback:
        named = run_git(git_cmd, ["symbolic-ref", "-q", "HEAD"], cwd=str(root), capture_output=True, text=True,
                        encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, timeout=60)
        ref = named.stdout.strip() if named.returncode == 0 else ""
    if pre:
        with suppress(OSError, subprocess.SubprocessError):  # no closure: the in-tree repair still runs
            publish_recovery_closure(git_cmd, root, pre)
    # The restore's only record, durable before git writes: temp + fsync + rename, never in place.
    # ``index_lock``: the lock generation already there before this move's git ran (empty: none). Only
    # THAT one is another git's for sure; a lock that appears later can be our own killed git's.
    write_durable_text(marker, f"pid={os.getpid()}\npre={pre or ''}\ntarget={target}\nstash={stash or ''}\n"
                       + f"index_lock={_lock_identity(marker.parent / 'index.lock')}\n"
                       + (f"rollback={rollback}\nref={ref}\n" if rollback else "")
                       + (f"git={git}\n" if (git := _absolute_git(git_cmd)) else ""))
    return marker


def publish_recovery_closure(git_cmd, root: Path, pre: str) -> Path:
    """Copy the launch repair's modules, as committed at ``pre``, beside the marker (outside the tree).

    The bytes come from the commit the repair restores, never from the working tree: a rollback move
    starts from a broken tree, and a second move in one run starts from code this process never ran.
    A manifest names each file's blob id; the launcher runs the closure only when every file hashes
    to it (``recovery_closure_verified``) and rebuilds it from git's objects otherwise. Files and
    directory are fsynced before the atomic rename, so a power loss during the move that follows
    finds the whole closure or none. An older run's closure (another ``pre``) is dropped.
    """
    import shutil

    if not is_object_id(pre):
        raise OSError(f"not a commit id: {pre!r}")
    dest = recovery_closure_dir(Path(root), pre)
    if recovery_closure_verified(dest, pre):
        return dest
    git_cmd = [*([git_cmd] if isinstance(git_cmd, str) else git_cmd), "--no-replace-objects"]
    listed = run_git(git_cmd, ["ls-tree", "-z", "--full-tree", pre, "--", *RECOVERY_CLOSURE], cwd=str(root),
                     capture_output=True, stdin=subprocess.DEVNULL, timeout=120)
    ids = {}
    for entry in (listed.stdout or b"").split(b"\0") if listed.returncode == 0 else ():
        meta, _, rel = entry.decode("utf-8", "replace").partition("\t")
        if meta.split()[1:2] == ["blob"]:
            ids[rel] = meta.split()[2]
    blobs = {RECOVERY_CLOSURE_INIT: b""}
    shown = read_target_files(git_cmd, root, pre, RECOVERY_CLOSURE)  # one spawn, not a cat-file per file
    for rel in RECOVERY_CLOSURE:
        if rel not in ids or shown[rel] is None or blob_id(shown[rel], pre) != ids[rel]:
            raise OSError(f"{rel} is not in {pre[:10]}")
        blobs[rel] = shown[rel]
    manifest = "".join(f"{blob_id(data, pre)} {rel}\n" for rel, data in blobs.items()).encode("utf-8")
    staging = dest.with_name(f".{pre}.{os.getpid()}.staging")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        for rel, data in {**blobs, RECOVERY_CLOSURE_MANIFEST: manifest}.items():
            (staging / rel).parent.mkdir(parents=True, exist_ok=True)
            with open(staging / rel, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        for directory in (staging / "hermes_cli", staging):
            _fsync_dir(directory)
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(staging, dest)
        _fsync_dir(dest.parent)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    for stale in dest.parent.iterdir():
        if stale != dest:
            shutil.rmtree(stale, ignore_errors=True)
    return dest


def _fsync_dir(directory: Path) -> None:
    """POSIX: make a directory's entries durable. Windows opens no directory handle; NTFS journals
    the rename itself."""
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _absolute_git(git_cmd) -> str:
    """The git this run moves the tree with, as an absolute path for the next launch's repair.

    A Windows install's git is often PM's store copy, on PATH only inside the updater
    (``expose_pm_git``); the repair cannot look it up through ``pm`` when the killed move tore a
    module ``pm`` imports (``hermes_constants``)."""
    import shutil

    head = (list(git_cmd) or ["git"])[0] if not isinstance(git_cmd, str) else git_cmd
    found = shutil.which(head)
    return os.path.abspath(found) if found else ""


def files_added_by(git_cmd, root: Path, pre: str, target: str | None) -> list[str]:
    """Paths ``target`` adds over ``pre``. A rollback's mixed reset un-tracks them and ``reset --hard``
    never touches untracked files, so ``drop_added_files`` removes them by name afterwards."""
    if not target or target == pre:
        return []
    cp = run_git(git_cmd, ["diff", "--name-only", "-z", "--no-renames", "--diff-filter=A", pre, target],
                        cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace",
                        stdin=subprocess.DEVNULL, timeout=120)
    return [p for p in cp.stdout.split("\0") if p] if cp.returncode == 0 else []


def drop_added_files(root: Path, added: list[str]) -> bool:
    """Every one is the move's own file: git refuses to move a checkout over an untracked file.

    False when one of them is still there (the caller keeps the marker; the restore retries)."""
    root = Path(root)
    gone = True
    for rel in added:
        try:
            (root / rel).unlink(missing_ok=True)
        except OSError:
            gone = False
    for parent in sorted({p for rel in added for p in Path(rel).parents if str(p) != "."},
                         key=lambda p: len(p.parts), reverse=True):
        try:
            (root / parent).rmdir()  # only when empty: anything else inside keeps it
        except OSError:
            pass
    return gone and not any(os.path.lexists(root / rel) for rel in added)


def tree_whole_at(git_cmd, root: Path, sha: str) -> bool:
    """HEAD is ``sha``, no ``index.lock`` is left and no tracked file differs from it."""
    def run(*args: str) -> subprocess.CompletedProcess:
        return run_git(git_cmd, [*args], cwd=str(root), capture_output=True, text=True, encoding="utf-8",
                              errors="replace", stdin=subprocess.DEVNULL, timeout=120)

    head = run("rev-parse", "-q", "--verify", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != sha or (interrupted_pull_marker(Path(root)).parent / "index.lock").exists():
        return False
    status = run("status", "--porcelain", "-z", "--untracked-files=no")
    return status.returncode == 0 and not status.stdout.strip("\0")


def settle_failed_tree_move(root: Path) -> bool:
    """Git exited without finishing a tree move: put back what it wrote, keep the marker if we can't.

    True when the tree is verified whole again (at the pre-move commit, or at the target when git got
    there anyway); False leaves the marker for the next launch's ``restore_interrupted_pull``.
    """
    marker = interrupted_pull_marker(Path(root))
    if not marker.is_file():
        return False  # no marker names the move: its absence proves nothing about the tree
    restore_interrupted_pull(Path(root), after_failure=True)
    return not marker.is_file()


def _requires_python_spec(pyproject: bytes | str | None) -> str | None:
    """The target's ``requires-python`` specifier, None when absent or unreadable."""
    if not pyproject:
        return None
    import tomllib

    try:
        text = pyproject.decode("utf-8") if isinstance(pyproject, bytes) else pyproject
        spec = tomllib.loads(text)["project"]["requires-python"]
    except (KeyError, TypeError, ValueError):  # TOMLDecodeError is a ValueError
        return None
    return spec if isinstance(spec, str) else None


def _spec_admits(spec: str, version: str) -> bool:
    import re

    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
    except ImportError:  # the lower bound is the part a Python bump moves
        floor = re.search(r">=\s*(\d+)\.(\d+)", spec)
        found = re.match(r"(\d+)\.(\d+)", version)
        return not floor or bool(found) and (int(found[1]), int(found[2])) >= (int(floor[1]), int(floor[2]))
    try:
        return SpecifierSet(spec).contains(version, prereleases=True)
    except InvalidSpecifier:
        return True  # unknown: keep judging with this interpreter, never wave a file through


def requires_other_python(pyproject: bytes | str | None) -> bool:
    """True when a target's ``requires-python`` excludes the running interpreter.

    Its startup modules may then use a newer Python's syntax that this interpreter's ``compile()``
    cannot judge: ``startup_syntax_error`` asks an installed interpreter the target admits instead.
    Unknown -> False: keep checking, so a real syntax error is never waved through on a guess.
    """
    import platform

    spec = _requires_python_spec(pyproject)
    return spec is not None and not _spec_admits(spec, platform.python_version())


_PYTHON_VERSION = "import platform; print(platform.python_version())"
# Run by the target's interpreter: compile every startup file sent on stdin, print the first error.
_COMPILE_ALL = (
    "import base64, json, sys\n"
    "verdict = None\n"
    "for rel, data in json.load(sys.stdin).items():\n"
    "    try:\n"
    "        compile(base64.b64decode(data), rel, 'exec', dont_inherit=True)\n"
    "    except (SyntaxError, ValueError) as exc:\n"
    "        verdict = [rel, f'{type(exc).__name__}: {exc}']\n"
    "        break\n"
    "print(json.dumps(verdict))\n")


def _run_python(argv: list[str], *, stdin: str = "", timeout: float = 60) -> str | None:
    """Stdout of a read-only interpreter child (it never writes the checkout), None when it failed."""
    from runtime.subprocess_compat import windows_hide_flags

    try:
        done = subprocess.run(argv, input=stdin, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, creationflags=windows_hide_flags())
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def _admitted_python(spec: str) -> str | None:
    """An installed ``python3.N`` the target's ``requires-python`` admits (newest first), or None
    when this machine has none. No uv tier: Hermes never resolves the user's uv
    (tests/test_managed_runtime_resolution.py), and a missing interpreter only means this one judges."""
    from hermes_platform.resolver import locate_command

    candidates = [res.command[0] for minor in range(40, sys.version_info[1], -1)
                  if (res := locate_command(f"python3.{minor}")).command][:3]
    for python in candidates:
        version = (_run_python([python, "-I", "-c", _PYTHON_VERSION], timeout=30) or "").strip()
        if version and _spec_admits(spec, version):
            return python
    return None


def _compile_here(sources: dict[str, bytes]) -> tuple[str, str] | None:
    for rel, data in sources.items():
        try:
            compile(data, rel, "exec", dont_inherit=True)
        except (SyntaxError, ValueError) as exc:
            return rel, f"{type(exc).__name__}: {exc}"
    return None


def startup_syntax_error(sources: dict[str, bytes | None], pyproject: bytes | None) -> tuple[str, str] | None:
    """``(path, error)`` for the first startup file that does not compile, else None.

    Judged by the interpreter that will run the target: this one when the target's
    ``requires-python`` admits it, else an installed interpreter it admits (review N15: a
    conflict-marker scan is not a syntax check). With none installed, this interpreter still
    judges: code it parses is code a newer Python parses, and a refusal names the Python to install
    so the release can be judged by its own interpreter. ``None`` values (absent files) are skipped.
    """
    import base64
    import json
    import platform

    present = {rel: data for rel, data in sources.items() if data is not None}
    spec = _requires_python_spec(pyproject)
    if spec is None or _spec_admits(spec, platform.python_version()):
        return _compile_here(present)
    if python := _admitted_python(spec):
        payload = json.dumps({rel: base64.b64encode(data).decode("ascii") for rel, data in present.items()})
        out = _run_python([python, "-I", "-c", _COMPILE_ALL], stdin=payload, timeout=120)
        try:
            verdict = json.loads((out or "").strip().splitlines()[-1])
        except (IndexError, ValueError):
            verdict = False  # the target's interpreter gave no verdict: judge here (below)
        if verdict is None:
            return None
        if isinstance(verdict, list) and len(verdict) == 2:
            return str(verdict[0]), f"{verdict[1]} (judged by {python})"
    broken = _compile_here(present)
    if broken is None:
        return None
    return broken[0], (f"{broken[1]}\nJudged by Python {platform.python_version()}: the target requires Python "
                       f"{spec} and no installed interpreter satisfies it. Install one (`uv python install "
                       f"'{spec}'`) and re-run `hermes update` so the release is judged by its own Python.")


def read_target_files(git_cmd, root: Path, target_ref: str, relpaths) -> dict[str, bytes | None]:
    """``{rel: bytes}`` of each path as committed at ``target_ref`` (None: absent there or unreadable),
    in ONE ``git cat-file --batch`` instead of one ``git show`` per file."""
    names = list(dict.fromkeys(relpaths))
    found: dict[str, bytes | None] = dict.fromkeys(names)
    request = "".join(f"{target_ref}:{rel}\n" for rel in names).encode("utf-8")
    cp = run_git(git_cmd, ["cat-file", "--batch"], cwd=str(root), input=request, capture_output=True, timeout=120)
    out, pos = (cp.stdout or b"") if cp.returncode == 0 else b"", 0
    for rel in names:
        end = out.find(b"\n", pos)
        if end < 0:
            break
        header, pos = out[pos:end].split(), end + 1
        if len(header) == 3 and header[2].isdigit():  # "<oid> <type> <size>", then the bytes and "\n"
            size = int(header[2])
            if header[1] == b"blob":
                found[rel] = out[pos:pos + size]
            pos += size + 1
    return found


def target_syntax_error(git_cmd, root: Path, target_ref: str, relpaths) -> tuple[str, str] | None:
    """``(path, error)`` for the first startup-critical file that does not compile at ``target_ref``.

    Read from the object store, never written to the tree: this runs BEFORE HEAD moves, so a broken
    release is refused with the install untouched (the post-pull rollback stays as the backstop).
    ``startup_syntax_error`` judges it under the interpreter the target admits. A file absent at the
    target (or unreadable) is skipped: the post-pull guard has the last word.
    """
    files = read_target_files(git_cmd, root, target_ref, ["pyproject.toml", *relpaths])
    pyproject = files.pop("pyproject.toml", None)
    return startup_syntax_error(files, pyproject)


def tree_syntax_error(root: Path, relpaths) -> tuple[str, str] | None:
    """``startup_syntax_error`` for the files in the worktree at ``root`` (the post-move backstop)."""
    root = Path(root)
    sources: dict[str, bytes | None] = {}
    for rel in relpaths:
        try:
            sources[rel] = (root / rel).read_bytes() if (root / rel).is_file() else None
        except OSError as exc:
            return str(rel), f"could not read: {exc}"
    pyproject = root / "pyproject.toml"
    return startup_syntax_error(sources, pyproject.read_bytes() if pyproject.is_file() else None)


def head_and_branch(git_cmd, root: Path) -> tuple[str, str]:
    def out(*args: str) -> str:
        cp = run_git(git_cmd, [*args], cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace",
                            stdin=subprocess.DEVNULL, timeout=60)
        return cp.stdout.strip() if cp.returncode == 0 else ""

    return out("rev-parse", "HEAD"), out("rev-parse", "--abbrev-ref", "HEAD")


def checkout_untouched(git_cmd, root: Path, start: tuple[str, str] | None) -> bool:
    """True when the checkout is exactly where this run found it: same HEAD, same branch, no
    autostash taken, no tree move armed. Only then may a failed git update fall back to ZIP."""
    from hermes_cli.update_cmd_stash import _unrestored_autostash_notice

    if start is None or not start[0]:
        return False
    if _unrestored_autostash_notice() is not None or interrupted_pull_marker(root).is_file():
        return False
    return head_and_branch(git_cmd, root) == start


def begin_update_attempt() -> None:
    """A new ``hermes update`` attempt owns nothing yet: no snapshot, no stake, no run start.

    The state above is per attempt, not per process: a second update in one interpreter (a
    long-lived caller) that inherited the first one's snapshot and owner would hand back the
    first update's committed debt as its own undo when its move is refused (review O5).
    """
    global _armed_snapshot, _run_start, _obligation_root, _owner
    _owner = ""
    _armed_snapshot = None
    _armed_bytes.clear()
    _run_start = None
    _obligation_root = None


def record_run_start(git_cmd, root: Path) -> None:
    global _run_start, _obligation_root
    _run_start = (list(git_cmd), head_and_branch(git_cmd, root))
    _obligation_root = Path(root)


def run_checkout_untouched(root: Path) -> bool:
    """``checkout_untouched`` for this run; True before the checkout phase (nothing can have moved)."""
    if _run_start is None:
        return True
    return checkout_untouched(_run_start[0], Path(root), _run_start[1])


def preflight_refusal(git_cmd, root: Path, target_ref: str, critical_files) -> str | None:
    """Why this update must not start, checked BEFORE the first tree move; ``None`` to proceed.

    * a venv owned by another OS user (the completion child used to refuse only after the swap);
    * a target whose startup-critical modules do not compile (rollback used to be the only guard).
    """
    root = Path(root)
    if not _owns_live_checkout(root):
        try:
            from hermes_cli.venv_sync import refuse_foreign_owned_venv

            refuse_foreign_owned_venv(root)
        except ImportError:
            pass
        except Exception as exc:  # health: allow BLE001 -- fail closed: any probe error refuses before the first move
            _record_stop("venv_foreign_owner")
            return f"✗ {exc}"  # pm's refusal carries its own remediation text
    broken = target_syntax_error(git_cmd, root, target_ref, critical_files)
    if broken is not None:
        path, error = broken
        _record_stop("target_syntax_error")
        return (f"✗ The update target has a syntax error in a critical file:\n  {path}\n    "
                + "\n    ".join(error.splitlines()[:6]))
    return None
