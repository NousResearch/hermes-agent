"""Kanban worker mount-namespace isolation (t_ff7d30cc).

Decided mechanism: an unprivileged user+mount namespace (``unshare -Urm``) is
entered by a small launcher that bind-mounts the production agent tree
read-only and re-opens a minimal set of rw carve-outs before ``exec``-ing the
worker. The launcher is applied as an argv wrapper in
``hermes_cli.kanban_db_dispatch._default_spawn`` so a worker cannot mutate
``HERMES_AGENT_HOME`` (the live production tree for the whole fleet) even if
it has a shell.

Rejected alternatives (measured; see ``artifacts/t_f4806674/design-decision.md``):

* ``systemd-run --user --scope --property=ReadOnlyPaths=`` -> ``Unknown
  assignment: ReadOnlyPaths`` -- transient scopes cannot carry namespace
  properties; ``--scope`` is already the dispatcher's restart-safety wrapper.
* chmod / separate uid -> dispatcher, gateways and workers are all uid 1000;
  sudo needs a password on this host, so no privilege separation is available.
* Transient systemd *service* with ``ReadOnlyPaths=`` -> works for EROFS but
  does not inherit the dispatcher-built worker env (HERMES_HOME, kanban/board/
  claim env, scrubbed secrets) and needs stdout plumbing.

Carve-out set (exactly this, no more):

* the per-task worktree (dispatcher-provisioned before the sandbox exists,
  because ``.git/worktrees`` is ro under the sandbox by design),
* ``<agent_home>/.git/objects`` -- shared object store a commit needs,
* ``<agent_home>/.git/logs`` -- reflog,
* ``<agent_home>/.git/worktrees/<this-worktree-name>`` -- THIS worktree's
  ``index``/``HEAD`` (NEVER the whole ``.git/worktrees`` directory: a writable
  whole dir lets a worker register an arbitrary new worktree -- observed
  escape during design, then closed),
* ``<agent_home>/.git/refs/heads/<branch-dir>`` -- ONLY the directory
  containing the worker's own branch ref (e.g. ``wt`` for ``wt/<task-id>``).
  ``.git/refs/heads`` as a whole is **NOT** carved out: a worker that can
  update arbitrary refs can ``git update-ref refs/heads/canonical-deploy``
  to a different commit (validated escape during integration -- the
  reference bash design missed it because the harness tested only no-op
  updates where the target SHA matched the current value). The narrow
  carve-out refuses cross-branch moves while still letting the worker
  commit on its own branch via the standard git plumbing.

Two traps (both measured):

* Carve-outs must be sourced from an **rw staging bind of the parent taken
  before the tree is flipped ro** -- a bind from an already-ro mount inherits
  ``MS_RDONLY``.
* ``.git/worktrees`` must be carved out **per worktree name**, not as a whole
  directory.
* ``.git/refs/heads`` must be carved out **per branch directory**, not as a
  whole directory (validated escape during integration; see above).

Config gate: ``kanban.worker_isolation = off|warn|enforce`` (default
``enforce`` on Linux when unprivileged userns is available). Emergency kill
switch: ``HERMES_WORKER_ISOLATION=off``.

Known caveats:

* Supplementary groups are lost inside the user namespace; local group-gated
  sockets (e.g. ``/var/snap/lxd/common/lxd/unix.socket``, group ``lxd``) keep
  working, but the kernel maps the worker's in-namespace uid 0 back to the
  host's ``serveradmin`` via ``SO_PEERCRED``, so lxd sees the worker as
  fully-trusted ``serveradmin`` (MORE trust than expected, not less).
  Mitigation if a flow needs lxd-as-another-user: map the gid via
  ``newgidmap``/subgid, or run that one op outside the sandbox. Remote access
  is unaffected (docker on this host is already an ssh context).
* ``git fetch``/``pull`` inside the worker's own worktree fails -- ``FETCH_HEAD``
  lives in the ro main ``.git``. By design the dispatcher fetches/creates the
  worktree; ``hermes -w`` from inside a worker is likewise unavailable.
* Cron-scheduled agents (``cron/scheduler.py`` Popen) are a second spawn path
  and are NOT covered here -- exposure is owned by ``t_efef6176``.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

log = logging.getLogger("hermes_cli.worker_isolate")

# ----------------------------------------------------------------------
# Public gate
# ----------------------------------------------------------------------

# Mode strings the config understands. ``enforce`` is the only mode that
# actually wraps the worker argv; the others are observation-only.
MODE_OFF = "off"
MODE_WARN = "warn"
MODE_ENFORCE = "enforce"
_VALID_MODES = (MODE_OFF, MODE_WARN, MODE_ENFORCE)

# Environment kill switch (sets the floor for any non-off config).
_KILL_SWITCH = "HERMES_WORKER_ISOLATION"


@dataclass(frozen=True)
class IsolationConfig:
    """Resolved isolation policy for a single spawn.

    ``mode`` is what the dispatcher asked for; ``effective`` is what it
    actually applied (after kill switch, platform/permission checks,
    degrade-once-and-log). They differ only when isolation was requested
    but the host cannot deliver it -- and that always degrades to ``off``
    with a single warning, never to silent pretend.
    """

    mode: str
    effective: str
    reason: str = ""


def resolve_isolation_mode(configured: Optional[str]) -> IsolationConfig:
    """Resolve the effective isolation mode for this spawn.

    Resolution order:
      1. ``HERMES_WORKER_ISOLATION`` env var overrides (kill switch).
      2. ``configured`` (the ``kanban.worker_isolation`` config value).
      3. Default: ``enforce`` on Linux when unprivileged userns is
         available; ``off`` otherwise.

    Anything that disables isolation goes through ``_degrade`` so a single
    warning is logged once per process -- never a per-spawn flood, never a
    silent pretend.
    """
    kill = os.environ.get(_KILL_SWITCH, "").strip().lower()
    if kill in (MODE_OFF, "0", "false", "no"):
        return _degrade(MODE_OFF, f"kill switch {_KILL_SWITCH}={kill!r}")
    if kill in (MODE_ENFORCE, "1", "true", "yes"):
        configured = MODE_ENFORCE

    mode = (configured or "").strip().lower() or _default_mode()
    if mode not in _VALID_MODES:
        return _degrade(MODE_OFF, f"unknown kanban.worker_isolation={mode!r}")
    if mode in (MODE_OFF, MODE_WARN):
        return IsolationConfig(mode=mode, effective=MODE_OFF, reason=f"configured mode={mode}")
    # mode == enforce -- actually need the kernel.
    if not _userns_available():
        return _degrade(MODE_OFF, "unprivileged user namespace unavailable on this host")
    return IsolationConfig(mode=MODE_ENFORCE, effective=MODE_ENFORCE)


# Single-shot warning -- don't flood the worker log every spawn.
_DEGRADE_WARNED: bool = False


def _degrade(effective: str, reason: str) -> IsolationConfig:
    global _DEGRADE_WARNED
    if not _DEGRADE_WARNED:
        log.warning(
            "kanban worker isolation degraded to %s: %s "
            "(set HERMES_WORKER_ISOLATION=off to silence, or fix the cause)",
            effective, reason,
        )
        _DEGRADE_WARNED = True
    return IsolationConfig(mode=MODE_OFF, effective=effective, reason=reason)


def _default_mode() -> str:
    """Linux + userns available -> enforce. Else -> off.

    Windows / macOS hosts simply don't have the unprivileged user
    namespace mechanism this card relies on; they fall through to ``off``
    and rely on the deploy-branch + kanban worktree conventions for
    protection (which the canonical-deploy-branch wiki page already
    documents).
    """
    if os.name != "posix":
        return MODE_OFF
    if not _is_linux():
        return MODE_OFF
    if not _userns_available():
        return MODE_OFF
    return MODE_ENFORCE


def _is_linux() -> bool:
    return os.sys.platform == "linux"


def _userns_available() -> bool:
    """Can *this* process enter an unprivileged user namespace?

    Cached for the lifetime of the process: probing requires ``unshare``
    itself, and probing once per spawn is acceptable overhead but pointless
    to repeat. Negative results are sticky; if you need a re-probe, restart
    the dispatcher.
    """
    global _USERN_PROBE
    if _USERN_PROBE is not None:
        return _USERN_PROBE
    binary = shutil.which("unshare")
    if binary is None:
        _USERN_PROBE = False
        return False
    try:
        import subprocess as _sp
        rc = _sp.run(
            [binary, "--user", "--map-root-user", "true"],
            capture_output=True,
            timeout=5,
            stdin=_sp.DEVNULL,
        ).returncode
    except Exception:
        rc = 1
    _USERN_PROBE = rc == 0
    return _USERN_PROBE


_USERN_PROBE: Optional[bool] = None


def load_configured_isolation_mode() -> Optional[str]:
    """Read ``kanban.worker_isolation`` from the resolved config.

    Mirrors ``configured_max_in_progress``'s shape so dispatcher integration
    is a one-line pattern. Returns ``None`` when unset / unparsable so the
    resolver's default-mode path runs.
    """
    try:
        from hermes_cli.config import load_config_readonly
        raw = (load_config_readonly() or {}).get("kanban", {}).get("worker_isolation")
    except Exception:
        return None
    if raw is None:
        return None
    if isinstance(raw, str):
        val = raw.strip().lower()
        return val or None
    if isinstance(raw, bool):
        # bool is a subclass of int -- handle it BEFORE the int check below.
        return MODE_ENFORCE if raw else MODE_OFF
    if isinstance(raw, int):
        return MODE_ENFORCE if raw else MODE_OFF
    return None


# ----------------------------------------------------------------------
# Argv builder (pure -- unit-testable without mounting)
# ----------------------------------------------------------------------


def build_isolated_worker_argv(
    command: Sequence[str],
    *,
    agent_home: str,
    worktree: Optional[str] = None,
    extra_rw: Sequence[str] = (),
) -> List[str]:
    """Return the launcher-prefixed worker argv.

    The nesting the dispatcher needs is ``[unshare, -Urm, launcher.py, --,
    command...]``. The launcher's own argv includes the carve-out paths so
    it can bind-mount them rw before flipping the tree ro.

    Pure function: no syscalls, no env reads -- unit-test the shape without
    ever touching the host mount table. The actual mount dance lives in
    :func:`run_isolated_launcher` and is exercised only on a real spawn.
    """
    if not command:
        raise ValueError("worker command must be a non-empty sequence")
    if not agent_home:
        raise ValueError("agent_home is required for worker isolation")
    launcher_argv = _build_launcher_argv(agent_home=agent_home, worktree=worktree, extra_rw=extra_rw)
    # ``--`` separates the launcher from the worker argv so a worker command
    # beginning with a flag never gets eaten by the launcher.
    return ["unshare", "-Urm", *launcher_argv, "--", *command]


def _build_launcher_argv(
    *,
    agent_home: str,
    worktree: Optional[str],
    extra_rw: Sequence[str],
) -> List[str]:
    """Argv for the launcher itself (without the ``unshare -Urm`` prefix).

    Layout: ``[<py>, -m, hermes_cli.worker_isolate, _launcher, <agent_home>,
    [--worktree <path>] [--rw <path>...] -- <worker argv...>]``.
    """
    argv: List[str] = list(_launcher_module_target())
    argv.append(agent_home)
    if worktree:
        argv += ["--worktree", worktree]
    if extra_rw:
        argv += ["--rw", *extra_rw]
    return argv


def _launcher_module_target() -> str:
    """Build the launcher's exec argv as two tokens.

    The dispatcher already guarantees the worker is on the same venv it is
    (``HERMES_HOME`` + resolved profile), so ``sys.executable`` is the right
    interpreter. We return the pair ``[sys.executable, '-m',
    'hermes_cli.worker_isolate', '_launcher']`` and let the caller splice
    it into the argv (the ``-m`` flag plus module path is what Python's
    runpy machinery actually wants).
    """
    import sys as _sys
    exe = _sys.executable or "python3"
    return [exe, "-m", "hermes_cli.worker_isolate", "_launcher"]


# ----------------------------------------------------------------------
# Launcher entry point -- runs in the new namespace, executes the worker
# ----------------------------------------------------------------------


def _launcher_main(argv: List[str]) -> int:
    """Standalone entry point for ``python -m hermes_cli.worker_isolate _launcher -- ...``.

    Runs inside the freshly-created user+mount namespace. The parent
    dispatcher already entered ``unshare -Urm``; here we do the bind dance
    and ``exec`` the worker argv.

    Argv layout::

        hermes_cli.worker_isolate _launcher <agent_home> [--worktree <path>]
            [--rw <path>...] -- <worker argv...>
    """
    if len(argv) < 3 or argv[1] != "_launcher":
        sys.stderr.write(
            "usage: python -m hermes_cli.worker_isolate _launcher "
            "<agent_home> [--worktree <path>] [--rw <path>...] -- <cmd...>\n"
        )
        return 64  # EX_USAGE

    args = argv[2:]
    try:
        agent_home, worktree, extra_rw, worker_argv = _parse_launcher_argv(args)
    except ValueError as exc:
        sys.stderr.write(f"hermes worker isolation: {exc}\n")
        return 64

    rc = _enter_isolation(agent_home=agent_home, worktree=worktree, extra_rw=extra_rw)
    if rc != 0:
        return rc

    # exec replaces the launcher with the worker; only an exec failure returns.
    try:
        os.execvp(worker_argv[0], worker_argv)
    except OSError as exc:
        sys.stderr.write(f"hermes worker isolation: exec {worker_argv[0]!r} failed: {exc}\n")
        return 127  # EX_NOTFOUND -- but only as a fallback; the worker IS the long-lived thing.


def _parse_launcher_argv(args: List[str]) -> tuple[str, Optional[str], List[str], List[str]]:
    """Parse the post-``_launcher`` argv into (agent_home, worktree, extras, worker_argv)."""
    if not args or args[0].startswith("-"):
        raise ValueError("first arg must be the agent home path")
    agent_home = args[0]
    worktree: Optional[str] = None
    extra_rw: List[str] = []
    i = 1
    while i < len(args) and args[i] != "--":
        a = args[i]
        if a == "--worktree":
            if i + 1 >= len(args):
                raise ValueError("--worktree requires a value")
            worktree = args[i + 1]
            i += 2
            continue
        if a == "--rw":
            i += 1
            while i < len(args) and not args[i].startswith("-"):
                extra_rw.append(args[i])
                i += 1
            continue
        raise ValueError(f"unknown launcher flag: {a!r}")
    if i >= len(args):
        raise ValueError("missing -- separator before worker argv")
    worker_argv = args[i + 1:]
    if not worker_argv:
        raise ValueError("worker argv after -- is empty")
    return agent_home, worktree, extra_rw, worker_argv


def _enter_isolation(
    *, agent_home: str, worktree: Optional[str], extra_rw: Sequence[str]
) -> int:
    """Apply the bind dance. Mirrors the validated reference launcher,
    with two security-tightening departures:

    1. ``.git/refs/heads`` is carved out **only for the worker's own branch**,
       not as a whole directory. The wider carve-out lets a worker
       ``update-ref refs/heads/canonical-deploy`` to a different commit
       (validated escape during integration testing -- the original bash
       design decision missed it because the harness tested only no-op
       updates where ``$CANON`` was the current value). The narrow
       carve-out refuses the cross-branch move while still letting the
       worker update its own branch ref via the standard git plumbing.

    2. ``.git/packed-refs`` is NOT carved out, so a worker can't
       resurrect a packed ref that the dispatcher unpacks to defang
       an update.

    Errors return distinct exit codes (91=stage bind, 92=prod bind, 93=remount ro,
    94=carve bind) so the launcher's first failing mount is identifiable from
    a worker log without re-running.
    """
    import tempfile

    parent = os.path.dirname(agent_home)
    if not os.path.isdir(parent):
        sys.stderr.write(f"hermes worker isolation: parent {parent!r} not a directory\n")
        return 90

    stage = tempfile.mkdtemp(prefix="hermes-isolate-stage.")
    try:
        # 1. rw staging view of the parent, taken BEFORE the tree is flipped
        #    ro. The carve-out sources bind from this rw staging view because
        #    a bind from an already-ro mount inherits MS_RDONLY.
        rc = _mount(["--bind", parent, stage])
        if rc != 0:
            return 91
        # 2. the production tree -> read-only.
        rc = _mount(["--bind", agent_home, agent_home])
        if rc != 0:
            return 92
        rc = _mount(["-o", "remount,bind,ro", agent_home])
        if rc != 0:
            return 93
        # 3. carve-outs, sourced from the rw staging bind.
        #    ``.git/refs/heads`` is intentionally NOT included (see docstring);
        #    the worker's own branch carve-out is added below when worktree
        #    is provided.
        carve: List[str] = [f"{agent_home}/.git/objects", f"{agent_home}/.git/logs"]
        if worktree:
            carve.append(worktree)
            carve.append(f"{agent_home}/.git/worktrees/{os.path.basename(worktree)}")
            # The worker's own branch ref lives under ``refs/heads/<branch>``.
            # We resolve the branch name from the worktree's HEAD so the
            # carve-out matches what git will actually try to update. If
            # the worktree has no HEAD yet (e.g. fresh worktree before any
            # checkout), the carve-out is skipped and the worker will get
            # the standard ro denial on its first ref write -- still safe.
            worker_ref_dir = _worker_branch_ref_dir(agent_home, worktree)
            if worker_ref_dir:
                # ``mkdir -p`` in the rw staging bind so the carve-out
                # path EXISTS (mount --bind onto a non-existent path
                # would fail, and even if it succeeded git can't create
                # ref files in a directory that doesn't exist on disk).
                _ensure_dir_in_stage(stage, parent, worker_ref_dir)
                carve.append(worker_ref_dir)
        for extra in extra_rw:
            carve.append(extra)
        for sub in carve:
            if not os.path.lexists(sub):
                # The carve-out path may legitimately not exist on this host
                # (e.g. .git/logs on a fresh worktree); skip silently.
                continue
            rel = sub
            if sub.startswith(parent + "/"):
                rel = sub[len(parent) + 1:]
            src = os.path.join(stage, rel)
            if not os.path.lexists(src):
                continue
            rc = _mount(["--bind", src, sub])
            if rc != 0:
                return 94
        return 0
    finally:
        # The staging dir was only a mount source -- leave it; nothing inside
        # is exposed.
        pass


def _ensure_dir_in_stage(stage: str, parent: str, target: str) -> None:
    """Create ``target``'s parent directories inside the rw staging bind.

    Carve-out targets must exist as paths inside the staging bind BEFORE
    we bind them over the (then-ro) production path. For directories
    holding ref files, that means we need to mkdir the path on the
    staging side first; otherwise git's ``fopen(<refpath>, 'w')`` would
    fail with ENOENT under the sandbox even though the file's directory
    itself was rw.

    No-op if ``target`` does not start with ``parent/`` (defensive).
    """
    if not target.startswith(parent + "/"):
        return
    rel = target[len(parent) + 1:]
    stage_path = os.path.join(stage, rel)
    if os.path.exists(stage_path):
        return
    try:
        os.makedirs(stage_path, exist_ok=True)
    except OSError:
        # Best effort -- if mkdir fails, the bind will silently skip the
        # missing source path. The worker's first ref write will then be
        # denied by the ro bind on the parent dir, which is the safe
        # default.
        pass


def _worker_branch_ref_dir(agent_home: str, worktree: str) -> Optional[str]:
    """Resolve the worker's own branch ref directory.

    Reads the worktree's HEAD (either the gitdir pointer + bookkeeping
    file, or a real HEAD for legacy layouts) and returns the on-disk
    directory that contains the branch ref (e.g.
    ``<agent_home>/.git/refs/heads/wt`` for a worker on ``wt/<task-id>``).
    Returns ``None`` when the worktree has no HEAD yet (fresh
    ``git worktree add``); the carve-out is skipped and the worker's
    first ref write will be denied by the standard ro bind, which is
    the safe default.
    """
    head_path: Optional[str] = None
    worktree_git = os.path.join(worktree, ".git")
    if os.path.isfile(worktree_git):
        # Modern linked-worktree layout: ``<worktree>/.git`` is a file
        # containing ``gitdir: <path>``. The real HEAD lives at
        # ``<path>/HEAD``.
        try:
            line = open(worktree_git, "r").read().strip()
        except OSError:
            return None
        if not line.startswith("gitdir:"):
            return None
        gitdir = line[len("gitdir:"):].strip()
        head_path = os.path.join(gitdir, "HEAD")
    elif os.path.isdir(worktree_git):
        # Legacy / non-linked layout: HEAD sits directly under .git.
        head_path = os.path.join(worktree_git, "HEAD")
    else:
        # Fall back to the worktree's own HEAD (uncommon, but cheap).
        head_path = os.path.join(worktree, "HEAD")
    if not head_path or not os.path.isfile(head_path):
        return None
    try:
        head = open(head_path, "r").read().strip()
    except OSError:
        return None
    # Detached HEAD (``<sha>``) -- no branch ref dir to carve out.
    if not head.startswith("ref:"):
        return None
    ref = head[len("ref:"):].strip()  # e.g. ``refs/heads/wt/t_x``
    if not ref.startswith("refs/heads/"):
        return None
    branch_path = ref[len("refs/heads/"):]  # e.g. ``wt/t_x``
    branch_dir, _ = os.path.split(branch_path)
    if not branch_dir:
        # Top-level branch like ``main`` -- the carve-out would be
        # ``<agent_home>/.git/refs/heads`` itself, which is exactly the
        # path we're refusing to expose. Skip and let the worker's
        # first write fail safe.
        return None
    return os.path.join(agent_home, ".git", "refs", "heads", branch_dir)


def _mount(args: List[str]) -> int:
    """Run ``mount <args>`` and return 0 on success, 1 on failure."""
    import subprocess as _sp
    binary = shutil.which("mount") or "/usr/bin/mount"
    try:
        return _sp.call([binary, *args], stdin=_sp.DEVNULL, stdout=_sp.DEVNULL, stderr=_sp.PIPE)
    except FileNotFoundError:
        return 1


# ----------------------------------------------------------------------
# Module-as-script: dispatch on the first argv token.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(_launcher_main(sys.argv))
