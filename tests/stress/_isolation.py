"""Fail-closed board isolation for the destructive kanban stress scripts.

These scripts claim, fail, reclaim and complete tasks in tight loops. Pointed
at the wrong database they are indistinguishable from a fleet of malicious
workers, so "which DB am I on?" is a safety property, not a convenience.

**Why ``HERMES_HOME`` was never enough.** ``kanban_db.kanban_db_path()``
resolves in this order:

1. ``HERMES_KANBAN_DB`` — pins the file directly,
2. the active board,
3. ``<root>/kanban.db``, where ``<root>`` comes from ``HERMES_KANBAN_HOME``
   or ``HERMES_HOME``.

The scripts set only ``HERMES_HOME`` (step 3). Anything that had already
exported ``HERMES_KANBAN_DB`` — most obviously the dispatcher, which injects
it into every worker's environment as defense in depth — therefore won at
step 1. A stress run launched from inside a worker printed a
``/var/folders/...`` home and drove the operator's real board: synthetic
workers marked live cards done. The same inheritance applies to
``HERMES_KANBAN_HOME``, ``HERMES_KANBAN_WORKSPACES_ROOT`` and friends.

So this module does not *ask* for isolation, it *proves* it:

* the home is a directory this module created under the system temp root,
  stamped with :data:`SANDBOX_MARKER`;
* the DB is a file that does not exist yet, directly inside that home;
* neither is the live default board or anywhere under the real ``~/.hermes``;
* every ``HERMES_KANBAN_*`` variable is scrubbed and only the sandbox ones
  are re-pinned — a *new* redirect var added upstream is scrubbed by default
  rather than silently inherited;
* and finally ``kanban_db_path()`` is *called* and its answer compared to the
  sandbox. That last check is what makes the guard survive a future change to
  the resolution order: if anything still out-votes the pin, the script
  refuses to start.

Every entry point calls :func:`isolate_stress_home` before importing
``hermes_cli.kanban_db``; every spawned worker calls
:func:`adopt_stress_home` rather than trusting what it inherited.
:class:`StressIsolationError` is fatal by design — there is no fallback to
"probably fine".
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

__all__ = [
    "SANDBOX_MARKER",
    "StressHome",
    "StressIsolationError",
    "adopt_stress_home",
    "assert_disposable_sandbox",
    "isolate_stress_home",
]

#: Stamped into every sandbox home. A child process is handed a path, not a
#: promise; the marker is how it tells a sandbox from any other directory.
SANDBOX_MARKER = ".hermes_stress_sandbox"

#: Board layout inside the sandbox. ``default`` resolves to ``<root>/kanban.db``,
#: so the sandbox mirrors a real board exactly.
_DB_NAME = "kanban.db"
_BOARD = "default"

#: Scrubbed wholesale before the sandbox values are pinned. Anything matching
#: this prefix either redirects path resolution or names a row on the live
#: board (``HERMES_KANBAN_TASK``, ``_RUN_ID``, ``_CLAIM_LOCK`` — a stress
#: worker inheriting those can complete somebody else's run).
_SCRUB_PREFIX = "HERMES_KANBAN"

#: Scrubbed by name: profile selection re-anchors ``HERMES_HOME`` at
#: ``<root>/profiles/<name>``, which is not the sandbox.
_SCRUB_EXACT = ("HERMES_PROFILE", "HERMES_PROFILE_NAME", "HERMES_HOME_ENSURED")


class StressIsolationError(RuntimeError):
    """The stress script could not prove its board is disposable.

    Always fatal. Catching this to continue would defeat the entire point.
    """


@dataclass(frozen=True)
class StressHome:
    """A proven-disposable sandbox and the environment that pins it."""

    home: Path
    db: Path
    env: Mapping[str, str] = field(default_factory=dict)

    def child_env(self) -> dict[str, str]:
        """Environment for a spawned worker: scrubbed, then re-pinned.

        Built from the sandbox values rather than copied from ``os.environ``
        verbatim, so a variable that reappeared after isolation cannot ride
        along into the child.
        """
        env = {
            key: value for key, value in os.environ.items()
            if not _is_scrubbed(key)
        }
        env.update(self.env)
        return env


def _is_scrubbed(key: str) -> bool:
    return key.startswith(_SCRUB_PREFIX) or key in _SCRUB_EXACT


def _is_within(path: Path, root: Path) -> bool:
    """True when ``path`` is strictly inside ``root`` (both resolved)."""
    try:
        resolved = path.resolve()
        root = root.resolve()
    except OSError:  # pragma: no cover - unreadable path is not a sandbox
        return False
    return root != resolved and root in resolved.parents


def _real_home() -> Path | None:
    """The operator's home, as it looks *before* ``HOME`` is rewritten.

    ``HERMES_REAL_HOME`` is what the dispatcher exports when it rewrites
    ``HOME``, so it survives that indirection; without it the OS home is
    still trustworthy. Resolved eagerly at isolation time and re-exported
    verbatim, because after ``_apply`` runs ``HOME`` names the sandbox and
    the answer would no longer be recoverable.
    """
    raw = os.environ.get("HERMES_REAL_HOME")
    if not raw:
        return _ambient_home()
    try:
        return Path(raw).expanduser()
    except (OSError, RuntimeError):  # pragma: no cover - unusable value
        return _ambient_home()


def _ambient_home() -> Path | None:
    """``Path.home()``, or ``None`` where the process has no home at all."""
    try:
        return Path.home()
    except (OSError, RuntimeError):  # pragma: no cover - no home at all
        return None


def _live_roots() -> Iterable[Path]:
    """Directories that certainly hold a real board.

    Both the exported real home and the ambient one, because before
    isolation they can legitimately differ and either may hold the board.
    Afterwards ``HOME`` is the sandbox, and the first candidate is the only
    one still pointing anywhere real.
    """
    seen: set[Path] = set()
    for base in (_real_home(), _ambient_home()):
        if base is None:
            continue
        root = base / ".hermes"
        if root not in seen:
            seen.add(root)
            yield root


def assert_disposable_sandbox(
    home: Path,
    db: Path,
    *,
    require_marker: bool = True,
    allow_existing_db: bool = False,
) -> None:
    """Raise :class:`StressIsolationError` unless ``db`` is safe to destroy.

    ``allow_existing_db`` is for :func:`adopt_stress_home`: by the time a
    spawned worker runs, the parent has already created the schema. Every
    other rule still binds, so an existing *live* board is refused there too
    — it fails the containment and marker checks first.
    """
    home = Path(home)
    db = Path(db)
    tmp_root = Path(tempfile.gettempdir())

    def refuse(why: str) -> None:
        raise StressIsolationError(
            f"refusing to run a destructive stress script: {why}\n"
            f"  home = {home}\n"
            f"  db   = {db}\n"
            f"  temp root = {tmp_root}\n"
            "This guard is fail-closed. It never falls back to the "
            "ambient HERMES_KANBAN_DB / HERMES_HOME."
        )

    if not home.is_dir():
        refuse("the sandbox home does not exist or is not a directory")
    if not _is_within(home, tmp_root):
        refuse(
            "the sandbox home is not under the system temp root, so it was "
            "not created by this guard"
        )
    if require_marker and not (home / SANDBOX_MARKER).is_file():
        refuse(
            f"the sandbox home carries no {SANDBOX_MARKER} marker — it is "
            "some other directory, not a home this guard created"
        )
    if db.parent.resolve() != home.resolve():
        refuse("the database is not directly inside the sandbox home")
    for live in _live_roots():
        if db.resolve() == (live / _DB_NAME).resolve() or _is_within(db, live):
            refuse(f"the database is the live board under {live}")
    if db.exists() and not allow_existing_db:
        refuse(
            "the database already exists — a fresh sandbox cannot contain a "
            "board, so this is somebody else's"
        )


def _sandbox_env(home: Path, db: Path) -> dict[str, str]:
    """The variables that pin every kanban path inside ``home``.

    ``HERMES_KANBAN_DB`` and ``HERMES_KANBAN_HOME`` are independent locks at
    different steps of the resolution order: even if one were ignored, the
    other still lands inside the sandbox.

    ``HERMES_REAL_HOME`` is the exception: it is not a redirect, it is the
    label on the tree that must never be touched. Pinning it at the sandbox
    would make every child agree that the operator's home *is* the sandbox,
    and :func:`_live_roots` — the only rule that names the live board —
    would then match nothing. It is carried through unchanged instead.
    """
    real_home = _real_home()
    return {
        "HOME": str(home),
        "HERMES_HOME": str(home),
        **({"HERMES_REAL_HOME": str(real_home)} if real_home else {}),
        "HERMES_KANBAN_HOME": str(home),
        "HERMES_KANBAN_DB": str(db),
        "HERMES_KANBAN_BOARD": _BOARD,
        "HERMES_KANBAN_WORKSPACES_ROOT": str(home / "kanban" / "workspaces"),
        "HERMES_KANBAN_ATTACHMENTS_ROOT": str(home / "kanban" / "attachments"),
    }


def _apply(env: Mapping[str, str]) -> None:
    """Scrub every inherited kanban variable, then pin the sandbox.

    Deliberately not restored on failure: leaving the process pointed at the
    sandbox is the safe end state, and the caller is expected to die anyway.
    """
    for key in [k for k in os.environ if _is_scrubbed(k)]:
        del os.environ[key]
    os.environ.update(env)


def _verify_resolver(db: Path) -> None:
    """Ask production code where the board is and insist on the sandbox.

    Imported here, after the environment is pinned, so nothing caches a live
    path on the way in. This is the check that does not depend on this
    module's list of variables being complete.
    """
    try:
        from hermes_cli import kanban_db as kb
    except Exception as exc:  # pragma: no cover - import failure is fatal
        raise StressIsolationError(
            f"cannot import hermes_cli.kanban_db to verify isolation: {exc}"
        ) from exc

    resolved = kb.kanban_db_path()
    if Path(resolved).resolve() != db.resolve():
        raise StressIsolationError(
            "kanban_db_path() does not agree with the sandbox — something "
            "still out-votes the pin.\n"
            f"  pinned   = {db}\n"
            f"  resolved = {resolved}\n"
            "Refusing to run rather than risk a live board."
        )


def isolate_stress_home(prefix: str) -> StressHome:
    """Create a sandbox board and pin the whole process to it.

    Call this **before** importing ``hermes_cli.kanban_db``, as the first
    statement of a stress entry point. Creating the schema stays the
    script's job; this only guarantees where it will land.
    """
    home = Path(tempfile.mkdtemp(prefix=prefix)).resolve()
    (home / SANDBOX_MARKER).write_text(
        f"disposable kanban board for {prefix}* (pid {os.getpid()})\n",
        encoding="utf-8",
    )
    db = home / _DB_NAME

    assert_disposable_sandbox(home, db)
    env = _sandbox_env(home, db)
    _apply(env)
    _verify_resolver(db)
    return StressHome(home=home, db=db, env=env)


def adopt_stress_home(home: str | Path) -> StressHome:
    """Re-assert the sandbox inside a spawned worker.

    A child inherits its parent's environment, so in practice it starts
    clean — but "in practice" is what the incident already disproved once.
    The child re-derives the DB from the path it was handed and re-runs
    every check, so a worker pointed anywhere else dies instead of working.
    """
    home = Path(home).resolve()
    db = home / _DB_NAME

    assert_disposable_sandbox(home, db, allow_existing_db=True)
    env = _sandbox_env(home, db)
    _apply(env)
    _verify_resolver(db)
    return StressHome(home=home, db=db, env=env)
