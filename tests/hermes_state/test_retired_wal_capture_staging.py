"""An interrupted retired-WAL capture must not leave its staging directory behind.

``capture_retired_wal_generation`` writes the retired WAL, the -shm and a copy of the main image into
a staging directory next to the LIVE database, then publishes it with one ``os.replace``. Until that
rename the directory holds a partial copy of ``state.db`` -- up to
``RETIRED_GENERATION_MAIN_IMAGE_MAX_BYTES`` (512 MB) -- with nothing else on the profile able to
collect it: the sweeper in ``hermes_cli.backup_restore`` only accepts *hidden* staging names (a
visible ``*.partial`` is a user artifact, and the published
``state.db.retired-wal-<ts>-<pid>/`` is a legitimate operator-recovery artifact), and the disk-hygiene
cron never matched the name either.

Measured on a real SIGTERM'd child writing a real WAL capture: before the fix the run left
``state.db.retired-wal-20261001-072448-3624811.partial/`` in the profile home and exited -15. These
tests pin the two guarantees that close it -- the termination guard unwinds through the ``finally``,
and the next capture collects what no handler could reach (SIGKILL).
"""

import importlib
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

import pytest

import hermes_state
import hermes_state_wal
from hermes_state import DeletedWalGenerationError, SessionDB
from hermes_state_dbfile import (
    RETIRED_GENERATION_MANIFEST,
    _prune_stale_retired_generation_staging,
    _retired_generation_staging,
    _staging_owner_pid,
)
from tests.hermes_state._wal_generation_harness import lose_sidecars, make_db, require_wal

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")

# The repo root of the checkout under test: the editable install's MetaPathFinder maps top-level
# *packages* but not root-level single-file modules (measured: a child with no PYTHONPATH fails
# ``import hermes_state_dbfile`` with ``No module named 'hermes_yaml'``), and a child *script* gets
# sys.path[0] = its own directory, never the cwd. Without this the signal tests would exercise the
# installed tree instead of the code under test.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])


@pytest.fixture
def force_wal(monkeypatch):
    """Pin WAL so this host's vulnerable SQLite still matches production topology."""
    pin = __import__("tests.hermes_state._wal_generation_harness", fromlist=["pin_wal"])
    pin.pin_wal(monkeypatch)


# A writer that opens state.db in WAL mode, loses its -wal/-shm generation to a non-Hermes opener,
# and is then SIGTERM'd (or SIGKILL'd) while the capture is copying. The copy is slowed through a
# module attribute so the signal lands after the staging directory exists but before it is published;
# production code is untouched.
_CHILD = textwrap.dedent(
    """
    import os, sys, time
    from pathlib import Path

    repo, db_path, ready_path = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, repo)
    os.environ["HERMES_HOME"] = str(Path(db_path).parent / "home")
    os.environ["HERMES_STATE_DB_GUARD_BYPASS"] = "1"

    import hermes_state_wal
    hermes_state_wal.is_sqlite_wal_reset_vulnerable = lambda version_info=None: False
    hermes_state_wal.resolve_journal_mode = lambda: "wal"

    import hermes_state_dbfile as dbfile
    from hermes_state import DeletedWalGenerationError, SessionDB

    def announce(state):
        Path(ready_path).write_text(state, encoding="utf-8")

    path = Path(db_path)
    db = SessionDB(db_path=path)
    if not db._wal_active:
        announce("skip")
        sys.exit(3)
    for sid in ("gw-0", "gw-1"):
        db.create_session(sid, "cli")
        db.append_message(sid, role="user", content="seed")
    db._try_wal_checkpoint()                      # history into state.db, like `sessions optimize`
    db._conn.execute("PRAGMA wal_autocheckpoint=0")
    for sid in ("gw-0", "gw-1"):
        db.append_message(sid, role="assistant", content="uncheckpointed " + "x" * 3000)
    for suffix in ("-wal", "-shm"):               # the field incident: the generation is taken away
        side = Path(str(path) + suffix)
        if side.exists():
            side.unlink()

    _real_copy = dbfile._copy_descriptor
    def _slow(fd, dest, *, size):
        if dest.name.endswith("-wal"):
            # The staging directory exists by now; announce it so the parent signals inside the copy.
            announce("staging")
            time.sleep(30)
        return _real_copy(fd, dest, size=size)
    dbfile._copy_descriptor = _slow

    try:
        db.append_message("gw-0", role="user", content="post-loss turn")
        announce("no-halt")
    except DeletedWalGenerationError:
        announce("halted")
    """
)


def _child_env() -> dict:
    """Environment for a child that must import the checkout the parent is running."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{_REPO_ROOT}{os.pathsep}{existing}" if existing else _REPO_ROOT
    return env


def _staging_candidates(db_path: Path) -> list[Path]:
    """Every retired-WAL staging directory beside *db_path* (ours, any owner)."""
    return sorted(p for p in db_path.parent.iterdir() if _staging_owner_pid(p.name) is not None)


def _published_captures(db_path: Path) -> list[Path]:
    """Published artifacts: the visible ``state.db.retired-wal-<ts>-<pid>/`` directories."""
    return sorted(p for p in db_path.parent.iterdir()
                  if p.is_dir() and ".retired-wal-" in p.name
                  and not p.name.startswith(".") and not p.name.endswith(".partial"))


def _start_capture(tmp_path: Path) -> tuple[subprocess.Popen, Path, Path]:
    """Spawn the writer child and return once its staging directory exists.

    Returns ``(proc, db_path, stderr_path)``. Skips the test when WAL is not active on this
    filesystem (same reason as ``require_wal`` in the shared harness).
    """
    db_path = tmp_path / "state.db"
    (tmp_path / "home").mkdir()
    child = tmp_path / "capture_child.py"
    child.write_text(_CHILD, encoding="utf-8")
    ready = tmp_path / "ready"
    stderr_path = tmp_path / "child-stderr.log"
    with stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(
            [sys.executable, str(child), _REPO_ROOT, str(db_path), str(ready)],
            cwd=_REPO_ROOT, env=_child_env(),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr,
        )
    deadline = time.monotonic() + 90
    state = ""
    while time.monotonic() < deadline and proc.poll() is None:
        if ready.exists():
            state = ready.read_text(encoding="utf-8")
            if state == "staging":
                break
        time.sleep(0.02)
    else:
        proc.kill()
        proc.wait(timeout=30)
        if state == "skip":
            pytest.skip("WAL not active on this filesystem")
        pytest.fail("the child never reached the staging copy\n" + stderr_path.read_text(encoding="utf-8"))
    assert _staging_candidates(db_path), (
        "the child announced the staging copy but no staging directory exists")
    return proc, db_path, stderr_path


def _reap(proc: subprocess.Popen) -> int:
    """Wait for *proc* and return its status; SIGKILL only if it ignores the signal for a minute.

    Not a poll()-then-kill: right after ``send_signal`` the child is still alive, so that order
    would replace the real exit status (-15 for an unhandled SIGTERM) with our own -9 and hide
    exactly the signal this test is about.
    """
    try:
        return proc.wait(timeout=60)
    except subprocess.TimeoutExpired:  # pragma: no cover - safety net
        proc.kill()
        return proc.wait(timeout=30)


# ── The staging name itself ─────────────────────────────────────────────────────────────────────


# The gate the backup sweeper applies, spelled out here instead of imported: PR 112365 (which
# carries hermes_cli.termination_guard and the sweeper) is still open, so on a plain main checkout
# neither symbol exists yet and the module under test would be untestable. These two lines are the
# sweeper's documented contract -- hidden names only, owner read from the LAST dotted component --
# and ``test_staging_name_also_satisfies_the_backup_sweep_when_it_is_present`` pins the agreement
# whenever the sweeper is in the tree.
def _backup_sweep_accepts(name: str) -> bool:
    return name.startswith(".") and name.endswith(".partial")


def _backup_sweep_owner(name: str) -> Optional[str]:
    match = re.search(r"\.(\d+)(?:-\d+)?\.partial$", name)
    return match.group(1) if match else None


def test_staging_name_is_hidden_pid_attributed_and_distinct_from_the_artifact() -> None:
    """The staging name must carry the properties the backup sweeper requires, without changing
    the published artifact's spelling.

    A visible ``*.partial`` is a user artifact and never removed by that sweeper (it also runs in
    the user's home), and its owner regex reads the LAST dotted component -- so the pid has to sit
    at the tail, or the timestamp in the middle gets probed as a pid and the residue is kept
    forever. The published artifact keeps the exact spelling backup.py's exclusion prefix, the
    operator-facing recovery flow and the capture tests key on.
    """
    final = Path("/tmp/state.db.retired-wal-20261001-072448-3624811")
    staging = _retired_generation_staging(final)

    assert _backup_sweep_accepts(staging.name), "a visible staging is never collected by any sweep"
    assert _backup_sweep_owner(staging.name) == str(os.getpid())
    assert _staging_owner_pid(staging.name) == os.getpid()
    assert _retired_generation_staging(final) != final, "the staging must be distinct from the artifact"
    # The published artifact is untouched: visible, no .partial, and not a sweep candidate.
    assert final.name == "state.db.retired-wal-20261001-072448-3624811"
    assert not _backup_sweep_accepts(final.name) and _staging_owner_pid(final.name) is None


def test_staging_name_also_satisfies_the_backup_sweep_when_it_is_present() -> None:
    """Same property, checked against the real sweeper whenever it is in the tree (PR 112365).

    Skipped on a checkout without the sweep, which is every ``main`` while that PR is open: the
    previous test is the one that always runs, and it encodes the same contract directly.
    """
    for module_name in ("hermes_cli.backup", "hermes_cli.backup_restore"):
        module = importlib.import_module(module_name)
        staging_kind = getattr(module, "_staging_kind", None)
        partial_pid_re = getattr(module, "_PARTIAL_PID_RE", None)
        if staging_kind is None or partial_pid_re is None:
            continue
        final = Path("/tmp/state.db.retired-wal-20261001-072448-3624811")
        staging = _retired_generation_staging(final)
        assert staging_kind(staging.name) == "partial", (
            f"{module_name}._staging_kind would never collect this name")
        assert partial_pid_re.search(staging.name).group(1) == str(os.getpid())
        assert staging_kind(final.name) is None, f"{module_name} would treat the published artifact as staging"
        return
    pytest.skip("the backup staging sweep is not in this tree yet (PR 112365 is open)")


def test_staging_owner_pid_rejects_everything_that_is_not_this_producer() -> None:
    """The sweep runs in the profile home, so only this module's exact shape is a candidate."""
    ts, pid = "20261001-072448", os.getpid()
    ours = f".state.db.retired-wal-{ts}-3624811.{pid}.partial"
    rejected = (
        "notes.partial",                                        # a user file
        "archive.partial",                                      # a user directory
        "20260915-203000-foo.partial",                          # a published snapshot whose label ends in .partial
        f"state.db.retired-wal-{ts}-3624811",                   # the PUBLISHED capture (no .partial)
        ".hermes-backup-2026.zip.1234-7.partial",               # the backup producer's staging
        ".state.db.abcdef.dbimport",                            # the import producer's staging
        ".state.db.snap_restore",                               # the restore fallback's copy
        f".state.db.retired-wal-{ts}-3624811",                  # ours, but pid-less (age-gate only)
        f"state.db.retired-wal-{ts}-3624811.partial",           # the OLD visible spelling
        f".state.db.retired-wal-{ts}-3624811.{pid}",            # pid in the wrong place
    )
    for name in rejected:
        assert _staging_owner_pid(name) is None, name
    assert _staging_owner_pid(ours) == pid


def test_published_capture_is_never_a_sweep_candidate(tmp_path, force_wal) -> None:
    """A real capture's artifact survives the sweep that runs on the same directory."""
    path = tmp_path / "state.db"
    db = make_db(path, "gw-0", "seed")
    try:
        require_wal(db)
        lose_sidecars(path, rename=False)
        with pytest.raises(DeletedWalGenerationError):
            db.append_message("gw-0", role="user", content="after the loss")
        published = _published_captures(path)
        assert published, "the capture published nothing"
        assert _prune_stale_retired_generation_staging(path) == 0
        assert published[0].is_dir(), "the sweep removed a published capture"
    finally:
        db.close()


def test_the_sweep_keeps_a_live_peer_and_a_user_partial(tmp_path, force_wal) -> None:
    """Only a dead owner's staging goes: a live pid's, and a visible user ``*.partial``, survive."""
    path = tmp_path / "state.db"
    db = make_db(path, "gw-0", "seed")
    try:
        require_wal(db)
        lose_sidecars(path, rename=False)
        with pytest.raises(DeletedWalGenerationError):
            db.append_message("gw-0", role="user", content="after the loss")
        final = _published_captures(path)[0]
        live = _retired_generation_staging(final)  # this process's pid: a live peer
        live.mkdir()
        (live / "in-flight").write_bytes(b"x" * 32)
        user = tmp_path / "notes.partial"
        user.write_bytes(b"user data")
        stranger = tmp_path / ".attic.partial"  # hidden, pid-less, a stranger's directory
        stranger.mkdir()
        (stranger / "keepme.txt").write_bytes(b"user dir")

        assert _prune_stale_retired_generation_staging(path) == 0
        assert (live / "in-flight").exists(), "a live run's staging was collected"
        assert user.exists() and (stranger / "keepme.txt").exists()
    finally:
        db.close()


# ── SIGTERM: the guard unwinds through the finally ───────────────────────────────────────────────


def test_sigterm_during_the_capture_leaves_no_staging_directory(tmp_path) -> None:
    """A real writer, SIGTERM'd mid-copy, leaves neither a staging directory nor a published artifact."""
    proc, db_path, stderr_path = _start_capture(tmp_path)
    proc.send_signal(signal.SIGTERM)
    rc = _reap(proc)

    assert rc == 128 + signal.SIGTERM, (
        f"the child died of the default disposition (rc={rc}, -SIGTERM={-signal.SIGTERM}): the "
        f"termination guard never turned the signal into an unwinding exception\n"
        + stderr_path.read_text(encoding="utf-8"))
    assert not _staging_candidates(db_path), "SIGTERM stranded the staging directory"
    assert not _published_captures(db_path), "an interrupted capture published an artifact"
    # No half-written per-file copy of the main image either: _copy_range stages through .part.
    assert not list(db_path.parent.glob("*.part")) and not list(db_path.parent.glob(".*.part"))


# ── SIGKILL: the next capture collects what no handler could ─────────────────────────────────────


def test_sigkill_residue_is_collected_by_the_next_run(tmp_path) -> None:
    """SIGKILL is the one death no handler can intercept: only the next run reclaims the residue.

    The guarantee under test is the CAP, not the cleanup: the stranded directory is real (measured on
    a real SIGKILLed child, not simulated) and the next capture removes it, while a live peer's
    staging directory and a published artifact survive untouched.
    """
    proc, db_path, stderr_path = _start_capture(tmp_path)
    proc.kill()  # SIGKILL: no handler, no finally, no ExitStack
    assert _reap(proc) == -signal.SIGKILL

    stranded = _staging_candidates(db_path)
    assert stranded, (
        "SIGKILL unexpectedly cleaned up after itself — the test would prove nothing\n"
        + stderr_path.read_text(encoding="utf-8"))
    assert not _published_captures(db_path)

    # The next capture is the collector. It is a REAL capture in THIS process: it sweeps the residue
    # first, then publishes its own artifact normally.
    published = tmp_path / "state.db.retired-wal-20260101-000000-1"
    published.mkdir()
    (published / RETIRED_GENERATION_MANIFEST).write_text("{}", encoding="utf-8")
    user = tmp_path / "notes.partial"
    user.write_bytes(b"user data")

    db = make_db(db_path, "gw-1", "seed")
    try:
        require_wal(db)
        lose_sidecars(db_path, rename=False)
        with pytest.raises(DeletedWalGenerationError):
            db.append_message("gw-1", role="user", content="after the loss")
        new_artifact = db._retired_generation_capture
    finally:
        db.close()

    assert new_artifact is not None and new_artifact.is_dir(), "the next capture published nothing"
    assert not stranded[0].exists(), "the sweep left the SIGKILL residue behind"
    assert not _staging_candidates(db_path), "the next run stranded its own staging"
    assert published.is_dir(), "a published capture was collected"
    assert user.exists(), "a user *.partial in the home was collected"
