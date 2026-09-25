"""Harness for the multi-process SQLite torture chamber (issue class C1: state.db integrity).

One real ``state.db`` under ``tmp_path``, in either journal mode Hermes deploys (see ``JOURNAL_MODES``); every
role is a separate OS process running ``_roles.py`` against it (see that module for the journal/report
protocol). This module owns:

* journal-mode selection through the PRODUCTION decision path: the ``delete`` arm pins the version the
  children's ``hermes_state_wal.is_sqlite_wal_reset_vulnerable()`` probe sees to a WAL-reset-vulnerable
  SQLite (the one CI's uv CPython 3.11.14 bundles), so ``apply_wal_with_fallback`` itself picks DELETE
  exactly as it does for every user on such a build — the same DELETE store its network/FUSE fallbacks end
  in; nothing in the harness issues a journal-mode pragma;

* process lifecycle (spawn / ready / stop / SIGTERM / SIGKILL, PIDs recorded, nothing else touched);
* a background ``/proc/<pid>/fd`` monitor over OUR children that records any ``(deleted)`` main-file or
  WAL descriptor, and SHM descriptors that remain deleted across a short confirmation window. SQLite's
  last-connection SHM cleanup unlinks before closing its own descriptor;
* the invariant checks, done in the test process with short-lived bare ``sqlite3`` connections only (the
  test process never imports ``hermes_state``, so it never becomes a foreign holder of the file).

Reuses the conformance harness's deadline polling / SIGKILL reaping (``tests/conformance/persistence``).
"""

from __future__ import annotations

import json
import os
import random
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path

from hermes_cli.sqlite_runtime import is_sqlite_wal_reset_vulnerable
from tests.conformance.persistence._harness import REPO_ROOT, kill9_and_reap, wait_for

ROLES = Path(__file__).with_name("_roles.py")
DEFAULT_SEED = 20260923
SEED_ENV = "HERMES_SQLITE_TORTURE_SEED"

JOURNAL_MODES = ("wal", "delete")
# Read ONLY by _roles.py in each child: the SQLite version its production version probe reports.
SQLITE_PIN_ENV = "HERMES_E2E_SQLITE_VERSION_PIN"
VULNERABLE_SQLITE = "3.50.4"  # bundled by uv's CPython 3.11.14 (the unit CI job): Hermes runs DELETE there
# DELETE mode holds an EXCLUSIVE lock for every commit's journal+db fsyncs and has no writer fairness: an unpaced
# append loop starves every other writer and reader. Gateway/TUI writers are paced by turns in the field.
DELETE_WRITER_PACE = 0.02
# SQLite unixShmUnmap unlinks before unixShmPurge closes hShm. Confirm persistence,
# not that legitimate instantaneous close window; main DB and WAL remain immediate.
# This is a sampling policy, not an upper bound on how long SQLite close can take.
SHM_CLOSE_GRACE_SECONDS = 0.1


def linked_sqlite_is_wal_capable() -> bool:
    """The production predicate on the SQLite this interpreter (and so every child) links."""
    return not is_sqlite_wal_reset_vulnerable(sqlite3.sqlite_version_info)


def skip_unless_deployable(journal: str) -> None:
    """WAL is not what Hermes runs on a vulnerable SQLite, so that arm is not deployable there; DELETE always is."""
    if journal == "wal" and not linked_sqlite_is_wal_capable():
        import pytest

        pytest.skip(f"linked SQLite {sqlite3.sqlite_version} runs Hermes in DELETE mode; the delete arm covers it")


def base_seed() -> int:
    return int(os.environ.get(SEED_ENV) or DEFAULT_SEED)


def episode_seed(label: str) -> int:
    return base_seed() + zlib.crc32(label.encode())


def child_env(home: Path, hermes_home: Path) -> dict:
    """Probe hygiene: private HOME/HERMES_HOME, no provider credentials, repo importable."""
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env.update({
        "HOME": str(home),
        "HERMES_HOME": str(hermes_home),
        "PYTHONUNBUFFERED": "1",
        # The child's HERMES_HOME *is* this chamber's private home, which the live-DB guard reads as
        # "the real Hermes root"; HOME/HERMES_HOME above already keep it off the production install.
        "HERMES_STATE_DB_GUARD_BYPASS": "1",
        "PYTHONPATH": os.pathsep.join(p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if p),
    })
    return env


class Chamber:
    """A private HERMES_HOME + state.db plus the processes playing roles against it."""

    def __init__(self, root: Path, *, journal: str = "wal"):
        assert journal in JOURNAL_MODES, journal
        self.root = root
        self.mode = journal
        self.home = root / "home"
        self.hermes_home = self.home / ".hermes"
        self.hermes_home.mkdir(parents=True, exist_ok=True)
        # The config default in BOTH arms: the delete arm is a default-config user on a vulnerable SQLite.
        (self.hermes_home / "config.yaml").write_text("database:\n  journal_mode: wal\n", encoding="utf-8")
        self.db = self.hermes_home / "state.db"
        self.work = root / "work"
        self.work.mkdir(exist_ok=True)
        self.env = child_env(self.home, self.hermes_home)
        if journal == "delete" and linked_sqlite_is_wal_capable():
            self.env[SQLITE_PIN_ENV] = VULNERABLE_SQLITE
        self.writer_pace = DELETE_WRITER_PACE if journal == "delete" else 0.0
        self.procs: dict[str, subprocess.Popen] = {}
        self.writer_runs: list[str] = []
        self.reader_seq = 0
        self.reader_name: str | None = None
        self.deleted_hits: list[tuple[str, int, str]] = []
        self.fd_samples: dict[str, list[int]] = {}
        self._lock = threading.Lock()
        self._monitor_stop = threading.Event()
        self._monitor = threading.Thread(target=self._scan_loop, name="deleted-fd-monitor", daemon=True)
        self._monitor.start()

    # -- lifecycle ---------------------------------------------------------------------------------
    def spawn(self, role: str, name: str, *, env: dict | None = None, **args) -> subprocess.Popen:
        assert name not in self.procs, f"duplicate role name {name}"
        if role == "writer":
            args.setdefault("pace", self.writer_pace)
        payload = {"workdir": str(self.work), "name": name, "db": str(self.db),
                   "stop": str(self.work / f"{name}.stop"), "busy_ok": self.mode == "delete", **args}
        stderr = open(self.work / f"{name}.stderr", "wb")  # noqa: SIM115 - closed in reap()
        proc = subprocess.Popen(
            [sys.executable, str(ROLES), role, json.dumps(payload)],
            cwd=str(REPO_ROOT), env={**self.env, **(env or {})}, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=stderr,
        )
        proc._stderr_file = stderr  # type: ignore[attr-defined]
        with self._lock:
            self.procs[name] = proc
        if role == "writer":
            self.writer_runs.append(name)
        return proc

    def spawn_cli(self, name: str, *argv: str) -> subprocess.Popen:
        """A real `hermes …` CLI subprocess against this HERMES_HOME (``hermes_cli.main`` run as ``__main__``
        by ``_roles.py cli`` so the journal-mode seam applies to it too)."""
        stderr = open(self.work / f"{name}.stderr", "wb")  # noqa: SIM115 - closed in reap()
        payload = {"workdir": str(self.work), "name": name, "argv": list(argv)}
        proc = subprocess.Popen(
            [sys.executable, str(ROLES), "cli", json.dumps(payload)], cwd=str(REPO_ROOT), env=self.env,
            stdin=subprocess.DEVNULL, stdout=open(self.work / f"{name}.stdout", "wb"), stderr=stderr,  # noqa: SIM115
        )
        proc._stderr_file = stderr  # type: ignore[attr-defined]
        with self._lock:
            self.procs[name] = proc
        return proc

    def stderr(self, name: str) -> str:
        path = self.work / f"{name}.stderr"
        return path.read_text(encoding="utf-8", errors="replace")[-3000:] if path.exists() else ""

    def wait_event(self, name: str, event: str, *, deadline: float = 60.0) -> dict:
        found: list[dict] = []

        def _has() -> bool:
            found[:] = [e for e in self.events(name) if e.get("event") in (event, "error")]
            return bool(found)

        self._wait(name, _has, f"{name}:{event}", deadline)
        assert found[0].get("event") == event, f"{name} reported {found[0]}\n{self.stderr(name)}"
        return found[0]

    def wait_acks(self, name: str, n: int, *, deadline: float = 60.0) -> None:
        self._wait(name, lambda: len(self.journal(name)[1]) >= n, f"{n} acks from {name}", deadline)

    def _wait(self, name: str, predicate, what: str, deadline: float) -> None:
        try:
            wait_for(predicate, deadline=deadline, what=what, child=self.procs.get(name))
        except AssertionError as exc:
            tail = [e for e in self.events(name) if e.get("event") == "error"][-1:]
            raise AssertionError(f"{exc}\n{name} stderr:\n{self.stderr(name)}\n{tail}") from None

    def request_stop(self, name: str) -> None:
        (self.work / f"{name}.stop").touch()

    def reap(self, name: str, *, deadline: float = 60.0) -> int:
        proc = self.procs[name]
        try:
            rc = proc.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            kill9_and_reap(proc)
            raise AssertionError(f"{name} did not exit within {deadline}s\n{self.stderr(name)}") from None
        finally:
            proc._stderr_file.close()  # type: ignore[attr-defined]
        return rc

    def stop(self, name: str, *, deadline: float = 60.0) -> int:
        self.request_stop(name)
        return self.reap(name, deadline=deadline)

    def sigterm(self, name: str) -> None:
        self.procs[name].send_signal(signal.SIGTERM)

    def kill9(self, name: str) -> None:
        proc = self.procs[name]
        kill9_and_reap(proc)
        proc._stderr_file.close()  # type: ignore[attr-defined]

    def live(self) -> list[tuple[str, subprocess.Popen]]:
        with self._lock:
            return [(n, p) for n, p in self.procs.items() if p.poll() is None]

    def shutdown(self) -> None:
        for name, proc in self.live():
            self.request_stop(name)
        end = time.monotonic() + 20
        for _name, proc in list(self.procs.items()):
            try:
                proc.wait(timeout=max(0.1, end - time.monotonic()))
            except subprocess.TimeoutExpired:
                kill9_and_reap(proc)
            f = getattr(proc, "_stderr_file", None)
            if f is not None and not f.closed:
                f.close()
        self._monitor_stop.set()
        self._monitor.join(timeout=5)

    # -- kernel truth: (deleted) sidecars held by our children --------------------------------------
    def _scan_loop(self) -> None:
        pending: dict[tuple, float] = {}
        while not self._monitor_stop.is_set():
            self._scan_deleted_fds(pending)
            self._monitor_stop.wait(0.02)

    def _scan_deleted_fds(self, pending: dict[tuple, float]) -> None:
        targets = {str(self.db)}
        if self.mode == "wal":
            targets |= {f"{self.db}-wal", f"{self.db}-shm"}
        still_pending: dict[tuple, float] = {}
        for name, proc in self.live():
            fd_dir = f"/proc/{proc.pid}/fd"
            try:
                fds = os.listdir(fd_dir)
            except OSError:
                continue
            for fd in fds:
                fd_path = f"{fd_dir}/{fd}"
                try:
                    link = os.readlink(fd_path)
                except OSError:
                    continue
                if not link.endswith(" (deleted)") or link[: -len(" (deleted)")] not in targets:
                    continue
                if link == f"{self.db}-shm (deleted)":
                    try:
                        identity = os.stat(fd_path)
                        if os.readlink(fd_path) != link:
                            continue
                    except OSError:
                        continue
                    if proc.poll() is not None:
                        continue
                    key = (proc, proc.pid, fd, identity.st_dev, identity.st_ino, link)
                    observed_at = time.monotonic()
                    first_seen = pending.get(key, observed_at)
                    still_pending[key] = first_seen
                    if observed_at - first_seen < SHM_CLOSE_GRACE_SECONDS:
                        continue
                elif proc.poll() is not None:
                    continue
                with self._lock:
                    self.deleted_hits.append((name, proc.pid, link))
        # A closed/reused descriptor or exited child must start a new observation.
        pending.clear()
        pending.update(still_pending)

    def deleted_hits_snapshot(self) -> list[tuple[str, int, str]]:
        with self._lock:
            return sorted(set(self.deleted_hits))

    # -- reports -------------------------------------------------------------------------------------
    def events(self, name: str) -> list[dict]:
        path = self.work / f"{name}.report"
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # torn final line of a SIGKILLed child
        return out

    def errors(self) -> list[tuple[str, dict]]:
        return [(name, e) for name in list(self.procs) for e in self.events(name) if e.get("event") == "error"]

    def journal(self, name: str) -> tuple[dict[str, str], set[str]]:
        """``(intents: tok -> sid, acked tokens)`` for one writer run."""
        path = self.work / f"{name}.journal"
        intents: dict[str, str] = {}
        acked: set[str] = set()
        if not path.exists():
            return intents, acked
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "I":
                intents[parts[1]] = parts[2]
            elif len(parts) == 2 and parts[0] == "A":
                acked.add(parts[1])
        return intents, acked


# -- invariants (bare sqlite3, short-lived connections) --------------------------------------------------


def _connect(db: Path, *, ro: bool = True) -> sqlite3.Connection:
    uri = f"file:{db}?mode=ro" if ro else f"file:{db}"
    return sqlite3.connect(uri, uri=True, timeout=30.0)


def integrity_rows(db: Path) -> list[str]:
    conn = _connect(db)
    try:
        return [r[0] for r in conn.execute("PRAGMA integrity_check").fetchall()]
    except sqlite3.DatabaseError as exc:
        return [f"integrity_check raised: {exc!r}"]
    finally:
        conn.close()


def journal_mode(db: Path) -> str:
    conn = _connect(db)
    try:
        return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        conn.close()


def counts(db: Path) -> dict[str, int]:
    """Per-session canonical row counts (plus ``__total__``)."""
    conn = _connect(db)
    try:
        rows = conn.execute("SELECT session_id, count(*) FROM messages GROUP BY session_id").fetchall()
    finally:
        conn.close()
    out = {sid: n for sid, n in rows}
    out["__total__"] = sum(out.values())
    return out


def token_counts(db: Path) -> dict[str, int]:
    """How many canonical rows carry each writer token (first word of the content)."""
    conn = _connect(db)
    try:
        rows = conn.execute("SELECT content FROM messages WHERE content LIKE 'TK%'").fetchall()
    finally:
        conn.close()
    out: dict[str, int] = {}
    for (content,) in rows:
        tok = content.split(" ", 1)[0]
        out[tok] = out.get(tok, 0) + 1
    return out


def fts_problems(db: Path, sample_tokens: list[str]) -> list[str]:
    """FTS must mirror canonical rows exactly: docsize == source rows for each external-content index,
    FTS5's own 'integrity-check' against the content, and a sample of acked tokens found exactly once."""
    problems: list[str] = []
    conn = _connect(db, ro=False)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master").fetchall()}
        meta = dict(conn.execute("SELECT key, value FROM state_meta").fetchall()) if "state_meta" in tables else {}
        pending = {k: v for k, v in meta.items() if k.startswith("fts_rebuild") or k in ("fts_stale",)}
        if pending:
            problems.append(f"FTS recovery still pending in state_meta: {pending}")
        canon = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
        pairs = [("messages_fts", "messages_fts_src"), ("messages_fts_trigram", "messages_fts_trigram_src")]
        for fts, src in pairs:
            if fts not in tables:
                problems.append(f"{fts} missing")
                continue
            indexed = conn.execute(f"SELECT count(*) FROM {fts}_docsize").fetchone()[0]
            expected = conn.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
            if indexed != expected:
                problems.append(f"{fts}: {indexed} indexed rows vs {expected} source rows (canonical {canon})")
            try:
                conn.execute(f"INSERT INTO {fts}({fts}, rank) VALUES('integrity-check', 1)")
            except sqlite3.DatabaseError as exc:
                problems.append(f"{fts} integrity-check: {exc}")
        for tok in sample_tokens:
            hits = conn.execute("SELECT count(*) FROM messages_fts WHERE messages_fts MATCH ?",
                                (f'"{tok}"',)).fetchone()[0]
            if hits != 1:
                problems.append(f"session search finds acked {tok} {hits}x (expected 1)")
        conn.rollback()
    finally:
        conn.close()
    return problems


def journal_mode_problems(chamber: Chamber, prefix: str = "") -> list[str]:
    """The store is still in the arm's journal mode, and every production SessionDB role (whose ``ready``
    event reports ``SessionDB._wal_active``) really ran in it — the seam took, the arm is not vacuous."""
    problems = []
    mode = journal_mode(chamber.db)
    if mode != chamber.mode:
        problems.append(f"journal_mode is {mode!r} after the episode (arm is {chamber.mode})")
    want = chamber.mode == "wal"
    for name in list(chamber.procs):
        if not name.startswith(prefix):
            continue
        ready = [e for e in chamber.events(name) if e.get("event") == "ready" and "wal" in e]
        if ready and ready[0]["wal"] != want:
            problems.append(f"{name} opened SessionDB with _wal_active={ready[0]['wal']} in the {chamber.mode} arm")
    return problems


def busy_summary(chamber: Chamber, prefix: str = "") -> dict[str, int]:
    """DELETE arm only: SQLITE_BUSY refusals the roles waited out, per operation (availability, not integrity)."""
    out: dict[str, int] = {}
    for name in list(chamber.procs):
        if name.startswith(prefix):
            for e in chamber.events(name):
                if e.get("event") == "busy":
                    out[e["op"]] = out.get(e["op"], 0) + 1
    return out


def exactly_once_problems(chamber: Chamber) -> list[str]:
    """Every acknowledged append is stored exactly once; an unacknowledged in-flight append at most once;
    no stored writer row that no writer ever intended."""
    stored = token_counts(chamber.db)
    problems: list[str] = []
    intended: set[str] = set()
    for run in chamber.writer_runs:
        intents, acked = chamber.journal(run)
        intended.update(intents)
        for tok in acked:
            if stored.get(tok, 0) != 1:
                problems.append(f"{run}: acked {tok} stored {stored.get(tok, 0)}x")
        for tok in set(intents) - acked:
            if stored.get(tok, 0) > 1:
                problems.append(f"{run}: in-flight {tok} stored {stored[tok]}x")
    phantom = sorted(set(stored) - intended)
    if phantom:
        problems.append(f"{len(phantom)} stored rows no writer intended, e.g. {phantom[:3]}")
    return problems


def compress_journal(chamber: Chamber, run: str) -> list[list[str]]:
    """``C`` lines of an agent run: the turn bases each ``/compress here N`` promised to keep verbatim."""
    path = chamber.work / f"{run}.journal"
    if not path.exists():
        return []
    return [line.split()[1:] for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("C ")]


def token_flags(db: Path, token: str) -> list[tuple[int, int]]:
    """``(active, compacted)`` of every canonical row whose content carries ``token``."""
    conn = _connect(db)
    try:
        rows = conn.execute("SELECT active, compacted FROM messages WHERE content LIKE ?",
                            (f"%{token}%",)).fetchall()
    finally:
        conn.close()
    return sorted((int(a), int(c)) for a, c in rows)


def acked_tokens(chamber: Chamber, runs: list[str]) -> list[str]:
    out: list[str] = []
    for run in runs:
        out.extend(sorted(chamber.journal(run)[1]))
    return out


def sample(rng: random.Random, items: list[str], k: int) -> list[str]:
    return rng.sample(items, min(k, len(items)))
