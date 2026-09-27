"""Child-process roles for the SQLite torture chamber.

Run as ``python _roles.py <role> <json-args>``. Every role is a real OS process that opens the shared
``state.db`` through the production ``SessionDB`` (or, for the non-hermes opener, a bare ``sqlite3``
connection) and reports through append-only files, so a ``kill -9`` loses nothing it already reported:

* ``<name>.journal`` — writers: ``I <tok> <sid> <nrows>`` before an append, ``A <tok>`` once it returned.
* ``<name>.report`` — JSON lines: ``ready`` / ``stats`` / ``error`` / ``closed`` events.

Tokens are single FTS words (``TK`` + alnum) so the test can look every acknowledged append up by content,
through ``messages`` and through the FTS indexes.

Journal-mode seam: with ``HERMES_E2E_SQLITE_VERSION_PIN`` set, the production version probe
``hermes_state_wal.is_sqlite_wal_reset_vulnerable()`` reports that SQLite version instead of the linked one;
the real range predicate and ``apply_wal_with_fallback`` then decide the journal mode as they would there.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

TOOL_FILLER = "y" * 3000  # longer than the FTS tool-content prefix, so the projection is exercised
_stop_requested = False


def _on_sigterm(_signum, _frame):
    # Gateway/TUI shape: SIGTERM is a graceful shutdown that ends in SessionDB.close() (checkpoint on close).
    global _stop_requested
    _stop_requested = True


class Out:
    """O_APPEND line writers: every line reaches the page cache before the next action, so it survives
    a SIGKILL of this process (only a kernel crash could lose it)."""

    def __init__(self, workdir: Path, name: str):
        self.journal_fd = os.open(workdir / f"{name}.journal", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        self.report_fd = os.open(workdir / f"{name}.report", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)

    def journal(self, line: str) -> None:
        os.write(self.journal_fd, (line + "\n").encode())

    def report(self, **event) -> None:
        event.setdefault("pid", os.getpid())
        event.setdefault("t", time.time())
        os.write(self.report_fd, (json.dumps(event) + "\n").encode())


def _apply_sqlite_version_pin() -> None:
    pin = os.environ.get("HERMES_E2E_SQLITE_VERSION_PIN")
    if not pin:
        return
    import hermes_state_wal

    pinned = tuple(int(p) for p in pin.split("."))
    probe = hermes_state_wal.is_sqlite_wal_reset_vulnerable

    def is_sqlite_wal_reset_vulnerable(version_info=None):
        return probe(pinned if version_info is None else version_info)

    hermes_state_wal.is_sqlite_wal_reset_vulnerable = is_sqlite_wal_reset_vulnerable


def _patient(a: dict, out: Out, op: str, fn, *, deadline: float = 90.0):
    """Run ``fn()``; in the DELETE arm (``busy_ok``) a SQLITE_BUSY refusal is reported as a ``busy`` event and
    retried. DELETE mode is documented to block readers on writes (hermes_state_wal), so a busy read/open is an
    availability event there, never an integrity one. In the WAL arm it propagates and fails the role."""
    end = time.monotonic() + deadline
    while True:
        try:
            return fn()
        except sqlite3.OperationalError as exc:
            # By result code, not text: SQLITE_BUSY also surfaces as "vtable constructor failed:
            # messages_fts" when the FTS5 table's config read hits the lock during an open.
            busy = getattr(exc, "sqlite_errorcode", None) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) or any(
                m in str(exc).lower() for m in ("database is locked", "database is busy"))
            if not (busy and a.get("busy_ok")) or time.monotonic() > end:
                raise
            out.report(event="busy", op=op, error=repr(exc))
            time.sleep(0.05)


def _fd_count() -> int:
    return len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else -1


def _stray_close(db_path: Path) -> None:
    """What a plugin, a tool read of ~/.hermes or a header probe does: open + close the live files in THIS
    process. POSIX drops every lock this process holds on the inode (sqlite.org/howtocorrupt.html §2.2)."""
    for suffix in ("", "-shm", "-wal"):
        try:
            os.close(os.open(str(db_path) + suffix, os.O_RDONLY))
        except OSError:
            pass


def _stopping(stop_file: Path) -> bool:
    return _stop_requested or stop_file.exists()


def role_writer(a: dict, out: Out) -> int:
    """Long-lived writer (gateway- or TUI-like): appends until told to stop, journaling intent + ack."""
    from hermes_state import SessionDB

    db_path, stop_file = Path(a["db"]), Path(a["stop"])
    tag = a["tag"]
    db = SessionDB(db_path=db_path)
    for sid in a["sessions"]:
        if db.get_session(sid) is None:
            db.create_session(sid, a.get("source", "cli"))
    out.report(event="ready", wal=bool(getattr(db, "_wal_active", False)), fds=_fd_count())
    i = 0
    max_appends = int(a.get("max_appends", 10**9))
    try:
        while not _stopping(stop_file) and i < max_appends:
            sid = a["sessions"][i % len(a["sessions"])]
            tok = f"TK{tag}x{os.getpid()}x{i}"
            if a.get("batch_every") and i % a["batch_every"] == 0:
                msgs = [
                    {"role": "user", "content": f"{tok}u please run it"},
                    {"role": "tool", "content": f"{tok}t {TOOL_FILLER}", "tool_name": "terminal",
                     "tool_call_id": f"c{i}"},
                ]
                out.journal(f"I {tok}u {sid} 1")
                out.journal(f"I {tok}t {sid} 1")
                db.append_messages_batch(sid, msgs)
                out.journal(f"A {tok}u")
                out.journal(f"A {tok}t")
            else:
                role = ("user", "assistant")[i % 2]
                out.journal(f"I {tok} {sid} 1")
                db.append_message(sid, role=role, content=f"{tok} turn {i} of {tag}")
                out.journal(f"A {tok}")
            i += 1
            if a.get("stray_every") and i % a["stray_every"] == 0:
                _stray_close(db_path)
            if a.get("pace"):
                time.sleep(a["pace"])
    except BaseException as exc:  # a refused/failed append is exactly what the suite exists to catch
        out.report(event="error", error=repr(exc), tb=traceback.format_exc()[-3000:], appends=i)
        return 2
    out.report(event="stats", appends=i, fds=_fd_count())
    try:
        db.close()
    except BaseException as exc:
        out.report(event="error", error=f"close: {exc!r}", tb=traceback.format_exc()[-3000:])
        return 2
    out.report(event="closed", appends=i)
    return 0


def role_reader(a: dict, out: Out) -> int:
    """Dashboard-like reader: opens a writable SessionDB at startup and polls until stopped. Reports any
    per-session count that went DOWN (no compaction runs in this chamber) and its fd count per pass."""
    from hermes_state import SessionDB

    db_path, stop_file = Path(a["db"]), Path(a["stop"])
    db = _patient(a, out, "open", lambda: SessionDB(db_path=db_path))
    out.report(event="ready", wal=bool(getattr(db, "_wal_active", False)), fds=_fd_count())
    seen: dict[str, int] = {}
    passes = 0

    def _pass() -> None:
        for row in db.list_sessions_rich(limit=200):
            sid = row["id"]
            n = db.message_count(sid)
            if n < seen.get(sid, 0):
                out.report(event="error", error=f"count went down for {sid}: {seen[sid]} -> {n}")
            seen[sid] = max(n, seen.get(sid, 0))

    try:
        while not _stopping(stop_file):
            _patient(a, out, "read", _pass)
            passes += 1
            if passes % 5 == 0:
                out.report(event="stats", passes=passes, fds=_fd_count(), total=sum(seen.values()))
            time.sleep(a.get("pace", 0.05))
    except BaseException as exc:
        out.report(event="error", error=repr(exc), tb=traceback.format_exc()[-3000:])
        return 2
    out.report(event="stats", passes=passes, fds=_fd_count(), total=sum(seen.values()))
    db.close()
    out.report(event="closed")
    return 0


def role_churn(a: dict, out: Out) -> int:
    """`hermes sessions list` / doctor / cron-guard shape: open, read, close — ``iterations`` times in one
    process, alternating the production SessionDB with a bare sqlite3 opener (sqlite3 shell, backup tool).
    The fd count must not grow with the number of cycles."""
    from hermes_state import SessionDB

    db_path = Path(a["db"])
    start_fds = _fd_count()
    fds_after_warmup = None

    def _hermes_cycle() -> None:
        db = SessionDB(db_path=db_path)
        try:
            db.message_count()
        finally:
            db.close()

    def _raw_cycle() -> None:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        try:
            conn.execute("SELECT count(*) FROM messages").fetchone()
        finally:
            conn.close()

    try:
        for i in range(int(a["iterations"])):
            _patient(a, out, "churn", _raw_cycle if i % 2 else _hermes_cycle)
            if i == 3:
                fds_after_warmup = _fd_count()
    except BaseException as exc:
        out.report(event="error", error=repr(exc), tb=traceback.format_exc()[-3000:])
        return 2
    out.report(event="stats", start_fds=start_fds, warm_fds=fds_after_warmup, end_fds=_fd_count(),
               iterations=a["iterations"])
    return 0


def role_opener(a: dict, out: Out) -> int:
    """One short-lived process: open, count, close, exit (the last-close checkpoint path)."""
    db_path = Path(a["db"])

    def _open_count_close() -> int:
        if a.get("raw"):
            conn = sqlite3.connect(str(db_path), timeout=30.0)
            try:
                return conn.execute("SELECT count(*) FROM messages").fetchone()[0]
            finally:
                conn.close()
        from hermes_state import SessionDB
        db = SessionDB(db_path=db_path)
        try:
            return db.message_count()
        finally:
            db.close()

    try:
        n = _patient(a, out, "open", _open_count_close)
    except BaseException as exc:
        # may_fail: the chmod episode opens a read-only file; a clean refusal is correct, damage is not.
        if a.get("may_fail"):
            out.report(event="refused", error=repr(exc))
            return 0
        out.report(event="error", error=repr(exc), tb=traceback.format_exc()[-3000:])
        return 2
    out.report(event="stats", count=n)
    return 0


def _raw_count(db_path: Path) -> int:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0)
    try:
        return conn.execute("SELECT count(*) FROM messages").fetchone()[0]
    finally:
        conn.close()


def role_fts(a: dict, out: Out) -> int:
    """Maintenance pass: full FTS rebuild + optimize through SessionDB (cross-process admission)."""
    from hermes_state import SessionDB

    db = _patient(a, out, "open", lambda: SessionDB(db_path=Path(a["db"])))
    out.report(event="ready")
    try:
        rebuilt = _patient(a, out, "fts", db.rebuild_fts)
        optimized = _patient(a, out, "fts", db.optimize_fts)
    except BaseException as exc:
        out.report(event="error", error=repr(exc), tb=traceback.format_exc()[-3000:])
        return 2
    db.close()
    out.report(event="stats", rebuilt=rebuilt, optimized=optimized)
    return 0


def role_repair(a: dict, out: Out) -> int:
    """`repair_state_db_schema` as the CLI/doctor/startup recovery calls it."""
    from hermes_state_repair import repair_state_db_schema

    db_path = Path(a["db"])
    try:
        before = _raw_count(db_path)
        result = repair_state_db_schema(db_path, backup=bool(a.get("backup", True)))
        after = _raw_count(db_path)
    except BaseException as exc:
        out.report(event="error", error=repr(exc), tb=traceback.format_exc()[-3000:])
        return 2
    out.report(event="stats", before=before, after=after,
               result={k: (str(v) if v is not None else None) for k, v in result.items()})
    return 0


def role_agent(a: dict, out: Out) -> int:
    """A real AIAgent turn loop on the shared state.db, the LLM behind the loopback fake provider.

    Every turn is ``user(<base>Q) -> tool call (terminal echo <base>T) -> answer(<base>A)``; the journal
    records ``I <base>`` before ``run_conversation`` and ``A <base>`` once it returned (the turn is durable).
    With ``compress_at`` the agent runs ``/compress here <keep>`` after that turn, exactly as the CLI/TUI
    slash command does, and journals ``C <kept bases…>``. ``resume`` reloads history from state.db first
    (a fresh process resuming the session)."""
    from agent.conversation_compression_manual import compress_now, parse_compress_args
    from hermes_state import SessionDB
    from run_agent import AIAgent

    stop_file = Path(a["stop"])
    db = SessionDB(db_path=Path(a["db"]))
    agent = AIAgent(base_url=a["base_url"], api_key="sk-fake-e2e", model="fake-model", quiet_mode=True,
                    session_db=db, session_id=a["session_id"], skip_context_files=True, skip_memory=True)
    history = db.get_messages_as_conversation(a["session_id"]) if a.get("resume") else None
    out.report(event="ready", micro=bool(getattr(agent.context_compressor, "_micro_compact_enabled", False)),
               resumed=len(history or []), wal=bool(getattr(db, "_wal_active", False)))
    bases: list[str] = []
    try:
        for i in range(int(a["turns"])):
            if _stopping(stop_file):
                break
            base = f"TA{a['tag']}{os.getpid()}N{i}"  # TA: agent turns; TK: plain writer rows
            out.journal(f"I {base} {a['session_id']} 3")
            result = agent.run_conversation(f"{base}Q question {i} " + "filler " * 120,
                                            conversation_history=history)
            history = result["messages"]
            if not any(base + "A" in str(m.get("content") or "") for m in history):
                raise AssertionError(f"turn {i} ended without its answer: {result.get('final_response')!r}")
            out.journal(f"A {base}")
            bases.append(base)
            if a.get("compress_at") is not None and i == int(a["compress_at"]):
                keep = int(a.get("keep", 2))
                res = compress_now(agent, history, parse_compress_args(f"here {keep}"),
                                   skip_without_window=True)
                if res.status != "compressed":
                    raise AssertionError(f"/compress here {keep} did not compress: {res.status}")
                history = res.after_messages
                out.journal("C " + " ".join(bases[-keep:]))
    except BaseException as exc:
        out.report(event="error", error=repr(exc), tb=traceback.format_exc()[-4000:])
        return 2
    out.report(event="stats", turns=len(bases))
    db.close()
    out.report(event="closed")
    return 0


def role_cli(a: dict, out: Out) -> int:
    """``hermes <argv>``: ``hermes_cli.main`` run as ``__main__``, i.e. ``python -m hermes_cli.main <argv>``."""
    import runpy

    sys.argv = ["hermes", *a["argv"]]
    try:
        runpy.run_module("hermes_cli.main", run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    return 0


ROLES = {
    "agent": role_agent, "cli": role_cli,
    "writer": role_writer, "reader": role_reader, "churn": role_churn, "opener": role_opener,
    "fts": role_fts, "repair": role_repair,
}


def main() -> int:
    role, args = sys.argv[1], json.loads(sys.argv[2])
    _apply_sqlite_version_pin()
    if role != "cli":  # the CLI keeps its own SIGTERM handling
        signal.signal(signal.SIGTERM, _on_sigterm)
    out = Out(Path(args["workdir"]), args["name"])
    return ROLES[role](args, out)


if __name__ == "__main__":
    sys.exit(main())
