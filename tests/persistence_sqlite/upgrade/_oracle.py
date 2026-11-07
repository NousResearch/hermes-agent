"""Upgrade and rollback oracle: main's store and the SQLite backend's holder each open a store in a
process of their own, one step per process, so nothing one side loads or caches reaches the other.

As a script, ``python _oracle.py <side> <step> <json-args>`` runs one step with PYTHONPATH at the tree that
side imports and prints its result as the last stdout line. Imported, it gives the tests ``run_step`` and the
store comparisons, which read files with plain ``sqlite3`` and import no store module.
"""

import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[3]
if __name__ != "__main__":
    # A step run as a script imports only its side's tree, which need not carry this package.
    from tests.persistence_sqlite._legacy_store import ORACLE_TREE, oracle_tree

FIXTURES = (
    "absent", "current", "v15", "v17", "v19", "v21", "v22", "v24", "v29", "v30", "v31", "usage_stale_pk",
    "usage_heal_interrupted", "routing_legacy_pk", "deferred_missing", "columns_missing", "newer_v33", "lazy_tables",
    "telegram_topic_v2", "fts_stale", "fts_v0_inline", "fts_v1", "fts_v2", "fts_orphan_shadow", "fts_trigram_dropped",
    "uid_backfill_partial", "optimize_interrupted", "jsonl_converted",
)

_RESULT_PREFIX = "ORACLE-RESULT "
_STEP_TIMEOUT_S = 180

# The session row main inserts (hermes_state_worker_lifecycle.insert_session_row_in_transaction) and the one
# message INSERT every main writer uses (hermes_state_messages._INSERT_MESSAGE_SQL), as raw SQL.
SESSION_INSERT_SQL = """INSERT INTO sessions (
   id, source, created_source, user_id, session_key, chat_id, chat_type, thread_id,
   model, model_config, system_prompt, system_prompt_hash,
   parent_session_id, cwd, profile_name, transport_profile, git_repo_root,
   origin_json, display_name, started_at
)
   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
MESSAGE_INSERT_SQL = """INSERT INTO messages (session_id, role, content, tool_call_id,
                   tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                   reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                   codex_message_items, platform_message_id, observed, _compressed_summary, active, api_content, display_kind,
                   display_metadata, display_identity, message_uid, absorbed_message_uids, tool_call_uids,
                   tool_call_uid)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

# Read back through each side's own read path; identical SQL, so the two answers compare row for row.
READS = {
    "schema_version": "SELECT version FROM schema_version",
    "sessions": "SELECT id, source, model, message_count FROM sessions ORDER BY id",
    "messages": "SELECT session_id, role, content FROM messages ORDER BY session_id, rowid",
    "state_meta": "SELECT key, value FROM state_meta ORDER BY key",
}

# state_meta values minted per store or per open: compared by presence only.
PER_STORE_META = {"store_instance_id", "store_created_at_utc", "db_file_generation"}
# Columns minted at random per row (by the uid backfill and the messages_message_uid_insert trigger).
RANDOM_COLUMNS = {"messages": {"message_uid"}}
# Values an open may legitimately rewrite in fixture rows, by the step of main's schema code that rewrites them:
# the version stamp, v16 tags, v18 fills NULLs, v25 moves prompts, the uid backfill, the NULL ``active`` heal,
# the usage PK heal's COALESCE(task, ''), the routing PK heal's COALESCE(scope, ''), and SQLite's own
# AUTOINCREMENT counter.
MIGRATED_COLUMNS = {
    "schema_version": {"version"},
    "sqlite_sequence": {"seq"},
    "sessions": {"model_config", "system_prompt", "system_prompt_hash", "session_key", "chat_id", "chat_type",
                 "thread_id", "display_name", "origin_json", "expiry_finalized"},
    "messages": {"message_uid", "active"},
    "session_model_usage": {"task"},
    "gateway_routing": {"scope"},
}
# state_meta keys an open consumes: the uid cursor once the backfill completes, the realign's markers
# and the stale breadcrumb a recovery clears.
CONSUMED_META = {"message_uid_backfill_id", "fts_rebuild_high_water", "fts_rebuild_progress",
                 "fts_tool_full_content_high_water", "fts_stale"}
_FTS_SHADOW_SUFFIXES = ("content", "data", "docsize", "idx", "config")
_FTS_FAMILIES = ("messages_fts", "messages_fts_trigram", "messages_fts_cjk")
FTS_PROBES = ("payment", "ayment", "迁移内容", "hello", "terminal")


def run_step(side: str, step: str, env: dict, **args) -> dict:
    """Run one step in a fresh interpreter importing *side*'s tree, the pinned main checkout for main's side (the
    test skips without one); return its JSON result."""
    tree = oracle_tree() if side == "m1" else WORKTREE
    proc = subprocess.run(
        [sys.executable, str(Path(__file__)), side, step, json.dumps(args)],
        env={**env, "PYTHONPATH": str(tree)}, cwd=env["HERMES_HOME"], capture_output=True, text=True, check=False,
        timeout=_STEP_TIMEOUT_S,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith(_RESULT_PREFIX)]
    if proc.returncode != 0 or not lines:
        raise AssertionError(
            f"{side} {step} {args} exited {proc.returncode}\n--- stdout\n{proc.stdout[-3000:]}\n--- stderr\n"
            f"{proc.stderr[-6000:]}"
        )
    return json.loads(lines[-1][len(_RESULT_PREFIX):])


def copy_store(src_dir: Path, dst_dir: Path) -> Path:
    """Copy a built fixture's home (the database, any sidecars and the home files beside them) to *dst_dir*."""
    shutil.copytree(src_dir, dst_dir)
    return dst_dir / "state.db"


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def file_digests(db: Path) -> dict:
    """sha256 of the main file and its -wal (None when absent)."""
    return {
        suffix: hashlib.sha256(Path(f"{db}{suffix}").read_bytes()).hexdigest()
        if Path(f"{db}{suffix}").exists() else None
        for suffix in ("", "-wal")
    }


def _sort_key(row) -> str:
    return repr(row)


def snapshot(db: Path) -> dict:
    """Schema and contents of a closed store, normalised so two stores migrated from one fixture compare
    equal: sqlite_master with whitespace collapsed, every ordinary table's rows (random and per-store values
    reduced to presence), FTS indexes by indexed rowid set and MATCH results, application_id presence."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        master = sorted(
            (kind, name, tbl, " ".join((sql or "").split()))
            for kind, name, tbl, sql in conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master")
        )
        vtables = {name for kind, name, _, sql in master if kind == "table" and sql.startswith("CREATE VIRTUAL TABLE")}
        shadows = {f"{family}_{suffix}" for family in _FTS_FAMILIES for suffix in _FTS_SHADOW_SUFFIXES}
        names = {name for _, name, _, _ in master}
        tables, columns = {}, {}
        for kind, name, _, _ in master:
            if kind != "table" or name in vtables or (name in shadows and name.rsplit("_", 1)[0] in vtables):
                continue
            cols = [row[1] for row in conn.execute(f'PRAGMA table_info("{name}")')]
            rows = []
            for row in conn.execute(f'SELECT * FROM "{name}"'):
                values = dict(zip(cols, row))
                for col in RANDOM_COLUMNS.get(name, ()):
                    if values.get(col) is not None:
                        values[col] = "<random>"
                if name == "state_meta" and values["key"] in PER_STORE_META:
                    values["value"] = "<per-store>" if values["value"] else values["value"]
                rows.append(tuple(values[c] for c in cols))
            tables[name] = sorted(rows, key=_sort_key)
            columns[name] = cols
        fts = {}
        for name in sorted(vtables):
            entry = {}
            if f"{name}_docsize" in names:
                entry["indexed"] = sorted(r[0] for r in conn.execute(f'SELECT id FROM "{name}_docsize"'))
            for probe in FTS_PROBES:
                try:
                    entry[probe] = sorted(
                        r[0] for r in conn.execute(f'SELECT rowid FROM "{name}" WHERE "{name}" MATCH ?', (f'"{probe}"',))
                    )
                except sqlite3.Error as exc:
                    entry[probe] = f"error: {exc}"
            fts[name] = entry
        app_id = conn.execute("PRAGMA application_id").fetchone()[0]
    finally:
        conn.close()
    return {"sqlite_master": master, "tables": tables, "columns": columns, "fts": fts, "application_id_set": app_id != 0}


def snapshot_diff(a: dict, b: dict, labels=("m1", "subject")) -> list:
    """Readable differences between two snapshots (empty when equal)."""
    out = []
    for key in sorted(set(a) | set(b)):
        left, right = a.get(key), b.get(key)
        if left == right:
            continue
        if isinstance(left, dict) and isinstance(right, dict):
            for sub in sorted(set(left) | set(right)):
                if left.get(sub) != right.get(sub):
                    out.append(f"{key}[{sub}]: {labels[0]}={left.get(sub)!r}\n    {labels[1]}={right.get(sub)!r}")
        elif isinstance(left, list) and isinstance(right, list):
            only_left = [x for x in left if x not in right]
            only_right = [x for x in right if x not in left]
            out.append(f"{key}: only in {labels[0]}={only_left!r}\n    only in {labels[1]}={only_right!r}")
        else:
            out.append(f"{key}: {labels[0]}={left!r} {labels[1]}={right!r}")
    return out


def lost_fixture_rows(before: dict, after: dict) -> list:
    """Fixture rows missing after the opens, each table projected on the fixture's columns minus the ones an
    open rewrites. FTS shadow tables (derived index state) and state_meta values are left out; state_meta keys
    are checked apart from the ones an open consumes."""
    lost = []
    for table, rows in before["tables"].items():
        if table == "state_meta":
            keys_after = {row[0] for row in after["tables"].get("state_meta", [])}
            lost += [f"state_meta key {row[0]!r}" for row in rows if row[0] not in keys_after | CONSUMED_META]
            continue
        if table.rsplit("_", 1)[-1] in _FTS_SHADOW_SUFFIXES and table.rsplit("_", 1)[0] in _FTS_FAMILIES:
            continue
        keep = [c for c in before["columns"][table] if c not in MIGRATED_COLUMNS.get(table, ())]
        cols_after = after["columns"].get(table)
        if cols_after is None or not set(keep) <= set(cols_after):
            lost.append(f"{table}: table or columns gone (after: {cols_after})")
            continue
        projected_after = [tuple(dict(zip(cols_after, r))[c] for c in keep) for r in after["tables"][table]]
        for row in rows:
            projected = tuple(dict(zip(before["columns"][table], row))[c] for c in keep)
            if projected in projected_after:
                projected_after.remove(projected)
            else:
                lost.append(f"{table}: {dict(zip(keep, projected))!r}")
    return lost


# ── Steps, run in the subprocess ─────────────────────────────────────────────────────────────────────────────


def _store_class(side: str):
    if side == "backend":
        from persistence.sqlite._holder import StoreHolder

        return lambda db, read_only=False: StoreHolder(Path(db), read_only=read_only)
    from hermes_state import SessionDB

    return lambda db, read_only=False: SessionDB(db_path=Path(db), read_only=read_only)


def _open(side: str, db: str, read_only: bool = False):
    return _store_class(side)(db, read_only=read_only)


def _step_build(side: str, db: str, name: str) -> dict:
    from _fixtures import FIXTURES as BUILDERS

    BUILDERS[name](Path(db))
    return {"built": name}


def _step_open(side: str, db: str) -> dict:
    _open(side, db).close()
    return {}


def _step_raw_write(side: str, db: str, tag: str) -> dict:
    """One transaction through the store's own writer: a session row and two messages with main's column sets,
    the counter bump main runs with them, then a state_meta key through the store's meta setter."""
    store = _open(side, db)
    try:
        now = time.time()
        session_id = f"{tag}-session"

        def write(conn):
            conn.execute(SESSION_INSERT_SQL, (
                session_id, "cli", "cli", None, None, None, None, None, f"{tag}-model", None, None, None, None,
                None, None, None, None, None, now,
            ))
            for offset, (role, content) in enumerate((("user", f"{tag} payment question"), ("assistant", f"{tag} answer"))):
                conn.execute(MESSAGE_INSERT_SQL, (
                    session_id, role, content, None, None, None, None, now + offset, None, None, None, None, None,
                    None, None, None, 0, 0, 1, None, None, None, None, uuid.uuid4().hex, None, None, None,
                ))
            conn.execute("UPDATE sessions SET message_count = message_count + 2 WHERE id = ?", (session_id,))

        store._execute_write(write)
        store.set_meta(f"{tag}-meta", tag)
    finally:
        store.close()
    return {"session": session_id}


def _step_api_write(side: str, db: str, tag: str) -> dict:
    """Main's public write API: create_session, append_message, set_meta."""
    assert side == "m1"
    store = _open(side, db)
    try:
        session_id = f"{tag}-session"
        store.create_session(session_id, "cli", model=f"{tag}-model")
        store.append_message(session_id, "user", content=f"{tag} payment question")
        store.append_message(session_id, "assistant", content=f"{tag} answer")
        store.set_meta(f"{tag}-meta", tag)
    finally:
        store.close()
    return {"session": session_id}


def _step_read(side: str, db: str) -> dict:
    """Open read-only and run READS through the store's read path; an open or read failure is the result
    (an import failure is not)."""
    open_store = _store_class(side)
    try:
        store = open_store(db, read_only=True)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    try:
        out = {key: [list(row) for row in store._read_all(sql, ())] for key, sql in READS.items()}
        out["meta_api"] = {key: store.get_meta(key) for key, _ in out["state_meta"]}
    except Exception as exc:
        out = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        store.close()
    return out


_STEPS = {"build": _step_build, "open": _step_open, "raw_write": _step_raw_write, "api_write": _step_api_write,
          "read": _step_read}


if __name__ == "__main__":
    _side, _step, _args = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    _result = _STEPS[_step](_side, **_args)
    print(_RESULT_PREFIX + json.dumps(_result), flush=True)
