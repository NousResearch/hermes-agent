"""One step of a crash-point case, run as a script in a process of its own: ``_driver.py <step> <home> [op]``.

The process imports whichever tree ``PYTHONPATH`` names and nothing from this test package, so the same step runs on
main's tree and on the head's. ``seed`` builds the starting home through main's own store (the test runs it on
main's tree for both sides); ``drive`` runs one operation through the entry point that
reaches it (the ``sitecustomize`` beside the test kills the process at the case's point); ``restart`` is the next
start in that home: a CLI start (``hermes sessions list``), then the gateway's start-up replay of delegation
completions, its state-database housekeeping pass and the first use of the hosted-room attachment store (which
prunes and sweeps that file once per process).
"""

import asyncio
import hashlib
import importlib.util
import json
import os
import queue
import sqlite3
import sys
import time
from pathlib import Path

DAY = 86400.0
OLD = 10 * DAY

# (session id, source, ended, old, messages, session key); every id but the never-active ones has a transcript file.
SESSIONS = (
    ("s-del", "cli", True, False, 2, None),
    ("s-keep", "cli", False, False, 1, None),
    ("s-empty", "cli", True, False, 0, None),
    ("s-prune1", "cron", True, True, 2, None),
    ("s-prune2", "cron", True, True, 1, None),
    ("s-arc1", "acp", True, True, 1, None),
    ("s-arc2", "acp", True, True, 1, None),
    ("s-na1", "telegram", False, True, 0, "agent:main:telegram:dm:101"),
    ("s-na2", "telegram", False, True, 0, "agent:main:telegram:dm:102"),
)
TRANSCRIPTS = ("s-del", "s-keep", "s-prune1", "s-prune2", "s-arc1", "s-arc2", "s-na1")
# (delegation id, completed seconds ago or None for a running unit, delivery state, delivery attempts)
DELEGATIONS = (
    ("d-run1", None, "pending", 0),
    ("d-run2", None, "pending", 0),
    ("d-old", 3 * DAY, "pending", 0),
    ("d-fresh", 120.0, "pending", 0),
    ("d-exh", 600.0, "pending", 8),
    ("d-delivered", 900.0, "delivered", 1),
)

_PLUGIN = '''import json, os


def _log(**kw):
    with open(os.path.join(os.environ["HERMES_HOME"], "hook.log"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({k: kw.get(k) for k in ("session_id", "reason", "deleted_ids")}, sort_keys=True) + "\\n")


def register(ctx):
    ctx.register_hook("on_session_delete", _log)
'''


def _write_config(home: Path, retention_days: int) -> None:
    (home / "config.yaml").write_text(
        "_config_version: 52\nupdates:\n  check: false\nplugins:\n  enabled:\n  - faultlog\n"
        f"sessions:\n  auto_prune: true\n  retention_days: {retention_days}\n  vacuum_after_prune: false\n",
        encoding="utf-8")


def seed(home: Path) -> dict:
    from hermes_state import SessionDB

    _write_config(home, 90)
    plugin = home / "plugins" / "faultlog"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: faultlog\nversion: 0.1.0\ndescription: crash-point hook log\n")
    (plugin / "__init__.py").write_text(_PLUGIN)
    now = time.time()
    db = SessionDB(db_path=home / "state.db")
    try:
        for sid, source, ended, _old, messages, key in SESSIONS:
            kwargs = {"session_key": key, "chat_id": key.rsplit(":", 1)[-1], "chat_type": "dm"} if key else {}
            db.create_session(sid, source, model="seed-model", **kwargs)
            for n in range(messages):
                db.append_message(sid, "user" if n % 2 == 0 else "assistant", content=f"{sid} message {n}")
            if ended:
                db.end_session(sid, "done")
            if key:
                db.save_gateway_routing_entry(key, json.dumps({"session_id": sid}))
    finally:
        db.close()
    sessions_dir = home / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    for sid in TRANSCRIPTS:
        (sessions_dir / f"{sid}.jsonl").write_text(json.dumps({"role": "user", "content": sid}) + "\n")
        (sessions_dir / f"request_dump_{sid}_1.json").write_text("{}")

    from gateway.session_ingress_media import native_inputs_root

    data = b"retired photo"
    digest = hashlib.sha256(data).hexdigest()
    media = native_inputs_root() / digest / "photo.png"
    media.parent.mkdir(parents=True)
    media.write_bytes(data)
    reference = {"path": str(media), "sha256": digest}

    from tools import async_delegation

    for delegation_id, ago, _state, _attempts in DELEGATIONS:
        async_delegation._persist_dispatch({
            "delegation_id": delegation_id, "dispatched_at": now - (ago or 60.0) - 60.0, "session_key": "",
            "origin_ui_session_id": "", "parent_session_id": "s-keep", "goal": f"goal of {delegation_id}"})
        if ago is not None:
            async_delegation._persist_completion(
                {"type": "async_delegation", "delegation_id": delegation_id, "status": "completed",
                 "summary": f"{delegation_id} done", "completed_at": now - ago}, {"status": "completed"})

    upload = _attachments(home).put(room_id="room-1", upload_id="upload-1", kind="file", name="first.txt",
                                    mime="text/plain", data=b"first attachment")
    (home / "attachment.json").write_text(json.dumps({"attachment_id": upload["attachment_id"]}))
    # A blob no row references, as a put killed between its blob write and its commit leaves one.
    (home / "hosted-room-attachments" / "blobs" / ("blob_" + "0" * 32)).write_bytes(b"orphan")

    conn = sqlite3.connect(home / "state.db", isolation_level=None)
    try:
        old = now - OLD
        for sid, _source, ended, is_old, _messages, _key in SESSIONS:
            if is_old:
                conn.execute("UPDATE sessions SET started_at = ?, ended_at = CASE WHEN ended_at IS NULL THEN NULL "
                             "ELSE ? END WHERE id = ?", (old, old + 60.0, sid))
                conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = ?", (old + 30.0, sid))
        conn.execute("INSERT INTO state_meta(key, value) VALUES (?, ?)",
                     ("gateway.media.retired.v1." + hashlib.sha256(reference["path"].encode()).hexdigest(),
                      json.dumps(reference)))
        for delegation_id, ago, state, attempts in DELEGATIONS:
            stale = now - (ago or 60.0)
            conn.execute("UPDATE async_delegations SET updated_at = ?, delivery_state = ?, delivery_attempts = ?, "
                         "delivered_at = CASE WHEN ? = 'delivered' THEN ? ELSE delivered_at END "
                         "WHERE delegation_id = ?", (stale, state, attempts, state, stale, delegation_id))
        conn.execute("UPDATE async_delegations SET delivery_claim = 'dead-consumer', delivery_claimed_at = ? "
                     "WHERE delegation_id = 'd-exh'", (now - 600.0,))
    finally:
        conn.close()
    return {"sessions": len(SESSIONS), "delegations": len(DELEGATIONS)}


def _attachments(home: Path, clock=None):
    """The hosted-room attachment store of the profile's file, opened the way each tree opens it."""
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore

    if importlib.util.find_spec("persistence_assembly") is None:
        return HostedRoomAttachmentStore(home / "state.db", **({"clock": clock} if clock else {}))
    from persistence_assembly import open_persistence, profile_scope

    return HostedRoomAttachmentStore(open_persistence(profile_scope()), "profile", clock=clock)


def _first_attachment_use(home: Path) -> str:
    """A read of the seeded upload, the first attachment use of the process; an upload no message committed is
    not readable, so the answer is the refusal's class name."""
    attachment_id = json.loads((home / "attachment.json").read_text())["attachment_id"]
    try:
        _attachments(home).describe(room_id="room-1", attachment_id=attachment_id, recipient_member_id="member-1",
                                    event_id="event-1")
    except Exception as exc:
        return type(exc).__name__
    return "readable"


def _put_second(home: Path, clock=None):
    return _attachments(home, clock).put(room_id="room-1", upload_id="upload-2", kind="file", name="second.txt",
                                         mime="text/plain", data=b"second attachment")


def _cli(*argv) -> int:
    from hermes_cli.main import main

    sys.argv = ["hermes", *argv]
    try:
        main()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    return 0


def _drained(target: queue.Queue) -> list:
    ids = []
    while not target.empty():
        evt = target.get_nowait()
        ids.append(evt.get("delegation_id") if isinstance(evt, dict) else repr(evt))
    return ids


def _housekeeping() -> None:
    from gateway.run import _housekeeping_state_db_maintenance

    _housekeeping_state_db_maintenance()


def drive(home: Path, op: str):
    from tools import async_delegation

    target = queue.Queue()
    if op == "delete":
        return _cli("sessions", "delete", "s-del", "--yes")
    if op == "prune":
        return _cli("sessions", "prune", "--source", "cron", "--older-than", "5", "--yes")
    if op == "never_active":
        return _cli("sessions", "prune", "--never-active", "--older-than", "5", "--yes")
    if op == "archive":
        return _cli("sessions", "archive", "--source", "acp", "--older-than", "5", "--yes")
    if op == "bulk_delete":
        from hermes_cli.web_models import BulkDeleteSessions
        from hermes_cli.web_routers.sessions import bulk_delete_sessions_endpoint

        return asyncio.run(bulk_delete_sessions_endpoint(BulkDeleteSessions(ids=["s-del"])))
    if op == "delete_empty":
        from hermes_cli.web_routers.sessions import delete_empty_sessions_endpoint

        return asyncio.run(delete_empty_sessions_endpoint())
    if op == "tui_archive":
        from tui_gateway import server

        return server.handle_request({"id": "r", "method": "session.archive", "params": {"session_id": "s-arc1"}})
    if op == "attachment_first_use":
        return _first_attachment_use(home)
    if op == "attachment_put":
        return _put_second(home)
    if op == "attachment_expired_put":
        return _put_second(home, clock=lambda: time.time() + 30 * DAY)
    if op == "auto_prune":
        _write_config(home, 5)
        return _housekeeping()
    if op == "recover":
        return async_delegation.recover_abandoned_delegations()
    if op == "restore":
        return [async_delegation.restore_undelivered_completions(target), _drained(target)]
    if op == "sweep":
        return [async_delegation.sweep_orphaned_completions(target, now=time.time() + 400.0), _drained(target)]
    if op == "requeue":
        return [async_delegation.requeue_admitted_completion("d-delivered", target), _drained(target)]
    raise SystemExit(f"unknown op {op}")


def restart(home: Path) -> dict:
    from tools import async_delegation

    print("== hermes sessions list", flush=True)
    listed = _cli("sessions", "list")
    print("== restore_undelivered_completions", flush=True)
    target = queue.Queue()
    restored = async_delegation.restore_undelivered_completions(target)
    print(restored, sorted(_drained(target)), flush=True)
    print("== state-database housekeeping", flush=True)
    _housekeeping()
    print("== first hosted-room attachment use", flush=True)
    print(_first_attachment_use(home), flush=True)
    return {"list_exit": listed, "restored": restored}


if __name__ == "__main__":
    _step, _home = sys.argv[1], Path(sys.argv[2])
    os.chdir(_home)
    if _step == "seed":
        _result = seed(_home)
    elif _step == "drive":
        _result = drive(_home, sys.argv[3])
    else:
        _result = restart(_home)
    print("DRIVER-RESULT " + json.dumps(_result, default=repr), flush=True)
