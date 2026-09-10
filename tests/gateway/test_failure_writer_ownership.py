"""Accepted inbound turns have one durable owner, including exception fallback."""

import json
import subprocess
import sys
from pathlib import Path


def test_gateway_failure_writer_preserves_accepted_turn_identity(tmp_path):
    root = Path(__file__).resolve().parents[2]
    receipt = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "evals/gateway_failure_ownership/probe.py"),
            str(root),
            str(receipt),
        ],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=90,
    )
    assert receipt.exists(), result.stdout + result.stderr
    data = json.loads(receipt.read_text())
    assert all(row["reached"] for row in data["observations"]), data
    assert data["passed"] == data["total"], data["observations"]
    assert result.returncode == 0, result.stdout + result.stderr


def test_failure_owner_follows_only_live_lineage_markers(tmp_path):
    import asyncio
    import sqlite3
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, SessionStore

    async def check():
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        runner = object.__new__(GatewayRunner)
        runner.session_store = store

        async def stop_typing(event, source):
            return None

        runner._hmwa_stop_typing_for_turn = stop_typing
        # An archived ancestor or the published live child can own a turn. Neither
        # a reaped sibling nor undone/observed/foreign-marker rows can own it.
        cases = ("ancestor", "middle-ancestor", "live-child", "reaped", "undone", "observed", "foreign", "unmarked")
        for index, (location, pid) in enumerate(
            (location, pid) for location in cases for pid in (None, "100")
        ):
            source = SessionSource(
                platform=Platform.TELEGRAM, chat_id=f"fixture-{index}", user_id="fixture"
            )
            entry = store.get_or_create_session(source)
            sid = entry.session_id
            db = store._db_for_session_id(sid)
            owner = f"accepted-owner-{index}"
            metadata = {"gateway_input_owner": owner}
            prepared = runner._PreparedTurn(
                [], "", "same", [{"type": "text", "text": "same"}],
                1700000000, None, sid, owner,
            )
            db.append_message(sid, "user", "same [screenshot]")
            middle = sid + "-middle"
            db.create_session(middle, source="telegram", parent_session_id=sid)
            orphan = sid + "-orphan"
            child = sid + "-live"
            db.create_session(orphan, source="telegram", parent_session_id=middle)
            db.end_session(orphan, "ws_orphan_reap")
            db.create_session(child, source="telegram", parent_session_id=middle)
            target = {"ancestor": sid, "middle-ancestor": middle, "reaped": orphan}.get(location, child)
            marker = {"gateway_input_owner": "foreign-writer"} if location == "foreign" else metadata
            current_id = db.append_message(
                target, "user", "same [screenshot]", platform_message_id=pid,
                observed=location == "observed",
                display_metadata=None if location == "unmarked" else marker,
            )
            db.end_session(sid, "compression")
            db.end_session(middle, "compression")
            if location in ("ancestor", "middle-ancestor", "undone"):
                with sqlite3.connect(db.db_path) as conn:
                    conn.execute(
                        "UPDATE messages SET active=0, compacted=? WHERE id=?",
                        (int(location != "undone"), current_id),
                    )
            store._publish_transcript_reroute(sid, child)
            owned = location in ("ancestor", "middle-ancestor", "live-child")
            assert store.has_input_owner(sid, owner) is owned, location
            # A restarted store has no published map and must pick the same live child.
            store._transcript_reroutes.clear()
            assert store.has_input_owner(sid, owner) is owned, location
            before = db.message_count()
            await runner._hmwa_agent_error_reply(
                RuntimeError("controlled post-compaction failure"),
                MessageEvent(text="same", source=source, message_id=pid),
                source, entry, entry.session_key, prepared,
            )
            assert db.message_count() == before + (not owned), location
            assert store.has_input_owner(sid, owner), location
            if not owned:
                latest = db.get_messages(child)[-1]
                assert latest["content"] == prepared.persist_user_message
                assert latest["display_metadata"]["gateway_input_owner"] == owner
        db.close()

    asyncio.run(check())
