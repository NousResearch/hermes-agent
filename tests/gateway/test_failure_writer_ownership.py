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


def test_failure_owner_uses_launch_rows_across_compaction(tmp_path):
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
        for index, (pid, parent_write) in enumerate(
            (pid, parent_write)
            for pid in (None, "current-platform-input")
            for parent_write in (False, True)
        ):
            source = SessionSource(
                platform=Platform.TELEGRAM,
                chat_id=f"fixture-{index}",
                user_id="fixture",
            )
            entry = store.get_or_create_session(source)
            sid = entry.session_id
            db = store._db_for_session_id(sid)
            # Existing identical input cannot supply current-turn ownership. Rich input's
            # durable text also differs from its prepared content.
            db.append_message(sid, "user", "same [screenshot]")
            baseline = frozenset(r["id"] for r in store.load_transcript(sid, raw=True))
            prepared = runner._PreparedTurn(
                [],
                "",
                "same",
                [{"type": "text", "text": "same"}],
                1700000000,
                None,
                sid,
                baseline,
            )
            if parent_write:
                current_id = db.append_message(
                    sid, "user", "same [screenshot]", platform_message_id=pid
                )
                with sqlite3.connect(db.db_path) as conn:
                    conn.execute(
                        "UPDATE messages SET active=0, compacted=1 WHERE id=?",
                        (current_id,),
                    )
            child = sid + "-child"
            db.end_session(sid, "compression")
            db.create_session(child, source="telegram", parent_session_id=sid)
            if not parent_write:
                current_id = db.append_message(
                    child, "user", "same [screenshot]", platform_message_id=pid
                )
            store._publish_transcript_reroute(sid, child)
            before = db.message_count()
            await runner._hmwa_agent_error_reply(
                RuntimeError("controlled post-compaction failure"),
                MessageEvent(text="same", source=source, message_id=pid),
                source,
                entry,
                entry.session_key,
                prepared,
            )
            assert db.message_count() == before
            assert current_id in {r["id"] for r in store.load_transcript(sid, raw=True)}
        db.close()

    asyncio.run(check())
