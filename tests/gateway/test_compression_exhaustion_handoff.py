"""Compression exhaustion should hand off usable context into the fresh session.

When the agent gives up after repeated compression attempts, the gateway must not
just say "try /new" or reset to an empty transcript. It should reset visibly and
seed the new session with a bounded, deterministic handoff: old-session pointer,
compact working state, todos/path references, and an assistant acknowledgement so
the next real user message has a clean role sequence.
"""

from __future__ import annotations

import ast
import inspect

from agent.compression_handoff import build_compression_handoff_messages
from gateway import run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from hermes_state import SessionDB


def _find_compression_exhausted_reset_block() -> ast.If:
    tree = ast.parse(inspect.getsource(gateway_run))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        consts = [
            n.value
            for n in ast.walk(node.test)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        ]
        if "compression_exhausted" not in consts:
            continue
        calls = {
            sub.func.attr
            for sub in ast.walk(node)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
        }
        if "reset_session" in calls:
            return node
    raise AssertionError("Could not locate compression-exhausted reset block")


def _make_agent_result() -> dict:
    huge_noise = "x" * 20_000
    return {
        "failed": True,
        "compression_exhausted": True,
        "error": "Context length exceeded (260,000 tokens). Cannot compress further.",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Bitte die Makler-Mail prüfen und nicht senden. "
                    "Arbeitsdatei: /Users/semih/Documents/Immobilien/Hoelderlinstrasse_28/"
                    "00_Arbeitsstand/Makler_Vrakas_Emailentwurf.txt\n" + huge_noise
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "[CONTEXT COMPACTION — REFERENCE ONLY]\n"
                    "## Historical Task Snapshot\n"
                    "Semih wollte automatische Session-Handoffs bei vollem Kontext.\n"
                    "## Historical Remaining Work\n"
                    "Tests ergänzen, Core implementieren, nicht committen."
                ),
                "_compressed_summary": True,
            },
            {
                "role": "user",
                "content": (
                    "[Your active task list was preserved across context compression]\n"
                    "- [>] 1. Auto-Handoff implementieren\n"
                    "- [ ] 2. Tests ausführen"
                ),
            },
        ],
    }


class TestCompressionExhaustionHandoffBuilder:
    def test_builds_bounded_role_safe_handoff_with_old_session_pointer(self):
        handoff = build_compression_handoff_messages(
            _make_agent_result(),
            old_session_id="old-session-123",
            new_session_id="new-session-456",
            profile="default",
        )

        assert [m["role"] for m in handoff] == ["user", "assistant"]
        assert "@session:default/old-session-123" in handoff[0]["content"]
        assert "new-session-456" in handoff[0]["content"]
        assert "Makler_Vrakas_Emailentwurf.txt" in handoff[0]["content"]
        assert "Auto-Handoff implementieren" in handoff[0]["content"]
        assert "x" * 1000 not in handoff[0]["content"]
        assert len(handoff[0]["content"]) < 12_000
        assert "Handoff übernommen" in handoff[1]["content"]


class TestGatewayAutoResetSeedsHandoff:
    def test_gateway_reset_block_builds_and_persists_handoff(self):
        block = _find_compression_exhausted_reset_block()
        name_calls = {
            sub.func.id
            for sub in ast.walk(block)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
        }
        attr_calls = {
            sub.func.attr
            for sub in ast.walk(block)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
        }

        assert "build_compression_handoff_messages" in name_calls
        assert "append_to_transcript" in attr_calls

    def test_seeded_handoff_is_loaded_by_fresh_session_without_old_bloat(self, tmp_path):
        store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
        store._db = SessionDB(db_path=tmp_path / "state.db")
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u1")
        entry = store.get_or_create_session(source)
        old_sid = entry.session_id
        store._db.create_session(session_id=old_sid, source="telegram", user_id="u1")
        store._db.replace_messages(
            old_sid,
            [{"role": "user", "content": "bloated" * 2000} for _ in range(80)],
        )

        new_entry = store.reset_session(entry.session_key)
        assert new_entry is not None
        handoff = build_compression_handoff_messages(
            _make_agent_result(),
            old_session_id=old_sid,
            new_session_id=new_entry.session_id,
            profile="default",
        )
        for msg in handoff:
            store.append_to_transcript(new_entry.session_id, msg)

        loaded = store.load_transcript(new_entry.session_id)
        assert [m["role"] for m in loaded] == ["user", "assistant"]
        assert "@session:default/" in loaded[0]["content"]
        assert "bloated" not in loaded[0]["content"]
        assert len(store.load_transcript(old_sid)) == 80
