"""Run from the checkout: PYTHONPATH=. python evals/codex_echo/probe.py.

Real subprocess pipes, protocol parser/projector, persistence and SQLite replay.
The peer is an offline fixture, not a Codex/provider validation.
"""
import json
import os
from pathlib import Path
import tempfile
import threading


def main():
    with tempfile.TemporaryDirectory(prefix="hermes-echo-") as tmp:
        os.environ.update(HOME=tmp, HERMES_HOME=tmp, CODEX_HOME=tmp, HERMES_DISABLE_PLUGINS="1")
        from agent import codex_runtime
        from agent.message_metadata import append_message
        from agent.session_persistence import SessionPersistenceMixin
        from agent.transports.codex_app_server_session import CodexAppServerSession
        from hermes_state import SessionDB

        observations = []
        for mode in ["echo", "assistant_only", "different", "later_equal"]:
            for rich in [False, True]:
                os.environ["ECHO_FIXTURE_MODE"] = mode
                sid = f"{mode}-{rich}"
                db = SessionDB(Path(tmp) / f"{sid}.db")
                db.create_session(session_id=sid, source="telegram", model="fixture")
                agent = SessionPersistenceMixin()
                agent.session_id = sid
                agent._session_db = db
                agent._session_db_created = True
                agent._last_flushed_db_idx = 0
                agent._session_persist_lock = threading.RLock()
                session = CodexAppServerSession(cwd=tmp, codex_home=tmp,
                    codex_bin=str(Path(__file__).with_name("fixture_server.py").resolve()))
                messages = []
                wire_input = [{"type": "text", "text": "caption"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}] if rich else "caption"
                try:
                    for accepted in range(2):
                        append_message(messages, {"role": "user", "content": "caption",
                            "platform_message_id": str(accepted) if not rich else None})
                        assert agent._flush_messages_to_session_db(messages)
                        turn = session.run_turn(wire_input, turn_timeout=10)
                        assert turn.error is None, turn.error
                        assert turn.final_text == "fixture reply", turn
                        codex_runtime._persist_projected_messages(agent, turn, messages)
                        assert agent._flush_messages_to_session_db(messages)
                    users = [m["content"] for m in db.get_messages_as_conversation(sid) if m["role"] == "user"]
                    expected_count = 4 if mode in {"different", "later_equal"} else 2
                    observations.append({"mode": mode, "rich_keyless": rich, "users": users,
                        "expected_user_rows": expected_count, "user_rows": len(users), "pass": len(users) == expected_count})
                finally:
                    session.close()
                    db.close()
        print(json.dumps({"module": codex_runtime.__file__, "cases": observations}, indent=2))


if __name__ == "__main__":
    main()
