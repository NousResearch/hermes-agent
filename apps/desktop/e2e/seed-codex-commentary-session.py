"""Seed one Codex commentary row into a disposable SessionDB for Electron E2E."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hermes_state import SessionDB

REASONING_SUMMARY = "Inspecting the transcript"
COMMENTARY = "I’m checking the persisted turn now."
CANONICAL_FINAL = "Persisted canonical answer."
FLATTENED_REASONING = f"{REASONING_SUMMARY}\n\n{COMMENTARY}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hermes-home", required=True, type=Path)
    parser.add_argument("--state-db", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hermes_home = args.hermes_home.resolve()
    state_db = args.state_db.resolve()
    if Path(os.environ["HERMES_HOME"]).resolve() != hermes_home:
        raise SystemExit("HERMES_HOME does not match --hermes-home")
    if state_db != hermes_home / "state.db" or not state_db.is_file():
        raise SystemExit("--state-db must identify the disposable HERMES_HOME/state.db")

    db = SessionDB(db_path=state_db)
    try:
        if db.get_session(args.session_id) is None:
            raise SystemExit(f"Durable session does not exist: {args.session_id}")
        row_id = db.append_message(
            session_id=args.session_id,
            role="assistant",
            content=CANONICAL_FINAL,
            reasoning=FLATTENED_REASONING,
            codex_reasoning_items=[{
                "type": "reasoning",
                "id": "rs_resume",
                "summary": [{"type": "summary_text", "text": REASONING_SUMMARY}],
                "encrypted_content": "E2E_ENCRYPTED_SENTINEL",
            }],
            codex_message_items=[
                {
                    "type": "message", "id": "analysis-e2e", "role": "assistant", "phase": "analysis",
                    "content": [{"type": "output_text", "text": "E2E_ANALYSIS_SENTINEL"}],
                },
                {
                    "type": "message", "id": "commentary-e2e", "role": "assistant", "phase": "commentary",
                    "content": [{"type": "output_text", "text": COMMENTARY}],
                },
            ],
            tool_calls=[{
                "id": "tc_commentary_e2e", "type": "function",
                "function": {"name": "terminal", "arguments": json.dumps({"command": "printf commentary-e2e"})},
            }],
        )
    finally:
        db.close()
    print(json.dumps({"row_id": row_id, "session_id": args.session_id}))


if __name__ == "__main__":
    main()
