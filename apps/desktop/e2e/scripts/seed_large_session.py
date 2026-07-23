#!/usr/bin/env python3
"""Seed a deterministic, tool-free large session into an isolated state.db."""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from hermes_state import SessionDB  # noqa: E402

SESSION_ID = "e2e-large-session"
SESSION_TITLE = "E2E large persisted session"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <state.db>")

    messages = []
    for index in range(53):
        role = "user" if index % 2 == 0 else "assistant"
        content = (
            f"E2E persisted user message {index}: audit the compatibility matrix"
            if role == "user"
            else f"E2E persisted assistant reply {index}: recorded the audit result"
        )
        messages.append({"role": role, "content": content, "timestamp": 1_700_000_000 + index})

    database = SessionDB(db_path=Path(sys.argv[1]))
    result = database.import_sessions(
        [
            {
                "id": SESSION_ID,
                "source": "desktop",
                "model": "mock-model",
                "started_at": 1_700_000_000,
                "title": SESSION_TITLE,
                "cwd": str(repo_root),
                "system_prompt": "",
                "messages": messages,
            }
        ]
    )
    database.close()

    if not result.get("ok") or result.get("imported") != 1:
        raise SystemExit(f"failed to seed large session: {result}")


if __name__ == "__main__":
    main()
