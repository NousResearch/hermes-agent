#!/usr/bin/env python3
"""Measure state.db savings from compact tool-result handles on noisy sessions.

The harness uses a temporary HERMES_HOME, writes synthetic noisy tool outputs
with handles disabled and enabled, and compares the persisted SQLite content
bytes. It does not call an LLM or external services.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _write_config(home: Path, enabled: bool, threshold: int, preview: int) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "tool_results:\n"
        f"  handles_enabled: {'true' if enabled else 'false'}\n"
        f"  handle_threshold_bytes: {threshold}\n"
        f"  preview_bytes: {preview}\n",
        encoding="utf-8",
    )


def _noisy_payload(session_index: int, tool_index: int, bytes_per_output: int) -> str:
    marker = f"SESSION-{session_index:03d}-TOOL-{tool_index:03d}-"
    line = marker + ("0123456789abcdef" * 8) + "\n"
    repeats = max(1, (bytes_per_output // len(line)) + 1)
    return (line * repeats)[:bytes_per_output]


def _run_case(root: Path, *, enabled: bool, sessions: int, tools_per_session: int, bytes_per_output: int, threshold: int, preview: int) -> dict[str, int | float | str | bool]:
    home = root / ("enabled" if enabled else "disabled")
    _write_config(home, enabled=enabled, threshold=threshold, preview=preview)
    os.environ["HERMES_HOME"] = str(home)

    # Import after HERMES_HOME is set so DEFAULT_DB_PATH and config helpers use
    # the temporary harness home.
    from agent.tool_result_handles import maybe_handle_tool_result
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    raw_total = 0
    stored_total = 0
    artifact_count = 0

    for sidx in range(sessions):
        sid = f"bench-{sidx:03d}"
        db.create_session(sid, "benchmark")
        for tidx in range(tools_per_session):
            raw = _noisy_payload(sidx, tidx, bytes_per_output)
            raw_total += len(raw.encode("utf-8"))
            content = maybe_handle_tool_result(
                raw,
                tool_name="terminal",
                args={"command": f"noise {sidx} {tidx}"},
                session_id=sid,
                tool_call_id=f"tc-{sidx}-{tidx}",
            )
            if content != raw:
                artifact_count += 1
            stored_total += len(content.encode("utf-8"))
            db.append_message(
                sid,
                "tool",
                content,
                tool_name="terminal",
                tool_call_id=f"tc-{sidx}-{tidx}",
            )

    conn = sqlite3.connect(home / "state.db")
    db_content_bytes = conn.execute(
        "SELECT COALESCE(SUM(LENGTH(CAST(content AS BLOB))), 0) FROM messages WHERE role = 'tool'"
    ).fetchone()[0]
    rows = conn.execute("SELECT COUNT(*) FROM messages WHERE role = 'tool'").fetchone()[0]
    conn.close()

    return {
        "enabled": enabled,
        "home": str(home),
        "rows": int(rows),
        "raw_output_bytes": raw_total,
        "stored_content_bytes": int(db_content_bytes),
        "python_counted_stored_bytes": stored_total,
        "artifact_count": artifact_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--tools-per-session", type=int, default=6)
    parser.add_argument("--bytes-per-output", type=int, default=48_000)
    parser.add_argument("--threshold", type=int, default=24_000)
    parser.add_argument("--preview", type=int, default=2_000)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="hermes-tool-result-bench-") as tmp:
        root = Path(tmp)
        disabled = _run_case(
            root,
            enabled=False,
            sessions=args.sessions,
            tools_per_session=args.tools_per_session,
            bytes_per_output=args.bytes_per_output,
            threshold=args.threshold,
            preview=args.preview,
        )
        enabled = _run_case(
            root,
            enabled=True,
            sessions=args.sessions,
            tools_per_session=args.tools_per_session,
            bytes_per_output=args.bytes_per_output,
            threshold=args.threshold,
            preview=args.preview,
        )

        before = int(disabled["stored_content_bytes"])
        after = int(enabled["stored_content_bytes"])
        saved = before - after
        savings_pct = (saved / before * 100.0) if before else 0.0
        result = {
            "parameters": vars(args),
            "disabled": disabled,
            "enabled": enabled,
            "saved_state_db_content_bytes": saved,
            "savings_pct": round(savings_pct, 2),
        }
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
