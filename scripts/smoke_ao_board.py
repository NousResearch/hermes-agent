#!/usr/bin/env python3
"""Spawn a bounded AO batch and verify Agent Board rows expose launch metadata.

Usage:
    API_SERVER_KEY=... .venv/bin/python scripts/smoke_ao_board.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ao_delegate_tool import ao_delegate_batch  # noqa: E402


def _request_json(base_url: str, path: str, *, api_key: str | None, method: str = "GET") -> dict[str, Any]:
    request = urllib.request.Request(
        urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/")),
        method=method,
        headers={"Accept": "application/json"},
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    if method == "POST":
        request.add_header("Content-Type", "application/json")
        request.data = b"{}"
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _board_rows(base_url: str, api_key: str | None) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({
        "runtime": "ao",
        "include_completed": "true",
        "limit": "500",
    })
    payload = _request_json(base_url, f"/v1/subagents/board?{query}", api_key=api_key)
    return list(payload.get("data") or [])


def _stop_sessions(base_url: str, api_key: str | None, session_ids: list[str]) -> None:
    for session_id in session_ids:
        encoded = urllib.parse.quote(session_id, safe="")
        try:
            _request_json(base_url, f"/v1/ao/sessions/{encoded}/stop", api_key=api_key, method="POST")
        except Exception as exc:
            print(f"cleanup warning: could not stop {session_id}: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("HERMES_GATEWAY_URL", "http://127.0.0.1:8642"))
    parser.add_argument("--api-key", default=os.getenv("API_SERVER_KEY") or os.getenv("HERMES_API_KEY"))
    parser.add_argument("--project-id", default="OrynWorkspace")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--keep-running", action="store_true")
    args = parser.parse_args()

    count = max(1, min(args.count, 5))
    tasks = [
        {
            "goal": f"Agent Board smoke worker {index}",
            "project_id": args.project_id,
            "prompt": (
                "Do not modify files. Reply with a concise final answer ending in "
                f"BOARD_SMOKE_{index}_DONE."
            ),
        }
        for index in range(1, count + 1)
    ]

    result = json.loads(ao_delegate_batch(tasks=tasks))
    session_ids = [
        item.get("session", {}).get("ao_session_id")
        for item in result.get("sessions", [])
        if item.get("session", {}).get("ao_session_id")
    ]
    if not session_ids:
        print(json.dumps(result, indent=2))
        return 1

    print(f"spawned: {', '.join(session_ids)}")
    deadline = time.time() + args.timeout
    rows_by_session: dict[str, dict[str, Any]] = {}
    try:
        while time.time() < deadline:
            try:
                rows = _board_rows(args.base_url, args.api_key)
            except urllib.error.HTTPError as exc:
                if exc.code == 401:
                    print("board API returned 401; pass --api-key or set API_SERVER_KEY", file=sys.stderr)
                raise
            rows_by_session = {
                row.get("ao_session_id"): row
                for row in rows
                if row.get("ao_session_id") in session_ids
            }
            if len(rows_by_session) == len(session_ids):
                break
            time.sleep(2)

        missing = [session_id for session_id in session_ids if session_id not in rows_by_session]
        if missing:
            print(f"missing board rows: {', '.join(missing)}", file=sys.stderr)
            return 1

        failures: list[str] = []
        for session_id in session_ids:
            row = rows_by_session[session_id]
            for field in ("agent", "model", "reasoning_effort"):
                if not row.get(field):
                    failures.append(f"{session_id} missing {field}")
            print(
                f"{session_id}: lane={row.get('lane')} "
                f"agent={row.get('agent')} model={row.get('model')} "
                f"reasoning={row.get('reasoning_effort')}"
            )
        if failures:
            print("\n".join(failures), file=sys.stderr)
            return 1
        print("AO board smoke passed")
        return 0
    finally:
        if not args.keep_running:
            _stop_sessions(args.base_url, args.api_key, session_ids)


if __name__ == "__main__":
    raise SystemExit(main())
