"""Reproducible HTTP payload benchmark using generated data, never a live store.

Run with the development Python from the repository root. Prints a JSON receipt;
fixture setup is excluded from timings. No provider, credential, or app access.
"""

import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    with tempfile.TemporaryDirectory(prefix="hermes-timeline-bench-") as directory:
        os.environ["HERMES_HOME"] = directory
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from hermes_state import SessionDB
        from hermes_cli.web_routers.sessions import manage_router

        sid = "generated-tool-heavy"
        prompt_count = 600
        rows_per_turn = 10
        with SessionDB(db_path=Path(directory) / "state.db") as db:
            db.create_session(session_id=sid, source="desktop")
            for start in range(0, prompt_count, 100):
                batch = []
                for turn in range(start, start + 100):
                    batch.append({"role": "user", "content": f"Investigate generated task {turn}", "timestamp": turn + 1})
                    for tool in range(4):
                        call_id = f"call-{turn}-{tool}"
                        batch.append({"role": "assistant", "content": "", "tool_calls": [{
                            "id": call_id, "type": "function", "function": {
                                "name": "terminal", "arguments": json.dumps({"command": "x" * 4096})}}]})
                        batch.append({"role": "tool", "tool_call_id": call_id,
                                      "content": "generated tool output\n" * 800})
                    batch.append({"role": "assistant", "content": f"Finished task {turn}"})
                db.append_messages_batch(sid, batch)
        app = FastAPI()
        app.include_router(manage_router)
        with TestClient(app) as client:
            def full_messages():
                total_bytes = count = requests = 0
                for offset in range(0, prompt_count * rows_per_turn + 1, 500):
                    response = client.get(f"/api/sessions/{sid}/messages", params={
                        "limit": 500, "offset": offset, "order": "oldest", "include_compacted": True})
                    response.raise_for_status()
                    page = response.json()["messages"]
                    total_bytes += len(response.content)
                    count += len(page)
                    requests += 1
                    if len(page) < 500:
                        break
                assert count == prompt_count * rows_per_turn
                return {"bytes": total_bytes, "rows": count, "requests": requests}

            def timeline():
                total_bytes = count = requests = 0
                cursor = 0
                seen = set()
                while True:
                    response = client.get(f"/api/sessions/{sid}/timeline", params={"limit": 500, "after_row_id": cursor})
                    response.raise_for_status()
                    page = response.json()
                    total_bytes += len(response.content)
                    count += len(page["entries"])
                    requests += 1
                    seen.update(entry["row_id"] for entry in page["entries"])
                    if not page["pagination"]["has_more"]:
                        break
                    cursor = page["pagination"]["next_cursor"]
                assert count == len(seen) == prompt_count
                return {"bytes": total_bytes, "rows": count, "requests": requests}

            samples = {"full_messages": [], "timeline": []}
            for _ in range(3):
                for name, procedure in (("full_messages", full_messages), ("timeline", timeline)):
                    started = time.perf_counter()
                    result = procedure()
                    result["elapsed_ms"] = (time.perf_counter() - started) * 1000
                    samples[name].append(result)
            receipt = {
                "fixture": {"prompts": prompt_count, "message_rows": prompt_count * rows_per_turn,
                            "tool_results": prompt_count * 4},
                **{name: {**runs[-1], "elapsed_ms": statistics.median(r["elapsed_ms"] for r in runs),
                          "samples_ms": [r["elapsed_ms"] for r in runs]} for name, runs in samples.items()},
            }
            receipt["payload_reduction_percent"] = 100 * (1 - receipt["timeline"]["bytes"] / receipt["full_messages"]["bytes"])
            receipt["speedup"] = receipt["full_messages"]["elapsed_ms"] / receipt["timeline"]["elapsed_ms"]
            print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
