"""A/B probe: does a freshly captured / restored native compaction checkpoint false-trigger
local compression on the next preflight? (#100611)

Runs the REAL ``AIAgent`` turn loop (``run_conversation``) against a local fake OpenAI
Responses SSE server that plays the ChatGPT Codex backend role (``provider="openai-codex"``
→ ``api_mode="codex_responses"``, ``is_codex_backend=True``). No mocks on the agent path
except a counting wrapper around ``_compress_context`` (a real summarizer call would need
a second LLM; the question under test is *whether it fires*, not what it writes).

Scenarios (all deterministic, no network beyond 127.0.0.1):

1. ``capture``  — turn 1 returns a ``compaction`` output item carrying N chars of
   ciphertext plus real usage below threshold; turn 2 in the SAME agent must reach the
   provider without local compression.
2. ``restore``  — the turn-1 transcript is written to a real ``SessionDB``, the DB is
   closed/reopened, a FRESH ``AIAgent`` resumes it; its first turn must reach the provider
   without local compression (idle pass armed too).
3. ``over_threshold`` (negative) — same as ``capture`` but the provider's real usage after
   the checkpoint is ABOVE the local threshold; local compression MUST still fire once real
   usage arrives (the deferral is one request, not a disable).

Usage (from a checkout root, venv python)::

    python evals/native_compaction/ab_checkpoint_preflight.py --out /tmp/result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

THRESHOLD = 204_000
CONTEXT_LENGTH = 400_000
# Reported field figure (#100611): 5,169,420 ciphertext chars → ~1.29M rough tokens.
CHECKPOINT_CHARS = 5_169_420


class _FakeResponses:
    """Local Responses API: every POST /responses answers one scripted SSE response."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.script: list[dict] = []
        self.lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):  # noqa: D401
                pass

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if not self.path.rstrip("/").endswith("/responses"):
                    self.send_response(404)
                    self.end_headers()
                    return
                with server.lock:
                    server.requests.append(body)
                    scripted = server.script.pop(0) if server.script else _text_response("ok", 1000)
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                events = [
                    {"type": "response.output_item.done", "output_index": i, "item": item}
                    for i, item in enumerate(scripted["output"])
                ] + [{"type": "response.completed", "response": scripted}]
                for ev in events:
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/backend-api/codex"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _text_response(text: str, input_tokens: int, *, compaction_chars: int = 0) -> dict:
    output = []
    if compaction_chars:
        output.append({"type": "compaction", "id": "cmp_1", "encrypted_content": "Z" * compaction_chars})
    output.append({
        "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    })
    return {
        "id": "resp_1", "object": "response", "created_at": 0, "status": "completed",
        "model": "gpt-5.6", "output": output,
        "usage": {"input_tokens": input_tokens, "output_tokens": 10, "total_tokens": input_tokens + 10},
    }


def _make_agent(base_url: str, session_id: str | None = None):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key", base_url=base_url, provider="openai-codex", model="gpt-5.6",
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[],
        max_iterations=3, session_id=session_id,
    )
    agent.compression_enabled = True
    agent.codex_responses_native_compaction = True
    cc = agent.context_compressor
    cc.context_length = CONTEXT_LENGTH
    cc.threshold_tokens = THRESHOLD
    calls: list[int] = []
    original = agent._compress_context

    def counting(messages, system_message, **kw):
        calls.append(int(kw.get("approx_tokens") or 0))
        return messages, kw.get("active_system_prompt") or (system_message.get("content") if isinstance(system_message, dict) else system_message)

    agent._compress_context = counting  # type: ignore[method-assign]
    agent._ab_compress_calls = calls
    agent._ab_original_compress = original
    return agent


def _request_facts(req: dict) -> dict:
    inp = req.get("input") or []
    return {
        "context_management": req.get("context_management"),
        "input_items": len(inp),
        "replayed_compaction_items": sum(1 for i in inp if isinstance(i, dict) and i.get("type") == "compaction"),
        "replayed_compaction_chars": sum(len(i.get("encrypted_content") or "") for i in inp if isinstance(i, dict) and i.get("type") == "compaction"),
    }


def _preflight_estimate(agent, messages) -> int | None:
    from agent.codex_responses_adapter import estimate_native_responses_preflight_tokens

    return estimate_native_responses_preflight_tokens(agent, messages, system_prompt="", tools=None)


def scenario_capture(wire: _FakeResponses, *, usage_after: int, reload_history: bool) -> dict:
    """``reload_history=True`` models the gateway: history is re-read from the DB before
    every turn, so message dicts are fresh objects and the usage anchor (keyed on ``id``)
    is stale — the rough estimator decides. ``False`` is the CLI shape (anchor protects)."""
    wire.requests.clear()
    wire.script[:] = [
        _text_response("checkpointed", 63_474, compaction_chars=CHECKPOINT_CHARS),
        _text_response("second", usage_after),
        _text_response("third", usage_after),
    ]
    agent = _make_agent(wire.base_url)
    r1 = agent.run_conversation("first request")
    history = r1["messages"]
    if reload_history:
        history = json.loads(json.dumps(history))
    carrier = next((m for m in history if m.get("role") == "assistant" and m.get("codex_reasoning_items")), None)
    est = _preflight_estimate(agent, history)
    latch_after_t1 = bool(agent.context_compressor.awaiting_real_usage_after_compression)
    compress_before_t2 = len(agent._ab_compress_calls)
    r2 = agent.run_conversation("second request", conversation_history=history)
    compress_t2 = len(agent._ab_compress_calls) - compress_before_t2
    history3 = r2["messages"]
    if reload_history:
        history3 = json.loads(json.dumps(history3))
    r3 = agent.run_conversation("third request", conversation_history=history3)
    return {
        "turn1_completed": bool(r1.get("completed")),
        "checkpoint_persisted": bool(carrier),
        "checkpoint_chars": len(carrier["codex_reasoning_items"][0]["encrypted_content"]) if carrier else 0,
        "preflight_estimate_before_turn2": est,
        "threshold": THRESHOLD,
        "latch_armed_after_turn1": latch_after_t1,
        "turn2_completed": bool(r2.get("completed")),
        "local_compress_calls_turn2": compress_t2,
        "turn3_completed": bool(r3.get("completed")),
        "local_compress_calls_turn3": len(agent._ab_compress_calls) - compress_before_t2 - compress_t2,
        "local_compress_approx_tokens": list(agent._ab_compress_calls),
        "provider_requests_total": len(wire.requests),
        "requests": [_request_facts(r) for r in wire.requests],
        "latch_after_turn3": bool(agent.context_compressor.awaiting_real_usage_after_compression),
        "last_real_prompt_tokens": agent.context_compressor.last_real_prompt_tokens,
    }


def scenario_restore(wire: _FakeResponses, tmp: Path) -> dict:
    from hermes_state import SessionDB

    wire.requests.clear()
    wire.script[:] = [
        _text_response("checkpointed", 63_474, compaction_chars=CHECKPOINT_CHARS),
        _text_response("resumed", 115_802),
    ]
    sid = "ab-native-restore"
    agent = _make_agent(wire.base_url, session_id=sid)
    r1 = agent.run_conversation("first request")
    db_path = tmp / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session(sid, source="cli")
    for m in r1["messages"]:
        if m.get("role") not in ("user", "assistant", "tool"):
            continue
        extra = {k: m[k] for k in ("codex_reasoning_items",) if m.get(k)}
        db.append_message(sid, m["role"], m.get("content") or "", **extra)
    db.close()
    reopened = SessionDB(db_path=db_path)
    history = reopened.get_messages_as_conversation(sid)
    reopened.close()
    restored_carrier = next((m for m in history if m.get("codex_reasoning_items")), None)
    fresh = _make_agent(wire.base_url, session_id=sid)
    # Idle pass armed: a long-idle restored session runs _idle_compaction before threshold preflight.
    fresh.compression_idle_compact_after_seconds = 1
    fresh._last_activity_ts = time.time() - 3600
    est = _preflight_estimate(fresh, history)
    n_req_before = len(wire.requests)
    r2 = fresh.run_conversation("after restart", conversation_history=history)
    return {
        "turn1_completed": bool(r1.get("completed")),
        "restored_checkpoint_chars": len(restored_carrier["codex_reasoning_items"][0]["encrypted_content"]) if restored_carrier else 0,
        "preflight_estimate_fresh_agent": est,
        "threshold": THRESHOLD,
        "resume_completed": bool(r2.get("completed")),
        "local_compress_calls_resume": len(fresh._ab_compress_calls),
        "local_compress_approx_tokens": list(fresh._ab_compress_calls),
        "provider_requests_resume": len(wire.requests) - n_req_before,
        "requests": [_request_facts(r) for r in wire.requests[n_req_before:]],
        "latch_after_resume": bool(fresh.context_compressor.awaiting_real_usage_after_compression),
        "last_real_prompt_tokens": fresh.context_compressor.last_real_prompt_tokens,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="ab-native-"))
    os.environ["HERMES_HOME"] = str(tmp / "home")
    (tmp / "home").mkdir(parents=True)
    import subprocess

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    wire = _FakeResponses()
    try:
        result = {
            "checkout": str(ROOT), "head": head,
            "capture_cli_same_objects": scenario_capture(wire, usage_after=115_802, reload_history=False),
            "capture_gateway_reloaded_history": scenario_capture(wire, usage_after=115_802, reload_history=True),
            "restore": scenario_restore(wire, tmp),
            # Negative: real usage after the checkpoint is STILL over threshold → local
            # compression must fire on the following turn (deferral is one request, not a disable).
            "over_threshold_negative": scenario_capture(wire, usage_after=THRESHOLD + 5_000, reload_history=True),
        }
    finally:
        wire.close()
    Path(args.out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else {
        kk: vv for kk, vv in v.items() if kk not in ("requests",)
    }) for k, v in result.items()}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
