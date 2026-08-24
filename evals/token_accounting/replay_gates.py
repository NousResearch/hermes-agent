"""Token-accounting replay: do the compaction gates follow the provider's REAL usage or the
local ``bytes/4`` estimate? (#104462)

Runs the REAL ``AIAgent`` turn loop against a local fake OpenAI chat-completions server whose
``usage.prompt_tokens`` is scripted per request and deliberately disagrees with the transcript
size. Gate scenarios replace ``_compress_context`` with a counting hook; recovery
scenarios run real compression with only summary generation replaced by fixed local text.

Shapes (each runs several turns):

* ``cli``      — same history list objects across turns (CLI REPL).
* ``gateway``  — history JSON round-tripped before every turn (fresh dict identities, as the
                 gateway re-reads the transcript from the DB each turn).
* ``restore``  — turn 1 persisted to a real ``SessionDB``, DB closed/reopened, a FRESH
                 ``AIAgent`` resumes (desktop per-turn ``serve`` / CLI ``--resume``).

Scenarios per shape:

* ``inflated``  — transcript ~3x the threshold by bytes/4, provider reports real usage well
                  UNDER threshold. No gate may fire (positive arm).
* ``past_window_inflated`` — whole-history estimate exceeds the entire model window.
* ``overflow_recovery`` / ``usage_less_fallback`` — real compression and local HTTP retry,
  with fixed local summary text (no inference); provider capability probes do not consume usage.
* ``deflated``  — transcript tiny by bytes/4, provider reports real usage OVER threshold.
                  Local compression MUST fire once real usage is known (negative arm).

Usage (from a checkout root, venv python)::

    python evals/token_accounting/replay_gates.py --out /tmp/result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CONTEXT_LENGTH = 128_000
THRESHOLD = 40_000
TURNS = 3
# ~3x threshold by bytes/4 (ASCII: 4 bytes per rough token).
INFLATED_CHARS = THRESHOLD * 4 * 3
REAL_UNDER = 9_000
REAL_OVER = THRESHOLD + 5_000


class _FakeChat:
    """Local chat-completions API: every POST answers one scripted (non-stream or SSE) response."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.usage_script: list[int | None | str] = []
        self.lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):  # noqa: D401
                pass

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                # Runtime capability probes are not model requests and must not consume usage.
                if "messages" not in body:
                    self.send_error(404, "Only chat completions are supported by this fixture")
                    return
                with server.lock:
                    server.requests.append(body)
                    prompt = server.usage_script.pop(0) if server.usage_script else REAL_UNDER
                if prompt == "overflow":
                    data = json.dumps({"error": {"message": "maximum context length exceeded",
                                                "type": "invalid_request_error",
                                                "code": "context_length_exceeded"}}).encode()
                    self.send_response(400)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                usage = ({"prompt_tokens": prompt, "completion_tokens": 5, "total_tokens": prompt + 5}
                         if isinstance(prompt, int) else None)
                msg = {"role": "assistant", "content": "ok"}
                if body.get("stream") is True:
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.end_headers()
                    for chunk in (
                        {"id": "m", "object": "chat.completion.chunk", "model": body.get("model", "m"),
                         "choices": [{"index": 0, "delta": msg, "finish_reason": None}]},
                        {"id": "m", "object": "chat.completion.chunk", "model": body.get("model", "m"),
                         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
                    ):
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return
                data = json.dumps({
                    "id": "x", "object": "chat.completion", "created": 0, "model": body.get("model", "m"),
                    "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}], "usage": usage,
                }).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _make_agent(base_url: str, session_id: str | None = None, session_db=None):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key", base_url=base_url, provider="custom", model="replay-model",
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[],
        max_iterations=3, session_id=session_id, session_db=session_db,
    )
    agent.compression_enabled = True
    cc = agent.context_compressor
    cc.context_length = CONTEXT_LENGTH
    cc.threshold_tokens = THRESHOLD
    calls: list[int] = []

    def counting(messages, system_message, **kw):
        calls.append(int(kw.get("approx_tokens") or 0))
        sys_prompt = system_message.get("content") if isinstance(system_message, dict) else system_message
        return messages, kw.get("active_system_prompt") or sys_prompt

    agent._compress_context = counting  # type: ignore[method-assign]
    agent._ab_compress_calls = calls
    return agent


def _seed_history(chars: int) -> list[dict]:
    """A prior transcript whose bytes/4 size is ``chars/4`` tokens."""
    return [
        {"role": "user", "content": "seed " + ("x" * chars)},
        {"role": "assistant", "content": "seeded"},
    ]


def _run_turns(wire: _FakeChat, agent_factory, history: list[dict], *, reload: bool, real: int) -> dict:
    wire.requests.clear()
    wire.usage_script[:] = [real] * (TURNS + 2)
    compress_by_turn: list[int] = []
    approx: list[int] = []
    agent = agent_factory()
    for t in range(TURNS):
        if reload:
            history = json.loads(json.dumps(history))
        before = len(agent._ab_compress_calls)
        r = agent.run_conversation(f"turn {t + 1}", conversation_history=history)
        assert r.get("completed"), r.get("error")
        compress_by_turn.append(len(agent._ab_compress_calls) - before)
        history = r["messages"]
        approx = list(agent._ab_compress_calls)
    return {
        "threshold": THRESHOLD, "real_prompt_tokens": real,
        "rough_estimate_turn1_messages": _rough(history),
        "local_compress_calls_by_turn": compress_by_turn,
        "local_compress_approx_tokens": approx,
        "provider_requests": len(wire.requests),
    }


def _rough(messages) -> int:
    from agent.model_metadata import estimate_messages_tokens_rough

    return estimate_messages_tokens_rough(messages)


def scenario(wire: _FakeChat, *, shape: str, real: int, chars: int, tmp: Path) -> dict:
    history = _seed_history(chars)
    if shape == "cli":
        return _run_turns(wire, lambda: _make_agent(wire.base_url), history, reload=False, real=real)
    if shape == "gateway":
        return _run_turns(wire, lambda: _make_agent(wire.base_url), history, reload=True, real=real)
    # restore: turn 1 in one process/agent, persisted; fresh agent + reopened DB for the rest.
    from hermes_state import SessionDB

    wire.requests.clear()
    wire.usage_script[:] = [real] * (TURNS + 2)
    sid = f"replay-{shape}-{real}-{chars}"
    db_path = tmp / f"state-{sid}.db"
    db = SessionDB(db_path=db_path)
    # The prior transcript is already durable (run_conversation treats ``conversation_history``
    # as persisted and flushes only the new rows).
    db.create_session(sid, source="cli")
    for m in history:
        db.append_message(sid, m["role"], m["content"])
    history = db.get_messages_as_conversation(sid)
    first = _make_agent(wire.base_url, session_id=sid, session_db=db)
    r1 = first.run_conversation("turn 1", conversation_history=history)
    assert r1.get("completed"), r1.get("error")
    calls_t1 = len(first._ab_compress_calls)
    db.close()
    reopened = SessionDB(db_path=db_path)
    persisted = reopened.get_messages_as_conversation(sid)
    compress_by_turn = [calls_t1]
    fresh = _make_agent(wire.base_url, session_id=sid, session_db=reopened)
    from agent.usage_anchor import restore_usage_anchor

    probe = _make_agent(wire.base_url, session_id=sid, session_db=reopened)
    restore_usage_anchor(probe, persisted)
    anchor_restored = probe._usage_anchor is not None
    hist = persisted
    for t in range(1, TURNS):
        before = len(fresh._ab_compress_calls)
        r = fresh.run_conversation(f"turn {t + 1}", conversation_history=hist)
        assert r.get("completed"), r.get("error")
        compress_by_turn.append(len(fresh._ab_compress_calls) - before)
        hist = r["messages"]
    reopened.close()
    return {
        "threshold": THRESHOLD, "real_prompt_tokens": real,
        "rough_estimate_turn1_messages": _rough(r1["messages"]),
        "persisted_messages": len(persisted),
        "anchor_restored_in_fresh_agent": anchor_restored,
        "local_compress_calls_by_turn": compress_by_turn,
        "local_compress_approx_tokens": list(first._ab_compress_calls) + list(fresh._ab_compress_calls),
        "provider_requests": len(wire.requests),
    }


def recovery_scenario(wire: _FakeChat, *, usage_less: bool) -> dict:
    """Real compression/retry, with only summary generation replaced by fixed local text."""
    from run_agent import AIAgent

    agent = _make_agent(wire.base_url)
    wire.requests.clear()
    wire.usage_script[:] = [None] * 8 if usage_less else ["overflow", REAL_UNDER]
    calls = []
    original = AIAgent._compress_context.__get__(agent, AIAgent)

    def compress(*args, **kwargs):
        calls.append(len(wire.requests))
        return original(*args, **kwargs)

    agent._compress_context = compress
    agent.context_compressor._generate_summary = lambda *a, **kw: "Local fixture summary of prior work."
    history = [{"role": "user" if i % 2 == 0 else "assistant",
                "content": f"record {i} " + "x" * 32_000} for i in range(20)]
    first = agent.run_conversation("continue", conversation_history=history)
    first_calls = len(calls)
    result = (agent.run_conversation("continue again", conversation_history=first["messages"])
              if usage_less else first)
    sizes = [len(json.dumps(r["messages"])) for r in wire.requests]
    return {"completed": result.get("completed"), "response": result.get("final_response"),
            "compression_after_provider_requests": calls, "first_turn_compressions": first_calls,
            "provider_request_sizes": sizes, "provider_omits_usage": agent.context_compressor._provider_omits_usage,
            "pass": bool(result.get("completed") and calls and calls[0] >= 1
                         and min(sizes[1:]) < sizes[0]
                         and (not usage_less or first_calls == 0))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="ab-token-accounting-"))
    os.environ["HERMES_HOME"] = str(tmp / "home")
    (tmp / "home").mkdir(parents=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    wire = _FakeChat()
    result: dict = {"checkout": str(ROOT), "head": head,
                    "compressor_sha256": hashlib.sha256(
                        (ROOT / "agent/context_compressor.py").read_bytes()).hexdigest()}
    try:
        for shape in ("cli", "gateway", "restore"):
            result[f"{shape}_past_window_inflated"] = scenario(
                wire, shape=shape, real=REAL_UNDER, chars=640_000, tmp=tmp)
            result[f"{shape}_inflated"] = scenario(wire, shape=shape, real=REAL_UNDER, chars=INFLATED_CHARS, tmp=tmp)
            result[f"{shape}_deflated"] = scenario(wire, shape=shape, real=REAL_OVER, chars=400, tmp=tmp)
        result["overflow_recovery"] = recovery_scenario(wire, usage_less=False)
        result["usage_less_fallback"] = recovery_scenario(wire, usage_less=True)
    finally:
        wire.close()
    verdict = {}
    for k, v in result.items():
        if not isinstance(v, dict):
            continue
        if "pass" in v:
            verdict[k] = "PASS" if v["pass"] else "FAIL"
            continue
        fired = sum(v["local_compress_calls_by_turn"])
        # inflated: real under threshold → no gate may fire. deflated: real over → must fire ≥1 after turn 1.
        verdict[k] = ("PASS" if fired == 0 else "FAIL") if k.endswith("inflated") else (
            "PASS" if sum(v["local_compress_calls_by_turn"][1:]) >= 1 else "FAIL")
    result["verdict"] = verdict
    Path(args.out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    return 0 if all(v == "PASS" for v in verdict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
