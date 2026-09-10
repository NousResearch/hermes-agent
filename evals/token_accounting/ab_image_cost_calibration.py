"""A/B probe: does the compaction trigger learn the real per-image cost from provider usage? (#70328)

Real ``AIAgent.run_conversation`` against a local fake chat-completions server whose
``usage.prompt_tokens`` prices every image at ``IMAGE_REAL`` tokens (a multimodal local model:
several thousand per screenshot) while the flat default is 1,500. A GUI loop appends one
screenshot per turn on a small window; the question is whether compaction fires BEFORE the real
prompt crosses the provider window (the fake returns a context-overflow 400 past it, like
llama.cpp), and what the estimator believes when it does.

    python evals/token_accounting/ab_image_cost_calibration.py --out /tmp/result.json
"""

from __future__ import annotations

import argparse
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

CONTEXT_LENGTH = 65_536
THRESHOLD = 40_000
IMAGE_REAL = 4_000
TEXT_PER_TURN = 400  # real tokens of text the provider sees per turn (system + user + reply)
TURNS = 14


def _count_images(messages) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            n += sum(1 for p in c if isinstance(p, dict) and p.get("type") in ("image_url", "image", "input_image"))
    return n


class _FakeVisionChat:
    """Prices a request as text_turns*TEXT_PER_TURN + images*IMAGE_REAL; 400s past the window."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.overflows = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):  # noqa: D401
                pass

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                msgs = body.get("messages") or []
                server.requests.append(body)
                images = _count_images(msgs)
                prompt = len(msgs) * TEXT_PER_TURN // 2 + images * IMAGE_REAL
                if os.environ.get("AB_DEBUG"):
                    sys.stderr.write(f"SERVER msgs={len(msgs)} images={images} prompt={prompt} roles={[m.get('role') for m in msgs]} types={[type(m.get('content')).__name__ for m in msgs]}\n")
                if prompt > CONTEXT_LENGTH:
                    server.overflows += 1
                    data = json.dumps({"error": {"message": f"request ({prompt} tokens) exceeds the available context size ({CONTEXT_LENGTH} tokens)", "type": "exceed_context_size"}}).encode()
                    self.send_response(400)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                usage = {"prompt_tokens": prompt, "completion_tokens": 5, "total_tokens": prompt + 5}
                if body.get("stream") is True:
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.end_headers()
                    for chunk in (
                        {"id": "m", "object": "chat.completion.chunk", "model": "m",
                         "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]},
                        {"id": "m", "object": "chat.completion.chunk", "model": "m",
                         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
                    ):
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return
                data = json.dumps({
                    "id": "x", "object": "chat.completion", "created": 0, "model": body.get("model", "m"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": usage,
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


def _screenshot_turn(i: int) -> list:
    return [{"type": "text", "text": f"screenshot {i}, click next"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 2_000}}]


def run(out_path: str) -> dict:
    from run_agent import AIAgent

    wire = _FakeVisionChat()
    agent = AIAgent(
        api_key="k", base_url=wire.base_url, provider="custom", model="vision-local-ab",
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[], max_iterations=3,
    )
    agent.compression_enabled = True
    cc = agent.context_compressor
    cc.context_length = CONTEXT_LENGTH
    cc.threshold_tokens = THRESHOLD
    compress_calls: list[dict] = []
    original = agent._compress_context

    def counting(messages, system_message, **kw):
        # Real compaction would need a summarizer; drop everything but the last 2 rows like one.
        compress_calls.append({"turn": len(wire.requests), "approx_tokens": int(kw.get("approx_tokens") or 0),
                               "images_in_history": _count_images(messages), "real_prompt_if_sent": len(messages) * TEXT_PER_TURN // 2 + _count_images(messages) * IMAGE_REAL})
        kept = [{"role": "user", "content": "[compressed summary]"}, {"role": "assistant", "content": "ok"}] + messages[-2:]
        sys_prompt = system_message.get("content") if isinstance(system_message, dict) else system_message
        return kept, kw.get("active_system_prompt") or sys_prompt

    agent._compress_context = counting  # type: ignore[method-assign]
    from agent.image_token_cost import current_image_token_cost, learned_image_token_cost
    history: list = []
    per_turn = []
    try:
        for i in range(TURNS):
            r = agent.run_conversation(_screenshot_turn(i), conversation_history=history)
            history = r["messages"]
            per_turn.append({"turn": i + 1, "completed": bool(r.get("completed")), "images": _count_images(history),
                             "provider_overflows_so_far": wire.overflows, "compress_calls_so_far": len(compress_calls)})
    finally:
        wire.close()
    # Compaction sizing: how many screenshots does the protected tail keep under the tail budget?
    # With images priced under their real cost the walk protects too many rows and the
    # "compacted" request is still over the window (the #70328 "cannot compress further" loop).
    from agent.image_token_cost import image_cost_context
    walk_history = [m for i in range(40) for m in ({"role": "user", "content": _screenshot_turn(i)}, {"role": "assistant", "content": "ok"})]
    tail_budget = int(THRESHOLD * cc.summary_target_ratio)
    with image_cost_context(learned_image_token_cost("vision-local-ab", wire.base_url)):
        cut = cc._find_tail_cut_by_tokens(walk_history, 0, token_budget=tail_budget)
    tail = walk_history[cut:]
    tail_real = _count_images(tail) * IMAGE_REAL + len(tail) * TEXT_PER_TURN // 2
    result = {
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "image_real_cost": IMAGE_REAL, "context_length": CONTEXT_LENGTH, "threshold": THRESHOLD,
        "learned_image_cost_after": learned_image_token_cost("vision-local-ab", wire.base_url),
        "provider_overflows": wire.overflows, "compress_calls": compress_calls, "per_turn": per_turn,
        "tail_budget": tail_budget, "tail_images_kept": _count_images(tail), "tail_real_tokens": tail_real,
        # The tail walk has an 8-row hard floor, so a screenshot-per-row tail can exceed the budget by
        # design; the invariant is that the walk's own accounting of that tail tracks the provider's.
        "tail_walk_estimate_error_pct": round(100 * (_count_images(tail) * learned_image_token_cost("vision-local-ab", wire.base_url) + len(tail) * TEXT_PER_TURN // 2 - tail_real) / tail_real, 1),
        "verdict": "PASS" if wire.overflows == 0 and compress_calls and abs(learned_image_token_cost("vision-local-ab", wire.base_url) - IMAGE_REAL) / IMAGE_REAL < 0.15 else "FAIL",
    }
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="ab-image-cost-"))
    os.environ["HERMES_HOME"] = str(tmp / "home")
    (tmp / "home").mkdir(parents=True)
    # An unknown custom model is treated as non-vision (images replaced by text); declare it.
    (tmp / "home" / "config.yaml").write_text("model:\n  supports_vision: true\n", encoding="utf-8")
    result = run(args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "per_turn"}, indent=2))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
