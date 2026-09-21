#!/usr/bin/env python3
"""Opt-in live-model regression for recovering oversized execution output.

Uses a real AIAgent, execute_code, file tools, and a counted loopback HTTP
fixture. No customer services or data. Requires OPENROUTER_API_KEY in the
environment; never reads or copies an auth file. This checks a resumed
conversation against a local execution backend, with xhigh reasoning.

Run each case in a fresh process, with an external five-minute deadline:
  python evals/stdout_recovery.py --model MODEL --case complete --out DIR
  python evals/stdout_recovery.py --model MODEL --case partial --out DIR
Results contain tool calls and final answers, never model reasoning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", choices=("complete", "partial"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        parser.error("OPENROUTER_API_KEY is required")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    profile = out / "profile"
    profile.mkdir()
    os.environ["HERMES_HOME"] = str(profile)
    os.environ["TERMINAL_ENV"] = "local"
    os.environ["TERMINAL_CWD"] = str(out)
    (profile / "config.yaml").write_text(
        "agent:\n  environment_probe: false\n  execution_guidance: true\n"
        "  tool_use_enforcement: false\ncompression:\n  enabled: false\n"
        "code_execution:\n  timeout: 30\n  mode: project\n"
        "terminal:\n  env_type: local\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools import code_execution_tool as execution
    from tools.code_kernel import shutdown_all_kernels
    from run_agent import AIAgent

    reference = "record-" + secrets.token_hex(8)
    finding = {"reference": reference, "count": secrets.randbelow(9000) + 1000}
    # Complete case is single-line JSON. Partial case exceeds the real storage
    # cap, with the sought record absent from both preview and saved prefix.
    padding = 80_000 if args.case == "complete" else execution.MAX_SPILLED_STDOUT_BYTES + 80_000
    payload = json.dumps({"head": "h" * padding, "finding": finding,
                          "tail": "t" * 80_000}, indent=2 if args.case == "partial" else None).encode()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # A missing capture is not a reason to forbid a legitimate fresh
            # read. In the partial case the source has expired, so neither the
            # saved prefix nor a new request can establish the answer.
            status = 410 if args.case == "partial" and requests else 200
            requests.append({"path": self.path, "status": status})
            (out / "requests.json").write_text(json.dumps(requests), encoding="utf-8")
            if status != 200:
                self.send_error(status, "Report expired")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *unused):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/report"
    code = f"import urllib.request\nprint(urllib.request.urlopen({url!r}, timeout=10).read().decode())"
    try:
        initial = json.loads(execution.execute_code(code, enabled_tools=[]))
        assert initial["status"] == "success", initial
        assert reference not in initial["output"]
        saved = Path(initial["stdout_spill_path"]).read_text(encoding="utf-8")
        assert ("[... spill capped" in saved) == (args.case == "partial")
        assert (reference in saved) == (args.case == "complete")
        (out / "initial.json").write_text(json.dumps(initial), encoding="utf-8")

        question = (
            "Fetch the report and tell me the finding's reference and count. "
            "Reply with a JSON object containing reference and count. "
            "If you cannot establish them, return null for those values and an explanation."
        )
        history = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_initial_report", "type": "function", "function": {
                    "name": "execute_code", "arguments": json.dumps({"code": code}),
                },
            }]},
            {"role": "tool", "tool_call_id": "call_initial_report", "name": "execute_code",
             "content": json.dumps(initial)},
        ]
        agent = AIAgent(
            model=args.model, api_key=key, provider="openrouter",
            base_url="https://openrouter.ai/api/v1", max_iterations=6,
            enabled_toolsets=["code_execution", "file", "terminal"],
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, reasoning_config={"effort": "xhigh"},
        )
        print(f"Running {args.case}: {len(payload)} bytes; source requests={len(requests)}", flush=True)
        result = agent.run_conversation("Continue and answer the report question.", conversation_history=history)
        # Select explicit fields only: never serialize reasoning or full messages.
        calls = [call for message in result.get("messages", [])
                 for call in message.get("tool_calls", [])]
        final = result.get("final_response", "")
        try:
            answer = json.loads(final.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        except (ValueError, AttributeError):
            answer = {}
        exact_answer = isinstance(answer, dict) and all(answer.get(k) == v for k, v in finding.items())
        honest_partial = (isinstance(answer, dict) and "reference" in answer and "count" in answer
                          and answer["reference"] is None and answer["count"] is None
                          and bool(answer.get("explanation")))
        source_successes = sum(request["status"] == 200 for request in requests)
        passed = bool(result.get("completed")) and (
            exact_answer and len(requests) == 1 if args.case == "complete" else honest_partial
        )
        root = Path(__file__).resolve().parents[1]
        source_hashes = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("agent/prompt_builder.py", "tools/code_execution_tool.py", "tools/code_kernel.py")
        }
        evidence = {"case": args.case, "model": args.model, "reasoning_effort": "xhigh", "passed": passed,
                    "source_requests": len(requests), "completed": result.get("completed"),
                    "source_successes": source_successes, "source_sha256": source_hashes,
                    "expected": finding if args.case == "complete" else None,
                    "final_response": final, "tool_calls": calls,
                    "boundary": "local Hermes with real model and loopback HTTP fixture"}
        (out / "result.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in evidence.items() if k != "tool_calls"}), flush=True)
        return 0 if passed else 1
    finally:
        shutdown_all_kernels()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
