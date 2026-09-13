"""Credential-free production CLI PTY / live slash display A/B.

Run with the checkout's Python and --out DIRECTORY. Provider counts below are
explicit fixture inputs, NOT measurements from a vendor. No inference is made.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pty
import sys
import tempfile


def child(out: Path) -> None:
    from cli import HermesCLI
    from run_agent import AIAgent
    from agent.context_breakdown import compute_session_context_breakdown
    from tui_gateway.server import _get_usage, _format_live_context_output
    import threading
    import asyncio
    from gateway.run import GatewayRunner
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore, SessionSource
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(Path(os.environ["HERMES_HOME"]) / "sessions", runner.config)
    runner._session_db = None
    runner.adapters = {}
    source = SessionSource(platform=Platform.TELEGRAM, user_id="fixture", chat_id="fixture", chat_type="dm")
    entry = runner.session_store.get_or_create_session(source)

    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class ProviderFixture(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            request_file = "provider-request-unmetered.json" if getattr(self.server, "omit_usage", False) else "provider-request.json"
            (out / request_file).write_text(json.dumps(request, indent=2))
            response = {"id": "fixture", "object": "chat.completion", "created": 0, "model": "fixture",
                        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "fixture answer"}}],
                        "usage": {"prompt_tokens": 1234, "completion_tokens": 20, "total_tokens": 1254}}
            if getattr(self.server, "omit_usage", False):
                response.pop("usage")
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), ProviderFixture)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    cli = HermesCLI(model="fixture", provider="openai-compat", api_key="fixture", base_url=base_url)
    agent = AIAgent(model="fixture", provider="openai-compat", api_key="fixture", base_url=base_url, enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False)
    agent.context_compressor._config_context_length = 100_000
    agent.context_compressor._resolved_context_length = 100_000
    agent._disable_streaming = True
    cli.agent = agent
    cli.conversation_history = [{"role": "user", "content": "fixture question"}]
    results = {}
    for scenario in ("local_estimate", "provider_usage", "provider_usage_plus_estimate"):
        if scenario == "local_estimate":
            agent.context_compressor.maybe_seed_preflight_display_tokens(1234)
        elif scenario == "provider_usage":
            result = agent.run_conversation("fixture question")
            assert result["completed"] and result["last_prompt_tokens"] == 1234
            cli.conversation_history = result["messages"]
        else:
            cli.conversation_history.append({"role": "user", "content": "new unpriced question"})
        agent._session_messages = cli.conversation_history
        print(f"\n=== {scenario}: CLI /context ===", flush=True)
        cli.process_command("/context")
        print("CLI status bar:", cli._build_status_bar_text(width=120), flush=True)
        usage = _get_usage(agent)
        session = {"agent": agent, "history": cli.conversation_history, "history_lock": threading.RLock()}
        print("=== TUI/Desktop live /context ===", flush=True)
        print(_format_live_context_output("fixture", session, ""), flush=True)
        runner._running_agents = {entry.session_key: agent}
        runner.session_store.rewrite_transcript(entry.session_id, cli.conversation_history)
        print("=== gateway /status ===", flush=True)
        print(asyncio.run(runner._handle_status_command(MessageEvent(text="/status", source=source, message_id="fixture"))), flush=True)
        print("=== gateway /context ===", flush=True)
        print(asyncio.run(runner._handle_context_command(MessageEvent(text="/context", source=source, message_id="fixture"))), flush=True)
        print("=== gateway /usage category block ===", flush=True)
        print("\n".join(runner._context_breakdown_lines(agent, source)), flush=True)
        results[scenario] = {"breakdown": compute_session_context_breakdown(agent, cli.conversation_history), "usage": usage}
    (out / "payloads.json").write_text(json.dumps(results, indent=2))
    server.omit_usage = True
    unmetered = AIAgent(model="fixture", provider="openai-compat", api_key="fixture", base_url=base_url, enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False)
    unmetered.context_compressor._config_context_length = 100_000
    unmetered.context_compressor._resolved_context_length = 100_000
    unmetered.context_compressor.maybe_seed_preflight_display_tokens(1234)
    unmetered._disable_streaming = True
    unmetered_result = unmetered.run_conversation("fixture without usage")
    assert unmetered_result["completed"]
    (out / "unmetered-result.json").write_text(json.dumps({"last_prompt_tokens": unmetered_result["last_prompt_tokens"]}))
    server.shutdown()
    server.server_close()
    print("DISPLAY_PROBE_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="context-display-") as home:
        pid, fd = pty.fork()
        if pid == 0:
            os.environ.clear()
            os.environ.update(HOME=home, HERMES_HOME=home + "/.hermes", PATH="/usr/bin:/bin", TERM="dumb", NO_COLOR="1")
            try:
                child(out)
            finally:
                sys.stdout.flush()
            os._exit(0)
        chunks = []
        while True:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            chunks.append(chunk)
        os.close(fd)
        _, status = os.waitpid(pid, 0)
        transcript = b"".join(chunks).decode(errors="replace")
        (out / "pty.txt").write_text(transcript)
        print(transcript)
        if status or "DISPLAY_PROBE_COMPLETE" not in transcript:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
