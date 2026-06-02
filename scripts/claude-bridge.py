#!/usr/bin/env python3
"""
Claude Code Bridge — Wraps `claude -p` as an OpenAI-compatible API endpoint.

Routes Hermes agent API calls through the Claude Code CLI using your existing
Claude Pro/Max subscription — zero API credits burned.

Architecture:
  Hermes profile (config.yaml: custom_provider → localhost:8800)
      ↓ HTTP POST /v1/chat/completions
  claude-bridge.py (this script)
      ↓ subprocess: claude -p --output-format json --model opus
  Claude Code CLI (authenticated via OAuth, subscription billing)

Usage:
  CLAUDE_BIN=/path/to/claude CLAUDE_HOME=/home/user PORT=8800 python3 claude-bridge.py

  All config is via environment variables:
    CLAUDE_BIN      — path to the `claude` CLI binary (default: finds in PATH)
    CLAUDE_HOME     — directory containing .claude/.credentials.json (required)
    PORT            — listen port (default: 8800)
    HOST            — bind address (default: 127.0.0.1)
    CLAUDE_MODEL    — model name for Claude Code (default: sonnet)
    TIMEOUT_SECONDS — subprocess timeout in seconds (default: 300)
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import subprocess
import os
import uuid
import time
import sys
import shutil


def _find_claude_bin() -> str:
    """Find the Claude Code CLI binary, preferring env var over PATH search."""
    bin_path = os.environ.get("CLAUDE_BIN", "")
    if bin_path and os.path.isfile(bin_path):
        return bin_path
    found = shutil.which("claude")
    if found:
        return found
    return "claude"  # will fail with a clear FileNotFoundError later


CLAUDE_BIN = _find_claude_bin()
CLAUDE_HOME = os.environ.get("CLAUDE_HOME", os.path.expanduser("~"))
PORT = int(os.environ.get("PORT", "8800"))
HOST = os.environ.get("HOST", "127.0.0.1")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "sonnet")

# Claude Code flags — print mode, non-interactive, structured JSON output
CLAUDE_FLAGS = [
    "--print",
    "--output-format", "json",
    "--model", CLAUDE_MODEL,
    "--no-session-persistence",
    "--dangerously-skip-permissions",
]

TIMEOUT_SECONDS = int(os.environ.get("TIMEOUT_SECONDS", "300"))


def build_prompt(messages: list) -> str:
    """Convert OpenAI-format messages into a single prompt for Claude Code."""
    parts = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Handle array content (multimodal / tool results)
        if isinstance(content, list):
            text_parts = []
            for c in content:
                t = c.get("type", "")
                if t == "text":
                    text_parts.append(c.get("text", ""))
                elif t == "tool_result":
                    text_parts.append(
                        f"[Tool Result: {c.get('tool_use_id', '')}]\n"
                        f"{c.get('content', '')}"
                    )
                elif t == "tool_use":
                    text_parts.append(
                        f"[Tool Call: {c.get('name', '')}]\n"
                        f"{json.dumps(c.get('input', {}), indent=2)}"
                    )
                elif t == "image_url":
                    text_parts.append(
                        f"[Image: {c.get('image_url', {}).get('url', '')}]"
                    )
            content = "\n".join(text_parts)

        content = content.strip()
        if not content:
            continue

        if role == "system":
            parts.append(content)
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")

    prompt = "\n\n".join(parts)

    # Prevent Claude Code from attempting tool use in -p mode
    prompt += (
        "\n\n---\n"
        "Respond directly to the user. Do not attempt to use any tools or "
        "filesystem operations — just provide your analysis and answer as text."
    )

    return prompt


class BridgeHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible /v1/chat/completions endpoint."""

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self._json_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._json_error(400, "Invalid JSON")
            return

        messages = data.get("messages", [])
        model = data.get("model", f"claude-{CLAUDE_MODEL}")
        stream = data.get("stream", False)

        if stream:
            self._json_error(501, "Streaming not supported — use non-streaming")
            return

        prompt = build_prompt(messages)
        cmd = [CLAUDE_BIN] + CLAUDE_FLAGS + [prompt]

        env = os.environ.copy()
        env["HOME"] = CLAUDE_HOME  # point Claude Code at OAuth credentials

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                env=env,
                cwd="/tmp",  # neutral cwd — avoids picking up project CLAUDE.md
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if not stdout and stderr:
                self._json_error(500, f"Claude Code error: {stderr[:500]}")
                return

            try:
                cc_output = json.loads(stdout)
            except json.JSONDecodeError:
                cc_output = {"result": stdout, "is_error": False}

            if cc_output.get("is_error", False):
                error_text = cc_output.get("result", stdout)[:500]
                self._json_error(500, error_text)
                return

            response_text = cc_output.get("result", stdout)
            usage = cc_output.get("usage", {})
            cost_usd = cc_output.get("total_cost_usd", 0)

            # Build OpenAI-format response
            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": (
                        usage.get("input_tokens", 0)
                        + usage.get("output_tokens", 0)
                    ),
                },
            }

            self._json_response(200, response)
            print(
                f"[{time.strftime('%H:%M:%S')}] ✓ {len(response_text)} chars, "
                f"${cost_usd:.4f} USD",
                flush=True,
            )

        except subprocess.TimeoutExpired:
            self._json_error(504, "Claude Code timed out")
        except FileNotFoundError:
            self._json_error(
                500,
                f"Claude binary not found at {CLAUDE_BIN}. "
                "Set CLAUDE_BIN or install with: npm install -g @anthropic-ai/claude-code",
            )
        except Exception as e:
            self._json_error(500, str(e))

    def do_GET(self):
        if self.path in ("/health", "/v1/health", "/"):
            self._json_response(
                200,
                {
                    "status": "ok",
                    "claude_bin": CLAUDE_BIN,
                    "claude_home": CLAUDE_HOME,
                    "model": CLAUDE_MODEL,
                },
            )
        else:
            self._json_error(404, "Not found")

    def _json_response(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_error(self, code, message):
        self._json_response(
            code,
            {
                "error": {
                    "message": message,
                    "type": "bridge_error",
                    "code": code,
                }
            },
        )

    def log_message(self, format, *args):
        """Suppress default HTTP logging to stdout."""
        pass


def main():
    print(f"🧠 Claude Code Bridge")
    print(f"   Listening: http://{HOST}:{PORT}/v1/chat/completions")
    print(f"   Claude:    {CLAUDE_BIN}")
    print(f"   Model:     {CLAUDE_MODEL} (subscription)")
    print(f"   Timeout:   {TIMEOUT_SECONDS}s")
    print()

    server = HTTPServer((HOST, PORT), BridgeHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        server.server_close()
        sys.exit(0)


if __name__ == "__main__":
    main()
