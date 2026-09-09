"""Offline native cache-prefix regression (no inference or account credentials).

Run: python3 evals/directsdk_cache_wire.py /absolute/path/to/claude
Uses the real DirectSDK and native binary, with a synthetic loopback Messages peer.
Checks exact cached-prefix reuse, not cache hit rates; usage is deliberately synthetic.
Only compact hash receipts go to stdout. No raw traces or fixtures are retained.
"""
import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading


THINKING = {"type": "thinking", "thinking": "PUBLIC SYNTHETIC THINKING",
            "signature": "public-fixture-opaque-signature"}
TOOLS = [{"type": "function", "function": {"name": "probe", "description": "Public probe",
          "parameters": {"type": "object", "properties": {}}}}]
USAGE = {"input_tokens": 101, "output_tokens": 37, "cache_read_input_tokens": 211,
         "cache_creation_input_tokens": 313}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def nodes(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from nodes(item)


def content(value):
    # Directives move to the newest breakpoint; they are not cached prompt text.
    if isinstance(value, dict):
        return {key: content(item) for key, item in value.items() if key != "cache_control"}
    if isinstance(value, list):
        return [content(item) for item in value]
    return value


class Peer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.split("?")[0] == "/v1/messages/count_tokens":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"input_tokens":100}')
            return
        if self.path.split("?")[0] != "/v1/messages":
            self.send_error(403)
            return
        self.server.wires.append(body)
        number = len(self.server.wires)
        blocks = [THINKING, {"type": "tool_use", "id": f"toolu_public_{number}",
                  "name": body["tools"][0]["name"], "input": {}}] if self.server.tools else [
                      {"type": "text", "text": "PUBLIC SYNTHETIC ANSWER"}]
        stop = "tool_use" if self.server.tools else "end_turn"
        message = {"id": f"msg_public_{number}", "type": "message", "role": "assistant",
                   "model": body["model"], "content": blocks, "stop_reason": stop,
                   "stop_sequence": None, "usage": USAGE}
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def emit(event):
            self.wfile.write(("event: " + event["type"] + "\ndata: " + json.dumps(event) + "\n\n").encode())
            self.wfile.flush()

        emit({"type": "message_start", "message": {**message, "content": [], "stop_reason": None}})
        for index, block in enumerate(blocks):
            kind = block["type"]
            start = ({"type": kind, "thinking": "", "signature": ""} if kind == "thinking"
                     else {**block, "input": {}} if kind == "tool_use" else {"type": kind, "text": ""})
            emit({"type": "content_block_start", "index": index, "content_block": start})
            deltas = ([{"type": "thinking_delta", "thinking": block["thinking"]},
                       {"type": "signature_delta", "signature": block["signature"]}] if kind == "thinking"
                      else [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
                      if kind == "tool_use" else [{"type": "text_delta", "text": block["text"]}])
            for delta in deltas:
                emit({"type": "content_block_delta", "index": index, "delta": delta})
            emit({"type": "content_block_stop", "index": index})
        emit({"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": USAGE})
        emit({"type": "message_stop"})


def run(binary, model):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'plugins/model-providers/claude-oauth-directsdk'))
    spec = importlib.util.spec_from_file_location(
        "cache_directsdk", root / "plugins/model-providers/claude-oauth-directsdk/directsdk.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    with tempfile.TemporaryDirectory(prefix="directsdk-cache-") as tmp, ThreadingHTTPServer(("127.0.0.1", 0), Peer) as peer:
        peer.wires, peer.tools = [], True
        worker = threading.Thread(target=peer.serve_forever, daemon=True)
        worker.start()
        env = {"PATH": os.defpath, "HOME": tmp, "HERMES_HOME": tmp,
               "CLAUDE_CONFIG_DIR": str(Path(tmp) / "config"), "XDG_CONFIG_HOME": tmp,
               "ANTHROPIC_API_KEY": "sk-ant-public-offline-fixture",
               "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{peer.server_port}",
               "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_TELEMETRY": "1",
               "DISABLE_ERROR_REPORTING": "1", "NO_PROXY": "127.0.0.1,localhost",
               "no_proxy": "127.0.0.1,localhost"}
        env.update({key: "http://127.0.0.1:1" for key in
                    ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")})
        version = subprocess.check_output([str(binary), "--version"], env=env,
                                          stdin=subprocess.DEVNULL, text=True, timeout=15).strip()
        client = native.Client(command=str(binary), env=env, timeout=45)
        history = [{"role": "system", "content": "PUBLIC SYNTHETIC SYSTEM\n" * 300}]
        for i in range(12):
            history.extend([{"role": "user", "content": f"Public earlier question {i}"},
                            {"role": "assistant", "content": f"Public earlier answer {i}"}])
        history.append({"role": "user", "content": "Run the public probe."})
        receipts, previous = [], None
        try:
            for round_number in range(6):
                # Pure tool continuation, then ordinary user turns, then tools again.
                peer.tools = round_number not in (2, 3)
                original = copy.deepcopy(history)
                start = len(peer.wires)
                result = client.create(model=model, messages=history, tools=TOOLS)
                assert history == original, "caller history mutated"
                assert len(peer.wires) == start + 1, "unexpected extra inference request"
                wire = peer.wires[-1]
                markers = [node["cache_control"] for node in nodes(wire) if "cache_control" in node]
                message_markers = [(i, j) for i, msg in enumerate(wire["messages"])
                                   for j, block in enumerate(msg["content"]) if "cache_control" in block]
                assert len(markers) <= 4 and len(message_markers) == 1, "extra cache breakpoints"
                assert all(marker == markers[0] for marker in markers), "mixed native cache TTLs"
                prompt = content({key: wire[key] for key in ("tools", "system", "messages")})
                if previous is not None:
                    old, old_markers, (i, j) = previous
                    candidate = copy.deepcopy(prompt)
                    candidate["messages"] = candidate["messages"][:i + 1]
                    candidate["messages"][-1]["content"] = candidate["messages"][-1]["content"][:j + 1]
                    assert candidate == old, f"cached prefix changed at round {round_number}: {digest(old)} != {digest(candidate)}"
                    assert markers == old_markers, "native cache TTL/breakpoint policy changed"
                    receipts.append({"round": round_number, "old_prefix_sha256": digest(old),
                                     "replayed_prefix_sha256": digest(candidate)})
                i, j = message_markers[-1]
                prefix = copy.deepcopy(prompt)
                prefix["messages"] = prefix["messages"][:i + 1]
                prefix["messages"][-1]["content"] = prefix["messages"][-1]["content"][:j + 1]
                previous = prefix, markers, (i, j)
                if round_number in (1, 2, 5):
                    assert THINKING in list(nodes(wire["messages"])), "signed thinking lost or changed"
                assistant = result.choices[0].message.model_dump()
                history.append(assistant)
                if peer.tools:
                    history.extend({"role": "tool", "tool_call_id": call["id"],
                                    "content": "Public tool result\n" * 1000}
                                   for call in assistant["tool_calls"])
                else:
                    history.append({"role": "user", "content": f"Public followup {round_number}"})
            assert all("<total_tokens>" not in json.dumps(wire) for wire in peer.wires), "native budget reminder present"
            assert sum(len(msg["content"]) for msg in peer.wires[0]["messages"]) > 20
            return {"native_version": version, "native_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                    "requests": len(peer.wires), "synthetic_usage_not_cache_measurement": True,
                    "cache_controls": markers, "prefix_receipts": receipts}
        finally:
            client.close()
            peer.shutdown()
            worker.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--model", default="claude-sonnet-5")
    args = parser.parse_args()
    print(json.dumps(run(args.binary.resolve(), args.model), indent=2))
