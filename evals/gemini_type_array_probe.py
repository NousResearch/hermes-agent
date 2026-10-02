"""Offline A/B native adapter + Google SDK schema validation (no API calls).

Run with httpx and google-genai==1.47.0 installed:
    python evals/gemini_type_array_probe.py --base origin/main
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class WireCapture(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        size = int(self.headers["Content-Length"])
        self.requests.append(json.loads(self.rfile.read(size)))
        # Intentionally stop at the wire boundary; never fake a model response.
        self.send_response(418)
        self.end_headers()
        self.wfile.write(b"Local schema capture only")

    def log_message(self, *_):
        pass


def wire_translate(adapter, tools):
    server = ThreadingHTTPServer(("127.0.0.1", 0), WireCapture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with adapter.GeminiNativeClient(
            api_key="local-probe-not-a-secret",
            base_url=f"http://127.0.0.1:{server.server_port}/v1beta",
        ) as client:
            try:
                client.chat.completions.create(
                    model="gemini-2.5-flash", tools=tools,
                    messages=[{"role": "user", "content": "Schema validation probe"}],
                )
            except adapter.GeminiAPIError as exc:
                assert exc.status_code == 418
        return WireCapture.requests.pop()["tools"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def probe(label, translate):
    from google.genai.types import FunctionDeclaration

    cases = {
        "nullable": {"type": ["number", "null"]},
        "nullable_enum": {"type": ["integer", "null"], "enum": [1, 2]},
        "union_enum": {"type": ["string", "integer"], "enum": ["one", 2]},
        "nested_items": {"type": "array", "items": {"type": ["string", "null"]}},
        "nested_anyof": {"anyOf": [{"type": ["string", "integer"]}, {"type": "boolean"}]},
        "constrained_union": {"type": ["string", "integer"], "anyOf": [{"enum": ["one"]}, {"enum": ["two"]}]},
        "structural_union": {"type": ["array", "object", "string"], "items": {"type": "integer"}, "properties": {"name": {"type": "string"}}, "required": ["name"]},
        "scalar_control": {"type": "number"},
    }
    results = {}
    for name, node in cases.items():
        parameters = {"type": "object", "properties": {"value": node}}
        original = copy.deepcopy(parameters)
        try:
            wire = translate([{"type": "function", "function": {"name": "probe", "parameters": parameters}}])
            declaration = wire[0]["functionDeclarations"][0]
            # Validate the actual adapter declaration, not a reconstructed schema.
            FunctionDeclaration.model_validate(json.loads(json.dumps(declaration)))
            results[name] = "accepted"
        except (TypeError, ValueError) as exc:
            results[name] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        assert parameters == original, name
    print(json.dumps({"label": label, "results": results}, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    for name in list(sys.modules):
        if name == "agent" or name.startswith(("agent.", "tools", "hermes")):
            del sys.modules[name]
    with tempfile.TemporaryDirectory(prefix="gemini-schema-home-") as home:
        os.environ["HERMES_HOME"] = home
        adapter = importlib.import_module("agent.gemini_native_adapter")
        baseline = types.ModuleType("baseline_gemini_schema")
        source = subprocess.check_output(
            ["git", "show", f"{args.base}:agent/gemini_schema.py"], cwd=root, text=True,
        )
        exec(compile(source, f"{args.base}:agent/gemini_schema.py", "exec"), baseline.__dict__)
        fixed = adapter.sanitize_gemini_tool_parameters
        adapter.sanitize_gemini_tool_parameters = baseline.sanitize_gemini_tool_parameters
        before = probe(args.base, lambda tools: wire_translate(adapter, tools))
        adapter.sanitize_gemini_tool_parameters = fixed
        after = probe("worktree", lambda tools: wire_translate(adapter, tools))
        assert before["scalar_control"] == "accepted"
        assert all(result != "accepted" for name, result in before.items() if name != "scalar_control")
        assert all(result == "accepted" for result in after.values())
        print(f"PASS: {len(before) - 1} failing base cases repaired; scalar control and inputs preserved.")
        print("LOCAL SDK VALIDATION ONLY: no Google API call or Windows Desktop exercise.")


if __name__ == "__main__":
    main()
