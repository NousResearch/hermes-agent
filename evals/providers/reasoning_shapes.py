"""Local HTTP/SDK reasoning-shape probe; no vendor inference or credentials.

Run: python evals/providers/reasoning_shapes.py --output /tmp/reasoning.json
Run the same file in a fresh interpreter on base and fix checkouts for A/B.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


CASES = {
    "string": ("**First**", "**Second**", "**First****Second**"),
    "text-parts": ([{"type": "text", "text": "**First**"}],
                   [{"type": "text", "text": "**Second**"}], "**First****Second**"),
    "split-fragments": (["Hel", "lo"], [" wor", "ld"], "Hello world"),
    "dict": ({"type": "text", "text": "Hello"}, {"text": " world"}, "Hello world"),
    "empty": ([], None, ""),
    "nontext-parts": ([{"type": "image_url", "image_url": {"url": "ignored"}}], [], ""),
}


def run_matrix(surfaces):
    from openai import AsyncOpenAI, OpenAI
    from run_agent import AIAgent
    from agent.agent_runtime_helpers import extract_reasoning
    from agent.auxiliary_client import _aggregate_chat_stream, _aggregate_chat_stream_async
    from agent.chat_completion_helpers import interruptible_streaming_api_call
    from agent.chat_completion_helpers_relay import RelayChatAccumulator

    active = {"message": {}, "deltas": [], "answer": "LOCAL_CAPTURE_OK"}
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "body": request})
            self.send_response(200)
            stream = request.get("stream", False)
            self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
            self.end_headers()
            common = {"id": "local", "created": 1, "model": "probe"}
            if stream:
                for delta in active["deltas"] + [{"content": active["answer"]}]:
                    chunk = {**common, "object": "chat.completion.chunk", "choices": [
                        {"index": 0, "delta": delta, "finish_reason": None}]}
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                chunk = {**common, "object": "chat.completion.chunk", "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}]}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())
            else:
                message = {"role": "assistant", "content": "LOCAL_CAPTURE_OK", **active["message"]}
                response = {**common, "object": "chat.completion", "choices": [
                    {"index": 0, "message": message, "finish_reason": "stop"}]}
                self.wfile.write(json.dumps(response).encode())
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    rows = []
    client = OpenAI(base_url=url, api_key="local-fixture", max_retries=0)
    agent = AIAgent(base_url=url, api_key="local-fixture", provider="custom", model="probe",
                    api_mode="chat_completions", enabled_toolsets=[], quiet_mode=True,
                    skip_memory=True, skip_context_files=True, skip_background_review=True)
    kwargs = {"model": "probe", "messages": [{"role": "user", "content": "probe"}]}

    async def aux_async():
        async with AsyncOpenAI(base_url=url, api_key="local-fixture", max_retries=0) as async_client:
            stream = await async_client.chat.completions.create(**kwargs, stream=True)
            return await _aggregate_chat_stream_async(stream, model="probe")

    try:
        for surface in surfaces:
            fields = ("reasoning", "reasoning_content")
            if surface == "nonstream":
                fields += tuple(f"reasoning_details.{key}" for key in ("summary", "thinking", "content", "text"))
            for field in fields:
                for name, (first, second, plain) in CASES.items():
                    active["deltas"] = [{field: first}, {field: second}]
                    active["answer"] = ([{"type": "text", "text": "LOCAL_"}, "CAPTURE_OK"]
                                        if name == "split-fragments" else "LOCAL_CAPTURE_OK")
                    # Completed responses exercise one field containing the same fragments.
                    active["message"] = ({"reasoning_details": [{field.split(".")[1]: first}]}
                                         if "." in field else {field: first})
                    callbacks = []
                    agent.reasoning_callback = callbacks.append
                    expected = plain
                    if surface in ("main", "relay") and name in ("string", "text-parts"):
                        expected = "**First**\n\n**Second**"
                    if surface == "nonstream":
                        expected = {"string": "**First**", "text-parts": "**First**",
                                    "split-fragments": "Hello", "dict": "Hello",
                                    "empty": "", "nontext-parts": ""}[name]
                    row = {"surface": surface, "field": field, "case": name, "expected": expected}
                    try:
                        if surface == "main":
                            response = interruptible_streaming_api_call(agent, dict(kwargs))
                            message = response.choices[0].message
                            reasoning = extract_reasoning(agent, message) or ""
                            row["callbacks"] = callbacks
                        elif surface == "relay":
                            acc = RelayChatAccumulator()
                            with client.chat.completions.create(**kwargs, stream=True) as stream:
                                for chunk in stream:
                                    acc.observe(chunk.model_dump(warnings=False))
                            message = acc.finalize()["choices"][0]["message"]
                            reasoning = message["reasoning_content"] or ""
                        elif surface == "nonstream":
                            message = client.chat.completions.create(**kwargs).choices[0].message
                            reasoning = extract_reasoning(agent, message) or ""
                        else:
                            if surface == "aux-sync":
                                stream = client.chat.completions.create(**kwargs, stream=True)
                                response = _aggregate_chat_stream(stream, model="probe")
                            else:
                                response = asyncio.run(aux_async())
                            message = response.choices[0].message
                            reasoning = message.reasoning or ""
                        content = message["content"] if isinstance(message, dict) else message.content
                        row.update(reasoning=reasoning, content=content)
                        row["ok"] = reasoning == expected and content == "LOCAL_CAPTURE_OK"
                        if surface == "main":
                            row["ok"] &= all(isinstance(value, str) for value in callbacks)
                            if row["ok"]:
                                row["ok"] &= "".join(callbacks) == expected
                    except Exception as error:
                        row.update(ok=False, error=f"{type(error).__name__}: {error}")
                    rows.append(row)
    finally:
        client.close()
        agent.client.close()
        server.shutdown()
        server.server_close()
        thread.join()
    return {"fidelity": "Local HTTP fixture + real SDK + production consumers; not vendor inference",
            "module": str(Path(__file__).resolve()), "requests": requests, "results": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for key in list(os.environ):
        if any(token in key.upper() for token in ("TOKEN", "SECRET", "API_KEY", "AUTH")):
            os.environ.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="reasoning-wire-") as home:
        os.environ.update(HOME=home, HERMES_HOME=home)
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        result = run_matrix(("main", "relay", "aux-sync", "aux-async", "nonstream"))
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(result["results"]),
                      "passed": sum(row["ok"] for row in result["results"]),
                      "output": args.output}))


if __name__ == "__main__":
    main()
