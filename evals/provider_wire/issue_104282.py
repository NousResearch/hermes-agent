"""Live localhost Responses adapter video rejection/control probe."""

import os, sys, tempfile, json, threading
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

root = Path(sys.argv[1]).resolve()
arm = sys.argv[2]
home = tempfile.mkdtemp(prefix="video-wire-")
os.environ.clear()
os.environ.update(HOME=home, HERMES_HOME=home, PATH="/usr/bin:/bin", NO_PROXY="*")
sys.path.insert(0, str(root))
os.chdir(home)
sys.dont_write_bytecode = True
captures = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        captures.append(body)
        response = {
            "id": "resp_fixture",
            "object": "response",
            "status": "completed",
            "model": "gpt-5.5",
            "output": [
                {
                    "type": "message",
                    "id": "msg_fixture",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "local fixture response",
                            "annotations": [],
                        }
                    ],
                }
            ],
        }
        events = [
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": response["output"][0],
            },
            {"type": "response.completed", "response": response},
        ]
        data = "".join(
            "event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n" for e in events
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
from openai import OpenAI
from agent.auxiliary_client import _CodexCompletionsAdapter
from agent.codex_responses_adapter import _preflight_codex_input_items

client = OpenAI(
    api_key="fixture",
    base_url=f"http://127.0.0.1:{server.server_port}/v1",
    max_retries=0,
)
adapter = _CodexCompletionsAdapter(client, "gpt-5.5")
outcomes = {}
for kind in ["video_url", "video", "input_video", "image_url"]:
    key = "image_url" if kind == "image_url" else kind
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": kind,
                    key: {
                        "url": "data:video/mp4;base64,AAAA"
                        if kind != "image_url"
                        else "data:image/png;base64,AAAA"
                    },
                },
                {"type": "text", "text": "Describe the input"},
            ],
        }
    ]
    prior = len(captures)
    try:
        result = adapter.create(messages=messages)
        outcomes[kind] = {
            "outcome": "success",
            "content": result.choices[0].message.content,
            "http_calls": len(captures) - prior,
        }
    except ValueError as e:
        outcomes[kind] = {
            "outcome": "rejected",
            "error": str(e),
            "http_calls": len(captures) - prior,
        }
    expected = "rejected" if arm == "fixed" and kind != "image_url" else "success"
    assert outcomes[kind]["outcome"] == expected, outcomes
assert captures[-1]["input"][0]["content"][0]["type"] == "input_image"
assert outcomes["image_url"]["content"] == "local fixture response"
print(
    json.dumps(
        {
            "repo": str(root),
            "outcomes": outcomes,
            "captures": captures,
            "adapter_module": sys.modules["agent.auxiliary_client"].__file__,
        },
        indent=2,
    )
)
client.close()
server.shutdown()
