"""A copy of the Solstice plugin installed in a profile's own ``$HERMES_HOME/plugins/model-providers/`` loads
under a top-level module name, so its transport may not reach a sibling plugin through a relative import.

Real discovery in a fresh interpreter (the registry and ``sys.modules`` stay untouched here), then
``create_client(...).chat.completions.create`` against a fake httpx transport."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

_SCRIPT = r"""
import json, sys
import httpx
import providers

profile = providers.get_provider_profile("solstice")
assert type(profile).__module__.startswith("_hermes_user_provider_"), type(profile).__module__
assert providers.plugin_load_error("solstice") == "", providers.plugin_load_error("solstice")
sent = []

def handler(request):
    sent.append(json.loads(request.content))
    if sent[-1]["stream"]:
        event = {"event_type": "interaction.completed", "interaction": {"id": "i1", "status": "completed", "usage": {}}}
        return httpx.Response(200, content=f"data: {json.dumps(event)}\n\n".encode(),
                              headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json={"id": "i1", "status": "completed", "usage": {},
                                     "steps": [{"type": "model_output", "content": [{"type": "text", "text": "hello"}]}]})

client = profile.create_client(api_key="ya29.test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
messages = [{"role": "user", "content": "hi"}]
unary = client.chat.completions.create(model="gemini-4-hydrogen", messages=messages)
chunks = list(client.chat.completions.create(model="gemini-4-hydrogen", messages=messages, stream=True))
print(json.dumps({"module": type(client).__module__, "text": unary.choices[0].message.content,
                  "finish": chunks[-1].choices[0].finish_reason, "store": [b["store"] for b in sent]}))
"""


def test_a_profile_local_solstice_plugin_creates_a_client_and_completes_a_turn(tmp_path):
    home = tmp_path / "home"
    shutil.copytree(ROOT / "plugins" / "model-providers" / "solstice", home / "plugins" / "model-providers" / "solstice",
                    ignore=shutil.ignore_patterns("__pycache__"))
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONDONTWRITEBYTECODE": "1"}

    result = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=ROOT, env=env, capture_output=True, text=True,
                            timeout=120, check=False)

    assert result.returncode == 0, result.stderr[-3000:]
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["module"].startswith("_hermes_user_provider_") and out["module"].endswith(".transport")
    assert out == {**out, "text": "hello", "finish": "stop", "store": [False, False]}
