"""Behavior tests for the Photon sidecar's native-reply dispatch."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict

_MODULE = Path("plugins/platforms/photon/sidecar/reply-target.mjs").resolve()


def _run_reply_harness(reply_to: str | None, cache_target: bool) -> Dict[str, Any]:
    harness = f"""
import {{ sendWithReply }} from {json.dumps(_MODULE.as_uri())};
const calls = [];
const space = {{ send: async builder => {{ calls.push(["send", builder]); return {{ id: "sent" }}; }} }};
const target = {{ reply: async builder => {{ calls.push(["reply", builder]); return {{ id: "replied" }}; }} }};
const knownMessages = new Map();
if ({json.dumps(cache_target)}) knownMessages.set("incoming-123", target);
const result = await sendWithReply(
  space,
  {{ type: "attachment" }},
  {json.dumps(reply_to)},
  knownMessages,
);
console.log(JSON.stringify({{ calls, result }}));
"""
    run = subprocess.run(
        ["node", "--input-type=module", "-e", harness],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def test_cached_target_uses_native_reply() -> None:
    result = _run_reply_harness("incoming-123", cache_target=True)

    assert result == {
        "calls": [["reply", {"type": "attachment"}]],
        "result": {"id": "replied"},
    }


def test_missing_or_expired_target_falls_back_to_space_send() -> None:
    result = _run_reply_harness("incoming-123", cache_target=False)

    assert result == {
        "calls": [["send", {"type": "attachment"}]],
        "result": {"id": "sent"},
    }


def test_proactive_send_uses_space_send() -> None:
    result = _run_reply_harness(None, cache_target=True)

    assert result["calls"] == [["send", {"type": "attachment"}]]
