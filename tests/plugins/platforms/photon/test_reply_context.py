"""Sidecar reply-target normalization regression tests.

These run the real Photon sidecar with a tiny Spectrum-compatible fake. They
cover both Spectrum 8.2.2 target shapes: a hydrated target with direction and
text, and the unresolved stub used when the target cannot be fetched.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


_ROOT_SIDECAR = Path("plugins/platforms/photon/sidecar")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _sidecar_env(port: int) -> dict[str, str]:
    return {
        **os.environ,
        "PHOTON_PROJECT_ID": "test-project",
        "PHOTON_PROJECT_SECRET": "test-secret",
        "PHOTON_SIDECAR_PORT": str(port),
        "PHOTON_SIDECAR_TOKEN": "test-token",
        # The fixture intentionally leaves the stream open after one event.
        "PHOTON_STREAM_SILENCE_PROBE_MS": "0",
    }


def _write_sidecar_fixture(tmp_path: Path, target_kind: str) -> Path:
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    shutil.copyfile(_ROOT_SIDECAR / "index.mjs", sidecar / "index.mjs")
    for helper in _ROOT_SIDECAR.glob("*.mjs"):
        if helper.name in {"index.mjs", "patch-spectrum-mixed-attachments.mjs"}:
            continue
        shutil.copyfile(helper, sidecar / helper.name)
    (sidecar / "patch-spectrum-mixed-attachments.mjs").write_text(
        "export function patchSpectrumTs() { return { patched: false }; }\n",
        encoding="utf-8",
    )

    target_id = f"target-{target_kind}"
    target_text = "hydrated outbound text " + ("x" * 2100)
    target_direction = 'direction: "outbound",' if target_kind == "hydrated" else ""
    target_content = (
        {"type": "text", "text": target_text}
        if target_kind == "hydrated"
        else {"type": "custom", "imessage_type": "reply-target", "stub": True}
    )
    fake_sdk = f"""
const TARGET_KIND = {json.dumps(target_kind)};
const target = {{
  id: {json.dumps(target_id)},
  {target_direction}
  content: {json.dumps(target_content)},
}};
const space = {{
  id: "space-1",
  type: "dm",
  phone: "+15555551212",
  __platform: "iMessage",
  async getMessage(id) {{
    if (TARGET_KIND === "hydrated" && id === target.id) return target;
    return undefined;
  }},
}};
const message = {{
  id: "inbound-reply",
  direction: "inbound",
  sender: {{ id: "+15555551212" }},
  space,
  timestamp: new Date("2026-08-05T12:00:00.000Z"),
  content: {{
    type: "reply",
    content: {{ type: "text", text: "user reply" }},
    target,
  }},
}};
let emitted = false;
const messages = {{
  [Symbol.asyncIterator]() {{
    return {{
      async next() {{
        if (!emitted) {{
          emitted = true;
          return {{ done: false, value: [space, message] }};
        }}
        await new Promise(() => {{}});
      }},
    }};
  }},
}};
export async function Spectrum() {{
  return {{ messages, stop: async () => undefined }};
}}
export const attachment = value => value;
export const voice = value => value;
export const text = value => value;
export const markdown = value => value;
export const richlink = value => value;
export const typing = value => value;
export const poll = value => value;
"""
    package = sidecar / "node_modules" / "spectrum-ts"
    (package / "providers").mkdir(parents=True)
    (package / "package.json").write_text(
        json.dumps(
            {
                "name": "spectrum-ts",
                "type": "module",
                "exports": {
                    ".": "./index.js",
                    "./providers/imessage": "./providers/imessage.js",
                },
            }
        ),
        encoding="utf-8",
    )
    (package / "index.js").write_text(fake_sdk.lstrip(), encoding="utf-8")
    (package / "providers" / "imessage.js").write_text(
        "export function imessage() { return {}; }\nimessage.config = () => ({});\n",
        encoding="utf-8",
    )
    return sidecar


def _run_one_event(tmp_path: Path, target_kind: str) -> dict:
    sidecar = _write_sidecar_fixture(tmp_path, target_kind)
    port = _free_port()
    proc = subprocess.Popen(
        ["node", "index.mjs"],
        cwd=sidecar,
        env=_sidecar_env(port),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/inbound",
        headers={"X-Hermes-Sidecar-Token": "test-token"},
        method="GET",
    )
    response = None
    try:
        deadline = time.monotonic() + 5
        while response is None:
            try:
                response = urllib.request.urlopen(request, timeout=0.5)
            except (OSError, urllib.error.URLError):
                if proc.poll() is not None or time.monotonic() >= deadline:
                    raise
        raw = response.readline()
        if not raw:
            raise AssertionError(
                f"sidecar exited before emitting {target_kind} reply event"
            )
        return json.loads(raw)
    finally:
        if response is not None:
            response.close()
        proc.terminate()
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate(timeout=5)


def test_sidecar_normalizes_hydrated_reply_target(tmp_path: Path) -> None:
    event = _run_one_event(tmp_path, "hydrated")

    content = event["content"]
    assert event["replyToMessageId"] == "target-hydrated"
    assert event["replyToIsOwnMessage"] is True
    assert len(event["replyToText"]) == 2000
    assert content["type"] == "reply"
    assert content["targetMessageId"] == "target-hydrated"
    assert content["targetDirection"] == "outbound"
    assert content["targetText"] == event["replyToText"]
    assert content["content"] == {"type": "text", "text": "user reply"}


def test_sidecar_preserves_stub_reply_target_for_adapter_fallback(
    tmp_path: Path,
) -> None:
    event = _run_one_event(tmp_path, "stub")

    content = event["content"]
    assert event["replyToMessageId"] == "target-stub"
    assert event["replyToText"] is None
    assert event["replyToIsOwnMessage"] is False
    assert content["type"] == "reply"
    assert content["targetMessageId"] == "target-stub"
    assert content["targetDirection"] is None
    assert content["targetText"] is None
    assert content["content"] == {"type": "text", "text": "user reply"}
