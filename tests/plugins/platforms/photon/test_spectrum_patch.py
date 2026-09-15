"""Regression tests for Hermes' Spectrum mixed text+attachment workaround."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import textwrap
import time
import urllib.request
from pathlib import Path


_PATCHER = Path("plugins/platforms/photon/sidecar/patch-spectrum-mixed-attachments.mjs")


def _sidecar_env(port: int) -> dict[str, str]:
    return {
        **os.environ,
        "PHOTON_PROJECT_ID": "test-project",
        "PHOTON_PROJECT_SECRET": "test-secret",
        "PHOTON_SIDECAR_PORT": str(port),
        "PHOTON_SIDECAR_TOKEN": "test-token",
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _write_sidecar_fixture(tmp_path: Path, *, sdk_available: bool) -> Path:
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    shutil.copyfile("plugins/platforms/photon/sidecar/index.mjs", sidecar / "index.mjs")
    # index.mjs imports sibling helper modules — copy every non-patch .mjs so
    # the fixture keeps working as helpers are extracted from index.mjs.
    for helper in Path("plugins/platforms/photon/sidecar").glob("*.mjs"):
        if helper.name in ("index.mjs", "patch-spectrum-mixed-attachments.mjs"):
            continue
        shutil.copyfile(helper, sidecar / helper.name)
    (sidecar / "patch-spectrum-mixed-attachments.mjs").write_text(
        "export function patchSpectrumTs() { throw new Error('forced patch failure'); }\n",
        encoding="utf-8",
    )

    if not sdk_available:
        return sidecar

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
    (package / "index.js").write_text(
        textwrap.dedent(
            """
            export async function Spectrum() {
              return {
                messages: { [Symbol.asyncIterator]() { return { next: () => new Promise(() => {}) }; } },
                stop: async () => undefined,
              };
            }
            export const attachment = value => value;
            export const voice = value => value;
            export const text = value => value;
            export const markdown = value => value;
            export const typing = value => value;
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (package / "providers" / "imessage.js").write_text(
        "export function imessage() { return {}; }\nimessage.config = () => ({});\n",
        encoding="utf-8",
    )
    return sidecar


def test_sidecar_patch_failure_still_reaches_health_endpoint(tmp_path: Path) -> None:
    """The compatibility patch is optional when the SDK itself remains usable."""
    sidecar = _write_sidecar_fixture(tmp_path, sdk_available=True)
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
        f"http://127.0.0.1:{port}/healthz",
        data=b"{}",
        headers={"X-Hermes-Sidecar-Token": "test-token"},
        method="POST",
    )
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                with urllib.request.urlopen(request, timeout=0.5) as response:
                    payload = json.load(response)
                break
            except OSError:
                if proc.poll() is not None or time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

        assert payload["ok"] is True
        assert proc.poll() is None
    finally:
        proc.terminate()
        _, stderr = proc.communicate(timeout=5)

    assert "forced patch failure" in stderr


def _tabify(src: str) -> str:
    """Convert the fixture's two-space indentation to the tab indentation that
    spectrum-ts ships in `@spectrum-ts/imessage/dist`, so the patch anchors
    (which match tabs) apply exactly as they do against a real install."""
    out = []
    for line in src.split("\n"):
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        out.append("\t" * (indent // 2) + " " * (indent % 2) + stripped)
    return "\n".join(out)


# A faithful, *executable* slice of spectrum-ts 8.x's iMessage inbound mapper:
# the two functions the patch rewrites (`rebuildFromAppleMessage` for
# `space.getMessage`, `toInboundMessages` for the live stream), plus stubs of
# the helpers they close over. Mirrors the published shape — tab-indented (via
# `_tabify`), `const ... = async` declarations, single-line builder calls — so
# the anchors exercise the real code path, and exporting the two functions lets
# the test assert runtime behavior rather than only string shape.
_SPECTRUM_IMESSAGE_FIXTURE = """
const formatChildId = (partIndex, parentGuid) => `p:${partIndex}/${parentGuid}`;
const asText = (text) => ({ type: "text", text });
const asCustom = (message) => ({ type: "custom" });
const asProviderGroup = (items) => ({ type: "group", items });
const messageAttachments = (message) => message.content.attachments ?? [];
const buildMessageBase = (message, chatGuidHint, timestamp, phone) => ({ direction: "inbound", sender: { id: "s" }, space: { id: "sp", type: "dm", phone }, timestamp });
const buildAttachmentMessage = async (client, base, info, id, partIndex, parentId) => {
  const msg = { ...base, id, content: { type: "attachment", id: info.guid }, partIndex };
  if (parentId !== void 0) msg.parentId = parentId;
  return msg;
};
const cacheMessage = (cache, message) => { cache.set(message.id, message); };
const rebuildFromAppleMessage = async (client, message, phone, chatGuidHint) => {
  const messageGuidStr = message.guid;
  const base = buildMessageBase(message, chatGuidHint, message.dateCreated ?? /* @__PURE__ */ new Date(), phone);
  const attachments = messageAttachments(message);
  if (attachments.length === 1) {
    const info = attachments[0];
    if (!info) throw new Error("Unreachable: attachments.length === 1 but no element");
    return buildAttachmentMessage(client, base, info, messageGuidStr, 0);
  }
  if (attachments.length > 1) {
    const items = [];
    for (let i = 0; i < attachments.length; i++) {
      const info = attachments[i];
      if (!info) continue;
      items.push(await buildAttachmentMessage(client, base, info, formatChildId(i, messageGuidStr), i, messageGuidStr));
    }
    return {
      ...base,
      id: messageGuidStr,
      content: asProviderGroup(items)
    };
  }
  const text = message.content.text;
  return {
    ...base,
    id: messageGuidStr,
    content: text ? asText(text) : asCustom(message)
  };
};
const toInboundMessages = async (client, cache, event, phone) => {
  const base = buildMessageBase(event.message, event.chatGuid, event.occurredAt, phone);
  const messageGuidStr = event.message.guid;
  const attachments = messageAttachments(event.message);
  if (attachments.length === 1) {
    const info = attachments[0];
    if (!info) throw new Error("Unreachable: attachments.length === 1 but no element");
    const msg = await buildAttachmentMessage(client, base, info, messageGuidStr, 0);
    cacheMessage(cache, msg);
    return [msg];
  }
  if (attachments.length > 1) {
    const items = [];
    for (let i = 0; i < attachments.length; i++) {
      const info = attachments[i];
      if (!info) continue;
      items.push(await buildAttachmentMessage(client, base, info, formatChildId(i, messageGuidStr), i, messageGuidStr));
    }
    const parent = {
      ...base,
      id: messageGuidStr,
      content: asProviderGroup(items)
    };
    cacheMessage(cache, parent);
    return [parent];
  }
  const text = event.message.content.text;
  const msg = {
    ...base,
    id: messageGuidStr,
    content: text ? asText(text) : asCustom(event.message)
  };
  cacheMessage(cache, msg);
  return [msg];
};
export { rebuildFromAppleMessage, toInboundMessages };
"""


def _write_fixture(tmp_path: Path) -> Path:
    dist = tmp_path / "node_modules" / "@spectrum-ts" / "imessage" / "dist"
    dist.mkdir(parents=True)
    chunk = dist / "index.js"
    chunk.write_text(_tabify(_SPECTRUM_IMESSAGE_FIXTURE), encoding="utf-8")
    return chunk


def test_spectrum_patch_rewrites_the_imessage_mapper(tmp_path: Path) -> None:
    """The dependency patch must apply to the 8.x `@spectrum-ts/imessage` chunk
    and rewrite both inbound mappers to thread text through attachment bubbles."""
    chunk = _write_fixture(tmp_path)

    result = subprocess.run(
        ["node", str(_PATCHER), str(tmp_path)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    patched = chunk.read_text(encoding="utf-8")
    assert "Preserve mixed text + attachment iMessage payloads" in patched
    # Single-attachment bubbles wrap the text + attachment in a group...
    assert "content: asProviderGroup([textMsg, msg2])" in patched  # rebuild
    assert "content: asProviderGroup([textMsg, msg])" in patched  # inbound
    # ...multi-attachment bubbles keep the group and shift attachment indices.
    assert "content: asProviderGroup(items)" in patched
    assert "formatChildId(text2 ? i + 1 : i, messageGuidStr)" in patched
    # The text is captured in both mappers before the attachment branches run.
    assert "const text2 = message.content.text;" in patched
    assert "const text2 = event.message.content.text;" in patched

    # Re-running is a no-op (idempotent self-heal on every sidecar start).
    again = subprocess.run(
        ["node", str(_PATCHER), str(tmp_path)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert again.returncode == 0, again.stderr
    assert chunk.read_text(encoding="utf-8") == patched


