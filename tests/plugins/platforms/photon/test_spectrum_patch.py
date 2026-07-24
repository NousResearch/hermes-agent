"""Regression tests for Hermes' Spectrum mixed text+attachment workaround."""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path


_PATCHER = Path("plugins/platforms/photon/sidecar/patch-spectrum-mixed-attachments.mjs")


def test_sidecar_applies_spectrum_patch_before_importing_sdk() -> None:
    """Existing installs should self-heal at runtime, not only during npm postinstall."""
    index = Path("plugins/platforms/photon/sidecar/index.mjs").read_text(encoding="utf-8")
    assert "import { patchSpectrumTs }" in index
    assert "patchSpectrumTs();" in index
    assert index.index("patchSpectrumTs();") < index.index('await import("spectrum-ts")')


def test_sidecar_healthz_reports_stream_health() -> None:
    """Local process health must include upstream stream health."""
    index = Path("plugins/platforms/photon/sidecar/index.mjs").read_text(encoding="utf-8")
    assert "function streamHealthSnapshot()" in index
    assert 'return ok(res, { stream: streamHealthSnapshot() });' in index
    assert "STREAM_INTERRUPTED_DEGRADE_COUNT" in index
    assert "process.exit(75);" in index


def test_sidecar_exposes_richlink_send_endpoint() -> None:
    """The loopback endpoint should wrap spectrum-ts' richlink() builder."""
    index = Path("plugins/platforms/photon/sidecar/index.mjs").read_text(encoding="utf-8")
    assert "richlink: spectrumRichlink" in index
    assert 'if (req.url === "/send-richlink")' in index
    assert "isHttpUrl(url)" in index
    assert "space.send(spectrumRichlink(url.trim()))" in index


def test_sidecar_intercepts_both_console_channels() -> None:
    """spectrum-ts routes its stream telemetry through @photon-ai/otel, which
    sends severity >= ERROR to console.error and WARN/INFO to console.log.
    The two lines the health monitor keys off land on *different* channels:
    `log.error("stream persistently failing")` -> console.error, but
    `log.warn("stream interrupted; reconnecting")` -> console.log. Patching
    only console.error would miss every interrupt burst (the primary silent-
    inbound symptom), so both channels must be intercepted.
    """
    index = Path("plugins/platforms/photon/sidecar/index.mjs").read_text(encoding="utf-8")
    assert "function classifyStreamLog(" in index
    assert "console.error = (...args) =>" in index
    assert "console.log = (...args) =>" in index
    # Both wrappers must feed the shared classifier.
    assert index.count("classifyStreamLog(text)") >= 2


def test_sidecar_labels_catchup_internal_errors_as_upstream_photon() -> None:
    """Photon cloud stream failures should not look like local auth problems."""
    index = Path("plugins/platforms/photon/sidecar/index.mjs").read_text(encoding="utf-8")
    assert "function inboundStreamErrorMessage" in index
    assert "EventService/CatchUpEvents" in index
    assert "this is upstream of Hermes" in index
    assert "PHOTON_ALLOWED_USERS" in index


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


def test_spectrum_patch_preserves_text_at_runtime(tmp_path: Path) -> None:
    """Execute the patched mappers and assert mixed bubbles become groups whose
    first child is the typed text, while text-free bubbles keep their exact
    original shape (id/partIndex/parentId) so message identity is unchanged."""
    chunk = _write_fixture(tmp_path)
    patch = subprocess.run(
        ["node", str(_PATCHER), str(tmp_path)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert patch.returncode == 0, patch.stderr

    harness = textwrap.dedent(
        f"""
        import {{ rebuildFromAppleMessage, toInboundMessages }} from {str(chunk)!r};
        const assert = (c, m) => {{ if (!c) {{ console.error("FAIL: " + m); process.exit(1); }} }};

        // Mixed text + single attachment -> group [text@0, attachment@1].
        let r = await rebuildFromAppleMessage(null, {{ guid: "G", content: {{ text: "hello", attachments: [{{ guid: "A0" }}] }} }}, "+1");
        assert(r.content.type === "group" && r.id === "G", "single+text -> group parent id=guid");
        assert(r.content.items.length === 2, "two items");
        assert(r.content.items[0].content.type === "text" && r.content.items[0].content.text === "hello" && r.content.items[0].partIndex === 0 && r.content.items[0].id === "p:0/G", "text child @0");
        assert(r.content.items[1].content.type === "attachment" && r.content.items[1].partIndex === 1 && r.content.items[1].id === "p:1/G" && r.content.items[1].parentId === "G", "attachment child @1");

        // Single attachment, no text -> unchanged bare attachment.
        r = await rebuildFromAppleMessage(null, {{ guid: "G", content: {{ text: "", attachments: [{{ guid: "A0" }}] }} }}, "+1");
        assert(r.content.type === "attachment" && r.id === "G" && r.partIndex === 0 && r.parentId === undefined, "no-text single attachment unchanged");

        // Multi attachment + text via the live stream -> group [text@0, att@1, att@2].
        let arr = await toInboundMessages(null, new Map(), {{ message: {{ guid: "G2", content: {{ text: "cap", attachments: [{{ guid: "A0" }}, {{ guid: "A1" }}] }} }} }}, "+1");
        assert(arr.length === 1 && arr[0].content.type === "group", "multi+text -> single group");
        let items = arr[0].content.items;
        assert(items.length === 3 && items[0].content.type === "text" && items[0].partIndex === 0, "text first @0");
        assert(items[1].partIndex === 1 && items[1].id === "p:1/G2" && items[2].partIndex === 2 && items[2].id === "p:2/G2", "attachments shifted to @1,@2");

        // Multi attachment, no text -> unchanged (attachments at @0,@1).
        arr = await toInboundMessages(null, new Map(), {{ message: {{ guid: "G3", content: {{ attachments: [{{ guid: "A0" }}, {{ guid: "A1" }}] }} }} }}, "+1");
        items = arr[0].content.items;
        assert(items.length === 2 && items[0].partIndex === 0 && items[0].id === "p:0/G3" && items[1].partIndex === 1, "no-text multi unchanged");

        // Text only, no attachments -> plain text (unchanged).
        r = await rebuildFromAppleMessage(null, {{ guid: "G4", content: {{ text: "just text", attachments: [] }} }}, "+1");
        assert(r.content.type === "text" && r.content.text === "just text" && r.id === "G4", "text-only unchanged");
        """
    )
    run = subprocess.run(
        ["node", "--input-type=module", "-e", harness],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
