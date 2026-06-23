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


def test_spectrum_patch_preserves_text_when_single_attachment(tmp_path: Path) -> None:
    """The sidecar dependency patch must turn text+one attachment into group content."""
    dist = tmp_path / "node_modules" / "spectrum-ts" / "dist"
    dist.mkdir(parents=True)
    chunk = dist / "chunk-test.js"
    chunk.write_text(
        textwrap.dedent(
            """
            var rebuildFromAppleMessage = async (client, message, phone, chatGuidHint) => {
              const messageGuidStr = message.guid;
              const timestamp = message.dateCreated ?? /* @__PURE__ */ new Date();
              const base = buildMessageBase(message, chatGuidHint, timestamp, phone);
              const attachments = messageAttachments(message);
              if (attachments.length === 1) {
                const info = attachments[0];
                if (!info) {
                  throw new Error("Unreachable: attachments.length === 1 but no element");
                }
                return buildAttachmentMessage(client, base, info, messageGuidStr, 0);
              }
              if (attachments.length > 1) {
                const items = [];
                for (let i = 0; i < attachments.length; i++) {
                  const info = attachments[i];
                  if (!info) {
                    continue;
                  }
                  items.push(
                    await buildAttachmentMessage(
                      client,
                      base,
                      info,
                      formatChildId(i, messageGuidStr),
                      i,
                      messageGuidStr
                    )
                  );
                }
                return {
                  ...base,
                  id: messageGuidStr,
                  content: asProviderGroup(items)
                };
              }
              if (getBalloonBundleId(message) === URL_BALLOON_BUNDLE_ID) {
                return toRichlinkMessage(message, base, messageGuidStr);
              }
              const text2 = message.content.text;
              return {
                ...base,
                id: messageGuidStr,
                content: text2 ? asText(text2) : asCustom(message)
              };
            };
            var toInboundMessages = async (client, cache, event, phone) => {
              const base = buildMessageBase(
                event.message,
                event.chatGuid,
                event.occurredAt,
                phone
              );
              const messageGuidStr = event.message.guid;
              if (getBalloonBundleId(event.message) === URL_BALLOON_BUNDLE_ID) {
                const msg2 = toRichlinkMessage(event.message, base, messageGuidStr);
                cacheMessage(cache, msg2);
                return [msg2];
              }
              const attachments = messageAttachments(event.message);
              if (attachments.length === 1) {
                const info = attachments[0];
                if (!info) {
                  throw new Error("Unreachable: attachments.length === 1 but no element");
                }
                const msg2 = await buildAttachmentMessage(
                  client,
                  base,
                  info,
                  messageGuidStr,
                  0
                );
                cacheMessage(cache, msg2);
                return [msg2];
              }
              if (attachments.length > 1) {
                const items = [];
                for (let i = 0; i < attachments.length; i++) {
                  const info = attachments[i];
                  if (!info) {
                    continue;
                  }
                  items.push(
                    await buildAttachmentMessage(
                      client,
                      base,
                      info,
                      formatChildId(i, messageGuidStr),
                      i,
                      messageGuidStr
                    )
                  );
                }
                const parent = {
                  ...base,
                  id: messageGuidStr,
                  content: asProviderGroup(items)
                };
                cacheMessage(cache, parent);
                return [parent];
              }
              const text2 = event.message.content.text;
              const msg = {
                ...base,
                id: messageGuidStr,
                content: text2 ? asText(text2) : asCustom(event.message)
              };
              cacheMessage(cache, msg);
              return [msg];
            };
            """
        ),
        encoding="utf-8",
    )

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
    assert "content: asProviderGroup([textMsg, msg2])" in patched
    assert "content: asProviderGroup(items)" in patched
    assert "formatChildId(text2 ? i + 1 : i, messageGuidStr)" in patched
