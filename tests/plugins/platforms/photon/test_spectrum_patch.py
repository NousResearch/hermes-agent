"""Regression tests for Hermes' Spectrum mixed text+attachment workaround."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


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


# spectrum-ts 5.x ships the iMessage inbound mapper in the bundled, tab-indented
# `@spectrum-ts/imessage/dist/index.js`. This fixture mirrors the upstream shape
# of the two branches the patch rewrites (`rebuildFromAppleMessage`, the
# on-demand path, and `toInboundMessages`, the live-stream path) so the patch's
# exact-match `replaceOnce` guards are exercised without a real npm install.
def _v5_imessage_fixture() -> str:
    t = "\t"
    lines = [
        "const buildMessageBase = () => ({});",
        "const rebuildFromAppleMessage = async (client, message, phone, chatGuidHint) => {",
        f"{t}const messageGuidStr = message.guid;",
        f"{t}const base = buildMessageBase(message, chatGuidHint, message.dateCreated ?? new Date(), phone);",
        f"{t}const attachments = messageAttachments(message);",
        f"{t}if (attachments.length === 1) {{",
        f"{t}{t}const info = attachments[0];",
        f'{t}{t}if (!info) throw new Error("Unreachable: attachments.length === 1 but no element");',
        f"{t}{t}return buildAttachmentMessage(client, base, info, messageGuidStr, 0);",
        f"{t}}}",
        f"{t}if (attachments.length > 1) {{",
        f"{t}{t}const items = [];",
        f"{t}{t}for (let i = 0; i < attachments.length; i++) {{",
        f"{t}{t}{t}const info = attachments[i];",
        f"{t}{t}{t}if (!info) continue;",
        f"{t}{t}{t}items.push(await buildAttachmentMessage(client, base, info, formatChildId(i, messageGuidStr), i, messageGuidStr));",
        f"{t}{t}}}",
        f"{t}{t}return {{",
        f"{t}{t}{t}...base,",
        f"{t}{t}{t}id: messageGuidStr,",
        f"{t}{t}{t}content: asProviderGroup(items)",
        f"{t}{t}}};",
        f"{t}}}",
        f"{t}const text = message.content.text;",
        f"{t}return {{ ...base, id: messageGuidStr, content: text ? asText(text) : asCustom(message) }};",
        "};",
        "const toInboundMessages = async (client, cache, event, phone) => {",
        f"{t}const base = buildMessageBase(event.message, event.chatGuid, event.occurredAt, phone);",
        f"{t}const messageGuidStr = event.message.guid;",
        f"{t}const attachments = messageAttachments(event.message);",
        f"{t}if (attachments.length === 1) {{",
        f"{t}{t}const info = attachments[0];",
        f'{t}{t}if (!info) throw new Error("Unreachable: attachments.length === 1 but no element");',
        f"{t}{t}const msg = await buildAttachmentMessage(client, base, info, messageGuidStr, 0);",
        f"{t}{t}cacheMessage(cache, msg);",
        f"{t}{t}return [msg];",
        f"{t}}}",
        f"{t}if (attachments.length > 1) {{",
        f"{t}{t}const items = [];",
        f"{t}{t}for (let i = 0; i < attachments.length; i++) {{",
        f"{t}{t}{t}const info = attachments[i];",
        f"{t}{t}{t}if (!info) continue;",
        f"{t}{t}{t}items.push(await buildAttachmentMessage(client, base, info, formatChildId(i, messageGuidStr), i, messageGuidStr));",
        f"{t}{t}}}",
        f"{t}{t}const parent = {{",
        f"{t}{t}{t}...base,",
        f"{t}{t}{t}id: messageGuidStr,",
        f"{t}{t}{t}content: asProviderGroup(items)",
        f"{t}{t}}};",
        f"{t}{t}cacheMessage(cache, parent);",
        f"{t}{t}return [parent];",
        f"{t}}}",
        f"{t}const text = event.message.content.text;",
        f"{t}const msg = {{ ...base, id: messageGuidStr, content: text ? asText(text) : asCustom(event.message) }};",
        f"{t}cacheMessage(cache, msg);",
        f"{t}return [msg];",
        "};",
    ]
    return "\n".join(lines) + "\n"


def _write_fixture(tmp_path: Path) -> Path:
    dist = tmp_path / "node_modules" / "@spectrum-ts" / "imessage" / "dist"
    dist.mkdir(parents=True)
    bundle = dist / "index.js"
    bundle.write_text(_v5_imessage_fixture(), encoding="utf-8")
    return bundle


def test_spectrum_patch_preserves_text_with_attachments(tmp_path: Path) -> None:
    """The sidecar dependency patch must turn text+attachment into group content."""
    if not shutil.which("node"):
        pytest.skip("node not available")
    bundle = _write_fixture(tmp_path)

    result = subprocess.run(
        ["node", str(_PATCHER), str(tmp_path)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    patched = bundle.read_text(encoding="utf-8")
    assert "Preserve mixed text + attachment iMessage payloads" in patched
    # Single attachment + text -> a group whose first child is the dropped text.
    assert "content: asProviderGroup([textMsg, attachmentMsg])" in patched
    assert "content: asText(text)" in patched
    assert "id: formatChildId(0, messageGuidStr)" in patched
    # Multi attachment + text -> text child at partIndex 0, attachments shifted.
    assert "const partIndex = text ? i + 1 : i;" in patched
    assert "content: asProviderGroup(items)" in patched

    # The patched bundle must still be syntactically valid JavaScript.
    check = subprocess.run(
        ["node", "--check", str(bundle)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr


def test_spectrum_patch_is_idempotent(tmp_path: Path) -> None:
    """A second run must no-op (marker guard), not double-apply or throw."""
    if not shutil.which("node"):
        pytest.skip("node not available")
    bundle = _write_fixture(tmp_path)

    first = subprocess.run(
        ["node", str(_PATCHER), str(tmp_path)],
        cwd=Path.cwd(), text=True, capture_output=True, check=False,
    )
    assert first.returncode == 0, first.stderr
    after_first = bundle.read_text(encoding="utf-8")

    second = subprocess.run(
        ["node", str(_PATCHER), str(tmp_path)],
        cwd=Path.cwd(), text=True, capture_output=True, check=False,
    )
    assert second.returncode == 0, second.stderr
    assert bundle.read_text(encoding="utf-8") == after_first
