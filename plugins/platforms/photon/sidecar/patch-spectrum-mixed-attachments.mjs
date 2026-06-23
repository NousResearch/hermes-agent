#!/usr/bin/env node
// Patch spectrum-ts' iMessage inbound mapper until upstream preserves mixed
// text + attachment Apple events. The mapper returns only the attachment
// Message whenever attachments are present, dropping `message.content.text`
// before Hermes can see it. We rewrite both inbound entry points
// (`toInboundMessages`, the live-stream path, and `rebuildFromAppleMessage`,
// the on-demand `getMessage`/cache path) to re-emit a grouped message whose
// first child is the dropped text.
//
// As of spectrum-ts 5.x the SDK is a package split: the iMessage provider — and
// this mapper — live in `@spectrum-ts/imessage`, shipped as a single bundled
// `dist/index.js` (tab-indented). We resolve and patch that file. The sidecar
// consumes already-mapped `Message` objects off `app.messages`, so the dropped
// text is unrecoverable downstream; patching the bundle is the only fix point.
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

const MARKER = "Hermes patch: Preserve mixed text + attachment iMessage payloads";

function scriptDir() {
  return path.dirname(fileURLToPath(import.meta.url));
}

// Resolve `@spectrum-ts/imessage`'s bundled entry. Prefer the install under
// `root` (so tests can point at a fixture and the committed install resolves
// without ambiguity); fall back to Node resolution, which follows npm hoisting
// and the nested `spectrum-ts/node_modules/...` case.
function resolveImessageBundle(root) {
  const direct = path.join(
    root,
    "node_modules",
    "@spectrum-ts",
    "imessage",
    "dist",
    "index.js"
  );
  if (fs.existsSync(direct)) return direct;
  try {
    const require = createRequire(import.meta.url);
    return require.resolve("@spectrum-ts/imessage");
  } catch {
    return null;
  }
}

// Tab-indented line builder — the bundle uses hard tabs, so matches must too.
const TAB = "\t";
const ind = (depth, code) => TAB.repeat(depth) + code;
const block = (lines) => lines.map(([depth, code]) => ind(depth, code)).join("\n");

// Replace exactly one occurrence; throw if the count isn't 1 so a shape change
// in a new spectrum-ts release fails closed (loud at install time) instead of
// silently leaving the mixed-attachment bug — or double-patching — in place.
function replaceOnce(source, from, to, label) {
  const count = source.split(from).length - 1;
  if (count !== 1) {
    throw new Error(`expected exactly one ${label} match, found ${count}`);
  }
  return source.replace(from, to);
}

// A text child message, partIndex 0, parented to the Apple message guid.
const textChild = (depth, lead, textExpr) =>
  block([
    [depth, `${lead}{`],
    [depth + 1, "...base,"],
    [depth + 1, "id: formatChildId(0, messageGuidStr),"],
    [depth + 1, `content: asText(${textExpr}),`],
    [depth + 1, "partIndex: 0,"],
    [depth + 1, "parentId: messageGuidStr"],
    [depth, lead.startsWith("items.push(") ? "});" : "};"],
  ]);

// --- rebuildFromAppleMessage (on-demand path, returns a single Message) ------

const REBUILD_SINGLE_FROM = block([
  [1, "if (attachments.length === 1) {"],
  [2, "const info = attachments[0];"],
  [2, 'if (!info) throw new Error("Unreachable: attachments.length === 1 but no element");'],
  [2, "return buildAttachmentMessage(client, base, info, messageGuidStr, 0);"],
  [1, "}"],
]);

const REBUILD_SINGLE_TO = [
  ind(1, "if (attachments.length === 1) {"),
  ind(2, "const info = attachments[0];"),
  ind(2, 'if (!info) throw new Error("Unreachable: attachments.length === 1 but no element");'),
  ind(2, "const text = message.content.text;"),
  ind(2, "const attachmentMsg = await buildAttachmentMessage(client, base, info, text ? formatChildId(1, messageGuidStr) : messageGuidStr, text ? 1 : 0, text ? messageGuidStr : void 0);"),
  ind(2, "if (text) {"),
  textChild(3, "const textMsg = ", "text"),
  ind(3, "return {"),
  ind(4, "...base,"),
  ind(4, "id: messageGuidStr,"),
  ind(4, "content: asProviderGroup([textMsg, attachmentMsg])"),
  ind(3, "};"),
  ind(2, "}"),
  ind(2, "return attachmentMsg;"),
  ind(1, "}"),
].join("\n");

const MULTI_LOOP = (textExpr) =>
  [
    ind(2, "const items = [];"),
    ind(2, "if (text) {"),
    textChild(3, "items.push(", "text"),
    ind(2, "}"),
    ind(2, "for (let i = 0; i < attachments.length; i++) {"),
    ind(3, "const info = attachments[i];"),
    ind(3, "if (!info) continue;"),
    ind(3, "const partIndex = text ? i + 1 : i;"),
    ind(3, "items.push(await buildAttachmentMessage(client, base, info, formatChildId(partIndex, messageGuidStr), partIndex, messageGuidStr));"),
    ind(2, "}"),
  ].join("\n");

const REBUILD_MULTI_FROM = block([
  [1, "if (attachments.length > 1) {"],
  [2, "const items = [];"],
  [2, "for (let i = 0; i < attachments.length; i++) {"],
  [3, "const info = attachments[i];"],
  [3, "if (!info) continue;"],
  [3, "items.push(await buildAttachmentMessage(client, base, info, formatChildId(i, messageGuidStr), i, messageGuidStr));"],
  [2, "}"],
  [2, "return {"],
  [3, "...base,"],
  [3, "id: messageGuidStr,"],
  [3, "content: asProviderGroup(items)"],
  [2, "};"],
  [1, "}"],
]);

const REBUILD_MULTI_TO = [
  ind(1, "if (attachments.length > 1) {"),
  ind(2, "const text = message.content.text;"),
  MULTI_LOOP("message.content.text"),
  ind(2, "return {"),
  ind(3, "...base,"),
  ind(3, "id: messageGuidStr,"),
  ind(3, "content: asProviderGroup(items)"),
  ind(2, "};"),
  ind(1, "}"),
].join("\n");

// --- toInboundMessages (live-stream path, returns Message[] + caches) --------

const INBOUND_SINGLE_FROM = block([
  [1, "if (attachments.length === 1) {"],
  [2, "const info = attachments[0];"],
  [2, 'if (!info) throw new Error("Unreachable: attachments.length === 1 but no element");'],
  [2, "const msg = await buildAttachmentMessage(client, base, info, messageGuidStr, 0);"],
  [2, "cacheMessage(cache, msg);"],
  [2, "return [msg];"],
  [1, "}"],
]);

const INBOUND_SINGLE_TO = [
  ind(1, "if (attachments.length === 1) {"),
  ind(2, "const info = attachments[0];"),
  ind(2, 'if (!info) throw new Error("Unreachable: attachments.length === 1 but no element");'),
  ind(2, "const text = event.message.content.text;"),
  ind(2, "const attachmentMsg = await buildAttachmentMessage(client, base, info, text ? formatChildId(1, messageGuidStr) : messageGuidStr, text ? 1 : 0, text ? messageGuidStr : void 0);"),
  ind(2, "if (text) {"),
  textChild(3, "const textMsg = ", "text"),
  ind(3, "const parent = {"),
  ind(4, "...base,"),
  ind(4, "id: messageGuidStr,"),
  ind(4, "content: asProviderGroup([textMsg, attachmentMsg])"),
  ind(3, "};"),
  ind(3, "cacheMessage(cache, parent);"),
  ind(3, "return [parent];"),
  ind(2, "}"),
  ind(2, "cacheMessage(cache, attachmentMsg);"),
  ind(2, "return [attachmentMsg];"),
  ind(1, "}"),
].join("\n");

const INBOUND_MULTI_FROM = block([
  [1, "if (attachments.length > 1) {"],
  [2, "const items = [];"],
  [2, "for (let i = 0; i < attachments.length; i++) {"],
  [3, "const info = attachments[i];"],
  [3, "if (!info) continue;"],
  [3, "items.push(await buildAttachmentMessage(client, base, info, formatChildId(i, messageGuidStr), i, messageGuidStr));"],
  [2, "}"],
  [2, "const parent = {"],
  [3, "...base,"],
  [3, "id: messageGuidStr,"],
  [3, "content: asProviderGroup(items)"],
  [2, "};"],
  [2, "cacheMessage(cache, parent);"],
  [2, "return [parent];"],
  [1, "}"],
]);

const INBOUND_MULTI_TO = [
  ind(1, "if (attachments.length > 1) {"),
  ind(2, "const text = event.message.content.text;"),
  MULTI_LOOP("event.message.content.text"),
  ind(2, "const parent = {"),
  ind(3, "...base,"),
  ind(3, "id: messageGuidStr,"),
  ind(3, "content: asProviderGroup(items)"),
  ind(2, "};"),
  ind(2, "cacheMessage(cache, parent);"),
  ind(2, "return [parent];"),
  ind(1, "}"),
].join("\n");

function applyPatch(source) {
  source = replaceOnce(source, REBUILD_SINGLE_FROM, REBUILD_SINGLE_TO, "rebuild single attachment");
  source = replaceOnce(source, REBUILD_MULTI_FROM, REBUILD_MULTI_TO, "rebuild multi attachment");
  source = replaceOnce(source, INBOUND_SINGLE_FROM, INBOUND_SINGLE_TO, "inbound single attachment");
  source = replaceOnce(source, INBOUND_MULTI_FROM, INBOUND_MULTI_TO, "inbound multi attachment");
  return source;
}

export function patchSpectrumTs(root = scriptDir()) {
  const file = resolveImessageBundle(root);
  if (!file || !fs.existsSync(file)) {
    throw new Error(
      `@spectrum-ts/imessage bundle not found (looked under ${root}). ` +
        "Run `npm install` inside plugins/platforms/photon/sidecar/."
    );
  }

  const raw = fs.readFileSync(file, "utf8");
  if (raw.includes(MARKER)) {
    return { patched: false, file, reason: "already patched" };
  }

  // Normalize to LF for matching so the patch works regardless of the
  // checkout's line-ending style (Windows git autocrlf produces CRLF, which
  // would otherwise defeat the \n-based search strings). The original EOL
  // style is restored on write.
  const CR = String.fromCharCode(13);
  const CRLF = CR + "\n";
  const usedCRLF = raw.includes(CRLF);
  const original = usedCRLF ? raw.split(CRLF).join("\n") : raw;

  if (
    !original.includes("const toInboundMessages = async") ||
    !original.includes("const rebuildFromAppleMessage = async")
  ) {
    throw new Error(
      `${file} does not look like the spectrum-ts iMessage inbound mapper ` +
        "(missing toInboundMessages/rebuildFromAppleMessage) — the SDK shape " +
        "changed; update patch-spectrum-mixed-attachments.mjs for this version."
    );
  }

  let patched = applyPatch(original);
  patched = `// ${MARKER}\n${patched}`;
  if (usedCRLF) {
    patched = patched.split("\n").join(CRLF);
  }
  fs.writeFileSync(file, patched, "utf8");
  return { patched: true, file };
}

const _invokedDirectly =
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href;
if (_invokedDirectly) {
  try {
    const root = process.argv[2] ? path.resolve(process.argv[2]) : scriptDir();
    const result = patchSpectrumTs(root);
    const action = result.patched ? "patched" : "ok";
    console.error(`photon-sidecar: spectrum mixed attachment patch ${action}: ${result.file}`);
  } catch (err) {
    console.error(`photon-sidecar: spectrum mixed attachment patch failed: ${err?.stack || err}`);
    process.exit(1);
  }
}
