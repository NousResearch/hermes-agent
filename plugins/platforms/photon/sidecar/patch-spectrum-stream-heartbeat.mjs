#!/usr/bin/env node
// Surface the iMessage streams' server heartbeats to the sidecar.
//
// Photon's iMessage streams send a heartbeat frame about every 30s even when
// nobody writes. @photon-ai/advanced-imessage's `createGrpcClient` accepts an
// `onHeartbeat` option for those frames, but spectrum-ts never passes one, so
// the sidecar cannot tell a quiet-but-live stream from a stalled one. This
// wraps the `createGrpcClient` import in `@spectrum-ts/imessage/dist/index.js`
// so every client spectrum-ts builds reports heartbeats to
// `globalThis.__hermesPhotonStreamHeartbeat`.
//
// Fail safe: the anchor must match exactly once or nothing is written; the
// sidecar's watchdog then keeps its silence-probe behavior.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const MARKER = "Hermes patch: Surface iMessage stream heartbeats";

const IMPORT_ANCHOR =
  'import { ErrorCode, NotFoundError, ValidationError, createGrpcClient } from "@photon-ai/advanced-imessage/grpc";\n';
const IMPORT_REPLACEMENT = [
  'import { ErrorCode, NotFoundError, ValidationError, createGrpcClient as hermesCreateGrpcClient } from "@photon-ai/advanced-imessage/grpc";',
  "const createGrpcClient = (options) => hermesCreateGrpcClient({",
  "\t...options,",
  "\tonHeartbeat: () => {",
  "\t\toptions?.onHeartbeat?.();",
  "\t\tglobalThis.__hermesPhotonStreamHeartbeat?.();",
  "\t}",
  "});",
  "",
].join("\n");

function scriptDir() {
  return path.dirname(fileURLToPath(import.meta.url));
}

export function patchSpectrumStreamHeartbeat(root = scriptDir()) {
  const file = path.join(root, "node_modules", "@spectrum-ts", "imessage", "dist", "index.js");
  if (!fs.existsSync(file)) {
    throw new Error(`@spectrum-ts/imessage dist not found: ${file}`);
  }
  const raw = fs.readFileSync(file, "utf8");
  if (raw.includes(MARKER)) {
    return { patched: false, file, reason: "already patched" };
  }
  // Match on LF and restore CRLF on write (Windows autocrlf checkouts).
  const CRLF = String.fromCharCode(13) + "\n";
  const usedCRLF = raw.includes(CRLF);
  const source = usedCRLF ? raw.split(CRLF).join("\n") : raw;
  const found = source.split(IMPORT_ANCHOR).length - 1;
  if (found !== 1) {
    throw new Error(`expected exactly one createGrpcClient import anchor, found ${found}`);
  }
  let patched = source.replace(IMPORT_ANCHOR, IMPORT_REPLACEMENT);
  patched = `// ${MARKER}\n${patched}`;
  if (usedCRLF) patched = patched.split("\n").join(CRLF);
  fs.writeFileSync(file, patched, "utf8");
  return { patched: true, file };
}

const _invokedDirectly =
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href;
if (_invokedDirectly) {
  // Never fail `npm ci`: without the patch the watchdog keeps its probe-only
  // behavior, which is what it did before.
  try {
    const root = process.argv[2] ? path.resolve(process.argv[2]) : scriptDir();
    const result = patchSpectrumStreamHeartbeat(root);
    console.error(
      `photon-sidecar: spectrum stream heartbeat patch ${result.patched ? "patched" : "ok"}: ${result.file}`
    );
  } catch (err) {
    console.error(
      "photon-sidecar: spectrum stream heartbeat patch skipped (stall watchdog " +
        `falls back to silence probes): ${err?.message || err}`
    );
  }
}
