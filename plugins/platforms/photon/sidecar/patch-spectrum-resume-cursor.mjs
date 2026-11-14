#!/usr/bin/env node
// Route spectrum-ts' private resume cursor through a hook so the sidecar can
// keep it across restarts (see catchup.mjs). `resumableOrderedStream` in
// `@spectrum-ts/core/dist/authoring.js` keeps `lastCursor` in a closure; three
// one-line edits ask the hook for a starting cursor, tell it about each item
// before it is emitted, and tell it when Photon refused the cursor. Every edit
// is optional-chained, so the patched file behaves exactly as before when no
// hook is installed.
//
// Applied from postinstall and again at sidecar start (like
// patch-spectrum-mixed-attachments.mjs), so an `npm ci` cannot drop it. The
// anchors match spectrum-ts 12.7.0's published output exactly; if a newer
// release reshapes the function, nothing is written and the sidecar logs the
// failure and runs live-only.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HOOK = "globalThis.__hermesPhotonResumeCursor";

const EDITS = [
  [
    "\tlet lastCursor;\n\tlet retryDelayMs = initialRetryDelayMs;",
    `\tlet lastCursor = ${HOOK}?.initial?.(label);\n\tlet retryDelayMs = initialRetryDelayMs;`,
  ],
  [
    "\tconst deliverItem = async (item, resetRetry, clearOnCursorAdvance) => {\n",
    "\tconst deliverItem = async (item, resetRetry, clearOnCursorAdvance) => {\n" +
      `\t\tif (!deliveredSinceCursor.has(item.id)) ${HOOK}?.note?.(label, item);\n`,
  ],
  [
    "\t\tif (error instanceof CursorRejectedError) {\n\t\t\tlastCursor = void 0;",
    "\t\tif (error instanceof CursorRejectedError) {\n" +
      `\t\t\t${HOOK}?.rejected?.(label);\n\t\t\tlastCursor = void 0;`,
  ],
];

function scriptDir() {
  return path.dirname(fileURLToPath(import.meta.url));
}

/** Patch authoring.js under `root`; returns {patched, file} or throws without writing. */
export function patchSpectrumResumeCursor(root = scriptDir()) {
  const file = path.join(root, "node_modules", "@spectrum-ts", "core", "dist", "authoring.js");
  let source = fs.readFileSync(file, "utf8");
  if (source.includes(HOOK)) return { patched: false, file };
  for (const [before, after] of EDITS) {
    const count = source.split(before).length - 1;
    if (count !== 1) {
      throw new Error(`unexpected spectrum-ts resumableOrderedStream source (anchor matched ${count} times)`);
    }
    source = source.replace(before, after);
  }
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, source, { mode: fs.statSync(file).mode & 0o777 });
  fs.renameSync(tmp, file);
  return { patched: true, file };
}

const _invokedDirectly = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (_invokedDirectly) {
  try {
    const root = process.argv[2] ? path.resolve(process.argv[2]) : scriptDir();
    const result = patchSpectrumResumeCursor(root);
    console.error(`photon-sidecar: spectrum resume cursor patch ${result.patched ? "patched" : "ok"}: ${result.file}`);
  } catch (err) {
    // Not fatal: without the hook the sidecar runs live-only, as before.
    console.error(`photon-sidecar: spectrum resume cursor patch failed; inbound catch-up across restarts is off: ${err?.message || err}`);
  }
}
