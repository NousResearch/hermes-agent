#!/usr/bin/env node
// Relax spectrum-ts 8.x poll schemas so empty/missing titles don't drop votes.
//
// Symptom (photon-hq/spectrum-ts#91, fix closed without merge in 8.0.0): when a
// recipient taps a native iMessage poll, the inbound path runs
// `toPollOptionMessage` → `resolvePoll` → `client.polls.get(...)` →
// `toCachedPoll` → `asPoll({title, options})`. The metadata returned by the
// cloud carries an empty `title`, and `pollSchema` rejected it via
// `z.string().nonempty()` (`too_small`, path `["title"]`). `resolvePoll` caught
// the throw, returned `undefined`, and `toPollOptionMessage` returned `[]` — the
// vote was silently dropped before the agent could ever see it.
//
// Upstream's intended fix (photon-hq/spectrum-ts PR #91) relaxed the three poll
// schemas to be lenient *on the wire* for inbound data, keeping outbound
// validation at the customer-facing builders (Hermes' `/send-poll` still rejects
// a blank title). We mirror that here against the compiled 8.0.0 chunk.
//
// The schemas live in @spectrum-ts/core/dist (chunk name is a content hash); we
// scan core/dist for the `pollChoiceSchema` anchor rather than hardcoding the
// chunk filename, and fail loudly if a future spectrum-ts reshapes them.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const MARKER = "Hermes patch: Relax spectrum-ts poll schemas for empty/missing inbound titles";

function scriptDir() {
  return path.dirname(fileURLToPath(import.meta.url));
}

function replaceOnce(source, from, to, label) {
  const count = source.split(from).length - 1;
  if (count !== 1) {
    throw new Error(`expected exactly one ${label} match, found ${count}`);
  }
  return source.replace(from, to);
}

export function patchSpectrumPollSchema(root = scriptDir()) {
  const dist = path.join(root, "node_modules", "@spectrum-ts", "core", "dist");
  if (!fs.existsSync(dist)) {
    throw new Error(`@spectrum-ts/core dist not found: ${dist}`);
  }
  const files = fs.readdirSync(dist)
    .filter((name) => name.endsWith(".js"))
    .map((name) => path.join(dist, name));

  for (const file of files) {
    const raw = fs.readFileSync(file, "utf8");
    if (raw.includes(MARKER)) {
      return { patched: false, file, reason: "already patched" };
    }
    // Normalize to LF for matching so the patch works regardless of the
    // checkout's line-ending style (CRLF would defeat \n-based anchors).
    const CR = String.fromCharCode(13);
    const CRLF = CR + "\n";
    const usedCRLF = raw.includes(CRLF);
    const original = usedCRLF ? raw.split(CRLF).join("\n") : raw;
    if (!original.includes("const pollChoiceSchema")) {
      continue;
    }

    let patched = original;
    // pollChoiceSchema.title: accept empty option text on the wire.
    patched = replaceOnce(
      patched,
      `const pollChoiceSchema = z.object({ title: z.string().nonempty() });`,
      `const pollChoiceSchema = z.object({ title: z.string() });`,
      "pollChoiceSchema.title"
    );
    // pollSchema.title: accept an empty/missing poll title on the wire.
    patched = replaceOnce(
      patched,
      `\ttitle: z.string().nonempty().max(300),`,
      `\ttitle: z.string().max(300).optional(),`,
      "pollSchema.title"
    );
    // pollOptionSchema.title: again lenient on the wire; the superRefine still
    //   enforces structural equality with option.title, so empty strings
    //   round-trip without silently dropping the vote.
    patched = replaceOnce(
      patched,
      `\tselected: z.boolean(),\n\ttitle: z.string().nonempty()`,
      `\tselected: z.boolean(),\n\ttitle: z.string()`,
      "pollOptionSchema.title"
    );

    patched = `// ${MARKER}\n${patched}`;
    if (usedCRLF) {
      patched = patched.split("\n").join(CRLF);
    }
    fs.writeFileSync(file, patched, "utf8");
    return { patched: true, file };
  }
  throw new Error("could not find spectrum-ts core poll schema chunk to patch");
}

const _invokedDirectly =
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href;
if (_invokedDirectly) {
  try {
    const root = process.argv[2] ? path.resolve(process.argv[2]) : scriptDir();
    const result = patchSpectrumPollSchema(root);
    const action = result.patched ? "patched" : "ok";
    console.error(`photon-sidecar: spectrum poll schema patch ${action}: ${result.file}`);
  } catch (err) {
    console.error(`photon-sidecar: spectrum poll schema patch failed: ${err?.stack || err}`);
    process.exit(1);
  }
}
