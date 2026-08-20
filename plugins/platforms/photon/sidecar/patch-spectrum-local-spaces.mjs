#!/usr/bin/env node
// LOCAL MODE EXPERIMENT: Spectrum 3.1.0 can send to a bare iMessage address
// locally, but its space.get() gate prevents Hermes cron delivery from
// obtaining the Space wrapper needed to call send(). Remove this entire file
// once Spectrum supports spaces in local iMessage mode.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const MARKER = "Hermes local iMessage workaround: allow space.get";

function scriptDir() {
  return path.dirname(fileURLToPath(import.meta.url));
}

export function patchSpectrumLocalSpaces(root = scriptDir()) {
  const nativeDist = path.join(
    root,
    "node_modules",
    "@spectrum-ts",
    "imessage-local",
    "dist"
  );
  if (fs.existsSync(nativeDist)) {
    const nativeFiles = fs.readdirSync(nativeDist)
      .filter((name) => name.endsWith(".js"))
      .map((name) => path.join(nativeDist, name));
    for (const file of nativeFiles) {
      const raw = fs.readFileSync(file, "utf8");
      if (
        raw.includes("get: async ({ input }) => ({") &&
        raw.includes("type: chatTypeFromGuid(input.id)")
      ) {
        return { patched: false, file, reason: "native support" };
      }
    }
  }

  const dist = path.join(root, "node_modules", "spectrum-ts", "dist");
  if (!fs.existsSync(dist)) {
    throw new Error(`spectrum-ts dist not found: ${dist}`);
  }

  const files = fs.readdirSync(dist)
    .filter((name) => name.endsWith(".js"))
    .map((name) => path.join(dist, name));

  for (const file of files) {
    const raw = fs.readFileSync(file, "utf8");
    if (raw.includes(MARKER)) {
      return { patched: false, file, reason: "already patched" };
    }
    if (!raw.includes("local mode only supports replying to existing messages")) {
      continue;
    }

    const CR = String.fromCharCode(13);
    const CRLF = CR + "\n";
    const usedCRLF = raw.includes(CRLF);
    const source = usedCRLF ? raw.split(CRLF).join("\n") : raw;
    const from = `      if (isLocal(client)) {
        throw UnsupportedError.action(
          "space.get",
          "iMessage (local mode)",
          "local mode only supports replying to existing messages"
        );
      }`;
    const to = `      if (isLocal(client)) {
        // ${MARKER}. IMessageSDK.send() accepts both bare DM addresses and
        // existing chat GUIDs; only Spectrum's Space construction gate is
        // missing. "local" satisfies the schema and is unused by local send.
        return {
          id: input.id,
          type: chatTypeFromGuid(input.id),
          phone: "local"
        };
      }`;
    const count = source.split(from).length - 1;
    if (count !== 1) {
      throw new Error(`expected exactly one local space.get gate, found ${count}`);
    }

    let patched = source.replace(from, to);
    if (usedCRLF) patched = patched.split("\n").join(CRLF);
    fs.writeFileSync(file, patched, "utf8");
    return { patched: true, file };
  }

  throw new Error("could not find spectrum-ts local iMessage space.get gate");
}

const invokedDirectly =
  process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (invokedDirectly) {
  try {
    const root = process.argv[2] ? path.resolve(process.argv[2]) : scriptDir();
    const result = patchSpectrumLocalSpaces(root);
    const action = result.patched ? "patched" : result.reason || "ok";
    console.error(`photon-sidecar: Spectrum local spaces patch ${action}: ${result.file}`);
  } catch (err) {
    console.error(`photon-sidecar: Spectrum local spaces patch failed: ${err?.stack || err}`);
    process.exit(1);
  }
}
