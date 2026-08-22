// Materialize Spectrum voice() input as a unique AAC/M4A file.
// Spectrum 8's uploadVoice() preserved source identity; keep this boundary
// explicit so the outbound voice attachment is never confused with inbound CAF.

import { randomUUID, createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ensureM4a } from "@spectrum-ts/core/authoring";

const M4A_MIME = "audio/mp4";

function sha256(buffer) {
  return createHash("sha256").update(buffer).digest("hex");
}

export async function prepareVoiceAttachment(sourcePath) {
  if (typeof sourcePath !== "string" || !sourcePath) {
    throw new Error("prepareVoiceAttachment: source path is required");
  }
  const raw = await readFile(sourcePath);
  const sourceSha = sha256(raw);
  const converted = await ensureM4a(raw, "");
  const brand = converted.buffer.toString("ascii", 8, 12);
  const uploadSha = sha256(converted.buffer);
  const dir = await mkdtemp(join(tmpdir(), "hermes-photon-voice-"));
  const fileName = `voice-${randomUUID()}.m4a`;
  const outPath = join(dir, fileName);
  await writeFile(outPath, converted.buffer);
  console.error(
    "photon-sidecar: voice identity " +
      `source=${sourcePath} sourceSha=${sourceSha} sourceBytes=${raw.length} ` +
      `uploadSha=${uploadSha} uploadBytes=${converted.buffer.length} brand=${JSON.stringify(brand)} ` +
      `fileName=${fileName}`
  );
  return {
    path: outPath,
    opts: { name: fileName, mimeType: M4A_MIME },
    sourceSha,
    uploadSha,
    uploadBytes: converted.buffer.length,
    brand,
    cleanup: async () => {
      await rm(dir, { recursive: true, force: true }).catch(() => {});
    },
  };
}
