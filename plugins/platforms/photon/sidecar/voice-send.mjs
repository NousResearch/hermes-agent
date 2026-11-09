// Spectrum converts voice bytes to M4A but preserves the source upload name.
// Prepare both bytes and identity together, so AAC isn't uploaded as .mp3/.wav.
import { randomUUID } from "node:crypto"
import { readFile } from "node:fs/promises"

const M4A_BRANDS = new Set(["M4A ", "M4B ", "M4P ", "mp42", "mp41", "isom", "iso2"])

export async function prepareVoiceAttachment(path, voice) {
  const { ensureM4a } = await import("@spectrum-ts/core/authoring")
  // Empty MIME makes Spectrum sniff bytes, never trust a misleading filename
  // or caller MIME. Other containers go through ffmpeg's own input detection.
  const converted = await ensureM4a(await readFile(path), "")
  const buffer = converted.buffer
  if (buffer.toString("ascii", 4, 8) !== "ftyp" ||
      !M4A_BRANDS.has(buffer.toString("ascii", 8, 12))) {
    throw new Error("voice conversion did not produce an M4A container")
  }
  return voice(buffer, {
    name: `voice-${randomUUID()}.m4a`,
    mimeType: "audio/mp4",
    duration: converted.duration,
  })
}
