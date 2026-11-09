// Spectrum converts voice bytes to M4A but preserves the source upload name.
// Prepare both bytes and identity together, so AAC isn't uploaded as .mp3/.wav.
import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { createReadStream } from "node:fs";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const MAX_INPUT_BYTES = 32 * 1024 * 1024;
const MAX_SECONDS = 300;
const PCM_BYTES_PER_SECOND = 24000 * 2;
const PROCESS_TIMEOUT_MS = 10000;

async function readInput(path) {
  if ((await stat(path)).size > MAX_INPUT_BYTES) throw new Error("Photon voice input exceeds 32 MiB");
  const chunks = [];
  let size = 0;
  for await (const chunk of createReadStream(path)) {
    size += chunk.length;
    if (size > MAX_INPUT_BYTES) throw new Error("Photon voice input exceeds 32 MiB");
    chunks.push(chunk);
  }
  if (!size) throw new Error("Photon voice input is empty");
  return Buffer.concat(chunks, size);
}

async function ffmpegBinary() {
  // Same optional installed binary/PATH resolution as Spectrum. No provisioning.
  try {
    return (await import("ffmpeg-static")).default || "ffmpeg";
  } catch {
    return "ffmpeg";
  }
}

function runFfmpeg(binary, args, stage, decode = false) {
  return new Promise((resolve, reject) => {
    const child = spawn(binary, args, { stdio: ["ignore", decode ? "pipe" : "ignore", "pipe"] });
    let failure;
    let pcmBytes = 0;
    let diagnosticBytes = 0;
    const stop = (reason) => {
      failure ??= new Error(`Photon voice ${stage}: ${reason}`);
      child.kill("SIGKILL");
    };
    const timer = setTimeout(() => stop("encoder time budget exceeded"), PROCESS_TIMEOUT_MS);
    child.on("error", () => {
      failure = new Error(`Photon voice ${stage}: ffmpeg is unavailable`);
    });
    child.stderr.on("data", (chunk) => {
      diagnosticBytes += chunk.length;
      if (diagnosticBytes > 16384) stop("encoder diagnostics budget exceeded");
    });
    child.stdout?.on("data", (chunk) => {
      pcmBytes += chunk.length;
      if (pcmBytes > MAX_SECONDS * PCM_BYTES_PER_SECOND) stop("audio exceeds 300 seconds");
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (failure) reject(failure);
      else if (code !== 0) reject(new Error(`Photon voice ${stage}: invalid or unsupported audio`));
      else resolve(pcmBytes);
    });
  });
}

export async function prepareVoiceAttachment(path, voice) {
  const raw = await readInput(path);
  const directory = await mkdtemp(join(tmpdir(), "hermes-photon-voice-"));
  try {
    const input = join(directory, "input");
    const output = join(directory, "voice.m4a");
    await writeFile(input, raw);
    const binary = await ffmpegBinary();
    const strict = [
      "-nostdin",
      "-v",
      "error",
      "-xerror",
      "-err_detect",
      "explode",
      "-protocol_whitelist",
      "file,pipe",
    ];
    // Sniff the input bytes and encode exactly once, including existing M4A.
    // The SDK's signature-only M4A shortcut isn't a usable-audio validation.
    await runFfmpeg(
      binary,
      [
        ...strict,
        "-i",
        input,
        "-map",
        "0:a:0",
        "-vn",
        "-map_metadata",
        "-1",
        "-t",
        String(MAX_SECONDS + 1),
        "-ac",
        "1",
        "-ar",
        "24000",
        "-c:a",
        "aac",
        "-f",
        "ipod",
        output,
      ],
      "conversion",
    );
    // Count decoded samples instead of trusting headers or claimed duration.
    const pcmBytes = await runFfmpeg(
      binary,
      [...strict, "-i", output, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "24000", "-f", "s16le", "pipe:1"],
      "validation",
      true,
    );
    const duration = pcmBytes / PCM_BYTES_PER_SECOND;
    if (!(duration > 0) || pcmBytes % 2 !== 0) throw new Error("Photon voice has no usable audio samples");
    const buffer = await readFile(output);
    if (buffer.toString("ascii", 4, 8) !== "ftyp") throw new Error("Photon voice conversion isn't M4A");
    return voice(buffer, { name: `voice-${randomUUID()}.m4a`, mimeType: "audio/mp4", duration });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}
