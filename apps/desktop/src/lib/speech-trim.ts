// Sentence-seam cleanup for client-direct speech.
//
// `gpt-4o-mini-tts` (and the OpenAI speech endpoint generally) wraps every
// synthesis in padding: measured on 2026-11-09 with model gpt-4o-mini-tts and
// voice marin, a clip carries ~0.35 s of near-silence at its head and 0.32-0.42 s
// after each sentence-final punctuation mark. Played one clip per sentence,
// that padding lands at BOTH ends of every seam — head-of-clip silence, then
// the next clip's synthesis round-trip — so a multi-sentence reply reads as
// stuttering instead of speech.
//
// These helpers trim that padding on the desktop side, where the clip is
// already decoded, so no provider setting and no server change is needed.
// Trimming is per-clip and never joins clips: each sentence still plays as its
// own element, so barge-in, stop, and the playback sequence contract are
// untouched.

/** Keep a little air after the last syllable so trims never clip a consonant. */
const TAIL_KEEP_SECONDS = 0.06
/** Below this RMS (linear, ~ -45 dBFS) a sample counts as silence. */
const SILENCE_RMS = 0.006
/** Analysis window, in seconds. Short enough to find a real onset, long enough to skip a single zero sample. */
const WINDOW_SECONDS = 0.005

function windowIsSilent(data: Float32Array, from: number, to: number): boolean {
  let sum = 0

  for (let i = from; i < to; i += 1) {
    sum += data[i] * data[i]
  }

  const count = to - from

  return count <= 0 || Math.sqrt(sum / count) < SILENCE_RMS
}

/**
 * Decode encoded audio (mp3) into raw samples. Returns null when the browser
 * cannot decode it, in which case the caller plays the original bytes — an
 * untrimmed clip is always better than a dropped one.
 */
export async function decodeAudioClip(bytes: ArrayBuffer, context: AudioContext): Promise<null | AudioBuffer> {
  try {
    // decodeAudioData detaches the buffer it is given in some engines — hand it a copy.
    return await context.decodeAudioData(bytes.slice(0))
  } catch {
    return null
  }
}

export interface TrimmedRange {
  /** Start offset in seconds. */
  start: number
  /** Length in seconds. */
  duration: number
  /** How much lead silence was removed — for tests and diagnostics. */
  trimmedHead: number
  trimmedTail: number
}

/** The speech-bearing span of a decoded clip, with the provider's edge padding removed. */
export function speechRange(buffer: AudioBuffer): TrimmedRange {
  const data = buffer.getChannelData(0)
  const perWindow = Math.max(1, Math.floor(buffer.sampleRate * WINDOW_SECONDS))
  const windows = Math.floor(data.length / perWindow)

  let firstSpeaking = -1
  let lastSpeaking = -1

  for (let w = 0; w < windows; w += 1) {
    const from = w * perWindow
    const to = from + perWindow

    if (!windowIsSilent(data, from, to)) {
      if (firstSpeaking === -1) {
        firstSpeaking = w
      }

      lastSpeaking = w
    }
  }

  // All silence (silence hallucination from the provider, or an unspeakable
  // sentence): keep the clip intact rather than emitting a zero-length element,
  // which would fire `ended` immediately and turn a seam into a click.
  if (firstSpeaking === -1) {
    return { duration: buffer.duration, start: 0, trimmedHead: 0, trimmedTail: 0 }
  }

  const startSample = firstSpeaking * perWindow
  const endSample = Math.min(data.length, (lastSpeaking + 1) * perWindow)
  const start = startSample / buffer.sampleRate
  const keepTail = Math.min(TAIL_KEEP_SECONDS, Math.max(0, buffer.duration - start))
  const end = Math.min(buffer.duration, endSample / buffer.sampleRate + keepTail)

  return {
    duration: Math.max(0.01, end - start),
    start,
    trimmedHead: start,
    trimmedTail: Math.max(0, buffer.duration - end)
  }
}
