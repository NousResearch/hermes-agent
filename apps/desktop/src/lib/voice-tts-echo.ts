// TTS self-echo guard for the voice-conversation barge monitor — a port of
// tools/voice_mode_transcript.is_tts_echo (the CLI's fix for the
// TTS -> STT -> TTS loop). Over speakers, Hermes' own reply bleeds into the
// mic, trips the playback-phase barge trigger, and gets transcribed; without
// this check the fragment is submitted as a user turn nobody spoke (#126708).
//
// Keep in lockstep with the Python rule: character-level similarity via a
// faithful difflib.SequenceMatcher(None, a, b).ratio() port (autojunk
// included), 0.6 threshold, and a transcript-sized sliding window over the
// spoken text for fragments of long replies.

/** tools/voice_mode_transcript.DEFAULT_TTS_ECHO_SIMILARITY_THRESHOLD */
export const DEFAULT_TTS_ECHO_SIMILARITY_THRESHOLD = 0.6

/** tools/voice_mode_transcript.MIN_FRAGMENT_LENGTH_FOR_ECHO — below this a
 *  genuine one-word barge ("yes") would trivially match a same-size window. */
export const MIN_FRAGMENT_LENGTH_FOR_ECHO = 10

// Code points, like Python str indexing (UTF-16 units would split CJK/emoji).
const normalizeForEchoCompare = (text: string): string[] => Array.from(text.replace(/\s+/gu, ' ').trim().toLowerCase())

/** difflib.SequenceMatcher(None, a, b).ratio() — isjunk=None, autojunk=True. */
export function sequenceMatcherRatio(a: readonly string[], b: readonly string[]): number {
  const total = a.length + b.length

  if (total === 0) {
    return 1
  }

  const b2j = new Map<string, number[]>()

  b.forEach((ch, j) => {
    const indices = b2j.get(ch)

    if (indices) {
      indices.push(j)
    } else {
      b2j.set(ch, [j])
    }
  })

  // autojunk: elements more common than 1% of a 200+-long b are "popular" and
  // never seed a match (they can still extend one, as in CPython).
  if (b.length >= 200) {
    const ntest = Math.floor(b.length / 100) + 1

    for (const [ch, indices] of b2j) {
      if (indices.length > ntest) {
        b2j.delete(ch)
      }
    }
  }

  const findLongestMatch = (alo: number, ahi: number, blo: number, bhi: number): [number, number, number] => {
    let besti = alo
    let bestj = blo
    let bestsize = 0
    let j2len = new Map<number, number>()

    for (let i = alo; i < ahi; i++) {
      const next = new Map<number, number>()

      for (const j of b2j.get(a[i]) ?? []) {
        if (j < blo) {
          continue
        }

        if (j >= bhi) {
          break
        }

        const k = (j2len.get(j - 1) ?? 0) + 1
        next.set(j, k)

        if (k > bestsize) {
          besti = i - k + 1
          bestj = j - k + 1
          bestsize = k
        }
      }

      j2len = next
    }

    // bjunk is empty (isjunk=None), so CPython's "extend with non-junk" loops
    // always run; this is how popular elements join a match.
    while (besti > alo && bestj > blo && a[besti - 1] === b[bestj - 1]) {
      besti--
      bestj--
      bestsize++
    }

    while (besti + bestsize < ahi && bestj + bestsize < bhi && a[besti + bestsize] === b[bestj + bestsize]) {
      bestsize++
    }

    return [besti, bestj, bestsize]
  }

  let matches = 0
  const queue: [number, number, number, number][] = [[0, a.length, 0, b.length]]

  while (queue.length > 0) {
    const [alo, ahi, blo, bhi] = queue.pop()!
    const [i, j, k] = findLongestMatch(alo, ahi, blo, bhi)

    if (k) {
      matches += k

      if (alo < i && blo < j) {
        queue.push([alo, i, blo, j])
      }

      if (i + k < ahi && j + k < bhi) {
        queue.push([i + k, ahi, j + k, bhi])
      }
    }
  }

  return (2 * matches) / total
}

/**
 * True when `transcript` looks like a self-capture of `spokenText` (the reply
 * Hermes was speaking when the barge tripped). A genuine interjection rarely
 * matches Hermes' own words, so a high ratio signals speaker bleed. Playback
 * captures span only pre-roll plus time-to-silence, so for long replies the
 * transcript is a FRAGMENT; when the whole-string ratio misses, a
 * transcript-sized window slides across the spoken text.
 */
export function isTtsEcho(
  transcript: string,
  spokenText: string,
  threshold: number = DEFAULT_TTS_ECHO_SIMILARITY_THRESHOLD
): boolean {
  const a = normalizeForEchoCompare(transcript || '')
  const b = normalizeForEchoCompare(spokenText || '')

  if (a.length === 0 || b.length === 0) {
    return false
  }

  const similar = (x: readonly string[], y: readonly string[]) => sequenceMatcherRatio(x, y) >= threshold

  if (similar(a, b)) {
    return true
  }

  if (a.length < MIN_FRAGMENT_LENGTH_FOR_ECHO || a.length >= b.length) {
    return false
  }

  for (let start = 0; start <= b.length - a.length; start++) {
    if (similar(a, b.slice(start, start + a.length))) {
      return true
    }
  }

  return false
}
