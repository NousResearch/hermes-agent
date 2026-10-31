/**
 * Which log tail is actually "recent activity"?
 *
 * `gui.log` is written only while a process runs in `mode="gui"` (the Desktop's own local
 * backend). When the Desktop attaches to the messaging gateway instead (`mode="gateway"`),
 * `gui.log` is frozen — and a status panel that tails it unconditionally shows days-old
 * WARNING lines as current activity, with the timestamps stripped before display so nothing
 * gives the staleness away (#146682).
 *
 * Rather than making every caller know which mode the backend runs in, compare the tails:
 * the newest `asctime`-prefixed line decides. This is formatting-agnostic about everything
 * else in the line, and a tail without any parseable stamp never "wins" — it can only be
 * shown when it is the only one with content.
 */

/** `2026-10-31 18:58:49,627 …` (Python asctime) or the dotted/ISO variants. */
const LOG_STAMP_RE = /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:[.,](\d{1,3}))?/

export interface LogTail {
  file: string
  lines: string[]
}

/** Newest timestamp in `lines`, in ms, or `null` when no line carries one. */
export function newestLogTimestamp(lines: readonly string[]): number | null {
  let newest: number | null = null

  for (const raw of lines) {
    const match = LOG_STAMP_RE.exec(raw.trim())

    if (!match) {
      continue
    }

    const [, year, month, day, hour, minute, second, fraction] = match
    // Local wall-clock on both sides of the comparison; only the ORDER matters here, so a
    // single fixed interpretation keeps two files comparable without a timezone guess.
    const at = Date.UTC(
      Number(year),
      Number(month) - 1,
      Number(day),
      Number(hour),
      Number(minute),
      Number(second),
      fraction ? Number(fraction.padEnd(3, '0')) : 0
    )

    if (newest === null || at > newest) {
      newest = at
    }
  }

  return newest
}

/**
 * The tail whose newest line is the most recent, or `null` when every tail is empty.
 * Untimestamped tails are only returned when nothing carries a stamp at all.
 */
export function pickFreshestLogTail(tails: readonly LogTail[]): LogTail | null {
  let best: { at: number; tail: LogTail } | null = null
  let fallback: LogTail | null = null

  for (const tail of tails) {
    if (!tail.lines?.length) {
      continue
    }

    fallback ??= tail

    const at = newestLogTimestamp(tail.lines)

    if (at === null) {
      continue
    }

    if (!best || at > best.at) {
      best = { at, tail }
    }
  }

  return best?.tail ?? fallback
}
