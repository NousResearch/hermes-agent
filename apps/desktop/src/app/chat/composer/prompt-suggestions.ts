import type { SessionInfo } from '@/types/hermes'

/** One click-to-fill starter, taken from a conversation the user actually had. */
export interface PromptSuggestion {
  /** The chip's label, and the text that lands in the composer when clicked. */
  text: string
  /** The conversation it came from. Identity for dedupe, and the chip's tooltip. */
  sessionId: string
}

export interface PromptSuggestionOptions {
  /** The conversation on screen: never suggest the one you are already in. */
  excludeSessionId?: null | string
  /** How many to return. The composer shows the first few as chips and hands the
   *  rest to the placeholder, so one pool feeds both. */
  limit?: number
  /** Clock injection for tests. */
  now?: number
}

/**
 * Sources that are NOT the user typing. Cron runs, delegate children and
 * `hermes chat -q` one-shots all write sessions with titles, and they are the
 * newest rows in the list, so without this filter the chips fill up with the
 * agent's own work ("Subagent: Fix ONLY these review findings…") on the first
 * busy day. Unknown sources stay in: a new platform adapter shipping tomorrow
 * should not silently empty the row.
 */
const NON_HUMAN_SOURCES = new Set(['cron', 'oneshot', 'subagent', 'worker', 'system', 'bot', 'eval'])

/** Titles the agent writes for itself, whatever the row's source says. */
const INTERNAL_TITLE = /^(subagent:|cronjob response|pulse delivery check|\[system|\[important)/i

/** `SessionInfo.last_active` is epoch SECONDS, not milliseconds: the backend's
 *  own liveness window is `time.time() - last_active < 300`. Comparing it
 *  against a millisecond window makes every row look weeks stale and empties the
 *  chips silently, so the unit is normalised here rather than assumed, and a
 *  backend that ever switches conventions just keeps working. */
function epochSeconds(value: number): number {
  return value > 1e12 ? value / 1000 : value
}

const WINDOW_SECONDS = 45 * 24 * 60 * 60
const MIN_LENGTH = 8
const MAX_LENGTH = 72
const DEFAULT_LIMIT = 6

/**
 * A title as a chip label: one line, no repeat counter, no truncation ellipsis
 * from the titler. Quotes are stripped because a title that wrapped the user's
 * words in them ("send the invoice") reads as a quotation in a button.
 */
function cleanLabel(raw: null | string | undefined): string {
  const text = (raw ?? '')
    .replace(/\s+/g, ' ')
    .replace(/\s+#\d+$/, '')
    .replace(/[.…]+\s*$/, '')
    .replace(/^["'“”‘’]+|["'“”‘’]+$/g, '')
    .trim()

  if (text.length <= MAX_LENGTH) {
    return text
  }

  // Cut on a word boundary: half a word reads as a rendering bug, a whole word
  // plus an ellipsis reads as "here is more of it".
  const clipped = text.slice(0, MAX_LENGTH - 1)
  const lastSpace = clipped.lastIndexOf(' ')

  return `${(lastSpace > MIN_LENGTH * 2 ? clipped.slice(0, lastSpace) : clipped).trimEnd()}…`
}

/** Dedupe identity: case, punctuation and the tail of a long title are noise. */
function dedupeKey(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9 ]+/g, ' ')
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 6)
    .join(' ')
}

/**
 * The user's own recent asks, newest first, for an empty composer.
 *
 * The i18n placeholder pool is generic by necessity (it ships to everyone). This
 * is the personal layer over it: the titles of recent real conversations, so the
 * first thing on screen is a task they recognise.
 *
 * Recency rather than a recurrence count on purpose. A theme ranker over these
 * titles is dominated by whatever the agent itself was doing (delegate children,
 * cron, one-shots) and surfaced the machine's work back at the user; "the last
 * few things you asked" is honest about what it is, needs no model call, and
 * stays fresh without a nightly job.
 */
export function promptSuggestions(
  sessions: readonly SessionInfo[],
  { excludeSessionId, limit = DEFAULT_LIMIT, now = Date.now() }: PromptSuggestionOptions = {}
): PromptSuggestion[] {
  const oldest = epochSeconds(now) - WINDOW_SECONDS
  const seen = new Set<string>()
  const out: PromptSuggestion[] = []

  const rows = [...sessions]
    .filter(session => {
      if (session.id === excludeSessionId || session.hidden) {
        return false
      }

      if (epochSeconds(session.last_active) < oldest) {
        return false
      }

      const source = (session.source ?? '').toLowerCase()

      if (NON_HUMAN_SOURCES.has(source)) {
        return false
      }

      // A session with no reply is a draft, not a conversation: its title is
      // either the first user words verbatim or a placeholder.
      return (session.message_count ?? 0) >= 2
    })
    .sort((a, b) => epochSeconds(b.last_active) - epochSeconds(a.last_active))

  for (const session of rows) {
    const title = session.title ?? ''

    if (INTERNAL_TITLE.test(title)) {
      continue
    }

    const text = cleanLabel(title)

    if (text.length < MIN_LENGTH || text.startsWith('/')) {
      continue
    }

    const key = dedupeKey(text)

    if (!key || seen.has(key)) {
      continue
    }

    seen.add(key)
    out.push({ sessionId: session.id, text })

    if (out.length >= limit) {
      break
    }
  }

  return out
}
