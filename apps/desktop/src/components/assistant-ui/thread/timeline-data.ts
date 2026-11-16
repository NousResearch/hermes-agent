// Pure timeline helpers — no React/DOM; tested in thread-timeline-data.test.ts.
import { PROCESS_NOTIFICATION_RE } from './content'

export interface TimelineSourceMessage {
  id: string
  rowId?: number
  role: string
  text: string
  /** The persisted row's `display_kind`, when the source carries it. */
  displayKind?: string
}

/** Who wrote a prompt: the user, or Hermes injecting a turn (cron, kanban,
 *  background results, gateway notices). */
export type TimelineAuthor = 'human' | 'system'

export interface TimelineEntry {
  id: string
  rowId?: number
  preview: string
  /** Absent only on the "Show earlier" placeholder. */
  author?: TimelineAuthor
  /** 1-based position among human prompts; absent on system entries and on
   *  entries whose place in the whole conversation is not yet known. */
  number?: number
}

const PREVIEW_MAX = 120

/** Typed rows that are still the user's own words (a mid-turn /steer, a skill invocation). */
const HUMAN_DISPLAY_KINDS: readonly string[] = ['steer', 'skill_invocation']

/**
 * Machine-injected prompts persisted without a `display_kind` (cron runs and
 * deliveries, kanban notices). Keep in step with `_MACHINE_PROMPT` in
 * hermes_state_timeline.py, which classifies the server timeline index.
 */
export const MACHINE_PROMPT_RE =
  /^\s*(?:\[IMPORTANT: You are running as a scheduled cron job\b|\[Cronjob "|\[kanban\b|\[IMPORTANT: Background process )/

export function timelineAuthor(text: string, displayKind?: string): TimelineAuthor {
  if (displayKind) {
    return HUMAN_DISPLAY_KINDS.includes(displayKind) ? 'human' : 'system'
  }

  return MACHINE_PROMPT_RE.test(text) ? 'system' : 'human'
}

/**
 * Number human prompts 1..N in rail order. Only the first `known` entries sit
 * at a known place in the whole conversation (the index pages oldest-first);
 * entries past that point are left unnumbered rather than numbered from a gap.
 * Takes unnumbered entries; the "Show earlier" placeholder has no author and is skipped.
 */
export function numberTimelineEntries(
  entries: readonly TimelineEntry[],
  known: number = entries.length
): TimelineEntry[] {
  let count = 0

  return entries.map((entry, index) =>
    entry.author === 'human' && index < known ? { ...entry, number: ++count } : entry
  )
}

/** Shared, localized focus curve; the active tick always retains its maximum. */
export function timelineBarWidth(index: number, activeIndex: number, hoverIndex: number | null): number {
  const curve = (distance: number) => (Math.abs(distance) > 3 ? 0 : Math.exp(-(distance ** 2) / 1.35))

  return 0.5 + 0.5 * Math.max(curve(index - activeIndex), hoverIndex === null ? 0 : curve(index - hoverIndex))
}

export const TIMELINE_REVEAL_EVENT = 'hermes:timeline-reveal'
export const EARLIER_TIMELINE_ID = '__earlier-history__'

export interface TimelineRevealRequest {
  id: string
  rowId?: number
  signal: AbortSignal
  complete: (revealedId: string | false) => void
}

export function timelinePreview(text: string, max: number = PREVIEW_MAX): string {
  const collapsed = text.replace(/\s+/g, ' ').trim()

  if (collapsed.length <= max) {
    return collapsed
  }

  return `${collapsed.slice(0, max - 1).trimEnd()}…`
}

export function deriveTimelineEntries(messages: readonly TimelineSourceMessage[]): TimelineEntry[] {
  const entries: TimelineEntry[] = []

  for (const message of messages) {
    if (message.role !== 'user') {
      continue
    }

    const text = message.text.trim()

    if (!text || PROCESS_NOTIFICATION_RE.test(text)) {
      continue
    }

    entries.push({
      id: message.id,
      preview: timelinePreview(text),
      author: timelineAuthor(text, message.displayKind),
      ...(message.rowId !== undefined ? { rowId: message.rowId } : {})
    })
  }

  return entries
}

/** Do two derivations describe the same rail? Lets a rebuild hand back the
 *  PREVIOUS array so an unchanged transcript costs zero re-renders. */
export function sameTimelineEntries(a: readonly TimelineEntry[], b: readonly TimelineEntry[]): boolean {
  if (a === b) {
    return true
  }

  if (a.length !== b.length) {
    return false
  }

  return a.every(
    (entry, index) =>
      entry.id === b[index].id &&
      entry.preview === b[index].preview &&
      entry.author === b[index].author &&
      entry.number === b[index].number
  )
}

/** Last user prompt at/above the viewport top (with slack); else first rendered. */
export function activeTimelineIndex(offsets: readonly (number | null)[], slack: number = 8): number {
  let active = -1
  let firstRendered = -1

  for (let i = 0; i < offsets.length; i++) {
    const offset = offsets[i]

    if (offset == null) {
      continue
    }

    if (firstRendered === -1) {
      firstRendered = i
    }

    if (offset <= slack) {
      active = i
    }
  }

  if (active !== -1) {
    return active
  }

  return firstRendered === -1 ? 0 : firstRendered
}
