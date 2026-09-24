// Pure timeline helpers — no React/DOM; tested in thread-timeline-data.test.ts.
import { PROCESS_NOTIFICATION_RE } from './content'

export interface TimelineSourceMessage {
  id: string
  rowId?: number
  role: string
  text: string
}

export interface TimelineEntry {
  id: string
  rowId?: number
  preview: string
}

const PREVIEW_MAX = 120

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

  return a.every((entry, index) => entry.id === b[index].id && entry.preview === b[index].preview)
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
