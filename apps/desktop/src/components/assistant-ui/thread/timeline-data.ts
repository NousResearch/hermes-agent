// Pure timeline helpers — no React/DOM; tested in thread-timeline-data.test.ts.

export interface TimelineSourceMessage {
  id: string
  role: string
  text: string
}

export interface TimelineEntry {
  id: string
  preview: string
}

// Injected as user messages for alternation; not human prompts (thread.tsx).
const PROCESS_NOTIFICATION_RE = /^\[IMPORTANT: Background process [\s\S]*\]$/
const COMPACTION_HANDOFF_RE = /^\[CONTEXT (?:COMPACTION|SUMMARY)/

const PREVIEW_MAX = 120

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

    if (!text || PROCESS_NOTIFICATION_RE.test(text) || COMPACTION_HANDOFF_RE.test(text)) {
      continue
    }

    entries.push({ id: message.id, preview: timelinePreview(text) })
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
