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

// The cron scheduler prepends a delivery/[SILENT] guidance paragraph to every
// job prompt (cron/scheduler.py cron_hint). It's agent plumbing, not something
// the human wrote — strip it so the UI shows only the actual job prompt.
const CRON_HINT_PREFIX = '[IMPORTANT: You are running as a scheduled cron job.'

export function stripCronPromptHint(text: string): string {
  if (!text.startsWith(CRON_HINT_PREFIX)) {
    return text
  }

  // The hint is a single paragraph whose closing `]` sits at a line end
  // (interior `[SILENT]` mentions are mid-sentence, never before a newline).
  const end = text.indexOf(']\n')

  return end === -1 ? '' : text.slice(end + 1).trimStart()
}

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

    const text = stripCronPromptHint(message.text.trim()).trim()

    if (!text || PROCESS_NOTIFICATION_RE.test(text)) {
      continue
    }

    entries.push({ id: message.id, preview: timelinePreview(text) })
  }

  return entries
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
