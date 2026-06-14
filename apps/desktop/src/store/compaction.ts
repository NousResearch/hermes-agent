import { atom, computed } from 'nanostores'

import { $activeSessionId } from './session'

// Status line for sessions whose agent is mid context-compaction, keyed by the
// runtime session id. Auto-compaction fires mid-turn and rewrites history to a
// summary — without a visible signal the transcript looks like it reset itself.
// Per-session (like clarify) so a background chat compacting can't clobber the
// foreground view; cleared when the turn starts, completes, or errors.
const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

export const $compactingSessions = atom<Record<string, string>>({})

// The compaction status for the currently-viewed session, or null. The thread
// loading indicator reads this focus-scoped view to swap to "Summarizing…".
export const $compactionStatus = computed(
  [$compactingSessions, $activeSessionId],
  (sessions, activeId) => sessions[keyFor(activeId)] ?? null
)

export function setSessionCompacting(sessionId: string | null | undefined, status: string | null): void {
  const key = keyFor(sessionId)
  const sessions = $compactingSessions.get()

  if (status) {
    if (sessions[key] === status) {
      return
    }

    $compactingSessions.set({ ...sessions, [key]: status })

    return
  }

  if (!(key in sessions)) {
    return
  }

  const next = { ...sessions }
  delete next[key]
  $compactingSessions.set(next)
}
