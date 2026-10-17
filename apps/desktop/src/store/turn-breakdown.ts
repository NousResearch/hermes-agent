// Per-session turn telemetry — the desktop-side derivations the backend's
// `session.usage` doesn't carry (issue #117224): when the last turn started,
// when it completed, and how much of it tools consumed. Keyed by session id
// and living outside the per-token `$sessionStates` projection so a turn's
// clock survives message-stream churn and is readable from any surface.
//
// Tool seconds accumulate from `tool.complete` payloads (`duration_s`, sent by
// `tui_gateway/tool_progress.py`); model time is then `turn wall clock − tool
// seconds`, reported with a `~` because the two windows overlap (a tool can
// run while the model streams).

import { atom } from 'nanostores'

export interface TurnBreakdown {
  /** Epoch ms the turn started (null while no turn has run). */
  startedAt: number | null
  /** Epoch ms the turn's last `message.complete` fired (null while running). */
  completedAt: number | null
  /** Sum of completed tool calls' `duration_s` for the current/last turn. */
  toolSeconds: number
}

const DEFAULT_BREAKDOWN: TurnBreakdown = { completedAt: null, startedAt: null, toolSeconds: 0 }

export const $turnBreakdownBySession = atom<Record<string, TurnBreakdown>>({})

function updateBreakdown(sessionId: string, updater: (prev: TurnBreakdown) => TurnBreakdown) {
  const prev = $turnBreakdownBySession.get()[sessionId] ?? DEFAULT_BREAKDOWN

  $turnBreakdownBySession.set({ ...$turnBreakdownBySession.get(), [sessionId]: updater(prev) })
}

/** A new turn is starting — reset the accumulation window. */
export function beginTurnBreakdown(sessionId: string) {
  updateBreakdown(sessionId, () => ({ completedAt: null, startedAt: Date.now(), toolSeconds: 0 }))
}

/** A turn ended — freeze the wall clock. */
export function endTurnBreakdown(sessionId: string) {
  updateBreakdown(sessionId, prev => ({ ...prev, completedAt: Date.now() }))
}

/** One tool call finished inside the turn — add its measured duration. */
export function addToolSeconds(sessionId: string, seconds: number) {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return
  }

  updateBreakdown(sessionId, prev => ({ ...prev, toolSeconds: prev.toolSeconds + seconds }))
}

/** Drop one session's breakdown (session archived / runtime torn down). */
export function clearTurnBreakdown(sessionId: string) {
  const next = { ...$turnBreakdownBySession.get() }

  delete next[sessionId]
  $turnBreakdownBySession.set(next)
}
