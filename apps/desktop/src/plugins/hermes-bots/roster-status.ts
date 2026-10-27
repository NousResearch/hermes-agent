/**
 * One status per roster row, and whether that row belongs in the Needs you
 * band. Pure: every input is a signal the pane already holds (the canonical
 * chat's resolved session status, the prompt that blocks it, the failure
 * badge, a live cron run, the turn/worker activity the face already reads,
 * the room prompt mirror and the room's @user flag), so the band and the
 * glyph can never disagree with each other or ask the gateway for anything.
 *
 * Priority, loudest first: needs you > routine running > working > failed >
 * unread > idle. A row shows exactly one of these.
 */

import type { RosterSessionSignals } from '@hermes/plugin-sdk'

export type BotStatusKind = 'failed' | 'idle' | 'needs-you' | 'routine' | 'unread' | 'working'

/** Why a row needs you. Prompt-derived waits (a clarify, an approval, a
 *  blocked delivery) stay distinct from an unread @user mention: a prompt
 *  clears when it is answered, a mention when the room is read. */
export type NeedsYouReason = 'approval' | 'blocked' | 'clarify' | 'mention'

export interface BotStatus {
  kind: BotStatusKind
  /** Lifted into the Needs you band. Only ever true for `needs-you`. */
  lifts: boolean
  reason?: NeedsYouReason
  /** What the row is waiting on, verbatim: the question, the command, the
   *  failure text, or the line that mentioned you. */
  quote?: string
  /** The row's source answered an earlier paint only; its signals are as old
   *  as that paint, so it is greyed and never lifted. */
  stale: boolean
}

/** Failure classes that wait on a decision rather than on a retry. Only a
 *  delivery the gateway refused (`agent_blocked`) is the user's call to make;
 *  auth/quota/config failures stay a quieter "failed" glyph with their hint. */
const DECISION_FAILURES: ReadonlySet<string> = new Set(['agent_blocked'])

const WORKING_DOTS: ReadonlySet<string> = new Set(['background', 'stalled', 'working'])

export interface BotStatusSignals {
  /** Resolved status of the bot's canonical chat (`needs-input`, `working`, …). */
  dot?: null | RosterSessionSignals['dots'][string]
  prompt?: null | RosterSessionSignals['prompts'][string]
  failure?: null | { message?: string; reason: string }
  routineRunning?: boolean
  /** The roster's own working read (focused turn, live worker, room turn). */
  working?: boolean
  /** `sourceReachable === false`: retained from an earlier paint. */
  stale?: boolean
  /** Hidden from the roster: lifts only while the reveal toggle is on. */
  hidden?: boolean
  revealHidden?: boolean
}

function needsYou(signals: BotStatusSignals): null | Pick<BotStatus, 'quote' | 'reason'> {
  if (signals.prompt) {
    return { reason: signals.prompt.kind, quote: signals.prompt.text }
  }

  if (signals.dot === 'needs-input') {
    // A sudo/secret prompt or a replay whose text has not arrived yet: the
    // wait is real even before there is anything to quote.
    return { reason: 'clarify' }
  }

  if (signals.failure && DECISION_FAILURES.has(signals.failure.reason)) {
    return { reason: 'blocked', quote: signals.failure.message || undefined }
  }

  return null
}

export function deriveBotStatus(signals: BotStatusSignals): BotStatus {
  const stale = Boolean(signals.stale)
  const canLift = !stale && (!signals.hidden || Boolean(signals.revealHidden))
  const waiting = needsYou(signals)

  if (waiting) {
    return { kind: 'needs-you', lifts: canLift, stale, ...waiting }
  }

  if (signals.routineRunning) {
    return { kind: 'routine', lifts: false, stale }
  }

  if (signals.working || WORKING_DOTS.has(String(signals.dot || ''))) {
    return { kind: 'working', lifts: false, stale }
  }

  if (signals.failure) {
    return { kind: 'failed', lifts: false, stale, quote: signals.failure.message || undefined }
  }

  if (signals.dot === 'unread') {
    return { kind: 'unread', lifts: false, stale }
  }

  return { kind: 'idle', lifts: false, stale }
}

export interface GroupStatusSignals {
  /** First pending member prompt in the room ($groupClarify mirror). */
  prompt?: null | { kind: 'approval' | 'clarify'; text: string }
  /** The room's latest member line addressed @user ($groupNeedsYou). */
  mentioned?: boolean
  mentionText?: string
  running?: boolean
}

/** A room lifts for a member's blocking prompt or an unread @user line; the
 *  prompt wins (it is the one that stalls the round). */
export function deriveGroupStatus(signals: GroupStatusSignals): BotStatus {
  if (signals.prompt) {
    return { kind: 'needs-you', lifts: true, stale: false, reason: signals.prompt.kind, quote: signals.prompt.text }
  }

  if (signals.mentioned) {
    return { kind: 'needs-you', lifts: true, stale: false, reason: 'mention', quote: signals.mentionText || undefined }
  }

  return { kind: signals.running ? 'working' : 'idle', lifts: false, stale: false }
}
