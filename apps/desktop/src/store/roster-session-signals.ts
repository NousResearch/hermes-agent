/**
 * ROSTER SESSION SIGNALS — the read-only view a roster (Bot Mode's Bots pane)
 * needs to say what each agent is doing without a new RPC:
 *
 * - `dots`: the one resolved status per stored session ($sessionDotStateById),
 *   so a bot row keyed on its canonical chat agrees with every other surface.
 * - `prompts`: the blocking prompt itself, per stored session (every lineage
 *   alias), so a row can QUOTE what it is waiting on — the clarify question or
 *   the command an approval would run. Prompts are parked under the runtime
 *   id that raised them; the runtime→stored bridge is the same one the dot
 *   state uses.
 * - `routines`: owner scopes with a cron run still in flight: not ended, and
 *   owned by the scheduler. The session LIST does not stamp `scheduler_owned`
 *   (only the session-detail read does), so a listed run falls back to the
 *   backend's 300 s `is_active` window, which goes dark during a tool call
 *   longer than that.
 *
 * Every input is reference-stable across stream deltas, so this recomputes on
 * prompt/status edges, not per token.
 */

import { computed } from 'nanostores'

import { stableArray, stableRecord } from '@/lib/stable-array'
import type { SessionInfo } from '@/types/hermes'

import { $clarifyRequests } from './clarify'
import { $approvalRequests } from './prompts'
import { $cronSessions, $sessions, lineageAliases } from './session'
import { $sessionDotStateById, type SessionDotState } from './session-dot-state'
import { $sessionStates } from './session-states'

export interface SessionPromptQuote {
  kind: 'approval' | 'clarify'
  /** The question asked, or the command the approval would run. */
  text: string
}

export interface RosterSessionSignals {
  dots: Readonly<Record<string, SessionDotState>>
  prompts: Readonly<Record<string, SessionPromptQuote>>
  /** `${connection_id ?? ''}::${profile}` for every cron run still in flight. */
  routines: readonly string[]
}

export const routineScopeKey = (connectionId: null | string | undefined, profile: null | string | undefined) =>
  `${String(connectionId || '').trim()}::${String(profile || 'default').trim() || 'default'}`

const cronRunInFlight = (row: SessionInfo) =>
  row.ended_at == null &&
  (typeof row.scheduler_owned === 'boolean' ? row.scheduler_owned : row.is_active === true)

let prevPrompts: Readonly<Record<string, SessionPromptQuote>> = {}
let prevRoutines: readonly string[] = []
let prevSignals: RosterSessionSignals | null = null

export const $rosterSessionSignals = computed(
  [$sessionDotStateById, $clarifyRequests, $approvalRequests, $sessionStates, $sessions, $cronSessions],
  (dots, clarifies, approvals, states, sessions, cron): RosterSessionSignals => {
    const prompts: Record<string, SessionPromptQuote> = {}

    // A still-live prompt keeps the quote object the last paint held, so
    // stableRecord holds; an answered one drops out with its text.
    const claim = (runtimeId: string, kind: SessionPromptQuote['kind'], text: string) => {
      if (!runtimeId) {
        return
      }

      for (const alias of lineageAliases(states[runtimeId]?.storedSessionId ?? runtimeId, sessions)) {
        const prev = prevPrompts[alias]

        prompts[alias] = prev?.kind === kind && prev.text === text ? prev : Object.freeze({ kind, text })
      }
    }

    for (const [runtimeId, request] of Object.entries(clarifies)) {
      // The question the bot still waits on: a batch answered one question at
      // a time is waiting on its first UNLOCKED one, as the composer reads it.
      const text = request.questions.find(q => q.question && request.lockedAnswers?.[q.qid] === undefined)?.question

      if (text) {
        claim(runtimeId, 'clarify', text)
      }
    }

    // Approval wins over a clarify on the same session: it names the command
    // that will run the moment the user says yes.
    for (const [runtimeId, request] of Object.entries(approvals)) {
      const text = String(request.command || request.description || '').trim()

      if (text) {
        claim(runtimeId, 'approval', text)
      }
    }

    const routines = cron
      .filter(cronRunInFlight)
      .map(row => routineScopeKey(row.connection_id, row.profile))

    const nextPrompts = (prevPrompts = stableRecord(prevPrompts, prompts))
    const nextRoutines = (prevRoutines = stableArray(prevRoutines, [...new Set(routines)].sort()))

    if (
      prevSignals &&
      prevSignals.dots === dots &&
      prevSignals.prompts === nextPrompts &&
      prevSignals.routines === nextRoutines
    ) {
      return prevSignals
    }

    return (prevSignals = Object.freeze({ dots, prompts: nextPrompts, routines: nextRoutines }))
  }
)
