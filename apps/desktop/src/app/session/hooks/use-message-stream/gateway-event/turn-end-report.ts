import type { ClientSessionState } from '../../../../types'

import type { GatewayEventContext } from './types'

/*
 * Every turn ends with message.complete and then the turn worker's end-of-turn
 * session.info (running=false). Core sends that report after it has released
 * the session, so a turn started right after the completion (a queued send the
 * composer drains at once) can see its message.start first. message.complete
 * marks the report owed (ClientSessionState.turnEndReportsOwed); these rules
 * keep a late report from settling the turn now live. Two short turns can both
 * complete before the first one's report arrives, so each completion owes its
 * own, and each running=false report pays one.
 */

/**
 * How long after its message.complete a turn's end-of-turn report is still
 * expected. Core sends it right after the completion (0.24-0.63 s measured over
 * a sandboxed backend). One that has not come by then is not coming (lost in a
 * reconnect, or it came ahead of a stopped turn's completion), and its mark must
 * not take a later turn's own report.
 */
export const TURN_END_REPORT_WAIT_MS = 30_000

const owedReports = (state: ClientSessionState | undefined, now: number): number[] =>
  (state?.turnEndReportsOwed ?? []).filter(completedAt => now - completedAt < TURN_END_REPORT_WAIT_MS)

/**
 * message.complete: this turn's end-of-turn report is still to come. Not when
 * the report already settled the turn ahead of this frame (#119569): the turn is
 * then neither live nor stopped here.
 */
export const oweTurnEndReport = (state: ClientSessionState, now = Date.now()): ClientSessionState =>
  state.turnLive || state.interrupted ? { ...state, turnEndReportsOwed: [...owedReports(state, now), now] } : state

const payOne = (state: ClientSessionState, now: number): ClientSessionState =>
  state.turnEndReportsOwed?.length ? { ...state, turnEndReportsOwed: owedReports(state, now).slice(1) } : state

/**
 * Whether a session.info's `running` field is an edge for the session's turn.
 * A completed turn's end-of-turn report that arrives after the next turn's
 * message.start says nothing about the turn now live: it neither ends that
 * turn's compaction phase, nor clears its prompts, nor settles it, and only
 * pays one owed mark.
 */
export function runningEdge(
  { deps, payload, sessionId }: GatewayEventContext,
  knownState: ClientSessionState | undefined,
  runningChanged: boolean
): boolean {
  const now = Date.now()

  const endedTurnReport =
    payload?.running === false && Boolean(knownState?.turnLive) && owedReports(knownState, now).length > 0

  if (endedTurnReport && sessionId) {
    deps.updateSessionState(sessionId, state => payOne(state, now), payload?.stored_session_id || undefined)
  }

  return runningChanged && !endedTurnReport
}

/** A running=false report pays one end report a completion left owed. */
export const payTurnEndReport = (state: ClientSessionState, busy: boolean): ClientSessionState =>
  busy ? state : payOne(state, Date.now())
