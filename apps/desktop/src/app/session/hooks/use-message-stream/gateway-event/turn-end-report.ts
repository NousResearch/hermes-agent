import type { ClientSessionState } from '../../../../types'

import type { GatewayEventContext } from './types'

/*
 * Every turn ends with message.complete and then the turn worker's end-of-turn
 * session.info (running=false). Core sends that report after it has released
 * the session, so a turn started right after the completion (a queued send the
 * composer drains at once) can see its message.start first. message.complete
 * marks the report owed (ClientSessionState.turnEndReportOwed); these rules
 * keep the late report from settling the turn now live.
 */

/**
 * Whether a session.info's `running` field is an edge for the session's turn.
 * A completed turn's end-of-turn report that arrives after the next turn's
 * message.start says nothing about the turn now live: it neither clears that
 * turn's prompts nor settles it, and only pays the owed mark.
 */
export function runningEdge(
  { deps, payload, sessionId }: GatewayEventContext,
  knownState: ClientSessionState | undefined,
  runningChanged: boolean
): boolean {
  const endedTurnReport = payload?.running === false && Boolean(knownState?.turnEndReportOwed && knownState.turnLive)

  if (endedTurnReport && sessionId) {
    deps.updateSessionState(
      sessionId,
      state => ({ ...state, turnEndReportOwed: false }),
      payload?.stored_session_id || undefined
    )
  }

  return runningChanged && !endedTurnReport
}

/** A running=false report pays the end report a completion left owed. */
export const payTurnEndReport = (state: ClientSessionState, busy: boolean): ClientSessionState =>
  !busy && state.turnEndReportOwed ? { ...state, turnEndReportOwed: false } : state
