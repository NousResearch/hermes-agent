export type SessionDotState = 'background' | 'idle' | 'needs-input' | 'stalled' | 'unread' | 'working'

interface SessionRowState {
  hasBackground: boolean
  isStalled: boolean
  isUnread: boolean
  isWorking: boolean
  needsInput: boolean
}

/** Resolve the sidebar dot's mutually-exclusive display state by priority. */
export function sessionDotState({
  hasBackground,
  isStalled,
  isUnread,
  isWorking,
  needsInput
}: SessionRowState): SessionDotState {
  if (needsInput) {
    return 'needs-input'
  }

  if (isWorking) {
    return isStalled ? 'stalled' : 'working'
  }

  if (hasBackground) {
    return 'background'
  }

  return isUnread ? 'unread' : 'idle'
}

/** A quiet turn is still authoritatively running. Keep the unmistakable row
 * arc until the gateway reports completion; only a blocking prompt suppresses
 * it in favour of the needs-input treatment. */
export function sessionShowsRunningArc({
  isWorking,
  needsInput
}: Pick<SessionRowState, 'isWorking' | 'needsInput'>): boolean {
  return isWorking && !needsInput
}
