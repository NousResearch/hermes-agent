import { afterEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $workingSessionIds, clearAllSessionStates, publishSessionState } from '@/store/session-states'

/**
 * The sidebar spinner and the row arc read `$workingSessionIds`, which projects
 * `$sessionStates` down to the ids surfaces actually key on. `message.start`
 * flips `busy` without carrying a stored id, so an unpersisted chat has none
 * yet — and until it does, its runtime id is that id.
 */
describe('$workingSessionIds — a busy runtime with no stored id', () => {
  afterEach(() => {
    clearAllSessionStates()
  })

  it('shows a new chat as running under its runtime id before it is persisted', () => {
    publishSessionState('runtime-unmapped', { ...createClientSessionState(null), busy: true })

    expect($workingSessionIds.get()).toEqual(['runtime-unmapped'])
  })

  it('shows the spinner as soon as the stored id is known', () => {
    publishSessionState('runtime-mapped', { ...createClientSessionState('stored-x'), busy: true })

    expect($workingSessionIds.get()).toEqual(['stored-x'])
  })
})
