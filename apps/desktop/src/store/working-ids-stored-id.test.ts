import { afterEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $workingSessionIds, clearAllSessionStates, publishSessionState } from '@/store/session-states'

/**
 * (C) The sidebar spinner reads `$workingSessionIds`, which projects
 * `$sessionStates` down to STORED session ids and drops any entry whose
 * `storedSessionId` is null. `message.start` flips `busy` without carrying a
 * stored id, so a runtime that was never seeded with one goes busy invisibly:
 * the backend works, the thread name stays bare.
 */
describe('$workingSessionIds — a busy runtime with no stored id', () => {
  afterEach(() => {
    clearAllSessionStates()
  })

  it('cannot show a spinner for a busy runtime that has no stored id', () => {
    publishSessionState('runtime-unmapped', { ...createClientSessionState(null), busy: true })

    // Documents the constraint rather than asserting the bug is fine: the
    // projection is keyed by stored id, so an unmapped runtime is unreachable
    // from the sidebar no matter how busy it is.
    expect($workingSessionIds.get()).toEqual([])
  })

  it('shows the spinner as soon as the stored id is known', () => {
    publishSessionState('runtime-mapped', { ...createClientSessionState('stored-x'), busy: true })

    expect($workingSessionIds.get()).toEqual(['stored-x'])
  })
})
