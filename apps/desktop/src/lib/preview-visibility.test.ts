import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $activeSessionId } from '@/store/session'
import { $sessionTiles, clearAllSessionStates } from '@/store/session-states'

import { sessionIsOnScreen } from './preview-visibility'

describe('sessionIsOnScreen', () => {
  beforeEach(() => {
    clearAllSessionStates()
    $activeSessionId.set(null)
    $sessionTiles.set([])
  })

  afterEach(() => {
    clearAllSessionStates()
    $activeSessionId.set(null)
    $sessionTiles.set([])
  })

  it('is true for the primary active session', () => {
    $activeSessionId.set('runtime-primary')

    expect(sessionIsOnScreen('runtime-primary')).toBe(true)
    expect(sessionIsOnScreen('runtime-hidden')).toBe(false)
  })

  it('is true for a visible session tile that is not the primary', () => {
    $activeSessionId.set('runtime-primary')
    $sessionTiles.set([{ runtimeId: 'runtime-tile', storedSessionId: 'stored-tile' }])

    expect(sessionIsOnScreen('runtime-tile')).toBe(true)
    expect(sessionIsOnScreen('runtime-primary')).toBe(true)
    expect(sessionIsOnScreen('runtime-background')).toBe(false)
  })

  it('is false for an empty session id', () => {
    $activeSessionId.set('runtime-primary')

    expect(sessionIsOnScreen('')).toBe(false)
  })
})
