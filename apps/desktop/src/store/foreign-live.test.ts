import { afterEach, describe, expect, it } from 'vitest'

import { $foreignLiveSessionIds } from './foreign-live'
import { $sessions } from './session'
import { $sessionDotStateById } from './session-dot-state'
import { $sessionStates } from './session-states'

const row = (id: string, isActive: boolean) => ({
  id,
  is_active: isActive,
  last_active: Date.now() / 1000,
  message_count: 3,
  profile: 'default',
  source: 'cli'
})

afterEach(() => {
  $sessions.set([])
  $sessionStates.set({})
})

describe('$foreignLiveSessionIds', () => {
  it('claims is_active rows with no runtime in this renderer', () => {
    $sessions.set([row('s-cli', true) as never])
    $sessionStates.set({})
    expect($foreignLiveSessionIds.get()).toContain('s-cli')
  })

  it('does NOT claim rows that have a runtime (event-owned)', () => {
    $sessions.set([row('s-desktop', true) as never])
    $sessionStates.set({
      'runtime-1': { storedSessionId: 's-desktop', busy: true } as never
    })
    expect($foreignLiveSessionIds.get()).not.toContain('s-desktop')
  })

  it('drops the claim when is_active goes false', () => {
    $sessions.set([row('s-cli', true) as never])
    $sessionStates.set({})
    expect($foreignLiveSessionIds.get()).toContain('s-cli')
    $sessions.set([row('s-cli', false) as never])
    expect($foreignLiveSessionIds.get()).not.toContain('s-cli')
  })

  it('paints the working dot through the dot-state pipeline', () => {
    $sessions.set([row('s-cron', true) as never])
    $sessionStates.set({})
    expect($sessionDotStateById.get()['s-cron']).toBe('working')
  })
})
