import { afterEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $gatewayState } from '@/store/session'
import { $sessionStates, dropSessionState, publishSessionState } from '@/store/session-states'

import { host } from './index'

describe('host.state busy vs gateway', () => {
  afterEach(() => {
    $sessionStates.set({})
    $gatewayState.set('idle')
  })

  it('exposes per-session turn-busy and does not treat gateway as busy', () => {
    const running = { ...createClientSessionState('stored-a'), busy: true }
    const idle = { ...createClientSessionState('stored-b'), busy: false }

    publishSessionState('runtime-a', running)
    publishSessionState('runtime-b', idle)
    $gatewayState.set('open')

    expect(host.state.busyBySession.get()).toEqual({ 'runtime-a': true, 'runtime-b': false })
    expect(host.state.gateway.get()).toBe('open')

    dropSessionState('runtime-a')
    expect(host.state.busyBySession.get()['runtime-a']).toBeUndefined()
    expect(host.state.gateway.get()).toBe('open')
  })
})
