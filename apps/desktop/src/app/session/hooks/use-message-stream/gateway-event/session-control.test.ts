import { afterEach, describe, expect, it, vi } from 'vitest'

import { $sessionControlBySession, clearAllSessionControl, type SessionControlSnapshot } from '@/store/session-control'

import { handleControlEvent } from './session-control'
import type { GatewayEventContext } from './types'

const SNAPSHOT: SessionControlSnapshot = {
  goal: null,
  heartbeat: null,
  loop: null,
  revision: 'event-revision',
  updated_at: 1_700_000_100
}

function context(overrides: Partial<GatewayEventContext> = {}): GatewayEventContext {
  return {
    deps: {} as GatewayEventContext['deps'],
    event: { payload: { control: SNAPSHOT }, session_id: 'event-session', type: 'session.control.update' },
    explicitSid: 'event-session',
    fromActiveSource: () => true,
    isActiveEvent: true,
    occurredAt: 1_700_000_100,
    payload: { control: SNAPSHOT } as GatewayEventContext['payload'],
    scheduleConfigRefresh: vi.fn(),
    sessionId: 'routed-session',
    ...overrides
  }
}

describe('handleControlEvent', () => {
  afterEach(() => {
    clearAllSessionControl()
  })

  it('does not claim non-control events', () => {
    expect(handleControlEvent(context({ event: { type: 'message.complete' } }))).toBe(false)
  })

  it('applies a valid event to the routed session and proves it supported', () => {
    expect(handleControlEvent(context())).toBe(true)

    expect($sessionControlBySession.get()).toMatchObject({
      'routed-session': { capability: 'supported', snapshot: { revision: 'event-revision' } }
    })
    expect($sessionControlBySession.get()['event-session']).toBeUndefined()
  })

  it('claims malformed control events without replacing the last good state', () => {
    handleControlEvent(context())
    const first = $sessionControlBySession.get()['routed-session']

    expect(
      handleControlEvent(
        context({
          payload: { control: { ...SNAPSHOT, updated_at: Number.POSITIVE_INFINITY } } as GatewayEventContext['payload']
        })
      )
    ).toBe(true)

    expect($sessionControlBySession.get()['routed-session']).toBe(first)
  })

  it('claims an unscoped control event without creating state', () => {
    expect(handleControlEvent(context({ sessionId: null }))).toBe(true)
    expect($sessionControlBySession.get()).toEqual({})
  })
})
