import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { sameCronSignature, shouldPollMessagingSessions } from './desktop-controller-utils'

const session = (id: string, title: string | null): SessionInfo => ({ id, title }) as SessionInfo

describe('sameCronSignature', () => {
  it('is false when the lengths differ', () => {
    expect(sameCronSignature([session('a', 't')], [])).toBe(false)
  })

  it('is true when ids and titles match in order', () => {
    const a = [session('a', 'one'), session('b', 'two')]
    const b = [session('a', 'one'), session('b', 'two')]
    expect(sameCronSignature(a, b)).toBe(true)
  })

  it('is false when a title changed', () => {
    const a = [session('a', 'one')]
    const b = [session('a', 'renamed')]
    expect(sameCronSignature(a, b)).toBe(false)
  })

  it('is false when order differs', () => {
    const a = [session('a', 't'), session('b', 't')]
    const b = [session('b', 't'), session('a', 't')]
    expect(sameCronSignature(a, b)).toBe(false)
  })
})

describe('shouldPollMessagingSessions', () => {
  const base = {
    activeIsMessaging: false,
    gatewayPlatforms: null,
    messagingSessionCount: 0,
    messagingViewOpen: false
  }

  it('is false when nothing indicates messaging is in use', () => {
    expect(shouldPollMessagingSessions(base)).toBe(false)
  })

  it('is true while the messaging view is open', () => {
    expect(shouldPollMessagingSessions({ ...base, messagingViewOpen: true })).toBe(true)
  })

  it('is true when a messaging session is already loaded', () => {
    expect(shouldPollMessagingSessions({ ...base, messagingSessionCount: 1 })).toBe(true)
  })

  it('is true when the active session is a messaging thread', () => {
    expect(shouldPollMessagingSessions({ ...base, activeIsMessaging: true })).toBe(true)
  })

  it('is true when status reports a configured gateway platform', () => {
    expect(
      shouldPollMessagingSessions({
        ...base,
        gatewayPlatforms: {
          telegram: {
            state: 'connected',
            updated_at: '2026-07-04T00:00:00+00:00'
          }
        }
      })
    ).toBe(true)
  })
})
