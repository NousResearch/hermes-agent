import { describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'

import {
  clearSettled,
  clearSettledGrace,
  deadLightweightKeys,
  getRecentlySettledSessionIds,
  LIGHTWEIGHT_STATES_CAP,
  markSettled,
  sweepDeadLightweightStates
} from './session-states-expiry'

const idle = (over: Partial<ClientSessionState> = {}): ClientSessionState =>
  ({
    storedSessionId: 's',
    messages: [],
    busy: false,
    awaitingResponse: false,
    needsInput: false,
    ...over
  }) as unknown as ClientSessionState

describe('getRecentlySettledSessionIds', () => {
  it('keeps ids for the grace window, then prunes on read', () => {
    vi.useFakeTimers()
    vi.setSystemTime(0)

    try {
      clearSettledGrace()
      markSettled('s1')
      expect(getRecentlySettledSessionIds()).toEqual(['s1'])

      vi.setSystemTime(29_000)
      expect(getRecentlySettledSessionIds()).toEqual(['s1'])

      vi.setSystemTime(31_000)
      expect(getRecentlySettledSessionIds()).toEqual([])
    } finally {
      vi.useRealTimers()
      clearSettledGrace()
    }
  })

  it('clearSettled drops a single id', () => {
    clearSettledGrace()
    markSettled('s1')
    markSettled('s2')
    clearSettled('s1')
    expect(getRecentlySettledSessionIds()).toEqual(['s2'])
    clearSettledGrace()
  })
})

describe('deadLightweightKeys', () => {
  it('returns nothing under the cap', () => {
    expect(deadLightweightKeys({ rt: idle() }, () => false)).toEqual([])
  })

  it('sweeps only fully-idle unreferenced entries, oldest-first', () => {
    const states: Record<string, ClientSessionState> = {}

    for (let i = 0; i < LIGHTWEIGHT_STATES_CAP + 3; i += 1) {
      states[`rt-${i}`] = idle({ storedSessionId: `s-${i}` })
    }

    // Pin the middle: a tile reference, a busy turn, and an attention dot
    // must all survive the sweep.
    states['rt-1'] = idle({ storedSessionId: 's-1' })
    const isReferenced = (runtimeId: string) => runtimeId === 'rt-1'
    states['rt-2'] = idle({ storedSessionId: 's-2', busy: true }) as ClientSessionState
    states[`rt-${LIGHTWEIGHT_STATES_CAP + 2}`] = idle({
      storedSessionId: 's-x',
      needsInput: true
    }) as ClientSessionState

    const dead = deadLightweightKeys(states, isReferenced)

    expect(dead).toHaveLength(3)
    expect(dead).not.toContain('rt-1')
    expect(dead).not.toContain('rt-2')
    expect(dead).not.toContain(`rt-${LIGHTWEIGHT_STATES_CAP + 2}`)
    // Oldest-first: the first key goes before later ones.
    expect(dead[0]).toBe('rt-0')
  })

  it('sweepDeadLightweightStates returns null when nothing qualifies', () => {
    expect(sweepDeadLightweightStates({ rt: idle({ busy: true }) }, () => false)).toBeNull()
    expect(sweepDeadLightweightStates({ rt: idle() }, () => false)).toBeNull()
  })
})
