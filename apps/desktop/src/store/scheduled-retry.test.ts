import { beforeEach, describe, expect, it } from 'vitest'

import { parseClockTime } from '@/components/assistant-ui/thread/scheduled-retry-action'

import {
  $scheduledRetries,
  decodeScheduledRetries,
  pruneScheduledRetries,
  sessionScheduledRetry,
  setScheduledRetry
} from './scheduled-retry'

const STORAGE_KEY = 'hermes.desktop.scheduledRetries'

const T0 = new Date('2026-08-30T15:00:00').getTime()

describe('scheduled retry store', () => {
  beforeEach(() => {
    window.localStorage.removeItem(STORAGE_KEY)
    $scheduledRetries.set({})
  })

  it('schedules one retry per session and persists it', () => {
    setScheduledRetry('s1', { at: T0, messageId: 'm1', sessionId: 's1' })

    expect(sessionScheduledRetry('s1').get()).toEqual({ at: T0, messageId: 'm1', sessionId: 's1' })
    expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY) || 'null')).toEqual({
      s1: { at: T0, messageId: 'm1', sessionId: 's1' }
    })
  })

  it('reschedules and cancels', () => {
    setScheduledRetry('s1', { at: T0, messageId: 'm1', sessionId: 's1' })
    setScheduledRetry('s1', { at: T0 + 1000, messageId: 'm1', sessionId: 's1' })
    expect(sessionScheduledRetry('s1').get()?.at).toBe(T0 + 1000)

    setScheduledRetry('s1', null)
    expect(sessionScheduledRetry('s1').get()).toBeNull()
    expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull()
  })

  it('scopes lookups per session and ignores null ids', () => {
    setScheduledRetry('s1', { at: T0, messageId: 'm1', sessionId: 's1' })

    expect(sessionScheduledRetry('s2').get()).toBeNull()
    expect(sessionScheduledRetry(null).get()).toBeNull()
  })

  it('drops schedules whose message no longer exists', () => {
    setScheduledRetry('s1', { at: T0, messageId: 'm1', sessionId: 's1' })
    setScheduledRetry('s2', { at: T0, messageId: 'm2', sessionId: 's2' })

    pruneScheduledRetries(new Set(['m2']))

    expect(sessionScheduledRetry('s1').get()).toBeNull()
    expect(sessionScheduledRetry('s2').get()).not.toBeNull()
  })

  it('rehydrates from storage and discards garbled shapes', () => {
    const restored = decodeScheduledRetries(
      JSON.stringify({
        s1: { at: T0, messageId: 'm1', sessionId: 's1' },
        s2: { at: 'not-a-number', messageId: 'm2', sessionId: 's2' },
        s3: 'garbage'
      })
    )

    expect(Object.keys(restored)).toEqual(['s1'])
    expect(restored.s1).toEqual({ at: T0, messageId: 'm1', sessionId: 's1' })
  })
})

describe('parseClockTime', () => {
  it('parses HH:mm later today', () => {
    const at = parseClockTime('18:30', T0)
    expect(at).not.toBeNull()
    expect(new Date(at!).getHours()).toBe(18)
    expect(new Date(at!).getMinutes()).toBe(30)
  })

  it('rolls a past time to tomorrow', () => {
    const at = parseClockTime('09:00', T0)
    expect(at).not.toBeNull()
    expect(at!).toBeGreaterThan(T0)
    expect(new Date(at!).getHours()).toBe(9)
  })

  it('accepts 24:00-style edge as tomorrow midnight', () => {
    const at = parseClockTime('23:59', T0)
    expect(at).not.toBeNull()
    expect(new Date(at!).getMinutes()).toBe(59)
  })

  it('accepts HHmm without a colon', () => {
    const at = parseClockTime('1830', T0)
    expect(at).not.toBeNull()
    expect(new Date(at!).getHours()).toBe(18)
  })

  it('rejects malformed values', () => {
    expect(parseClockTime('25:00', T0)).toBeNull()
    expect(parseClockTime('12:61', T0)).toBeNull()
    expect(parseClockTime('soon', T0)).toBeNull()
    expect(parseClockTime('', T0)).toBeNull()
  })
})
