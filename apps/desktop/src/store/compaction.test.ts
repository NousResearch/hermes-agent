import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $compactingSessions, sessionCompacting, setSessionCompacting } from './compaction'

describe('compaction store', () => {
  beforeEach(() => $compactingSessions.set({}))

  afterEach(() => $compactingSessions.set({}))

  it('tracks session ids that collide with object prototype keys', () => {
    expect(sessionCompacting('toString').get()).toBe(false)

    setSessionCompacting('toString', true)

    expect(Object.hasOwn($compactingSessions.get(), 'toString')).toBe(true)
    expect(sessionCompacting('toString').get()).toBe(true)
  })

  it('scopes the view to the session asked for, not whichever is active', () => {
    setSessionCompacting('session-a', true)

    expect(sessionCompacting('session-a').get()).toBe(true)
    expect(sessionCompacting('session-b').get()).toBe(false)
  })

  it('clears a session without disturbing the others', () => {
    setSessionCompacting('session-a', true)
    setSessionCompacting('session-b', true)

    setSessionCompacting('session-a', false)

    expect($compactingSessions.get()).toEqual({ 'session-b': true })
  })
})
