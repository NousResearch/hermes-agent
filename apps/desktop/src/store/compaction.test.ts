import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $compactingSessions, sessionCompacting, setSessionCompacting } from './compaction'

describe('compaction store', () => {
  beforeEach(() => $compactingSessions.set({}))

  afterEach(() => $compactingSessions.set({}))

  it('tracks compaction per session independently', () => {
    setSessionCompacting('session-a', true)
    setSessionCompacting('session-b', true)

    expect($compactingSessions.get()).toEqual({ 'session-a': true, 'session-b': true })
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

  it('is a no-op when clearing an unknown session', () => {
    setSessionCompacting('session-a', true)
    const before = $compactingSessions.get()

    setSessionCompacting('session-missing', false)

    expect($compactingSessions.get()).toBe(before)
  })
})
