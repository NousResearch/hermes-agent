import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $compactingSessions, $compactionActive, setSessionCompacting } from './compaction'
import { $activeSessionId } from './session'

describe('compaction store', () => {
  beforeEach(() => {
    $compactingSessions.set({})
    $activeSessionId.set(null)
  })

  afterEach(() => {
    $compactingSessions.set({})
    $activeSessionId.set(null)
  })

  it('tracks compaction per session independently', () => {
    setSessionCompacting('session-a', true)
    setSessionCompacting('session-b', true)

    expect($compactingSessions.get()).toEqual({ 'session-a': true, 'session-b': true })
  })

  it('exposes only the active session via the focus-scoped view', () => {
    setSessionCompacting('session-a', true)

    expect($compactionActive.get()).toBe(false)

    $activeSessionId.set('session-a')
    expect($compactionActive.get()).toBe(true)

    $activeSessionId.set('session-b')
    expect($compactionActive.get()).toBe(false)
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

  it('does not report a prototype-named session as active (#58441)', () => {
    // 'toString' is on Object.prototype; an empty map must not look like it
    // already contains it, and setting it must store an own key.
    expect(Object.hasOwn($compactingSessions.get(), 'toString')).toBe(false)

    setSessionCompacting('toString', true)

    expect($compactingSessions.get().toString).toBe(true)
  })
})
