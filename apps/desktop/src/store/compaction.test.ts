import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $compactingSessions,
  $compactionProgressById,
  sessionCompacting,
  sessionCompactionProgress,
  setSessionCompacting,
  setSessionCompactionProgress
} from './compaction'

describe('compaction store', () => {
  beforeEach(() => {
    $compactingSessions.set({})
    $compactionProgressById.set({})
  })

  afterEach(() => {
    $compactingSessions.set({})
    $compactionProgressById.set({})
  })

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

  it('keeps progress per session and clears it with the compacting flag', () => {
    setSessionCompacting('session-a', true)
    setSessionCompactionProgress('session-a', 'compressing 42 messages (~120,000 tok)…')
    setSessionCompacting('session-b', true)
    setSessionCompactionProgress('session-b', 'compressing 8 messages…')

    expect(sessionCompactionProgress('session-a').get()?.text).toBe('compressing 42 messages (~120,000 tok)…')
    expect(sessionCompactionProgress('session-b').get()?.text).toBe('compressing 8 messages…')

    setSessionCompacting('session-a', false)

    expect(sessionCompactionProgress('session-a').get()).toBeNull()
    expect(sessionCompactionProgress('session-b').get()).not.toBeNull()
  })

  it('replaces the progress text without moving the timer origin', () => {
    setSessionCompacting('session-a', true)
    setSessionCompactionProgress('session-a', 'compressing 42 messages…')
    const since = sessionCompactionProgress('session-a').get()?.since

    setSessionCompactionProgress('session-a', 'compressing 17 messages…')

    const progress = sessionCompactionProgress('session-a').get()
    expect(progress?.text).toBe('compressing 17 messages…')
    expect(progress?.since).toBe(since)
  })
})
