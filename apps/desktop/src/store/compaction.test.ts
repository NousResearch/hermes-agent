import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $compactingSessions, $compactionStatus, setSessionCompacting } from './compaction'
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

  it('tracks compaction status per session independently', () => {
    setSessionCompacting('session-a', 'Summarizing a…')
    setSessionCompacting('session-b', 'Summarizing b…')

    expect($compactingSessions.get()['session-a']).toBe('Summarizing a…')
    expect($compactingSessions.get()['session-b']).toBe('Summarizing b…')
  })

  it('exposes only the active session via the focus-scoped view', () => {
    setSessionCompacting('session-a', 'Summarizing a…')

    expect($compactionStatus.get()).toBeNull()

    $activeSessionId.set('session-a')
    expect($compactionStatus.get()).toBe('Summarizing a…')

    $activeSessionId.set('session-b')
    expect($compactionStatus.get()).toBeNull()
  })

  it('clears a session without disturbing the others', () => {
    setSessionCompacting('session-a', 'Summarizing a…')
    setSessionCompacting('session-b', 'Summarizing b…')

    setSessionCompacting('session-a', null)

    expect($compactingSessions.get()['session-a']).toBeUndefined()
    expect($compactingSessions.get()['session-b']).toBe('Summarizing b…')
  })

  it('is a no-op when clearing an unknown session', () => {
    setSessionCompacting('session-a', 'Summarizing a…')
    const before = $compactingSessions.get()

    setSessionCompacting('session-missing', null)

    expect($compactingSessions.get()).toBe(before)
  })
})
