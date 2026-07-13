import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $petUnread } from './pet'
import {
  $petLiveSessions,
  acknowledgePetLiveSession,
  beginPetLiveSession,
  clearPetLiveSessionActivity,
  completePetLiveSession,
  reconcilePetLiveSessionFocus,
  replacePetLiveSessionRuntime,
  resetPetLiveSessions,
  setPetLiveSessionActivity,
  syncPetLiveSessionState
} from './pet-live-session'

const state = (
  overrides: Partial<Parameters<typeof syncPetLiveSessionState>[0]> = {}
): Parameters<typeof syncPetLiveSessionState>[0] => ({
  profile: 'default',
  runtimeSessionId: 'runtime',
  storedSessionId: 'stored',
  busy: false,
  needsInput: false,
  awaitingResponse: false,
  turnStartedAt: null,
  ...overrides
})

describe('pet live-session snapshots', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(100)
    resetPetLiveSessions()
    $petUnread.set(false)
  })

  afterEach(() => {
    resetPetLiveSessions()
    $petUnread.set(false)
    vi.useRealTimers()
  })

  it('keeps equal runtime and stored ids isolated by normalized profile', () => {
    syncPetLiveSessionState(state({ busy: true }), { profile: 'default', runtimeSessionId: 'runtime' })
    syncPetLiveSessionState(state({ profile: ' work ', busy: true }), {
      profile: 'work',
      runtimeSessionId: 'runtime'
    })

    expect($petLiveSessions.get()).toEqual([
      expect.objectContaining({ profile: 'default', runtimeSessionId: 'runtime', storedSessionId: 'stored' }),
      expect.objectContaining({ profile: 'work', runtimeSessionId: 'runtime', storedSessionId: 'stored' })
    ])

    setPetLiveSessionActivity('work', 'runtime', 'tool', '  terminal  ')
    completePetLiveSession('work', 'runtime', 'failed')

    expect($petLiveSessions.get().find(item => item.profile === 'default')).toEqual(
      expect.objectContaining({ activityKind: null, outcome: null })
    )
    expect($petLiveSessions.get().find(item => item.profile === 'work')).toEqual(
      expect.objectContaining({ activityKind: null, activityName: null, outcome: 'failed' })
    )
  })

  it('bounds direct activity metadata and never exposes transcript-shaped fields', () => {
    syncPetLiveSessionState(state({ busy: true }), { profile: 'default', runtimeSessionId: 'runtime' })
    setPetLiveSessionActivity('default', 'runtime', 'tool', `  ${'x'.repeat(500)}  `)

    const snapshot = $petLiveSessions.get()[0]!
    const serialized = JSON.stringify(snapshot)

    expect(snapshot.activityName).toHaveLength(120)
    expect(serialized).not.toContain('messages')
    expect(serialized).not.toContain('reasoningText')
    expect(serialized).not.toContain('toolOutput')
    expect(serialized).not.toContain('token')

    clearPetLiveSessionActivity('default', 'runtime', 'tool', 'different')
    expect($petLiveSessions.get()[0]?.activityName).toHaveLength(120)

    clearPetLiveSessionActivity('default', 'runtime', 'tool')
    expect($petLiveSessions.get()[0]?.activityName).toHaveLength(120)

    clearPetLiveSessionActivity('default', 'runtime', 'tool', 'x'.repeat(500))
    expect($petLiveSessions.get()[0]).toEqual(expect.objectContaining({ activityKind: null, activityName: null }))
  })

  it('preserves outcome through a terminal cache update and clears it on the next busy turn', () => {
    syncPetLiveSessionState(state({ busy: true, turnStartedAt: 50 }), {
      profile: 'default',
      runtimeSessionId: 'runtime'
    })
    completePetLiveSession('default', 'runtime', 'done')
    syncPetLiveSessionState(state(), { profile: 'default', runtimeSessionId: 'runtime' })

    expect($petLiveSessions.get()[0]).toEqual(expect.objectContaining({ busy: false, outcome: 'done' }))

    syncPetLiveSessionState(state({ busy: true, turnStartedAt: 110 }), {
      profile: 'default',
      runtimeSessionId: 'runtime'
    })
    expect($petLiveSessions.get()[0]).toEqual(expect.objectContaining({ busy: true, outcome: null }))
  })

  it('begins with a verified stored id and atomically rotates the focused runtime', () => {
    syncPetLiveSessionState(state(), { profile: 'default', runtimeSessionId: 'runtime' })

    replacePetLiveSessionRuntime('default', 'runtime', 'fresh-runtime', 'stored')

    expect($petLiveSessions.get()).toEqual([
      expect.objectContaining({
        awaitingResponse: true,
        busy: true,
        profile: 'default',
        runtimeSessionId: 'fresh-runtime',
        storedSessionId: 'stored'
      })
    ])

    syncPetLiveSessionState(state({ runtimeSessionId: 'fresh-runtime' }))

    expect($petLiveSessions.get()).toEqual([
      expect.objectContaining({
        busy: false,
        runtimeSessionId: 'fresh-runtime',
        storedSessionId: 'stored'
      })
    ])

    resetPetLiveSessions()
    beginPetLiveSession('work', 'new-runtime', 'verified-stored')

    expect($petLiveSessions.get()).toEqual([
      expect.objectContaining({
        busy: true,
        profile: 'work',
        runtimeSessionId: 'new-runtime',
        storedSessionId: 'verified-stored'
      })
    ])
  })

  it('retains exact active idle, prunes prior inactive idle, and syncs a cached new active state', () => {
    syncPetLiveSessionState(state(), { profile: 'default', runtimeSessionId: 'runtime' })
    const activeIdle = $petLiveSessions.get()

    syncPetLiveSessionState(state(), { profile: 'default', runtimeSessionId: 'runtime' })
    expect($petLiveSessions.get()).toBe(activeIdle)

    reconcilePetLiveSessionFocus(
      { profile: 'work', runtimeSessionId: 'work-runtime' },
      state({ profile: 'work', runtimeSessionId: 'work-runtime', storedSessionId: 'work-stored' })
    )

    expect($petLiveSessions.get()).toEqual([
      expect.objectContaining({ profile: 'work', runtimeSessionId: 'work-runtime', storedSessionId: 'work-stored' })
    ])
  })

  it('prunes inactive idle snapshots without outcomes but retains busy, waiting, and outcomes', () => {
    syncPetLiveSessionState(state({ runtimeSessionId: 'idle' }), { profile: 'default', runtimeSessionId: 'idle' })
    syncPetLiveSessionState(state({ runtimeSessionId: 'busy', busy: true }), null)
    syncPetLiveSessionState(state({ runtimeSessionId: 'waiting', needsInput: true }), null)
    syncPetLiveSessionState(state({ runtimeSessionId: 'done' }), { profile: 'default', runtimeSessionId: 'done' })
    completePetLiveSession('default', 'done', 'done')

    reconcilePetLiveSessionFocus(null)

    expect($petLiveSessions.get().map(item => item.runtimeSessionId)).toEqual(['busy', 'waiting', 'done'])
  })

  it('acknowledges only one outcome and clears unread only after the final outcome', () => {
    for (const profile of ['default', 'work']) {
      syncPetLiveSessionState(state({ profile, busy: true }), null)
      completePetLiveSession(profile, 'runtime', profile === 'default' ? 'done' : 'failed')
    }

    $petUnread.set(true)

    acknowledgePetLiveSession('default', 'runtime')

    expect($petLiveSessions.get()).toEqual([expect.objectContaining({ profile: 'work', outcome: 'failed' })])
    expect($petUnread.get()).toBe(true)

    acknowledgePetLiveSession('work', 'runtime')

    expect($petLiveSessions.get()).toEqual([])
    expect($petUnread.get()).toBe(false)
  })
})
