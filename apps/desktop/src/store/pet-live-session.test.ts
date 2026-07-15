import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $petUnread } from './pet'
import {
  $petLiveSessions,
  acknowledgePetLiveSession,
  appendPetLiveSessionReply,
  armNextReplyForProfile,
  armPetLiveSessionReply,
  beginPetLiveSession,
  beginPetLiveSessionReply,
  clearPetLiveSessionActivity,
  completePetLiveSession,
  completePetLiveSessionReply,
  consumePetReplyArm,
  disarmNextReplyForProfile,
  disarmPetReplyArm,
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

  it('consumes an exact session reply arm only once', () => {
    armPetLiveSessionReply(' default ', ' runtime ')

    expect(consumePetReplyArm('work', 'runtime')).toBe(false)
    expect(consumePetReplyArm('default', 'runtime')).toBe(true)
    expect(consumePetReplyArm('default', 'runtime')).toBe(false)
  })

  it('consumes a profile-level next-session reply arm only once', () => {
    armNextReplyForProfile(' work ')

    expect(consumePetReplyArm('default', 'new-runtime')).toBe(false)
    expect(consumePetReplyArm('work', 'new-runtime')).toBe(true)
    expect(consumePetReplyArm('work', 'another-runtime')).toBe(false)
  })

  it('consumes an exact arm before preserving a separate next-session arm for the same profile', () => {
    armPetLiveSessionReply('work', 'exact-runtime')
    armNextReplyForProfile('work')

    expect(consumePetReplyArm('work', 'exact-runtime')).toBe(true)
    expect(consumePetReplyArm('work', 'new-runtime')).toBe(true)
  })

  it('disarms failed exact-session and next-session captures without touching siblings', () => {
    armPetLiveSessionReply('default', 'runtime')
    armPetLiveSessionReply('default', 'sibling')
    armNextReplyForProfile('work')

    disarmPetReplyArm('default', 'runtime')
    disarmNextReplyForProfile('work')

    expect(consumePetReplyArm('default', 'runtime')).toBe(false)
    expect(consumePetReplyArm('work', 'new-runtime')).toBe(false)
    expect(consumePetReplyArm('default', 'sibling')).toBe(true)
  })

  it('preserves streamed text when a fast message.start re-begins an armed pet reply', () => {
    beginPetLiveSessionReply('default', 'runtime', 'stored')
    appendPetLiveSessionReply('default', 'runtime', 'Already streamed')

    beginPetLiveSessionReply('default', 'runtime', 'stored')

    expect($petLiveSessions.get()[0]).toEqual(
      expect.objectContaining({ reply: { streaming: true, text: 'Already streamed' } })
    )
  })

  it('uses streamed text when the terminal payload has no final text', () => {
    beginPetLiveSessionReply('default', 'runtime', 'stored')
    appendPetLiveSessionReply('default', 'runtime', 'streamed fallback')

    completePetLiveSessionReply('default', 'runtime', '   ')

    expect($petLiveSessions.get()[0]?.reply).toEqual({ streaming: false, text: 'streamed fallback' })
  })

  it('drops an unfinished pet reply when the next turn is not pet-armed', () => {
    beginPetLiveSessionReply('default', 'runtime', 'stored')
    appendPetLiveSessionReply('default', 'runtime', 'stale partial')

    beginPetLiveSession('default', 'runtime', 'stored')
    appendPetLiveSessionReply('default', 'runtime', 'ordinary reply')

    expect($petLiveSessions.get()[0]?.reply).toBeNull()
  })

  it('clears pending reply arms when live sessions reset', () => {
    armPetLiveSessionReply('default', 'runtime')
    armNextReplyForProfile('work')

    resetPetLiveSessions()

    expect(consumePetReplyArm('default', 'runtime')).toBe(false)
    expect(consumePetReplyArm('work', 'new-runtime')).toBe(false)
  })

  it('clears reply arming when a turn completes', () => {
    beginPetLiveSessionReply('default', 'runtime', 'stored')

    completePetLiveSession('default', 'runtime', 'done')

    expect($petLiveSessions.get()[0]).toEqual(expect.objectContaining({ replyArmed: false }))
  })

  it('captures only the exact pet-started turn reply and clears it on the next ordinary turn', () => {
    syncPetLiveSessionState(state(), { profile: 'default', runtimeSessionId: 'runtime' })
    syncPetLiveSessionState(state({ profile: 'work' }), null)

    beginPetLiveSessionReply('default', 'runtime', 'stored')
    appendPetLiveSessionReply('work', 'runtime', 'wrong profile')
    appendPetLiveSessionReply('default', 'different-runtime', 'wrong runtime')
    appendPetLiveSessionReply('default', 'runtime', 'Hello ')
    appendPetLiveSessionReply('default', 'runtime', 'from Nox')

    expect($petLiveSessions.get().find(item => item.profile === 'default')).toEqual(
      expect.objectContaining({ reply: { streaming: true, text: 'Hello from Nox' } })
    )
    expect(JSON.stringify($petLiveSessions.get())).not.toContain('wrong profile')
    expect(JSON.stringify($petLiveSessions.get())).not.toContain('wrong runtime')

    completePetLiveSessionReply('default', 'runtime', 'Final answer')
    completePetLiveSession('default', 'runtime', 'done')

    expect($petLiveSessions.get().find(item => item.profile === 'default')).toEqual(
      expect.objectContaining({ reply: { streaming: false, text: 'Final answer' } })
    )

    beginPetLiveSession('default', 'runtime', 'stored')

    expect($petLiveSessions.get().find(item => item.profile === 'default')).toEqual(
      expect.objectContaining({ reply: null })
    )
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
