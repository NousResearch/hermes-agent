import { afterEach, describe, expect, it } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'

import {
  $companionActivity,
  $companionAtRest,
  $companionMotion,
  $companionState,
  deriveCompanionState,
  flashCompanionActivity,
  hasCompanionSpriteForMeta,
  mergeCompanionInfoMeta,
  setCompanionActivity
} from './companion'
import { $activeSessionId, $busy } from './session'
import { clearAllSessionStates, publishSessionState } from './session-states'

describe('deriveCompanionState', () => {
  it('rests at idle by default and uses waiting when awaiting input', () => {
    expect(deriveCompanionState({})).toBe('idle')
    expect(deriveCompanionState({ awaitingInput: true })).toBe('waiting')
  })

  it('runs when busy or a tool is executing', () => {
    expect(deriveCompanionState({ busy: true })).toBe('run')
    expect(deriveCompanionState({ toolRunning: true })).toBe('run')
  })

  it('reviews while reasoning (below tool, above bare busy)', () => {
    expect(deriveCompanionState({ reasoning: true })).toBe('review')
    expect(deriveCompanionState({ reasoning: true, busy: true })).toBe('review')
    expect(deriveCompanionState({ reasoning: true, toolRunning: true })).toBe('run')
  })

  it('waits (blocked on the user) above the in-flight signals', () => {
    expect(deriveCompanionState({ awaitingInput: true, toolRunning: true, busy: true })).toBe('waiting')
    // but a finish beat still wins over waiting
    expect(deriveCompanionState({ justCompleted: true, awaitingInput: true })).toBe('wave')
  })

  it('honors the full priority chain: error > celebrate > complete > tool', () => {
    expect(deriveCompanionState({ error: true, celebrate: true, busy: true })).toBe('failed')
    expect(deriveCompanionState({ celebrate: true, justCompleted: true, toolRunning: true })).toBe('jump')
    expect(deriveCompanionState({ justCompleted: true, toolRunning: true })).toBe('wave')
  })
})

describe('roam motion', () => {
  it('only reports at-rest when the agent-driven state is plain idle', () => {
    $companionActivity.set({})
    expect($companionAtRest.get()).toBe(true)

    $companionActivity.set({ busy: true })
    expect($companionAtRest.get()).toBe(false)

    $companionActivity.set({})
    expect($companionAtRest.get()).toBe(true)
  })

  it('shows the roam pose while wandering, but never overrides real activity', () => {
    $companionActivity.set({})
    $companionMotion.set('run')
    expect($companionState.get()).toBe('run')

    // Hops surface the jump pose.
    $companionMotion.set('jump')
    expect($companionState.get()).toBe('jump')

    // Activity wins over a wander in progress.
    $companionActivity.set({ reasoning: true, busy: true })
    expect($companionState.get()).toBe('review')

    // Back at rest, the wander resumes its pose; clearing it returns to idle.
    $companionActivity.set({})
    expect($companionState.get()).toBe('jump')
    $companionMotion.set(null)
    expect($companionState.get()).toBe('idle')

    $companionActivity.set({})
  })
})

describe('companion info metadata cache helpers', () => {
  it('treats matching slug and spritesheet revision as a reusable sprite payload', () => {
    const current = {
      enabled: true,
      slug: 'boba',
      displayName: 'Old Boba',
      scale: 0.33,
      spritesheetBase64: 'large-sprite-payload',
      spritesheetRevision: '100:2048'
    }

    const meta = {
      enabled: true,
      slug: 'boba',
      displayName: 'Boba',
      scale: 0.5,
      spritesheetRevision: '100:2048'
    }

    expect(hasCompanionSpriteForMeta(current, meta)).toBe(true)
    expect(mergeCompanionInfoMeta(current, meta)).toMatchObject({
      enabled: true,
      slug: 'boba',
      displayName: 'Boba',
      scale: 0.5,
      spritesheetBase64: 'large-sprite-payload',
      spritesheetRevision: '100:2048'
    })
  })
})

describe('flashCompanionActivity', () => {
  it('clears stale sibling beats so a completion never inherits a prior error', () => {
    // A turn errors (sad), then the next turn finishes cleanly. The celebrate
    // beat must win — error is highest priority, so a merge-only flash would
    // keep the companion on the failed pose.
    setCompanionActivity({ error: true })
    flashCompanionActivity({ celebrate: true })

    expect($companionActivity.get().error).toBe(false)
    expect($companionState.get()).toBe('jump')

    setCompanionActivity({})
  })
})

describe('$companionState reads the active runtime slice, not the $busy mirror (#84434 / #84438)', () => {
  const runtimeId = 'runtime-live'
  const slice = (busy: boolean) => ({ ...createClientSessionState(null), busy, storedSessionId: 'stored-live' })

  afterEach(() => {
    clearAllSessionStates()
    $activeSessionId.set(null)
    $busy.set(false)
    $companionActivity.set({})
  })

  it('follows the active slice, not the $busy mirror: runs while it is busy, idles the moment it settles', () => {
    $activeSessionId.set(runtimeId)
    publishSessionState(runtimeId, slice(true))
    $companionActivity.set({ toolRunning: true })

    expect($companionState.get()).toBe('run')
    expect($companionAtRest.get()).toBe(false)

    // An interrupted turn flips busy→false without a tool.complete / message.complete.
    publishSessionState(runtimeId, slice(false))
    expect($companionState.get()).toBe('idle')
    expect($companionAtRest.get()).toBe(true)
  })
})
