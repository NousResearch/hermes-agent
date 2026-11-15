import { afterEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'

import {
  $petActivity,
  $petAtRest,
  $petMotion,
  $petSpriteTarget,
  $petState,
  clearPetReaction,
  derivePetState,
  flashPetActivity,
  hasPetSpriteForMeta,
  mergePetInfoMeta,
  reactPet,
  setPetActivity,
  setPetHandling,
  setPetInfo
} from './pet'
import { $activeSessionId, $busy } from './session'
import { clearAllSessionStates, publishSessionState } from './session-states'

describe('derivePetState', () => {
  it('rests at idle by default and uses waiting when awaiting input', () => {
    expect(derivePetState({})).toBe('idle')
    expect(derivePetState({ awaitingInput: true })).toBe('waiting')
  })

  it('runs when busy or a tool is executing', () => {
    expect(derivePetState({ busy: true })).toBe('run')
    expect(derivePetState({ toolRunning: true })).toBe('run')
  })

  it('reviews while reasoning (below tool, above bare busy)', () => {
    expect(derivePetState({ reasoning: true })).toBe('review')
    expect(derivePetState({ reasoning: true, busy: true })).toBe('review')
    expect(derivePetState({ reasoning: true, toolRunning: true })).toBe('run')
  })

  it('waits (blocked on the user) above the in-flight signals', () => {
    expect(derivePetState({ awaitingInput: true, toolRunning: true, busy: true })).toBe('waiting')
    // but a finish beat still wins over waiting
    expect(derivePetState({ justCompleted: true, awaitingInput: true })).toBe('wave')
  })

  it('honors the full priority chain: error > celebrate > complete > tool', () => {
    expect(derivePetState({ error: true, celebrate: true, busy: true })).toBe('failed')
    expect(derivePetState({ celebrate: true, justCompleted: true, toolRunning: true })).toBe('jump')
    expect(derivePetState({ justCompleted: true, toolRunning: true })).toBe('wave')
  })
})

describe('roam motion', () => {
  it('only reports at-rest when the agent-driven state is plain idle', () => {
    $petActivity.set({})
    expect($petAtRest.get()).toBe(true)

    $petActivity.set({ busy: true })
    expect($petAtRest.get()).toBe(false)

    $petActivity.set({})
    expect($petAtRest.get()).toBe(true)
  })

  it('shows the roam pose while wandering, but never overrides real activity', () => {
    $petActivity.set({})
    $petMotion.set('run')
    expect($petState.get()).toBe('run')

    // Hops surface the jump pose.
    $petMotion.set('jump')
    expect($petState.get()).toBe('jump')

    // Activity wins over a wander in progress.
    $petActivity.set({ reasoning: true, busy: true })
    expect($petState.get()).toBe('review')

    // Back at rest, the wander resumes its pose; clearing it returns to idle.
    $petActivity.set({})
    expect($petState.get()).toBe('jump')
    $petMotion.set(null)
    expect($petState.get()).toBe('idle')

    $petActivity.set({})
  })
})

describe('pet info metadata cache helpers', () => {
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

    expect(hasPetSpriteForMeta(current, meta)).toBe(true)
    expect(mergePetInfoMeta(current, meta)).toMatchObject({
      enabled: true,
      slug: 'boba',
      displayName: 'Boba',
      scale: 0.5,
      spritesheetBase64: 'large-sprite-payload',
      spritesheetRevision: '100:2048'
    })
  })
})

describe('flashPetActivity', () => {
  it('clears stale sibling beats so a completion never inherits a prior error', () => {
    // A turn errors (sad), then the next turn finishes cleanly. The celebrate
    // beat must win — error is highest priority, so a merge-only flash would
    // keep the pet on the failed pose.
    setPetActivity({ error: true })
    flashPetActivity({ celebrate: true })

    expect($petActivity.get().error).toBe(false)
    expect($petState.get()).toBe('jump')

    setPetActivity({})
  })
})

describe('$petState reads the active runtime slice, not the $busy mirror (#84434 / #84438)', () => {
  const runtimeId = 'runtime-live'
  const slice = (busy: boolean) => ({ ...createClientSessionState(null), busy, storedSessionId: 'stored-live' })

  afterEach(() => {
    clearAllSessionStates()
    $activeSessionId.set(null)
    $busy.set(false)
    $petActivity.set({})
  })

  it('follows the active slice, not the $busy mirror: runs while it is busy, idles the moment it settles', () => {
    $activeSessionId.set(runtimeId)
    publishSessionState(runtimeId, slice(true))
    $petActivity.set({ toolRunning: true })

    expect($petState.get()).toBe('run')
    expect($petAtRest.get()).toBe(false)

    // An interrupted turn flips busy→false without a tool.complete / message.complete.
    publishSessionState(runtimeId, slice(false))
    expect($petState.get()).toBe('idle')
    expect($petAtRest.get()).toBe(true)
  })
})

describe('sleepy (pet.json behaviors.sleep)', () => {
  it('naps only at rest; a napping pet neither wanders nor shows the roam pose', () => {
    expect(derivePetState({ sleepy: true, busy: true })).toBe('run')
    expect(derivePetState({ sleepy: true, awaitingInput: true })).toBe('waiting')

    $petActivity.set({ sleepy: true })
    $petMotion.set('jump')

    expect($petAtRest.get()).toBe(false)
    expect($petState.get()).toBe('sleep')

    $petActivity.set({})
    $petMotion.set(null)
  })
})

describe('$petSpriteTarget: handling states and plugin reactions over the live state', () => {
  const pose = () => $petSpriteTarget.get().state

  afterEach(() => {
    vi.useRealTimers()
    setPetHandling(null)
    clearPetReaction('a')
    clearPetReaction('b')
    $petActivity.set({})
    setPetInfo({ enabled: false })
  })

  it('plays a reaction over idle / working for one loop, never over failed, waiting or sleep', () => {
    vi.useFakeTimers()
    setPetInfo({ enabled: true, loopMs: 1000 })

    reactPet('a', 'wave')
    expect(pose()).toBe('wave')
    vi.advanceTimersByTime(999)
    expect(pose()).toBe('wave')
    vi.advanceTimersByTime(1)
    expect(pose()).toBe('idle')

    $petActivity.set({ busy: true })
    reactPet('a', 'jump', 500)
    expect(pose()).toBe('jump')

    for (const [activity, state] of [
      [{ error: true }, 'failed'],
      [{ awaitingInput: true }, 'waiting'],
      [{ sleepy: true }, 'sleep']
    ] as const) {
      $petActivity.set(activity)
      reactPet('a', 'jump', 500)
      expect(pose()).toBe(state)
    }
  })

  it("cancels early, and an unload clears only that plugin's reaction", () => {
    const cancel = reactPet('a', 'wave', 5000)
    cancel()
    expect(pose()).toBe('idle')

    reactPet('a', 'wave', 5000)
    reactPet('b', 'jump', 5000)
    clearPetReaction('a')
    expect(pose()).toBe('jump')
    clearPetReaction('b')
    expect(pose()).toBe('idle')
  })

  it('a mapped handling state outranks everything visually but never reaches $petState', () => {
    setPetHandling('poke')
    expect(pose()).toBe('idle') // unmapped: recorded, not drawn

    setPetInfo({ enabled: true, behaviors: { rows: { held: ['dangle'] } } })
    $petActivity.set({ awaitingInput: true })
    reactPet('a', 'wave', 5000)
    setPetHandling('held')

    expect(pose()).toBe('held')
    // The bubble and the approval flow keep reading the real state.
    expect($petState.get()).toBe('waiting')

    setPetHandling(null)
    expect(pose()).toBe('waiting')
  })
})
