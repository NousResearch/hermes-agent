import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $petActivity,
  $petAtRest,
  $petBubble,
  $petControls,
  $petDismissed,
  $petInfo,
  $petMotion,
  $petState,
  derivePetState,
  flashPetActivity,
  setPetActivity,
  setPetBubble,
  setPetControls,
  type PetInfo
} from './pet'

import { setPetEnabled, $petGallery, type PetGallery } from './pet-gallery'

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

describe('pet disable via setPetEnabled — Hide button regression (#61267)', () => {
  // Regression: the Hide button in FloatingPet must route through
  // setPetEnabled(request, false, ...) so pet.disable fires on the
  // gateway. Without it, $petDismissed never receives the enabled:false
  // response it waits for and stays locked forever.
  //
  // This test verifies the store-level contract: after setPetEnabled
  // completes with on=false, pet.info is called via syncInfo and
  // $petInfo reflects the disabled state. The FloatingPet poll guard
  // relies on this to clear $petDismissed and resume normal polling.

  beforeEach(() => {
    $petGallery.set({
      enabled: true,
      active: 'boba',
      pets: [{ slug: 'boba', displayName: 'Boba', installed: true }]
    })
    $petInfo.set({ enabled: true, slug: 'boba', displayName: 'Boba' })
    $petDismissed.set(false)
  })

  it('calls pet.disable and updates $petInfo to enabled:false', async () => {
    const calls: string[] = []

    const mockRequest = <T>(method: string): Promise<T> => {
      calls.push(method)

      if (method === 'pet.info') {
        return Promise.resolve({ enabled: false, slug: 'boba' } as T)
      }

      return Promise.resolve({} as T)
    }

    const ok = await setPetEnabled(mockRequest, false, {
      noneAvailable: '',
      fallback: 'Could not turn the pet off.'
    })

    expect(ok).toBe(true)
    expect(calls).toContain('pet.disable')
    expect(calls).toContain('pet.info')
    expect($petInfo.get().enabled).toBe(false)
  })

  it('$petDismissed clearable when pet info reports disabled', () => {
    // Simulate what the FloatingPet poll guard does after a Hide:
    // $petDismissed is set true, then a poll returns enabled:false
    $petDismissed.set(true)
    $petInfo.set({ enabled: false, slug: 'boba' })

    // Guard logic (mirrors floating-pet.tsx ~line 188):
    if ($petDismissed.get() && !$petInfo.get().enabled) {
      $petDismissed.set(false)
    }

    expect($petDismissed.get()).toBe(false)
  })

  it('$petDismissed stays true when poll still reports enabled', () => {
    // If the gateway never received pet.disable, it keeps returning
    // enabled:true. The guard would never clear $petDismissed.
    $petDismissed.set(true)
    $petInfo.set({ enabled: true, slug: 'boba' })

    // Guard logic — enabled stays true, so dismiss flag persists
    if ($petDismissed.get() && !$petInfo.get().enabled) {
      $petDismissed.set(false)
    }

    expect($petDismissed.get()).toBe(true)
  })
})
