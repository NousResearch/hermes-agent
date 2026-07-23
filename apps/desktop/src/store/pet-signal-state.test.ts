import { afterEach, describe, expect, it, vi } from 'vitest'

import { $petActivity, $petAtRest, $petMotion, $petState, deriveEffectivePetState } from './pet'
import { $petSignals, clearPetSignals, upsertPetSignal } from './pet-signals'

let signalSourceSequence = 0

const clearTestSignals = () => {
  for (const source of new Set($petSignals.get().map(signal => signal.source))) {
    clearPetSignals(source)
  }
}

afterEach(() => {
  clearTestSignals()
  $petActivity.set({})
  $petMotion.set(null)
  vi.useRealTimers()
})

describe('native and external pet state precedence', () => {
  it('uses external work and thinking only while native Hermes is idle', () => {
    expect(deriveEffectivePetState({}, false, 'run')).toBe('run')
    expect(deriveEffectivePetState({}, false, 'review')).toBe('review')
  })

  it('keeps native work above an external blocked signal', () => {
    expect(deriveEffectivePetState({ toolRunning: true }, true, 'waiting')).toBe('run')
    expect(deriveEffectivePetState({ reasoning: true }, true, 'waiting')).toBe('review')
  })

  it('keeps native approval, error, celebration, and completion above external work', () => {
    expect(deriveEffectivePetState({ awaitingInput: true }, true, 'run')).toBe('waiting')
    expect(deriveEffectivePetState({ error: true }, false, 'run')).toBe('failed')
    expect(deriveEffectivePetState({ celebrate: true }, false, 'run')).toBe('jump')
    expect(deriveEffectivePetState({ justCompleted: true }, false, 'run')).toBe('wave')
  })

  it('lets a live external signal drive the pet and stop idle roaming', () => {
    const source = `pet-state-${++signalSourceSequence}`
    $petActivity.set({})
    $petMotion.set('jump')

    upsertPetSignal({
      createdAt: Date.now(),
      expiresAt: Date.now() + 10_000,
      id: 'job',
      priority: 0,
      source,
      state: 'working'
    })

    expect($petAtRest.get()).toBe(false)
    expect($petState.get()).toBe('run')

    clearPetSignals(source)
    expect($petAtRest.get()).toBe(true)
    expect($petState.get()).toBe('jump')
  })

  it('returns both surfaces to the native roam state when a failure expires', () => {
    vi.useFakeTimers()
    vi.setSystemTime(5_000)
    const source = `pet-expiry-${++signalSourceSequence}`
    $petActivity.set({})
    $petMotion.set('jump')

    upsertPetSignal({
      createdAt: 5_000,
      expiresAt: 5_100,
      id: 'failure',
      priority: 10,
      source,
      state: 'failed'
    })

    expect($petState.get()).toBe('failed')
    expect($petAtRest.get()).toBe(false)

    vi.advanceTimersByTime(100)
    expect($petState.get()).toBe('jump')
    expect($petAtRest.get()).toBe(true)
  })

  it('preserves all existing native behavior when no signal exists', () => {
    expect(deriveEffectivePetState({ reasoning: true }, true, 'idle')).toBe('review')
    expect(deriveEffectivePetState({}, false, 'idle')).toBe('idle')
  })
})
