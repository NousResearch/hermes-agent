import { beforeEach, describe, expect, it } from 'vitest'

import { onPersistenceEvent, type PersistenceEvent } from '@/lib/storage'

import { $translucency, setTranslucency, TRANSLUCENCY_MAX, TRANSLUCENCY_MIN, TRANSLUCENCY_STEP } from './translucency'

const KEY = 'hermes.desktop.translucency.v1'

describe('window translucency lever', () => {
  beforeEach(() => setTranslucency(TRANSLUCENCY_MIN))

  it('steps in single percent so the readable low end is reachable', () => {
    expect(TRANSLUCENCY_STEP).toBe(1)
    expect(TRANSLUCENCY_MIN).toBe(0)
    expect(TRANSLUCENCY_MAX).toBe(100)
  })

  it('defaults to off', () => {
    expect($translucency.get()).toBe(TRANSLUCENCY_MIN)
  })

  it('accepts every step the slider can emit', () => {
    for (let intensity = TRANSLUCENCY_MIN; intensity <= TRANSLUCENCY_MAX; intensity += TRANSLUCENCY_STEP) {
      setTranslucency(intensity)
      expect($translucency.get()).toBe(intensity)
    }
  })

  it('clamps out-of-range input to the lever bounds', () => {
    setTranslucency(-40)
    expect($translucency.get()).toBe(TRANSLUCENCY_MIN)

    setTranslucency(240)
    expect($translucency.get()).toBe(TRANSLUCENCY_MAX)
  })

  it('rounds fractional input to a whole percent', () => {
    setTranslucency(35.6)
    expect($translucency.get()).toBe(36)
  })

  it('persists under a stable key so a saved setting survives upgrades', () => {
    const writes: PersistenceEvent[] = []

    const stop = onPersistenceEvent(event => {
      if (event.op === 'write') {
        writes.push(event)
      }
    })

    try {
      setTranslucency(23)
    } finally {
      stop()
    }

    expect(writes).toContainEqual({ key: KEY, op: 'write', value: '23' })
  })
})
