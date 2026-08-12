// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest'

import { onPersistenceEvent, type PersistenceEvent } from '@/lib/storage'

import {
  $translucency,
  $translucencyMode,
  GLASS_SUPPORTED,
  glassSurfaceKeep,
  setTranslucency,
  setTranslucencyMode,
  TRANSLUCENCY_MAX,
  TRANSLUCENCY_MIN,
  TRANSLUCENCY_STEP
} from './translucency'

const KEY = 'hermes.desktop.translucency.v1'

describe('window translucency lever', () => {
  beforeEach(() => {
    setTranslucency(TRANSLUCENCY_MIN)
    setTranslucencyMode('clear')
  })

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

  it('mirrors intensity and mode to the desktop bridge', () => {
    const calls: Array<{ intensity: number; mode?: string }> = []
    window.hermesDesktop = {
      setTranslucency: (payload: { intensity: number; mode?: 'clear' | 'glass' }) => calls.push(payload)
    } as never

    setTranslucency(40)
    expect(calls.at(-1)).toEqual({ intensity: 40, mode: 'clear' })
  })

  it('rejects glass off macOS and applies it on macOS', () => {
    setTranslucency(50)
    setTranslucencyMode('glass')

    if (GLASS_SUPPORTED) {
      expect($translucencyMode.get()).toBe('glass')
      expect(document.documentElement.hasAttribute('data-hermes-glass')).toBe(true)
      expect(document.documentElement.style.getPropertyValue('--translucency-glass-keep')).toBe('65%')
    } else {
      expect($translucencyMode.get()).toBe('clear')
      expect(document.documentElement.hasAttribute('data-hermes-glass')).toBe(false)
    }
  })

  it('removes the glass attribute at zero intensity or back on clear', () => {
    setTranslucency(50)
    setTranslucencyMode('glass')
    setTranslucency(0)
    expect(document.documentElement.hasAttribute('data-hermes-glass')).toBe(false)

    setTranslucency(50)
    setTranslucencyMode('clear')
    expect(document.documentElement.hasAttribute('data-hermes-glass')).toBe(false)
  })

  it('marks clear mode on <html> so the overlay scrim can compensate', () => {
    setTranslucency(50)
    setTranslucencyMode('clear')
    expect(document.documentElement.hasAttribute('data-hermes-clear')).toBe(true)

    setTranslucency(0)
    expect(document.documentElement.hasAttribute('data-hermes-clear')).toBe(false)

    setTranslucency(50)

    if (GLASS_SUPPORTED) {
      setTranslucencyMode('glass')
      expect(document.documentElement.hasAttribute('data-hermes-clear')).toBe(false)
    }
  })
})

describe('glassSurfaceKeep', () => {
  it('mirrors the clear-mode opacity ramp with its 30% floor', () => {
    expect(glassSurfaceKeep(0)).toBe(100)
    expect(glassSurfaceKeep(50)).toBe(65)
    expect(glassSurfaceKeep(100)).toBeCloseTo(30)
  })
})
