// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest'

import { onPersistenceEvent, type PersistenceEvent } from '@/lib/storage'

import {
  $translucency,
  $translucencyPeek,
  beginTranslucencyPeek,
  DEFAULT_GLASS_MATERIAL,
  DEFAULT_GLASS_SCOPE,
  endTranslucencyPeek,
  GLASS_SUPPORTED,
  isChatWindow,
  setTranslucency,
  setTranslucencyMaterial,
  setTranslucencyMode,
  setTranslucencyScope,
  TRANSLUCENCY_MAX,
  TRANSLUCENCY_MIN,
  TRANSLUCENCY_STEP
} from './translucency'

const KEY = 'hermes.desktop.translucency.v1'

const glassAttr = () => document.documentElement.hasAttribute('data-hermes-glass')
const clearAttr = () => document.documentElement.hasAttribute('data-hermes-clear')
const keep = () => document.documentElement.style.getPropertyValue('--translucency-glass-keep')

describe('window translucency lever', () => {
  beforeEach(() => {
    setTranslucencyMode('clear')
    setTranslucency(TRANSLUCENCY_MIN)
  })

  it('steps in single percent so the readable low end is reachable', () => {
    expect(TRANSLUCENCY_STEP).toBe(1)
    expect(TRANSLUCENCY_MIN).toBe(0)
    expect(TRANSLUCENCY_MAX).toBe(100)
  })

  it('defaults to off and clear', () => {
    expect($translucency.get()).toEqual({
      intensity: TRANSLUCENCY_MIN,
      mode: 'clear',
      material: DEFAULT_GLASS_MATERIAL,
      scope: DEFAULT_GLASS_SCOPE
    })
  })

  it('accepts every step the slider can emit', () => {
    for (let intensity = TRANSLUCENCY_MIN; intensity <= TRANSLUCENCY_MAX; intensity += TRANSLUCENCY_STEP) {
      setTranslucency(intensity)
      expect($translucency.get().intensity).toBe(intensity)
    }
  })

  it('clamps out-of-range input and rounds fractions', () => {
    setTranslucency(-40)
    expect($translucency.get().intensity).toBe(TRANSLUCENCY_MIN)

    setTranslucency(240)
    expect($translucency.get().intensity).toBe(TRANSLUCENCY_MAX)

    setTranslucency(35.6)
    expect($translucency.get().intensity).toBe(36)
  })

  it('changing one half of the state leaves the other alone', () => {
    setTranslucency(42)
    setTranslucencyMode('glass')
    expect($translucency.get().intensity).toBe(42)

    setTranslucency(43)
    expect($translucency.get().mode).toBe(GLASS_SUPPORTED ? 'glass' : 'clear')
  })

  it('persists intensity and mode together under one stable key', () => {
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

    expect(writes.at(-1)).toEqual({
      key: KEY,
      op: 'write',
      value: JSON.stringify({
        intensity: 23,
        mode: 'clear',
        material: DEFAULT_GLASS_MATERIAL,
        scope: DEFAULT_GLASS_SCOPE
      })
    })
  })

  it('mirrors the whole state to the desktop bridge', () => {
    const calls: unknown[] = []
    window.hermesDesktop = { setTranslucency: (payload: unknown) => calls.push(payload) } as never

    setTranslucency(40)
    expect(calls.at(-1)).toEqual({
      intensity: 40,
      mode: 'clear',
      material: DEFAULT_GLASS_MATERIAL,
      scope: DEFAULT_GLASS_SCOPE
    })
  })
})

describe('glass mode', () => {
  beforeEach(() => {
    setTranslucencyMode('clear')
    setTranslucency(TRANSLUCENCY_MIN)
  })

  it('rejects glass off macOS and applies it on macOS', () => {
    setTranslucency(50)
    setTranslucencyMode('glass')

    if (GLASS_SUPPORTED) {
      expect($translucency.get().mode).toBe('glass')
      expect(glassAttr()).toBe(true)
      expect(keep()).toBe('50%')
    } else {
      expect($translucency.get().mode).toBe('clear')
      expect(glassAttr()).toBe(false)
    }
  })

  it('drops the glass attribute at zero intensity or back on clear', () => {
    setTranslucency(50)
    setTranslucencyMode('glass')

    setTranslucency(0)
    expect(glassAttr()).toBe(false)
    expect(keep()).toBe('')

    setTranslucency(50)
    setTranslucencyMode('clear')
    expect(glassAttr()).toBe(false)
  })

  // Clear and glass are mutually exclusive page states: clear strengthens the
  // overlay scrim, glass thins the field. Both at once would fight.
  it('marks clear mode separately, and never both at once', () => {
    setTranslucency(50)
    expect(clearAttr()).toBe(true)
    expect(glassAttr()).toBe(false)

    setTranslucencyMode('glass')

    if (GLASS_SUPPORTED) {
      expect(clearAttr()).toBe(false)
      expect(glassAttr()).toBe(true)
    }

    setTranslucency(0)
    expect(clearAttr()).toBe(false)
    expect(glassAttr()).toBe(false)
  })
})

describe('frost and area', () => {
  beforeEach(() => {
    setTranslucencyMode('clear')
    setTranslucency(TRANSLUCENCY_MIN)
    setTranslucencyMaterial(DEFAULT_GLASS_MATERIAL)
    setTranslucencyScope(DEFAULT_GLASS_SCOPE)
  })

  it('carries the chosen material without disturbing the rest of the state', () => {
    setTranslucency(40)
    setTranslucencyMaterial('header')

    expect($translucency.get().material).toBe('header')
    expect($translucency.get().intensity).toBe(40)
  })

  it('rejects a material outside the shipped ladder', () => {
    setTranslucencyMaterial('sidebar' as never)
    expect($translucency.get().material).toBe(DEFAULT_GLASS_MATERIAL)
  })

  // The sidebar scope is the Finder shape, and styles.css keys its split-paint
  // gradient off this attribute — so the attribute IS the contract.
  it('publishes the area as an attribute only while glass is on', () => {
    setTranslucency(50)
    setTranslucencyScope('sidebar')

    if (GLASS_SUPPORTED) {
      setTranslucencyMode('glass')
      expect(document.documentElement.getAttribute('data-hermes-glass-scope')).toBe('sidebar')

      setTranslucencyScope('window')
      expect(document.documentElement.getAttribute('data-hermes-glass-scope')).toBe('window')

      setTranslucencyMode('clear')
    }

    expect(document.documentElement.hasAttribute('data-hermes-glass-scope')).toBe(false)
  })
})

// A held slider drag and a timed pulse from a picker click can overlap, which
// is why the peek counts rather than toggling: the drag must not be cancelled
// by a pulse expiring underneath it.
describe('translucency peek', () => {
  it('stays open until every overlapping hold has ended', () => {
    beginTranslucencyPeek()
    beginTranslucencyPeek()
    expect(document.documentElement.hasAttribute('data-hermes-translucency-peek')).toBe(true)

    endTranslucencyPeek()
    expect(document.documentElement.hasAttribute('data-hermes-translucency-peek')).toBe(true)

    endTranslucencyPeek()
    expect(document.documentElement.hasAttribute('data-hermes-translucency-peek')).toBe(false)
  })

  it('never goes negative, so a stray release cannot wedge the next peek open', () => {
    endTranslucencyPeek()
    endTranslucencyPeek()
    expect($translucencyPeek.get()).toBe(0)

    beginTranslucencyPeek()
    expect(document.documentElement.hasAttribute('data-hermes-translucency-peek')).toBe(true)
    endTranslucencyPeek()
    expect(document.documentElement.hasAttribute('data-hermes-translucency-peek')).toBe(false)
  })
})

// Glass rewrites page surfaces, which is only correct in a real chat window.
// The HUD, pet overlay, quick entry and wake indicator are transparent
// special-purpose windows that own their own backgrounds.
describe('isChatWindow', () => {
  it('accepts the primary window and secondary session windows', () => {
    expect(isChatWindow('')).toBe(true)
    expect(isChatWindow('?theme=dark')).toBe(true)
    expect(isChatWindow('?win=secondary')).toBe(true)
    expect(isChatWindow('?win=secondary&session=abc')).toBe(true)
  })

  it('rejects every special-purpose window kind', () => {
    for (const win of ['hud', 'pet', 'quick-entry', 'wake', 'anything-new']) {
      expect(isChatWindow(`?win=${win}`), win).toBe(false)
    }
  })
})
