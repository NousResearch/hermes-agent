// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// jsdom reports an EMPTY navigator.platform and a userAgent of "darwin", so
// GLASS_SUPPORTED resolves false and every glass assertion in this file
// silently turns into a no-op — on a Mac too. Pin a real macOS platform
// BEFORE the store module is evaluated (hoisted above the imports), so the
// glass path is the one actually under test.
vi.hoisted(() => {
  Object.defineProperty(globalThis.navigator, 'platform', { configurable: true, value: 'MacIntel' })
})

import { DEFAULT_GLASS_MATERIAL, DEFAULT_GLASS_SCOPE } from '@hermes/shared/translucency'

import { onPersistenceEvent, type PersistenceEvent } from '@/lib/storage'

import {
  $translucency,
  $translucencyBook,
  $translucencyPeek,
  beginTranslucencyPeek,
  defaultTranslucencyValues,
  endTranslucencyPeek,
  GLASS_SUPPORTED,
  isChatWindow,
  resetTranslucencyPeek,
  setAppearance,
  setTranslucency,
  setTranslucencyFade,
  setTranslucencyMaterial,
  setTranslucencyMode,
  setTranslucencyScope,
  TRANSLUCENCY_MAX,
  TRANSLUCENCY_MIN,
  TRANSLUCENCY_STEP
} from './translucency'

const KEY = 'hermes.desktop.translucency.v2'
const LEGACY_KEY = 'hermes.desktop.translucency.v1'

// The book is per-appearance; the tests below drive one appearance at a time.
// Dark is the store's initial appearance, so it is also the reset target.
// This suite pins navigator.platform to a Mac before import, so the Mac table
// is the one in play — `GLASS_IS_WINDOWS` resolves false throughout.
const DARK = defaultTranslucencyValues('dark', false)
const LIGHT = defaultTranslucencyValues('light', false)

const glassAttr = () => document.documentElement.hasAttribute('data-hermes-glass')
const clearAttr = () => document.documentElement.hasAttribute('data-hermes-clear')
const keep = () => document.documentElement.style.getPropertyValue('--translucency-glass-keep')

// Snapshotted at import time: every describe below mutates the store, so the
// default can only be observed here.
const initialTranslucency = $translucency.get()

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

  // NB: this asserts the module's INITIAL value, so it deliberately reads the
  // atom before the beforeEach above can touch it — the previous version of
  // this test ran after the reset and so proved nothing about the default.
  it('starts on the dark appearance defaults, glass-backed on macOS', () => {
    expect(initialTranslucency).toEqual({ ...DARK, mode: GLASS_SUPPORTED ? 'glass' : 'clear' })
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

  // Persistence is debounced: the slider emits ~100 updates per drag and
  // localStorage.setItem is synchronous, so writing per tick janks the drag.
  // Nothing reads the stored value until a cold launch.
  it('persists intensity and mode together under one stable key, once the drag settles', () => {
    vi.useFakeTimers()
    const writes: PersistenceEvent[] = []

    const stop = onPersistenceEvent(event => {
      if (event.op === 'write') {
        writes.push(event)
      }
    })

    try {
      // A drag: every intermediate value paints, none of them hit storage.
      for (let intensity = 18; intensity <= 23; intensity += 1) {
        setTranslucency(intensity)
      }

      expect(writes).toHaveLength(0)

      vi.advanceTimersByTime(200)

      // One write, carrying the value the hand landed on — recorded against
      // the appearance being painted, not against the shared base.
      expect(writes).toHaveLength(1)
      expect(writes.at(-1)).toEqual({
        key: KEY,
        op: 'write',
        value: JSON.stringify({ ...$translucencyBook.get(), dark: { intensity: 23 } })
      })
    } finally {
      stop()
      vi.useRealTimers()
    }
  })

  // Flipped from an earlier version that asserted the bridge was debounced:
  // in clear mode the native window fade IS the effect, and main can only
  // move it when told — a debounced send froze the fade mid-drag. Main diffs
  // and debounces its own disk write, so per-tick sends are cheap.
  it('mirrors every tick to the desktop bridge so the clear-mode fade tracks the drag', () => {
    vi.useFakeTimers()
    const calls: unknown[] = []
    window.hermesDesktop = { setTranslucency: (payload: unknown) => calls.push(payload) } as never

    try {
      for (let intensity = 36; intensity <= 40; intensity += 1) {
        setTranslucency(intensity)
      }

      expect(calls).toHaveLength(5)
      expect(calls.at(-1)).toEqual({ ...DARK, intensity: 40, mode: 'clear' })
    } finally {
      vi.useRealTimers()
    }
  })

  // The page paint is the one thing that must NOT wait for the debounce —
  // the field has to track the hand.
  it('paints every tick even though it persists once', () => {
    vi.useFakeTimers()

    try {
      setTranslucencyMode('glass')
      setTranslucency(30)
      expect(keep()).toBe('70%')

      setTranslucency(80)
      expect(keep()).toBe('20%')
    } finally {
      vi.useRealTimers()
      setTranslucencyMode('clear')
      setTranslucency(TRANSLUCENCY_MIN)
    }
  })
})

describe('glass mode', () => {
  // Guard the guard: if this env ever stops reporting mac, every assertion in
  // this block quietly turns into a no-op instead of failing.
  it('runs its glass assertions for real in this environment', () => {
    expect(GLASS_SUPPORTED).toBe(true)
  })

  beforeEach(() => {
    setTranslucencyMode('clear')
    setTranslucency(TRANSLUCENCY_MIN)
  })

  it('rejects glass when the platform cannot back it', () => {
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
  // gradient off this attribute — so the attribute IS the contract. It must
  // also be REMOVED whenever glass goes away: a stale scope left on <html>
  // keeps the split-paint gradient selector live over a non-glass field.
  it('publishes the area as an attribute only while glass is on', () => {
    setTranslucency(50)
    setTranslucencyScope('sidebar')

    if (!GLASS_SUPPORTED) {
      expect(document.documentElement.hasAttribute('data-hermes-glass-scope')).toBe(false)

      return
    }

    setTranslucencyMode('glass')
    expect(document.documentElement.getAttribute('data-hermes-glass-scope')).toBe('sidebar')

    setTranslucencyScope('window')
    expect(document.documentElement.getAttribute('data-hermes-glass-scope')).toBe('window')

    // Each of the three ways glass can end has to clear it, independently.
    setTranslucency(0)
    expect(document.documentElement.hasAttribute('data-hermes-glass-scope')).toBe(false)

    setTranslucency(50)
    expect(document.documentElement.hasAttribute('data-hermes-glass-scope')).toBe(true)

    setTranslucencyMode('clear')
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

describe('peek reset', () => {
  // Escape mid-drag unmounts the slider before its pointerup ever fires; the
  // settings surface calls reset on unmount so the counter can't stay wedged
  // and ghost the next overlay at 8% opacity.
  it('drops every outstanding hold at once', () => {
    beginTranslucencyPeek()
    beginTranslucencyPeek()
    beginTranslucencyPeek()
    expect(document.documentElement.hasAttribute('data-hermes-translucency-peek')).toBe(true)

    resetTranslucencyPeek()
    expect($translucencyPeek.get()).toBe(0)
    expect(document.documentElement.hasAttribute('data-hermes-translucency-peek')).toBe(false)

    // A pulse timer expiring after the reset must not push the counter negative
    // or resurrect the attribute.
    endTranslucencyPeek()
    expect($translucencyPeek.get()).toBe(0)
    expect(document.documentElement.hasAttribute('data-hermes-translucency-peek')).toBe(false)
  })
})

describe('cross-window sync', () => {
  afterEach(() => {
    setTranslucencyMode('clear')
    setTranslucency(TRANSLUCENCY_MIN)
  })

  // Under glass main touches nothing native on an intensity change, so a
  // sibling window's storage event is the ONLY way this window hears about it.
  it("adopts a sibling window's persisted state", () => {
    window.localStorage.setItem(KEY, JSON.stringify({ mode: 'clear', base: {}, light: {}, dark: { intensity: 77 } }))

    window.dispatchEvent(new StorageEvent('storage', { key: KEY, newValue: 'x' }))

    expect($translucency.get().intensity).toBe(77)
  })

  it('ignores storage events for other keys', () => {
    setTranslucency(12)
    window.localStorage.setItem(KEY, JSON.stringify({ mode: 'clear', base: {}, light: {}, dark: { intensity: 99 } }))

    window.dispatchEvent(new StorageEvent('storage', { key: 'hermes.desktop.zoom.v1', newValue: 'x' }))

    expect($translucency.get().intensity).toBe(12)
  })
})

// The store must actually CONSULT isChatWindow, not merely export it: glass
// rewrites page-level surfaces, and the HUD / pet overlay / quick entry own
// their own backgrounds. Driving window.location.search proves the wiring.
describe('glass is confined to chat windows', () => {
  const setSearch = (search: string) => {
    Object.defineProperty(window, 'location', {
      configurable: true,
      value: { ...window.location, search }
    })
  }

  const originalLocation = window.location

  afterEach(() => {
    Object.defineProperty(window, 'location', { configurable: true, value: originalLocation })
    setTranslucencyMode('clear')
    setTranslucency(TRANSLUCENCY_MIN)
  })

  it('does not thin surfaces in a special-purpose window', () => {
    setSearch('?win=hud')
    setTranslucency(60)
    setTranslucencyMode('glass')

    // The mode is still the user's choice — only the page rewrite is withheld.
    expect($translucency.get().mode).toBe('glass')
    expect(document.documentElement.hasAttribute('data-hermes-glass')).toBe(false)
  })

  // The HUD paints its band from the app's field mix, so it needs the setting
  // and the tint number even though its surfaces must not be rewritten. The
  // two flags are what keep those separable: keying the band off
  // `data-hermes-glass` would silently never match.
  it('still publishes the live setting and the tint to a special-purpose window', () => {
    setSearch('?win=hud')
    setTranslucency(60)
    setTranslucencyMode('glass')

    expect(document.documentElement.hasAttribute('data-hermes-glass-on')).toBe(true)
    expect(document.documentElement.style.getPropertyValue('--translucency-glass-keep')).toBe('40%')
  })

  it('withdraws both flags when glass is switched off', () => {
    setSearch('?win=hud')
    setTranslucency(60)
    setTranslucencyMode('glass')
    setTranslucencyMode('clear')

    expect(document.documentElement.hasAttribute('data-hermes-glass-on')).toBe(false)
    expect(document.documentElement.style.getPropertyValue('--translucency-glass-keep')).toBe('')
  })

  it('thins them in the primary window', () => {
    setSearch('')
    setTranslucency(60)
    setTranslucencyMode('glass')

    expect(document.documentElement.hasAttribute('data-hermes-glass')).toBe(true)
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

// A tint that reads as a whisper over a dark palette is a milky sheet over a
// light one, so the same lever has to mean a different amount in each
// appearance. These are the contract for that split.
describe('per-appearance settings', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $translucencyBook.set({ mode: 'glass', base: {}, light: {}, dark: {} })
    setAppearance('dark')
  })

  afterEach(() => setAppearance('dark'))

  it('ships each appearance its own defaults', () => {
    expect($translucency.get()).toEqual({ ...DARK, mode: 'glass' })

    setAppearance('light')
    expect($translucency.get()).toEqual({ ...LIGHT, mode: 'glass' })
  })

  it('ships glass ON, so the feature is visible without being found first', () => {
    expect($translucency.get().mode).toBe('glass')
    expect($translucency.get().intensity).toBeGreaterThan(0)

    setAppearance('light')
    expect($translucency.get().intensity).toBeGreaterThan(0)
  })

  it('scopes an edit to the appearance it was made in', () => {
    setAppearance('light')
    setTranslucency(90)
    setTranslucencyFade(7)

    expect($translucency.get().intensity).toBe(90)
    expect($translucency.get().fade).toBe(7)

    // Dark never heard about it and keeps its own default.
    setAppearance('dark')
    expect($translucency.get().intensity).toBe(DARK.intensity)
    expect($translucency.get().fade).toBe(DARK.fade)

    // ...and light still holds the edit on the way back.
    setAppearance('light')
    expect($translucency.get().intensity).toBe(90)
  })

  it('carries an unset key over from the shared base', () => {
    $translucencyBook.set({ mode: 'glass', base: { intensity: 44, scope: 'sidebar' }, light: {}, dark: {} })

    // Neither appearance has an opinion, so both inherit base.
    expect($translucency.get().intensity).toBe(44)
    expect($translucency.get().scope).toBe('sidebar')

    setAppearance('light')
    expect($translucency.get().intensity).toBe(44)

    // An edit overrides only the key it touches; the rest keeps inheriting.
    setTranslucency(12)
    expect($translucency.get().intensity).toBe(12)
    expect($translucency.get().scope).toBe('sidebar')

    setAppearance('dark')
    expect($translucency.get().intensity).toBe(44)
  })

  // Clear vs glass is a choice about the window, not about the palette —
  // splitting it would mean flipping appearance silently changed the mode.
  it('keeps the mode global across appearances', () => {
    setTranslucencyMode('clear')
    expect($translucency.get().mode).toBe('clear')

    setAppearance('light')
    expect($translucency.get().mode).toBe('clear')
  })

  // Switching appearance changes what is painted but not what is stored, so
  // it must not schedule a write.
  it('does not persist on an appearance switch alone', () => {
    vi.useFakeTimers()
    const writes: PersistenceEvent[] = []

    const stop = onPersistenceEvent(event => {
      if (event.op === 'write') {
        writes.push(event)
      }
    })

    try {
      setAppearance('light')
      setAppearance('dark')
      vi.advanceTimersByTime(400)

      expect(writes).toHaveLength(0)
    } finally {
      stop()
      vi.useRealTimers()
    }
  })
})

// A window someone already tuned must survive the upgrade unchanged — the new
// per-appearance defaults may only fill in where nothing was ever set.
describe('v1 → v2 migration', () => {
  afterEach(() => {
    window.localStorage.clear()
    setAppearance('dark')
  })

  it('lands a tuned v1 state in base, so both appearances inherit it', async () => {
    window.localStorage.clear()
    window.localStorage.setItem(
      LEGACY_KEY,
      JSON.stringify({ intensity: 31, fade: 4, mode: 'glass', material: 'popover', scope: 'sidebar' })
    )

    vi.resetModules()
    const fresh = await import('./translucency')

    expect(fresh.$translucencyBook.get()).toEqual({
      mode: 'glass',
      base: { intensity: 31, fade: 4, material: 'popover', scope: 'sidebar' },
      light: {},
      dark: {}
    })

    fresh.setAppearance('light')
    expect(fresh.$translucency.get()).toEqual({
      intensity: 31,
      fade: 4,
      mode: 'glass',
      material: 'popover',
      scope: 'sidebar'
    })
  })

  // The legacy rule: a non-zero v1 intensity with no mode was rendering as
  // clear all along and has to keep doing so.
  it('keeps a legacy mode-less state on clear', async () => {
    window.localStorage.clear()
    window.localStorage.setItem(LEGACY_KEY, JSON.stringify({ intensity: 25 }))

    vi.resetModules()
    const fresh = await import('./translucency')

    expect(fresh.$translucency.get().mode).toBe('clear')
    expect(fresh.$translucency.get().intensity).toBe(25)
  })

  it('prefers a v2 book over a stale v1 state', async () => {
    window.localStorage.clear()
    window.localStorage.setItem(LEGACY_KEY, JSON.stringify({ intensity: 25, mode: 'clear' }))
    window.localStorage.setItem(KEY, JSON.stringify({ mode: 'glass', base: {}, light: {}, dark: { intensity: 8 } }))

    vi.resetModules()
    const fresh = await import('./translucency')

    expect(fresh.$translucency.get().mode).toBe('glass')
    expect(fresh.$translucency.get().intensity).toBe(8)
  })
})
