import { beforeEach, describe, expect, it } from 'vitest'

import { modePref, skinPref } from './context'
import { DEFAULT_SKIN_NAME } from './presets'

// Skin and mode share one per-profile contract, so assert it once over both.
interface Pref {
  resolve: (profile: string) => string
  assign: (profile: string, value: string) => void
}

const cases = [
  {
    name: 'skin',
    pref: skinPref as unknown as Pref,
    fallback: DEFAULT_SKIN_NAME,
    a: 'ember',
    b: 'catppuccin'
  },
  { name: 'mode', pref: modePref as unknown as Pref, fallback: 'system', a: 'dark', b: 'light' }
]

describe.each(cases)('per-profile $name', ({ pref, fallback, a, b }) => {
  beforeEach(() => window.localStorage.clear())

  it('falls back to the default when unassigned', () => {
    expect(pref.resolve('default')).toBe(fallback)
    expect(pref.resolve('work')).toBe(fallback)
  })

  it('keeps each profile on its own value', () => {
    pref.assign('work', a)
    pref.assign('default', b)
    expect(pref.resolve('work')).toBe(a)
    expect(pref.resolve('default')).toBe(b)
  })

  it('lets unassigned profiles inherit the default profile as the global fallback', () => {
    pref.assign('default', a)
    expect(pref.resolve('never-themed')).toBe(a)
  })
})

describe('mode validation', () => {
  beforeEach(() => window.localStorage.clear())

  it('normalizes an unknown stored mode back to system', () => {
    modePref.assign('work', 'dusk' as never)
    expect(modePref.resolve('work')).toBe('system')
  })
})

describe('custom backend skin persistence', () => {
  beforeEach(() => window.localStorage.clear())

  it('preserves a stored skin name that may be registered by the backend after startup', () => {
    skinPref.assign('work', 'custom-backend-skin')
    expect(skinPref.resolve('work')).toBe('custom-backend-skin')
  })
})

// A fresh profile follows the OS. This defaulted to `light`, so a dark-mode
// desktop got a white window on first launch — and, once translucency became
// per-appearance, light's much heavier tint along with it. Main already
// defaulted its own themeSource to 'system', so the two disagreed at boot.
describe('a profile that has never chosen a mode', () => {
  beforeEach(() => window.localStorage.clear())

  it('follows the OS rather than forcing light', () => {
    expect(modePref.resolve('default')).toBe('system')
    expect(modePref.resolve('work')).toBe('system')
  })

  it('still honours an explicit choice', () => {
    modePref.assign('default', 'light')
    expect(modePref.resolve('default')).toBe('light')
  })
})
