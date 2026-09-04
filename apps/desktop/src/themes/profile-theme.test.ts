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
    b: 'catppuccin',
    junk: 'nope'
  },
  { name: 'mode', pref: modePref as unknown as Pref, fallback: 'system', a: 'dark', b: 'light', junk: 'dusk' }
]

describe.each(cases)('per-profile $name', ({ pref, fallback, a, b, junk }) => {
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

  it('normalizes an unknown stored value back to the default', () => {
    pref.assign('work', junk)
    expect(pref.resolve('work')).toBe(fallback)
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

// #101216: named-profile assign must seed the global fallback so a Bot Mode
// gateway hop onto a never-themed bot does not resolve `system`.
describe('named-profile appearance seeds the global fallback (#101216)', () => {
  beforeEach(() => window.localStorage.clear())

  it('lets a never-themed bot profile inherit Dark set on the launch profile', () => {
    modePref.assign('forge', 'dark')
    expect(modePref.resolve('atlas')).toBe('dark')
  })

  it('keeps an explicit per-profile override over the global fallback', () => {
    modePref.assign('forge', 'dark')
    modePref.assign('atlas', 'light')
    expect(modePref.resolve('forge')).toBe('dark')
    expect(modePref.resolve('atlas')).toBe('light')
  })

  it('promotes a unanimous pre-existing per-profile mode into the empty global slot', () => {
    window.localStorage.setItem('hermes-desktop-profile-modes-v1', JSON.stringify({ forge: 'dark' }))
    expect(modePref.resolve('atlas')).toBe('dark')
  })
})
