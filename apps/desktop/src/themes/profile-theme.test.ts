import { beforeEach, describe, expect, it } from 'vitest'

import { modePref, skinPref } from './context'
import { DEFAULT_SKIN_NAME } from './presets'

// Skin and mode share one desktop-wide contract, so assert it once over both.
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

describe.each(cases)('desktop-wide $name', ({ pref, fallback, a, b, junk }) => {
  beforeEach(() => window.localStorage.clear())

  it('falls back to the default when unassigned', () => {
    expect(pref.resolve('default')).toBe(fallback)
    expect(pref.resolve('work')).toBe(fallback)
  })

  it('uses the latest assignment across every profile', () => {
    pref.assign('work', a)
    expect(pref.resolve('default')).toBe(a)
    pref.assign('default', b)
    expect(pref.resolve('work')).toBe(b)
    expect(pref.resolve('default')).toBe(b)
  })

  it('applies a choice to profiles that have never selected an appearance', () => {
    pref.assign('default', a)
    expect(pref.resolve('never-themed')).toBe(a)
  })

  it('normalizes an unknown stored value back to the default', () => {
    pref.assign('work', junk)
    expect(pref.resolve('default')).toBe(fallback)
  })
})

// A fresh desktop follows the OS. This defaulted to `light`, so a dark-mode
// desktop got a white window on first launch — and, once translucency became
// per-appearance, light's much heavier tint along with it. Main already
// defaulted its own themeSource to 'system', so the two disagreed at boot.
describe('a desktop that has never chosen a mode', () => {
  beforeEach(() => window.localStorage.clear())

  it('follows the OS rather than forcing light', () => {
    expect(modePref.resolve('default')).toBe('system')
    expect(modePref.resolve('work')).toBe('system')
  })

  it('still honours an explicit choice', () => {
    modePref.assign('default', 'light')
    expect(modePref.resolve('default')).toBe('light')
  })

  it('migrates the last active profile appearance once', () => {
    window.localStorage.setItem('hermes-desktop-active-profile-v1', 'jarvis')
    window.localStorage.setItem('hermes-desktop-profile-themes-v1', JSON.stringify({ jarvis: 'mono', seo: 'nous' }))
    window.localStorage.setItem('hermes-desktop-profile-modes-v1', JSON.stringify({ jarvis: 'light', seo: 'dark' }))

    expect(skinPref.resolve('seo')).toBe('mono')
    expect(modePref.resolve('seo')).toBe('light')
    expect(window.localStorage.getItem('team-hermes-global-appearance-v1')).toBe('1')
  })
})
