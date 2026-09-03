import { beforeEach, describe, expect, it } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'

import { $themeScope, __resetThemeScope, modePref, setThemeScope, skinPref } from './context'
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

// Theme scope: per-profile (the default above) or one look shared by every
// profile and gateway. Shared parks both prefs on the global slot so the
// per-profile record survives a round trip untouched.
const THEME_SCOPE_KEY = 'hermes-desktop-theme-scope-v1'

describe.each(cases)('theme scope for $name', ({ pref, fallback, a, b }) => {
  beforeEach(() => {
    window.localStorage.clear()
    __resetThemeScope()
    $activeGatewayProfile.set('default')
  })

  it('defaults to per-profile', () => {
    expect($themeScope.get()).toBe('per-profile')
  })

  it("promotes the live profile's look to the global slot when going shared", () => {
    pref.assign('default', b)
    pref.assign('work', a)
    $activeGatewayProfile.set('work')

    setThemeScope('shared')

    expect(pref.resolve('default')).toBe(a)
    expect(pref.resolve('work')).toBe(a)
    expect(pref.resolve('anything')).toBe(a)
  })

  it('makes an assignment from any profile visible to every profile under shared', () => {
    setThemeScope('shared')
    pref.assign('work', a)

    expect(pref.resolve('default')).toBe(a)
    expect(pref.resolve('work')).toBe(a)
    expect(pref.resolve('never-themed')).toBe(a)
  })

  it("restores each profile's own assignment when going back to per-profile", () => {
    pref.assign('default', a)
    pref.assign('work', b)
    $activeGatewayProfile.set('work')

    setThemeScope('shared')
    expect(pref.resolve('default')).toBe(b)

    setThemeScope('per-profile')
    expect(pref.resolve('work')).toBe(b)
    // The global now carries what was promoted; only the record was left alone.
    expect(pref.resolve('default')).toBe(b)
    expect(pref.resolve('never-themed')).toBe(b)
  })

  it('leaves the per-profile record untouched while shared', () => {
    pref.assign('work', a)
    $activeGatewayProfile.set('work')

    setThemeScope('shared')
    pref.assign('work', b)
    pref.assign('other', b)
    setThemeScope('per-profile')

    expect(pref.resolve('work')).toBe(a)
    expect(pref.resolve('other')).toBe(b)
  })

  it('is a no-op when set to the current scope', () => {
    pref.assign('work', a)
    $activeGatewayProfile.set('work')

    setThemeScope('per-profile')

    expect(pref.resolve('default')).toBe(fallback)
    expect(window.localStorage.getItem(THEME_SCOPE_KEY)).toBeNull()
  })
})

describe('theme scope persistence', () => {
  beforeEach(() => {
    window.localStorage.clear()
    __resetThemeScope()
    $activeGatewayProfile.set('default')
  })

  it('persists under its own scoped key', () => {
    setThemeScope('shared')
    expect(window.localStorage.getItem(THEME_SCOPE_KEY)).toBe('shared')

    __resetThemeScope()
    expect($themeScope.get()).toBe('shared')

    setThemeScope('per-profile')
    expect(window.localStorage.getItem(THEME_SCOPE_KEY)).toBe('per-profile')
  })

  it('normalizes a junk stored value to per-profile', () => {
    window.localStorage.setItem(THEME_SCOPE_KEY, 'everywhere')
    __resetThemeScope()
    expect($themeScope.get()).toBe('per-profile')
  })
})
