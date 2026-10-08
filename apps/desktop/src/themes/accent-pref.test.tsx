import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $accentOverride } from './accent-override'
import { accentPref } from './accent-pref'
import { hexToOklch, hueDelta, luminance } from './color'
import { modePref, ThemeProvider, useTheme } from './context'

const ACCENTS_KEY = 'hermes-desktop-profile-accents-v1'
const WIZARD_KEY = 'hermes-onboarding-wizard-answers-v1'
const PINK = '#e0218a'
const GREEN = '#2ea043'

const primary = () => window.document.documentElement.style.getPropertyValue('--theme-primary')

const storedKeys = () => Array.from({ length: window.localStorage.length }, (_, i) => window.localStorage.key(i))

const hueOf = (hex: string) => hexToOklch(hex)?.h ?? Number.NaN

let theme: ReturnType<typeof useTheme> | null = null

function Probe() {
  theme = useTheme()

  return null
}

const mount = () =>
  render(
    <ThemeProvider>
      <Probe />
    </ThemeProvider>
  )

const themePrimary = (mode: 'light' | 'dark') => {
  window.localStorage.clear()
  modePref.assign('default', mode)
  const view = mount()
  const value = primary()
  view.unmount()
  window.localStorage.clear()

  return value
}

describe('per-profile accent preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $accentOverride.set(null)
    theme = null
  })

  afterEach(() => {
    cleanup()
    $accentOverride.set(null)
  })

  it('repaints the stored accent after the provider is re-created', () => {
    const untinted = themePrimary('light')

    modePref.assign('default', 'light')
    accentPref.assign('default', 'pink')

    mount().unmount()
    mount()

    expect(primary()).not.toBe(untinted)
    expect(Math.abs(hueDelta(hueOf(primary()), hueOf(PINK)))).toBeLessThan(25)
  })

  it('stores a pick from Settings for the live profile', () => {
    mount()

    act(() => theme!.setAccent('green'))

    expect(accentPref.stored('default')).toBe('green')
    expect(theme!.accent).toBe('green')
    expect(Math.abs(hueDelta(hueOf(primary()), hueOf(GREEN)))).toBeLessThan(25)
  })

  it('flips Mono with the light/dark mode', () => {
    modePref.assign('default', 'light')
    accentPref.assign('default', 'mono')
    mount()

    expect(luminance(primary())).toBeLessThan(0.2)

    act(() => theme!.setMode('dark'))

    expect(luminance(primary())).toBeGreaterThan(0.6)
  })

  it('leaves another profile on its theme accent and writes no global slot', () => {
    accentPref.assign('default', 'pink')

    expect(accentPref.stored('work')).toBeNull()
    expect(storedKeys()).toEqual([ACCENTS_KEY])

    accentPref.assign('work', 'green')

    expect(accentPref.stored('default')).toBe('pink')
    expect(accentPref.stored('never-themed')).toBeNull()
    expect(storedKeys()).toEqual([ACCENTS_KEY])
  })

  it('repaints when a peer window changes the accent', () => {
    mount()
    const before = primary()

    window.localStorage.setItem(ACCENTS_KEY, JSON.stringify({ default: 'pink' }))
    act(() => window.dispatchEvent(new StorageEvent('storage', { key: ACCENTS_KEY })))

    expect(primary()).not.toBe(before)
    expect(theme!.accent).toBe('pink')
  })

  it('stores Nous blue and Theme default as no pick', () => {
    accentPref.assign('default', 'pink')
    accentPref.assign('default', 'nous')
    expect(accentPref.stored('default')).toBeNull()

    accentPref.assign('default', 'custom:#0053FD')
    expect(accentPref.stored('default')).toBeNull()

    accentPref.assign('default', 'custom:#E0218A')
    expect(accentPref.stored('default')).toBe('custom:#e0218a')

    accentPref.assign('default', null)
    expect(accentPref.stored('default')).toBeNull()
  })

  it('lets the override beat the pick, and the pick beat the theme', () => {
    const untinted = themePrimary('light')

    modePref.assign('default', 'light')
    accentPref.assign('default', 'pink')
    mount()
    const picked = primary()

    act(() => $accentOverride.set(GREEN))
    expect(Math.abs(hueDelta(hueOf(primary()), hueOf(GREEN)))).toBeLessThan(25)

    act(() => $accentOverride.set(null))
    expect(primary()).toBe(picked)

    act(() => theme!.setAccent(null))
    expect(primary()).toBe(untinted)
  })
})

describe('boot pre-paint', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => window.document.documentElement.removeAttribute('style'))

  const boot = async () => {
    await import('./context')

    return primary()
  }

  it('deletes the old wizard answers key without applying its accent', async () => {
    window.localStorage.setItem('hermes-desktop-mode-v1', 'light')
    const untinted = await boot()

    vi.resetModules()
    window.localStorage.setItem(WIZARD_KEY, JSON.stringify({ accent: PINK }))

    expect(await boot()).toBe(untinted)
    expect(window.localStorage.getItem(WIZARD_KEY)).toBeNull()
  })

  it('paints the last profile accent before the provider mounts', async () => {
    window.localStorage.setItem('hermes-desktop-mode-v1', 'light')
    const untinted = await boot()

    vi.resetModules()
    window.localStorage.setItem(ACCENTS_KEY, JSON.stringify({ default: 'pink' }))

    const painted = await boot()

    expect(painted).not.toBe(untinted)
    expect(Math.abs(hueDelta(hueOf(painted), hueOf(PINK)))).toBeLessThan(25)

    vi.resetModules()
    window.localStorage.setItem('hermes-desktop-active-profile-v1', 'work')

    expect(await boot()).toBe(untinted)
  })
})
