import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'

import { $accentOverride, setAccentOverride } from './accent-override'
import { __resetBackendSkinSync, ingestBackendSkin } from './backend-sync'
import { skinPref, ThemeProvider, useTheme } from './context'
import { ownAccent, persistAccent } from './persist-accent'
import { nousTheme } from './presets'
import type { DesktopTheme } from './types'
import { $userThemes, installUserTheme, resolveTheme } from './user-themes'

let theme: ReturnType<typeof useTheme>

function ThemeProbe() {
  theme = useTheme()

  return null
}

beforeEach(() => {
  window.localStorage.clear()
  $userThemes.set({})
  $activeGatewayProfile.set('default')
})

afterEach(() => {
  cleanup()
  setAccentOverride(null)
})

describe('persistAccent', () => {
  it('previews, saves, reloads and resets the One Dark highlight without changing its neutral primary', () => {
    __resetBackendSkinSync()
    ingestBackendSkin({ name: 'one-dark', colors: { ui_accent: '#c678dd' } }, { apply: false })
    skinPref.assign('default', 'one-dark')
    const view = render(
      <ThemeProvider>
        <ThemeProbe />
      </ThemeProvider>
    )
    expect(ownAccent()).toBe('#e06c75')
    act(() => setAccentOverride('#2ea043'))
    const preview = theme.theme.colors
    expect(preview.primary).toBe('#404754')
    expect(preview.midground).not.toBe('#c678dd')
    act(() => persistAccent('#2ea043'))
    expect(theme.theme.colors).toEqual(preview)
    expect(theme.themeName).toBe('one-dark-accent-2ea043')
    const root = document.documentElement.style
    expect(root.getPropertyValue('--theme-midground')).toBe(preview.midground)
    expect(root.getPropertyValue('--theme-accent-soft')).toBe(preview.accent)
    expect(root.getPropertyValue('--theme-primary')).toBe('#404754')
    expect(root.getPropertyValue('--dt-input')).toBe('#181a1f')
    expect(root.getPropertyValue('--ui-success')).toBe('#98c379')
    expect(root.getPropertyValue('--ui-warning')).toBe('#d19a66')
    const stored = JSON.parse(window.localStorage.getItem('hermes-desktop-user-themes-v1')!)
    expect(stored[theme.themeName].accentOf).toBe('one-dark')
    expect(stored[theme.themeName].colors.primary).toBe('#404754')
    view.unmount()
    render(
      <ThemeProvider>
        <ThemeProbe />
      </ThemeProvider>
    )
    expect(theme.theme.colors).toEqual(preview)
    act(() => persistAccent('#e06c75'))
    expect(theme.themeName).toBe('one-dark')
    expect(theme.theme.colors.accent).toBe('#e06c75')
    act(() => persistAccent('#e0218a'))
    act(() => persistAccent(null))
    expect(theme.themeName).toBe('one-dark')
  })

  it('saves an accent as the profile theme and returns to Nous on null', () => {
    render(
      <ThemeProvider>
        <ThemeProbe />
      </ThemeProvider>
    )

    act(() => {
      setAccentOverride('#8a2be2')
      persistAccent('#8a2be2')
    })
    const saved = theme.themeName

    expect(saved).not.toBe(nousTheme.name)
    expect(skinPref.resolve('default')).toBe(saved)
    expect($accentOverride.get()).toBeNull()

    act(() => persistAccent('not a colour'))
    expect(theme.themeName).toBe(saved)

    act(() => persistAccent(null))
    expect(theme.themeName).toBe(nousTheme.name)
    expect(skinPref.resolve('default')).toBe(nousTheme.name)
  })

  it('retints the active theme, keeping its own colours and CSS, exactly as the preview painted it', () => {
    const ember: DesktopTheme = {
      ...nousTheme,
      name: 'my-campfire',
      label: 'My campfire',
      description: 'A custom skin',
      colors: { ...nousTheme.colors, background: '#fff4e8', foreground: '#2b1a0e', card: '#fffaf3', border: '#e8cfb5' },
      darkColors: { ...nousTheme.darkColors!, background: '#1c120b', foreground: '#f5e6d6', card: '#24170e' },
      customCSS: '.campfire-only { letter-spacing: 0.02em; }'
    }

    installUserTheme(ember)
    skinPref.assign('default', ember.name)

    render(
      <ThemeProvider>
        <ThemeProbe />
      </ThemeProvider>
    )

    expect(theme.themeName).toBe(ember.name)

    act(() => setAccentOverride('#e0218a'))
    const previewed = theme.theme

    act(() => persistAccent('#e0218a'))
    const saved = resolveTheme(theme.themeName)!

    expect(saved.name).not.toBe(ember.name)
    expect(saved.customCSS).toBe(ember.customCSS)
    expect(saved.colors.background).toBe(ember.colors.background)
    expect(saved.colors.foreground).toBe(ember.colors.foreground)
    expect(saved.darkColors?.background).toBe(ember.darkColors!.background)
    expect(theme.theme.colors).toEqual(previewed.colors)
    expect(theme.theme.customCSS).toBe(previewed.customCSS)

    // A different colour makes its own copy; the same base and colour reuses it.
    const pink = theme.themeName
    act(() => persistAccent('#2ea043'))
    const green = theme.themeName
    expect(green).not.toBe(pink)
    expect(resolveTheme(green)!.colors.background).toBe(ember.colors.background)
    act(() => persistAccent('#e0218a'))
    expect(theme.themeName).toBe(pink)
    expect(Object.keys($userThemes.get()).sort()).toEqual([ember.name, pink, green].sort())

    // The theme's own accent puts the theme itself back.
    act(() => persistAccent(null))
    expect(theme.themeName).toBe(ember.name)
  })

  it("never overwrites another profile's saved accent", () => {
    $activeGatewayProfile.set('home')
    render(
      <ThemeProvider>
        <ThemeProbe />
      </ThemeProvider>
    )

    act(() => persistAccent('#8a2be2'))
    const violet = skinPref.resolve('home')

    act(() => $activeGatewayProfile.set('work'))
    act(() => persistAccent('#e0218a'))

    expect(skinPref.resolve('work')).not.toBe(violet)
    expect(skinPref.resolve('home')).toBe(violet)
    expect(resolveTheme(violet)!.accentOf).toBe(nousTheme.name)
    expect(resolveTheme(skinPref.resolve('work'))!.label).toContain('#e0218a')
    expect(resolveTheme(violet)!.label).toContain('#8a2be2')
  })
})
