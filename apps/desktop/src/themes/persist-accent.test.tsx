import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'

import { $accentOverride, setAccentOverride } from './accent-override'
import { skinPref, ThemeProvider, useTheme } from './context'
import { persistAccent } from './persist-accent'
import { nousTheme } from './presets'
import { $userThemes } from './user-themes'

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
})
