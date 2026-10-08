import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { accentPref } from '@/themes/accent-pref'
import { ThemeProvider, useTheme } from '@/themes/context'

import { type AccentTarget, commitAccent, previewAccent } from './apply'

const primary = () => window.document.documentElement.style.getPropertyValue('--theme-primary')

let theme: ReturnType<typeof useTheme> | null = null

function Probe() {
  theme = useTheme()

  return null
}

function target(): AccentTarget {
  if (!theme) {
    throw new Error('ThemeProvider is not mounted')
  }

  const { accent, clearAccentPreview, previewAccent: preview, setAccent } = theme

  return { clearPreview: clearAccentPreview, current: accent, live: true, preview, setAccent }
}

beforeEach(() => {
  window.localStorage.clear()
  theme = null
  render(
    <ThemeProvider>
      <Probe />
    </ThemeProvider>
  )
})

afterEach(cleanup)

describe('questionnaire accent preview', () => {
  it('paints a previewed swatch without storing it', () => {
    const untinted = primary()

    const reverts: Array<(() => void) | null> = []
    act(() => void reverts.push(previewAccent(target(), 'pink')))

    expect(primary()).not.toBe(untinted)
    expect(accentPref.stored('default')).toBeNull()

    act(() => reverts[0]?.())

    expect(primary()).toBe(untinted)
  })

  it('stores the swatch only on Confirm', () => {
    act(() => void previewAccent(target(), 'pink'))
    act(() => commitAccent(target(), 'pink'))

    expect(accentPref.stored('default')).toBe('pink')
  })
})
