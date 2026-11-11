import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it } from 'vitest'

import { $shellChoice } from '@/store/shell-variant'

import { type ThemeContextValue, ThemeProvider, useTheme } from './context'

const SKIN_KEY = 'hermes-desktop-theme-v2'

let seen: ThemeContextValue | null = null

function Probe() {
  seen = useTheme()

  return null
}

const mount = () =>
  render(
    <ThemeProvider>
      <Probe />
    </ThemeProvider>
  )

beforeEach(() => {
  window.localStorage.clear()
  $shellChoice.set(null)
})

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
  seen = null
})

it('paints the redesign in its own palette until the user picks a skin, without saving one', () => {
  mount()
  expect(seen?.themeName).toBe('portal')
  expect(seen?.defaultThemeName).toBe('portal')

  act(() => $shellChoice.set(false))
  expect(seen?.themeName).toBe('nous')
  expect(seen?.defaultThemeName).toBe('nous')

  act(() => $shellChoice.set(true))
  expect(seen?.themeName).toBe('portal')
  expect(window.localStorage.getItem(SKIN_KEY)).toBeNull()
})

it('keeps an explicit pick in both shells', () => {
  mount()
  act(() => seen?.setTheme('everforest'))
  expect(seen?.themeName).toBe('everforest')

  act(() => $shellChoice.set(false))
  expect(seen?.themeName).toBe('everforest')
  expect(seen?.defaultThemeName).toBe('nous')
})
