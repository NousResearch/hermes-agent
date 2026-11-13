import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { __resetBackendSkinSync, ingestBackendSkin } from './backend-sync'
import { ThemeProvider, useTheme } from './context'

let theme: ReturnType<typeof useTheme>

function Probe() {
  theme = useTheme()

  return null
}

beforeEach(() => {
  window.localStorage.clear()
  __resetBackendSkinSync()
})
afterEach(cleanup)

describe('Arabic skin appearance', () => {
  it('switches the dark Arabic skin to light and back, including before its sibling is seeded', () => {
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>
    )
    act(() => {
      ingestBackendSkin(
        { name: 'hermes-arabi', colors: { background: '#121212', ui_text: '#f5f5f5' } },
        { apply: true }
      )
      theme.setMode('dark')
    })
    const root = window.document.documentElement

    expect(root.dataset.hermesMode).toBe('dark')
    act(() => theme.setMode('light'))
    expect(root.dataset.hermesMode).toBe('light')
    expect(root.classList.contains('dark')).toBe(false)
    act(() =>
      ingestBackendSkin(
        { name: 'hermes-arabi-light', colors: { background: '#fcfcfc', ui_text: '#1c1c1c' } },
        { apply: false }
      )
    )
    expect(root.style.getPropertyValue('--theme-background-seed')).toBe('#fcfcfc')
    act(() => theme.setMode('dark'))
    expect(root.dataset.hermesMode).toBe('dark')
    expect(root.style.getPropertyValue('--theme-background-seed')).toBe('#121212')
  })

  it('paints an explicitly activated light skin regardless of the saved dark mode', () => {
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>
    )
    act(() => {
      theme.setMode('dark')
      ingestBackendSkin(
        { name: 'hermes-arabi-light', colors: { background: '#fcfcfc', ui_text: '#1c1c1c' } },
        { apply: true }
      )
    })
    expect(window.document.documentElement.dataset.hermesTheme).toBe('hermes-arabi-light')
    expect(window.document.documentElement.dataset.hermesMode).toBe('light')
    expect(window.document.documentElement.classList.contains('dark')).toBe(false)
  })
})
