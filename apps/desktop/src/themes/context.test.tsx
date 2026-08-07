import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { __resetBackendSkinSync, ingestBackendSkin } from './backend-sync'
import { ThemeProvider } from './context'
import { $themePreview, buildCustomTheme, createCustomThemeDefinition } from './custom-themes'
import { nousTheme } from './presets'

// The live-authoring loop: Hermes writes/edits one skin file and every surface
// repaints. An in-place edit keeps the NAME — only the palette moves.
const bloomberg = (foreground: string) => ({
  name: 'bloomberg',
  colors: { background: '#000000', ui_text: foreground, ui_accent: '#ff8000' }
})

const cssVar = (name: string) => window.document.documentElement.style.getPropertyValue(name)

describe('ThemeProvider ← backend skin sync', () => {
  beforeEach(() => {
    window.localStorage.clear()
    __resetBackendSkinSync()
  })

  afterEach(() => {
    $themePreview.set(null)
    cleanup()
  })

  it('applies an activated backend skin', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    act(() => ingestBackendSkin(bloomberg('#ff9f0a'), { apply: true }))

    expect(cssVar('--theme-foreground')).toBe('#ff9f0a')
    expect(cssVar('--theme-background-seed')).toBe('#000000')
  })

  it('repaints an in-place edit of the ACTIVE skin (same name, new palette)', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    act(() => ingestBackendSkin(bloomberg('#ff9f0a'), { apply: true }))
    expect(cssVar('--theme-foreground')).toBe('#ff9f0a')

    // Recolor the same skin file. The same-name apply guard correctly no-ops
    // (protects manual desktop picks), so the repaint must come from the
    // registry update reaching the active theme derivation.
    act(() => ingestBackendSkin(bloomberg('#ff2d95'), { apply: true }))
    expect(cssVar('--theme-foreground')).toBe('#ff2d95')
  })

  it('does not repaint an edit to an INACTIVE skin', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    act(() => ingestBackendSkin(bloomberg('#ff9f0a'), { apply: true }))

    // A different skin registered without apply (e.g. seeded on reconnect)
    // must not touch the painted theme.
    act(() =>
      ingestBackendSkin({ name: 'forest', colors: { background: '#001100', ui_text: '#66ff66' } }, { apply: false })
    )
    expect(cssVar('--theme-foreground')).toBe('#ff9f0a')
  })

  it('previews without persisting boot paint and restores immediately on cancel', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    const originalBackground = cssVar('--theme-background-seed')
    const storedBootBackground = window.localStorage.getItem('hermes-boot-background')

    const draft = createCustomThemeDefinition({
      source: nousTheme,
      lightColors: nousTheme.colors,
      darkColors: nousTheme.darkColors ?? nousTheme.colors
    })

    draft.light.background = '#fff1dc'
    const preview = buildCustomTheme(draft)

    act(() => $themePreview.set({ mode: 'light', theme: preview }))

    expect(cssVar('--theme-background-seed')).toBe('#fff1dc')
    expect(window.localStorage.getItem('hermes-boot-background')).toBe(storedBootBackground)

    act(() => $themePreview.set(null))

    expect(cssVar('--theme-background-seed')).toBe(originalBackground)
  })
})
