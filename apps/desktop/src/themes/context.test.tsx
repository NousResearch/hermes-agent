import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { __resetBackendSkinSync, ingestBackendSkin } from './backend-sync'
import { ThemeProvider, useTheme } from './context'
import { DEFAULT_SKIN_NAME } from './presets'

// The live-authoring loop: Hermes writes/edits one skin file and every surface
// repaints. An in-place edit keeps the NAME — only the palette moves.
const bloomberg = (foreground: string) => ({
  name: 'bloomberg',
  colors: { background: '#000000', ui_text: foreground, ui_accent: '#ff8000' }
})

const classicDefault = {
  name: 'default',
  description: 'Classic Hermes — gold and kawaii',
  colors: { background: '#1a1a2e', ui_accent: '#FFBF00', ui_text: '#FFF8DC', banner_text: '#FFF8DC' }
}

const cssVar = (name: string) => window.document.documentElement.style.getPropertyValue(name)

/** Probe so tests can call setTheme through the real provider path (not only skinPref). */
function ThemeProbe({ onReady }: { onReady: (api: ReturnType<typeof useTheme>) => void }) {
  const api = useTheme()
  onReady(api)
  return null
}

describe('ThemeProvider ← backend skin sync', () => {
  beforeEach(() => {
    window.localStorage.clear()
    __resetBackendSkinSync()
  })

  afterEach(cleanup)

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

  it('keeps setTheme(default) active and paints Classic Hermes (#76579 / #76743)', () => {
    // Review salvage: selecting registered `default` must go through setTheme /
    // normalizeSkin (not only resolveTheme/skinPref) and must NOT collapse to nous.
    let latest: ReturnType<typeof useTheme> | null = null

    render(
      <ThemeProvider>
        <ThemeProbe onReady={api => {
          latest = api
        }} />
      </ThemeProvider>
    )

    act(() => ingestBackendSkin(classicDefault, { apply: false }))

    expect(latest).not.toBeNull()
    // Boot default remains nous until the user (or Appearance) selects classic.
    expect(DEFAULT_SKIN_NAME).toBe('nous')

    act(() => {
      latest!.setTheme('default')
    })

    // Active selection stays `default` (not retired → nous). deriveTheme may
    // mode-suffix the seed (default-light / Classic Hermes Light); the
    // selection key and painted palette are what Appearance must preserve.
    expect(latest!.themeName).toBe('default')
    expect(latest!.themeName).not.toBe(DEFAULT_SKIN_NAME)
    expect(latest!.theme.label.toLowerCase()).toContain('classic hermes')
    expect(latest!.availableThemes.some(t => t.name === 'default' && t.label === 'Classic Hermes')).toBe(
      true
    )
    // Classic gold palette reaches CSS (ui_accent / ui_text → primary / foreground).
    // Light-mode derivation may mix the navy background; accent/text stay gold.
    expect(cssVar('--theme-foreground').toLowerCase()).toBe('#fff8dc')
    expect(cssVar('--theme-primary').toLowerCase()).toBe('#ffbf00')
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
})
