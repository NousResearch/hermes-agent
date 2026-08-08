import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $backendThemes, __resetBackendSkinSync, ingestBackendSkin } from './backend-sync'
import { ThemeProvider } from './context'

// The live-authoring loop: Hermes writes/edits one skin file and every surface
// repaints. An in-place edit keeps the NAME — only the palette moves.
const bloomberg = (foreground: string) => ({
  name: 'bloomberg',
  colors: { background: '#000000', ui_text: foreground, ui_accent: '#ff8000' }
})

const cssVar = (name: string) => window.document.documentElement.style.getPropertyValue(name)

const customStyleEl = () =>
  window.document.getElementById('hermes-desktop-custom-css') as HTMLStyleElement | null

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

  it('injects customCSS from an applied backend skin', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    act(() =>
      ingestBackendSkin({ ...bloomberg('#ff9f0a'), customCSS: '.chat-input { font-size: 16px; }' }, { apply: true })
    )

    expect(customStyleEl()?.textContent).toBe('.chat-input { font-size: 16px; }')
  })

  it('replaces customCSS in the SAME <style> tag when the active skin changes', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    act(() => ingestBackendSkin({ ...bloomberg('#ff9f0a'), customCSS: 'a { color: red; }' }, { apply: true }))
    const first = customStyleEl()
    expect(first?.textContent).toBe('a { color: red; }')

    act(() =>
      ingestBackendSkin(
        { name: 'forest', colors: { background: '#001100', ui_text: '#66ff66' }, customCSS: 'b { color: blue; }' },
        { apply: true }
      )
    )

    const second = customStyleEl()
    expect(second).not.toBeNull()
    expect(second?.id).toBe(first?.id) // one tag reused — no accumulation
    expect(second?.textContent).toBe('b { color: blue; }')
  })

  it('removes the style tag when switching to a CSS-less skin', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    act(() => ingestBackendSkin({ ...bloomberg('#ff9f0a'), customCSS: 'a { color: red; }' }, { apply: true }))
    expect(customStyleEl()).not.toBeNull()

    act(() => ingestBackendSkin(bloomberg('#00ff00'), { apply: true }))

    expect(customStyleEl()).toBeNull()
  })

  it('applies customCSS for a built-in-named user skin without shadowing the built-in palette', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    act(() =>
      ingestBackendSkin(
        { name: 'mono', colors: { background: '#ff00ff', ui_text: '#00ff00' }, customCSS: '.status-bar { background: black; }' },
        { apply: true }
      )
    )

    // The user's CSS lands…
    expect(customStyleEl()?.textContent).toBe('.status-bar { background: black; }')
    // …but the palette policy still holds: the backend theme is never
    // registered under a built-in name, so the painted background is the
    // desktop's built-in mono, not the user YAML's #ff00ff.
    expect($backendThemes.get().mono).toBeUndefined()
    expect(cssVar('--theme-background-seed')).not.toBe('#ff00ff')
  })

  it('applies customCSS from a default-named user skin under the desktop default', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    act(() =>
      ingestBackendSkin({ name: 'default', colors: { background: '#123456' }, customCSS: '.chat-input { font-size: 18px; }' }, { apply: true })
    )

    expect(customStyleEl()?.textContent).toBe('.chat-input { font-size: 18px; }')
  })
})
