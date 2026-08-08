import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { __resetBackendSkinSync, ingestBackendSkin } from './backend-sync'
import { ThemeProvider } from './context'
import { $backendSkinCache, $userThemes } from './user-themes'

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
    $backendSkinCache.set({})
    $userThemes.set({})
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

  it('adopts the backend skin on first connect when no desktop choice exists', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    // gateway.ready seed (apply:false) with nothing persisted: the provider
    // adopts the backend's active skin, paints it, and caches name + converted
    // record so the choice survives relaunch (backend skins are unresolvable
    // at boot without the cache).
    act(() => ingestBackendSkin(bloomberg('#ff9f0a'), { apply: false }))

    expect(cssVar('--theme-foreground')).toBe('#ff9f0a')
    expect(window.localStorage.getItem('hermes-desktop-theme-v2')).toBe('bloomberg')
    expect(window.localStorage.getItem('hermes-desktop-backend-skin-cache-v1')).toContain('bloomberg')
  })

  it('does not adopt when the user has an explicit resolvable desktop choice', () => {
    window.localStorage.setItem('hermes-desktop-theme-v2', 'nous')
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    act(() => ingestBackendSkin(bloomberg('#ff9f0a'), { apply: false }))

    // The persisted pick wins; the seed registers but never paints.
    expect(cssVar('--theme-foreground')).not.toBe('#ff9f0a')
    expect(window.localStorage.getItem('hermes-desktop-theme-v2')).toBe('nous')
  })

  it('re-adopts when the persisted name only resolves via the just-seeded backend skin', () => {
    // Old-binary state: a bare backend-skin name persisted with no cached
    // record. Boot normalized it to the default (backend skins aren't
    // resolvable pre-connect), and once the gateway seeds it, the name only
    // resolves through the LIVE registry — boot-time sources still say "no
    // real choice". Adoption must fire anyway and cache the record so the
    // next relaunch boots straight into the skin.
    window.localStorage.setItem('hermes-desktop-theme-v2', 'bloomberg')
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    act(() => ingestBackendSkin(bloomberg('#ff9f0a'), { apply: false }))

    expect(cssVar('--theme-foreground')).toBe('#ff9f0a')
    const stored = JSON.parse(window.localStorage.getItem('hermes-desktop-backend-skin-cache-v1') ?? '{}')
    expect(stored.bloomberg).toBeDefined()
    expect(window.localStorage.getItem('hermes-desktop-theme-v2')).toBe('bloomberg')
  })

  it('persists the converted record when a backend skin is applied', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    act(() => ingestBackendSkin(bloomberg('#ff9f0a'), { apply: true }))

    // The name alone can't resolve at boot (no gateway yet); the converted
    // record must land in the backend-skin cache so boot paints it directly.
    const stored = JSON.parse(window.localStorage.getItem('hermes-desktop-backend-skin-cache-v1') ?? '{}')
    expect(stored.bloomberg).toBeDefined()
    expect(stored.bloomberg.colors.foreground).toBe('#ff9f0a')
    expect(window.localStorage.getItem('hermes-desktop-theme-v2')).toBe('bloomberg')
  })
})
