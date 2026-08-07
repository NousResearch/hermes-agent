import type { HermesSkin, SkinColors } from '@hermes/shared/skin'
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { __resetBackendSkinSync, ingestBackendSkin } from './backend-sync'
import { getBaseColors, ThemeProvider } from './context'
import { BUILTIN_THEMES, DEFAULT_SKIN_NAME, exMachinaTheme } from './presets'
import { skinToDesktopTheme } from './skin'
import { resolveTheme } from './user-themes'

const cssVar = (name: string) => window.document.documentElement.style.getPropertyValue(name)

/**
 * The canonical Ex Machina design tokens, in the skin-shaped key space
 * `skinToDesktopTheme` consumes. This is the SOURCE the bundled preset is
 * derived from — not a copy of some file elsewhere — so the parity test below
 * fails loudly if the preset is hand-edited away from what the converter
 * produces. Change a color here and re-derive the preset, never the reverse.
 */
const EX_MACHINA_SKIN_COLORS = {
  background: '#050505',
  banner_border: '#303030',
  banner_title: '#F5F5F5',
  banner_accent: '#FF2020',
  banner_dim: '#747474',
  banner_text: '#E8E8E8',
  ui_accent: '#FF2020',
  ui_label: '#F5F5F5',
  ui_text: '#E8E8E8',
  ui_primary: '#FFFFFF',
  ui_border: '#303030',
  ui_tool: '#FF2020',
  ui_thinking: '#8A8A8A',
  ui_ok: '#83C7A0',
  ui_error: '#FF2020',
  ui_warn: '#D9B866',
  prompt: '#FF2020',
  input_rule: '#5A1010',
  response_border: '#3D3D3D',
  status_bar_bg: '#0B0B0B',
  status_bar_text: '#D7D7D7',
  status_bar_strong: '#FF2020',
  status_bar_dim: '#666666',
  status_bar_good: '#83C7A0',
  status_bar_warn: '#D9B866',
  status_bar_bad: '#FF6262',
  status_bar_critical: '#FF2020',
  session_label: '#FF2020',
  session_border: '#565656',
  voice_status_bg: '#0B0B0B',
  selection_bg: '#351010',
  completion_menu_bg: '#0B0B0B',
  completion_menu_current_bg: '#351010',
  completion_menu_meta_bg: '#111111',
  completion_menu_meta_current_bg: '#461313',
  diff_added: '#112219',
  diff_removed: '#351010',
  diff_added_word: '#83C7A0',
  diff_removed_word: '#FF6262',
  syntax_string: '#E8E8E8',
  syntax_number: '#FF6262',
  syntax_keyword: '#FF2020',
  syntax_comment: '#747474',
  shell_dollar: '#FF2020'
} satisfies SkinColors

const EX_MACHINA_SKIN: HermesSkin = { name: 'ex-machina', colors: EX_MACHINA_SKIN_COLORS }

describe('Ex Machina bundled preset', () => {
  it('is a built-in resolvable by name', () => {
    expect(BUILTIN_THEMES['ex-machina']).toBe(exMachinaTheme)
    expect(resolveTheme('ex-machina')).toBe(exMachinaTheme)
  })

  it('matches what the skin converter produces from its design tokens', () => {
    const converted = skinToDesktopTheme(EX_MACHINA_SKIN)

    // Drift here means someone hand-edited a hex in the preset instead of
    // re-deriving, so the bundled paint no longer follows the rules every synced
    // skin is converted by — and since ingestBackendSkin won't shadow a built-in
    // name, that hand-edit is what wins on screen.
    expect(exMachinaTheme.colors).toEqual(converted?.colors)
  })

  it('carries the signal red accent from the skin', () => {
    expect(exMachinaTheme.colors.primary).toBe('#ff2020')
    expect(exMachinaTheme.colors.midground).toBe('#ff2020')
    expect(exMachinaTheme.colors.background).toBe('#050505')
  })

  it('is single-mode: the light toggle cannot invert the identity', () => {
    expect(exMachinaTheme.darkColors).toEqual(exMachinaTheme.colors)
    // Even asked for light, the seed palette stays black — renderedModeFor then
    // resolves `.dark` from the background luminance.
    expect(getBaseColors('ex-machina', 'light').background).toBe('#050505')
    expect(getBaseColors('ex-machina', 'dark').background).toBe('#050505')
  })
})

describe('Ex Machina as the startup default', () => {
  beforeEach(() => {
    window.localStorage.clear()
    __resetBackendSkinSync()
    window.document.documentElement.removeAttribute('style')
    window.document.documentElement.classList.remove('dark')
  })

  afterEach(cleanup)

  it('is DEFAULT_SKIN_NAME', () => {
    expect(DEFAULT_SKIN_NAME).toBe('ex-machina')
  })

  it('paints Ex Machina on a clean profile, before any backend sync', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    expect(cssVar('--theme-background-seed')).toBe('#050505')
    expect(cssVar('--theme-midground')).toBe('#ff2020')
    expect(window.document.documentElement.dataset.hermesTheme).toBe('ex-machina')
    expect(window.document.documentElement.dataset.hermesMode).toBe('dark')
    expect(window.document.documentElement.classList.contains('dark')).toBe(true)
  })

  it('records a dark boot background so the next cold start has no white flash', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    // index.html's inline pre-paint reads these before the bundle loads; its
    // no-storage fallback is hardcoded to the same value.
    expect(window.localStorage.getItem('hermes-boot-background')).toBe('#070707')
    expect(window.localStorage.getItem('hermes-boot-color-scheme')).toBe('dark')
  })

  it('keeps the desktop preset when the backend syncs the same skin', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    act(() => {
      ingestBackendSkin(EX_MACHINA_SKIN, { apply: true })
    })

    expect(cssVar('--theme-background-seed')).toBe('#050505')
    expect(cssVar('--theme-midground')).toBe('#ff2020')
  })
})

describe('Ex Machina semantic colors', () => {
  beforeEach(() => {
    window.localStorage.clear()
    __resetBackendSkinSync()
    window.document.documentElement.removeAttribute('style')
  })

  afterEach(cleanup)

  it('leaves success/error and diff tokens to styles.css', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    // These stay green/red globally. If applyTheme ever started writing them
    // from the skin, ex-machina's all-red palette would turn "success" red.
    for (const token of [
      '--ui-green',
      '--ui-red',
      '--ui-diff-add-foreground',
      '--ui-diff-remove-foreground',
      '--ui-cyan'
    ]) {
      expect(cssVar(token)).toBe('')
    }
  })

  it('routes the skin red to destructive only', () => {
    render(
      <ThemeProvider>
        <div />
      </ThemeProvider>
    )

    expect(cssVar('--dt-destructive')).toBe('#ff2020')
    expect(cssVar('--dt-destructive-foreground')).toBe('#ffffff')
  })
})
