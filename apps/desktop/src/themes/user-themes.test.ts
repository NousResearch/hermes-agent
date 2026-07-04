import { beforeEach, describe, expect, it } from 'vitest'

import { BUILTIN_THEMES, DEFAULT_SKIN_NAME } from './presets'
import {
  $marketplaceInstalls,
  $userThemes,
  installUserTheme,
  isUserTheme,
  listAllThemes,
  marketplaceIdOf,
  removeUserTheme,
  resolveTheme
} from './user-themes'
import { convertVscodeColorTheme } from './vscode'

const makeTheme = (label: string, source?: string) =>
  convertVscodeColorTheme(
    {
      name: label,
      type: 'dark',
      colors: { 'editor.background': '#101014', 'editor.foreground': '#fafafa', focusBorder: '#7aa2f7' }
    },
    source ? { source } : undefined
  ).theme

describe('user theme registry', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $userThemes.set({})
  })

  it('installs a theme into the merged registry and persists it', () => {
    const theme = installUserTheme(makeTheme('Tokyo Night'))

    expect(isUserTheme(theme.name)).toBe(true)
    expect(resolveTheme(theme.name)).toEqual(theme)
    expect(listAllThemes().map(t => t.name)).toContain(theme.name)
    expect(window.localStorage.getItem('hermes-desktop-user-themes-v1')).toContain(theme.name)
  })

  it('lists built-ins before user themes', () => {
    installUserTheme(makeTheme('Custom'))
    const names = listAllThemes().map(t => t.name)

    expect(names.slice(0, Object.keys(BUILTIN_THEMES).length)).toEqual(Object.keys(BUILTIN_THEMES))
    expect(names.at(-1)).toBe('vsc-custom')
  })

  it('removes a theme', () => {
    const theme = installUserTheme(makeTheme('Throwaway'))
    removeUserTheme(theme.name)

    expect(isUserTheme(theme.name)).toBe(false)
    expect(resolveTheme(theme.name)).toBeUndefined()
  })

  it('resolves built-ins through the same lookup', () => {
    expect(resolveTheme(DEFAULT_SKIN_NAME)).toBe(BUILTIN_THEMES[DEFAULT_SKIN_NAME])
  })

  it('refuses to shadow a built-in name', () => {
    const builtinName = makeTheme('x')
    builtinName.name = DEFAULT_SKIN_NAME

    expect(() => installUserTheme(builtinName)).toThrow(/built-in/)
  })

  it('rejects a theme missing required colors', () => {
    const broken = makeTheme('Broken')
    // @ts-expect-error — intentionally corrupt the palette for the test.
    broken.colors = { background: '#000000' }

    expect(() => installUserTheme(broken)).toThrow(/colors/)
  })
})

describe('marketplace install tracking', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $userThemes.set({})
  })

  it('recovers the extension id only from Marketplace-sourced themes', () => {
    expect(marketplaceIdOf(makeTheme('Dracula', 'dracula-theme.theme-dracula'))).toBe('dracula-theme.theme-dracula')
    // A pasted (non-Marketplace) import has no extension id to report.
    expect(marketplaceIdOf(makeTheme('Pasted'))).toBeNull()
  })

  it('maps installed Marketplace extension ids to their theme, reactively', () => {
    expect($marketplaceInstalls.get().size).toBe(0)

    const theme = installUserTheme(makeTheme('Dracula', 'dracula-theme.theme-dracula'))
    const map = $marketplaceInstalls.get()

    expect(map.get('dracula-theme.theme-dracula')).toEqual(theme)

    removeUserTheme(theme.name)
    expect($marketplaceInstalls.get().has('dracula-theme.theme-dracula')).toBe(false)
  })

  it('omits pasted imports (no extension id) from the map', () => {
    installUserTheme(makeTheme('Pasted'))
    expect($marketplaceInstalls.get().size).toBe(0)
  })
})
