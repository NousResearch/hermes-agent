import { beforeEach, describe, expect, it } from 'vitest'

import { BUILTIN_THEMES, DEFAULT_SKIN_NAME } from './presets'
import {
  $backendSkinCache,
  $marketplaceInstalls,
  $userThemes,
  cacheBackendSkin,
  installUserTheme,
  isUserTheme,
  listAllThemes,
  marketplaceIdOf,
  pruneBackendSkinCache,
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
    $backendSkinCache.set({})
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

  it('resolves a cached backend skin before the gateway connects (boot path)', async () => {
    // Simulates a relaunch: $backendThemes is empty (no gateway yet) and only
    // the persisted cache from a previous session exists. Boot-time
    // resolution (normalizeSkin / deriveTheme) must still find the skin.
    const theme = makeTheme('Backend Skin')

    cacheBackendSkin(theme)
    expect(resolveTheme(theme.name)).toEqual(theme)
    expect(window.localStorage.getItem('hermes-desktop-backend-skin-cache-v1')).toContain(theme.name)

    // A live backend conversion wins over the stale cache once connected.
    const fresher = { ...theme, colors: { ...theme.colors, foreground: '#00ff00' } }
    const { $backendThemes } = await import('./backend-sync')
    $backendThemes.set({ [theme.name]: fresher })

    expect(resolveTheme(theme.name)).toEqual(fresher)
  })

  it('lists cached backend skins in the picker before the gateway connects', () => {
    // Regression: the picker (listAllThemes) must include skins that were
    // applied in a previous session and only exist in the boot cache — a
    // relaunch must not drop them from Appearance / Cmd-K / /skin just
    // because the gateway hasn't re-broadcast them yet.
    const theme = makeTheme('Previously Applied Skin')

    cacheBackendSkin(theme)
    expect(listAllThemes().map(t => t.name)).toContain(theme.name)
  })

  it('prunes cached backend skins that no longer exist on disk', () => {
    // Regression: a skin file deleted from $HERMES_HOME/skins/ must stop
    // ghosting in the picker once the backend broadcasts the authoritative
    // available-skin list (deleted files are absent from it).
    const alive = makeTheme('Still Here')
    const deleted = makeTheme('Deleted Skin')

    cacheBackendSkin(alive)
    cacheBackendSkin(deleted)
    expect(listAllThemes().map(t => t.name)).toContain(deleted.name)

    pruneBackendSkinCache([alive.name, 'nous', 'custom-green'])

    expect(listAllThemes().map(t => t.name)).not.toContain(deleted.name)
    expect(listAllThemes().map(t => t.name)).toContain(alive.name)
    // The pruned record is gone from persistence too, so a relaunch boots
    // without the ghost.
    expect(window.localStorage.getItem('hermes-desktop-backend-skin-cache-v1')).not.toContain(deleted.name)
  })

  it('leaves the cache untouched when every cached skin still exists', () => {
    const theme = makeTheme('Only Skin')

    cacheBackendSkin(theme)
    const before = window.localStorage.getItem('hermes-desktop-backend-skin-cache-v1')

    pruneBackendSkinCache([theme.name])

    expect(window.localStorage.getItem('hermes-desktop-backend-skin-cache-v1')).toBe(before)
    expect(listAllThemes().map(t => t.name)).toContain(theme.name)
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
