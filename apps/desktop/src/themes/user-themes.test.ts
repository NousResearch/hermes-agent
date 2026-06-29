import { beforeEach, describe, expect, it } from 'vitest'

import { BUILTIN_THEMES, DEFAULT_SKIN_NAME } from './presets'
import { $userThemes, installUserTheme, isUserTheme, listAllThemes, removeUserTheme, resolveTheme } from './user-themes'
import { convertVscodeColorTheme } from './vscode'

// jsdom in this environment does not provide a working localStorage.clear()
// (pre-existing baseline issue). Install a full mock so storage-backed code
// paths (user theme persistence) work in tests.
function installStorageMock() {
  let store: Record<string, string> = {}

  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    value: {
      clear: () => {
        store = {}
      },
      getItem: (key: string) => store[key] ?? null,
      key: (index: number) => Object.keys(store)[index] ?? null,
      removeItem: (key: string) => {
        delete store[key]
      },
      setItem: (key: string, value: string) => {
        store[key] = String(value)
      }
    }
  })
}

const makeTheme = (label: string) =>
  convertVscodeColorTheme({
    name: label,
    type: 'dark',
    colors: { 'editor.background': '#101014', 'editor.foreground': '#fafafa', focusBorder: '#7aa2f7' }
  }).theme

describe('user theme registry', () => {
  beforeEach(() => {
    installStorageMock()
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
