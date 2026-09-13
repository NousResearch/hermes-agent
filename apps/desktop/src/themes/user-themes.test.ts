import { beforeEach, describe, expect, it, vi } from 'vitest'

import { onPersistenceEvent, type PersistenceEvent } from '@/lib/storage'

import { BUILTIN_THEMES, DEFAULT_SKIN_NAME } from './presets'
import { PROFILE_SKINS_KEY, SKIN_KEY, USER_THEMES_KEY } from './storage-keys'
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
    const theme = makeTheme('Tokyo Night')
    const persistenceEvents: PersistenceEvent[] = []
    const unsubscribe = onPersistenceEvent(event => persistenceEvents.push(event))

    try {
      installUserTheme(theme)
    } finally {
      unsubscribe()
    }

    expect(isUserTheme(theme.name)).toBe(true)
    expect(resolveTheme(theme.name)).toEqual(theme)
    expect(listAllThemes().map(t => t.name)).toContain(theme.name)
    const stored = JSON.stringify({ [theme.name]: theme })
    expect(window.localStorage.getItem(USER_THEMES_KEY)).toBe(stored)
    expect(persistenceEvents).toEqual([{ key: USER_THEMES_KEY, op: 'write', value: stored }])
  })

  it('lists built-ins before user themes', () => {
    installUserTheme(makeTheme('Custom'))
    const names = listAllThemes().map(t => t.name)

    expect(names.slice(0, Object.keys(BUILTIN_THEMES).length)).toEqual(Object.keys(BUILTIN_THEMES))
    expect(names.at(-1)).toBe('vsc-custom')
  })

  it('removes a theme and scrubs profile and global skin assignments', () => {
    const theme = installUserTheme(makeTheme('Throwaway'))
    window.localStorage.setItem(PROFILE_SKINS_KEY, JSON.stringify({ default: theme.name, custom: 'other-theme' }))
    window.localStorage.setItem(SKIN_KEY, theme.name)
    const persistenceEvents: PersistenceEvent[] = []
    const unsubscribe = onPersistenceEvent(event => persistenceEvents.push(event))

    try {
      removeUserTheme(theme.name)
    } finally {
      unsubscribe()
    }

    expect(isUserTheme(theme.name)).toBe(false)
    expect(resolveTheme(theme.name)).toBeUndefined()
    expect(JSON.parse(window.localStorage.getItem(PROFILE_SKINS_KEY) || '{}')).toEqual({ custom: 'other-theme' })
    expect(window.localStorage.getItem(SKIN_KEY)).toBeNull()
    expect(persistenceEvents).toEqual([
      { key: USER_THEMES_KEY, op: 'remove', value: null },
      {
        key: PROFILE_SKINS_KEY,
        op: 'read',
        value: JSON.stringify({ default: theme.name, custom: 'other-theme' })
      },
      { key: PROFILE_SKINS_KEY, op: 'write', value: JSON.stringify({ custom: 'other-theme' }) },
      { key: SKIN_KEY, op: 'read', value: theme.name },
      { key: SKIN_KEY, op: 'remove', value: null }
    ])
  })

  it.each(['__proto__', 'constructor', 'toString'])('is a true no-op for inherited prototype name %s', name => {
    const retained = makeTheme('Retained')
    const current = { [retained.name]: retained }
    const rawUserThemes = JSON.stringify(current)
    const rawProfiles = JSON.stringify({ default: name, custom: retained.name })
    $userThemes.set(current)
    window.localStorage.setItem(USER_THEMES_KEY, rawUserThemes)
    window.localStorage.setItem(PROFILE_SKINS_KEY, rawProfiles)
    window.localStorage.setItem(SKIN_KEY, name)
    const persistenceEvents: PersistenceEvent[] = []
    const unsubscribe = onPersistenceEvent(event => persistenceEvents.push(event))

    try {
      removeUserTheme(name)
    } finally {
      unsubscribe()
    }

    expect($userThemes.get()).toBe(current)
    expect($userThemes.get()).toEqual({ [retained.name]: retained })
    expect(window.localStorage.getItem(USER_THEMES_KEY)).toBe(rawUserThemes)
    expect(window.localStorage.getItem(PROFILE_SKINS_KEY)).toBe(rawProfiles)
    expect(window.localStorage.getItem(SKIN_KEY)).toBe(name)
    expect(persistenceEvents).toEqual([])
  })

  it('removes an own prototype-named theme from a null-prototype registry', () => {
    const theme = makeTheme('Prototype collision')
    theme.name = '__proto__'
    const current = Object.create(null) as Record<string, typeof theme>
    current[theme.name] = theme
    $userThemes.set(current)
    window.localStorage.setItem(USER_THEMES_KEY, JSON.stringify(current))
    const persistenceEvents: PersistenceEvent[] = []
    const unsubscribe = onPersistenceEvent(event => persistenceEvents.push(event))

    try {
      removeUserTheme(theme.name)
    } finally {
      unsubscribe()
    }

    expect(Object.hasOwn($userThemes.get(), theme.name)).toBe(false)
    expect(persistenceEvents).toEqual([
      { key: USER_THEMES_KEY, op: 'remove', value: null },
      { key: PROFILE_SKINS_KEY, op: 'read', value: null },
      { key: SKIN_KEY, op: 'read', value: null }
    ])
  })

  it('still scrubs the global skin assignment when profile storage contains corrupted JSON', () => {
    const theme = installUserTheme(makeTheme('Corrupted profiles'))
    const rawProfiles = '{not-json'
    window.localStorage.setItem(PROFILE_SKINS_KEY, rawProfiles)
    window.localStorage.setItem(SKIN_KEY, theme.name)
    const persistenceEvents: PersistenceEvent[] = []
    const unsubscribe = onPersistenceEvent(event => persistenceEvents.push(event))

    try {
      removeUserTheme(theme.name)
    } finally {
      unsubscribe()
    }

    expect(isUserTheme(theme.name)).toBe(false)
    expect(window.localStorage.getItem(PROFILE_SKINS_KEY)).toBe(rawProfiles)
    expect(window.localStorage.getItem(SKIN_KEY)).toBeNull()
    expect(persistenceEvents).toEqual([
      { key: USER_THEMES_KEY, op: 'remove', value: null },
      { key: PROFILE_SKINS_KEY, op: 'read', value: rawProfiles },
      { key: SKIN_KEY, op: 'read', value: theme.name },
      { key: SKIN_KEY, op: 'remove', value: null }
    ])
  })

  it.each<[string, (themeName: string) => unknown]>([
    ['null', () => null],
    ['a primitive', () => 'invalid'],
    ['an array', themeName => [themeName]]
  ])('still scrubs the global skin assignment when profile storage contains %s', (_, makeStored) => {
    const theme = installUserTheme(makeTheme('Invalid profiles'))
    const rawProfiles = JSON.stringify(makeStored(theme.name))
    window.localStorage.setItem(PROFILE_SKINS_KEY, rawProfiles)
    window.localStorage.setItem(SKIN_KEY, theme.name)

    removeUserTheme(theme.name)

    expect(isUserTheme(theme.name)).toBe(false)
    expect(window.localStorage.getItem(PROFILE_SKINS_KEY)).toBe(rawProfiles)
    expect(window.localStorage.getItem(SKIN_KEY)).toBeNull()
  })

  it('still removes the theme when localStorage access is restricted', () => {
    const theme = installUserTheme(makeTheme('Restricted storage'))
    const localStorageDescriptor = Object.getOwnPropertyDescriptor(window, 'localStorage')
    const persistenceEvents: PersistenceEvent[] = []
    const unsubscribe = onPersistenceEvent(event => persistenceEvents.push(event))

    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      get: () => {
        throw new DOMException('Storage access denied', 'SecurityError')
      }
    })

    try {
      expect(() => removeUserTheme(theme.name)).not.toThrow()
      expect(isUserTheme(theme.name)).toBe(false)
      expect(persistenceEvents).toEqual([
        { key: USER_THEMES_KEY, op: 'remove', value: null },
        { key: PROFILE_SKINS_KEY, op: 'read', value: null },
        { key: SKIN_KEY, op: 'read', value: null }
      ])
    } finally {
      unsubscribe()

      if (localStorageDescriptor) {
        Object.defineProperty(window, 'localStorage', localStorageDescriptor)
      }
    }
  })

  it('writes the remaining installed themes when removing one of several', () => {
    const removed = installUserTheme(makeTheme('Removed'))
    const retained = installUserTheme(makeTheme('Retained'))
    const persistenceEvents: PersistenceEvent[] = []
    const unsubscribe = onPersistenceEvent(event => persistenceEvents.push(event))

    try {
      removeUserTheme(removed.name)
    } finally {
      unsubscribe()
    }

    const stored = JSON.stringify({ [retained.name]: retained })
    expect(window.localStorage.getItem(USER_THEMES_KEY)).toBe(stored)
    expect(persistenceEvents).toEqual([
      { key: USER_THEMES_KEY, op: 'write', value: stored },
      { key: PROFILE_SKINS_KEY, op: 'read', value: null },
      { key: SKIN_KEY, op: 'read', value: null }
    ])
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

describe('user theme storage hydration', () => {
  it('reads installed themes through the persistence choke point', async () => {
    const theme = makeTheme('Hydrated')
    const stored = JSON.stringify({ [theme.name]: theme })
    window.localStorage.setItem(USER_THEMES_KEY, stored)
    vi.resetModules()
    const { onPersistenceEvent: subscribe } = await import('@/lib/storage')
    const persistenceEvents: PersistenceEvent[] = []
    const unsubscribe = subscribe(event => persistenceEvents.push(event))

    try {
      const { $userThemes: hydratedThemes } = await import('./user-themes')

      expect(hydratedThemes.get()).toEqual({ [theme.name]: theme })
    } finally {
      unsubscribe()
    }

    expect(persistenceEvents).toContainEqual({ key: USER_THEMES_KEY, op: 'read', value: stored })
  })
})
