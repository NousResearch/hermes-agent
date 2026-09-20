import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { parseLinkOpenMode } from '@/store/link-open-mode'

const KEY = 'hermes.desktop.link-open-mode'

const loadStore = async () => {
  vi.resetModules()

  return import('@/store/link-open-mode')
}

describe('link open mode', () => {
  beforeAll(() => {
    const store = new Map<string, string>()
    const storage: Storage = {
      get length() {
        return store.size
      },
      clear: () => store.clear(),
      getItem: key => store.get(String(key)) ?? null,
      key: index => [...store.keys()][index] ?? null,
      removeItem: key => void store.delete(String(key)),
      setItem: (key, value) => void store.set(String(key), String(value))
    }

    Object.defineProperty(window, 'localStorage', { configurable: true, value: storage })
  })

  beforeEach(() => {
    window.localStorage.clear()
  })

  it('defaults to in-app so existing installs keep the preview pane', async () => {
    expect((await loadStore()).$linkOpenMode.get()).toBe('in-app')
  })

  it('persists external and reads it back after reload', async () => {
    const first = await loadStore()

    first.setLinkOpenMode('external')
    expect(first.$linkOpenMode.get()).toBe('external')
    expect(window.localStorage.getItem(KEY)).toBe('external')
    expect((await loadStore()).$linkOpenMode.get()).toBe('external')
  })

  it('persists an explicit in-app choice', async () => {
    const first = await loadStore()

    first.setLinkOpenMode('external')
    first.setLinkOpenMode('in-app')
    expect(first.$linkOpenMode.get()).toBe('in-app')
    expect(window.localStorage.getItem(KEY)).toBe('in-app')
    expect((await loadStore()).$linkOpenMode.get()).toBe('in-app')
  })

  it('fail-opens missing and illegal values as in-app', async () => {
    expect(parseLinkOpenMode(null)).toBe('in-app')
    expect(parseLinkOpenMode(undefined)).toBe('in-app')
    expect(parseLinkOpenMode('')).toBe('in-app')
    expect(parseLinkOpenMode('browser')).toBe('in-app')
    expect(parseLinkOpenMode('native')).toBe('in-app')
    expect(parseLinkOpenMode('in-app')).toBe('in-app')
    expect(parseLinkOpenMode('external')).toBe('external')

    window.localStorage.setItem(KEY, 'browser')
    expect((await loadStore()).$linkOpenMode.get()).toBe('in-app')
  })
})
