import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const POPOUT_ENABLED_STORAGE_KEY = 'hermes.desktop.composerPopout.enabled'

function installLocalStorage() {
  const entries = new Map<string, string>()
  const storage: Storage = {
    get length() {
      return entries.size
    },
    clear: () => entries.clear(),
    getItem: key => (entries.has(key) ? entries.get(key)! : null),
    key: index => Array.from(entries.keys())[index] ?? null,
    removeItem: key => entries.delete(key),
    setItem: (key, value) => entries.set(key, String(value))
  }

  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    value: storage
  })
}

describe('composer popout store', () => {
  beforeEach(() => {
    installLocalStorage()
  })

  afterEach(() => {
    window.localStorage.removeItem(POPOUT_ENABLED_STORAGE_KEY)
    vi.resetModules()
  })

  it('ignores a stale persisted enabled flag on module import', async () => {
    window.localStorage.setItem(POPOUT_ENABLED_STORAGE_KEY, 'true')
    vi.resetModules()

    const { $composerPoppedOut } = await import('./composer-popout')

    expect($composerPoppedOut.get()).toBe(false)
    expect(window.localStorage.getItem(POPOUT_ENABLED_STORAGE_KEY)).toBe('true')
  })

  it('does not persist the enabled flag when toggled', async () => {
    window.localStorage.removeItem(POPOUT_ENABLED_STORAGE_KEY)
    vi.resetModules()

    const { $composerPoppedOut, setComposerPoppedOut } = await import('./composer-popout')

    setComposerPoppedOut(true)

    expect($composerPoppedOut.get()).toBe(true)
    expect(window.localStorage.getItem(POPOUT_ENABLED_STORAGE_KEY)).toBeNull()

    setComposerPoppedOut(false)

    expect($composerPoppedOut.get()).toBe(false)
    expect(window.localStorage.getItem(POPOUT_ENABLED_STORAGE_KEY)).toBeNull()
  })
})
