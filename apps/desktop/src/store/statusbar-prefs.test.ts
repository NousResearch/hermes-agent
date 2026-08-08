import { beforeEach, describe, expect, it, vi } from 'vitest'

// In-memory storage so each test controls the persisted state that
// persistentAtom reads at module-init time. Mirrors the pattern in
// updates.test.ts.
const storage = new Map<string, string>()

vi.mock('@/lib/storage', () => ({
  readKey: (key: string) => storage.get(key) ?? null,
  writeKey: (key: string, value: null | string) => {
    if (value === null) {
      storage.delete(key)
    } else {
      storage.set(key, value)
    }
  },
  onPersistenceEvent: () => () => {}
}))

describe('$statusbarVisible', () => {
  beforeEach(() => {
    storage.clear()
    vi.resetModules()
  })

  it('defaults to true when no stored preference exists (upgrading users)', async () => {
    const { $statusbarVisible } = await import('./statusbar-prefs')
    expect($statusbarVisible.get()).toBe(true)
  })

  it('is false when a user explicitly hid the bar in a previous session', async () => {
    storage.set('hermes.desktop.statusbarVisible', 'false')
    const { $statusbarVisible } = await import('./statusbar-prefs')
    expect($statusbarVisible.get()).toBe(false)
  })

  it('is true when a user explicitly showed the bar', async () => {
    storage.set('hermes.desktop.statusbarVisible', 'true')
    const { $statusbarVisible } = await import('./statusbar-prefs')
    expect($statusbarVisible.get()).toBe(true)
  })
})

describe('toggleStatusbarVisible', () => {
  beforeEach(() => {
    storage.clear()
    vi.resetModules()
  })

  it('toggles from true to false', async () => {
    const { $statusbarVisible, toggleStatusbarVisible } = await import('./statusbar-prefs')
    expect($statusbarVisible.get()).toBe(true)
    toggleStatusbarVisible()
    expect($statusbarVisible.get()).toBe(false)
  })

  it('toggles from false to true', async () => {
    storage.set('hermes.desktop.statusbarVisible', 'false')
    const { $statusbarVisible, toggleStatusbarVisible } = await import('./statusbar-prefs')
    expect($statusbarVisible.get()).toBe(false)
    toggleStatusbarVisible()
    expect($statusbarVisible.get()).toBe(true)
  })
})
