import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { persistentAtom } from '../persisted'
import { writeKey } from '../storage'

import { startUiPreferencesSync, type UiPreferencesBridge, type UiPreferencesSnapshot } from './sync'

// A bridge over an in-memory document with the store's semantics: per-key
// merge, null deletes, `seed` applies only while no document exists.
function memoryStore(initial: null | Record<string, string>) {
  let doc: null | UiPreferencesSnapshot = initial ? { exists: true, revision: 1, values: { ...initial } } : null
  const writes: { seed?: boolean; values: Record<string, null | string> }[] = []
  const listeners = new Set<(snapshot: UiPreferencesSnapshot) => void>()
  const snapshot = (): UiPreferencesSnapshot => doc ?? { exists: false, revision: 0, values: {} }

  const bridge: UiPreferencesBridge = {
    read: async () => snapshot(),
    write: async payload => {
      writes.push(payload)

      if (payload.seed && doc) {
        return snapshot()
      }

      const values = { ...(doc?.values ?? {}) }

      for (const [key, value] of Object.entries(payload.values)) {
        if (value === null) {
          delete values[key]
        } else {
          values[key] = value
        }
      }

      doc = { exists: true, revision: (doc?.revision ?? 0) + 1, values }

      return snapshot()
    },
    onChanged: callback => {
      listeners.add(callback)

      return () => listeners.delete(callback)
    }
  }

  const external = (values: Record<string, string>) => {
    doc = { exists: true, revision: (doc?.revision ?? 0) + 1, values }
    listeners.forEach(listener => listener(snapshot()))
  }

  return { bridge, external, snapshot, writes }
}

const DENSITY = 'hermes.desktop.sessionListDensity'
const MODE = 'hermes-desktop-mode-v1'
const DRAFT = 'hermes.desktop.composerQueue.v1'

describe('machine UI preferences sync', () => {
  let stop: () => void = () => undefined

  beforeEach(() => {
    window.localStorage.clear()
    vi.useFakeTimers()
  })

  afterEach(() => {
    stop()
    vi.useRealTimers()
  })

  it('seeds a missing store with only this window’s shared preferences', async () => {
    window.localStorage.setItem(DENSITY, 'detailed')
    window.localStorage.setItem(DRAFT, '[{"text":"unsent"}]')
    const store = memoryStore(null)

    stop = startUiPreferencesSync(store.bridge)
    await vi.runAllTimersAsync()

    expect(store.writes[0]).toEqual({ seed: true, values: { [DENSITY]: 'detailed' } })
    expect(store.snapshot().values).toEqual({ [DENSITY]: 'detailed' })
  })

  it('adopts an existing store over this window’s cache instead of overwriting it', async () => {
    window.localStorage.setItem(DENSITY, 'detailed')
    window.localStorage.setItem(MODE, 'light')
    const $density = persistentAtom(DENSITY, 'compact')
    const store = memoryStore({ [MODE]: 'dark' })

    stop = startUiPreferencesSync(store.bridge)
    await vi.runAllTimersAsync()

    expect(store.writes).toEqual([])
    expect(store.snapshot().values).toEqual({ [MODE]: 'dark' })
    expect(window.localStorage.getItem(MODE)).toBe('dark')
    expect(window.localStorage.getItem(DENSITY)).toBeNull()
    expect($density.get()).toBe('compact')
  })

  it('merges local changes per key and follows changes made elsewhere live', async () => {
    const $density = persistentAtom(DENSITY, 'compact', { decode: raw => raw, encode: value => value })
    const store = memoryStore({ [MODE]: 'dark' })

    stop = startUiPreferencesSync(store.bridge)
    await vi.runAllTimersAsync()

    $density.set('detailed')
    writeKey(DRAFT, 'local only')
    await vi.runAllTimersAsync()

    expect(store.writes).toEqual([{ values: { [DENSITY]: 'detailed' } }])
    expect(store.snapshot().values).toEqual({ [DENSITY]: 'detailed', [MODE]: 'dark' })

    store.external({ [DENSITY]: 'comfortable', [MODE]: 'dark' })
    await vi.runAllTimersAsync()

    expect($density.get()).toBe('comfortable')
    expect(store.writes).toHaveLength(1)
  })
})
