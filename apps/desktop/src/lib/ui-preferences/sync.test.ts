import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { persistentAtom } from '../persisted'
import { writeKey } from '../storage'

import { startUiPreferencesSync, type UiPreferencesBridge, type UiPreferencesSnapshot } from './sync'

// A bridge over an in-memory document with the store's semantics: per-key
// merge, null deletes, `seed` only adds the keys the store lacks.
function memoryStore(initial: null | Record<string, string>) {
  let doc: null | UiPreferencesSnapshot = initial
    ? { exists: true, generation: 'g1', revision: 1, values: { ...initial } }
    : null

  const writes: { seed?: boolean; values: Record<string, null | string> }[] = []
  const listeners = new Set<(snapshot: UiPreferencesSnapshot) => void>()
  const snapshot = (): UiPreferencesSnapshot => doc ?? { exists: false, revision: 0, values: {} }

  const bridge: UiPreferencesBridge = {
    read: async () => snapshot(),
    write: async payload => {
      writes.push(payload)

      const values = { ...(doc?.values ?? {}) }

      for (const [key, value] of Object.entries(payload.values)) {
        if (payload.seed) {
          if (value !== null && !(key in values)) {
            values[key] = value
          }
        } else if (value === null) {
          delete values[key]
        } else {
          values[key] = value
        }
      }

      doc = { exists: true, generation: doc?.generation ?? 'g1', revision: (doc?.revision ?? 0) + 1, values }

      return snapshot()
    },
    onChanged: callback => {
      listeners.add(callback)

      return () => listeners.delete(callback)
    }
  }

  const external = (values: Record<string, string>) => {
    doc = { exists: true, generation: doc?.generation ?? 'g1', revision: (doc?.revision ?? 0) + 1, values }
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

  it('adopts an existing store’s values and imports only the keys it lacks', async () => {
    window.localStorage.setItem(DENSITY, 'detailed')
    window.localStorage.setItem(MODE, 'light')
    const store = memoryStore({ [MODE]: 'dark' })

    stop = startUiPreferencesSync(store.bridge)
    await vi.runAllTimersAsync()

    expect(store.writes).toEqual([{ seed: true, values: { [DENSITY]: 'detailed', [MODE]: 'light' } }])
    expect(store.snapshot().values).toEqual({ [DENSITY]: 'detailed', [MODE]: 'dark' })
    expect(window.localStorage.getItem(MODE)).toBe('dark')
    expect(window.localStorage.getItem(DENSITY)).toBe('detailed')
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

  it('keeps a change made right before a reload and writes it on the next load', async () => {
    const store = memoryStore({ [DENSITY]: 'compact' })

    stop = startUiPreferencesSync(store.bridge)
    await vi.runAllTimersAsync()
    writeKey(DENSITY, 'detailed')
    stop() // the tab reloads before the debounced write leaves

    expect(store.writes).toEqual([])

    stop = startUiPreferencesSync(store.bridge)
    await vi.runAllTimersAsync()

    expect(window.localStorage.getItem(DENSITY)).toBe('detailed')
    expect(store.snapshot().values).toEqual({ [DENSITY]: 'detailed' })
  })

  it('keeps one write in flight so a newer edit is never overwritten by an older batch', async () => {
    const store = memoryStore({ [DENSITY]: 'compact' })
    const write = store.bridge.write
    let release: () => void = () => undefined
    let held = true
    const sent: (null | string)[] = []

    store.bridge.write = payload => {
      sent.push(payload.values[DENSITY] ?? null)

      return held
        ? new Promise(resolve => {
            release = () => {
              held = false
              resolve(write(payload))
            }
          })
        : write(payload)
    }

    stop = startUiPreferencesSync(store.bridge)
    await vi.runAllTimersAsync()
    writeKey(DENSITY, 'comfortable')
    await vi.advanceTimersByTimeAsync(200)
    writeKey(DENSITY, 'detailed')
    await vi.advanceTimersByTimeAsync(2000)

    expect(sent).toEqual(['comfortable'])

    release()
    await vi.runAllTimersAsync()

    expect(sent).toEqual(['comfortable', 'detailed'])
    expect(store.snapshot().values[DENSITY]).toBe('detailed')
    expect(window.localStorage.getItem(DENSITY)).toBe('detailed')
  })

  it('does not hold later changes behind a batch the store refused', async () => {
    const store = memoryStore({ [DENSITY]: 'compact' })
    const write = store.bridge.write
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)

    store.bridge.write = payload =>
      MODE in payload.values ? Promise.reject(new Error('400: unsupported value for ' + MODE)) : write(payload)

    stop = startUiPreferencesSync(store.bridge)
    await vi.runAllTimersAsync()
    writeKey(MODE, 'x'.repeat(10))
    await vi.runAllTimersAsync()
    writeKey(DENSITY, 'detailed')
    await vi.runAllTimersAsync()

    expect(store.snapshot().values).toEqual({ [DENSITY]: 'detailed' })
    expect(window.localStorage.getItem(MODE)).toBe('x'.repeat(10))
  })
})
