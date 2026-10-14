import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { storedString, writeKey } from '@/lib/storage'

const KEY = 'hermes.desktop.keepAwakeMode.v1'
const LEGACY_KEY = 'hermes.desktop.keepAwake.v1'
const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop
const setKeepAwakeBridge = vi.fn()

/** The atom reads storage at import time, so migration tests need a fresh module. */
async function loadStore() {
  vi.resetModules()

  return import('./keep-awake')
}

beforeEach(() => {
  desktopWindow.hermesDesktop = { setKeepAwake: setKeepAwakeBridge } as unknown as Window['hermesDesktop']
  writeKey(KEY, null)
  writeKey(LEGACY_KEY, null)
  setKeepAwakeBridge.mockClear()
})

afterEach(() => {
  desktopWindow.hermesDesktop = initialHermesDesktop
})

describe('keep-awake store', () => {
  it('persists the mode and mirrors it to the main process', async () => {
    const { $keepAwakeMode, setKeepAwakeMode } = await loadStore()

    setKeepAwakeMode('while-working')
    expect($keepAwakeMode.get()).toBe('while-working')
    expect(storedString(KEY)).toBe('while-working')
    expect(setKeepAwakeBridge).toHaveBeenLastCalledWith('while-working')

    setKeepAwakeMode('off')
    expect(storedString(KEY)).toBe('off')
    expect(setKeepAwakeBridge).toHaveBeenLastCalledWith('off')
  })

  it('defaults to off on a machine with no preference', async () => {
    const { $keepAwakeMode } = await loadStore()

    expect($keepAwakeMode.get()).toBe('off')
  })

  it('migrates the pre-mode toggle: on meant always, off meant off', async () => {
    writeKey(LEGACY_KEY, 'true')
    expect((await loadStore()).$keepAwakeMode.get()).toBe('always')
    // Loading persisted the migrated mode under the new key (the subscriber runs
    // on init) — that is the one-shot migration working. Clear it to test the
    // other legacy value from a clean slate.
    writeKey(KEY, null)

    writeKey(LEGACY_KEY, 'false')
    expect((await loadStore()).$keepAwakeMode.get()).toBe('off')
  })

  it('prefers a saved mode over a stale legacy flag beside it', async () => {
    writeKey(LEGACY_KEY, 'true')
    writeKey(KEY, 'while-working')

    expect((await loadStore()).$keepAwakeMode.get()).toBe('while-working')
  })

  it('ignores an unrecognised saved mode instead of trusting it', async () => {
    writeKey(KEY, 'sometimes')

    expect((await loadStore()).$keepAwakeMode.get()).toBe('off')
  })
})
