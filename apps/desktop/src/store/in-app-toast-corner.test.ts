import { beforeEach, describe, expect, it, vi } from 'vitest'

const STORAGE_KEY = 'hermes.desktop.inAppToastCorner'

async function loadStore() {
  return import('./in-app-toast-corner')
}

describe('in-app toast corner preference', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('accepts every supported corner and rejects stale values', async () => {
    const { DEFAULT_IN_APP_TOAST_CORNER, resolveInAppToastCorner } = await loadStore()

    expect(resolveInAppToastCorner('top-left')).toBe('top-left')
    expect(resolveInAppToastCorner('top-right')).toBe('top-right')
    expect(resolveInAppToastCorner('bottom-left')).toBe('bottom-left')
    expect(resolveInAppToastCorner('bottom-right')).toBe('bottom-right')
    expect(resolveInAppToastCorner('center')).toBe(DEFAULT_IN_APP_TOAST_CORNER)
    expect(resolveInAppToastCorner(null)).toBe(DEFAULT_IN_APP_TOAST_CORNER)
  })

  it('persists a device-local corner selection', async () => {
    const { $inAppToastCorner, setInAppToastCorner } = await loadStore()

    setInAppToastCorner('top-right')

    expect($inAppToastCorner.get()).toBe('top-right')
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('top-right')
  })

  it('hydrates the saved corner in a fresh renderer module', async () => {
    window.localStorage.setItem(STORAGE_KEY, 'top-left')

    const { $inAppToastCorner } = await loadStore()

    expect($inAppToastCorner.get()).toBe('top-left')
  })

  it('falls back safely when the saved value is invalid', async () => {
    window.localStorage.setItem(STORAGE_KEY, 'center')

    const { $inAppToastCorner, DEFAULT_IN_APP_TOAST_CORNER } = await loadStore()

    expect($inAppToastCorner.get()).toBe(DEFAULT_IN_APP_TOAST_CORNER)
  })
})
