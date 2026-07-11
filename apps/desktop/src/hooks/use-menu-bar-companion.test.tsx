// @vitest-environment jsdom
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const broadcastDesktopStateChange = vi.fn()
const getMenuBarCompanionEnabled = vi.fn()
const setMenuBarCompanionEnabled = vi.fn()
let syncHandler: ((message: Record<string, unknown>) => void) | null = null

vi.mock('@/lib/desktop-state-sync', () => ({
  broadcastDesktopStateChange: (domain: string, options?: unknown) => broadcastDesktopStateChange(domain, options),
  onDesktopStateSync: (handler: (message: Record<string, unknown>) => void) => {
    syncHandler = handler

    return () => {
      syncHandler = null
    }
  }
}))

import { useMenuBarCompanion } from './use-menu-bar-companion'

beforeEach(() => {
  getMenuBarCompanionEnabled.mockResolvedValue({ enabled: true })
  setMenuBarCompanionEnabled.mockImplementation(async enabled => ({ enabled }))
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      settings: {
        getMenuBarCompanionEnabled,
        setMenuBarCompanionEnabled
      }
    }
  })
})

afterEach(() => {
  cleanup()
  syncHandler = null
  vi.clearAllMocks()
})

describe('useMenuBarCompanion', () => {
  it('loads, saves, and synchronizes the shared Desktop preference', async () => {
    const { result } = renderHook(() => useMenuBarCompanion())

    await waitFor(() => expect(getMenuBarCompanionEnabled).toHaveBeenCalled())

    await act(async () => {
      await result.current.setEnabled(false)
    })

    expect(result.current.enabled).toBe(false)
    expect(setMenuBarCompanionEnabled).toHaveBeenCalledWith(false)
    expect(broadcastDesktopStateChange).toHaveBeenCalledWith('menu-bar-companion', { value: false })

    act(() => {
      syncHandler?.({ type: 'changed', domain: 'menu-bar-companion', value: true })
    })

    expect(result.current.enabled).toBe(true)
  })

  it('fails closed when the Desktop preference bridge is unavailable', async () => {
    getMenuBarCompanionEnabled.mockRejectedValueOnce(new Error('bridge unavailable'))

    const { result } = renderHook(() => useMenuBarCompanion())

    await waitFor(() => expect(getMenuBarCompanionEnabled).toHaveBeenCalled())
    expect(result.current.enabled).toBe(false)
  })
})
