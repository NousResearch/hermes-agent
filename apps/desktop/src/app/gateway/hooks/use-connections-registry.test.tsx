import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useConnectionsRegistry } from './use-connections-registry'

vi.mock('@/store/connections', () => ({
  initializeConnectionsRegistry: vi.fn(async () => null),
  refreshConnectionsRegistry: vi.fn(async () => null)
}))
vi.mock('@/store/boot', () => ({ $desktopBoot: atom({ running: true }) }))
vi.mock('@/store/windows', () => ({
  isAuxiliaryWindow: vi.fn(() => false),
  isPeerInstanceWindow: vi.fn(() => false)
}))

const { $desktopBoot } = await import('@/store/boot')
const connections = await import('@/store/connections')
const windows = await import('@/store/windows')
const refresh = vi.mocked(connections.refreshConnectionsRegistry)
const initialize = vi.mocked(connections.initializeConnectionsRegistry)

beforeEach(() => {
  vi.clearAllMocks()
  refresh.mockResolvedValue(null)
  $desktopBoot.set({ ...$desktopBoot.get(), running: true })
  vi.mocked(windows.isAuxiliaryWindow).mockReturnValue(false)
  vi.mocked(windows.isPeerInstanceWindow).mockReturnValue(false)
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('window-owned connection registry', () => {
  it('waits for primary boot fetches before restoring the launch source', async () => {
    renderHook(useConnectionsRegistry)
    expect(refresh).toHaveBeenCalledTimes(1)
    expect(initialize).not.toHaveBeenCalled()

    act(() => $desktopBoot.set({ ...$desktopBoot.get(), running: false }))
    await waitFor(() => expect(initialize).toHaveBeenCalledTimes(1))
  })

  it.each(['isPeerInstanceWindow', 'isAuxiliaryWindow'] as const)(
    'loads the cache without replaying app-launch preferences when %s',
    async kind => {
      vi.mocked(windows[kind]).mockReturnValue(true)
      $desktopBoot.set({ ...$desktopBoot.get(), running: false })
      renderHook(useConnectionsRegistry)

      await waitFor(() => expect(refresh).toHaveBeenCalledTimes(1))
      expect(initialize).not.toHaveBeenCalled()
    }
  )

  it('bounds failed reads and recovers on focus without polling a healthy registry', async () => {
    vi.useFakeTimers()
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    refresh.mockRejectedValue(new Error('Registry IPC unavailable'))
    const view = renderHook(useConnectionsRegistry)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000)
    })
    expect(refresh).toHaveBeenCalledTimes(3)
    expect(warn).toHaveBeenCalled()

    refresh.mockResolvedValue(null)
    await act(async () => {
      window.dispatchEvent(new Event('focus'))
    })
    expect(refresh).toHaveBeenCalledTimes(4)

    await act(async () => {
      window.dispatchEvent(new Event('focus'))
      await vi.advanceTimersByTimeAsync(60_000)
    })
    expect(refresh).toHaveBeenCalledTimes(4)

    view.unmount()
    await act(async () => {
      window.dispatchEvent(new Event('focus'))
      await vi.advanceTimersByTimeAsync(60_000)
    })
    expect(refresh).toHaveBeenCalledTimes(4)
  })
})
