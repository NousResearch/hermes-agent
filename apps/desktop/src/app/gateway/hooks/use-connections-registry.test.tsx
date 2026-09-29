import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useConnectionsRegistry } from './use-connections-registry'

const { activeConnectionId, getOnChanged, setOnChanged } = vi.hoisted(() => {
  let active = 'deleted' as string | null
  const activeConnectionId = { get: () => active, set: (value: string | null) => (active = value) }
  let onChanged: ((payload: { connectionId: string; reason: 'removed' | 'saved' | 'updated' }) => void) | undefined
  return { activeConnectionId, getOnChanged: () => onChanged, setOnChanged: (value: typeof onChanged) => (onChanged = value) }
})

vi.mock('@/store/connections', () => ({
  $activeConnectionId: activeConnectionId,
  forgetConnection: vi.fn(),
  initializeConnectionsRegistry: vi.fn(async () => null),
  refreshConnectionsRegistry: vi.fn(async () => null),
  selectConnection: vi.fn(async () => undefined)
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
  setOnChanged(undefined)
  activeConnectionId.set('deleted')
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

  it('re-homes the active window after the main process removes its connection', async () => {
    const surviving = { primary: 'local', connections: [{ id: 'local' }] }
    refresh.mockResolvedValue(surviving as never)
    window.hermesDesktop = {
      connections: {
        onChanged: (callback: (payload: { connectionId: string; reason: 'removed' | 'saved' | 'updated' }) => void) => {
          setOnChanged(callback)
          return () => undefined
        }
      }
    } as never

    renderHook(useConnectionsRegistry)
    await act(async () => getOnChanged()?.({ connectionId: 'deleted', reason: 'removed' }))

    expect(connections.forgetConnection).toHaveBeenCalledWith('deleted')
    await waitFor(() => expect(connections.selectConnection).toHaveBeenCalledWith('local'))
  })

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
  setOnChanged(undefined)
  activeConnectionId.set('deleted')
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
