import { act, renderHook, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $busyInputConfig, busyInputOwnerKey } from '@/store/busy-input-mode'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { useBusyInputMode } from './use-busy-input-mode'

vi.mock('@/store/profile', () => ({ $activeGatewayProfile: atom('builder') }))
vi.mock('@/store/session', () => ({
  $connection: atom({ connectionId: 'local' }),
  $gatewayState: atom('open'),
  knownSessionOwner: () => undefined,
  ownerLookupSessionRows: () => []
}))
vi.mock('@/store/session-states', () => ({ sessionTileOwnerRoute: () => undefined }))

describe('busy mode lifecycle', () => {
  beforeEach(() => {
    $busyInputConfig.set(null)
    $connection.set({ ...$connection.get()!, connectionId: 'local' })
    $activeGatewayProfile.set('builder')
    $gatewayState.set('open')
  })

  it('reuses existing config refreshes without reading on Send', async () => {
    const requestGateway = vi.fn()
    $busyInputConfig.set({ connection: $connection.get(), owner: busyInputOwnerKey('local', 'builder'), mode: 'steer' })

    const { result } = renderHook(() =>
      useBusyInputMode({ sessionId: 'sid', storedSessionId: 'stored', requestGateway })
    )

    expect(result.current('sid')).toBe('steer')
    expect(result.current('sid')).toBe('steer')
    expect(requestGateway).not.toHaveBeenCalled()
    act(() =>
      $busyInputConfig.set({
        connection: $connection.get(),
        owner: busyInputOwnerKey('local', 'builder'),
        mode: 'queue'
      })
    )
    expect(result.current('sid')).toBe('queue')
  })

  it('loads at initialization without another lookup on Send', async () => {
    const requestGateway = vi.fn().mockResolvedValue({ value: 'steer' })

    const { result } = renderHook(() =>
      useBusyInputMode({ sessionId: 'sid', storedSessionId: 'stored', requestGateway })
    )

    await waitFor(() => expect(result.current('sid')).toBe('steer'))
    expect(requestGateway).toHaveBeenCalledTimes(1)
    expect(result.current('another-session')).toBe(null)
  })

  it('does not expose a previous profile mode or publish its late reply', async () => {
    let resolveOld!: (value: unknown) => void

    const requestGateway = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise(resolve => {
            resolveOld = resolve
          })
      )
      .mockResolvedValue({ value: 'queue' })

    const { result } = renderHook(() =>
      useBusyInputMode({ sessionId: 'sid', storedSessionId: 'stored', requestGateway })
    )

    expect(result.current('sid')).toBe(null)
    act(() => $activeGatewayProfile.set('other'))
    await waitFor(() => expect(result.current('sid')).toBe('queue'))
    await act(async () => resolveOld({ value: 'steer' }))
    expect(result.current('sid')).toBe('queue')
  })

  it('does not reuse a config snapshot after replacing the connection', async () => {
    $busyInputConfig.set({ connection: $connection.get(), owner: busyInputOwnerKey('local', 'builder'), mode: 'steer' })
    const requestGateway = vi.fn().mockRejectedValue(new Error('unavailable'))

    const { result } = renderHook(() =>
      useBusyInputMode({ sessionId: 'sid', storedSessionId: 'stored', requestGateway })
    )

    expect(result.current('sid')).toBe('steer')
    act(() => $connection.set({ ...$connection.get()!, connectionId: 'other' }))
    await act(async () => {
      await Promise.resolve()
    })
    expect(result.current('sid')).toBe(null)
  })

  it('keeps a failed initial load unknown rather than defaulting to interrupt', async () => {
    const requestGateway = vi.fn().mockRejectedValue(new Error('offline'))

    const { result } = renderHook(() =>
      useBusyInputMode({ sessionId: 'sid', storedSessionId: 'stored', requestGateway })
    )

    await act(async () => {
      await Promise.resolve()
    })
    expect(result.current('sid')).toBe(null)
  })
})
