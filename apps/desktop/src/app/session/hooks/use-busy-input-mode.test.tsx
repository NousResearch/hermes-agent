import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $busyInputConfig } from '@/store/busy-input-mode'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $gatewayState } from '@/store/session'

import { useBusyInputMode } from './use-busy-input-mode'

vi.mock('@/store/session-request-router', () => ({
  requestForSessionProfile: (
    _owner: unknown,
    request: (method: string, params: unknown) => Promise<unknown>,
    method: string,
    params: unknown
  ) => request(method, params)
}))

afterEach(() => {
  cleanup()
  $busyInputConfig.set(null)
  $activeGatewayProfile.set('default')
  $gatewayState.set('closed')
})

it('canonical busy policy is session scoped even with a cached global mode', async () => {
  $connection.set({ mode: 'local', wsUrl: 'ws://localhost/api/ws?native_dial=fixture' } as never)
  $gatewayState.set('open')
  $busyInputConfig.set({ owner: JSON.stringify(['', 'default']), connection: $connection.get(), mode: 'queue' })
  let resolveNext!: (result: { value: string }) => void
  const request = vi.fn().mockResolvedValueOnce({ value: 'steer' }).mockImplementationOnce(() => new Promise(resolve => { resolveNext = resolve }))
  const h = renderHook(({ sessionId }) => useBusyInputMode({ sessionId, storedSessionId: null, requestGateway: request }), { initialProps: { sessionId: 'first' } })

  try {
    await waitFor(() => expect(h.result.current).toBe('steer'))
    h.rerender({ sessionId: 'second' })
    expect(h.result.current).toBeNull()
    await act(async () => resolveNext({ value: 'interrupt' }))
    expect(h.result.current).toBe('interrupt')
    expect(request).toHaveBeenLastCalledWith('config.get', { key: 'busy', session_id: 'second' })
  } finally { h.unmount(); $connection.set(null) }
})

it('recovers a transient busy policy read without reconnecting', async () => {
  vi.useFakeTimers()
  $gatewayState.set('open')
  const request = vi.fn().mockRejectedValueOnce(new Error('temporarily unavailable')).mockResolvedValue({ value: 'steer' })
  const hook = renderHook(() => useBusyInputMode({ sessionId: 'runtime', storedSessionId: null, requestGateway: request }))

  try {
    await act(async () => { await vi.advanceTimersByTimeAsync(2000) })
    expect(hook.result.current).toBe('steer')
    expect(request).toHaveBeenCalledTimes(2)
  } finally { hook.unmount(); vi.useRealTimers() }
})

it('bounds failed reads and cancels retries when their session leaves', async () => {
  vi.useFakeTimers()
  $gatewayState.set('open')
  const request = vi.fn().mockRejectedValue(new Error('unavailable'))
  const hook = renderHook(({ sessionId }) => useBusyInputMode({ sessionId, storedSessionId: null, requestGateway: request }), { initialProps: { sessionId: 'first' } })

  try {
    await act(async () => { await vi.advanceTimersByTimeAsync(10000) })
    expect(request).toHaveBeenCalledTimes(3)
    hook.rerender({ sessionId: 'second' })
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    hook.unmount()
    await act(async () => { await vi.advanceTimersByTimeAsync(10000) })
    expect(request).toHaveBeenCalledTimes(4)
  } finally { hook.unmount(); vi.useRealTimers() }
})

it('uses only the current owner config and ignores a late response from the previous profile', async () => {
  $gatewayState.set('open')
  let resolveOld!: (value: { value: string }) => void

  const request = vi
    .fn()
    .mockImplementationOnce(
      () =>
        new Promise(resolve => {
          resolveOld = resolve
        })
    )
    .mockResolvedValue({ value: 'steer' })

  const hook = renderHook(() =>
    useBusyInputMode({ sessionId: 'runtime', storedSessionId: null, requestGateway: request })
  )

  expect(hook.result.current).toBeNull()
  act(() => $activeGatewayProfile.set('other'))
  await waitFor(() => expect(hook.result.current).toBe('steer'))
  await act(async () => resolveOld({ value: 'queue' }))
  expect(hook.result.current).toBe('steer')
  act(() =>
    $busyInputConfig.set({
      owner: JSON.stringify([$connection.get()?.connectionId ?? '', 'other']),
      connection: $connection.get(),
      mode: 'queue'
    })
  )
  expect(hook.result.current).toBe('queue')
  act(() => $busyInputConfig.set({ owner: 'foreign-owner', connection: $connection.get(), mode: 'interrupt' }))
  expect(hook.result.current).toBe('steer')
})
