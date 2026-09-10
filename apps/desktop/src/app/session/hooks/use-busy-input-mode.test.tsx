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
