import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'
import { $daybreakByProfile, daybreakKey } from '@/store/daybreak'
import { $notifications } from '@/store/notifications'
import { $activeSessionId, $currentModel, $currentProvider } from '@/store/session'

import { useModelMenuController } from './use-model-menu-controller'

it('a level picked while a reply streams applies from the next message instead of failing (MP-24)', async () => {
  $activeSessionId.set('runtime-1')
  $currentModel.set('gemini-3.1-pro')
  $currentProvider.set('google')
  $notifications.set([])

  const requestGateway = vi.fn(async (method: string) =>
    method === 'config.set'
      ? { ok: true, operation: 'reasoning', status: 'deferred', value: 'high' }
      : { providers: [] }
  )

  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )

  const { result } = renderHook(
    () => useModelMenuController({ onSelectModel: vi.fn(), requestGateway: requestGateway as never }),
    { wrapper }
  )

  result.current.controller.setOptions(
    { effort: 'high' },
    { isActive: true, model: 'gemini-3.1-pro', provider: 'google' }
  )

  await waitFor(() =>
    expect($notifications.get()).toContainEqual(
      expect.objectContaining({ kind: 'info', message: 'high applies from your next message' })
    )
  )
  expect($notifications.get().some(n => n.kind === 'error')).toBe(false)
})

it('a queued Daybreak flip goes to the connection that was clicked, not the one focused later', async () => {
  $daybreakByProfile.set({})

  let releaseFirst!: (value: unknown) => void

  const firstReply = new Promise(resolve => {
    releaseFirst = resolve
  })

  const local = {
    request: vi.fn((method: string, params?: Record<string, unknown>) =>
      method === 'config.set' && params?.value === true ? firstReply : Promise.resolve({ value: params?.value })
    )
  }

  const remote = {
    request: vi.fn(async (_method: string, params?: Record<string, unknown>) => ({ value: params?.value }))
  }

  // The primary menu's shared requester follows focus: it reaches whichever connection is focused when called.
  const focusedRequest = vi.fn(async () => ({ providers: [] }))
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )

  const { rerender, result } = renderHook(
    ({ gateway, connection }: { connection: string; gateway: typeof local }) =>
      useModelMenuController({
        gateway: gateway as unknown as HermesGateway,
        onSelectModel: vi.fn(),
        ownerConnectionId: connection,
        profile: 'default',
        requestGateway: focusedRequest as never
      }),
    { initialProps: { connection: 'local', gateway: local }, wrapper }
  )

  result.current.controller.daybreak?.set(true)
  result.current.controller.daybreak?.set(false) // queued behind the first write
  rerender({ connection: 'remote', gateway: remote }) // focus moves to another connection's `default`

  releaseFirst({ value: true })
  await waitFor(() => expect(local.request).toHaveBeenCalledTimes(2))

  expect(local.request).toHaveBeenLastCalledWith('config.set', { key: 'daybreak', profile: 'default', value: false })
  expect(remote.request).not.toHaveBeenCalledWith('config.set', expect.anything())
  expect(focusedRequest).not.toHaveBeenCalledWith('config.set', expect.anything())
  await waitFor(() => expect($daybreakByProfile.get()[daybreakKey('default', 'local')]).toBe(false))
})
