import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { expect, it, vi } from 'vitest'

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
