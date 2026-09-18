import { act, cleanup, renderHook } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

vi.mock('@/store/gateway', () => ({ $gateway: atom<unknown>(null) }))
vi.mock('@/store/profile', () => ({ $activeGatewayProfile: atom('default') }))
vi.mock('@/store/inbox', async original => ({
  ...(await original<Record<string, unknown>>()),
  refreshInbox: vi.fn().mockResolvedValue({ published: false, snapshot: null })
}))

import { $gateway } from '@/store/gateway'
import { $inbox, clearInbox, refreshInbox } from '@/store/inbox'
import { $activeGatewayProfile } from '@/store/profile'

import { useInbox } from './use-inbox'

beforeEach(() => {
  vi.useFakeTimers()
  clearInbox()
  $gateway.set(null)
  $activeGatewayProfile.set('default')
  vi.clearAllMocks()
})
afterEach(() => { cleanup(); vi.useRealTimers() })

function seed() {
  $inbox.set({ capability: 'supported', error: null, loading: false, snapshot: {
    badge: 'amber', counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
    coverage: { profile: 'default', connection_scope: '', approval_scope: '', clarify_scope: '', errors: [], partial: false, scanned_sessions: 1 },
    items: []
  } })
}

it('actual gateway atom disconnect clears loaded data without unmounting', () => {
  act(() => $gateway.set({ request: vi.fn() } as never))
  renderHook(() => useInbox())
  act(seed)
  expect($inbox.get().snapshot).not.toBeNull()
  act(() => $gateway.set(null))
  expect($inbox.get().snapshot).toBeNull()
  const calls = vi.mocked(refreshInbox).mock.calls.length
  act(() => vi.advanceTimersByTime(60_000))
  expect(refreshInbox).toHaveBeenCalledTimes(calls)
})

it('actual profile atom switch clears old data and requests the new profile', () => {
  act(() => $gateway.set({ request: vi.fn() } as never))
  renderHook(() => useInbox())
  act(seed)
  act(() => $activeGatewayProfile.set('other'))
  expect($inbox.get().snapshot).toBeNull()
  expect(refreshInbox).toHaveBeenLastCalledWith('other')
})
