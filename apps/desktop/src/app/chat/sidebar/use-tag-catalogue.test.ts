import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { requestGatewayForAgent } from '@/store/gateway'
import { $profileScope } from '@/store/profile'

import { useTagCatalogue } from './use-tag-catalogue'

vi.mock('@/store/profile', () => ({
  $profiles: atom([{ name: 'a' }, { name: 'b' }]),
  $profileScope: atom('*'),
  ALL_PROFILES: '*',
  normalizeProfileKey: (s: string) => s
}))
vi.mock('@/store/session', () => ({
  $connection: atom({ connectionId: 'local', profile: 'a' }),
  $sessions: atom([{ connection_id: 'remote', profile: 'b' }]),
  $cronSessions: atom([]),
  $messagingSessions: atom([])
}))
vi.mock('@/store/sidebar-archive', () => ({ $archivedSessions: atom([]) }))
vi.mock('@/store/gateway', () => ({ requestGatewayForAgent: vi.fn() }))
afterEach(cleanup)
it('unions every shown owner catalogue and refreshes on reopen without retaining stale profile results', async () => {
  vi.mocked(requestGatewayForAgent).mockImplementation(async (id, profile) => ({ tags: [`${id}:${profile}`] }))
  const { result, rerender } = renderHook(({ open }) => useTagCatalogue(open), { initialProps: { open: true } })
  await waitFor(() => expect(result.current.tags).toEqual(['local:a', 'local:b', 'remote:b']))
  expect(requestGatewayForAgent).toHaveBeenCalledWith('remote', 'b', 'session.tags.list', { profile: 'b' })
  rerender({ open: false })
  vi.mocked(requestGatewayForAgent).mockImplementation(async () => ({ tags: ['fresh'] }))
  rerender({ open: true })
  await waitFor(() => expect(result.current.tags).toEqual(['fresh']))
  let resolve!: (value: { tags: string[] }) => void
  vi.mocked(requestGatewayForAgent).mockImplementation(
    () =>
      new Promise(r => {
        resolve = r
      })
  )
  act(() => ($profileScope as ReturnType<typeof atom<string>>).set('b'))
  vi.mocked(requestGatewayForAgent).mockImplementation(async () => ({ tags: ['a-only'] }))
  act(() => ($profileScope as ReturnType<typeof atom<string>>).set('a'))
  await waitFor(() => expect(result.current.tags).toEqual(['a-only']))
  await act(async () => resolve({ tags: ['stale-b'] }))
  expect(result.current.tags).toEqual(['a-only'])
})
