import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import { requestGatewayForAgent } from '@/store/gateway'
import { $profileScope } from '@/store/profile'

import { useTagCatalogue } from './use-tag-catalogue'

vi.mock('@/store/profile', () => ({ $profileScope: atom('a') }))
vi.mock('@/store/session', () => ({
  $connection: atom({ connectionId: 'local', profile: 'a' }),
  $sessions: atom([]),
  $cronSessions: atom([]),
  $messagingSessions: atom([])
}))
vi.mock('@/store/sidebar-archive', () => ({ $archivedSessions: atom([]) }))
vi.mock('@/store/gateway', () => ({ requestGatewayForAgent: vi.fn() }))
afterEach(cleanup)
it('offers the full server catalogue with no loaded rows and never changes the profile filter', async () => {
  vi.mocked(requestGatewayForAgent).mockResolvedValue({ tags: ['from-b'] })
  const { result, rerender } = renderHook(({ open }) => useTagCatalogue(open), { initialProps: { open: true } })
  await waitFor(() => expect(result.current.tags).toEqual(['from-b']))
  expect(requestGatewayForAgent).toHaveBeenCalledTimes(1)
  expect($profileScope.get()).toBe('a')
  act(() => ($profileScope as ReturnType<typeof atom<string>>).set('b'))
  expect(result.current.tags).toEqual(['from-b'])
  expect(requestGatewayForAgent).toHaveBeenCalledTimes(1)
  rerender({ open: false })
  vi.mocked(requestGatewayForAgent).mockResolvedValue({ tags: ['fresh'] })
  rerender({ open: true })
  await waitFor(() => expect(result.current.tags).toEqual(['fresh']))
})
