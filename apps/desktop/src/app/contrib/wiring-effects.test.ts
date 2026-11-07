import { act, cleanup, renderHook } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, expect, it, vi } from 'vitest'

import type { FreshSessionRequestOptions } from '@/store/profile'

vi.mock('@/store/profile', () => ({
  $freshSessionRequest: atom(0),
  $freshSessionRequestOptions: atom<FreshSessionRequestOptions>({}),
  refreshActiveProfile: vi.fn()
}))
vi.mock('../right-sidebar/files/use-project-tree', () => ({ resetProjectTreeState: vi.fn() }))
vi.mock('./panes', () => ({ $restartPreviewServer: atom(null) }))

const { $freshSessionRequest, $freshSessionRequestOptions } = await import('@/store/profile')
const { useFreshSessionRequest } = await import('./wiring-effects')

afterEach(cleanup)

it('delivers coding choices through the extracted fresh-draft hook without replaying the initial request', () => {
  const start = vi.fn()
  renderHook(() => useFreshSessionRequest(start))
  expect(start).not.toHaveBeenCalled()

  const options = { codingWorkspaceControls: true, workspaceTarget: null } as const
  act(() => {
    $freshSessionRequestOptions.set(options)
    $freshSessionRequest.set($freshSessionRequest.get() + 1)
  })
  expect(start).toHaveBeenCalledExactlyOnceWith(options)

  act(() => {
    $freshSessionRequestOptions.set({})
    $freshSessionRequest.set($freshSessionRequest.get() + 1)
  })
  expect(start).toHaveBeenLastCalledWith({})
  expect(start).toHaveBeenCalledTimes(2)
})
