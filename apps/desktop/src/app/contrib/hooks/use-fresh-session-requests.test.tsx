import { act, renderHook } from '@testing-library/react'
import { StrictMode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $freshSessionRequest, requestFreshSession } from '@/store/profile'

import { useFreshSessionRequests } from './use-fresh-session-requests'

describe('useFreshSessionRequests', () => {
  beforeEach(() => {
    $freshSessionRequest.set(0)
  })

  it('clears the old chat synchronously when a profile switch requests a fresh draft', () => {
    const startFreshSessionDraft = vi.fn()

    renderHook(() => useFreshSessionRequests(startFreshSessionDraft))

    act(() => {
      requestFreshSession()

      // The clear must happen in the same stack as profile selection. Deferring
      // it to a React effect leaves the old profile route live for a fast Enter.
      expect(startFreshSessionDraft).toHaveBeenCalledTimes(1)
    })
  })

  it('keeps exactly one live listener through StrictMode effect replay', () => {
    const startFreshSessionDraft = vi.fn()

    renderHook(() => useFreshSessionRequests(startFreshSessionDraft), { wrapper: StrictMode })

    act(() => requestFreshSession())

    expect(startFreshSessionDraft).toHaveBeenCalledTimes(1)
  })

  it('removes the synchronous listener on unmount', () => {
    const startFreshSessionDraft = vi.fn()
    const { unmount } = renderHook(() => useFreshSessionRequests(startFreshSessionDraft))

    unmount()
    requestFreshSession()

    expect(startFreshSessionDraft).not.toHaveBeenCalled()
  })
})
