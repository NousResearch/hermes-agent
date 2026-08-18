import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const { refreshActiveProfile } = vi.hoisted(() => ({
  refreshActiveProfile: vi.fn().mockResolvedValue(undefined)
}))

vi.mock('@/store/profile', () => ({ refreshActiveProfile }))

import { useProfileRailRefreshOnActive } from './use-profile-rail-refresh-on-active'

describe('useProfileRailRefreshOnActive', () => {
  afterEach(() => {
    refreshActiveProfile.mockClear()
    vi.restoreAllMocks()
  })

  it('refreshes once on mount', () => {
    renderHook(() => useProfileRailRefreshOnActive())

    expect(refreshActiveProfile).toHaveBeenCalledTimes(1)
  })

  it('refreshes again when the window regains focus', async () => {
    renderHook(() => useProfileRailRefreshOnActive())
    refreshActiveProfile.mockClear()

    await act(async () => {
      window.dispatchEvent(new Event('focus'))
    })

    expect(refreshActiveProfile).toHaveBeenCalledTimes(1)
  })

  it('refreshes when visibilitychange fires while the document is visible', async () => {
    vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
    renderHook(() => useProfileRailRefreshOnActive())
    refreshActiveProfile.mockClear()

    await act(async () => {
      document.dispatchEvent(new Event('visibilitychange'))
    })

    expect(refreshActiveProfile).toHaveBeenCalledTimes(1)
  })

  it('does NOT refresh when visibilitychange fires while the document is hidden', async () => {
    // Backgrounding/tab-switching away must not trigger a redundant refresh
    // -- only becoming visible/focused again should.
    vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden')
    renderHook(() => useProfileRailRefreshOnActive())
    refreshActiveProfile.mockClear()

    await act(async () => {
      document.dispatchEvent(new Event('visibilitychange'))
    })

    expect(refreshActiveProfile).not.toHaveBeenCalled()
  })

  it('removes both listeners on unmount, so a later focus/visibilitychange event does not refresh', async () => {
    vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
    const { unmount } = renderHook(() => useProfileRailRefreshOnActive())
    refreshActiveProfile.mockClear()

    unmount()

    await act(async () => {
      window.dispatchEvent(new Event('focus'))
      document.dispatchEvent(new Event('visibilitychange'))
    })

    expect(refreshActiveProfile).not.toHaveBeenCalled()
  })

  it('does not accumulate listeners across repeated mount/unmount cycles', async () => {
    // A leaked listener from a prior mount would double- (or N-times-)
    // refresh on a single focus event after remounting.
    const first = renderHook(() => useProfileRailRefreshOnActive())
    first.unmount()

    renderHook(() => useProfileRailRefreshOnActive())
    refreshActiveProfile.mockClear()

    await act(async () => {
      window.dispatchEvent(new Event('focus'))
    })

    expect(refreshActiveProfile).toHaveBeenCalledTimes(1)
  })
})
