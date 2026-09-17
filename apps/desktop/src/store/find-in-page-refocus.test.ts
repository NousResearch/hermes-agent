import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/lib/find-in-page-scope', () => ({
  captureFindScope: vi.fn(),
  currentFindScope: vi.fn(() => null),
  performScopedFind: vi.fn(() => ({ activeOrdinal: 0, count: 0 })),
  releaseFindScope: vi.fn()
}))

import { captureFindScope } from '@/lib/find-in-page-scope'
import { $findBarFocusRequest, $findInPage, closeFindBar, openFindBar } from '@/store/find-in-page'

beforeEach(() => {
  closeFindBar()
  $findInPage.set({ active: false, query: '', matchOrdinal: 0, matchCount: 0 })
  $findBarFocusRequest.set(0)
  vi.clearAllMocks()
})

describe('openFindBar', () => {
  it('opens with a cleared query and captures the scope', () => {
    openFindBar()

    const state = $findInPage.get()

    expect(state.active).toBe(true)
    expect(state.query).toBe('')
    expect(captureFindScope).toHaveBeenCalledTimes(1)
  })

  // Upstream #93273: ⌘F while the bar is open-but-unfocused did nothing,
  // because `active` never changed so the focus effect never re-fired.
  it('bumps focusRequest when already open, instead of doing nothing', () => {
    openFindBar()
    const first = $findBarFocusRequest.get()

    openFindBar()

    expect($findBarFocusRequest.get()).toBe(first + 1)
  })

  it('keeps the typed query on a repeated open', () => {
    openFindBar()
    $findInPage.set({ ...$findInPage.get(), query: 'advisor' })

    openFindBar()

    expect($findInPage.get().query).toBe('advisor')
  })

  it('keeps a live scope on a repeated open', () => {
    openFindBar()
    vi.clearAllMocks()

    openFindBar()

    // The bar stays searchable: callers that flip `active` themselves before
    // calling (and the route-stable re-open) must still have a captured scope.
    expect(captureFindScope).toHaveBeenCalledTimes(1)
  })

  it('starts a fresh session after the bar is closed', () => {
    openFindBar()
    $findInPage.set({ ...$findInPage.get(), query: 'stale' })
    closeFindBar()

    openFindBar()

    expect($findInPage.get().query).toBe('')
    expect($findInPage.get().active).toBe(true)
  })
})
