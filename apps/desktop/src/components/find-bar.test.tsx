import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useNavigate } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { FindBar } from '@/components/find-bar'
import { I18nProvider } from '@/i18n'
import { en } from '@/i18n/en'
import { zh } from '@/i18n/zh'
import { findBarClaimsCombo, findBarKeyAction, formatMatchLabel } from '@/lib/find-in-page'
import { KEYBIND_ACTIONS } from '@/lib/keybinds/actions'
import { actionAllowedInInput } from '@/lib/keybinds/combo'
import {
  $findInPage,
  closeFindBar,
  findInPageListenerCount,
  findNext,
  findPrevious,
  initFindInPageListener,
  openFindBar,
  resetFindInPageListenerForTest,
  setFindQuery,
  updateFindResults
} from '@/store/find-in-page'

// ── Bridge double ───────────────────────────────────────────────────────────
// Stands in for the preload `hermesDesktop` surface. `onFoundInPage` records
// its subscribers so the tests can assert the listener refcount and drive
// results back into the store the way the main process would.

interface FoundResult {
  activeMatchOrdinal: number
  count: number
}

function installBridge() {
  const findInPage = vi.fn().mockResolvedValue({ count: 0 })
  const stopFindInPage = vi.fn().mockResolvedValue(undefined)
  const subscribers = new Set<(result: FoundResult) => void>()

  const onFoundInPage = vi.fn((callback: (result: FoundResult) => void) => {
    subscribers.add(callback)

    return () => subscribers.delete(callback)
  })

  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    findInPage,
    stopFindInPage,
    onFoundInPage
  }

  return {
    findInPage,
    stopFindInPage,
    onFoundInPage,
    subscribers,
    emit(result: FoundResult) {
      for (const callback of [...subscribers]) {
        callback(result)
      }
    }
  }
}

function resetStore() {
  $findInPage.set({ active: false, query: '', matchOrdinal: 0, matchCount: 0 })
}

// Zero the bridge refcount so a leaked subscription can't bleed between tests.
function drainListeners() {
  resetFindInPageListenerForTest()
}

let bridge: ReturnType<typeof installBridge>

beforeEach(() => {
  bridge = installBridge()
  resetStore()
})

afterEach(() => {
  cleanup()
  resetStore()
  drainListeners()
  vi.restoreAllMocks()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

// ── Pure: match-count formatting ────────────────────────────────────────────

describe('formatMatchLabel', () => {
  it('hides the counter entirely when there is no query', () => {
    expect(formatMatchLabel('', 0, 0)).toBe('')
    // Even if stale counts linger from a previous search.
    expect(formatMatchLabel('', 3, 12)).toBe('')
  })

  it('reports an explicit zero when a query matches nothing', () => {
    expect(formatMatchLabel('nope', 0, 0)).toBe('0/0')
  })

  it('formats ordinal over count', () => {
    expect(formatMatchLabel('hit', 3, 12)).toBe('3/12')
    expect(formatMatchLabel('hit', 1, 1)).toBe('1/1')
  })

  it('shows 0 ordinal for the frame before the first match is selected', () => {
    // Electron legitimately reports matches with activeMatchOrdinal 0 on the
    // first (non-final) update of a fresh search.
    expect(formatMatchLabel('hit', 0, 5)).toBe('0/5')
  })

  it('clamps an out-of-range ordinal into the count', () => {
    expect(formatMatchLabel('hit', 99, 5)).toBe('5/5')
    expect(formatMatchLabel('hit', -3, 5)).toBe('0/5')
  })

  it('never emits NaN for non-finite input', () => {
    expect(formatMatchLabel('hit', Number.NaN, 4)).toBe('0/4')
    expect(formatMatchLabel('hit', 2, Number.NaN)).toBe('0/0')
    expect(formatMatchLabel('hit', 2, Number.POSITIVE_INFINITY)).toBe('0/0')
  })

  it('floors fractional counts rather than rendering decimals', () => {
    expect(formatMatchLabel('hit', 2.7, 9.9)).toBe('2/9')
  })
})

// ── Pure: keybinding matcher ────────────────────────────────────────────────

describe('findBarKeyAction', () => {
  it('maps Escape to close', () => {
    expect(findBarKeyAction({ key: 'Escape' })).toBe('close')
  })

  it('maps Cmd+G / Ctrl+G to next from anywhere', () => {
    expect(findBarKeyAction({ key: 'g', metaKey: true })).toBe('next')
    expect(findBarKeyAction({ key: 'g', ctrlKey: true })).toBe('next')
  })

  it('maps Cmd+Shift+G / Ctrl+Shift+G to previous', () => {
    // event.key is uppercase 'G' when Shift is held — matched case-insensitively.
    expect(findBarKeyAction({ key: 'G', metaKey: true, shiftKey: true })).toBe('previous')
    expect(findBarKeyAction({ key: 'g', ctrlKey: true, shiftKey: true })).toBe('previous')
  })

  it('steps with Enter / Shift+Enter only while focus is in the input', () => {
    expect(findBarKeyAction({ key: 'Enter' }, { inInput: true })).toBe('next')
    expect(findBarKeyAction({ key: 'Enter', shiftKey: true }, { inInput: true })).toBe('previous')
    // Bare Enter outside the input belongs to the composer, not the find bar.
    expect(findBarKeyAction({ key: 'Enter' })).toBeNull()
  })

  it('ignores a bare g so typing never triggers a step', () => {
    expect(findBarKeyAction({ key: 'g' })).toBeNull()
    expect(findBarKeyAction({ key: 'g' }, { inInput: true })).toBeNull()
    expect(findBarKeyAction({ key: 'G', shiftKey: true }, { inInput: true })).toBeNull()
  })

  it('falls through when Alt is held so ⌥⌘G is not swallowed', () => {
    expect(findBarKeyAction({ key: 'g', metaKey: true, altKey: true })).toBeNull()
    expect(findBarKeyAction({ key: 'Escape', altKey: true })).toBeNull()
  })

  it('does not treat Cmd+Escape as close', () => {
    expect(findBarKeyAction({ key: 'Escape', metaKey: true })).toBeNull()
  })

  it('ignores unrelated keys', () => {
    expect(findBarKeyAction({ key: 'f', metaKey: true })).toBeNull()
    expect(findBarKeyAction({ key: 'Tab' }, { inInput: true })).toBeNull()
  })
})

describe('findBarClaimsCombo', () => {
  it('claims the step accelerators and Escape', () => {
    expect(findBarClaimsCombo('mod+g')).toBe(true)
    expect(findBarClaimsCombo('mod+shift+g')).toBe(true)
    expect(findBarClaimsCombo('escape')).toBe(true)
  })

  it('leaves every other combo to the keybind registry', () => {
    // ⌘F must still reach view.findInPage even while the bar is open.
    expect(findBarClaimsCombo('mod+f')).toBe(false)
    expect(findBarClaimsCombo('mod+k')).toBe(false)
    expect(findBarClaimsCombo('mod+b')).toBe(false)
    expect(findBarClaimsCombo('enter')).toBe(false)
  })
})

// ── Keybind registration ────────────────────────────────────────────────────

describe('find-in-page keybind registration', () => {
  const byId = new Map(KEYBIND_ACTIONS.map(action => [action.id, action]))

  it('registers view.findInPage on mod+f in the view category', () => {
    const action = byId.get('view.findInPage')

    expect(action).toBeTruthy()
    expect(action?.category).toBe('view')
    expect(action?.defaults).toEqual(['mod+f'])
  })

  it('mod+f fires from inside a textarea (browser find behavior)', () => {
    // The runtime consults actionAllowedInInput before dispatching while an
    // editable element owns focus; ⌘F should still open find from the composer.
    expect(actionAllowedInInput('view.findInPage', 'mod+f')).toBe(true)
  })

  it('registers the step pair unbound so it cannot conflict with view.toggleReview', () => {
    const next = byId.get('view.findNext')
    const previous = byId.get('view.findPrevious')

    expect(next?.category).toBe('view')
    expect(previous?.category).toBe('view')
    // mod+g stays with view.toggleReview by default; the open find bar claims
    // it at dispatch time instead (findBarClaimsCombo above).
    expect(next?.defaults).toEqual([])
    expect(previous?.defaults).toEqual([])
    expect(byId.get('view.toggleReview')?.defaults).toEqual(['mod+g'])
  })

  it('every registered find action has an i18n label (keybinds panel row)', () => {
    for (const id of ['view.findInPage', 'view.findNext', 'view.findPrevious']) {
      expect(en.keybinds.actions[id], id).toBeTruthy()
      expect(zh.keybinds.actions[id], id).toBeTruthy()
    }
  })
})

// ── Store: open/close state + dispatch ──────────────────────────────────────

describe('find-in-page store', () => {
  it('opens with a cleared query and counters', () => {
    updateFindResults(3, 12)
    openFindBar()

    expect($findInPage.get()).toEqual({ active: true, query: '', matchOrdinal: 0, matchCount: 0 })
  })

  it('closing clears state and stops the native find (clears selection)', () => {
    openFindBar()
    setFindQuery('needle')
    updateFindResults(2, 7)

    closeFindBar()

    expect($findInPage.get().active).toBe(false)
    expect($findInPage.get().query).toBe('')
    expect($findInPage.get().matchCount).toBe(0)
    expect(bridge.stopFindInPage).toHaveBeenCalledTimes(1)
  })

  it('closing an already-closed bar does not re-issue stopFindInPage', () => {
    openFindBar()
    closeFindBar()
    closeFindBar()

    expect(bridge.stopFindInPage).toHaveBeenCalledTimes(1)
  })

  it('a fresh query searches from scratch (findNext false)', () => {
    openFindBar()
    setFindQuery('needle')

    expect(bridge.findInPage).toHaveBeenCalledWith('needle', { forward: true, findNext: false })
  })

  it('clearing the query stops the find instead of searching for empty', () => {
    openFindBar()
    setFindQuery('needle')
    bridge.findInPage.mockClear()

    setFindQuery('')

    expect(bridge.findInPage).not.toHaveBeenCalled()
    expect(bridge.stopFindInPage).toHaveBeenCalled()
    expect($findInPage.get().matchCount).toBe(0)
  })

  it('findNext steps forward and findPrevious steps backward on the same query', () => {
    openFindBar()
    setFindQuery('needle')
    bridge.findInPage.mockClear()

    findNext()
    expect(bridge.findInPage).toHaveBeenLastCalledWith('needle', { forward: true, findNext: true })

    findPrevious()
    expect(bridge.findInPage).toHaveBeenLastCalledWith('needle', { forward: false, findNext: true })
  })

  it('stepping with no query is a no-op (never searches invisibly)', () => {
    openFindBar()

    findNext()
    findPrevious()

    expect(bridge.findInPage).not.toHaveBeenCalled()
  })

  it('setFindQuery on a closed bar never searches', () => {
    // A debounce timer that already fired, or any late caller, must not
    // re-highlight the page after the user dismissed the bar.
    setFindQuery('needle')

    expect(bridge.findInPage).not.toHaveBeenCalled()
    expect($findInPage.get().query).toBe('')
  })

  it('found-in-page results land on the store', () => {
    openFindBar()
    setFindQuery('needle')
    const release = initFindInPageListener()

    bridge.emit({ activeMatchOrdinal: 4, count: 9 })

    expect($findInPage.get().matchOrdinal).toBe(4)
    expect($findInPage.get().matchCount).toBe(9)
    release()
  })

  it('refcounts the bridge listener so remounts cannot stack subscriptions', () => {
    const first = initFindInPageListener()
    const second = initFindInPageListener()

    // One real bridge subscription regardless of subscriber count.
    expect(bridge.onFoundInPage).toHaveBeenCalledTimes(1)
    expect(bridge.subscribers.size).toBe(1)
    expect(findInPageListenerCount()).toBe(2)

    first()
    // Still one holder → the bridge listener stays installed.
    expect(bridge.subscribers.size).toBe(1)

    second()
    // Last holder released → the listener is detached, nothing leaks.
    expect(bridge.subscribers.size).toBe(0)
    expect(findInPageListenerCount()).toBe(0)
  })

  it('releasing the same subscription twice cannot drive the refcount negative', () => {
    const release = initFindInPageListener()
    release()
    release()

    expect(findInPageListenerCount()).toBe(0)

    // A fresh subscribe after a double-release still installs exactly one.
    const next = initFindInPageListener()
    expect(bridge.subscribers.size).toBe(1)
    next()
  })
})

// ── Component ───────────────────────────────────────────────────────────────

// Store mutations that a MOUNTED FindBar subscribes to must be act()-wrapped
// so React flushes the resulting re-render inside the test.
function actStore(mutate: () => void) {
  act(() => {
    mutate()
  })
}

function renderFindBar(initialPath = '/') {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <I18nProvider configClient={null} initialLocale="en">
        <FindBar />
      </I18nProvider>
    </MemoryRouter>
  )
}

/** Harness that can navigate the route the mounted FindBar observes. */
function renderFindBarWithNavigation(initialPath = '/session/a') {
  let navigateRef: ReturnType<typeof useNavigate> | undefined

  function CaptureNavigate() {
    navigateRef = useNavigate()

    return null
  }

  const view = render(
    <MemoryRouter initialEntries={[initialPath]}>
      <I18nProvider configClient={null} initialLocale="en">
        <CaptureNavigate />
        <FindBar />
      </I18nProvider>
    </MemoryRouter>
  )

  return { ...view, navigate: (to: string) => act(() => navigateRef?.(to)) }
}

describe('FindBar', () => {
  it('renders nothing while closed', () => {
    renderFindBar()

    expect(screen.queryByRole('search')).toBeNull()
  })

  it('renders input, counter and close button when open with results', async () => {
    openFindBar()
    renderFindBar()

    const input = await screen.findByRole('textbox', { name: /find in page/i })
    expect(input).toBeTruthy()
    expect(screen.getByRole('button', { name: /close/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /next match/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /previous match/i })).toBeTruthy()

    // Counter appears once a query + results exist.
    expect(screen.queryByText('3/12')).toBeNull()
    actStore(() => $findInPage.set({ active: true, query: 'needle', matchOrdinal: 3, matchCount: 12 }))
    await waitFor(() => expect(screen.getByText('3/12')).toBeTruthy())
  })

  it('focuses the input on open', async () => {
    openFindBar()
    renderFindBar()

    const input = await screen.findByRole('textbox', { name: /find in page/i })
    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    await waitFor(() => expect(document.activeElement).toBe(input))
  })

  it('debounces typing into a single findInPage call', async () => {
    vi.useFakeTimers()

    try {
      openFindBar()
      renderFindBar()

      const input = screen.getByRole('textbox', { name: /find in page/i })
      fireEvent.change(input, { target: { value: 'n' } })
      fireEvent.change(input, { target: { value: 'ne' } })
      fireEvent.change(input, { target: { value: 'nee' } })

      expect(bridge.findInPage).not.toHaveBeenCalled()

      // Flushing the debounce updates the store, which re-renders the bar.
      act(() => {
        vi.advanceTimersByTime(200)
      })

      expect(bridge.findInPage).toHaveBeenCalledTimes(1)
      expect(bridge.findInPage).toHaveBeenCalledWith('nee', { forward: true, findNext: false })
    } finally {
      vi.useRealTimers()
    }
  })

  it('does not fire a pending search after the bar closes', async () => {
    vi.useFakeTimers()

    try {
      openFindBar()
      renderFindBar()

      fireEvent.change(screen.getByRole('textbox', { name: /find in page/i }), {
        target: { value: 'needle' }
      })

      actStore(closeFindBar)
      vi.advanceTimersByTime(500)

      expect(bridge.findInPage).not.toHaveBeenCalled()
    } finally {
      vi.useRealTimers()
    }
  })

  it('Enter dispatches next and Shift+Enter dispatches previous', async () => {
    $findInPage.set({ active: true, query: 'needle', matchOrdinal: 1, matchCount: 4 })
    renderFindBar()

    const input = screen.getByRole('textbox', { name: /find in page/i })

    fireEvent.keyDown(input, { key: 'Enter' })
    expect(bridge.findInPage).toHaveBeenLastCalledWith('needle', { forward: true, findNext: true })

    fireEvent.keyDown(input, { key: 'Enter', shiftKey: true })
    expect(bridge.findInPage).toHaveBeenLastCalledWith('needle', { forward: false, findNext: true })
  })

  it('Cmd+G / Cmd+Shift+G step from outside the input while the bar is open', async () => {
    $findInPage.set({ active: true, query: 'needle', matchOrdinal: 1, matchCount: 4 })
    renderFindBar()

    // Fired on window (not the input) — the accelerator must not require focus.
    fireEvent.keyDown(window, { key: 'g', metaKey: true })
    expect(bridge.findInPage).toHaveBeenLastCalledWith('needle', { forward: true, findNext: true })

    fireEvent.keyDown(window, { key: 'G', metaKey: true, shiftKey: true })
    expect(bridge.findInPage).toHaveBeenLastCalledWith('needle', { forward: false, findNext: true })
  })

  it('Escape closes the bar and clears the native selection', async () => {
    openFindBar()
    renderFindBar()

    fireEvent.keyDown(window, { key: 'Escape' })

    expect($findInPage.get().active).toBe(false)
    expect(bridge.stopFindInPage).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(screen.queryByRole('search')).toBeNull())
  })

  it('the close button clears state and stops the find', async () => {
    openFindBar()
    renderFindBar()

    fireEvent.click(screen.getByRole('button', { name: /close/i }))

    expect($findInPage.get().active).toBe(false)
    expect(bridge.stopFindInPage).toHaveBeenCalledTimes(1)
  })

  it('the next / previous buttons dispatch a step', () => {
    $findInPage.set({ active: true, query: 'needle', matchOrdinal: 1, matchCount: 4 })
    renderFindBar()

    fireEvent.click(screen.getByRole('button', { name: /next match/i }))
    expect(bridge.findInPage).toHaveBeenLastCalledWith('needle', { forward: true, findNext: true })

    fireEvent.click(screen.getByRole('button', { name: /previous match/i }))
    expect(bridge.findInPage).toHaveBeenLastCalledWith('needle', { forward: false, findNext: true })
  })

  it('keeps exactly one bridge subscription across an open/close cycle', async () => {
    renderFindBar()

    // Mounted-but-closed already holds the subscription: results for an
    // in-flight search must still land after the bar hides.
    expect(bridge.subscribers.size).toBe(1)

    actStore(openFindBar)
    await waitFor(() => expect(screen.getByRole('search')).toBeTruthy())
    actStore(closeFindBar)
    await waitFor(() => expect(screen.queryByRole('search')).toBeNull())
    actStore(openFindBar)

    // Toggling visibility must not stack listeners — the subscription is
    // mount-scoped, not active-scoped.
    expect(bridge.onFoundInPage).toHaveBeenCalledTimes(1)
    expect(bridge.subscribers.size).toBe(1)
  })

  it('unmount releases the bridge subscription (no leak across route changes)', () => {
    const { unmount } = renderFindBar()
    expect(bridge.subscribers.size).toBe(1)

    unmount()

    expect(bridge.subscribers.size).toBe(0)
    expect(findInPageListenerCount()).toBe(0)
  })

  it('a remount does not stack subscriptions', () => {
    const first = renderFindBar()
    first.unmount()

    const second = renderFindBar()

    expect(bridge.subscribers.size).toBe(1)
    second.unmount()
    expect(bridge.subscribers.size).toBe(0)
  })

  it('the window key listener is removed on unmount', () => {
    openFindBar()
    const { unmount } = renderFindBar()

    unmount()
    bridge.stopFindInPage.mockClear()

    // Reopen the store WITHOUT a mounted bar; a leaked listener would still
    // handle Escape and call into the bridge.
    actStore(openFindBar)
    fireEvent.keyDown(window, { key: 'Escape' })

    expect(bridge.stopFindInPage).not.toHaveBeenCalled()
  })

  it('navigating to another route closes the bar and clears the highlights', async () => {
    const { navigate } = renderFindBarWithNavigation('/session/a')

    actStore(openFindBar)
    actStore(() => $findInPage.set({ active: true, query: 'needle', matchOrdinal: 2, matchCount: 7 }))
    await waitFor(() => expect(screen.getByRole('search')).toBeTruthy())

    navigate('/session/b')

    // Bar gone, state reset, and the native selection cleared — stale
    // highlights must not survive a session switch.
    await waitFor(() => expect(screen.queryByRole('search')).toBeNull())
    expect($findInPage.get()).toEqual({ active: false, query: '', matchOrdinal: 0, matchCount: 0 })
    expect(bridge.stopFindInPage).toHaveBeenCalledTimes(1)
  })

  it('navigation while the bar is closed does not reach into the bridge', async () => {
    const { navigate } = renderFindBarWithNavigation('/session/a')

    navigate('/session/b')

    await waitFor(() => expect(screen.queryByRole('search')).toBeNull())
    expect(bridge.stopFindInPage).not.toHaveBeenCalled()
  })
})
