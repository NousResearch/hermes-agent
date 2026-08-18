import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useNavigate } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type KeybindRuntimeDeps, useKeybinds } from '@/app/hooks/use-keybinds'
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

// useKeybinds only needs the theme context for resolvedMode/setMode; the real
// provider persists themes and subscribes to backend sync, which is unrelated
// to the find-in-page gate under test.
vi.mock('@/themes/context', () => ({
  useTheme: () => ({ resolvedMode: 'dark', setMode: vi.fn() })
}))

// ── Bridge double ───────────────────────────────────────────────────────────
// Stands in for the preload `hermesDesktop` surface. The primary window's
// renderer-side walker no longer calls `findInPage` / `stopFindInPage`
// (#81726); only `onFoundInPage` is still wired, for secondary session
// windows that drive the search through Electron. `findInPage` /
// `stopFindInPage` remain on the bridge for those secondary windows —
// regression: the listener-refcount tests still want the bridge surface
// defined so a multi-window secondary renderer could install it.

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

/**
 * Plant a chat surface in the DOM. The scoped walker resolves its scope from
 * the foreground `[data-chat-surface]`, so any test that exercises a real
 * search needs one — even just an empty container — to make sure the store
 * has something to walk. `releaseFindScope` is called in afterEach via
 * `cleanup` + a body reset, but tests that close the bar mid-flight also
 * rely on `document.body.innerHTML = ''` between cases (see pane-visibility
 * tests for the same pattern).
 */
function plantSurface(id = 'surface') {
  const root = document.createElement('div')

  root.setAttribute('data-chat-surface', '')
  root.id = id
  document.body.appendChild(root)

  return root
}

let bridge: ReturnType<typeof installBridge>

beforeEach(() => {
  bridge = installBridge()
  resetStore()
})

afterEach(() => {
  cleanup()
  document.body.innerHTML = ''
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
    plantSurface()
    openFindBar()

    expect($findInPage.get()).toEqual({ active: true, query: '', matchOrdinal: 0, matchCount: 0 })
  })

  it('closing clears state and tears down the scoped highlights', () => {
    const surface = plantSurface()
    surface.textContent = 'needle in a haystack, needle again'
    openFindBar()
    setFindQuery('needle')

    expect($findInPage.get().matchCount).toBe(2)
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(2)

    closeFindBar()

    expect($findInPage.get().active).toBe(false)
    expect($findInPage.get().query).toBe('')
    expect($findInPage.get().matchCount).toBe(0)
    // Highlights are unwrapped; the original text is restored.
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(0)
    expect(surface.textContent).toBe('needle in a haystack, needle again')
  })

  it('closing an already-closed bar does not re-strip highlights', () => {
    plantSurface()
    openFindBar()
    closeFindBar()
    closeFindBar()

    // No thrown exception, no doubled DOM churn — the early return in
    // closeFindBar short-circuits when the bar is already inactive.
    expect($findInPage.get().active).toBe(false)
  })

  it('a fresh query wraps matches and counts them', () => {
    const surface = plantSurface()
    surface.textContent = 'needle haystack needle hay needle'
    openFindBar()
    setFindQuery('needle')

    expect($findInPage.get().query).toBe('needle')
    expect($findInPage.get().matchCount).toBe(3)
    expect($findInPage.get().matchOrdinal).toBe(1)
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(3)
  })

  it('clearing the query unwraps highlights and zeroes the counter', () => {
    const surface = plantSurface()
    surface.textContent = 'needle haystack needle'
    openFindBar()
    setFindQuery('needle')
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(2)

    setFindQuery('')

    expect($findInPage.get().matchCount).toBe(0)
    expect($findInPage.get().query).toBe('')
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(0)
    expect(surface.textContent).toBe('needle haystack needle')
  })

  it('findNext advances the active ordinal and findPrevious walks it back', () => {
    const surface = plantSurface()
    surface.textContent = 'needle needle needle'
    openFindBar()
    setFindQuery('needle')

    expect($findInPage.get().matchOrdinal).toBe(1)

    findNext()
    expect($findInPage.get().matchOrdinal).toBe(2)

    findNext()
    expect($findInPage.get().matchOrdinal).toBe(3)

    // Wraps around at the end.
    findNext()
    expect($findInPage.get().matchOrdinal).toBe(1)

    findPrevious()
    // Wraps backward from 1 → 3.
    expect($findInPage.get().matchOrdinal).toBe(3)

    findPrevious()
    expect($findInPage.get().matchOrdinal).toBe(2)
  })

  it('stepping with no query is a no-op (never searches invisibly)', () => {
    plantSurface()
    openFindBar()

    findNext()
    findPrevious()

    expect($findInPage.get().matchOrdinal).toBe(0)
    expect($findInPage.get().matchCount).toBe(0)
  })

  it('setFindQuery on a closed bar never searches', () => {
    // A debounce timer that already fired, or any late caller, must not
    // re-highlight the page after the user dismissed the bar.
    const surface = plantSurface()
    surface.textContent = 'needle needle'

    setFindQuery('needle')

    expect($findInPage.get().query).toBe('')
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(0)
  })

  it('scopes the search to the active chat surface, never the whole document', () => {
    // Regression (#81726): every active chat surface is mounted at once
    // (keep-alive), so a global Ctrl+F used to match across all of them.
    // The scoped walker must only wrap matches inside the surface the bar
    // was opened against.
    const foreground = plantSurface('foreground')
    const hidden = document.createElement('div')

    hidden.setAttribute('data-pane-hidden', '')
    hidden.setAttribute('data-chat-surface', '')
    hidden.id = 'background'
    hidden.textContent = 'needle in background chat needle'
    document.body.appendChild(hidden)

    foreground.textContent = 'needle in foreground chat needle'

    openFindBar()
    setFindQuery('needle')

    // Foreground: both matches wrapped. Background: untouched.
    expect(foreground.querySelectorAll('mark.find-hit').length).toBe(2)
    expect(hidden.querySelectorAll('mark.find-hit').length).toBe(0)
    expect($findInPage.get().matchCount).toBe(2)

    closeFindBar()
    expect(foreground.querySelectorAll('mark.find-hit').length).toBe(0)
    expect(hidden.querySelectorAll('mark.find-hit').length).toBe(0)
  })

  it('opening the bar twice on different surfaces retargets the scope', () => {
    // The walker resolves the scope at openFindBar() time, so a bar opened
    // while the foreground pane is chat-A searches chat-A, and a fresh bar
    // opened after the pane flips to chat-B searches chat-B. The previous
    // bar's highlights are already cleared by closeFindBar().
    const surfaceA = plantSurface('a')
    const surfaceB = document.createElement('div')

    surfaceB.setAttribute('data-chat-surface', '')
    surfaceB.id = 'b'
    surfaceB.textContent = 'target-b needle'
    document.body.appendChild(surfaceB)

    surfaceA.textContent = 'target-a needle'

    openFindBar()
    setFindQuery('needle')

    // Surface A is foreground; only its hit is wrapped.
    expect(surfaceA.querySelectorAll('mark.find-hit').length).toBe(1)
    expect(surfaceB.querySelectorAll('mark.find-hit').length).toBe(0)
    closeFindBar()

    // Flip visibility: surface B is now the foreground.
    surfaceA.setAttribute('data-pane-hidden', '')

    openFindBar()
    setFindQuery('needle')

    expect(surfaceA.querySelectorAll('mark.find-hit').length).toBe(0)
    expect(surfaceB.querySelectorAll('mark.find-hit').length).toBe(1)
    expect($findInPage.get().matchCount).toBe(1)
  })

  it('re-targets the scope after a keep-alive tab flip while the bar stays open', () => {
    // Regression (#81726): the bar only closes on a ROUTE change; a keep-
    // alive tab flip hides the captured surface without changing the
    // pathname, so a typed query used to resolve no scope and report 0/0
    // while the viewed surface contains the text. The scope must re-resolve
    // to the now-foreground surface.
    const surfaceA = plantSurface('a')
    const surfaceB = document.createElement('div')

    surfaceB.setAttribute('data-chat-surface', '')
    surfaceB.id = 'b'
    document.body.appendChild(surfaceB)

    surfaceA.textContent = 'needle in A'
    surfaceB.textContent = 'needle in B needle'

    openFindBar()
    setFindQuery('needle')

    // Surface A is foreground; only its match is wrapped.
    expect(surfaceA.querySelectorAll('mark.find-hit').length).toBe(1)
    expect(surfaceB.querySelectorAll('mark.find-hit').length).toBe(0)
    expect($findInPage.get().matchCount).toBe(1)

    // Tab flip while the bar stays open: A's pane layer hides, B is now the
    // foreground surface the user is reading.
    surfaceA.setAttribute('data-pane-hidden', '')

    setFindQuery('needle')

    // The scope re-targeted to B — its matches are wrapped, and A's stale
    // highlights are cleared rather than left to resurface on its next visit.
    expect(surfaceB.querySelectorAll('mark.find-hit').length).toBe(2)
    expect(surfaceA.querySelectorAll('mark.find-hit').length).toBe(0)
    expect($findInPage.get().matchCount).toBe(2)

    // Stepping still works on the re-targeted surface.
    findNext()
    expect($findInPage.get().matchOrdinal).toBe(2)
  })

  it('a no-match query leaves the count at zero and the DOM untouched', () => {
    const surface = plantSurface()
    surface.textContent = 'hello world'

    openFindBar()
    setFindQuery('nothing-here')

    expect($findInPage.get().matchCount).toBe(0)
    expect($findInPage.get().matchOrdinal).toBe(0)
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(0)
  })

  it('match is case-insensitive', () => {
    const surface = plantSurface()
    surface.textContent = 'NEEDLE needle Needle'

    openFindBar()
    setFindQuery('needle')

    expect($findInPage.get().matchCount).toBe(3)
  })

  it('a fresh query against existing highlights rewrites the marks (no leak)', () => {
    const surface = plantSurface()
    surface.textContent = 'alpha beta alpha beta'
    openFindBar()
    setFindQuery('alpha')
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(2)

    setFindQuery('beta')

    // The previous marks were unwrapped before the new query highlighted.
    expect($findInPage.get().matchCount).toBe(2)
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(2)
    // Original text intact (no doubled marks / no lost text).
    expect(surface.textContent).toBe('alpha beta alpha beta')
  })

  it('a query typed with no chat surface stays zero-count (parity with the bridge path)', () => {
    // No chat surface in the DOM. The bar still tracks the typed query
    // (matches the contract users had with the Electron bridge), but
    // obviously cannot find anything — the view is whatever they're in.
    openFindBar()
    setFindQuery('needle')

    expect($findInPage.get().query).toBe('needle')
    expect($findInPage.get().matchCount).toBe(0)
    expect($findInPage.get().matchOrdinal).toBe(0)
  })

  it('found-in-page results from a secondary window still land on the store', () => {
    // Retained path: secondary session windows drive the Electron bridge,
    // not the renderer-side walker. The bridge event still maps onto the
    // same store shape.
    plantSurface()
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
    const surface = plantSurface()
    surface.textContent = 'one two three four'
    openFindBar()
    renderFindBar()

    const input = await screen.findByRole('searchbox', { name: /find in page/i })
    expect(input).toBeTruthy()
    expect(input.getAttribute('type')).toBe('search')
    expect(input.getAttribute('role')).toBeNull()
    expect(screen.getByRole('search').className).toContain('top-[calc(var(--titlebar-height,34px)+0.5rem)]')
    expect(screen.getByRole('button', { name: /close/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /next match/i })).toBeTruthy()
    expect(screen.getByRole('button', { name: /previous match/i })).toBeTruthy()

    // Counter appears once a query + results exist.
    expect(screen.queryByText('3/12')).toBeNull()
    actStore(() => $findInPage.set({ active: true, query: 'two', matchOrdinal: 1, matchCount: 1 }))
    await waitFor(() => expect(screen.getByText('1/1')).toBeTruthy())
  })

  it('focuses the input on open', async () => {
    plantSurface()
    openFindBar()
    renderFindBar()

    const input = await screen.findByRole('searchbox', { name: /find in page/i })
    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    await waitFor(() => expect(document.activeElement).toBe(input))
  })

  it('debounces typing into a single scoped find', async () => {
    vi.useFakeTimers()
    const surface = plantSurface()
    surface.textContent = 'one nee needle'

    try {
      openFindBar()
      renderFindBar()

      const input = screen.getByRole('searchbox', { name: /find in page/i })
      fireEvent.change(input, { target: { value: 'n' } })
      fireEvent.change(input, { target: { value: 'ne' } })
      fireEvent.change(input, { target: { value: 'nee' } })

      // No marks yet — debounced.
      expect(surface.querySelectorAll('mark.find-hit').length).toBe(0)

      // Flushing the debounce updates the store, which re-renders the bar.
      act(() => {
        vi.advanceTimersByTime(200)
      })

      // Now the scoped walker has wrapped the matches.
      expect($findInPage.get().query).toBe('nee')
      expect($findInPage.get().matchCount).toBe(2)
      expect(surface.querySelectorAll('mark.find-hit').length).toBe(2)
    } finally {
      vi.useRealTimers()
    }
  })

  it('excludes the semantic searchbox only while Chromium indexes the page', async () => {
    vi.useFakeTimers()
    let resolveFind: (() => void) | undefined
    bridge.findInPage.mockReturnValueOnce(new Promise(resolve => (resolveFind = () => resolve({ count: 0 }))))

    try {
      openFindBar()
      renderFindBar()

      const input = screen.getByRole('searchbox', { name: /find in page/i }) as HTMLInputElement
      input.focus()
      fireEvent.change(input, { target: { value: 'needle' } })
      input.setSelectionRange(6, 6)

      act(() => vi.advanceTimersByTime(200))

      expect(input.inert).toBe(true)
      expect(input.type).toBe('search')
      expect(input.getAttribute('role')).toBeNull()

      await act(async () => resolveFind?.())
      expect(input.inert).toBe(false)
      // eslint-disable-next-line no-restricted-globals -- the exclusion cycle must restore real focus
      expect(document.activeElement).toBe(input)
      expect(input.selectionStart).toBe(6)
    } finally {
      vi.useRealTimers()
    }
  })

  it('does not fire a pending search after the bar closes', async () => {
    vi.useFakeTimers()
    const surface = plantSurface()
    surface.textContent = 'needle'

    try {
      openFindBar()
      renderFindBar()

      fireEvent.change(screen.getByRole('searchbox', { name: /find in page/i }), {
        target: { value: 'needle' }
      })

      actStore(closeFindBar)
      vi.advanceTimersByTime(500)

      // The debounced timer that already fired is gated by `active`, so
      // closing the bar prevents the walker from running.
      expect(surface.querySelectorAll('mark.find-hit').length).toBe(0)
      expect($findInPage.get().query).toBe('')
    } finally {
      vi.useRealTimers()
    }
  })

  it('Enter dispatches next and Shift+Enter dispatches previous', () => {
    const surface = plantSurface()
    surface.textContent = 'needle needle needle'
    $findInPage.set({ active: true, query: 'needle', matchOrdinal: 1, matchCount: 3 })
    // Manually replay the open+query path so the marks exist for stepping.
    openFindBar()
    setFindQuery('needle')

    renderFindBar()
    const input = screen.getByRole('searchbox', { name: /find in page/i })

    fireEvent.keyDown(input, { key: 'Enter' })
    expect($findInPage.get().matchOrdinal).toBe(2)

    fireEvent.keyDown(input, { key: 'Enter', shiftKey: true })
    expect($findInPage.get().matchOrdinal).toBe(1)
  })

  it('Cmd+G / Cmd+Shift+G step from outside the input while the bar is open', async () => {
    const surface = plantSurface()
    surface.textContent = 'needle needle needle'

    openFindBar()
    setFindQuery('needle')
    expect($findInPage.get().matchOrdinal).toBe(1)

    renderFindBar()

    // Fired on window (not the input) — the accelerator must not require focus.
    fireEvent.keyDown(window, { key: 'g', metaKey: true })
    expect($findInPage.get().matchOrdinal).toBe(2)

    fireEvent.keyDown(window, { key: 'G', metaKey: true, shiftKey: true })
    expect($findInPage.get().matchOrdinal).toBe(1)
  })

  it('Escape closes the bar and unwraps the highlights', async () => {
    const surface = plantSurface()
    surface.textContent = 'needle haystack'
    openFindBar()
    setFindQuery('needle')
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(1)

    renderFindBar()
    fireEvent.keyDown(window, { key: 'Escape' })

    expect($findInPage.get().active).toBe(false)
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(0)
    await waitFor(() => expect(screen.queryByRole('search')).toBeNull())
  })

  it('the close button clears state and unwraps the highlights', async () => {
    const surface = plantSurface()
    surface.textContent = 'needle haystack'
    openFindBar()
    setFindQuery('needle')
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(1)

    renderFindBar()
    fireEvent.click(screen.getByRole('button', { name: /close/i }))

    expect($findInPage.get().active).toBe(false)
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(0)
  })

  it('the next / previous buttons dispatch a step', () => {
    const surface = plantSurface()
    surface.textContent = 'needle needle needle'
    openFindBar()
    setFindQuery('needle')

    renderFindBar()

    fireEvent.click(screen.getByRole('button', { name: /next match/i }))
    expect($findInPage.get().matchOrdinal).toBe(2)

    fireEvent.click(screen.getByRole('button', { name: /previous match/i }))
    expect($findInPage.get().matchOrdinal).toBe(1)
  })

  it('keeps exactly one bridge subscription across an open/close cycle', async () => {
    plantSurface()
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
    plantSurface()
    openFindBar()
    const { unmount } = renderFindBar()

    expect($findInPage.get().active).toBe(true)

    unmount()

    // Reopen the store WITHOUT a mounted bar; a leaked listener would
    // intercept Escape and call closeFindBar, flipping $findInPage back to
    // inactive. With the listener removed, the Escape bubbles into the
    // void and the store stays open — proving the bar's capture-phase
    // window listener did not leak.
    actStore(openFindBar)
    fireEvent.keyDown(window, { key: 'Escape' })

    expect($findInPage.get().active).toBe(true)
  })

  it('navigating to another route closes the bar and clears the highlights', async () => {
    const surface = plantSurface()
    surface.textContent = 'needle haystack'
    openFindBar()
    setFindQuery('needle')
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(1)

    const { navigate } = renderFindBarWithNavigation('/session/a')
    await waitFor(() => expect(screen.getByRole('search')).toBeTruthy())

    navigate('/session/b')

    // Bar gone, state reset, and the highlights stripped — stale marks
    // must not survive a session switch.
    await waitFor(() => expect(screen.queryByRole('search')).toBeNull())
    expect($findInPage.get()).toEqual({ active: false, query: '', matchOrdinal: 0, matchCount: 0 })
    expect(surface.querySelectorAll('mark.find-hit').length).toBe(0)
  })

  it('navigation while the bar is closed does not reach into the bridge', async () => {
    plantSurface()
    const { navigate } = renderFindBarWithNavigation('/session/a')

    navigate('/session/b')

    await waitFor(() => expect(screen.queryByRole('search')).toBeNull())
    // No thrown exception, no DOM churn — the route-change cleanup is a
    // no-op when there's no bar to close.
    expect($findInPage.get().active).toBe(false)
  })

  it('does not render on overlay routes (settings, command center, …)', () => {
    // Match isOverlayView: agents, command-center, cron, profiles, settings,
    // starmap, webhooks. Test with the most commonly hit one.
    openFindBar()
    renderFindBar('/settings')

    expect(screen.queryByRole('search')).toBeNull()
    // The store still has active=true but the component guard suppresses
    // rendering — a harmless state ghost that the next pathname-change
    // cleanup will drain via closeFindBar.
    expect($findInPage.get().active).toBe(true)
  })
})

// ── Keybind gate: view.findInPage on overlay routes ─────────────────────────
// The component guard above proves hidden RENDERING; this proves the keybind
// itself never OPENS the bar on an overlay route. Mounted against the real
// useKeybinds listener + combo index so a regression in either the handler
// wiring or the route classification fails the test.

function KeybindHarness({ deps }: { deps: KeybindRuntimeDeps }) {
  useKeybinds(deps)

  return null
}

describe('view.findInPage keybind gate', () => {
  function renderKeybinds(pathname: string) {
    const deps: KeybindRuntimeDeps = {
      toggleCommandCenter: vi.fn(),
      startFreshSession: vi.fn(),
      openNewSessionTab: vi.fn(),
      toggleSelectedPin: vi.fn(),
      archiveSelectedSession: vi.fn()
    }

    render(
      <MemoryRouter initialEntries={[pathname]}>
        <I18nProvider configClient={null} initialLocale="en">
          <KeybindHarness deps={deps} />
        </I18nProvider>
      </MemoryRouter>
    )

    return deps
  }

  it('does not open the find bar for mod+f while an overlay route is showing', () => {
    renderKeybinds('/settings')

    fireEvent.keyDown(window, { key: 'f', metaKey: true })

    expect($findInPage.get().active).toBe(false)
  })

  it('opens the find bar for mod+f on a normal chat route', () => {
    renderKeybinds('/session/a')

    fireEvent.keyDown(window, { key: 'f', metaKey: true })

    expect($findInPage.get().active).toBe(true)
  })
})

// ── Files-pane-aware positioning ────────────────────────────────────────────

describe('FindBar files pane positioning', () => {
  function mountAside(width = 240) {
    // eslint-disable-next-line no-restricted-globals -- the component queries the live document for the aside
    const aside = document.createElement('aside')
    aside.setAttribute('aria-label', 'Right sidebar')
    // jsdom has no layout — stub the rect the component measures.
    aside.getBoundingClientRect = () => ({ left: window.innerWidth - width, width }) as DOMRect
    // eslint-disable-next-line no-restricted-globals -- must land in the real DOM the component measures
    document.body.appendChild(aside)

    return aside
  }

  afterEach(() => {
    // eslint-disable-next-line no-restricted-globals -- cleanup of the real DOM the component measures
    for (const aside of document.querySelectorAll('aside[aria-label="Right sidebar"]')) {
      aside.remove()
    }
  })

  it('keeps the default right-4 position when the pane is closed', async () => {
    openFindBar()
    renderFindBar()

    const bar = await screen.findByRole('search')
    expect(bar.style.right).toBe('')
  })

  it('parks the bar left of the pane when it is open', async () => {
    mountAside(240)
    openFindBar()
    renderFindBar()

    const bar = await screen.findByRole('search')
    await waitFor(() => expect(bar.style.right).toBe('calc(240px + 0.75rem)'))
  })

  it('re-measures when the pane opens or closes while the bar is up', async () => {
    openFindBar()
    renderFindBar()

    const bar = await screen.findByRole('search')
    expect(bar.style.right).toBe('')

    // Pane opens mid-session: the MutationObserver path must re-measure.
    const aside = mountAside(300)
    await waitFor(() => expect(bar.style.right).toBe('calc(300px + 0.75rem)'))

    // Pane closes again: the bar falls back to the default position.
    aside.remove()
    await waitFor(() => expect(bar.style.right).toBe(''))
  })
})
