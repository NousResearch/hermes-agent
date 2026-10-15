import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  $canGoBack,
  $viewHistory,
  backVisible,
  goBackToPreviousView,
  noteActiveViewPane,
  noteViewChange,
  resetViewPaneMemory,
  viewPaneFor
} from './view-history'

/** The whole state is the atom — the cursor lives in it, not in a module var. */
const reset = () => {
  $viewHistory.set({ entries: [], index: -1 })
  resetViewPaneMemory()
}

describe('view history', () => {
  beforeEach(reset)

  it('has nothing behind the first route of a window', () => {
    noteViewChange('/')

    expect($canGoBack.get()).toBe(false)
    expect(goBackToPreviousView(vi.fn())).toBe(false)
  })

  it('keeps the view it is leaving behind the cursor', () => {
    noteViewChange('/')
    noteViewChange('/settings')

    expect($viewHistory.get()).toEqual({ entries: ['/', '/settings'], index: 1 })
    expect($canGoBack.get()).toBe(true)
  })

  it('returns from a page to the view it was opened from', () => {
    const navigate = vi.fn()

    noteViewChange('/session-1')
    noteViewChange('/capabilities')

    expect(goBackToPreviousView(navigate)).toBe(true)
    expect(navigate).toHaveBeenCalledWith('/session-1')
  })

  it('is not re-recorded by the navigation it causes, so Back cannot ping-pong', () => {
    noteViewChange('/session-1')
    noteViewChange('/capabilities')
    goBackToPreviousView(vi.fn())

    // The route change Back itself just made — the window is on the session view again.
    noteViewChange('/session-1')

    expect($viewHistory.get()).toEqual({ entries: ['/session-1', '/capabilities'], index: 0 })
    expect($canGoBack.get()).toBe(false)
  })

  it('walks several views back, one step at a time', () => {
    const navigate = vi.fn()

    noteViewChange('/session-1')
    noteViewChange('/capabilities')
    noteViewChange('/artifacts')

    goBackToPreviousView(navigate)
    goBackToPreviousView(navigate)

    expect(navigate.mock.calls).toEqual([['/capabilities'], ['/session-1']])
  })

  it('records no entry for a dismissed overlay, so Back cannot reopen it', () => {
    noteViewChange('/session-1')
    noteViewChange('/settings')
    // Closing the overlay is a `replace` onto the view it was covering.
    noteViewChange('/session-1')

    expect($viewHistory.get()).toEqual({ entries: ['/session-1', '/settings'], index: 0 })

    noteViewChange('/capabilities')

    expect($viewHistory.get()).toEqual({ entries: ['/session-1', '/capabilities'], index: 1 })
    expect(goBackToPreviousView(vi.fn())).toBe(true)
  })

  it('drops the views ahead of the cursor when a new one opens', () => {
    noteViewChange('/a')
    noteViewChange('/b')
    noteViewChange('/c')
    goBackToPreviousView(vi.fn())

    noteViewChange('/d')

    expect($viewHistory.get()).toEqual({ entries: ['/a', '/b', '/d'], index: 2 })
  })

  it('does not record a re-entry into the route it is already on', () => {
    noteViewChange('/settings')
    noteViewChange('/settings')

    expect($viewHistory.get()).toEqual({ entries: ['/settings'], index: 0 })
  })

  it('caps the history so a long session cannot grow it without limit', () => {
    for (let index = 0; index < 40; index += 1) {
      noteViewChange(`/session-${index}`)
    }

    const { entries, index } = $viewHistory.get()

    expect(entries).toHaveLength(32)
    expect(index).toBe(31)
    expect(entries.at(-1)).toBe('/session-39')
    expect(entries.at(0)).toBe('/session-8')
  })
})

describe('the main-strip tab a view fronted', () => {
  beforeEach(reset)

  it('is restored alongside the route by Back, so a session tile is not abandoned', () => {
    const navigate = vi.fn()
    const restorePane = vi.fn()

    noteViewChange('/session-1')
    noteActiveViewPane('session-tile:other')
    noteViewChange('/capabilities')
    noteActiveViewPane('workspace')

    expect(goBackToPreviousView(navigate, restorePane)).toBe(true)
    expect(navigate).toHaveBeenCalledWith('/session-1')
    expect(restorePane).toHaveBeenCalledWith('session-tile:other')
  })

  it('is remembered per view, not globally', () => {
    noteViewChange('/session-1')
    noteActiveViewPane('session-tile:a')

    noteViewChange('/capabilities')
    noteActiveViewPane('workspace')

    expect(viewPaneFor('/session-1')).toBe('session-tile:a')
    expect(viewPaneFor('/capabilities')).toBe('workspace')
  })

  it('leaves nothing to restore when the view never fronted a tab', () => {
    const restorePane = vi.fn()

    noteViewChange('/session-1')
    noteViewChange('/capabilities')

    goBackToPreviousView(vi.fn(), restorePane)

    expect(restorePane).not.toHaveBeenCalled()
  })

  it('is forgotten once the capped history has dropped the view', () => {
    noteViewChange('/old')
    noteActiveViewPane('session-tile:old')

    for (let index = 0; index < 40; index += 1) {
      noteViewChange(`/session-${index}`)
      noteActiveViewPane(`session-tile:${index}`)
    }

    expect(viewPaneFor('/old')).toBeNull()
  })
})

describe('backVisible', () => {
  it('hides the affordance on the session view and when nothing is behind it', () => {
    expect(backVisible('chat', true)).toBe(false)
    expect(backVisible('settings', false)).toBe(false)
    expect(backVisible('settings', true)).toBe(true)
  })
})
