import { beforeEach, describe, expect, it } from 'vitest'

import { $rightRailActiveTabId } from './layout'
import {
  $browserPages,
  $previewTabs,
  closeRightRail,
  closeRightRailTab,
  markBrowserTabPopped,
  openBrowserTab,
  openPreview,
  setPreviewTabCloser,
  toggleBrowserTab
} from './preview'

beforeEach(() => {
  closeRightRail()
  setPreviewTabCloser(tabId => {
    const { [tabId]: gone, ...rest } = $browserPages.get()

    if (gone) {
      $browserPages.set(rest)
    }

    closeRightRailTab(tabId)
  })
})

describe('toggleBrowserTab', () => {
  it('opens a blank browser when there is no browser tab yet', () => {
    toggleBrowserTab()

    const tabs = $previewTabs.get()

    expect(tabs).toHaveLength(1)
    expect(tabs[0]?.target.url).toBe('about:blank')
  })

  // The whole point of a titlebar toggle: press it again while the Browser is
  // the tab you're looking at and it goes away, the same shape as every other
  // show/hide affordance in the shell (GitHub pane, terminal, files).
  it('closes the browser tab when it is already the active tab', () => {
    toggleBrowserTab()
    expect($previewTabs.get()).toHaveLength(1)

    toggleBrowserTab()

    expect($previewTabs.get()).toHaveLength(0)
    expect($rightRailActiveTabId.get()).toBeNull()
  })

  // A hidden Browser (you're looking at something else) must re-front, not
  // vanish — "toggle" means the active-vs-not state, never a second press
  // while unrelated content is focused.
  it('re-fronts the browser instead of closing it when another tab is active', () => {
    openBrowserTab()
    openPreview({ kind: 'file', label: 'notes.md', source: '/work/notes.md', url: 'file:///work/notes.md' })
    expect($rightRailActiveTabId.get()).not.toBe($previewTabs.get()[0]?.id)

    toggleBrowserTab()

    const tabs = $previewTabs.get()

    expect(tabs).toHaveLength(2)
    expect($rightRailActiveTabId.get()).toBe(tabs.find(tab => tab.target.kind === 'url')?.id)
  })

  // A close is a real close — the tab and its page are gone, same as
  // closing any other tab. Reopening after that starts fresh on
  // about:blank; toggleBrowserTab does not secretly remember a closed page.
  it('starts fresh on about:blank after a close/reopen cycle', () => {
    openPreview({ kind: 'url', label: 'Example', source: 'https://example.com', url: 'https://example.com' })
    toggleBrowserTab() // closes it
    toggleBrowserTab() // reopens it

    const tabs = $previewTabs.get()

    expect(tabs).toHaveLength(1)
    expect(tabs[0]?.target.url).toBe('about:blank')
  })

  // Regression guard: `openBrowserTab` itself must stay open-only. Chat
  // links and other callers rely on it never toggling a page closed under
  // them — only the dedicated toggle wrapper may collapse the tab.
  it('does not regress openBrowserTab into a toggle', () => {
    openBrowserTab()
    openBrowserTab()
    openBrowserTab()

    expect($previewTabs.get()).toHaveLength(1)
  })

  // Popping a tab out leaves it in the tab list and still active — only a
  // "popped" flag hides it from the docked view. Closing it here would drop
  // the tab while its window survives, empty, with nothing that can close it.
  it('leaves a popped-out browser tab alone', () => {
    toggleBrowserTab()
    const id = $previewTabs.get()[0]?.id

    markBrowserTabPopped(id!, true)

    toggleBrowserTab()

    expect($previewTabs.get().map(tab => tab.id)).toEqual([id])
    expect($rightRailActiveTabId.get()).toBe(id)
  })

  // The page record is keyed by a random id, so a close that drops the tab
  // without dropping it leaks one entry per press for the life of the window.
  // The closer the pane registers is what knows about that record.
  it('drops the page record when it closes', () => {
    openPreview({ kind: 'url', label: 'Example', source: 'https://example.com', url: 'https://example.com' })
    const id = $previewTabs.get()[0]?.id

    $browserPages.set({ [id!]: { title: 'Example', url: 'https://example.com' } })

    toggleBrowserTab()

    expect($previewTabs.get()).toHaveLength(0)
    expect($browserPages.get()[id!]).toBeUndefined()
  })
})
