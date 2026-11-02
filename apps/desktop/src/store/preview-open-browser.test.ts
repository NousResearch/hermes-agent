import { beforeEach, describe, expect, it } from 'vitest'

import { $previewTabs, closeRightRail, openBrowserTab, openPreview } from './preview'

beforeEach(() => {
  closeRightRail()
})

describe('openBrowserTab', () => {
  it('opens a blank browser when there is no browser tab yet', () => {
    openBrowserTab()

    const tabs = $previewTabs.get()

    expect(tabs).toHaveLength(1)
    // about:blank, so the pane lands on its empty state inviting an address
    // rather than a white void.
    expect(tabs[0]?.target.url).toBe('about:blank')
  })

  // The hotkey is "show me the browser", not "reset the browser" — pressing it
  // while a page is loaded must not throw that page away.
  it('re-fronts the existing page instead of blanking it', () => {
    openPreview({ kind: 'url', label: 'Example', source: 'https://example.com', url: 'https://example.com' })
    openBrowserTab()

    const tabs = $previewTabs.get()

    expect(tabs).toHaveLength(1)
    expect(tabs[0]?.target.url).toBe('https://example.com')
  })

  // "Show me the browser" is idempotent — the "+" is how you ask for another.
  it('shows the browser it already has rather than stacking duplicates', () => {
    openBrowserTab()
    openBrowserTab()
    openBrowserTab()

    expect($previewTabs.get()).toHaveLength(1)
  })

  it('leaves other preview tabs alone', () => {
    openPreview({ kind: 'file', label: 'notes.md', source: '/work/notes.md', url: 'file:///work/notes.md' })
    openBrowserTab()

    const tabs = $previewTabs.get()

    expect(tabs).toHaveLength(2)
    expect(tabs.map(tab => tab.target.url)).toContain('file:///work/notes.md')
  })
})
