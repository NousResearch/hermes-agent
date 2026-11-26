import { beforeEach, describe, expect, it } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { $dismissedPanes, $layoutTree, isPaneVisible } from '@/components/pane-shell/tree/store'

import { $previewTabs, closeRightRail, openBrowserTab, openPreview, toggleBrowserTab } from './preview'

beforeEach(() => {
  closeRightRail()
  $dismissedPanes.set(new Set())
  $layoutTree.set(null)
})

describe('toggleBrowserTab', () => {
  it('opens a blank browser when there is no browser tab yet', () => {
    toggleBrowserTab()

    const tabs = $previewTabs.get()

    expect(tabs).toHaveLength(1)
    expect(tabs[0]?.target.url).toBe('about:blank')
  })

  // Close must not cost the page: the TAB survives so the next toggle re-fronts
  // what was showing instead of landing on about:blank.
  it('dismisses the visible browser pane but keeps its tab', () => {
    openBrowserTab()

    const paneId = `preview-tile:${$previewTabs.get()[0]?.id}`

    // The shape watchPreviewTiles maintains: the mirrored pane in the tree,
    // fronted in its zone.
    $layoutTree.set(group([paneId, 'workspace'], { active: paneId, id: 'g-main' }))
    expect(isPaneVisible(paneId)).toBe(true)

    toggleBrowserTab()

    expect($previewTabs.get()).toHaveLength(1)
    expect(isPaneVisible(paneId)).toBe(false)
  })

  it('re-fronts the existing page instead of blanking it', () => {
    openPreview({ kind: 'url', label: 'Example', source: 'https://example.com', url: 'https://example.com' })

    // Browser away (no tree, pane not on screen): the toggle brings it back.
    toggleBrowserTab()

    const tabs = $previewTabs.get()

    expect(tabs).toHaveLength(1)
    expect(tabs[0]?.target.url).toBe('https://example.com')
  })

  it('leaves other preview tabs alone', () => {
    openPreview({ kind: 'file', label: 'notes.md', source: '/work/notes.md', url: 'file:///work/notes.md' })
    toggleBrowserTab()

    const tabs = $previewTabs.get()

    expect(tabs).toHaveLength(2)
    expect(tabs.map(tab => tab.target.url)).toContain('file:///work/notes.md')
  })
})
