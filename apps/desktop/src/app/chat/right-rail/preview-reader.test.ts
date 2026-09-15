import { beforeEach, describe, expect, it } from 'vitest'

import { findGroup, findGroupOfPane } from '@/components/pane-shell/tree/model'
import * as tree from '@/components/pane-shell/tree/store'
import { $rightRailActiveTabId, selectRightRailTab } from '@/store/layout'
import { $previewTabs, closeRightRail, openPreview, type PreviewTarget } from '@/store/preview'

import {
  PREVIEW_READ_MAX_CHARS,
  readActivePreview,
  registerPreviewPageReader,
  resolveActivePreviewTab
} from './preview-reader'

function urlTarget(url: string): PreviewTarget {
  return { kind: 'url', label: 'Browser', source: url, url }
}

function fileTarget(path: string): PreviewTarget {
  return { kind: 'file', label: path, path, previewKind: 'text', source: path, url: `file://${path}` }
}

/** Browser tab id minted by openPreview (url:browser-<uuid>), not a fixed singleton. */
function activeTabId(): string {
  const id = $rightRailActiveTabId.get()
  if (!id) {
    throw new Error('expected an active right-rail tab')
  }

  return id
}

describe('readActivePreview (read_preview tool)', () => {
  // URL targets share a Browser vessel when one is already active, so a reader
  // registered in one test would answer the next — unregister whatever a test installed.
  let cleanups: Array<() => void> = []

  const register = (tabId: string, reader: Parameters<typeof registerPreviewPageReader>[1]) => {
    const unregister = registerPreviewPageReader(tabId, reader)

    cleanups.push(unregister)

    return unregister
  }

  beforeEach(() => {
    for (const cleanup of cleanups) {
      cleanup()
    }

    cleanups = []
    closeRightRail()
    window.localStorage.clear()
    tree.noteActiveTreeGroup(null)
    tree.noteHoveredTreeGroup(null)
    // declareDefaultTree only fully replaces when no current tree exists.
    tree.$layoutTree.set(null)
  })

  it('answers null when nothing is open, so the tool reports it cleanly', async () => {
    expect(await readActivePreview()).toBeNull()
  })

  it('serializes the Browser tab through its registered page reader', async () => {
    openPreview(urlTarget('https://news.ycombinator.com'), 'tool-result')
    const browserId = activeTabId()
    register(browserId, async () => ({
      text: 'Top stories…',
      title: 'Hacker News',
      url: 'https://news.ycombinator.com/news'
    }))

    expect(await readActivePreview()).toMatchObject({
      active_tab_id: browserId,
      kind: 'url',
      text: 'Top stories…',
      title: 'Hacker News',
      total_chars: 12,
      // The live address wins over the target (in-page navigation).
      url: 'https://news.ycombinator.com/news'
    })
  })

  it('windows long pages with start/count and reports the full length', async () => {
    openPreview(urlTarget('https://example.com'), 'tool-result')
    register(activeTabId(), async () => ({
      text: 'abcdefghij',
      title: 't',
      url: ''
    }))

    expect(await readActivePreview({ count: 4, start: 2 })).toMatchObject({
      end: 6,
      start: 2,
      text: 'cdef',
      total_chars: 10
    })
  })

  it('caps a single read at PREVIEW_READ_MAX_CHARS even when asked for more', async () => {
    openPreview(urlTarget('https://example.com'), 'tool-result')
    register(activeTabId(), async () => ({
      text: 'x'.repeat(PREVIEW_READ_MAX_CHARS + 5000),
      title: 't',
      url: ''
    }))

    const result = await readActivePreview({ count: PREVIEW_READ_MAX_CHARS + 5000 })

    expect(result?.text).toHaveLength(PREVIEW_READ_MAX_CHARS)
    expect(result?.total_chars).toBe(PREVIEW_READ_MAX_CHARS + 5000)
  })

  it('answers identity + retry note for a Browser tab whose pane is not mounted', async () => {
    openPreview(urlTarget('https://example.com'), 'tool-result')

    expect(await readActivePreview()).toMatchObject({
      kind: 'url',
      note: expect.stringContaining('retry') as string,
      text: '',
      url: 'https://example.com'
    })
  })

  it('answers a file tab with its identity and points at read_file', async () => {
    openPreview(fileTarget('/work/notes.md'), 'file-browser')

    expect(await readActivePreview()).toMatchObject({
      kind: 'file',
      note: expect.stringContaining('read_file') as string,
      path: '/work/notes.md'
    })
  })

  it('reads the tab the user is LOOKING at, not the last one opened', async () => {
    openPreview(fileTarget('/work/one.md'), 'file-browser')
    openPreview(fileTarget('/work/two.md'), 'file-browser')
    selectRightRailTab('file:file:///work/one.md')

    expect(await readActivePreview()).toMatchObject({ path: '/work/one.md' })
  })

  it('falls back to the identity answer when the reader throws (webview booting)', async () => {
    openPreview(urlTarget('https://example.com'), 'tool-result')
    register(activeTabId(), async () => {
      throw new Error('webview gone')
    })

    expect(await readActivePreview()).toMatchObject({ note: expect.stringContaining('retry') as string, text: '' })
  })

  it('unregister is idempotent and scoped to the same reader', async () => {
    openPreview(urlTarget('https://example.com'), 'tool-result')
    const tabId = activeTabId()
    const first = register(tabId, async () => ({ text: 'first', title: '', url: '' }))

    register(tabId, async () => ({ text: 'second', title: '', url: '' }))
    // Unregistering the STALE reader must not evict the live one.
    first()

    expect(await readActivePreview()).toMatchObject({ text: 'second' })
  })

  it('includes tabs[] metadata when more than one preview is open', async () => {
    openPreview(fileTarget('/work/project-network.html'), 'tool-result')
    const fileId = activeTabId()
    openPreview(urlTarget('https://example.com/tickets'), 'tool-result')
    const browserId = activeTabId()

    const result = await readActivePreview()
    expect(result).toMatchObject({
      active_tab_id: browserId,
      kind: 'url',
      url: 'https://example.com/tickets'
    })
    expect(result?.tabs?.map(t => t.id).sort()).toEqual([fileId, browserId].sort())
    expect(result?.note).toMatch(/Multiple preview tabs/i)
  })

  it('honors the hovered preview zone over a stale global file selection', async () => {
    const model = await import('@/components/pane-shell/tree/model')

    openPreview(fileTarget('/work/a.md'), 'file-browser')
    const fileId = activeTabId()
    openPreview(urlTarget('https://example.com'), 'tool-result')
    const browserId = activeTabId()
    // Stale global active (pre-fix clobber) while Browser remains open.
    selectRightRailTab(fileId)

    tree.declareDefaultTree(
      model.split('row', [
        model.group([`preview-tile:${browserId}`], {
          active: `preview-tile:${browserId}`,
          id: 'grp-browser'
        }),
        model.group([`preview-tile:${fileId}`], { active: `preview-tile:${fileId}`, id: 'grp-file' })
      ])
    )

    tree.noteHoveredTreeGroup('grp-browser')
    expect(resolveActivePreviewTab()?.id).toBe(browserId)
    expect(await readActivePreview()).toMatchObject({ active_tab_id: browserId, kind: 'url' })
  })

  it('honors the focused preview zone over a stale global file selection', async () => {
    const model = await import('@/components/pane-shell/tree/model')

    openPreview(fileTarget('/work/a.md'), 'file-browser')
    const fileId = activeTabId()
    openPreview(urlTarget('https://example.com'), 'tool-result')
    const browserId = activeTabId()
    selectRightRailTab(fileId)

    tree.declareDefaultTree(
      model.split('row', [
        model.group([`preview-tile:${browserId}`], {
          active: `preview-tile:${browserId}`,
          id: 'grp-browser'
        }),
        model.group([`preview-tile:${fileId}`], { active: `preview-tile:${fileId}`, id: 'grp-file' })
      ])
    )

    tree.noteActiveTreeGroup('grp-browser')
    expect(resolveActivePreviewTab()?.id).toBe(browserId)
  })

  it('falls back to the last open Browser when global active is a missing id', async () => {
    openPreview(urlTarget('https://example.com'), 'tool-result')
    const browserId = activeTabId()
    // Simulate a stale global pointer (old singleton id, closed tab, etc.).
    selectRightRailTab('url:browser' as typeof browserId)

    expect($previewTabs.get().some(t => t.id === browserId)).toBe(true)
    expect(resolveActivePreviewTab()?.id).toBe(browserId)
  })
})

describe('openPreview reveal retargets activeTreeGroup (split-zone clobber)', () => {
  beforeEach(() => {
    closeRightRail()
    window.localStorage.clear()
    tree.noteActiveTreeGroup(null)
    tree.noteHoveredTreeGroup(null)
    tree.$layoutTree.set(null)
  })

  it('notes the Browser zone on reveal so follow cannot snap global active back to a file sibling', async () => {
    const model = await import('@/components/pane-shell/tree/model')

    openPreview(fileTarget('/work/project-network.html'), 'tool-result')
    const fileId = activeTabId()

    // Pre-create the Browser vessel so the split layout can name its pane before open.
    openPreview(urlTarget('about:blank'), 'manual')
    const browserId = activeTabId()
    const browserPane = `preview-tile:${browserId}`
    const filePane = `preview-tile:${fileId}`

    // File zone is what the user last touched; Browser will re-front in a sibling zone.
    tree.declareDefaultTree(
      model.split('row', [
        model.group([browserPane], { active: browserPane, id: 'grp-browser' }),
        model.group([filePane], { active: filePane, id: 'grp-file' })
      ])
    )
    tree.noteActiveTreeGroup('grp-file')
    selectRightRailTab(fileId)

    // Mirror watchPreviewTiles reveal (noteActiveTreeGroup after revealTreePane).
    const reveal = (tabId: string) => {
      const paneId = `preview-tile:${tabId}`
      tree.revealTreePane(paneId)
      const t = tree.$layoutTree.get()
      const group = t ? findGroupOfPane(t, paneId) : null
      if (group) {
        tree.noteActiveTreeGroup(group.id)
      }
    }

    openPreview(urlTarget('https://example.com/tickets'), 'tool-result')
    expect($rightRailActiveTabId.get()).toBe(browserId)
    reveal(browserId)

    // follow from layout would previously use stale grp-file and clobber Browser.
    const follow = () => {
      const t = tree.$layoutTree.get()
      const groupId = tree.$activeTreeGroup.get()
      const active = groupId && t ? findGroup(t, groupId)?.active : undefined
      if (!active?.startsWith('preview-tile:')) return
      const tabId = active.slice('preview-tile:'.length)
      if ($rightRailActiveTabId.get() !== tabId) {
        selectRightRailTab(tabId as typeof fileId)
      }
    }

    follow()

    expect(tree.$activeTreeGroup.get()).toBe('grp-browser')
    expect($rightRailActiveTabId.get()).toBe(browserId)
    expect(await readActivePreview()).toMatchObject({
      active_tab_id: browserId,
      kind: 'url',
      url: 'https://example.com/tickets'
    })
  })
})
