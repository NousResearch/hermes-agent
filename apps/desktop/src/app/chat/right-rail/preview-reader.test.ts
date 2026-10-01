import { beforeEach, describe, expect, it } from 'vitest'

import { group, split } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup, noteHoveredTreeGroup } from '@/components/pane-shell/tree/store'
import { $rightRailActiveTabId, selectRightRailTab } from '@/store/layout'
import { $previewTabs, closeRightRail, openPreview, type PreviewTarget } from '@/store/preview'

import { watchPreviewTiles } from '../preview-tile'

import { PREVIEW_READ_MAX_CHARS, readActivePreview, registerPreviewPageReader } from './preview-reader'

function urlTarget(url: string): PreviewTarget {
  return { kind: 'url', label: 'Browser', source: url, url }
}

function fileTarget(path: string): PreviewTarget {
  return { kind: 'file', label: path, path, previewKind: 'text', source: path, url: `file://${path}` }
}

describe('readActivePreview (read_preview tool)', () => {
  // All URL targets share the singleton Browser tab id, so a reader registered
  // in one test would answer the next — unregister whatever a test installed.
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
    noteActiveTreeGroup(null)
    noteHoveredTreeGroup(null)
    $layoutTree.set(null)
  })

  it('answers null when nothing is open, so the tool reports it cleanly', async () => {
    expect(await readActivePreview()).toBeNull()
  })

  it('serializes the Browser tab through its registered page reader', async () => {
    openPreview(urlTarget('https://news.ycombinator.com'))
    register($rightRailActiveTabId.get()!, async () => ({
      text: 'Top stories…',
      title: 'Hacker News',
      url: 'https://news.ycombinator.com/news'
    }))

    expect(await readActivePreview()).toMatchObject({
      kind: 'url',
      text: 'Top stories…',
      title: 'Hacker News',
      total_chars: 12,
      // The live address wins over the target (in-page navigation).
      url: 'https://news.ycombinator.com/news'
    })
  })

  it('windows long pages with start/count and reports the full length', async () => {
    openPreview(urlTarget('https://example.com'))
    register($rightRailActiveTabId.get()!, async () => ({
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
    openPreview(urlTarget('https://example.com'))
    register($rightRailActiveTabId.get()!, async () => ({
      text: 'x'.repeat(PREVIEW_READ_MAX_CHARS + 5000),
      title: 't',
      url: ''
    }))

    const result = await readActivePreview({ count: PREVIEW_READ_MAX_CHARS + 5000 })

    expect(result?.text).toHaveLength(PREVIEW_READ_MAX_CHARS)
    expect(result?.total_chars).toBe(PREVIEW_READ_MAX_CHARS + 5000)
  })

  it('answers identity + retry note for a Browser tab whose pane is not mounted', async () => {
    openPreview(urlTarget('https://example.com'))

    expect(await readActivePreview()).toMatchObject({
      kind: 'url',
      note: expect.stringContaining('retry') as string,
      text: '',
      url: 'https://example.com'
    })
  })

  it('answers a file tab with its identity and points at read_file', async () => {
    openPreview(fileTarget('/work/notes.md'))

    expect(await readActivePreview()).toMatchObject({
      kind: 'file',
      note: expect.stringContaining('read_file') as string,
      path: '/work/notes.md'
    })
  })

  it('reads the tab the user is LOOKING at, not the last one opened', async () => {
    openPreview(fileTarget('/work/one.md'))
    openPreview(fileTarget('/work/two.md'))
    selectRightRailTab('file:file:///work/one.md')

    expect(await readActivePreview()).toMatchObject({ path: '/work/one.md' })
  })

  it('falls back to the identity answer when the reader throws (webview booting)', async () => {
    openPreview(urlTarget('https://example.com'))
    register($rightRailActiveTabId.get()!, async () => {
      throw new Error('webview gone')
    })

    expect(await readActivePreview()).toMatchObject({ note: expect.stringContaining('retry') as string, text: '' })
  })

  it('unregister is idempotent and scoped to the same reader', async () => {
    openPreview(urlTarget('https://example.com'))
    const tabId = $rightRailActiveTabId.get()!
    const first = register(tabId, async () => ({ text: 'first', title: '', url: '' }))

    register(tabId, async () => ({ text: 'second', title: '', url: '' }))
    // Unregistering the STALE reader must not evict the live one.
    first()

    expect(await readActivePreview()).toMatchObject({ text: 'second' })
  })

  it('reads the hovered preview zone instead of the global right-rail tab', async () => {
    openPreview(fileTarget('/work/a.md'))
    const fileId = $rightRailActiveTabId.get()!
    openPreview(urlTarget('https://example.com/tickets'))
    const browserId = $rightRailActiveTabId.get()!
    selectRightRailTab(fileId)
    mountSplit(fileId, browserId)
    noteHoveredTreeGroup('grp-browser')

    expect(await readActivePreview()).toMatchObject({ kind: 'url', url: 'https://example.com/tickets' })
  })

  it('reads the focused preview zone instead of a stale global file tab', async () => {
    openPreview(fileTarget('/work/a.md'))
    const fileId = $rightRailActiveTabId.get()!
    openPreview(urlTarget('https://example.com/tickets'))
    const browserId = $rightRailActiveTabId.get()!
    selectRightRailTab(fileId)
    mountSplit(fileId, browserId)
    noteActiveTreeGroup('grp-browser')

    expect(await readActivePreview()).toMatchObject({ kind: 'url', url: 'https://example.com/tickets' })
  })

  it('reads the hovered file when that zone is what the user is looking at', async () => {
    openPreview(fileTarget('/work/a.md'))
    const fileId = $rightRailActiveTabId.get()!
    openPreview(urlTarget('https://example.com/tickets'))
    const browserId = $rightRailActiveTabId.get()!
    mountSplit(fileId, browserId)
    noteHoveredTreeGroup('grp-file')

    expect(await readActivePreview()).toMatchObject({ kind: 'file', path: '/work/a.md' })
  })

  it('returns active_tab_id and the open tab list when more than one preview is mounted', async () => {
    openPreview(fileTarget('/work/project-network.html'))
    const fileId = $rightRailActiveTabId.get()!
    openPreview(urlTarget('https://example.com/tickets'))
    const browserId = $rightRailActiveTabId.get()!

    expect(await readActivePreview()).toMatchObject({
      active_tab_id: browserId,
      kind: 'url',
      tabs: [
        { id: fileId, kind: 'file', label: '/work/project-network.html', url: 'file:///work/project-network.html' },
        { id: browserId, kind: 'url', label: 'Browser', url: 'https://example.com/tickets' }
      ],
      url: 'https://example.com/tickets'
    })
    expect($previewTabs.get()).toHaveLength(2)
  })
})

describe('follow() does not overwrite an explicit open in another group', () => {
  beforeEach(() => {
    closeRightRail()
    window.localStorage.clear()
    noteActiveTreeGroup(null)
    noteHoveredTreeGroup(null)
    $layoutTree.set(null)
  })

  it('keeps the opened URL when the other group is still the interacted zone', async () => {
    watchPreviewTiles()
    openPreview(fileTarget('/work/project-network.html'))
    const fileId = $rightRailActiveTabId.get()!
    openPreview(urlTarget('about:blank'))
    const browserId = $rightRailActiveTabId.get()!
    mountSplit(fileId, browserId)
    noteActiveTreeGroup('grp-file')

    openPreview(urlTarget('https://example.com/tickets'))
    // reveal may not commit when the pane is already fronted; the layout
    // listener is what copies the interacted zone. Fire that same listener.
    $layoutTree.set(mountSplit(fileId, browserId))

    expect($rightRailActiveTabId.get()).toBe(browserId)
    expect(await readActivePreview()).toMatchObject({
      active_tab_id: browserId,
      kind: 'url',
      url: 'https://example.com/tickets'
    })
  })
})

function mountSplit(fileId: string, browserId: string) {
  const tree = split('row', [
    group([`preview-tile:${browserId}`], { active: `preview-tile:${browserId}`, id: 'grp-browser' }),
    group([`preview-tile:${fileId}`], { active: `preview-tile:${fileId}`, id: 'grp-file' })
  ])

  $layoutTree.set(tree)

  return tree
}
