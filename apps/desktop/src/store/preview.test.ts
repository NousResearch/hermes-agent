import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $rightRailActiveTabId } from './layout'
import {
  $previewServerRestart,
  $previewServerRestartStatus,
  $previewTabs,
  $previewTarget,
  beginPreviewServerRestart,
  closePreviewForSource,
  closePreviewMatching,
  closeRightRail,
  closeRightRailTab,
  openPreview,
  previewTabId,
  type PreviewTarget,
  progressPreviewServerRestart
} from './preview'

function fileTarget(source: string): PreviewTarget {
  return { kind: 'file', label: source, path: source, previewKind: 'html', source, url: `file://${source}` }
}

function urlTarget(source: string): PreviewTarget {
  return { kind: 'url', label: source, source, url: source }
}

function artifactTarget(id: string): PreviewTarget {
  return { kind: 'artifact', label: id, source: id, url: id }
}

describe('preview store', () => {
  beforeEach(() => {
    $previewServerRestart.set(null)
    closeRightRail()
    window.localStorage.clear()
  })

  afterEach(() => {
    $previewServerRestart.set(null)
    closeRightRail()
    window.localStorage.clear()
  })

  it('does not notify status subscribers for restart progress text', () => {
    const statuses: string[] = []
    const unsubscribe = $previewServerRestartStatus.subscribe(status => statuses.push(status))

    beginPreviewServerRestart('task-1', 'http://localhost:5174')
    progressPreviewServerRestart('task-1', 'first line')
    progressPreviewServerRestart('task-1', 'second line')
    unsubscribe()

    expect(statuses).toEqual(['idle', 'running'])
  })

  it('opens the pane and fronts the new tab', () => {
    openPreview(fileTarget('/work/demo.html'), 'tool-result')

    expect($rightRailActiveTabId.get()).toBe('file:file:///work/demo.html')
    expect($previewTarget.get()?.path).toBe('/work/demo.html')
  })

  it('gives every kind of target its own tab, side by side', () => {
    openPreview(fileTarget('/work/demo.html'), 'file-browser')
    openPreview(urlTarget('http://localhost:5174'), 'tool-result')
    openPreview(artifactTarget('session-1:dashboard'))

    expect($previewTabs.get().map(tab => tab.target.kind)).toEqual(['file', 'url', 'artifact'])
  })

  // The Browser is a SINGLETON: the tab names the surface, not the page, so a
  // second URL navigates the browser it already has instead of stacking a
  // second Browser tab beside the first.
  it('keeps one Browser tab — a second url swaps its target instead of adding a tab', () => {
    openPreview(urlTarget('https://news.ycombinator.com'), 'tool-result')
    openPreview(urlTarget('https://www.reddit.com'), 'tool-result')

    const urlTabs = $previewTabs.get().filter(tab => tab.target.kind === 'url')

    expect(urlTabs).toHaveLength(1)
    expect(urlTabs[0].target.url).toBe('https://www.reddit.com')
    expect($rightRailActiveTabId.get()).toBe(urlTabs[0].id)
  })

  it('re-fronts an existing tab instead of duplicating it, refreshing its target', () => {
    openPreview({ ...fileTarget('/work/demo.html'), label: 'old' }, 'file-browser')
    openPreview({ ...fileTarget('/work/demo.html'), label: 'new' }, 'file-browser')

    expect($previewTabs.get()).toHaveLength(1)
    expect($previewTarget.get()?.label).toBe('new')
  })

  // Browsing to an HTML file means "let me read it"; a tool or link handing you
  // one means "run it". Same road, different render mode on the target.
  it('renders browsed html as source and handed-over html live', () => {
    openPreview(fileTarget('/work/browsed.html'), 'file-browser')
    expect($previewTarget.get()?.renderMode).toBe('source')

    openPreview(fileTarget('/work/handed.html'), 'tool-result')
    expect($previewTarget.get()?.renderMode).toBe('preview')
  })

  it('falls back to a neighbouring tab when the active one closes, and clears the selection on the last', () => {
    openPreview(fileTarget('/work/one.html'), 'file-browser')
    openPreview(fileTarget('/work/two.html'), 'file-browser')

    closeRightRailTab(previewTabId(fileTarget('/work/two.html')))

    expect($previewTarget.get()?.path).toBe('/work/one.html')

    closeRightRailTab(previewTabId(fileTarget('/work/one.html')))
    expect($previewTarget.get()).toBeNull()
    expect($rightRailActiveTabId.get()).toBeNull()
  })

  it('ignores a close for a tab that is not open, so the shortcut falls through', () => {
    closeRightRailTab('file:file:///nowhere.html')

    expect($previewTabs.get()).toHaveLength(0)
  })

  it('closes by the raw source the composer rows were handed', () => {
    openPreview(urlTarget('http://localhost:5174'), 'tool-result')

    expect(closePreviewForSource('http://localhost:5174')).toBe(true)
    expect($previewTabs.get()).toHaveLength(0)
    expect(closePreviewForSource('http://localhost:5174')).toBe(false)
  })

  it('closes a tab whose url or label matches even when source differs', () => {
    openPreview(
      { kind: 'url', label: 'HN', source: 'https://news.ycombinator.com', url: 'https://news.ycombinator.com/' },
      'tool-result'
    )

    expect(closePreviewMatching('https://news.ycombinator.com/')).toBe(true)
    expect($previewTabs.get()).toHaveLength(0)

    openPreview({ ...fileTarget('/work/demo.html'), label: 'Demo' }, 'tool-result')

    expect(closePreviewMatching('Demo')).toBe(true)
    expect($previewTabs.get()).toHaveLength(0)
  })

  it('does not wipe the rail on an empty or unknown close query', () => {
    openPreview(fileTarget('/work/keep.html'), 'file-browser')

    expect(closePreviewMatching()).toBe(false)
    expect(closePreviewMatching('   ')).toBe(false)
    expect(closePreviewMatching('https://missing.example')).toBe(false)
    expect($previewTabs.get()).toHaveLength(1)
  })

  it('persists file and url tabs but never artifacts, whose content is memory-only', () => {
    openPreview(fileTarget('/work/demo.html'), 'file-browser')
    openPreview(urlTarget('http://localhost:5174'), 'tool-result')
    openPreview(artifactTarget('session-1:dashboard'))

    const stored = window.localStorage.getItem('hermes.desktop.previewTabs.v2') ?? ''

    expect(stored).toContain('/work/demo.html')
    expect(stored).toContain('localhost:5174')
    expect(stored).not.toContain('dashboard')
  })

  it('strips inline image bytes rather than pushing megabytes into storage', () => {
    openPreview({ ...fileTarget('/work/shot.png'), dataUrl: 'data:image/png;base64,AAAA', previewKind: 'image' })

    expect(window.localStorage.getItem('hermes.desktop.previewTabs.v2') ?? '').not.toContain('base64')
  })

  it('does not persist remote HTML without its in-memory document', () => {
    openPreview({ ...fileTarget('/remote/report.html'), dataUrl: 'data:text/html;base64,PGgxPnJlbW90ZTwvaDE+' })

    expect(window.localStorage.getItem('hermes.desktop.previewTabs.v2')).toBe('[]')
  })

  it('preserves an explicit HTML source fallback', () => {
    openPreview({ ...fileTarget('/remote/report.html'), renderMode: 'source' }, 'tool-result')

    expect($previewTarget.get()?.renderMode).toBe('source')
  })

  it('does not persist transient remote HTML source fallbacks', () => {
    const target = { ...fileTarget('/remote/report.html'), renderMode: 'source' as const, transient: true }

    openPreview(target, 'tool-result')

    expect(window.localStorage.getItem('hermes.desktop.previewTabs.v2')).toBe('[]')
  })
})
