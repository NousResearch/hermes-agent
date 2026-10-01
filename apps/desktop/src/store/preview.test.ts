import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'

import { $rightRailActiveTabId, selectRightRailTab } from './layout'
import {
  $browserPages,
  $previewServerRestart,
  $previewServerRestartStatus,
  $previewTabs,
  $previewTarget,
  $visiblePreviewTabs,
  adoptDraftPreviewTabs,
  beginPreviewServerRestart,
  closeAgentPreview,
  closeBrowserPreviewMatchingLiveUrl,
  closePreviewForSource,
  closePreviewMatching,
  closeRightRail,
  closeRightRailTab,
  commitBrowserTabLocation,
  decodePreviewTabs,
  markPreviewTabMissing,
  newBrowserTab,
  noteBrowserPage,
  openPreview,
  previewTabId,
  type PreviewTarget,
  progressPreviewServerRestart,
  prunePreviewTabsForSession,
  rekeyPreviewTabsSession,
  renderedHtmlTarget,
  setPreviewRenderMode,
  setPreviewTabPinned
} from './preview'
import { $selectedStoredSessionId } from './session'

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
    $browserPages.set({})
    $previewServerRestart.set(null)
    $selectedStoredSessionId.set(null)
    closeRightRail()
    window.localStorage.clear()
  })

  afterEach(() => {
    $browserPages.set({})
    $previewServerRestart.set(null)
    $selectedStoredSessionId.set(null)
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
    openPreview(fileTarget('/work/demo.html'))

    expect($rightRailActiveTabId.get()).toBe('file:file:///work/demo.html')
    expect($previewTarget.get()?.path).toBe('/work/demo.html')
  })

  it('gives every kind of target its own tab, side by side', () => {
    openPreview(fileTarget('/work/demo.html'))
    openPreview(urlTarget('http://localhost:5174'))
    openPreview(artifactTarget('session-1:dashboard'))

    expect($previewTabs.get().map(tab => tab.target.kind)).toEqual(['file', 'url', 'artifact'])
  })

  // A Browser tab is a VESSEL, so a link hands its page to the browser you are
  // already looking at. New tabs are something you ask for (`newBrowserTab`) —
  // otherwise an agent opening five pages leaves five Browsers behind.
  it('navigates the open Browser rather than stacking a second one', () => {
    openPreview(urlTarget('https://news.ycombinator.com'))
    openPreview(urlTarget('https://www.reddit.com'))

    const urlTabs = $previewTabs.get().filter(tab => tab.target.kind === 'url')

    expect(urlTabs).toHaveLength(1)
    expect(urlTabs[0].target.url).toBe('https://www.reddit.com')
    expect($rightRailActiveTabId.get()).toBe(urlTabs[0].id)
  })

  it('commits the live page onto a Browser tab without changing its id', () => {
    openPreview(urlTarget('https://news.ycombinator.com'))
    const id = $previewTabs.get()[0].id

    commitBrowserTabLocation(id, 'https://news.ycombinator.com/item?id=1', 'Item')

    expect($previewTabs.get()).toHaveLength(1)
    expect($previewTabs.get()[0].id).toBe(id)
    expect($previewTabs.get()[0].target.url).toBe('https://news.ycombinator.com/item?id=1')
    expect($previewTabs.get()[0].target.label).toBe('Item')
  })

  it('opens more than one Browser on request, each holding its own page', () => {
    openPreview(urlTarget('https://news.ycombinator.com'))
    newBrowserTab()
    openPreview(urlTarget('https://www.reddit.com'))

    const urlTabs = $previewTabs.get().filter(tab => tab.target.kind === 'url')

    expect(urlTabs.map(tab => tab.target.url)).toEqual(['https://news.ycombinator.com', 'https://www.reddit.com'])
    expect(new Set(urlTabs.map(tab => tab.id)).size).toBe(2)
  })

  // Which Browser a link lands in: the one on screen. Selecting the older tab
  // must send the next page there, not to whichever was opened most recently.
  it('navigates the Browser you are looking at', () => {
    openPreview(urlTarget('https://news.ycombinator.com'))
    const first = $previewTabs.get()[0].id

    newBrowserTab()
    selectRightRailTab(first)
    openPreview(urlTarget('https://www.reddit.com'))

    expect($previewTabs.get().find(tab => tab.id === first)?.target.url).toBe('https://www.reddit.com')
    expect($previewTabs.get()).toHaveLength(2)
  })

  // A Browser id is minted rather than derived, so it must never be handed out
  // twice: per-tab state keyed by it would resurface under an unrelated tab.
  it('never reuses a Browser id, even after one is closed', () => {
    newBrowserTab()
    const first = $previewTabs.get()[0].id

    newBrowserTab()
    closeRightRailTab(first)
    newBrowserTab()

    const ids = $previewTabs.get().map(tab => tab.id)

    expect(ids).not.toContain(first)
    expect(new Set(ids).size).toBe(ids.length)
  })

  it('re-fronts an existing tab instead of duplicating it, refreshing its target', () => {
    openPreview({ ...fileTarget('/work/demo.html'), label: 'old' })
    openPreview({ ...fileTarget('/work/demo.html'), label: 'new' })

    expect($previewTabs.get()).toHaveLength(1)
    expect($previewTarget.get()?.label).toBe('new')
  })

  // Local HTML files default to a live Render, whether opened from the file
  // browser or handed over by a tool. Source is an explicit fallback only.
  it('renders browsed html and handed-over html live', () => {
    openPreview(fileTarget('/work/browsed.html'))
    expect($previewTarget.get()?.renderMode).toBe('preview')

    openPreview(fileTarget('/work/handed.html'))
    expect($previewTarget.get()?.renderMode).toBe('preview')

    openPreview(fileTarget('/work/manual.html'))
    expect($previewTarget.get()?.renderMode).toBe('preview')
  })

  it('preserves an explicit HTML source fallback from the file browser', () => {
    openPreview({ ...fileTarget('/work/fallback.html'), renderMode: 'source' })

    expect($previewTarget.get()?.renderMode).toBe('source')
  })

  it('switches render mode on the same tab without duplicating it', () => {
    openPreview(fileTarget('/work/toggle.html'))

    const tabId = previewTabId(fileTarget('/work/toggle.html'))

    expect($previewTabs.get()).toHaveLength(1)
    expect($previewTarget.get()?.renderMode).toBe('preview')

    setPreviewRenderMode(tabId, 'source')

    expect($previewTabs.get()).toHaveLength(1)
    expect($previewTabs.get()[0]?.id).toBe(tabId)
    expect($previewTarget.get()?.renderMode).toBe('source')

    setPreviewRenderMode(tabId, 'preview')

    expect($previewTabs.get()).toHaveLength(1)
    expect($previewTabs.get()[0]?.id).toBe(tabId)
    expect($previewTarget.get()?.renderMode).toBe('preview')
  })

  it('keeps a tab in Source when the same file is opened again', () => {
    const target = fileTarget('/work/again.html')

    openPreview(target)
    setPreviewRenderMode(previewTabId(target), 'source')
    openPreview({ ...target, label: 'again.html (renamed)' })

    expect($previewTabs.get()).toHaveLength(1)
    expect($previewTarget.get()?.label).toBe('again.html (renamed)')
    expect($previewTarget.get()?.renderMode).toBe('source')

    openPreview({ ...target, renderMode: 'preview' })

    expect($previewTarget.get()?.renderMode).toBe('preview')
  })

  it('renders an agent hand-over of an HTML file even when its tab sits in Source', () => {
    const target = fileTarget('/work/handed.html')

    openPreview(target)
    setPreviewRenderMode(previewTabId(target), 'source')
    openPreview(renderedHtmlTarget(target))

    expect($previewTabs.get()).toHaveLength(1)
    expect($previewTarget.get()?.renderMode).toBe('preview')

    // An explicit mode and non-HTML targets pass through untouched.
    expect(renderedHtmlTarget({ ...target, renderMode: 'source' }).renderMode).toBe('source')
    expect(renderedHtmlTarget({ ...fileTarget('/work/notes.md'), previewKind: 'text' }).renderMode).toBeUndefined()
  })

  it('falls back to a neighbouring tab when the active one closes, and clears the selection on the last', () => {
    openPreview(fileTarget('/work/one.html'))
    openPreview(fileTarget('/work/two.html'))

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
    openPreview(urlTarget('http://localhost:5174'))

    expect(closePreviewForSource('http://localhost:5174')).toBe(true)
    expect($previewTabs.get()).toHaveLength(0)
    expect(closePreviewForSource('http://localhost:5174')).toBe(false)
  })

  it('closes a tab whose url or label matches even when source differs', () => {
    openPreview({
      kind: 'url',
      label: 'HN',
      source: 'https://news.ycombinator.com',
      url: 'https://news.ycombinator.com/'
    })

    expect(closePreviewMatching('https://news.ycombinator.com/')).toBe(true)
    expect($previewTabs.get()).toHaveLength(0)

    openPreview({ ...fileTarget('/work/demo.html'), label: 'Demo' })

    expect(closePreviewMatching('Demo')).toBe(true)
    expect($previewTabs.get()).toHaveLength(0)
  })

  it('closes a Browser tab by its live navigated URL and removes it from persistence', () => {
    openPreview(urlTarget('https://example.com'))
    const tabId = $previewTabs.get()[0].id

    noteBrowserPage(tabId, {
      title: 'Dashboard',
      url: 'https://example.com/dashboard'
    })

    expect(closeBrowserPreviewMatchingLiveUrl('https://example.com/dashboard')).toBe(true)
    expect($previewTabs.get()).toHaveLength(0)
    expect($browserPages.get()[tabId]).toBeUndefined()
    expect(window.localStorage.getItem('hermes.desktop.previewTabs.v2')).toBeNull()
  })

  it('prefers the tab currently showing a URL over one that navigated away from it', () => {
    openPreview(urlTarget('https://example.com'))
    const first = $previewTabs.get()[0].id
    noteBrowserPage(first, { title: 'Elsewhere', url: 'https://elsewhere.example/' })

    newBrowserTab()
    openPreview(urlTarget('https://example.com'))
    const second = $previewTabs.get()[1].id
    noteBrowserPage(second, { title: 'Example', url: 'https://example.com/' })

    expect(closeBrowserPreviewMatchingLiveUrl('https://example.com')).toBe(true)
    expect($previewTabs.get().map(tab => tab.id)).toEqual([first])
    expect($browserPages.get()[first]?.url).toBe('https://elsewhere.example/')
  })

  it('does not wipe the rail on an empty or unknown close query', () => {
    openPreview(fileTarget('/work/keep.html'))

    expect(closePreviewMatching()).toBe(false)
    expect(closePreviewMatching('   ')).toBe(false)
    expect(closePreviewMatching('https://missing.example')).toBe(false)
    expect($previewTabs.get()).toHaveLength(1)
  })

  it('persists file and url tabs but never artifacts, whose content is memory-only', () => {
    openPreview(fileTarget('/work/demo.html'))
    openPreview(urlTarget('http://localhost:5174'))
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

    // Nothing persistable, so the profile's bucket is empty and the key is
    // removed rather than stored as an empty list (matching the tiles store).
    expect(window.localStorage.getItem('hermes.desktop.previewTabs.v2')).toBeNull()
  })

  it('preserves an explicit HTML source fallback', () => {
    openPreview({ ...fileTarget('/remote/report.html'), renderMode: 'source' })

    expect($previewTarget.get()?.renderMode).toBe('source')
  })

  it('does not persist transient remote HTML source fallbacks', () => {
    const target = { ...fileTarget('/remote/report.html'), renderMode: 'source' as const, transient: true }

    openPreview(target)

    // Nothing persistable, so the profile's bucket is empty and the key is
    // removed rather than stored as an empty list (matching the tiles store).
    expect(window.localStorage.getItem('hermes.desktop.previewTabs.v2')).toBeNull()
  })

  it('tombstones a confirmed-missing tab in place without closing it', () => {
    openPreview(fileTarget('/work/demo.html'))
    openPreview(fileTarget('/work/keep.html'))
    const demoId = 'file:file:///work/demo.html'

    markPreviewTabMissing(demoId)

    const tabs = $previewTabs.get()

    expect(tabs).toHaveLength(2)
    expect(tabs.find(tab => tab.id === demoId)?.target.missing).toBe(true)
    expect(tabs.find(tab => tab.id !== demoId)?.target.missing).toBeFalsy()

    // Idempotent: a second tombstone must not rewrite the list.
    const before = JSON.stringify($previewTabs.get())
    markPreviewTabMissing(demoId)
    expect(JSON.stringify($previewTabs.get())).toBe(before)
  })

  it('ignores a tombstone for a tab that is not open', () => {
    markPreviewTabMissing('file:file:///nowhere.html')

    expect($previewTabs.get()).toHaveLength(0)
  })

  it('drops tombstoned file tabs at restore so dead paths are not re-probed next boot', () => {
    const raw = JSON.stringify([
      { id: 'file:file:///work/gone.html', target: { ...fileTarget('/work/gone.html'), missing: true } },
      { id: 'file:file:///work/alive.html', target: fileTarget('/work/alive.html') }
    ])

    const restored = decodePreviewTabs(raw)

    expect(restored.map(tab => tab.target.path)).toEqual(['/work/alive.html'])
  })
})

describe('preview session scoping (#73890)', () => {
  afterEach(() => {
    $selectedStoredSessionId.set(null)
    closeRightRail()
    window.localStorage.clear()
  })

  const paths = (tabs: readonly { target: PreviewTarget }[]) => tabs.map(tab => tab.target.path ?? tab.target.url)

  it('stamps the active session on tabs it opens', () => {
    $selectedStoredSessionId.set('sess-1')
    openPreview(fileTarget('/work/demo.html'))

    expect($previewTabs.get()[0]?.sessionId).toBe('sess-1')
  })

  it('shows only the focused session tabs plus pinned ones, keeping hidden tabs alive', () => {
    $selectedStoredSessionId.set('sess-1')
    openPreview(fileTarget('/work/a.html'))
    $selectedStoredSessionId.set('sess-2')
    openPreview(fileTarget('/work/b.html'))

    expect(paths($visiblePreviewTabs.get())).toEqual(['/work/b.html'])
    expect($previewTabs.get()).toHaveLength(2)

    $selectedStoredSessionId.set('sess-1')
    expect(paths($visiblePreviewTabs.get())).toEqual(['/work/a.html'])

    const aTab = $previewTabs.get().find(tab => tab.target.path === '/work/a.html')!
    setPreviewTabPinned(aTab.id, true)
    $selectedStoredSessionId.set('sess-2')
    expect(paths($visiblePreviewTabs.get()).sort()).toEqual(['/work/a.html', '/work/b.html'])

    setPreviewTabPinned(aTab.id, false)
    expect(paths($visiblePreviewTabs.get())).toEqual(['/work/b.html'])
  })

  it('opens the same file as one tab per session, and re-fronts it within one', () => {
    $selectedStoredSessionId.set('sess-1')
    openPreview(fileTarget('/work/demo.html'))
    openPreview({ ...fileTarget('/work/demo.html'), url: '/work/demo.html', label: 'refreshed' })
    $selectedStoredSessionId.set('sess-2')
    openPreview(fileTarget('/work/demo.html'))

    const tabs = $previewTabs.get()

    expect(tabs.map(tab => tab.sessionId)).toEqual(['sess-1', 'sess-2'])
    expect(new Set(tabs.map(tab => tab.id)).size).toBe(2)
    expect(tabs[0]?.target.label).toBe('refreshed')
  })

  it('reuses a pinned file from another session without changing its owner', () => {
    $selectedStoredSessionId.set('sess-1')
    openPreview(fileTarget('/work/a.html'))
    const id = $previewTabs.get()[0]!.id
    setPreviewTabPinned(id, true)

    $selectedStoredSessionId.set('sess-2')
    openPreview({ ...fileTarget('/work/a.html'), label: 'refreshed' })

    expect($previewTabs.get()).toHaveLength(1)
    expect($previewTabs.get()[0]).toMatchObject({ id, pinned: true, sessionId: 'sess-1' })
    expect($rightRailActiveTabId.get()).toBe(id)
  })

  it('prunes a deleted session tabs but keeps its pins', () => {
    $selectedStoredSessionId.set('sess-1')
    openPreview(fileTarget('/work/a.html'))
    setPreviewTabPinned($previewTabs.get()[0]!.id, true)
    $selectedStoredSessionId.set('sess-2')
    openPreview(fileTarget('/work/b.html'))

    prunePreviewTabsForSession('sess-2')
    expect(paths($previewTabs.get())).toEqual(['/work/a.html'])

    prunePreviewTabsForSession('sess-1')
    expect(paths($previewTabs.get())).toEqual(['/work/a.html'])
  })

  it('keeps a draft tabs in the draft until that draft itself gets its stored id', () => {
    openPreview(fileTarget('/work/draft.html'))
    newBrowserTab()
    expect($previewTabs.get().map(tab => tab.sessionId)).toEqual([undefined, undefined])

    // Focus moving elsewhere — an existing session clicked in the sidebar, or
    // a side tile — must not take the draft's tabs with it.
    $selectedStoredSessionId.set('existing-x')
    expect($visiblePreviewTabs.get()).toHaveLength(0)

    const previousTree = $layoutTree.get()

    try {
      $layoutTree.set(group(['workspace', 'session-tile:tile-t'], { active: 'session-tile:tile-t', id: 'grp-draft' }))
      noteActiveTreeGroup('grp-draft')
      expect($visiblePreviewTabs.get()).toHaveLength(0)
    } finally {
      noteActiveTreeGroup(null)
      $layoutTree.set(previousTree)
    }

    $selectedStoredSessionId.set(null)
    expect(paths($visiblePreviewTabs.get())).toEqual(['/work/draft.html', 'about:blank'])

    // The draft's first send assigns its stored id: now the tabs are its own.
    adoptDraftPreviewTabs('sess-new')
    $selectedStoredSessionId.set('sess-new')

    expect($previewTabs.get().map(tab => tab.sessionId)).toEqual(['sess-new', 'sess-new'])
    expect(paths($visiblePreviewTabs.get())).toEqual(['/work/draft.html', 'about:blank'])
  })

  it('owns a strip "+" Browser by the focused session, never adopting it elsewhere', () => {
    $selectedStoredSessionId.set('sess-x')
    newBrowserTab()
    const id = $previewTabs.get()[0]!.id

    expect($previewTabs.get()[0]?.sessionId).toBe('sess-x')
    expect($visiblePreviewTabs.get().map(tab => tab.id)).toEqual([id])

    $selectedStoredSessionId.set('sess-y')
    expect($visiblePreviewTabs.get()).toHaveLength(0)
    expect($previewTabs.get()[0]?.sessionId).toBe('sess-x')
  })

  it('never navigates another session Browser for a URL open', () => {
    $selectedStoredSessionId.set('sess-a')
    openPreview(urlTarget('https://a.example'))
    const aBrowser = $previewTabs.get()[0]!

    $selectedStoredSessionId.set('sess-b')
    openPreview(urlTarget('https://b.example'))

    expect($previewTabs.get()).toHaveLength(2)
    expect($previewTabs.get().find(tab => tab.id === aBrowser.id)).toMatchObject({
      sessionId: 'sess-a',
      target: { url: 'https://a.example' }
    })

    $selectedStoredSessionId.set('sess-a')
    expect(paths($visiblePreviewTabs.get())).toEqual(['https://a.example'])
  })

  it('opens for an explicit owner without touching the focused drawer', () => {
    $selectedStoredSessionId.set('sess-focused')
    openPreview(fileTarget('/work/focused.html'))
    const focusedId = $rightRailActiveTabId.get()

    openPreview(fileTarget('/work/tile.html'), 'sess-tile')

    expect($previewTabs.get().find(tab => tab.target.path === '/work/tile.html')?.sessionId).toBe('sess-tile')
    expect(paths($visiblePreviewTabs.get())).toEqual(['/work/focused.html'])
    expect($rightRailActiveTabId.get()).toBe(focusedId)
  })

  it('agent close without a url closes only that session own tabs', () => {
    $selectedStoredSessionId.set('sess-a')
    openPreview(fileTarget('/work/pinned.html'))
    setPreviewTabPinned($previewTabs.get()[0]!.id, true)
    openPreview(fileTarget('/work/a.html'))
    $selectedStoredSessionId.set('sess-b')
    openPreview(fileTarget('/work/b.html'))

    closeAgentPreview('sess-a', [])

    expect(paths($previewTabs.get()).sort()).toEqual(['/work/b.html', '/work/pinned.html'])
  })

  it('agent close by url reaches only the tabs that session can see', () => {
    $selectedStoredSessionId.set('sess-a')
    openPreview(fileTarget('/work/same.html'))
    $selectedStoredSessionId.set('sess-b')
    openPreview(fileTarget('/work/same.html'))

    closeAgentPreview('sess-a', ['/work/same.html'])

    expect($previewTabs.get().map(tab => tab.sessionId)).toEqual(['sess-b'])
  })

  it('keeps a conversation tabs visible across a compression tip rotation', () => {
    $selectedStoredSessionId.set('tip-1')
    openPreview(fileTarget('/work/a.html'))
    newBrowserTab()

    rekeyPreviewTabsSession('tip-1', 'tip-2')

    // Focus still names the old tip until the route follows; the tabs must
    // not blink out in between (a hidden Browser loses its page).
    expect($visiblePreviewTabs.get()).toHaveLength(2)

    $selectedStoredSessionId.set('tip-2')
    expect($visiblePreviewTabs.get()).toHaveLength(2)
    expect($previewTabs.get().map(tab => tab.sessionId)).toEqual(['tip-2', 'tip-2'])
  })

  it('survives a rotation that points back at an earlier tip', () => {
    $selectedStoredSessionId.set('tip-a')
    openPreview(fileTarget('/work/a.html'))

    rekeyPreviewTabsSession('tip-a', 'tip-b')
    rekeyPreviewTabsSession('tip-b', 'tip-c')
    rekeyPreviewTabsSession('tip-c', 'tip-a')
    rekeyPreviewTabsSession('tip-b', 'tip-a')

    for (const id of ['tip-a', 'tip-b', 'tip-c']) {
      $selectedStoredSessionId.set(id)
      expect(paths($visiblePreviewTabs.get())).toEqual(['/work/a.html'])
    }
  })

  it('stamps a tab opened under a rotated-away id with the new tip, so it outlives a relaunch', async () => {
    $selectedStoredSessionId.set('tip-1')
    rekeyPreviewTabsSession('tip-1', 'tip-2')
    // Focus has not followed the rotation yet.
    openPreview(fileTarget('/work/in-window.html'))
    newBrowserTab()

    expect($previewTabs.get().map(tab => tab.sessionId)).toEqual(['tip-2', 'tip-2'])

    const relaunched = await relaunchedPreviewStore()
    const { $selectedStoredSessionId: selected } = await import('./session')

    selected.set('tip-2')
    expect(relaunched.$visiblePreviewTabs.get()).toHaveLength(2)
    selected.set(null)
  })

  it('prunes a deleted conversation by any of its ids', () => {
    $selectedStoredSessionId.set('tip-1')
    openPreview(fileTarget('/work/a.html'))
    rekeyPreviewTabsSession('tip-1', 'tip-2')
    openPreview(fileTarget('/work/b.html'))

    prunePreviewTabsForSession('tip-1')

    expect($previewTabs.get()).toHaveLength(0)
  })

  it('agent close without a url writes the tab list once and fronts what the session still sees', () => {
    $selectedStoredSessionId.set('sess-a')
    openPreview(fileTarget('/work/pinned.html'))
    const pinned = $previewTabs.get()[0]!.id
    setPreviewTabPinned(pinned, true)

    for (let i = 0; i < 5; i++) {
      openPreview(fileTarget(`/work/a${i}.html`))
    }

    let writes = 0
    const unbind = $previewTabs.listen(() => writes++)

    try {
      closeAgentPreview('sess-a', [])
    } finally {
      unbind()
    }

    expect(writes).toBe(1)
    expect(paths($previewTabs.get())).toEqual(['/work/pinned.html'])
    expect($rightRailActiveTabId.get()).toBe(pinned)
  })

  it('tombstones a missing file by the url the file pane reports', () => {
    $selectedStoredSessionId.set('sess-1')
    openPreview(fileTarget('/work/gone.html'))

    markPreviewTabMissing(fileTarget('/work/gone.html').url)

    expect($previewTabs.get()[0]?.target.missing).toBe(true)
  })

  // A relaunch (or a fresh pop-out renderer) reads storage at module load.
  async function relaunchedPreviewStore() {
    vi.resetModules()

    return import('./preview')
  }

  it('restores every Browser tab of every session and every pin on relaunch', async () => {
    $selectedStoredSessionId.set('sess-a')
    openPreview(urlTarget('https://a.example'))
    newBrowserTab()
    openPreview(urlTarget('https://a2.example'))
    $selectedStoredSessionId.set('sess-b')
    openPreview(urlTarget('https://b.example'))
    setPreviewTabPinned($previewTabs.get()[2]!.id, true)
    openPreview(fileTarget('/work/b.html'))

    const live = $previewTabs.get().map(tab => [tab.id, tab.sessionId, Boolean(tab.pinned)])
    const relaunched = await relaunchedPreviewStore()

    expect(relaunched.$previewTabs.get().map(tab => [tab.id, tab.sessionId, Boolean(tab.pinned)])).toEqual(live)
  })

  it('pops out a Browser tab that is not the last one persisted', async () => {
    $selectedStoredSessionId.set('sess-a')
    openPreview(urlTarget('https://first.example'))
    const first = $previewTabs.get()[0]!.id
    newBrowserTab()
    openPreview(urlTarget('https://second.example'))

    // A fresh pop-out renderer: no session focus, its view on the default
    // bucket, the tab in another profile's bucket — found by id in storage.
    window.localStorage.setItem('hermes.desktop.previewTabs.v2', JSON.stringify({ work: $previewTabs.get() }))
    const fresh = await relaunchedPreviewStore()

    expect(fresh.$previewTabs.get()).toHaveLength(0)
    fresh.adoptPersistedBrowserTab(first)

    expect(fresh.$previewTabs.get().find(tab => tab.id === first)?.target.url).toBe('https://first.example')
  })

  it('follows the focused session tile over the sidebar selection', () => {
    $selectedStoredSessionId.set('sess-1')
    openPreview(fileTarget('/work/a.html'))
    $selectedStoredSessionId.set('sess-2')
    openPreview(fileTarget('/work/b.html'))

    const previousTree = $layoutTree.get()

    try {
      $layoutTree.set(group(['session-tile:sess-1'], { active: 'session-tile:sess-1', id: 'grp-main' }))
      noteActiveTreeGroup('grp-main')

      expect(paths($visiblePreviewTabs.get())).toEqual(['/work/a.html'])

      noteActiveTreeGroup(null)
      expect(paths($visiblePreviewTabs.get())).toEqual(['/work/b.html'])
    } finally {
      noteActiveTreeGroup(null)
      $layoutTree.set(previousTree)
    }
  })

  it('migrates legacy unscoped tabs to pinned and keeps their ids', () => {
    const raw = JSON.stringify([
      { id: 'file:file:///work/a.html', target: fileTarget('/work/a.html') },
      { id: 'file:file:///work/b.html', target: fileTarget('/work/b.html'), sessionId: 'sess-9' },
      { id: 'file:file:///work/c.html', target: fileTarget('/work/c.html'), sessionId: 'sess-1', pinned: false }
    ])

    const decoded = decodePreviewTabs(raw)

    expect(decoded.map(tab => [tab.id, tab.sessionId, tab.pinned])).toEqual([
      ['file:file:///work/a.html', undefined, true],
      ['file:file:///work/b.html', 'sess-9', undefined],
      ['file:file:///work/c.html', 'sess-1', false]
    ])
  })
})
