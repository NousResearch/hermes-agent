import { afterEach, beforeEach, describe, expect, it } from 'vitest'

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
  beginPreviewServerRestart,
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

    expect($rightRailActiveTabId.get()).toBe('file:/work/demo.html')
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

  // Every Watch mints a fresh `#ticket=`; the viewer is the same page, so the
  // new Watch must replace the old (ticketless, dead) tab instead of stacking.
  it('reuses an isolated viewer tab for the same location with a new ticket', () => {
    const viewer = (url: string): PreviewTarget => ({ ...urlTarget(url), browserContext: 'isolated', transient: true })

    openPreview(viewer('http://127.0.0.1:9876/viewer.html?realm=a#ticket=one'))
    openPreview(viewer('http://127.0.0.1:9876/viewer.html?realm=a#ticket=two'))
    openPreview(viewer('http://127.0.0.1:9876/viewer.html?realm=b#ticket=three'))

    expect($previewTabs.get().map(tab => tab.target.url)).toEqual([
      'http://127.0.0.1:9876/viewer.html?realm=a#ticket=two',
      'http://127.0.0.1:9876/viewer.html?realm=b#ticket=three'
    ])
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
    // File tab ids are session-scoped; read the real one instead of rebuilding it.
    const demoId = $previewTabs.get().find(tab => tab.target.url === fileTarget('/work/demo.html').url)!.id

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

describe('preview session scoping', () => {
  afterEach(() => {
    $selectedStoredSessionId.set(null)
    closeRightRail()
    window.localStorage.clear()
  })

  it('stamps the active session on tabs it opens', () => {
    $selectedStoredSessionId.set('sess-1')
    openPreview(fileTarget('/work/demo.html'))

    expect($previewTabs.get()[0]?.sessionId).toBe('sess-1')
  })

  it('shows only the active session tabs plus pinned ones', () => {
    $selectedStoredSessionId.set('sess-1')
    openPreview(fileTarget('/work/a.html'))
    $selectedStoredSessionId.set('sess-2')
    openPreview(fileTarget('/work/b.html'))

    expect($visiblePreviewTabs.get().map(tab => tab.target.path)).toEqual(['/work/b.html'])

    // Pinning makes the first session's tab visible in every session.
    const aTab = $previewTabs.get().find(tab => tab.target.path === '/work/a.html')
    setPreviewTabPinned(aTab!.id, true)

    expect(
      $visiblePreviewTabs
        .get()
        .map(tab => tab.target.path)
        .sort()
    ).toEqual(['/work/a.html', '/work/b.html'])

    // Unpinning hides it from the other session again.
    setPreviewTabPinned(aTab!.id, false)
    expect($visiblePreviewTabs.get().map(tab => tab.target.path)).toEqual(['/work/b.html'])
  })

  it('prunes a deleted session tabs but keeps its pins', () => {
    $selectedStoredSessionId.set('sess-1')
    openPreview(fileTarget('/work/a.html'))
    setPreviewTabPinned($previewTabs.get()[0]!.id, true)
    $selectedStoredSessionId.set('sess-2')
    openPreview(fileTarget('/work/b.html'))

    prunePreviewTabsForSession('sess-2')
    expect($previewTabs.get().map(tab => tab.target.path)).toEqual(['/work/a.html'])

    // Pinned tabs belong to the workspace, not the session that opened them.
    prunePreviewTabsForSession('sess-1')
    expect($previewTabs.get().map(tab => tab.target.path)).toEqual(['/work/a.html'])
  })

  it('adopts ownerless draft tabs when a session appears', () => {
    openPreview(fileTarget('/work/draft.html'))
    expect($previewTabs.get()[0]?.sessionId).toBeUndefined()

    $selectedStoredSessionId.set('sess-new')
    expect($previewTabs.get()[0]?.sessionId).toBe('sess-new')
    // The id is rekeyed onto the session-scoped form so a later open of the
    // same file dedupes instead of stacking.
    expect($previewTabs.get()[0]?.id).toBe('file:sess-new:/work/draft.html')
    expect($visiblePreviewTabs.get().map(tab => tab.target.path)).toEqual(['/work/draft.html'])
  })

  it('canonicalizes file identities across entry points', () => {
    expect(previewTabId(fileTarget('/work/demo.html'))).toBe('file:/work/demo.html')

    const viaPlainPath = previewTabId({ ...fileTarget('/work/demo.html'), url: '/work/demo.html' })
    expect(viaPlainPath).toBe('file:/work/demo.html')
  })

  it('scopes file tab ids to the session, so two sessions can open the same file', () => {
    expect(previewTabId(fileTarget('/work/demo.html'), 'sess-1')).toBe('file:sess-1:/work/demo.html')
    expect(previewTabId(fileTarget('/work/demo.html'), 'sess-2')).toBe('file:sess-2:/work/demo.html')

    $selectedStoredSessionId.set('sess-1')
    openPreview(fileTarget('/work/demo.html'))
    $selectedStoredSessionId.set('sess-2')
    openPreview(fileTarget('/work/demo.html'))

    const tabs = $previewTabs.get()

    expect(tabs).toHaveLength(2)
    expect(tabs[0]).toMatchObject({ sessionId: 'sess-1', id: 'file:sess-1:/work/demo.html' })
    expect(tabs[1]).toMatchObject({ sessionId: 'sess-2', id: 'file:sess-2:/work/demo.html' })
    expect($visiblePreviewTabs.get()).toHaveLength(1)
  })

  it('re-opening the same file in the same session keeps the tab owner and pin', () => {
    $selectedStoredSessionId.set('sess-1')
    openPreview(fileTarget('/work/demo.html'))
    setPreviewTabPinned($previewTabs.get()[0]!.id, true)

    openPreview({ ...fileTarget('/work/demo.html'), label: 'refreshed' })

    expect($previewTabs.get()).toHaveLength(1)
    expect($previewTabs.get()[0]).toMatchObject({ sessionId: 'sess-1', pinned: true })
    expect($previewTabs.get()[0]?.target.label).toBe('refreshed')
  })

  it('unpinning an ownerless tab adopts the current session', () => {
    openPreview(fileTarget('/work/legacy.html'))
    setPreviewTabPinned($previewTabs.get()[0]!.id, true)

    $selectedStoredSessionId.set('sess-1')
    setPreviewTabPinned($previewTabs.get()[0]!.id, false)

    expect($previewTabs.get()[0]).toMatchObject({ pinned: false, sessionId: 'sess-1' })
    expect($previewTabs.get()[0]?.id).toBe('file:sess-1:/work/legacy.html')
  })

  it('closes by source only within the current session', () => {
    $selectedStoredSessionId.set('sess-1')
    openPreview(fileTarget('/work/a.html'))
    $selectedStoredSessionId.set('sess-2')
    openPreview(fileTarget('/work/a.html'))

    // The hidden session's tab must not be closed by the other session's row.
    expect(closePreviewForSource('/work/a.html')).toBe(true)
    expect($previewTabs.get()).toHaveLength(1)
    expect($previewTabs.get()[0]?.sessionId).toBe('sess-1')
  })

  it('reopens a pinned legacy row instead of stacking a second tab', () => {
    // A migrated legacy row: pinned, unprefixed id, no owner.
    $previewTabs.set([{ id: 'file:/work/a.html', target: fileTarget('/work/a.html'), pinned: true }])

    $selectedStoredSessionId.set('sess-1')
    openPreview(fileTarget('/work/a.html'))

    expect($previewTabs.get()).toHaveLength(1)
    expect($previewTabs.get()[0]).toMatchObject({ pinned: true, sessionId: 'sess-1' })
    expect($previewTabs.get()[0]?.id).toBe('file:sess-1:/work/a.html')
    expect($visiblePreviewTabs.get()).toHaveLength(1)
  })

  it('reusing an owned pinned row from another session keeps the owner id', () => {
    $selectedStoredSessionId.set('sess-1')
    openPreview(fileTarget('/work/a.html'))
    setPreviewTabPinned($previewTabs.get()[0]!.id, true)

    $selectedStoredSessionId.set('sess-2')
    openPreview({ ...fileTarget('/work/a.html'), label: 'refreshed' })

    expect($previewTabs.get()).toHaveLength(1)
    // The row keeps its OWNER's id — never an id/sessionId split (the id
    // would say sess-2 while the owner says sess-1 until the next unpin).
    expect($previewTabs.get()[0]).toMatchObject({
      id: 'file:sess-1:/work/a.html',
      sessionId: 'sess-1',
      pinned: true
    })
    expect($previewTabs.get()[0]?.target.label).toBe('refreshed')
    expect($rightRailActiveTabId.get()).toBe('file:sess-1:/work/a.html')
  })

  it('follows the focused session tile over the sidebar selection', () => {
    $selectedStoredSessionId.set('sess-1')
    openPreview(fileTarget('/work/a.html'))
    $selectedStoredSessionId.set('sess-2')
    openPreview(fileTarget('/work/b.html'))

    expect($visiblePreviewTabs.get().map(tab => tab.target.path)).toEqual(['/work/b.html'])

    // A sess-1 session TILE is active in the interacted zone while sess-2 is
    // selected in the sidebar: the drawer belongs to the conversation being
    // typed in, so the visible set follows the FOCUSED session.
    const previousTree = $layoutTree.get()

    try {
      $layoutTree.set(group(['session-tile:sess-1'], { active: 'session-tile:sess-1', id: 'grp-main' }))
      noteActiveTreeGroup('grp-main')

      expect($visiblePreviewTabs.get().map(tab => tab.target.path)).toEqual(['/work/a.html'])

      // Leaving the tile (workspace/route active) falls back to the selection.
      noteActiveTreeGroup(null)
      expect($visiblePreviewTabs.get().map(tab => tab.target.path)).toEqual(['/work/b.html'])
    } finally {
      // A failed assertion must not leak the tree/group state into the next
      // test — the focused derivation reads both.
      noteActiveTreeGroup(null)
      $layoutTree.set(previousTree)
    }
  })

  it('re-owns the singleton Browser to the session that navigates it', () => {
    $selectedStoredSessionId.set('sess-1')
    openPreview(urlTarget('http://localhost:5174'))

    // A second session navigating the Browser takes the surface with it:
    // keeping the original owner would leave the new URL invisible in the
    // session that just opened it.
    $selectedStoredSessionId.set('sess-2')
    openPreview(urlTarget('http://localhost:9999'))

    expect($previewTabs.get()).toHaveLength(1)
    // The Browser id is minted (never reused), so assert the stable parts:
    // one surface, re-owned to the navigating session.
    expect($previewTabs.get()[0]?.id).toMatch(/^url:browser-/)
    expect($previewTabs.get()[0]).toMatchObject({ sessionId: 'sess-2' })
    expect($previewTabs.get()[0]?.target.url).toBe('http://localhost:9999')
    expect($visiblePreviewTabs.get()).toHaveLength(1)

    // And sess-1 no longer sees the navigated surface — no context leak.
    $selectedStoredSessionId.set('sess-1')
    expect($visiblePreviewTabs.get()).toHaveLength(0)
  })

  it('adoption dedupes against an already session-scoped row', () => {
    // Both rows for the same file coexist before the session appears.
    $previewTabs.set([
      { id: 'file:/work/a.html', target: fileTarget('/work/a.html') },
      { id: 'file:sess-1:/work/a.html', target: fileTarget('/work/a.html'), sessionId: 'sess-1' }
    ])

    $selectedStoredSessionId.set('sess-1')

    expect($previewTabs.get()).toHaveLength(1)
    expect($previewTabs.get()[0]?.id).toBe('file:sess-1:/work/a.html')
  })

  it('adoption dedupe keeps the active selection valid', () => {
    // The colliding row is the ACTIVE tab when adoption rekeys the draft
    // onto the same id — the survivor carries that id, so the selection must
    // not dangle.
    $previewTabs.set([
      { id: 'file:/work/a.html', target: fileTarget('/work/a.html') },
      { id: 'file:sess-1:/work/a.html', target: fileTarget('/work/a.html'), sessionId: 'sess-1' }
    ])
    selectRightRailTab('file:sess-1:/work/a.html')

    $selectedStoredSessionId.set('sess-1')

    expect($previewTabs.get()).toHaveLength(1)
    expect($rightRailActiveTabId.get()).toBe('file:sess-1:/work/a.html')
  })

  it('migrates legacy unscoped tabs to pinned and rekeys their ids', () => {
    const raw = JSON.stringify([
      { id: 'file:file:///work/a.html', target: fileTarget('/work/a.html') },
      { id: 'file:file:///work/b.html', target: fileTarget('/work/b.html'), sessionId: 'sess-9' }
    ])

    const decoded = decodePreviewTabs(raw)

    expect(decoded).toHaveLength(2)
    expect(decoded[0]).toMatchObject({ id: 'file:/work/a.html', pinned: true })
    expect(decoded[1]?.id).toBe('file:sess-9:/work/b.html')
    expect(decoded[1]?.sessionId).toBe('sess-9')
    expect(decoded[1]?.pinned).toBeUndefined()
  })

  it('does not re-pin a persisted unpinned row', () => {
    const raw = JSON.stringify([
      { id: 'file:/work/a.html', target: fileTarget('/work/a.html'), sessionId: 'sess-1', pinned: false }
    ])

    const decoded = decodePreviewTabs(raw)

    expect(decoded[0]?.pinned).toBe(false)
  })

  it('dedupes the same file persisted under both old and canonical ids', () => {
    const raw = JSON.stringify([
      { id: 'file:file:///work/a.html', target: fileTarget('/work/a.html') },
      { id: 'file:/work/a.html', target: { ...fileTarget('/work/a.html'), label: 'updated' } }
    ])

    const decoded = decodePreviewTabs(raw)

    expect(decoded).toHaveLength(1)
    expect(decoded[0]?.id).toBe('file:/work/a.html')
    expect(decoded[0]?.target.label).toBe('updated')
  })
})
