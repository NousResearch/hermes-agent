import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { BrowserWorkspace, BrowserWorkspaceTab } from '../../electron/browser-workspace-types'

const key = 'hermes.desktop.previewTabs.v2'

const tab = (id: string, url = `https://example.test/${id}`): BrowserWorkspaceTab => ({
  id: `url:${id}`,
  pinned: false,
  target: { kind: 'url', label: id, source: url, url }
})

const workspace = (id: string, scope: string, tabs = [tab(id)]): BrowserWorkspace => ({
  id,
  revision: 1,
  selectionVersion: 0,
  selectionIntentVersion: 0,
  tabs: tabs.map(tab => ({ ...tab, sessionId: `session-${scope}` })),
  activeTabId: tabs[0]!.id,
  removed: [],
  docked: [],
  closed: false,
  owner: {
    conversation: { kind: 'session', id: `session-${scope}`, connectionId: `connection-${scope}`, profile: scope },
    connectionId: `connection-${scope}`,
    profile: scope,
    scope,
    destination: { kind: 'composer', surfaceId: `surface-${scope}`, target: `session-${scope}`, windowId: 'chat' }
  }
})

beforeEach(() => {
  vi.resetModules()
  window.localStorage.clear()
  window.history.replaceState({}, '', '/')
})
afterEach(() => {
  window.localStorage.clear()
  window.history.replaceState({}, '', '/')
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

describe('workspace snapshot integration', () => {
  it("persists only the owning scope and preserves another opener's newer page state", async () => {
    window.localStorage.setItem(key, JSON.stringify({ alpha: [tab('a')], beta: [tab('b')] }))
    const { receiveBrowserWorkspace } = await import('./browser-workspaces')
    const { setPreviewScope, $previewTabs, $poppedBrowserTabIds } = await import('./preview')
    setPreviewScope('alpha')
    // Another owner writes after this module captured its startup buckets.
    window.localStorage.setItem(
      key,
      JSON.stringify({ alpha: [tab('a')], beta: [tab('b', 'https://example.test/new')] })
    )
    const state = workspace('window-a', 'alpha', [tab('a'), tab('new')])
    receiveBrowserWorkspace(state)
    expect($previewTabs.get().map(item => item.id)).toEqual(['url:a', 'url:new'])
    expect($poppedBrowserTabIds.get().has('url:new')).toBe(true)
    const stored = JSON.parse(window.localStorage.getItem(key)!)
    expect(stored.beta[0].target.url).toBe('https://example.test/new')
    expect(stored.alpha.map((item: { id: string }) => item.id)).toEqual(['url:a', 'url:new'])
  })

  it('distinguishes delete, one-tab docking and native close without duplicates; docked comments retain origin', async () => {
    const { receiveBrowserWorkspace } = await import('./browser-workspaces')
    const { setPreviewScope, $previewTabs, $poppedBrowserTabIds } = await import('./preview')
    const { readPreviewAnnotateDestination } = await import('@/lib/preview-annotate/handoff')
    setPreviewScope('alpha')
    const initial = workspace('window-a', 'alpha', [tab('seed'), tab('b'), tab('c')])
    receiveBrowserWorkspace(initial)
    const docked = { ...initial, revision: 2, selectionVersion: 1, selectionIntentVersion: 1, tabs: [tab('seed'), tab('c')], docked: [tab('b')] }
    receiveBrowserWorkspace(docked)
    expect($poppedBrowserTabIds.get().has('url:b')).toBe(false)
    expect(readPreviewAnnotateDestination('url:b')).toEqual(initial.owner.destination)
    receiveBrowserWorkspace({
      ...docked,
      revision: 3,
      selectionVersion: 2,
      selectionIntentVersion: 2,
      tabs: [tab('c')],
      removed: ['url:seed'],
      activeTabId: 'url:c'
    })

    const closed = {
      ...initial,
      revision: 4,
      selectionVersion: 3,
      selectionIntentVersion: 3,
      tabs: [],
      removed: ['url:seed'],
      docked: [tab('b'), tab('c')],
      activeTabId: null,
      closed: true
    }

    receiveBrowserWorkspace(closed)
    receiveBrowserWorkspace(closed)
    expect($previewTabs.get().map(item => item.id)).toEqual(['url:b', 'url:c'])
    expect([...$poppedBrowserTabIds.get()]).toEqual([])
    expect(readPreviewAnnotateDestination('url:c')).toEqual(initial.owner.destination)
    expect(readPreviewAnnotateDestination('url:seed')).toBeNull()
  })

  it('selects one owned window explicitly, fails closed when ambiguous, and never overrides a local tab', async () => {
    const composer = window.document.createElement('div')
    composer.dataset.composerTarget = 'session-alpha'
    composer.dataset.composerSurfaceId = 'surface-alpha'
    composer.dataset.browserSessionId = 'session-alpha'
    window.document.body.append(composer)

    try {
      const { capturePreviewAnnotateDestination } = await import('@/lib/preview-annotate/handoff')
      const { setSessionOwnerHint } = await import('./session')
      await import('./session-states')
      setSessionOwnerHint('session-alpha', { connectionId: 'connection-alpha', profile: 'alpha' })
      const { receiveBrowserWorkspace, selectedPopoutTarget } = await import('./browser-workspaces')
      const { setPreviewScope, newBrowserTab } = await import('./preview')
      const { selectRightRailTab } = await import('./layout')
      setPreviewScope('alpha')
      const first = workspace('one', 'alpha')
      first.owner.destination = capturePreviewAnnotateDestination()
      const second = workspace('two', 'alpha')
      second.owner.destination = first.owner.destination
      receiveBrowserWorkspace(first)
      receiveBrowserWorkspace(second)
      selectRightRailTab(null)
      expect(selectedPopoutTarget()).toBeNull()
      selectRightRailTab('url:two')
      expect(selectedPopoutTarget()).toMatchObject({ windowId: 'two', tabId: 'url:two' })
      newBrowserTab()
      expect(selectedPopoutTarget()).toBeNull()
    } finally { composer.remove() }
  })

  it('keeps detached user selection through layout/page updates but yields to a docked interaction', async () => {
    const composer = document.createElement('div')
    composer.dataset.composerTarget = 'session-alpha'
    composer.dataset.composerSurfaceId = 'surface-alpha'
    composer.dataset.browserSessionId = 'session-alpha'
    document.body.append(composer)
    const pane = document.createElement('div')
    pane.dataset.treeGroup = 'local-preview'
    document.body.append(pane)

    try {
      const { $selectedStoredSessionId, setSessionOwnerHint } = await import('./session')
      await import('./session-states')
      setSessionOwnerHint('session-alpha', { connectionId: 'connection-alpha', profile: 'alpha' })
      $selectedStoredSessionId.set('session-alpha')
      const { receiveBrowserWorkspace, selectedPopoutTarget } = await import('./browser-workspaces')
      const { setPreviewScope, $previewTabs } = await import('./preview')
      const { $rightRailActiveTabId, selectRightRailTab } = await import('./layout')
      const { $layoutTree, noteActiveTreeGroup } = await import('@/components/pane-shell/tree/store')
      const { group } = await import('@/components/pane-shell/tree/model')
      const { watchPreviewTiles } = await import('@/app/chat/preview-tile')
      setPreviewScope('alpha')
      $previewTabs.set([{ ...tab('local'), sessionId: 'session-alpha' }])
      watchPreviewTiles()
      $layoutTree.set(group(['preview-tile:url:local'], { id: 'local-preview' }))
      noteActiveTreeGroup('local-preview')
      selectRightRailTab('url:local')
      const initial = workspace('one', 'alpha', [tab('seed'), tab('second')])
      receiveBrowserWorkspace(initial)
      const selected = { ...initial, revision: 2, selectionVersion: 1, selectionIntentVersion: 1, activeTabId: 'url:second' }
      receiveBrowserWorkspace(selected)
      expect(selectedPopoutTarget()).toMatchObject({ windowId: 'one', tabId: 'url:second' })
      // Layout-only emissions from the old docked zone are not user intent.
      $layoutTree.set(group(['preview-tile:url:local'], { id: 'local-preview' }))
      receiveBrowserWorkspace({
        ...selected,
        revision: 3,
        tabs: selected.tabs.map(item => item.id === 'url:second'
          ? { ...item, target: tab('second', 'https://example.test/new').target }
          : item)
      })
      expect(selectedPopoutTarget()).toMatchObject({ tabId: 'url:second' })
      // Clicking the already-active docked zone must reclaim selection too.
      pane.dispatchEvent(new Event('pointerdown', { bubbles: true }))
      expect($rightRailActiveTabId.get()).toBe('url:local')
      expect(selectedPopoutTarget()).toBeNull()
      receiveBrowserWorkspace({ ...selected, revision: 4 })
      expect(selectedPopoutTarget()).toBeNull()
      // Explicit selection of the same detached tab is a new gesture.
      receiveBrowserWorkspace({ ...selected, revision: 5, selectionIntentVersion: 2 })
      expect(selectedPopoutTarget()).toMatchObject({ tabId: 'url:second' })
      expect(selectedPopoutTarget()?.selectionVersion).toBe(selected.selectionVersion)
      // A preview tab can share a group whose current pane is not a preview.
      $layoutTree.set(group(['terminal', 'preview-tile:url:local'], { id: 'local-preview', active: 'terminal' }))
      pane.dataset.treeTab = 'preview-tile:url:local'
      pane.dispatchEvent(new Event('pointerdown', { bubbles: true }))
      expect($rightRailActiveTabId.get()).toBe('url:local')
      composer.dataset.browserSessionId = 'someone-else'
      selectRightRailTab('url:local')
      receiveBrowserWorkspace({ ...selected, revision: 6, selectionIntentVersion: 3 })
      expect($rightRailActiveTabId.get()).toBe('url:local')
    } finally {
      composer.remove()
      pane.remove()
    }
  })

  it('subscribes before cold snapshots and ignores an older response after a live update', async () => {
    const initial = workspace('window-a', 'alpha')
    let listener!: (state: BrowserWorkspace) => void
    let resolve!: (states: BrowserWorkspace[]) => void
    const order: string[] = []

    const stop = vi.fn()

    ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
      browserWorkspace: {
        onChanged: (callback: typeof listener) => {
          order.push('subscribe')
          listener = callback

          return stop
        },
        snapshots: () => {
          order.push('snapshot')

          return new Promise<BrowserWorkspace[]>(done => {
            resolve = done
          })
        }
      }
    }
    window.history.replaceState({}, '', '/?win=browser&browserWindow=window-a')
    const { installBrowserWorkspaceSync, $browserWorkspaces } = await import('./browser-workspaces')
    const cleanup = installBrowserWorkspaceSync()
    listener({ ...initial, revision: 2, tabs: [tab('a'), tab('b')], activeTabId: 'url:b' })
    resolve([initial, workspace('stranger', 'beta')])
    await Promise.resolve()
    expect(order).toEqual(['subscribe', 'snapshot'])
    expect($browserWorkspaces.get()['window-a']?.revision).toBe(2)
    expect($browserWorkspaces.get().stranger).toBeUndefined()
    expect(window.localStorage.getItem(key)).toBeNull()
    cleanup()
    expect(stop).toHaveBeenCalledOnce()
  })
})
