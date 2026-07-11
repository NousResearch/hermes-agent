import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $rightRailActiveTabId, PREVIEW_PANE_ID, RIGHT_RAIL_PREVIEW_TAB_ID } from './layout'
import { $paneOpen } from './panes'
import {
  $filePreviewTabs,
  $filePreviewTarget,
  $previewServerRestart,
  $previewServerRestartStatus,
  $previewSurfaceLayouts,
  $previewTarget,
  $sessionPreviewRegistry,
  $utilityPreviewTabs,
  $webPreviewTabs,
  beginPreviewServerRestart,
  clearSessionPreviewRegistry,
  closeActiveRightRailTab,
  closeRightRail,
  closeRightRailTab,
  detachRightRailTab,
  dismissPreviewTarget,
  getSessionPreviewRecord,
  maximizeRightRailTab,
  minimizeRightRailTab,
  openUtilityPreviewTab,
  type PreviewTarget,
  progressPreviewServerRestart,
  registerSessionPreview,
  restoreRightRailTab,
  setCurrentSessionPreviewTarget,
  setPreviewWorkspaceScope,
  setRightRailTabFloatingGeometry
} from './preview'
import { $activeGatewayProfile } from './profile'
import { $activeSessionId, $connection, $selectedStoredSessionId } from './session'

function previewTarget(source: string): PreviewTarget {
  const isUrl = /^https?:\/\//i.test(source)

  return {
    kind: isUrl ? 'url' : 'file',
    label: source,
    path: isUrl ? undefined : source,
    previewKind: isUrl ? undefined : 'html',
    source,
    url: isUrl ? source : `file://${source}`
  }
}

function withRenderMode(target: PreviewTarget, renderMode: PreviewTarget['renderMode']): PreviewTarget {
  return { ...target, renderMode }
}

describe('preview store', () => {
  beforeEach(() => {
    $activeGatewayProfile.set('default')
    $connection.set(null)
    $previewServerRestart.set(null)
    $activeSessionId.set('session-1')
    $selectedStoredSessionId.set(null)
    window.localStorage.clear()
    clearSessionPreviewRegistry()
  })

  afterEach(() => {
    $activeGatewayProfile.set('default')
    $connection.set(null)
    $previewServerRestart.set(null)
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    window.localStorage.clear()
    clearSessionPreviewRegistry()
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

  it('persists registered previews and dismissal per session', () => {
    const target = previewTarget('/work/demo.html')

    setCurrentSessionPreviewTarget(target, 'tool-result')

    expect($previewTarget.get()).toEqual(withRenderMode(target, 'preview'))
    expect($paneOpen(PREVIEW_PANE_ID).get()).toBe(true)
    expect(getSessionPreviewRecord('session-1')?.normalized).toEqual(withRenderMode(target, 'preview'))
    expect(window.localStorage.getItem('hermes.desktop.sessionPreviews.v1')).toContain('/work/demo.html')

    dismissPreviewTarget()

    expect($previewTarget.get()).toBeNull()
    expect($paneOpen(PREVIEW_PANE_ID).get()).toBe(false)
    expect(getSessionPreviewRecord('session-1')).toBeNull()
    expect(Object.values($sessionPreviewRegistry.get())[0]?.[0]?.dismissedAt).toEqual(expect.any(Number))

    setCurrentSessionPreviewTarget(target, 'tool-result')

    expect(getSessionPreviewRecord('session-1')?.dismissedAt).toBeUndefined()
  })

  it('uses opaque record ids and isolates records by connection identity', () => {
    const target = previewTarget('https://example.com/app')

    $connection.set({ baseUrl: 'https://one.example', mode: 'remote', profile: 'one' } as never)
    const record = registerSessionPreview('shared-session', target, 'manual')

    expect(record?.id).toMatch(/^session-preview:/)
    expect(record?.id).not.toContain('shared-session')
    expect(record?.id).not.toContain(target.url)
    expect(getSessionPreviewRecord('shared-session')?.normalized.url).toBe(target.url)

    $connection.set({ baseUrl: 'https://two.example', mode: 'remote', profile: 'two' } as never)
    expect(getSessionPreviewRecord('shared-session')).toBeNull()

    $connection.set(null)
    $activeGatewayProfile.set('profile-one')
    registerSessionPreview('local-shared-session', target, 'manual')
    expect(getSessionPreviewRecord('local-shared-session')).not.toBeNull()

    $activeGatewayProfile.set('profile-two')
    expect(getSessionPreviewRecord('local-shared-session')).toBeNull()
  })

  it('clears tabs and layouts when the session or connection scope changes', () => {
    setPreviewWorkspaceScope('session-1', 'remote:profile-a:https://one.example')
    setCurrentSessionPreviewTarget(previewTarget('/work/private.txt'), 'manual')
    const tabId = $filePreviewTabs.get()[0]!.id

    detachRightRailTab(tabId)
    expect($filePreviewTabs.get()).toHaveLength(1)
    expect($previewSurfaceLayouts.get()[tabId]?.placement).toBe('floating')

    setPreviewWorkspaceScope('session-1', 'remote:profile-a:https://one.example')
    expect($filePreviewTabs.get()).toHaveLength(1)

    setPreviewWorkspaceScope('session-2', 'remote:profile-a:https://one.example')
    expect($filePreviewTabs.get()).toEqual([])
    expect($webPreviewTabs.get()).toEqual([])
    expect($previewSurfaceLayouts.get()).toEqual({})

    const storedScope = window.localStorage.getItem('hermes.desktop.previewWorkspaceScope.v1') ?? ''
    expect(storedScope).toMatch(/^scope:/)
    expect(storedScope).not.toContain('session-2')
    expect(storedScope).not.toContain('one.example')
  })

  it('preserves global utility tabs when the chat preview scope changes', () => {
    setPreviewWorkspaceScope('session-1', 'remote:profile-a:https://one.example')
    setCurrentSessionPreviewTarget(previewTarget('/work/private.txt'), 'manual')
    openUtilityPreviewTab('terminal')
    openUtilityPreviewTab('host-vnc')

    setPreviewWorkspaceScope('session-2', 'remote:profile-a:https://one.example')

    expect($filePreviewTabs.get()).toEqual([])
    expect($webPreviewTabs.get()).toEqual([])
    expect($utilityPreviewTabs.get()).toEqual([
      { id: 'utility:terminal', kind: 'terminal' },
      { id: 'utility:host-vnc', kind: 'host-vnc' }
    ])
    expect($rightRailActiveTabId.get()).toBe('utility:host-vnc')
    expect($paneOpen(PREVIEW_PANE_ID).get()).toBe(true)
  })

  it('replaces the session preview instead of keeping a back stack', () => {
    const first = previewTarget('/work/first.html')
    const second = previewTarget('/work/second.html')

    setCurrentSessionPreviewTarget(first, 'tool-result')
    setCurrentSessionPreviewTarget(second, 'tool-result')

    expect(Object.values($sessionPreviewRegistry.get())[0]).toHaveLength(1)
    expect(getSessionPreviewRecord('session-1')?.normalized).toEqual(withRenderMode(second, 'preview'))

    dismissPreviewTarget()

    expect($previewTarget.get()).toBeNull()
    expect(getSessionPreviewRecord('session-1')).toBeNull()
    expect(Object.values($sessionPreviewRegistry.get())[0]?.map(record => record.normalized.url)).toEqual([
      'file:///work/second.html'
    ])
  })

  it('keeps file inspection separate from live preview', () => {
    const target = previewTarget('/work/demo.html')
    const preview = previewTarget('/work/live.html')

    setCurrentSessionPreviewTarget(preview, 'tool-result')

    setCurrentSessionPreviewTarget(target, 'manual')

    expect($filePreviewTarget.get()).toEqual(withRenderMode(target, 'source'))
    expect($previewTarget.get()).toEqual(withRenderMode(preview, 'preview'))
    expect(getSessionPreviewRecord('session-1')?.normalized).toEqual(withRenderMode(preview, 'preview'))

    closeActiveRightRailTab()

    expect($filePreviewTarget.get()).toBeNull()
    expect($previewTarget.get()).toEqual(withRenderMode(preview, 'preview'))
  })

  it('keeps file tabs when a live preview opens', () => {
    const file = previewTarget('/work/file.html')
    const live = previewTarget('/work/live.html')

    setCurrentSessionPreviewTarget(file, 'manual')
    setCurrentSessionPreviewTarget(live, 'tool-result')

    expect($filePreviewTabs.get().map(tab => tab.target)).toEqual([withRenderMode(file, 'source')])
    expect($filePreviewTarget.get()).toBeNull()
    expect($rightRailActiveTabId.get()).toBe(RIGHT_RAIL_PREVIEW_TAB_ID)
    expect($previewTarget.get()).toEqual(withRenderMode(live, 'preview'))
  })

  it('keeps multiple browser preview tabs instead of overwriting the live target', () => {
    const first = previewTarget('https://example.com/one')
    const second = previewTarget('https://example.com/two')

    setCurrentSessionPreviewTarget(first, 'manual')
    setCurrentSessionPreviewTarget(second, 'manual')

    expect($webPreviewTabs.get().map(tab => tab.target.url)).toEqual([
      'https://example.com/one',
      'https://example.com/two'
    ])
    const activeId = $rightRailActiveTabId.get()
    expect(activeId).toMatch(/^preview:tab-/)
    expect(activeId).not.toContain('https://example.com/two')
  })

  it('opens terminal and host VNC as peer workspace tabs', () => {
    setCurrentSessionPreviewTarget(previewTarget('/work/file.txt'), 'manual')
    const fileTabId = $filePreviewTabs.get()[0]!.id

    openUtilityPreviewTab('terminal')
    openUtilityPreviewTab('host-vnc')

    expect($utilityPreviewTabs.get()).toEqual([
      { id: 'utility:terminal', kind: 'terminal' },
      { id: 'utility:host-vnc', kind: 'host-vnc' }
    ])
    expect($filePreviewTabs.get()[0]?.id).toBe(fileTabId)
    expect($rightRailActiveTabId.get()).toBe('utility:host-vnc')
    expect($paneOpen(PREVIEW_PANE_ID).get()).toBe(true)
  })

  it('closes one utility tab without closing files or another utility tab', () => {
    setCurrentSessionPreviewTarget(previewTarget('/work/file.txt'), 'manual')
    const fileTabId = $filePreviewTabs.get()[0]!.id
    openUtilityPreviewTab('terminal')
    openUtilityPreviewTab('host-vnc')

    closeRightRailTab('utility:terminal')

    expect($utilityPreviewTabs.get()).toEqual([{ id: 'utility:host-vnc', kind: 'host-vnc' }])
    expect($filePreviewTabs.get()[0]?.id).toBe(fileTabId)
    expect($rightRailActiveTabId.get()).toBe('utility:host-vnc')
    expect($paneOpen(PREVIEW_PANE_ID).get()).toBe(true)
  })

  it('dismisses a browser session record when its tab is closed', () => {
    const target = previewTarget('https://example.com/closed')
    setCurrentSessionPreviewTarget(target, 'manual')
    registerSessionPreview('session-2', target, 'manual')
    const tabId = $webPreviewTabs.get()[0]!.id

    closeRightRailTab(tabId)

    expect($webPreviewTabs.get()).toEqual([])
    expect(getSessionPreviewRecord('session-1')).toBeNull()
    expect(getSessionPreviewRecord('session-2')).toBeNull()
  })

  it('keeps credential-bearing URLs out of persisted tabs and layout keys', () => {
    const sensitive = previewTarget('https://user:pass@example.com/app?token=secret-value#private')

    setCurrentSessionPreviewTarget(sensitive, 'manual')
    const tabId = $webPreviewTabs.get()[0]!.id
    detachRightRailTab(tabId)

    expect(tabId).toMatch(/^preview:tab-/)
    expect(tabId).not.toContain(sensitive.url)
    expect($webPreviewTabs.get()[0]!.target.url).toBe(sensitive.url)
    expect(window.localStorage.getItem('hermes.desktop.webPreviewTabs.v1') ?? '').not.toContain('secret-value')
    expect(window.localStorage.getItem('hermes.desktop.sessionPreviews.v1') ?? '').not.toContain('secret-value')
    expect(window.localStorage.getItem('hermes.desktop.sessionPreviews.v1') ?? '').not.toContain(sensitive.url)
    expect(window.localStorage.getItem('hermes.desktop.previewSurfaceLayouts.v1') ?? '').not.toContain('secret-value')
    expect(window.localStorage.getItem('hermes.desktop.previewSurfaceLayouts.v1') ?? '').not.toContain(sensitive.url)
  })

  it('keeps data URL contents out of every preview persistence key', () => {
    const dataUrl = 'data:text/html,<h1>private-inline-content</h1>'

    const sensitive: PreviewTarget = {
      dataUrl,
      kind: 'url',
      label: 'Private inline preview',
      source: dataUrl,
      url: dataUrl
    }

    setCurrentSessionPreviewTarget(sensitive, 'manual')
    setRightRailTabFloatingGeometry(
      $webPreviewTabs.get()[0]!.id,
      { height: 480, width: 640, x: 80, y: 90 },
      { height: 900, width: 1400 }
    )

    for (const key of [
      'hermes.desktop.webPreviewTabs.v1',
      'hermes.desktop.sessionPreviews.v1',
      'hermes.desktop.previewSurfaceLayouts.v1'
    ]) {
      expect(window.localStorage.getItem(key) ?? '').not.toContain('private-inline-content')
    }
  })

  it('restores floating geometry after maximize and minimize', () => {
    const target = previewTarget('/work/layout.html')

    setCurrentSessionPreviewTarget(target, 'manual')
    const tabId = $filePreviewTabs.get()[0]!.id
    const geometry = { height: 480, width: 720, x: 96, y: 88 }

    detachRightRailTab(tabId)
    setRightRailTabFloatingGeometry(tabId, geometry, { height: 900, width: 1400 })
    maximizeRightRailTab(tabId)

    expect($previewSurfaceLayouts.get()[tabId]).toMatchObject({
      placement: 'maximized',
      restore: { geometry, placement: 'floating' }
    })

    restoreRightRailTab(tabId)
    minimizeRightRailTab(tabId)
    restoreRightRailTab(tabId)

    expect($previewSurfaceLayouts.get()[tabId]).toMatchObject({ geometry, placement: 'floating' })
    expect($rightRailActiveTabId.get()).toBe(tabId)
  })

  it('moves activation away from a minimized active tab when another surface is available', () => {
    setCurrentSessionPreviewTarget(previewTarget('/work/first.txt'), 'manual')
    const firstId = $filePreviewTabs.get()[0]!.id
    setCurrentSessionPreviewTarget(previewTarget('/work/second.txt'), 'manual')
    const secondId = $filePreviewTabs.get()[1]!.id

    minimizeRightRailTab(secondId)

    expect($previewSurfaceLayouts.get()[secondId]?.placement).toBe('minimized')
    expect($rightRailActiveTabId.get()).toBe(firstId)
  })

  it('reconciles the active tab after closing the last file surface', () => {
    setCurrentSessionPreviewTarget(previewTarget('/work/only.txt'), 'manual')
    const onlyId = $filePreviewTabs.get()[0]!.id

    closeRightRailTab(onlyId)

    expect($filePreviewTabs.get()).toEqual([])
    expect($rightRailActiveTabId.get()).toBe(RIGHT_RAIL_PREVIEW_TAB_ID)
  })

  it('removes layouts on close and clears every layout on close-all', () => {
    setCurrentSessionPreviewTarget(previewTarget('/work/first.txt'), 'manual')
    const firstId = $filePreviewTabs.get()[0]!.id
    setCurrentSessionPreviewTarget(previewTarget('/work/second.txt'), 'manual')
    const secondId = $filePreviewTabs.get()[1]!.id

    detachRightRailTab(firstId)
    maximizeRightRailTab(secondId)
    closeRightRailTab(secondId)

    expect($previewSurfaceLayouts.get()[secondId]).toBeUndefined()
    expect($rightRailActiveTabId.get()).toBe(firstId)

    closeRightRail()

    expect($previewSurfaceLayouts.get()).toEqual({})
    expect($filePreviewTabs.get()).toEqual([])
    expect($webPreviewTabs.get()).toEqual([])
  })
})
