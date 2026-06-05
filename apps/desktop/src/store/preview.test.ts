import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $rightRailActiveTabId, PREVIEW_PANE_ID, RIGHT_RAIL_PREVIEW_TAB_ID } from './layout'
import { $paneOpen } from './panes'
import {
  $filePreviewTabs,
  $filePreviewTarget,
  $previewServerRestart,
  $previewServerRestartStatus,
  $previewTarget,
  $sessionPreviewRegistry,
  beginPreviewServerRestart,
  clearSessionPreviewRegistry,
  closeActiveRightRailTab,
  decodeFilePreviewTabs,
  dismissPreviewTarget,
  getSessionPreviewRecord,
  type PreviewTarget,
  progressPreviewServerRestart,
  setCurrentSessionPreviewTarget,
  syncFilePreviewTabsForSession
} from './preview'
import { $activeSessionId, $selectedStoredSessionId } from './session'

function previewTarget(source: string): PreviewTarget {
  return {
    kind: 'file',
    label: source,
    path: source,
    previewKind: 'html',
    source,
    url: `file://${source}`
  }
}

function withRenderMode(target: PreviewTarget, renderMode: PreviewTarget['renderMode']): PreviewTarget {
  return { ...target, renderMode }
}

describe('preview store', () => {
  beforeEach(() => {
    $previewServerRestart.set(null)
    $activeSessionId.set('session-1')
    $selectedStoredSessionId.set(null)
    window.localStorage.clear()
    clearSessionPreviewRegistry()
  })

  afterEach(() => {
    $previewServerRestart.set(null)
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    clearSessionPreviewRegistry()
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
    expect($sessionPreviewRegistry.get()['session-1']?.[0]?.dismissedAt).toEqual(expect.any(Number))

    setCurrentSessionPreviewTarget(target, 'tool-result')

    expect(getSessionPreviewRecord('session-1')?.dismissedAt).toBeUndefined()
  })

  it('replaces the session preview instead of keeping a back stack', () => {
    const first = previewTarget('/work/first.html')
    const second = previewTarget('/work/second.html')

    setCurrentSessionPreviewTarget(first, 'tool-result')
    setCurrentSessionPreviewTarget(second, 'tool-result')

    expect($sessionPreviewRegistry.get()['session-1']).toHaveLength(1)
    expect(getSessionPreviewRecord('session-1')?.normalized).toEqual(withRenderMode(second, 'preview'))

    dismissPreviewTarget()

    expect($previewTarget.get()).toBeNull()
    expect(getSessionPreviewRecord('session-1')).toBeNull()
    expect($sessionPreviewRegistry.get()['session-1']?.map(record => record.normalized.url)).toEqual([
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

  it('drops file preview tabs from other sessions when the routed session changes', () => {
    const file = previewTarget('/work/file.html')

    setCurrentSessionPreviewTarget(file, 'manual')

    expect($filePreviewTabs.get()).toHaveLength(1)
    expect($filePreviewTabs.get()[0]?.sessionId).toBe('session-1')
    expect($rightRailActiveTabId.get()).toBe(`file:${file.url}`)

    // Routing to a conversation that does not contain the attachment must
    // dismiss the leaked file tab and fall back to the live preview tab.
    syncFilePreviewTabsForSession('session-2')

    expect($filePreviewTabs.get()).toEqual([])
    expect($rightRailActiveTabId.get()).toBe(RIGHT_RAIL_PREVIEW_TAB_ID)
  })

  it('keeps file preview tabs when re-syncing the same session', () => {
    const file = previewTarget('/work/file.html')

    setCurrentSessionPreviewTarget(file, 'manual')

    // The routing effect re-runs on every registry update within the same
    // session, so syncing the unchanged session must be a no-op.
    syncFilePreviewTabsForSession('session-1')

    expect($filePreviewTabs.get()).toHaveLength(1)
    expect($rightRailActiveTabId.get()).toBe(`file:${file.url}`)
  })

  it('filters file tabs by the session id it is given', () => {
    const file = previewTarget('/work/file.html')

    setCurrentSessionPreviewTarget(file, 'manual')
    expect($filePreviewTabs.get()[0]?.sessionId).toBe('session-1')

    // The function scopes purely by its argument — the caller (the routing
    // effect) is responsible for passing the session it is routing to. Syncing
    // against a non-matching id drops the tab.
    syncFilePreviewTabsForSession('session-1')
    expect($filePreviewTabs.get()).toHaveLength(1)

    syncFilePreviewTabsForSession('other-session')
    expect($filePreviewTabs.get()).toEqual([])
  })

  it('drops legacy and blank-session persisted file tabs on decode', () => {
    const scoped = {
      id: 'file:///work/new.html',
      sessionId: 'session-1',
      target: previewTarget('/work/new.html')
    }

    // A pre-scoping row carries no sessionId, and a blank sessionId is just as
    // unattributable — neither can be scoped to a conversation, so decode must
    // reject both rather than surface an unscoped tab.
    const legacy = { id: 'file:///work/old.html', target: previewTarget('/work/old.html') }
    const blank = { id: 'file:///work/blank.html', sessionId: '', target: previewTarget('/work/blank.html') }

    const decoded = decodeFilePreviewTabs(JSON.stringify([scoped, legacy, blank]))

    expect(decoded).toEqual([scoped])
  })
})
