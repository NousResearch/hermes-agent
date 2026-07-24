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
  dismissPreviewTarget,
  getSessionPreviewRecord,
  openPreviewUrl,
  type PreviewTarget,
  progressPreviewServerRestart,
  setCurrentSessionPreviewTarget
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

  describe('openPreviewUrl', () => {
    it('opens a URL in the live preview pane', () => {
      openPreviewUrl('https://example.com', 'Example')

      expect($previewTarget.get()).toEqual({
        kind: 'url',
        label: 'Example',
        source: 'inline-open',
        url: 'https://example.com'
      })
      expect($paneOpen(PREVIEW_PANE_ID).get()).toBe(true)
      expect($rightRailActiveTabId.get()).toBe(RIGHT_RAIL_PREVIEW_TAB_ID)
    })

    it('uses the URL as label when no label is provided', () => {
      openPreviewUrl('https://example.com/page')

      expect($previewTarget.get()?.label).toBe('https://example.com/page')
      expect($previewTarget.get()?.url).toBe('https://example.com/page')
    })

    it('accepts http URLs', () => {
      openPreviewUrl('http://localhost:5173')

      expect($previewTarget.get()).toEqual({
        kind: 'url',
        label: 'http://localhost:5173',
        source: 'inline-open',
        url: 'http://localhost:5173'
      })
    })

    it('accepts blob: URLs', () => {
      openPreviewUrl('blob:https://example.com/uuid')

      expect($previewTarget.get()?.url).toBe('blob:https://example.com/uuid')
    })

    it('accepts devtools: URLs', () => {
      openPreviewUrl('devtools://devtools/bundled/inspector.html')

      expect($previewTarget.get()?.url).toBe('devtools://devtools/bundled/inspector.html')
    })

    it('blocks javascript: protocol URLs', () => {
      const prev = $previewTarget.get()
      openPreviewUrl('javascript:alert(1)')

      expect($previewTarget.get()).toBe(prev)
    })

    it('blocks data: protocol URLs', () => {
      const prev = $previewTarget.get()
      openPreviewUrl('data:text/html,<script>alert(1)</script>')

      expect($previewTarget.get()).toBe(prev)
    })

    it('blocks file: protocol URLs', () => {
      const prev = $previewTarget.get()
      openPreviewUrl('file:///etc/passwd')

      expect($previewTarget.get()).toBe(prev)
    })

    it('blocks empty URLs', () => {
      const prev = $previewTarget.get()
      openPreviewUrl('')

      expect($previewTarget.get()).toBe(prev)
    })
  })
})
