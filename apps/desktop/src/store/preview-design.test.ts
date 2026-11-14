import { beforeEach, describe, expect, it } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { $layoutTree, noteActiveTreeGroup } from '@/components/pane-shell/tree/store'
import { $rightRailActiveTabId } from '@/store/layout'

import { $poppedBrowserTabIds, $previewTabs, closeRightRail, openPreview, setPreviewTabPinned } from './preview'
import { $previewDesignRequest, openFocusedBrowserInDesignMode } from './preview-design'

function focusSession(sessionId: string) {
  $layoutTree.set(group([`session-tile:${sessionId}`, 'workspace'], { active: `session-tile:${sessionId}`, id: 'design-test' }))
  noteActiveTreeGroup('design-test')
}

beforeEach(() => {
  noteActiveTreeGroup(null)
  $layoutTree.set(null)
  $previewDesignRequest.set(null)
  $poppedBrowserTabIds.set(new Set())
  closeRightRail()
})

describe('openFocusedBrowserInDesignMode', () => {
  it('reuses the focused conversation Browser and preserves its page', () => {
    focusSession('session-a')
    openPreview(
      { kind: 'url', label: 'Existing page', source: 'https://example.com/current', url: 'https://example.com/current' },
      'session-a'
    )
    const existing = $previewTabs.get()[0]!

    expect(openFocusedBrowserInDesignMode('session-a')).toBe('ok')
    expect($previewTabs.get()).toHaveLength(1)
    expect($previewTabs.get()[0]).toMatchObject({ id: existing.id, target: { url: 'https://example.com/current' } })
    expect($previewDesignRequest.get()).toEqual({ sessionId: 'session-a', tabId: existing.id })
  })

  it('is idempotent when invoked repeatedly', () => {
    focusSession('session-a')

    expect(openFocusedBrowserInDesignMode('session-a')).toBe('ok')
    const first = $previewDesignRequest.get()
    expect(openFocusedBrowserInDesignMode('session-a')).toBe('ok')

    expect($previewTabs.get()).toHaveLength(1)
    expect($previewDesignRequest.get()).toEqual(first)
  })

  it('does not open or steer a Browser for another conversation', () => {
    focusSession('session-b')

    expect(openFocusedBrowserInDesignMode('session-a')).toBe('another-conversation')
    expect($previewTabs.get()).toEqual([])
    expect($previewDesignRequest.get()).toBeNull()
  })

  it('prefers the conversation Browser over an active pinned Browser owned elsewhere', () => {
    focusSession('session-a')
    openPreview({ kind: 'url', label: 'Own', source: 'https://own.test', url: 'https://own.test' }, 'session-a')
    const own = $previewTabs.get()[0]!
    openPreview({ kind: 'url', label: 'Pinned', source: 'https://pinned.test', url: 'https://pinned.test' }, 'session-b')
    const foreign = $previewTabs.get().find(tab => tab.sessionId === 'session-b')!
    setPreviewTabPinned(foreign.id, true)
    $rightRailActiveTabId.set(foreign.id)

    expect(openFocusedBrowserInDesignMode('session-a')).toBe('ok')
    expect($rightRailActiveTabId.get()).toBe(own.id)
    expect($previewDesignRequest.get()).toEqual({ sessionId: 'session-a', tabId: own.id })
  })

  it('requires a popped-out Browser to be docked before Design Mode', () => {
    focusSession('session-a')
    openPreview({ kind: 'url', label: 'Popped', source: 'https://example.com', url: 'https://example.com' }, 'session-a')
    const tabId = $previewTabs.get()[0]!.id
    $poppedBrowserTabIds.set(new Set([tabId]))

    expect(openFocusedBrowserInDesignMode('session-a')).toBe('popped-out')
    expect($previewDesignRequest.get()).toBeNull()
  })
})
