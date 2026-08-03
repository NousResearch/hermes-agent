import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $rightRailActiveTabId, RIGHT_RAIL_PREVIEW_TAB_ID } from '@/store/layout'
import {
  $filePreviewTabs,
  $previewTarget,
  clearSessionPreviewRegistry,
  type PreviewTarget,
  setCurrentSessionPreviewTarget
} from '@/store/preview'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'

import { closeActiveTab } from './close-tab'

function fileTarget(path: string): PreviewTarget {
  return {
    kind: 'file',
    label: path,
    path,
    previewKind: 'text',
    source: path,
    url: `file://${path}`
  }
}

describe('closeActiveTab', () => {
  beforeEach(() => {
    vi.stubGlobal('document', { activeElement: null })
    $activeSessionId.set('session-1')
    $selectedStoredSessionId.set(null)
    window.localStorage.clear()
    clearSessionPreviewRegistry()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    clearSessionPreviewRegistry()
    window.localStorage.clear()
  })

  it('closes the active file preview tab (⌘W happy path)', () => {
    setCurrentSessionPreviewTarget(fileTarget('/work/notes.md'), 'manual')

    expect($filePreviewTabs.get()).toHaveLength(1)
    expect($rightRailActiveTabId.get()).toBe('file:file:///work/notes.md')

    expect(closeActiveTab()).toBe(true)
    expect($filePreviewTabs.get()).toHaveLength(0)
  })

  it('closes the visible file tab when active selection is a ghost preview', () => {
    // Active tab id stuck on live-preview after that target was cleared, while
    // file tabs remain (UI falls back to tabs[0] until React syncs). ⌘W must
    // close the visible file tab instead of no-op'ing via closeWorkspaceTab().
    setCurrentSessionPreviewTarget(fileTarget('/work/notes.md'), 'manual')
    $previewTarget.set(null)
    $rightRailActiveTabId.set(RIGHT_RAIL_PREVIEW_TAB_ID)

    expect($filePreviewTabs.get()).toHaveLength(1)
    expect(closeActiveTab()).toBe(true)
    expect($filePreviewTabs.get()).toHaveLength(0)
  })
})
