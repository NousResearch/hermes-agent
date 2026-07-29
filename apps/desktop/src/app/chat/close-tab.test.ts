import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $rightRailActiveTabId } from '@/store/layout'
import { $previewTabs, closeRightRail, openPreview, type PreviewTarget } from '@/store/preview'

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
    closeRightRail()
    window.localStorage.clear()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    closeRightRail()
    window.localStorage.clear()
  })

  it('closes the active file preview tab (⌘W happy path)', () => {
    openPreview(fileTarget('/work/notes.md'), 'manual')

    expect($previewTabs.get()).toHaveLength(1)
    expect($rightRailActiveTabId.get()).toBe('file:file:///work/notes.md')

    expect(closeActiveTab()).toBe(true)
    expect($previewTabs.get()).toHaveLength(0)
  })

  it('closes the visible tab when the active selection points at a tab that is gone', () => {
    // The rail falls back to tabs[0] until React syncs the selection, so ⌘W has
    // to act on what is actually on screen rather than no-op'ing.
    openPreview(fileTarget('/work/notes.md'), 'manual')
    $rightRailActiveTabId.set('file:file:///work/stale.md')

    expect($previewTabs.get()).toHaveLength(1)
    expect(closeActiveTab()).toBe(true)
    expect($previewTabs.get()).toHaveLength(0)
  })
})
