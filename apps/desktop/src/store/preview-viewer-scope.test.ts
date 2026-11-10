import { afterEach, beforeEach, expect, it } from 'vitest'

import {
  $previewTabs,
  closeRightRail,
  closeRightRailTab,
  dropPreviewTabsForProfile,
  migratePreviewTabsForProfile,
  openPreview,
  setPreviewScope
} from './preview'

beforeEach(() => {
  setPreviewScope('viewer-owner')
  closeRightRail()
  window.localStorage.clear()
})
afterEach(() => {
  closeRightRail()
  dropPreviewTabsForProfile('viewer-owner')
  dropPreviewTabsForProfile('viewer-other')
  dropPreviewTabsForProfile('viewer-renamed')
  setPreviewScope('default')
  window.localStorage.clear()
})

function openViewer() {
  return openPreview({
    kind: 'url',
    label: 'Viewer',
    source: 'viewer',
    url: 'https://viewer.example/view#ticket=fixture',
    transient: true,
    browserContext: 'isolated'
  })
}

it('keeps the exact explicitly opened viewer across profiles, but scopes ordinary tabs', () => {
  const file = openPreview({ kind: 'file', label: 'File', source: '/work/file.txt', url: 'file:///work/file.txt' })
  const viewer = openViewer()
  setPreviewScope('viewer-other')
  expect($previewTabs.get()).toEqual([viewer])
  expect($previewTabs.get()[0]).toBe(viewer)
  expect(window.localStorage.getItem('hermes.desktop.previewTabs.v2')).not.toContain('ticket')
  closeRightRailTab(viewer.id)
  setPreviewScope('viewer-owner')
  expect($previewTabs.get()).toEqual([file])
})

it('does not make ordinary transient tabs global', () => {
  const tab = openPreview({
    kind: 'url',
    label: 'Temporary',
    source: 'temporary',
    url: 'https://example.test',
    transient: true
  })

  setPreviewScope('viewer-other')
  expect($previewTabs.get()).toEqual([])
  setPreviewScope('viewer-owner')
  expect($previewTabs.get()[0]).toBe(tab)
})

it('retires a viewer when its owning profile is deleted from another scope', () => {
  openViewer()
  setPreviewScope('viewer-other')
  expect($previewTabs.get()).toHaveLength(1)
  dropPreviewTabsForProfile('viewer-owner')
  expect($previewTabs.get()).toEqual([])
})

it('preserves viewer ownership through profile rename, without remounting it', () => {
  const viewer = openViewer()
  setPreviewScope('viewer-other')
  migratePreviewTabsForProfile('viewer-owner', 'viewer-renamed')
  expect($previewTabs.get()[0]).toBe(viewer)
  dropPreviewTabsForProfile('viewer-owner')
  expect($previewTabs.get()[0]).toBe(viewer)
  dropPreviewTabsForProfile('viewer-renamed')
  expect($previewTabs.get()).toEqual([])
})
