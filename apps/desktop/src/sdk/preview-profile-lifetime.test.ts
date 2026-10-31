import { afterEach, expect, it, vi } from 'vitest'

import {
  $browserPages,
  $previewTabs,
  $visiblePreviewTabs,
  closeRightRailTab,
  noteBrowserPage,
  previewTabsFor,
  setPreviewScope
} from '@/store/preview'
import { $selectedStoredSessionId } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'

import { openPluginPreview } from './preview'

const session = { runtimeSessionId: 'runtime', storedSessionId: 'stored', connectionId: 'local', profile: 'worker' }
afterEach(() => {
  setPreviewScope('worker')
  $previewTabs.set([])
  $browserPages.set({})
  $sessionTiles.set([])
  $selectedStoredSessionId.set(null)
  vi.clearAllTimers()
  vi.useRealTimers()
})
it('binds viewer ownership to the validated session, not the profile or session currently focused', async () => {
  $sessionTiles.set([
    { storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute: { connectionId: 'local', profile: 'worker' } }
  ])
  setPreviewScope('other')
  $selectedStoredSessionId.set('elsewhere')
  expect(await openPluginPreview({ url: 'https://viewer.example/view#ticket=fixture', session })).toBe(true)
  expect($previewTabs.get()[0]?.sessionId).toBe('stored')
  expect(previewTabsFor()).toEqual([])
  expect($visiblePreviewTabs.get()).toEqual([])
  setPreviewScope('worker')
  $selectedStoredSessionId.set('stored')
  expect(previewTabsFor()).toHaveLength(1)
  expect($visiblePreviewTabs.get()).toHaveLength(1)
})

it('keeps the SAME mounted viewer document renewing across profile away/back', async () => {
  vi.useFakeTimers()
  $sessionTiles.set([
    { storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute: { connectionId: 'local', profile: 'worker' } }
  ])
  setPreviewScope('worker')
  const renew = vi.fn(async () => {})
  const url = 'https://viewer.example/view#ticket=fixture'
  await openPluginPreview({ url, session, onKeepAlive: renew })
  const tab = $previewTabs.get()[0]!
  const document = { isLive: () => true }
  noteBrowserPage(tab.id, { title: 'Viewer', url, document })
  await vi.advanceTimersByTimeAsync(0)
  expect(renew).toHaveBeenCalledOnce()
  setPreviewScope('other')
  expect($browserPages.get()[tab.id]?.document).toBe(document)
  expect(document.isLive()).toBe(true)
  await vi.advanceTimersByTimeAsync(60_000)
  setPreviewScope('worker')
  expect($previewTabs.get()[0]).toBe(tab)
  await vi.advanceTimersByTimeAsync(60_000)
  expect(renew).toHaveBeenCalledTimes(3)
  closeRightRailTab(tab.id)
  await vi.advanceTimersByTimeAsync(60_000)
  expect(renew).toHaveBeenCalledTimes(3)
})
