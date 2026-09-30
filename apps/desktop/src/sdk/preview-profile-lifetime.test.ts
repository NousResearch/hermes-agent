import { afterEach, expect, it, vi } from 'vitest'

import {
  $browserPages,
  $previewTabs,
  closeRightRailTab,
  noteBrowserPage,
  previewTabsForAgent,
  setPreviewScope
} from '@/store/preview'
import { $sessionTiles } from '@/store/session-states'

import { openPluginPreview } from './preview'

const session = { runtimeSessionId: 'runtime', storedSessionId: 'stored', connectionId: 'local', profile: 'worker' }
afterEach(() => {
  setPreviewScope('worker')
  $previewTabs.set([])
  $browserPages.set({})
  $sessionTiles.set([])
  vi.clearAllTimers()
  vi.useRealTimers()
})
it('binds viewer ownership to the validated session, not the profile currently focused', async () => {
  $sessionTiles.set([
    { storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute: { connectionId: 'local', profile: 'worker' } }
  ])
  setPreviewScope('other')
  expect(await openPluginPreview({ url: 'https://viewer.example/view#ticket=fixture', session })).toBe(true)
  expect(previewTabsForAgent()).toEqual([])
  setPreviewScope('worker')
  expect(previewTabsForAgent()).toHaveLength(1)
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
