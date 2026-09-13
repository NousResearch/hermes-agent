import { afterEach, expect, it, vi } from 'vitest'

import { $browserPages, $previewTabs, closeRightRailTab, newBrowserTab, noteBrowserPage } from '@/store/preview'
import { $sessionTiles } from '@/store/session-states'

import { host } from './index'

const session = { runtimeSessionId: 'runtime', storedSessionId: 'stored', connectionId: 'local', profile: 'worker' }
afterEach(() => {
  $previewTabs.set([])
  $sessionTiles.set([])
  $browserPages.set({})
  vi.clearAllTimers()
  vi.useRealTimers()
})
it('opens a transient ticket URL through preview tabs and refuses unsafe or stale session actions', async () => {
  expect(host.openPreview).toBeTypeOf('function')
  $sessionTiles.set([
    { storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute: { connectionId: 'local', profile: 'worker' } }
  ])
  const url = 'http://127.0.0.1:9876/viewer?ticket=secret'
  expect(await host.openPreview({ url, label: 'Viewer', session })).toBe(true)
  expect($previewTabs.get().at(-1)?.target).toMatchObject({ kind: 'url', url, label: 'Viewer', transient: true })
  expect(window.localStorage.getItem('hermes.desktop.previewTabs.v2')).not.toContain('secret')

  for (const unsafe of [
    'javascript:alert(1)',
    'file:///etc/passwd',
    'data:text/html,x',
    'https://user:pass@example.org',
    '/relative'
  ]) {
    expect(await host.openPreview({ url: unsafe, session })).toBe(false)
  }

  expect(await host.openPreview({ url, session: { ...session, profile: 'other' } })).toBe(false)
  expect(await host.openPreview({ url, session: { ...session, runtimeSessionId: 'stale' } })).toBe(false)
})

it('keeps parked previews alive independently of foreground session and stops irreversibly on guest navigation', async () => {
  vi.useFakeTimers()
  $sessionTiles.set([
    { storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute: { connectionId: 'local', profile: 'worker' } }
  ])
  const url = 'https://viewer.example/view?mode=watch#ticket=secret'
  const onKeepAlive = vi.fn(async () => {})
  expect(await host.openPreview({ url, session, onKeepAlive })).toBe(true)
  const tab = $previewTabs.get()[0]!
  const document = { isLive: () => true }
  noteBrowserPage(tab.id, { title: 'Viewer', url, document })
  await vi.advanceTimersByTimeAsync(0)
  expect(onKeepAlive).toHaveBeenCalledOnce()
  newBrowserTab()
  $sessionTiles.set([])
  noteBrowserPage(tab.id, { title: 'Viewer', url: url.split('#')[0]!, document })
  await vi.advanceTimersByTimeAsync(60_000)
  expect(onKeepAlive).toHaveBeenCalledTimes(2)
  noteBrowserPage(tab.id, { title: 'Other', url: 'https://viewer.example/other', document })
  noteBrowserPage(tab.id, { title: 'Viewer', url, document })
  await vi.advanceTimersByTimeAsync(180_000)
  expect(onKeepAlive).toHaveBeenCalledTimes(2)
})

it('retires the original preview on same-URL replacement or close, including pending callbacks', async () => {
  vi.useFakeTimers()
  $sessionTiles.set([
    { storedSessionId: 'stored', runtimeId: 'runtime', ownerRoute: { connectionId: 'local', profile: 'worker' } }
  ])
  const url = 'https://viewer.example/view#ticket=one'
  let finish!: () => void

  const old = vi.fn(
    () =>
      new Promise<void>(resolve => {
        finish = resolve
      })
  )

  await host.openPreview({ url, session, onKeepAlive: old })
  const tab = $previewTabs.get()[0]!
  noteBrowserPage(tab.id, { title: 'Viewer', url, document: { isLive: () => true } })
  await vi.advanceTimersByTimeAsync(0)
  const current = vi.fn(async () => {})
  await host.openPreview({ url, session, onKeepAlive: current })
  finish()
  await vi.advanceTimersByTimeAsync(60_000)
  expect(old).toHaveBeenCalledOnce()
  expect(current).toHaveBeenCalledTimes(2)
  closeRightRailTab(tab.id)
  await vi.advanceTimersByTimeAsync(180_000)
  expect(current).toHaveBeenCalledTimes(2)
  expect(vi.getTimerCount()).toBe(0)
})
