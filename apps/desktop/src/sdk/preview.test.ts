import { afterEach, expect, it } from 'vitest'

import { $previewTabs } from '@/store/preview'
import { $sessionTiles } from '@/store/session-states'

import { host } from './index'

const session = { runtimeSessionId: 'runtime', storedSessionId: 'stored', connectionId: 'local', profile: 'worker' }
afterEach(() => {
  $previewTabs.set([])
  $sessionTiles.set([])
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
